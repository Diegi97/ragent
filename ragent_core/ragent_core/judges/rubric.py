import asyncio
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any, Self

import verifiers.v1 as vf
from pydantic import BaseModel, Field, model_validator
from verifiers.v1.utils.retries import retrying

RUBRIC_PROMPT = """Given a task, a response, and grading criteria, determine whether the response satisfies each criterion.

Task:
```
{question}
```

Response:
```
{response}
```

Criteria:
{criteria}

Verdict options: {negative_verdict}, {positive_verdict}

For each criterion ID, provide a brief explanation of your assessment and the verdict that best matches it.

Return exactly one evaluation per criterion, using each criterion's exact ID. The
number of <criterion> children must equal the number of supplied criteria: one supplied
criterion requires one child, and four supplied criteria require four children. For
example, a batch of two criteria has this XML structure:

<criteria>
  <criterion>
    <id>C-001</id>
    <reason>Brief explanation of the assessment</reason>
    <verdict>selected verdict</verdict>
  </criterion>
  <criterion>
    <id>C-002</id>
    <reason>Brief explanation of the assessment</reason>
    <verdict>selected verdict</verdict>
  </criterion>
</criteria>
"""

_CRITERIA_XML_RE = re.compile(
    r"<criteria(?:\s[^>]*)?>.*?</criteria\s*>",
    re.DOTALL,
)
_CRITERION_XML_RE = re.compile(
    r"<criterion(?:\s[^>]*)?>(.*?)</criterion\s*>",
    re.DOTALL,
)
_FIELD_XML_RE = {
    field: re.compile(
        rf"<{field}(?:\s[^>]*)?>(.*?)</{field}\s*>",
        re.DOTALL,
    )
    for field in ("id", "reason", "verdict")
}


class JudgeCriterion(BaseModel):
    id: str
    text: str
    weight: float = Field(default=1.0, gt=0)


class CriterionVerdict(BaseModel):
    id: str
    reason: str
    verdict: str


class RubricJudgeConfig(vf.JudgeConfig):
    """Configuration placed under the task's singular ``judge`` field."""

    name: str = "rubric"
    criteria_field: str = "rubric"
    question_field: str = "question"
    view: vf.JudgeView = "last_reply"
    negative_verdict: str = "no"
    positive_verdict: str = "yes"
    max_criteria: int | None = 4
    """Maximum criteria per judge call. ``None`` grades all criteria in one call."""
    max_retries: int = 5
    """Retries after a malformed or otherwise invalid judge verdict."""

    @model_validator(mode="after")
    def validate_verdicts(self) -> Self:
        if not self.negative_verdict.strip() or not self.positive_verdict.strip():
            raise ValueError("rubric verdicts must be non-empty")
        if (
            self.negative_verdict.strip().casefold()
            == self.positive_verdict.strip().casefold()
        ):
            raise ValueError("negative_verdict and positive_verdict must differ")
        if self.max_criteria is not None and self.max_criteria < 1:
            raise ValueError("max_criteria must be at least 1 or None")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        return self


def _parse_unescaped_xml_verdicts(block: str) -> list[CriterionVerdict]:
    """Parse judge fields as text when XML entities were not escaped."""

    verdicts: list[CriterionVerdict] = []
    for criterion_match in _CRITERION_XML_RE.finditer(block):
        body = criterion_match.group(1)
        values: dict[str, str] = {}
        for field, pattern in _FIELD_XML_RE.items():
            if match := pattern.search(body):
                values[field] = match.group(1).strip()
        verdicts.append(CriterionVerdict.model_validate(values))
    return verdicts


def parse_xml_verdicts(text: str) -> list[CriterionVerdict]:
    """Extract the last valid ``<criteria>`` block from a model response.

    Prose and Markdown fences outside the XML are ignored. Nested markup inside a
    field is reduced to its combined text, so the reason is free to span lines.
    """

    parsed: list[CriterionVerdict] | None = None
    for match in _CRITERIA_XML_RE.finditer(text):
        block = match.group()
        verdicts: list[CriterionVerdict] = []
        try:
            root = ET.fromstring(block)
            for element in root.findall("./criterion"):
                values: dict[str, str] = {}
                for field in ("id", "reason", "verdict"):
                    child = element.find(field)
                    if child is not None:
                        values[field] = "".join(child.itertext()).strip()
                verdicts.append(CriterionVerdict.model_validate(values))
        except (ET.ParseError, ValueError):
            try:
                verdicts = _parse_unescaped_xml_verdicts(block)
            except ValueError:
                continue

        if verdicts:
            parsed = verdicts

    if parsed is None:
        raise ValueError(f"judge returned no parseable criteria XML: {text!r}")
    return parsed


class RubricJudge(vf.Judge[list[CriterionVerdict], RubricJudgeConfig]):
    prompt = RUBRIC_PROMPT

    def parse(
        self,
        response: vf.JudgeResponse[list[CriterionVerdict]],
    ) -> list[CriterionVerdict]:
        return parse_xml_verdicts(response.text)

    def _criteria(self, task: vf.TaskData) -> list[JudgeCriterion]:
        raw_items = getattr(task, self.config.criteria_field, None)
        if raw_items is None:
            raise ValueError(
                f"Rubric judge found no {self.config.criteria_field!r} field "
                "on the task"
            )

        def criterion_text(raw_item: Any) -> str:
            text = (
                raw_item.get("criterion")
                if isinstance(raw_item, Mapping)
                else getattr(raw_item, "criterion", None)
            )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "each task rubric item must contain a non-empty 'criterion'"
                )
            return text

        criteria = [
            JudgeCriterion(
                id=f"C-{index:03d}",
                text=criterion_text(raw_item),
            )
            for index, raw_item in enumerate(raw_items, start=1)
        ]

        if not criteria:
            raise ValueError("task rubric contains no criteria")
        return criteria

    def _question(self, task: vf.TaskData) -> str:
        if not self.config.question_field:
            return task.prompt_text
        question = getattr(task, self.config.question_field, None)
        if question is None:
            raise ValueError(
                f"Rubric judge found no {self.config.question_field!r} "
                "question field on the task"
            )
        return str(question)

    def _response(self, trace: vf.Trace) -> str:
        return (
            trace.transcript if self.config.view == "full_trace" else trace.last_reply
        )

    async def _grade_batch(
        self,
        *,
        trace: vf.Trace,
        question: str,
        response: str,
        batch: list[JudgeCriterion],
    ) -> dict[str, float]:
        rendered_criteria = "\n".join(
            f"- ID: {criterion.id}\n"
            "  Match criteria: PASS if the response satisfies this requirement: "
            f"{criterion.text} FAIL if it does not."
            for criterion in batch
        )
        async for attempt in retrying(
            on=ValueError,
            retries=self.config.max_retries,
            label="rubric judge batch",
        ):
            with attempt:
                result = await self.evaluate(
                    trace=trace,
                    question=question,
                    response=response,
                    criteria=rendered_criteria,
                    negative_verdict=self.config.negative_verdict,
                    positive_verdict=self.config.positive_verdict,
                )
                verdicts = result.parsed or []

                by_id = {criterion.id: criterion for criterion in batch}
                actual_ids = sorted(verdict.id for verdict in verdicts)
                expected_ids = sorted(by_id)
                if actual_ids != expected_ids:
                    raise ValueError(
                        f"judge returned verdicts for {actual_ids}; "
                        f"expected {expected_ids}"
                    )

                negative = self.config.negative_verdict.strip().casefold()
                positive = self.config.positive_verdict.strip().casefold()
                scores: dict[str, float] = {}
                for verdict in verdicts:
                    normalized = verdict.verdict.strip().casefold()
                    if normalized == positive:
                        scores[verdict.id] = 1.0
                    elif normalized == negative:
                        scores[verdict.id] = 0.0
                    else:
                        raise ValueError(
                            f"judge returned verdict {verdict.verdict!r} for "
                            f"{verdict.id!r}; expected "
                            f"{self.config.negative_verdict!r} or "
                            f"{self.config.positive_verdict!r}"
                        )
                return scores

        raise RuntimeError("rubric judge retry loop ended without a result")

    async def score(self, task: vf.TaskData, trace: vf.Trace) -> float:
        criteria = self._criteria(task)
        response = self._response(trace)
        if not response.strip():
            return 0.0
        question = self._question(task)

        batch_size = self.config.max_criteria
        batches = (
            [criteria]
            if batch_size is None
            else [
                criteria[start : start + batch_size]
                for start in range(0, len(criteria), batch_size)
            ]
        )
        pending = [
            asyncio.ensure_future(
                self._grade_batch(
                    trace=trace,
                    question=question,
                    response=response,
                    batch=batch,
                )
            )
            for batch in batches
        ]
        try:
            results = await asyncio.gather(*pending)
        except BaseException:
            for future in pending:
                future.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise

        scores = {
            name: score
            for batch_scores in results
            for name, score in batch_scores.items()
        }
        for criterion in criteria:
            trace.record_metric(
                f"{self.reward_name}/{criterion.id}", scores[criterion.id]
            )

        total_weight = sum(criterion.weight for criterion in criteria)
        return (
            sum(criterion.weight * scores[criterion.id] for criterion in criteria)
            / total_weight
        )
