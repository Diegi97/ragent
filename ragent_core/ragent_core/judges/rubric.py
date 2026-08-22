import asyncio
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any, Self

import verifiers.v1 as vf
from pydantic import model_validator
from verifiers.v1.judges.rubric import Criterion, CriterionVerdict
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

For each criterion, provide a brief explanation of your assessment and the verdict that best matches it.

Return exactly one evaluation per criterion, using each criterion's exact name. The
number of <criterion> children must equal the number of supplied criteria: one supplied
criterion requires one child, and four supplied criteria require four children. For
example, a batch of two criteria has this XML structure:

<criteria>
  <criterion>
    <name>exact name of the first supplied criterion</name>
    <reason>Brief explanation of the assessment</reason>
    <verdict>selected verdict</verdict>
  </criterion>
  <criterion>
    <name>exact name of the second supplied criterion</name>
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
    for field in ("name", "reason", "verdict")
}


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
                for field in ("name", "reason", "verdict"):
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

    def _criteria(self, task: vf.TaskData) -> list[Criterion]:
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
            Criterion(
                name=f"criterion_{index:02d}",
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
        batch: list[Criterion],
    ) -> dict[str, float]:
        rendered_criteria = "\n".join(
            f"- {criterion.name}: {criterion.text}" for criterion in batch
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

                by_name = {criterion.name: criterion for criterion in batch}
                actual_names = sorted(verdict.name for verdict in verdicts)
                expected_names = sorted(by_name)
                if actual_names != expected_names:
                    raise ValueError(
                        f"judge returned verdicts for {actual_names}; "
                        f"expected {expected_names}"
                    )

                negative = self.config.negative_verdict.strip().casefold()
                positive = self.config.positive_verdict.strip().casefold()
                scores: dict[str, float] = {}
                for verdict in verdicts:
                    normalized = verdict.verdict.strip().casefold()
                    if normalized == positive:
                        scores[verdict.name] = 1.0
                    elif normalized == negative:
                        scores[verdict.name] = 0.0
                    else:
                        raise ValueError(
                            f"judge returned verdict {verdict.verdict!r} for "
                            f"{verdict.name!r}; expected "
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
                f"{self.reward_name}/{criterion.name}", scores[criterion.name]
            )

        total_weight = sum(criterion.weight for criterion in criteria)
        return (
            sum(criterion.weight * scores[criterion.name] for criterion in criteria)
            / total_weight
        )
