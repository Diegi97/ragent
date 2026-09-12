import asyncio
from collections.abc import Mapping
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.utils.retries import retrying

from ragent_core.judges.criteria import Verdict, criterion_id, criterion_metric_name
from ragent_core.judges.rubric.config import RubricJudgeConfig
from ragent_core.judges.rubric.contracts import CriterionVerdict, JudgeCriterion
from ragent_core.judges.rubric.prompt import RUBRIC_PROMPT


class RubricJudge(vf.Judge[list[CriterionVerdict], RubricJudgeConfig]):
    prompt = RUBRIC_PROMPT

    def parse(
        self,
        response: vf.JudgeResponse[list[CriterionVerdict]],
    ) -> list[CriterionVerdict]:
        return CriterionVerdict.parse_xml(response.text)

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
                self.grade_batch(
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
                criterion_metric_name(criterion.id, self.reward_name),
                scores[criterion.id],
            )

        total_weight = sum(criterion.weight for criterion in criteria)
        return (
            sum(criterion.weight * scores[criterion.id] for criterion in criteria)
            / total_weight
        )

    async def grade_batch(
        self,
        *,
        trace: vf.Trace,
        question: str,
        response: str,
        batch: list[JudgeCriterion],
    ) -> dict[str, float]:
        rendered_criteria = "\n".join(
            f"- ID: {criterion.id}\n"
            f"  Match criteria: {Verdict.PASS} if the response satisfies this requirement: "
            f"{criterion.text} {Verdict.FAIL} if it does not."
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

                scores = {verdict.id: verdict.verdict.score for verdict in verdicts}
                return scores

        raise RuntimeError("rubric judge retry loop ended without a result")

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
                id=criterion_id(index),
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
