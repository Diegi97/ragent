import re
from pathlib import Path
from typing import cast

from verifiers.v1.judge import (
    Judge,
    JudgeConfig,
    JudgeResponse,
    JudgeView,
    judge_question,
    judge_response,
)
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import ID

CORRECTNESS_PROMPT = (
    Path(__file__).resolve().parent / "answer_correctness.txt"
).read_text(encoding="utf-8")
JUDGMENT_RE = re.compile(
    r"<judgment>\s*(CORRECT|INCORRECT)\s*</judgment>", re.IGNORECASE
)


class AnswerCorrectnessJudgeConfig(JudgeConfig):
    id: ID = "answer_correctness"
    """Pinned to the built-in, so a code-level default entry needs no explicit id."""
    answer_field: str = "answer"
    """The task field holding the authoritative ground-truth answer."""
    question_field: str = ""
    """Task field to fill `{question}`; empty uses `TaskData.prompt_text`."""
    view: JudgeView = "last_reply"
    """How much of the rollout fills `{response}` (see `JudgeView`)."""


class AnswerCorrectnessJudge(Judge[float, AnswerCorrectnessJudgeConfig]):
    prompt = CORRECTNESS_PROMPT

    def parse(self, response: JudgeResponse[float]) -> float:
        judgments = JUDGMENT_RE.findall(response.text)
        if len(judgments) != 1:
            raise ValueError(
                "correctness judge expected exactly one CORRECT/INCORRECT judgment tag, "
                f"got: {response.text!r}"
            )
        return float(judgments[0].upper() == "CORRECT")

    async def score(self, task: TaskData, trace: Trace) -> float:
        answer = getattr(task, self.config.answer_field, None)
        if answer is None:
            raise ValueError(
                f"correctness judge found no {self.config.answer_field!r} field on the task; "
                "point `answer_field` at the task's ground-truth field"
            )
        if isinstance(answer, (list, tuple)):
            answer = "\n".join(str(item) for item in answer)
        response = judge_response(trace, self.config.view)
        if not response.strip():
            return 0.0
        result = await self.evaluate(
            trace=trace,
            question=judge_question(task, self.config.question_field),
            answer=answer,
            response=response,
        )
        return cast(float, result.parsed)


__all__ = ["AnswerCorrectnessJudge", "AnswerCorrectnessJudgeConfig"]
