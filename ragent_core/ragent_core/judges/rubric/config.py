from typing import Self

import verifiers.v1 as vf
from pydantic import model_validator

from ragent_core.judges.criteria import Verdict


class RubricJudgeConfig(vf.JudgeConfig):
    """Configuration placed under the task's singular ``judge`` field."""

    name: str = "rubric"
    criteria_field: str = "rubric"
    question_field: str = "question"
    view: vf.JudgeView = "last_reply"
    negative_verdict: Verdict = Verdict.FAIL
    positive_verdict: Verdict = Verdict.PASS
    max_criteria: int | None = 4
    """Maximum criteria per judge call. ``None`` grades all criteria in one call."""
    max_retries: int = 5
    """Retries after a malformed or otherwise invalid judge verdict."""

    @model_validator(mode="after")
    def validate_verdicts(self) -> Self:
        if (
            self.negative_verdict is not Verdict.FAIL
            or self.positive_verdict is not Verdict.PASS
        ):
            raise ValueError(
                "Rubric verdict configuration must use the canonical positive/negative labels"
            )
        if self.max_criteria is not None and self.max_criteria < 1:
            raise ValueError("max_criteria must be at least 1 or None")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        return self
