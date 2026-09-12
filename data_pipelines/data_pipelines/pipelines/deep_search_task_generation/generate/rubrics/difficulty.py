from dataclasses import dataclass
from enum import StrEnum

MAX_REJECTED_SOLVER_PASS_PERCENT = 10


class DifficultyBandName(StrEnum):
    EASY = "easy"
    MIDDLE = "middle"
    HARD = "hard"
    VERY_HARD = "very_hard"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DifficultyBand:
    name: DifficultyBandName
    lower: int
    upper: int
    includes_upper: bool = False

    @property
    def label(self) -> str:
        upper_prefix = "" if self.includes_upper else "<"
        return f"{self.lower}-{upper_prefix}{self.upper}"


DIFFICULTY_BANDS = (
    DifficultyBand(DifficultyBandName.EASY, 85, 100, True),
    DifficultyBand(DifficultyBandName.MIDDLE, 60, 85),
    DifficultyBand(DifficultyBandName.HARD, 40, 60),
    DifficultyBand(DifficultyBandName.VERY_HARD, 0, 40),
)


def difficulty_band(percent_passed: float) -> DifficultyBandName:
    """Classify a percentage validated at the solver-audit boundary."""
    return next(band.name for band in DIFFICULTY_BANDS if percent_passed >= band.lower)


def difficulty_thresholds() -> dict[DifficultyBandName, str]:
    return {band.name: band.label for band in DIFFICULTY_BANDS}


def calibration_band_description() -> str:
    return "; ".join(
        f"{band.label}% is {band.name.replace('_', ' ')}" for band in DIFFICULTY_BANDS
    )
