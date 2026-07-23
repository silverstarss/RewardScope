"""RewardScope package."""

from rewardscope.schemas import (
    ExtractionResult,
    ExtractionStatus,
    RewardBreakdown,
    RolloutRecord,
    VerificationResult,
)
from rewardscope.extraction import extract_numeric_answer, parse_numeric_value

__all__ = [
    "ExtractionResult",
    "ExtractionStatus",
    "RewardBreakdown",
    "RolloutRecord",
    "VerificationResult",
    "extract_numeric_answer",
    "parse_numeric_value",
]

from rewardscope.verification import (
    verify_extracted_numeric_answer,
    verify_numeric_answer,
)

__all__ += ["verify_extracted_numeric_answer", "verify_numeric_answer"]
