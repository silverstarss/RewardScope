"""Shared data schemas for RewardScope rollout diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    """Structured result returned by an answer verifier."""

    extracted_answer: str | None
    extraction_ok: bool
    format_ok: bool
    is_correct: bool
    error_type: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RolloutRecord:
    """One generated response plus verifier, reward, and token metadata."""

    prompt_id: str
    sample_id: int
    model_name: str
    dataset_name: str
    split: str
    seed: int
    temperature: float
    top_p: float
    max_new_tokens: int
    prompt: str
    response: str
    ground_truth: str
    extracted_answer: str | None
    extraction_ok: bool
    format_ok: bool
    raw_correctness: bool
    format_reward: float
    length_penalty: float
    final_reward: float
    prompt_tokens: int
    response_tokens: int
    hit_max_length: bool
    latency_seconds: float
    verifier_error_type: str | None

    def __post_init__(self) -> None:
        """Validate fields that affect metrics and cost accounting."""
        _require_non_negative_int("sample_id", self.sample_id)
        _require_non_negative_int("seed", self.seed)
        _require_non_negative_int("max_new_tokens", self.max_new_tokens)
        _require_non_negative_int("prompt_tokens", self.prompt_tokens)
        _require_non_negative_int("response_tokens", self.response_tokens)
        _require_non_negative_float("temperature", self.temperature)
        _require_probability("top_p", self.top_p)
        _require_non_negative_float("latency_seconds", self.latency_seconds)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


def _require_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _require_non_negative_float(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative number.")


def _require_probability(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number between 0 and 1.")
