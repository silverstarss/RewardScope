"""Normalized dataset example contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetExample:
    """One dataset item ready to become one prompt's rollout group."""

    dataset_name: str
    split: str
    source_index: int
    prompt_id: str
    question: str
    prompt: str
    ground_truth: str
    reference_solution: str

    def __post_init__(self) -> None:
        for name in (
            "dataset_name",
            "split",
            "prompt_id",
            "question",
            "prompt",
            "ground_truth",
            "reference_solution",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if (
            not isinstance(self.source_index, int)
            or isinstance(self.source_index, bool)
            or self.source_index < 0
        ):
            raise ValueError("source_index must be a non-negative integer.")


@dataclass(frozen=True)
class DatasetLoadResult:
    """Selected examples plus reproducibility metadata from a dataset adapter."""

    examples: tuple[DatasetExample, ...]
    source_count: int
    fingerprint: str | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_count, int)
            or isinstance(self.source_count, bool)
            or self.source_count < 0
        ):
            raise ValueError("source_count must be a non-negative integer.")
        if self.fingerprint is not None and (
            not isinstance(self.fingerprint, str) or not self.fingerprint
        ):
            raise ValueError("fingerprint must be a non-empty string or None.")
