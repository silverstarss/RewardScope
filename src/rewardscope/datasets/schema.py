"""Normalized dataset example contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetExample:
    """One dataset item ready to become one prompt's rollout group."""

    dataset_name: str
    split: str
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
