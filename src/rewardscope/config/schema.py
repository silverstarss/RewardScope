"""Validated configuration objects for RewardScope runs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

from rewardscope.rewards import RewardConfig


@dataclass(frozen=True)
class ModelConfig:
    name: str
    tokenizer_name: str | None = None
    prompt_format: Literal["chat", "plain", "auto"] = "auto"
    context_window: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str("name", self.name)
        _require_optional_non_empty_str("tokenizer_name", self.tokenizer_name)
        if self.prompt_format not in {"chat", "plain", "auto"}:
            raise ValueError("prompt_format must be one of: chat, plain, auto.")
        _require_optional_positive_int("context_window", self.context_window)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split: str
    max_prompts: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str("name", self.name)
        _require_non_empty_str("split", self.split)
        _require_optional_positive_int("max_prompts", self.max_prompts)


@dataclass(frozen=True)
class SamplingConfig:
    num_samples: int
    seed: int
    temperature: float
    top_p: float
    max_new_tokens: int
    batch_size: int

    def __post_init__(self) -> None:
        _require_positive_int("num_samples", self.num_samples)
        _require_non_negative_int("seed", self.seed)
        _require_non_negative_finite_number("temperature", self.temperature)
        _require_probability("top_p", self.top_p)
        _require_positive_int("max_new_tokens", self.max_new_tokens)
        _require_positive_int("batch_size", self.batch_size)
        if self.temperature == 0 and self.num_samples != 1:
            raise ValueError("num_samples must be 1 when temperature is 0.")


@dataclass(frozen=True)
class OutputConfig:
    run_id: str
    output_dir: Path

    def __post_init__(self) -> None:
        _require_non_empty_str("run_id", self.run_id)
        if not isinstance(self.output_dir, Path):
            raise ValueError("output_dir must be a Path.")
        if not str(self.output_dir):
            raise ValueError("output_dir must not be empty.")


@dataclass(frozen=True)
class AnalysisConfig:
    strict: bool = False
    k_values: tuple[int, ...] = (1, 4, 8)

    def __post_init__(self) -> None:
        if not isinstance(self.strict, bool):
            raise ValueError("strict must be a boolean.")
        _require_k_values(self.k_values)


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    dataset: DatasetConfig
    sampling: SamplingConfig
    reward: RewardConfig
    output: OutputConfig
    analysis: AnalysisConfig

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelConfig):
            raise ValueError("model must be a ModelConfig.")
        if not isinstance(self.dataset, DatasetConfig):
            raise ValueError("dataset must be a DatasetConfig.")
        if not isinstance(self.sampling, SamplingConfig):
            raise ValueError("sampling must be a SamplingConfig.")
        if not isinstance(self.reward, RewardConfig):
            raise ValueError("reward must be a RewardConfig.")
        if not isinstance(self.output, OutputConfig):
            raise ValueError("output must be an OutputConfig.")
        if not isinstance(self.analysis, AnalysisConfig):
            raise ValueError("analysis must be an AnalysisConfig.")
        if any(k > self.sampling.num_samples for k in self.analysis.k_values):
            raise ValueError("analysis.k_values cannot exceed sampling.num_samples.")


def _require_non_empty_str(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_optional_non_empty_str(name: str, value: object) -> None:
    if value is not None:
        _require_non_empty_str(name, value)


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _require_optional_positive_int(name: str, value: object) -> None:
    if value is not None:
        _require_positive_int(name, value)


def _require_non_negative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _require_non_negative_finite_number(name: str, value: object) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number.")


def _require_probability(name: str, value: object) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or not 0 < value <= 1
    ):
        raise ValueError(f"{name} must be a number in the interval (0, 1].")


def _require_k_values(value: object) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError("k_values must be a non-empty tuple of positive integers.")
    if any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in value):
        raise ValueError("k_values must be a non-empty tuple of positive integers.")
    if len(set(value)) != len(value):
        raise ValueError("k_values must not contain duplicates.")
