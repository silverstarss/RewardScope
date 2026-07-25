"""Dataset-config dispatch for normalized RewardScope examples."""

from __future__ import annotations

from rewardscope.config import DatasetConfig
from rewardscope.datasets.gsm8k import (
    DEFAULT_GSM8K_PROMPT_TEMPLATE,
    GSM8K_COT_4SHOT_PROMPT_TEMPLATE,
    GSM8K_COT_4SHOT_TERMINAL_PROMPT_TEMPLATE,
    STRICT_GSM8K_PROMPT_TEMPLATE,
    load_gsm8k_examples,
    load_gsm8k_result,
)
from rewardscope.datasets.schema import DatasetExample, DatasetLoadResult


def load_dataset_examples(config: DatasetConfig) -> list[DatasetExample]:
    """Load examples for one supported dataset configuration."""
    if not isinstance(config, DatasetConfig):
        raise TypeError("config must be a DatasetConfig.")
    if config.name.lower() != "gsm8k":
        raise ValueError(f"Unsupported dataset: {config.name!r}. Supported datasets: gsm8k.")
    return load_gsm8k_examples(
        split=config.split,
        config_name=config.config,
        revision=config.revision,
        max_examples=config.max_examples,
        selection=config.selection,
        dataset_seed=config.dataset_seed,
        source_indices=config.source_indices,
        prompt_template=_prompt_template_for(config),
    )


def load_dataset_result(config: DatasetConfig) -> DatasetLoadResult:
    """Load selected examples and adapter metadata for experiment provenance."""
    if not isinstance(config, DatasetConfig):
        raise TypeError("config must be a DatasetConfig.")
    if config.name.lower() != "gsm8k":
        raise ValueError(f"Unsupported dataset: {config.name!r}. Supported datasets: gsm8k.")
    return load_gsm8k_result(
        split=config.split, config_name=config.config, revision=config.revision,
        max_examples=config.max_examples, selection=config.selection,
        dataset_seed=config.dataset_seed,
        source_indices=config.source_indices,
        prompt_template=_prompt_template_for(config),
    )


def _prompt_template_for(config: DatasetConfig) -> str:
    templates = {
        "baseline": DEFAULT_GSM8K_PROMPT_TEMPLATE,
        "strict": STRICT_GSM8K_PROMPT_TEMPLATE,
        "gsm8k_cot_4shot": GSM8K_COT_4SHOT_PROMPT_TEMPLATE,
        "gsm8k_cot_4shot_terminal": GSM8K_COT_4SHOT_TERMINAL_PROMPT_TEMPLATE,
    }
    return templates[config.prompt_template]
