"""Dataset-config dispatch for normalized RewardScope examples."""

from __future__ import annotations

from rewardscope.config import DatasetConfig
from rewardscope.datasets.gsm8k import load_gsm8k_examples
from rewardscope.datasets.schema import DatasetExample


def load_dataset_examples(config: DatasetConfig) -> list[DatasetExample]:
    """Load examples for one supported dataset configuration."""
    if not isinstance(config, DatasetConfig):
        raise TypeError("config must be a DatasetConfig.")
    if config.name.lower() != "gsm8k":
        raise ValueError(f"Unsupported dataset: {config.name!r}. Supported datasets: gsm8k.")
    return load_gsm8k_examples(
        split=config.split,
        max_prompts=config.max_prompts,
    )
