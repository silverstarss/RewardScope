"""Dataset adapters that produce RewardScope examples."""

from rewardscope.datasets.gsm8k import (
    DEFAULT_GSM8K_PROMPT_TEMPLATE,
    load_gsm8k_examples,
)
from rewardscope.datasets.load import load_dataset_examples
from rewardscope.datasets.schema import DatasetExample

__all__ = [
    "DEFAULT_GSM8K_PROMPT_TEMPLATE",
    "DatasetExample",
    "load_dataset_examples",
    "load_gsm8k_examples",
]
