"""Dataset adapters that produce RewardScope examples."""

from rewardscope.datasets.gsm8k import (
    DEFAULT_GSM8K_PROMPT_TEMPLATE,
    GSM8K_COT_4SHOT_PROMPT_TEMPLATE,
    STRICT_GSM8K_PROMPT_TEMPLATE,
    load_gsm8k_examples,
)
from rewardscope.datasets.load import load_dataset_examples, load_dataset_result
from rewardscope.datasets.schema import DatasetExample, DatasetLoadResult

__all__ = [
    "DEFAULT_GSM8K_PROMPT_TEMPLATE",
    "GSM8K_COT_4SHOT_PROMPT_TEMPLATE",
    "STRICT_GSM8K_PROMPT_TEMPLATE",
    "DatasetExample",
    "DatasetLoadResult",
    "load_dataset_examples",
    "load_dataset_result",
    "load_gsm8k_examples",
]
