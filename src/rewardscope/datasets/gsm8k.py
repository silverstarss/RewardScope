"""GSM8K normalization and optional Hugging Face dataset loading."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rewardscope.datasets.schema import DatasetExample
from rewardscope.extraction import extract_numeric_answer


DEFAULT_GSM8K_PROMPT_TEMPLATE = """Solve the following problem.
Give the final answer as: Answer: <number>

Question: {question}
"""


def load_gsm8k_examples(
    split: str,
    *,
    max_prompts: int | None = None,
    prompt_template: str = DEFAULT_GSM8K_PROMPT_TEMPLATE,
) -> list[DatasetExample]:
    """Load and normalize one GSM8K ``main`` split into RewardScope examples."""
    _require_non_empty_str("split", split)
    _require_optional_positive_int("max_prompts", max_prompts)
    _require_prompt_template(prompt_template)

    dataset = _load_hf_dataset(split)
    limit = len(dataset) if max_prompts is None else min(len(dataset), max_prompts)
    return [
        _normalize_gsm8k_row(
            row=dataset[index],
            split=split,
            index=index,
            prompt_template=prompt_template,
        )
        for index in range(limit)
    ]


def _load_hf_dataset(split: str) -> Sequence[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'GSM8K loading requires the optional data dependency. Run: pip install -e ".[data]"'
        ) from error

    return load_dataset("openai/gsm8k", "main", split=split)


def _normalize_gsm8k_row(
    *,
    row: object,
    split: str,
    index: int,
    prompt_template: str,
) -> DatasetExample:
    if not isinstance(row, dict):
        raise ValueError(f"GSM8K {split} example {index} must be a mapping.")
    question = row.get("question")
    reference_solution = row.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"GSM8K {split} example {index} has an invalid question.")
    if not isinstance(reference_solution, str) or not reference_solution.strip():
        raise ValueError(f"GSM8K {split} example {index} has an invalid answer.")

    extraction = extract_numeric_answer(reference_solution)
    if not extraction.extraction_ok or extraction.normalized_answer is None:
        raise ValueError(
            f"GSM8K {split} example {index} has no parseable final numeric answer."
        )

    return DatasetExample(
        dataset_name="gsm8k",
        split=split,
        prompt_id=f"gsm8k-{split}-{index:06d}",
        question=question,
        prompt=prompt_template.format(question=question),
        ground_truth=extraction.normalized_answer,
        reference_solution=reference_solution,
    )


def _require_non_empty_str(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_optional_positive_int(name: str, value: object) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer or None.")


def _require_prompt_template(prompt_template: object) -> None:
    _require_non_empty_str("prompt_template", prompt_template)
    if "{question}" not in prompt_template:
        raise ValueError("prompt_template must include a {question} placeholder.")
    try:
        prompt_template.format(question="example")
    except (KeyError, ValueError) as error:
        raise ValueError("prompt_template must be format-compatible with {question}.") from error
