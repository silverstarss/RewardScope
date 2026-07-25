"""GSM8K normalization and optional Hugging Face dataset loading."""

from __future__ import annotations

from collections.abc import Sequence
from random import Random
from typing import Any

from rewardscope.datasets.schema import DatasetExample, DatasetLoadResult
from rewardscope.extraction import extract_numeric_answer


DEFAULT_GSM8K_PROMPT_TEMPLATE = """Solve the following problem.
Give the final answer as: Answer: <number>

Question: {question}
"""

STRICT_GSM8K_PROMPT_TEMPLATE = """Solve the following problem.

Question: {question}

End your response with exactly one final line in this exact format:
Answer: <number>
In <number>, use only the numeric value: no units, commas, currency symbols, percent signs, or explanations.
"""

# Four fixed demonstrations make this a reproducible GSM8K CoT sanity prompt.
GSM8K_COT_4SHOT_PROMPT_TEMPLATE = """Question: There are 15 trees in the grove. Grove workers will plant trees today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. #### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. #### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. #### 39

Question: Jason had 20 tennis balls. He bought 2 more cans of tennis balls. Each can had 3 tennis balls. How many tennis balls does he have now?
Answer: Jason started with 20 tennis balls. 2 cans of 3 tennis balls each is 6 tennis balls. 20 + 6 = 26. The answer is 26. #### 26

Question: {question}
Answer:
"""

GSM8K_COT_4SHOT_TERMINAL_PROMPT_TEMPLATE = GSM8K_COT_4SHOT_PROMPT_TEMPLATE.removesuffix(
    "Question: {question}\nAnswer:\n"
) + """Question: {question}

Solve the problem step by step.
End your response with exactly one final line in this format:

#### <number>

The final line must contain only the number after ####.
Do not include units, commas, currency symbols, percent signs, or explanations on the final line.
"""


def load_gsm8k_examples(
    split: str,
    *,
    config_name: str | None = "main",
    revision: str | None = None,
    max_examples: int | None = None,
    selection: str = "first",
    dataset_seed: int = 0,
    source_indices: tuple[int, ...] | None = None,
    prompt_template: str = DEFAULT_GSM8K_PROMPT_TEMPLATE,
) -> list[DatasetExample]:
    """Load and normalize one GSM8K ``main`` split into RewardScope examples."""
    return list(
        load_gsm8k_result(
            config_name=config_name,
            split=split,
            revision=revision,
            max_examples=max_examples,
            selection=selection,
            dataset_seed=dataset_seed,
            source_indices=source_indices,
            prompt_template=prompt_template,
        ).examples
    )


def load_gsm8k_result(
    *, config_name: str | None, split: str, revision: str | None,
    max_examples: int | None, selection: str, dataset_seed: int,
    source_indices: tuple[int, ...] | None = None,
    prompt_template: str = DEFAULT_GSM8K_PROMPT_TEMPLATE,
) -> DatasetLoadResult:
    """Load GSM8K while preserving selection and source metadata."""
    _require_non_empty_str("split", split)
    _require_optional_non_empty_str("config_name", config_name)
    _require_optional_non_empty_str("revision", revision)
    _require_optional_positive_int("max_examples", max_examples)
    _require_selection(selection)
    _require_non_negative_int("dataset_seed", dataset_seed)
    _require_optional_source_indices(source_indices)
    _require_prompt_template(prompt_template)

    dataset = _load_hf_dataset(split, config_name=config_name, revision=revision)
    selected_indices = _select_indices(
        len(dataset), max_examples=max_examples, selection=selection, dataset_seed=dataset_seed,
        source_indices=source_indices,
    )
    examples = [
        _normalize_gsm8k_row(
            row=dataset[index],
            split=split,
            index=index,
            prompt_template=prompt_template,
        )
        for index in selected_indices
    ]
    fingerprint = getattr(dataset, "_fingerprint", None)
    return DatasetLoadResult(
        examples=tuple(examples), source_count=len(dataset),
        fingerprint=fingerprint if isinstance(fingerprint, str) and fingerprint else None,
    )


def _load_hf_dataset(
    split: str, *, config_name: str | None = "main", revision: str | None = None
) -> Sequence[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'GSM8K loading requires the optional data dependency. Run: pip install -e ".[data]"'
        ) from error

    return load_dataset("openai/gsm8k", config_name, split=split, revision=revision)


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
        source_index=index,
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


def _require_optional_non_empty_str(name: str, value: object) -> None:
    if value is not None:
        _require_non_empty_str(name, value)


def _require_non_negative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _require_optional_source_indices(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, tuple) or not value:
        raise ValueError("source_indices must be a non-empty tuple of non-negative integers or None.")
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in value):
        raise ValueError("source_indices must be a non-empty tuple of non-negative integers or None.")
    if len(set(value)) != len(value):
        raise ValueError("source_indices must not contain duplicates.")


def _require_selection(value: object) -> None:
    if value not in {"first", "random"}:
        raise ValueError("selection must be one of: first, random.")


def _select_indices(
    source_count: int, *, max_examples: int | None, selection: str, dataset_seed: int,
    source_indices: tuple[int, ...] | None = None,
) -> list[int]:
    if source_indices is not None:
        out_of_range = [index for index in source_indices if index >= source_count]
        if out_of_range:
            raise ValueError(f"source_indices contains out-of-range values: {out_of_range}.")
        return list(source_indices)
    limit = source_count if max_examples is None else min(source_count, max_examples)
    if selection == "first":
        return list(range(limit))
    return sorted(Random(dataset_seed).sample(range(source_count), limit))


def _require_prompt_template(prompt_template: object) -> None:
    _require_non_empty_str("prompt_template", prompt_template)
    if "{question}" not in prompt_template:
        raise ValueError("prompt_template must include a {question} placeholder.")
    try:
        prompt_template.format(question="example")
    except (KeyError, ValueError) as error:
        raise ValueError("prompt_template must be format-compatible with {question}.") from error
