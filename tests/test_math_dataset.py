import pytest

from rewardscope import DatasetConfig, load_dataset_result
from rewardscope.datasets import math


FAKE_MATH_ROWS = [
    {
        "problem": "Find the set.",
        "solution": "An intermediate value is \\boxed{1}. Thus \\boxed{\\{1,2\\}}.",
        "level": "Level 1",
    },
    {
        "problem": "Find the interval.",
        "solution": "Therefore the answer is \\boxed{(1,2]}.",
        "level": "Level 2",
    },
    {
        "problem": "Bad gold.",
        "solution": "No final box is present.",
        "level": "Level 1",
    },
    {
        "problem": "Harder question.",
        "solution": "Therefore \\boxed{3}.",
        "level": "Level 3",
    },
]


def test_math_adapter_filters_unparseable_gold_and_records_audit_rate(monkeypatch):
    monkeypatch.setattr(
        math,
        "_load_math_datasets",
        lambda **kwargs: [("algebra", FAKE_MATH_ROWS)],
    )
    monkeypatch.setattr(
        math,
        "extract_final_boxed_latex_gold",
        lambda solution: None if solution.startswith("No") else solution.rsplit(" ", 1)[-1].rstrip("."),
    )

    result = load_dataset_result(
        DatasetConfig(
            name="math", config="all", split="train", max_examples=128,
            levels=("Level 1", "Level 2"), prompt_template="zero_shot_boxed",
        )
    )

    assert result.source_count == 3
    assert result.gold_parse_attempt_count == 3
    assert result.gold_parse_failure_count == 1
    assert result.gold_parse_failure_rate == pytest.approx(1 / 3)
    assert [example.source_index for example in result.examples] == [0, 1]
    assert result.examples[0].ground_truth == r"\boxed{\{1,2\}}"
    assert result.examples[0].prompt.startswith("Solve the problem step by step")


def test_math_adapter_rejects_requested_gold_that_was_filtered(monkeypatch):
    monkeypatch.setattr(
        math,
        "_load_math_datasets",
        lambda **kwargs: [("algebra", FAKE_MATH_ROWS)],
    )
    monkeypatch.setattr(math, "extract_final_boxed_latex_gold", lambda solution: None)

    with pytest.raises(ValueError, match="gold-unparseable"):
        load_dataset_result(
            DatasetConfig(
                name="math", config="all", split="train", source_indices=(0,),
                levels=("Level 1",), prompt_template="zero_shot_boxed",
            )
        )


def test_modelscope_loader_requires_its_optional_dependency(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "modelscope", None)

    with pytest.raises(RuntimeError, match="pip install 'modelscope>=1.13.2'"):
        math._load_modelscope_math_dataset(
            split="train", config_name="all", revision=None
        )
