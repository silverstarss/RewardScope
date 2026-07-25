import sys

import pytest

from rewardscope import DatasetConfig, load_dataset_examples, load_gsm8k_examples
from rewardscope.datasets import gsm8k


FAKE_GSM8K_ROWS = [
    {
        "question": "What is 2 divided by 4?",
        "answer": "We divide.\n#### 2/4",
    },
    {
        "question": "What is 40 plus 2?",
        "answer": "The calculation is simple.\n#### 42",
    },
]


def test_gsm8k_adapter_normalizes_reference_answers_and_builds_stable_ids(monkeypatch):
    monkeypatch.setattr(gsm8k, "_load_hf_dataset", lambda *args, **kwargs: FAKE_GSM8K_ROWS)

    examples = load_gsm8k_examples("test")

    assert [example.prompt_id for example in examples] == [
        "gsm8k-test-000000",
        "gsm8k-test-000001",
    ]
    assert examples[0].ground_truth == "1/2"
    assert examples[0].question == "What is 2 divided by 4?"
    assert "Question: What is 2 divided by 4?" in examples[0].prompt
    assert examples[0].reference_solution.endswith("#### 2/4")


def test_gsm8k_adapter_limits_examples_deterministically(monkeypatch):
    monkeypatch.setattr(gsm8k, "_load_hf_dataset", lambda *args, **kwargs: FAKE_GSM8K_ROWS)

    examples = load_gsm8k_examples("test", max_examples=1)

    assert len(examples) == 1
    assert examples[0].prompt_id == "gsm8k-test-000000"


def test_configured_dataset_loader_dispatches_to_gsm8k_adapter(monkeypatch):
    monkeypatch.setattr(gsm8k, "_load_hf_dataset", lambda *args, **kwargs: FAKE_GSM8K_ROWS)
    config = DatasetConfig(name="GSM8K", config="main", split="test", max_examples=1)

    examples = load_dataset_examples(config)

    assert len(examples) == 1
    assert examples[0].dataset_name == "gsm8k"


def test_strict_prompt_template_uses_the_configured_terminal_answer_contract(monkeypatch):
    monkeypatch.setattr(gsm8k, "_load_hf_dataset", lambda *args, **kwargs: FAKE_GSM8K_ROWS)

    examples = load_dataset_examples(
        DatasetConfig(name="gsm8k", config="main", split="test", prompt_template="strict")
    )

    assert "exactly one final line" in examples[0].prompt
    assert "Answer: <number>" in examples[0].prompt
    assert "no units, commas, currency symbols" in examples[0].prompt


def test_cot_four_shot_template_has_fixed_demonstrations_and_an_open_answer(monkeypatch):
    monkeypatch.setattr(gsm8k, "_load_hf_dataset", lambda *args, **kwargs: FAKE_GSM8K_ROWS)

    examples = load_dataset_examples(
        DatasetConfig(
            name="gsm8k", config="main", split="test", prompt_template="gsm8k_cot_4shot"
        )
    )

    assert examples[0].prompt.count("Question:") == 5
    assert examples[0].prompt.count("####") == 4
    assert examples[0].prompt.endswith("Answer:\n")


def test_unsupported_dataset_is_rejected_without_importing_hugging_face_data():
    config = DatasetConfig(name="math", config=None, split="test")

    with pytest.raises(ValueError, match="Unsupported dataset: 'math'"):
        load_dataset_examples(config)


@pytest.mark.parametrize(
    ("row", "error_message"),
    [
        ({"answer": "#### 42"}, "invalid question"),
        ({"question": "Question"}, "invalid answer"),
        ({"question": "Question", "answer": "No final answer."}, "no parseable"),
    ],
)
def test_gsm8k_adapter_rejects_invalid_source_rows(monkeypatch, row, error_message):
    monkeypatch.setattr(gsm8k, "_load_hf_dataset", lambda *args, **kwargs: [row])

    with pytest.raises(ValueError, match=error_message):
        load_gsm8k_examples("test")


def test_gsm8k_adapter_validates_template_and_limit_before_loading_data(monkeypatch):
    def unexpected_load(split):
        raise AssertionError("dataset loader should not run")

    monkeypatch.setattr(gsm8k, "_load_hf_dataset", unexpected_load)

    with pytest.raises(ValueError, match=r"must include a \{question\} placeholder"):
        load_gsm8k_examples("test", prompt_template="Solve this")
    with pytest.raises(ValueError, match="max_examples must be a positive integer"):
        load_gsm8k_examples("test", max_examples=0)


def test_missing_datasets_dependency_reports_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)

    with pytest.raises(RuntimeError, match=r'pip install -e "\.\[data\]"'):
        gsm8k._load_hf_dataset("test")
