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


def test_strict_prompt_template_uses_the_configured_boxed_answer_contract(monkeypatch):
    monkeypatch.setattr(gsm8k, "_load_hf_dataset", lambda *args, **kwargs: FAKE_GSM8K_ROWS)

    examples = load_dataset_examples(
        DatasetConfig(name="gsm8k", config="main", split="test", prompt_template="strict")
    )

    assert "Please reason step by step" in examples[0].prompt
    assert r"\boxed{}" in examples[0].prompt


def test_zero_shot_boxed_template_uses_the_training_prompt_verbatim(monkeypatch):
    monkeypatch.setattr(gsm8k, "_load_hf_dataset", lambda *args, **kwargs: FAKE_GSM8K_ROWS)

    examples = load_dataset_examples(
        DatasetConfig(
            name="gsm8k", config="main", split="test",
            prompt_template="gsm8k_zero_shot_boxed",
        )
    )

    assert examples[0].prompt == (
        "Solve the problem step by step and put your final answer within \\boxed{}.\n\n"
        "Question: What is 2 divided by 4?\n"
    )


def test_cot_four_shot_template_has_fixed_boxed_demonstrations(monkeypatch):
    monkeypatch.setattr(gsm8k, "_load_hf_dataset", lambda *args, **kwargs: FAKE_GSM8K_ROWS)

    examples = load_dataset_examples(
        DatasetConfig(
            name="gsm8k", config="main", split="test", prompt_template="gsm8k_cot_4shot"
        )
    )

    assert examples[0].prompt.count("Question:") == 5
    assert examples[0].prompt.count(r"\boxed{") == 5
    assert examples[0].prompt.endswith(r"within \boxed{}." + "\n")


def test_terminal_cot_template_is_compatible_with_the_boxed_answer_contract(monkeypatch):
    monkeypatch.setattr(gsm8k, "_load_hf_dataset", lambda *args, **kwargs: FAKE_GSM8K_ROWS)

    examples = load_dataset_examples(
        DatasetConfig(
            name="gsm8k", config="main", split="test",
            prompt_template="gsm8k_cot_4shot_terminal",
        )
    )

    assert examples[0].prompt.count("Question:") == 5
    assert examples[0].prompt.count(r"\boxed{") == 5
    assert examples[0].prompt.endswith(r"within \boxed{}." + "\n")


def test_multiturn_terminal_template_uses_role_separated_demonstrations(monkeypatch):
    monkeypatch.setattr(gsm8k, "_load_hf_dataset", lambda *args, **kwargs: FAKE_GSM8K_ROWS)

    examples = load_dataset_examples(
        DatasetConfig(
            name="gsm8k",
            config="main",
            split="test",
            prompt_template="gsm8k_cot_4shot_multiturn_terminal",
        )
    )

    messages = examples[0].messages
    assert messages is not None
    assert [message.role for message in messages] == [
        "user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant", "user",
    ]
    assert all(message.content.splitlines()[-1].startswith(r"\boxed{") for message in messages[1:-1:2])
    assert messages[-1].content == examples[0].prompt
    assert messages[-1].content.endswith(r"within \boxed{}." + "\n")


def test_gsm8k_adapter_uses_explicit_source_indices_in_requested_order(monkeypatch):
    monkeypatch.setattr(gsm8k, "_load_hf_dataset", lambda *args, **kwargs: FAKE_GSM8K_ROWS)

    examples = load_gsm8k_examples("test", source_indices=(1, 0))

    assert [example.source_index for example in examples] == [1, 0]


def test_unsupported_dataset_is_rejected_without_importing_hugging_face_data():
    config = DatasetConfig(name="algebra", config=None, split="test")

    with pytest.raises(ValueError, match="Unsupported dataset: 'algebra'"):
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
