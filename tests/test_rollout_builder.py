import pytest

from rewardscope import (
    RewardConfig,
    RolloutInput,
    build_numeric_rollout,
)


def make_rollout_input(**overrides):
    fields = {
        "run_id": "gsm8k-smoke-001",
        "prompt_id": "gsm8k-0001",
        "sample_id": 0,
        "model_name": "Qwen2.5-1.5B-Instruct",
        "dataset_name": "GSM8K",
        "split": "test",
        "seed": 123,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 256,
        "batch_size": 4,
        "prompt": "What is 40 + 2?",
        "response": "The final answer is 42.",
        "ground_truth": "42",
        "prompt_tokens": 12,
        "response_tokens": 6,
        "hit_max_length": False,
    }
    fields.update(overrides)
    return RolloutInput(**fields)


def test_builder_returns_complete_correct_rollout_record():
    rollout_input = make_rollout_input()

    record = build_numeric_rollout(rollout_input)

    assert record.run_id == "gsm8k-smoke-001"
    assert record.prompt_id == "gsm8k-0001"
    assert record.response == "The final answer is 42."
    assert record.verification.is_correct is True
    assert record.verification.extraction.normalized_answer == "42"
    assert record.reward.final_reward == 1.0


def test_builder_preserves_implicit_answer_format_status_in_complete_record():
    rollout_input = make_rollout_input(response="2 + 40 = 42")
    config = RewardConfig(format_compliance_reward=0.2)

    record = build_numeric_rollout(rollout_input, reward_config=config)

    assert record.verification.is_correct is True
    assert record.verification.extraction.format_ok is False
    assert record.reward.format_reward == 0.0
    assert record.reward.final_reward == 1.0


def test_builder_preserves_wrong_answer_and_reward_details():
    rollout_input = make_rollout_input(response="Answer: 41")
    config = RewardConfig(incorrect_answer_reward=-0.5)

    record = build_numeric_rollout(rollout_input, reward_config=config)

    assert record.verification.is_correct is False
    assert record.verification.error_type == "wrong_answer"
    assert record.reward.correctness_reward == -0.5
    assert record.reward.final_reward == -0.5


def test_builder_output_can_be_serialized_as_a_complete_nested_dictionary():
    record = build_numeric_rollout(make_rollout_input())

    assert record.to_dict()["verification"]["is_correct"] is True
    assert record.to_dict()["reward"]["final_reward"] == 1.0


@pytest.mark.parametrize(
    ("overrides", "error_message"),
    [
        ({"prompt_id": ""}, "prompt_id must be a non-empty string"),
        ({"response_tokens": -1}, "response_tokens must be a non-negative integer"),
        ({"top_p": 0}, r"top_p must be a number in the interval \(0, 1\]"),
    ],
)
def test_builder_preserves_rollout_record_validation(overrides, error_message):
    with pytest.raises(ValueError, match=error_message):
        build_numeric_rollout(make_rollout_input(**overrides))


def test_builder_requires_a_rollout_input():
    with pytest.raises(TypeError, match="rollout_input must be a RolloutInput"):
        build_numeric_rollout(None)  # type: ignore[arg-type]
