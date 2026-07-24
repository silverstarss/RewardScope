import pytest

from rewardscope import RewardConfig, compute_reward, verify_numeric_answer


def test_default_reward_is_one_for_a_correct_answer():
    verification = verify_numeric_answer("The final answer is 42.", "42")

    reward = compute_reward(verification, response_tokens=6)

    assert reward.correctness_reward == 1.0
    assert reward.format_reward == 0.0
    assert reward.length_penalty == 0.0
    assert reward.final_reward == 1.0


def test_default_reward_is_zero_for_an_incorrect_answer():
    verification = verify_numeric_answer("The final answer is 41.", "42")

    reward = compute_reward(verification, response_tokens=6)

    assert reward.correctness_reward == 0.0
    assert reward.final_reward == 0.0


def test_format_reward_is_independent_from_correctness():
    verification = verify_numeric_answer("Answer: 41", "42")
    config = RewardConfig(
        incorrect_answer_reward=-0.5,
        format_compliance_reward=0.2,
        length_penalty_per_token=0.01,
    )

    reward = compute_reward(verification, response_tokens=10, config=config)

    assert reward.correctness_reward == -0.5
    assert reward.format_reward == 0.2
    assert reward.length_penalty == pytest.approx(-0.1)
    assert reward.final_reward == pytest.approx(-0.4)


def test_implicit_terminal_answer_is_correct_but_receives_no_format_reward():
    verification = verify_numeric_answer("2 + 40 = 42", "42")
    config = RewardConfig(format_compliance_reward=0.2)

    reward = compute_reward(verification, response_tokens=5, config=config)

    assert verification.is_correct is True
    assert reward.correctness_reward == 1.0
    assert reward.format_reward == 0.0
    assert reward.final_reward == 1.0


def test_length_penalty_reduces_final_reward_by_response_length():
    verification = verify_numeric_answer("The final answer is 42.", "42")
    config = RewardConfig(length_penalty_per_token=0.02)

    reward = compute_reward(verification, response_tokens=25, config=config)

    assert reward.length_penalty == pytest.approx(-0.5)
    assert reward.final_reward == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("field_name", "value", "error_message"),
    [
        ("correct_answer_reward", float("nan"), "must be a finite number"),
        ("format_compliance_reward", float("inf"), "must be a finite number"),
        (
            "length_penalty_per_token",
            -0.1,
            "length_penalty_per_token must be non-negative",
        ),
    ],
)
def test_reward_config_validates_weights(field_name, value, error_message):
    with pytest.raises(ValueError, match=error_message):
        RewardConfig(**{field_name: value})


@pytest.mark.parametrize("response_tokens", [-1, "6", True])
def test_compute_reward_requires_non_negative_integer_response_tokens(response_tokens):
    verification = verify_numeric_answer("Answer: 42", "42")

    with pytest.raises(ValueError, match="response_tokens must be a non-negative integer"):
        compute_reward(verification, response_tokens=response_tokens)


def test_compute_reward_validates_its_object_inputs():
    verification = verify_numeric_answer("Answer: 42", "42")

    with pytest.raises(TypeError, match="verification must be a VerificationResult"):
        compute_reward(None, response_tokens=1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="config must be a RewardConfig"):
        compute_reward(verification, response_tokens=1, config=None)  # type: ignore[arg-type]
