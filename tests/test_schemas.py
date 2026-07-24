from fractions import Fraction

import pytest

from rewardscope import (
    ExtractionResult,
    ExtractionStatus,
    RewardBreakdown,
    RolloutRecord,
    VerificationResult,
)


def make_extraction(**overrides):
    fields = {
        "raw_answer": "42",
        "normalized_answer": "42",
        "parsed_value": Fraction(42),
        "extraction_status": ExtractionStatus.EXPLICIT_FINAL,
        "format_ok": True,
    }
    fields.update(overrides)
    return ExtractionResult(**fields)


def make_verification(**overrides):
    fields = {
        "extraction": make_extraction(),
        "is_correct": True,
        "error_type": None,
    }
    fields.update(overrides)
    return VerificationResult(**fields)


def make_reward(**overrides):
    fields = {
        "correctness_reward": 1.0,
        "format_reward": 1.0,
        "length_penalty": 0.0,
        "final_reward": 1.0,
    }
    fields.update(overrides)
    return RewardBreakdown(**fields)


def make_rollout(**overrides):
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
        "response": "The answer is 42.",
        "ground_truth": "42",
        "verification": make_verification(),
        "reward": make_reward(),
        "prompt_tokens": 12,
        "response_tokens": 6,
        "hit_max_length": False,
    }
    fields.update(overrides)
    return RolloutRecord(**fields)


def test_extraction_result_keeps_raw_normalized_and_exact_values():
    result = make_extraction(
        raw_answer=r"\frac{1}{2}",
        normalized_answer="1/2",
        parsed_value=Fraction(1, 2),
        extraction_status=ExtractionStatus.BOXED,
    )

    assert result.extraction_ok is True
    assert result.to_dict()["parsed_value"] == "1/2"


def test_implicit_terminal_answer_is_usable_but_not_format_compliant():
    result = make_extraction(
        raw_answer="42",
        normalized_answer="42",
        parsed_value=Fraction(42),
        extraction_status=ExtractionStatus.IMPLICIT_TERMINAL,
        format_ok=False,
    )

    assert result.extraction_ok is True
    assert result.format_ok is False


def test_failed_extraction_cannot_include_a_normalized_or_parsed_answer():
    with pytest.raises(ValueError, match="Failed extraction cannot have a parsed_value"):
        make_extraction(
            raw_answer=None,
            normalized_answer=None,
            parsed_value=Fraction(42),
            extraction_status=ExtractionStatus.MISSING,
            format_ok=False,
        )


def test_correct_verification_requires_successful_extraction():
    failed_extraction = make_extraction(
        raw_answer=None,
        normalized_answer=None,
        parsed_value=None,
        extraction_status=ExtractionStatus.MISSING,
        format_ok=False,
    )

    with pytest.raises(ValueError, match="cannot be correct when extraction failed"):
        make_verification(extraction=failed_extraction)


def test_rollout_record_serializes_nested_verification_and_reward():
    record = make_rollout()

    assert record.to_dict()["verification"] == {
        "extraction": {
            "raw_answer": "42",
            "normalized_answer": "42",
            "parsed_value": "42",
            "extraction_status": "explicit_final",
            "format_ok": True,
        },
        "is_correct": True,
        "error_type": None,
    }
    assert record.to_dict()["reward"]["correctness_reward"] == 1.0


@pytest.mark.parametrize("field_name", ["prompt_tokens", "response_tokens"])
def test_token_counts_must_be_non_negative_integers(field_name):
    with pytest.raises(ValueError, match=f"{field_name} must be a non-negative integer"):
        make_rollout(**{field_name: "6"})

    with pytest.raises(ValueError, match=f"{field_name} must be a non-negative integer"):
        make_rollout(**{field_name: -1})


@pytest.mark.parametrize(
    ("value", "error_message"),
    [
        (0, r"top_p must be a number in the interval \(0, 1\]"),
        (1.5, r"top_p must be a number in the interval \(0, 1\]"),
        (float("nan"), "top_p must be a finite number"),
        (float("inf"), "top_p must be a finite number"),
    ],
)
def test_top_p_must_be_a_finite_value_in_open_closed_unit_interval(
    value, error_message
):
    with pytest.raises(ValueError, match=error_message):
        make_rollout(top_p=value)


def test_max_new_tokens_must_be_positive():
    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        make_rollout(max_new_tokens=0)


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        make_rollout(batch_size=0)


def test_reward_values_must_be_finite():
    with pytest.raises(ValueError, match="final_reward must be a finite number"):
        make_reward(final_reward=float("nan"))


def test_boolean_fields_must_be_booleans():
    with pytest.raises(ValueError, match="hit_max_length must be a boolean"):
        make_rollout(hit_max_length="no")
