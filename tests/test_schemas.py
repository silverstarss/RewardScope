import pytest

from rewardscope import RolloutRecord, VerificationResult


def make_rollout(**overrides):
    fields = {
        "prompt_id": "gsm8k-0001",
        "sample_id": 0,
        "model_name": "Qwen2.5-1.5B-Instruct",
        "dataset_name": "GSM8K",
        "split": "test",
        "seed": 123,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 256,
        "prompt": "What is 40 + 2?",
        "response": "The answer is 42.",
        "ground_truth": "42",
        "extracted_answer": "42",
        "extraction_ok": True,
        "format_ok": True,
        "raw_correctness": True,
        "format_reward": 1.0,
        "length_penalty": 0.0,
        "final_reward": 1.0,
        "prompt_tokens": 12,
        "response_tokens": 6,
        "hit_max_length": False,
        "latency_seconds": 0.25,
        "verifier_error_type": None,
    }
    fields.update(overrides)
    return RolloutRecord(**fields)


def test_verification_result_can_be_serialized_to_dict():
    result = VerificationResult(
        extracted_answer="42",
        extraction_ok=True,
        format_ok=True,
        is_correct=True,
        error_type=None,
    )

    assert result.to_dict() == {
        "extracted_answer": "42",
        "extraction_ok": True,
        "format_ok": True,
        "is_correct": True,
        "error_type": None,
    }


def test_rollout_record_can_be_serialized_to_dict():
    record = make_rollout()

    data = record.to_dict()

    assert data["prompt_id"] == "gsm8k-0001"
    assert data["response_tokens"] == 6
    assert data["verifier_error_type"] is None


def test_rollout_record_allows_missing_extracted_answer():
    record = make_rollout(
        extracted_answer=None,
        extraction_ok=False,
        format_ok=False,
        raw_correctness=False,
        final_reward=0.0,
        verifier_error_type="no_answer_found",
    )

    assert record.extracted_answer is None
    assert record.to_dict()["verifier_error_type"] == "no_answer_found"


@pytest.mark.parametrize("field_name", ["prompt_tokens", "response_tokens"])
def test_token_counts_must_be_non_negative_integers(field_name):
    with pytest.raises(ValueError, match=f"{field_name} must be a non-negative integer"):
        make_rollout(**{field_name: "6"})

    with pytest.raises(ValueError, match=f"{field_name} must be a non-negative integer"):
        make_rollout(**{field_name: -1})


def test_top_p_must_be_probability():
    with pytest.raises(ValueError, match="top_p must be a number between 0 and 1"):
        make_rollout(top_p=1.5)
