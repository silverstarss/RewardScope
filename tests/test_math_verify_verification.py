import json
from fractions import Fraction

import pytest

from rewardscope import (
    MathVerifyLatexVerifier,
    MathVerifyNumericVerifier,
    RewardConfig,
    RolloutInput,
    build_math_verify_numeric_rollout,
    build_numeric_rollout,
    rescore_completed_run_with_math_verify,
    write_rollouts_jsonl,
)
from rewardscope.verification import math_verify


class FakeLatexExtractionConfig:
    def __init__(self, *, boxed_match_priority=50):
        self.boxed_match_priority = boxed_match_priority


class FakeExprExtractionConfig:
    pass


def fake_parse(text, *, extraction_config, raise_on_error):
    assert raise_on_error is True
    if text == "INTERNAL_ERROR":
        raise RuntimeError("parser exploded")
    boxed = math_verify._boxed_contents(text)
    candidate = boxed[-1] if boxed else text.rsplit(" ", 1)[-1]
    values = {
        "42": Fraction(42),
        "1/2": Fraction(1, 2),
        "1": Fraction(1),
        "2/4": Fraction(1, 2),
        "8": Fraction(8),
        r"\{1,2\}": r"\{1,2\}",
        "(1,2]": "(1,2]",
        r"\text{yes}": r"\text{yes}",
        "bad": None,
    }
    value = values.get(candidate.strip())
    return [] if value is None else [value]


def fake_verify(gold, prediction):
    return gold[0] == prediction[0]


@pytest.fixture(autouse=True)
def fake_math_verify_backend(monkeypatch):
    backend = math_verify._MathVerifyBackend(
        parse=fake_parse,
        verify=fake_verify,
        LatexExtractionConfig=FakeLatexExtractionConfig,
        ExprExtractionConfig=FakeExprExtractionConfig,
    )
    monkeypatch.setattr(math_verify, "_load_math_verify_backend", lambda: backend)


def test_evaluation_prefers_a_boxed_math_answer_over_earlier_text():
    result = MathVerifyNumericVerifier(mode="evaluation").verify(
        "Answer: To solve this use 99.\\n\\boxed{2/4}", "1/2"
    )

    assert result.is_correct is True
    assert result.extraction.extraction_status.value == "boxed"
    assert result.extraction.normalized_answer == "1/2"
    assert result.extraction.format_ok is True


def test_evaluation_allows_an_unboxed_expression_but_marks_format_incomplete():
    result = MathVerifyNumericVerifier(mode="evaluation").verify("Answer: 2/4", "1/2")

    assert result.is_correct is True
    assert result.extraction.format_ok is False


def test_training_requires_a_parseable_boxed_answer_for_reward():
    record = build_math_verify_numeric_rollout(
        _rollout_input(response="Answer: 2/4"),
        reward_config=RewardConfig(),
        mode="training",
    )

    assert record.verification.is_correct is False
    assert record.verification.error_type == "missing_boxed_answer"
    assert record.reward.final_reward == 0.0


def test_training_accepts_math_equivalent_boxed_answer():
    result = MathVerifyNumericVerifier(mode="training").verify(r"\boxed{2/4}", "1/2")

    assert result.is_correct is True
    assert result.extraction.format_ok is True


def test_training_distinguishes_malformed_box_from_missing_box():
    result = MathVerifyNumericVerifier(mode="training").verify(r"\boxed{bad}", "1/2")

    assert result.is_correct is False
    assert result.error_type == "boxed_answer_parse_error"


def test_latex_verifier_accepts_a_set_gold_without_numeric_coercion():
    result = MathVerifyLatexVerifier(mode="training").verify(
        r"\boxed{\{1,2\}}", r"\boxed{\{1,2\}}"
    )

    assert result.is_correct is True
    assert result.extraction.parsed_value == r"\{1,2\}"


def test_latex_verifier_remains_boxed_only_in_training_mode():
    result = MathVerifyLatexVerifier(mode="training").verify(
        r"The answer is \{1,2\}.", r"\boxed{\{1,2\}}"
    )

    assert result.is_correct is False
    assert result.error_type == "missing_boxed_answer"


@pytest.mark.parametrize(
    "answer",
    [r"\{1,2\}", "(1,2]", r"\text{yes}"],
)
def test_latex_verifier_retains_non_numeric_gold_shapes(answer):
    result = MathVerifyLatexVerifier(mode="training").verify(
        rf"\boxed{{{answer}}}", rf"\boxed{{{answer}}}"
    )

    assert result.is_correct is True


def test_latex_verifier_does_not_reward_a_partial_set_answer():
    result = MathVerifyLatexVerifier(mode="training").verify(
        r"\boxed{1}", r"\boxed{\{1,2\}}"
    )

    assert result.is_correct is False


def test_math_verify_internal_parser_errors_are_not_silently_scored_as_wrong_answers():
    with pytest.raises(RuntimeError, match="parser exploded"):
        MathVerifyNumericVerifier().verify("INTERNAL_ERROR", "42")


def test_invalid_gsm8k_ground_truth_is_a_schema_error():
    with pytest.raises(ValueError, match="ground_truth"):
        MathVerifyNumericVerifier().verify(r"\boxed{42}", "not-a-number")


def test_math_verify_rescore_writes_an_offline_comparison_without_sampling(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_rollouts_jsonl(
        source / "rollouts.jsonl",
        [build_numeric_rollout(_rollout_input(response=r"\boxed{2/4}"))],
    )
    (source / "config_snapshot.json").write_text(
        json.dumps(
            {
                "resolved": {
                    "sampling": {"num_samples": 1},
                    "analysis": {"k_values": [1], "strict": True},
                    "reward": {
                        "correct_answer_reward": 1.0,
                        "incorrect_answer_reward": 0.0,
                        "format_compliance_reward": 0.0,
                        "length_penalty_per_token": 0.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    artifacts = rescore_completed_run_with_math_verify(source)

    comparison = json.loads(artifacts.comparison_json.read_text(encoding="utf-8"))
    rescored = json.loads(artifacts.rescored_rollouts_jsonl.read_text(encoding="utf-8"))
    assert comparison["verifier"] == "math_verify"
    assert comparison["mode"] == "evaluation"
    assert rescored["finish_reason"] == "eos"


def _rollout_input(*, response: str) -> RolloutInput:
    return RolloutInput(
        run_id="run", prompt_id="prompt", sample_id=0, model_name="model",
        dataset_name="gsm8k", split="test", generation_seed=1,
        temperature=0.0, top_p=1.0, max_new_tokens=32, batch_size=1,
        prompt="question", response=response, ground_truth="1/2",
        prompt_tokens=4, response_tokens=3, hit_max_length=False, finish_reason="eos",
    )
