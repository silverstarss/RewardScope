from copy import deepcopy

import pytest

from rewardscope import (
    RewardConfig,
    RolloutInput,
    build_numeric_rollout,
    compute_prompt_group_metrics,
    summarize_prompt_group_metrics,
)


def make_row(**overrides):
    reward_config = overrides.pop("reward_config", RewardConfig())
    fields = {
        "run_id": "run-001",
        "prompt_id": "gsm8k-0001",
        "sample_id": 0,
        "model_name": "Qwen2.5-1.5B-Instruct",
        "dataset_name": "GSM8K",
        "split": "test",
        "generation_seed": 123,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 256,
        "batch_size": 4,
        "prompt": "What is 40 + 2?",
        "response": "Answer: 42",
        "ground_truth": "42",
        "prompt_tokens": 12,
        "response_tokens": 6,
        "hit_max_length": False,
    }
    fields.update(overrides)
    return build_numeric_rollout(
        RolloutInput(**fields), reward_config=reward_config
    ).to_dict()


def single_group(result):
    assert len(result.groups) == 1
    return result.groups[0]


def test_group_metrics_classify_mixed_groups_and_use_population_statistics():
    rows = [
        make_row(response_tokens=5),
        make_row(sample_id=1, response="Answer: 41", response_tokens=7, hit_max_length=True),
    ]

    group = single_group(compute_prompt_group_metrics(rows))

    assert group.any_correct is True
    assert group.all_correct is False
    assert group.all_wrong is False
    assert group.mixed is True
    assert group.correct_count == 1
    assert group.raw_reward_mean == pytest.approx(0.5)
    assert group.raw_reward_variance == pytest.approx(0.25)
    assert group.raw_reward_std == pytest.approx(0.5)
    assert group.raw_reward_range == pytest.approx(1.0)
    assert group.final_reward_variance == pytest.approx(0.25)
    assert group.unique_valid_answer_count == 2
    assert group.response_tokens_total == 12
    assert group.response_tokens_mean == pytest.approx(6.0)
    assert group.hit_max_length_count == 1
    assert group.hit_max_length_rate == pytest.approx(0.5)
    assert "multiple_valid_answers" in group.bad_case_tags
    assert "zero_raw_reward_variance" not in group.bad_case_tags


def test_all_wrong_uses_raw_correctness_even_when_shaping_creates_final_variance():
    config = RewardConfig(format_compliance_reward=0.2)
    rows = [
        make_row(response="Answer: 41", reward_config=config),
        make_row(sample_id=1, response="2 + 40 = 41", reward_config=config),
    ]

    group = single_group(compute_prompt_group_metrics(rows))

    assert group.all_wrong is True
    assert group.mixed is False
    assert group.raw_reward_variance == 0.0
    assert group.final_reward_variance == pytest.approx(0.01)
    assert group.final_reward_range == pytest.approx(0.2)
    assert "all_wrong" in group.bad_case_tags
    assert "zero_raw_reward_variance" in group.bad_case_tags


def test_unique_answers_only_count_successful_extractions():
    rows = [
        make_row(response="Answer: 42"),
        make_row(sample_id=1, response="43"),
        make_row(sample_id=2, response="I cannot solve this."),
    ]

    group = single_group(compute_prompt_group_metrics(rows))

    assert group.extraction_failure_count == 1
    assert group.extraction_failure_rate == pytest.approx(1 / 3)
    assert group.unique_valid_answer_count == 2
    assert group.format_error_count == 2
    assert "extraction_failures" in group.bad_case_tags


def test_same_prompt_id_from_different_runs_is_not_merged():
    rows = [
        make_row(run_id="run-a"),
        make_row(run_id="run-b"),
    ]

    result = compute_prompt_group_metrics(rows)

    assert [(group.run_id, group.prompt_id) for group in result.groups] == [
        ("run-a", "gsm8k-0001"),
        ("run-b", "gsm8k-0001"),
    ]


@pytest.mark.parametrize(
    ("field_name", "value", "expected_issue"),
    [
        ("ground_truth", "43", "inconsistent_ground_truth"),
        ("model_name", "other-model", "inconsistent_model"),
        ("dataset_name", "MATH", "inconsistent_dataset"),
        ("temperature", 0.8, "inconsistent_generation_config"),
        ("batch_size", 8, "inconsistent_generation_config"),
    ],
)
def test_non_strict_mode_records_and_excludes_inconsistent_group_members(
    field_name, value, expected_issue
):
    conflicting_row = make_row(sample_id=1)
    conflicting_row[field_name] = value

    result = compute_prompt_group_metrics([make_row(), conflicting_row])

    group = single_group(result)
    assert group.sample_count == 1
    assert expected_issue in group.bad_case_tags
    assert result.issues[0].code == expected_issue

    with pytest.raises(ValueError, match=expected_issue):
        compute_prompt_group_metrics([make_row(), conflicting_row], strict=True)


def test_duplicate_sample_ids_are_excluded_and_expected_group_size_is_checked():
    duplicate = make_row(sample_id=0)

    result = compute_prompt_group_metrics(
        [make_row(), duplicate], expected_group_size=2
    )

    group = single_group(result)
    assert group.sample_count == 1
    assert "duplicate_sample_id" in group.bad_case_tags
    assert "unexpected_group_size" in group.bad_case_tags
    assert {issue.code for issue in result.issues} == {
        "duplicate_sample_id",
        "unexpected_group_size",
    }


def test_non_strict_mode_records_malformed_rows_and_strict_mode_raises():
    malformed = deepcopy(make_row(sample_id=1))
    del malformed["run_id"]

    result = compute_prompt_group_metrics([make_row(), malformed])

    assert single_group(result).sample_count == 1
    assert result.issues[0].code == "missing_or_invalid_field"

    with pytest.raises(ValueError, match="missing_or_invalid_field"):
        compute_prompt_group_metrics([make_row(), malformed], strict=True)


def test_summary_reports_group_outcomes_pass_at_k_and_effective_token_cost():
    mixed_rows = [
        make_row(prompt_id="mixed", response_tokens=3),
        make_row(prompt_id="mixed", sample_id=1, response="Answer: 41", response_tokens=4),
    ]
    all_wrong_rows = [
        make_row(prompt_id="wrong", response="Answer: 41", response_tokens=5),
        make_row(
            prompt_id="wrong", sample_id=1, response="Answer: 40",
            response_tokens=6, hit_max_length=True,
        ),
    ]

    result = compute_prompt_group_metrics(mixed_rows + all_wrong_rows)
    summary = summarize_prompt_group_metrics(result, k_values=(1, 2, 4))

    assert summary is not None
    assert summary.group_count == 2
    assert summary.all_wrong_rate == pytest.approx(0.5)
    assert summary.all_correct_rate == 0.0
    assert summary.mixed_rate == pytest.approx(0.5)
    assert summary.pass_at_k == {1: pytest.approx(0.25), 2: pytest.approx(0.5), 4: None}
    assert summary.pass_at_k_eligible_group_count == {1: 2, 2: 2, 4: 0}
    assert summary.hit_max_length_count == 1
    assert summary.hit_max_length_rate == pytest.approx(1 / 4)
    assert summary.groups_with_hit_max_length_count == 1
    assert summary.total_response_tokens == 18
    assert summary.effective_response_tokens == 7
    assert summary.effective_token_ratio == pytest.approx(7 / 18)
    assert summary.token_cost_per_mixed_prompt == pytest.approx(18.0)
    assert summary.bad_case_counts["all_wrong"] == 1


def test_empty_result_has_no_summary():
    result = compute_prompt_group_metrics([])

    assert result.groups == ()
    assert result.issues == ()
    assert summarize_prompt_group_metrics(result) is None


def test_summary_rejects_invalid_pass_at_k_configuration():
    result = compute_prompt_group_metrics([make_row()])

    with pytest.raises(ValueError, match="k_values"):
        summarize_prompt_group_metrics(result, k_values=(1, 1))
