import csv
import json

import pytest

from rewardscope import (
    RolloutInput,
    analyze_rollouts_jsonl,
    build_numeric_rollout,
    compute_prompt_group_metrics,
    summarize_prompt_group_metrics,
    write_analysis_report,
    write_rollouts_jsonl,
)


def make_record(**overrides):
    fields = {
        "run_id": "run-001",
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
        "response": "Answer: 42",
        "ground_truth": "42",
        "prompt_tokens": 12,
        "response_tokens": 6,
        "hit_max_length": False,
    }
    fields.update(overrides)
    return build_numeric_rollout(RolloutInput(**fields))


def test_analysis_report_writes_csv_summary_and_issues(tmp_path):
    rows = [
        make_record().to_dict(),
        make_record(sample_id=1, response="Answer: 41").to_dict(),
    ]
    result = compute_prompt_group_metrics(rows)
    summary = summarize_prompt_group_metrics(result, k_values=(1, 2))

    artifacts = write_analysis_report(tmp_path / "report", result, summary)

    with artifacts.prompt_group_metrics_csv.open(encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    assert artifacts.group_count == 1
    assert artifacts.issue_count == 0
    assert csv_rows[0]["prompt_id"] == "gsm8k-0001"
    assert csv_rows[0]["bad_case_tags"] == "multiple_valid_answers"

    summary_data = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    assert summary_data["group_count"] == 1
    assert summary_data["pass_at_k"] == {"1": 0.5, "2": 1.0}
    assert artifacts.issues_jsonl.read_text(encoding="utf-8") == ""


def test_analyze_rollouts_jsonl_reads_persisted_records_and_forwards_options(tmp_path):
    input_path = tmp_path / "rollouts.jsonl"
    write_rollouts_jsonl(input_path, [make_record()])

    result, summary = analyze_rollouts_jsonl(
        input_path, expected_group_size=2, k_values=(1,)
    )

    assert result.groups[0].sample_count == 1
    assert result.issues[0].code == "unexpected_group_size"
    assert summary is not None
    assert summary.pass_at_k == {1: 1.0}


def test_empty_analysis_writes_header_null_summary_and_empty_issue_log(tmp_path):
    result = compute_prompt_group_metrics([])
    artifacts = write_analysis_report(tmp_path, result, summary=None)

    csv_lines = artifacts.prompt_group_metrics_csv.read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == 1
    assert json.loads(artifacts.summary_json.read_text(encoding="utf-8")) is None
    assert artifacts.issues_jsonl.read_text(encoding="utf-8") == ""


def test_report_writer_validates_metric_result_and_summary_types(tmp_path):
    result = compute_prompt_group_metrics([])

    with pytest.raises(TypeError, match="result must be a PromptGroupMetricsResult"):
        write_analysis_report(tmp_path, None, None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="summary must be a PromptGroupSummary or None"):
        write_analysis_report(tmp_path, result, "not a summary")  # type: ignore[arg-type]
