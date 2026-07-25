import json

import pytest

from rewardscope import (
    RolloutInput,
    build_numeric_rollout,
    read_rollouts_jsonl,
    write_rollouts_jsonl,
)


def make_record(**overrides):
    fields = {
        "run_id": "gsm8k-smoke-001",
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
        "response": "The final answer is 42.",
        "ground_truth": "42",
        "prompt_tokens": 12,
        "response_tokens": 6,
        "hit_max_length": False,
    }
    fields.update(overrides)
    return build_numeric_rollout(RolloutInput(**fields))


def test_write_then_read_preserves_complete_nested_rollout_data(tmp_path):
    path = tmp_path / "runs" / "rollouts.jsonl"
    records = [make_record(), make_record(prompt_id="gsm8k-0002", sample_id=1)]

    written_count = write_rollouts_jsonl(path, records)
    loaded_records = read_rollouts_jsonl(path)

    assert written_count == 2
    assert loaded_records[0]["prompt_id"] == "gsm8k-0001"
    assert loaded_records[1]["sample_id"] == 1
    assert loaded_records[0]["verification"]["is_correct"] is True
    assert loaded_records[0]["reward"]["final_reward"] == 1.0


def test_default_write_mode_replaces_existing_records(tmp_path):
    path = tmp_path / "rollouts.jsonl"

    write_rollouts_jsonl(path, [make_record(prompt_id="first")])
    write_rollouts_jsonl(path, [make_record(prompt_id="second")])

    assert [record["prompt_id"] for record in read_rollouts_jsonl(path)] == ["second"]


def test_append_mode_preserves_existing_records(tmp_path):
    path = tmp_path / "rollouts.jsonl"

    write_rollouts_jsonl(path, [make_record(prompt_id="first")])
    write_rollouts_jsonl(path, [make_record(prompt_id="second")], append=True)

    assert [record["prompt_id"] for record in read_rollouts_jsonl(path)] == [
        "first",
        "second",
    ]


def test_invalid_record_does_not_overwrite_existing_file(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    write_rollouts_jsonl(path, [make_record(prompt_id="original")])

    with pytest.raises(TypeError, match=r"records\[0\] must be a RolloutRecord"):
        write_rollouts_jsonl(path, ["not a record"])

    assert read_rollouts_jsonl(path)[0]["prompt_id"] == "original"


def test_reader_skips_blank_lines(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    record = make_record().to_dict()
    path.write_text(f"\n{json.dumps(record)}\n\n", encoding="utf-8")

    assert read_rollouts_jsonl(path) == [record]


def test_reader_reports_invalid_json_with_line_number(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    path.write_text('{"prompt_id": "valid"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid JSON .* line 2"):
        read_rollouts_jsonl(path)


def test_reader_rejects_non_object_json_lines(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    path.write_text('["not", "a", "record"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"Expected a JSON object .* line 1"):
        read_rollouts_jsonl(path)


def test_writer_validates_append_flag(tmp_path):
    with pytest.raises(TypeError, match="append must be a boolean"):
        write_rollouts_jsonl(tmp_path / "rollouts.jsonl", [], append="yes")
