import json

from rewardscope import (
    RolloutInput,
    build_numeric_rollout,
    rescore_completed_run,
    write_rollouts_jsonl,
)


def make_record(*, sample_id: int, response: str):
    return build_numeric_rollout(
        RolloutInput(
            run_id="gsm8k-run",
            prompt_id="gsm8k-0001",
            sample_id=sample_id,
            model_name="local-qwen",
            dataset_name="gsm8k",
            split="test",
            generation_seed=7,
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=512,
            batch_size=2,
            prompt="What is the percentage?",
            response=response,
            ground_truth="35",
            prompt_tokens=10,
            response_tokens=4,
            hit_max_length=False,
        )
    )


def test_rescore_completed_gsm8k_run_uses_literal_percentages_without_sampling(tmp_path):
    source = tmp_path / "completed-run"
    source.mkdir()
    write_rollouts_jsonl(
        source / "rollouts.jsonl",
        [
            make_record(sample_id=0, response="Answer: 35%"),
            make_record(sample_id=1, response="Answer: 4"),
        ],
    )
    (source / "config_snapshot.json").write_text(
        json.dumps(
            {
                "resolved": {
                    "dataset": {"name": "gsm8k"},
                    "sampling": {"num_samples": 2},
                    "analysis": {"k_values": [1, 2], "strict": False},
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

    artifacts = rescore_completed_run(source)

    comparison = json.loads(artifacts.comparison_json.read_text(encoding="utf-8"))
    migration = json.loads(artifacts.migration_json.read_text(encoding="utf-8"))
    changed = [
        json.loads(line)
        for line in artifacts.changed_samples_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    rescored_rows = [
        json.loads(line)
        for line in artifacts.rescored_rollouts_jsonl.read_text(encoding="utf-8").splitlines()
    ]

    assert comparison["percentage_policy"] == "literal"
    assert comparison["before"]["extraction_failure_rate"] == 0.5
    assert comparison["after"]["extraction_failure_rate"] == 0.0
    assert comparison["before"]["all_wrong_rate"] == 1.0
    assert comparison["after"]["mixed_rate"] == 1.0
    assert comparison["after"]["pass_at_k"] == {"1": 0.5, "2": 1.0}
    assert comparison["changed_sample_count"] == 1
    assert migration["sample"]["decision_transitions"] == [
        {
            "before": {
                "extraction_status": "explicit_final",
                "format_ok": True,
                "raw_correctness": False,
            },
            "after": {
                "extraction_status": "explicit_final",
                "format_ok": True,
                "raw_correctness": False,
            },
            "count": 1,
        },
        {
            "before": {
                "extraction_status": "parse_error",
                "format_ok": False,
                "raw_correctness": False,
            },
            "after": {
                "extraction_status": "explicit_final",
                "format_ok": False,
                "raw_correctness": True,
            },
            "count": 1,
        },
    ]
    assert migration["group"]["outcome_transitions"] == [
        {"before": "all_wrong", "after": "mixed", "count": 1}
    ]
    assert changed[0]["sample_id"] == 0
    assert changed[0]["after"]["extraction"]["normalized_answer"] == "35"
    assert rescored_rows[0]["verification"]["is_correct"] is True
    assert (artifacts.output_dir / "analysis" / "summary.json").is_file()
