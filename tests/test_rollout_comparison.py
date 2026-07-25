import json

from rewardscope import (
    RolloutInput,
    build_numeric_rollout,
    compare_rollouts_jsonl,
    write_rollouts_jsonl,
)


def make_record(*, prompt_id: str, response: str, response_tokens: int):
    return build_numeric_rollout(
        RolloutInput(
            run_id="run",
            prompt_id=prompt_id,
            sample_id=0,
            model_name="model",
            dataset_name="gsm8k",
            split="test",
            generation_seed=1,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=512,
            batch_size=1,
            prompt="Question",
            response=response,
            ground_truth="42",
            prompt_tokens=3,
            response_tokens=response_tokens,
            hit_max_length=False,
        )
    )


def test_offline_rollout_comparison_aligns_prompt_rows_and_preserves_responses(tmp_path):
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    write_rollouts_jsonl(
        baseline_path,
        [
            make_record(prompt_id="p-0", response="Answer: 42", response_tokens=8),
            make_record(prompt_id="p-1", response="Answer: 41", response_tokens=12),
        ],
    )
    write_rollouts_jsonl(
        candidate_path,
        [
            make_record(prompt_id="p-0", response="#### 41", response_tokens=2),
            make_record(prompt_id="p-1", response="#### 42", response_tokens=3),
        ],
    )

    artifacts = compare_rollouts_jsonl(
        baseline_path, candidate_path, tmp_path / "comparison"
    )

    summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in artifacts.sample_comparison_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert summary["sample_count"] == 2
    assert summary["baseline_accuracy"] == 0.5
    assert summary["candidate_accuracy"] == 0.5
    assert summary["correctness_transition_counts"] == {
        "correct_to_incorrect": 1,
        "incorrect_to_correct": 1,
    }
    assert summary["response_tokens_delta_total"] == -15
    assert rows[0]["correctness_transition"] == "correct_to_incorrect"
    assert rows[0]["baseline_response"] == "Answer: 42"
    assert rows[0]["candidate_response"] == "#### 41"
