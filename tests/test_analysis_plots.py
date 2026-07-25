import sys

import pytest

from rewardscope import (
    RewardConfig,
    RolloutInput,
    build_numeric_rollout,
    compute_prompt_group_metrics,
    write_analysis_plots,
)
from rewardscope.reports import plots


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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


def assert_png(path):
    assert path is not None
    assert path.is_file()
    assert path.stat().st_size > len(PNG_SIGNATURE)
    assert path.read_bytes()[: len(PNG_SIGNATURE)] == PNG_SIGNATURE


def test_plot_writer_creates_four_non_empty_pngs_for_group_metrics(tmp_path):
    pytest.importorskip("matplotlib")
    rows = [
        make_row(prompt_id="wrong", response="Answer: 41"),
        make_row(prompt_id="wrong", sample_id=1, response="Answer: 40"),
        make_row(prompt_id="mixed"),
        make_row(prompt_id="mixed", sample_id=1, response="Answer: 41"),
        make_row(prompt_id="correct"),
        make_row(prompt_id="correct", sample_id=1),
    ]
    result = compute_prompt_group_metrics(rows)

    artifacts = write_analysis_plots(tmp_path, result)

    assert_png(artifacts.outcome_distribution_png)
    assert_png(artifacts.prompt_pass_rate_distribution_png)
    assert_png(artifacts.reward_variance_png)
    assert_png(artifacts.token_efficiency_png)


def test_empty_metrics_do_not_require_matplotlib_or_create_pngs(tmp_path):
    result = compute_prompt_group_metrics([])

    artifacts = write_analysis_plots(tmp_path, result)

    assert artifacts.outcome_distribution_png is None
    assert artifacts.prompt_pass_rate_distribution_png is None
    assert artifacts.reward_variance_png is None
    assert artifacts.token_efficiency_png is None


def test_all_wrong_zero_variance_and_no_mixed_group_still_plot(tmp_path):
    pytest.importorskip("matplotlib")
    rows = [
        make_row(response="Answer: 41"),
        make_row(sample_id=1, response="Answer: 41"),
    ]
    result = compute_prompt_group_metrics(rows)

    artifacts = write_analysis_plots(tmp_path, result)

    assert_png(artifacts.reward_variance_png)
    assert_png(artifacts.token_efficiency_png)


def test_variable_group_sizes_and_repeated_variance_coordinates_still_plot(tmp_path):
    pytest.importorskip("matplotlib")
    rows = [
        make_row(prompt_id="one", response="Answer: 41"),
        make_row(prompt_id="two", response="Answer: 41"),
        make_row(prompt_id="two", sample_id=1, response="Answer: 41"),
    ]
    result = compute_prompt_group_metrics(rows)

    artifacts = write_analysis_plots(tmp_path, result)

    assert_png(artifacts.prompt_pass_rate_distribution_png)
    assert_png(artifacts.reward_variance_png)


def test_missing_matplotlib_reports_the_optional_dependency_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    with pytest.raises(RuntimeError, match=r'pip install -e "\.\[analysis\]"'):
        plots._load_pyplot()
