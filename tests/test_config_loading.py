from pathlib import Path

import pytest

from rewardscope import load_run_config


VALID_CONFIG = """
model:
  name: Qwen/Qwen2.5-1.5B-Instruct
  tokenizer_name: Qwen/Qwen2.5-1.5B-Instruct
  prompt_format: chat
  context_window: 32768
dataset:
  name: gsm8k
  config: main
  split: test
  max_examples: 16
  selection: first
  dataset_seed: 9
sampling:
  num_samples: 8
  generation_seed: 42
  temperature: 0.7
  top_p: 0.95
  max_new_tokens: 256
  batch_size: 4
reward:
  correct_answer_reward: 1.0
  format_compliance_reward: 0.1
output:
  run_id: smoke-run
  output_dir: outputs/smoke-run
analysis:
  strict: false
  k_values: [1, 4, 8]
"""


def write_config(tmp_path, content=VALID_CONFIG):
    path = tmp_path / "run.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_run_config_builds_validated_nested_dataclasses(tmp_path):
    config = load_run_config(write_config(tmp_path))

    assert config.model.name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.model.tokenizer_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.model.prompt_format == "chat"
    assert config.model.context_window == 32768
    assert config.dataset.max_examples == 16
    assert config.dataset.dataset_seed == 9
    assert config.dataset.prompt_template == "baseline"
    assert config.sampling.num_samples == 8
    assert config.sampling.batch_size == 4
    assert config.reward.format_compliance_reward == 0.1
    assert config.reward.incorrect_answer_reward == 0.0
    assert config.output.output_dir == Path("outputs/smoke-run")
    assert config.analysis.k_values == (1, 4, 8)


def test_reward_and_analysis_sections_use_defaults_when_omitted(tmp_path):
    content = VALID_CONFIG.replace(
        "reward:\n  correct_answer_reward: 1.0\n  format_compliance_reward: 0.1\n", ""
    ).replace("analysis:\n  strict: false\n  k_values: [1, 4, 8]\n", "")

    config = load_run_config(write_config(tmp_path, content))

    assert config.reward.correct_answer_reward == 1.0
    assert config.analysis.strict is False
    assert config.analysis.k_values == (1, 4, 8)


def test_loader_rejects_unknown_fields(tmp_path):
    content = VALID_CONFIG.replace("  max_examples: 16", "  max_examples: 16\n  typo: true")

    with pytest.raises(ValueError, match="dataset has unknown fields: typo"):
        load_run_config(write_config(tmp_path, content))


def test_loader_rejects_missing_required_fields(tmp_path):
    content = VALID_CONFIG.replace("  max_new_tokens: 256\n", "")

    with pytest.raises(ValueError, match="sampling is missing required fields: max_new_tokens"):
        load_run_config(write_config(tmp_path, content))


@pytest.mark.parametrize(
    ("old", "new", "error_message"),
    [
        ("  num_samples: 8", "  num_samples: 0", "num_samples must be a positive integer"),
        ("  top_p: 0.95", "  top_p: 0", "top_p must be a number in the interval"),
        ("  batch_size: 4", "  batch_size: 0", "batch_size must be a positive integer"),
        (
            "  prompt_format: chat",
            "  prompt_format: markdown",
            "prompt_format must be one of",
        ),
        (
            "  dataset_seed: 9",
            "  dataset_seed: 9\n  prompt_template: verbose",
            "prompt_template must be one of",
        ),
        (
            "  k_values: [1, 4, 8]",
            "  k_values: [1, 9]",
            "analysis.k_values cannot exceed sampling.num_samples",
        ),
    ],
)
def test_loader_rejects_invalid_values(tmp_path, old, new, error_message):
    with pytest.raises(ValueError, match=error_message):
        load_run_config(write_config(tmp_path, VALID_CONFIG.replace(old, new)))


def test_loader_requires_analysis_k_values_to_be_a_yaml_list(tmp_path):
    content = VALID_CONFIG.replace("  k_values: [1, 4, 8]", "  k_values: 8")

    with pytest.raises(ValueError, match="analysis.k_values must be a list"):
        load_run_config(write_config(tmp_path, content))


def test_loader_reports_invalid_yaml(tmp_path):
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_run_config(write_config(tmp_path, "model: [unterminated"))


def test_loader_requires_top_level_mapping(tmp_path):
    with pytest.raises(ValueError, match="configuration must be a mapping"):
        load_run_config(write_config(tmp_path, "- not\n- a mapping\n"))


def test_gsm8k_sanity_config_locks_the_intended_evaluation_contract():
    config = load_run_config(
        Path(__file__).parents[1] / "configs" / "gsm8k-sanity.yaml"
    )

    assert config.dataset.max_examples == 128
    assert config.dataset.prompt_template == "gsm8k_cot_4shot"
    assert config.sampling.temperature == 0.0
    assert config.sampling.num_samples == 1
    assert config.sampling.max_new_tokens == 512
    assert config.analysis.k_values == (1,)


def test_loader_parses_explicit_source_indices(tmp_path):
    content = VALID_CONFIG.replace(
        "  dataset_seed: 9", "  dataset_seed: 9\n  source_indices: [3, 1]"
    ).replace("  max_examples: 16\n", "")

    config = load_run_config(write_config(tmp_path, content))

    assert config.dataset.source_indices == (3, 1)


def test_explicit_source_indices_reject_conflicting_max_examples(tmp_path):
    content = VALID_CONFIG.replace(
        "  dataset_seed: 9", "  dataset_seed: 9\n  source_indices: [3, 1]"
    )

    with pytest.raises(ValueError, match="max_examples must be None"):
        load_run_config(write_config(tmp_path, content))


def test_format_calibration_config_locks_the_fixed_subset_and_decoding_contract():
    config = load_run_config(
        Path(__file__).parents[1] / "configs" / "gsm8k-format-calibration.yaml"
    )

    assert config.model.prompt_format == "chat"
    assert config.dataset.source_indices == (
        2, 4, 6, 7, 10, 12, 13, 15, 16, 21, 41, 60, 0, 1, 3, 5
    )
    assert config.dataset.prompt_template == "gsm8k_cot_4shot_terminal"
    assert config.sampling.temperature == 0.0
    assert config.sampling.num_samples == 1
    assert config.sampling.max_new_tokens == 512


def test_multiturn_format_calibration_changes_only_the_prompt_message_structure():
    config = load_run_config(
        Path(__file__).parents[1] / "configs" / "gsm8k-format-calibration-multiturn.yaml"
    )

    assert config.dataset.source_indices == (
        2, 4, 6, 7, 10, 12, 13, 15, 16, 21, 41, 60, 0, 1, 3, 5
    )
    assert config.dataset.prompt_template == "gsm8k_cot_4shot_multiturn_terminal"
    assert config.sampling.temperature == 0.0
    assert config.sampling.num_samples == 1
    assert config.sampling.max_new_tokens == 512
    assert config.sampling.batch_size == 4
