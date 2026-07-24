from pathlib import Path

import pytest

from rewardscope import load_run_config


VALID_CONFIG = """
model:
  name: Qwen/Qwen2.5-1.5B-Instruct
  tokenizer_name: Qwen/Qwen2.5-1.5B-Instruct
dataset:
  name: gsm8k
  split: test
  max_prompts: 16
sampling:
  num_samples: 8
  seed: 42
  temperature: 0.7
  top_p: 0.95
  max_new_tokens: 256
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
    assert config.dataset.max_prompts == 16
    assert config.sampling.num_samples == 8
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
    content = VALID_CONFIG.replace("  max_prompts: 16", "  max_prompts: 16\n  typo: true")

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
