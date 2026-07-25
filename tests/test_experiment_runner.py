import json

import pytest

from rewardscope import (
    AnalysisConfig,
    DatasetConfig,
    DatasetExample,
    ModelConfig,
    OutputConfig,
    RewardConfig,
    RunConfig,
    SamplingConfig,
)
from rewardscope.datasets.schema import DatasetLoadResult
from rewardscope.runners import experiment
from rewardscope.sampling import GeneratedResponse


class FakeModel:
    dtype = "float32"


class FakeTokenizer:
    padding_side = "left"
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "fake-template"


class FakeSampler:
    called = False
    bad_output = False

    def __init__(self):
        self._model = FakeModel()
        self._tokenizer = FakeTokenizer()

    @classmethod
    def from_pretrained(cls, model_config):
        cls.called = True
        return cls()

    def _resolve_prompt_format(self):
        return "chat"

    def _eos_token_id(self):
        return 2

    def render_prompt(self, prompt):
        return f"<user>{prompt}</user><assistant>"

    def generate(self, prompts, sampling):
        responses = [
            GeneratedResponse(prompt_index, sample_index, "Answer: 42", 5, 3, "eos")
            for prompt_index in range(len(prompts))
            for sample_index in range(sampling.num_samples)
        ]
        return responses[:-1] if self.bad_output else responses


def make_examples(*, duplicate_ids=False):
    first = DatasetExample("gsm8k", "test", 10, "p-0", "What is 40 + 2?", "Question: 40 + 2", "42", "#### 42")
    second = DatasetExample("gsm8k", "test", 11, "p-0" if duplicate_ids else "p-1", "What is 20 + 22?", "Question: 20 + 22", "42", "#### 42")
    return DatasetLoadResult((first, second), source_count=20, fingerprint="fake-fingerprint")


def make_config(tmp_path, **overrides):
    fields = {
        "model": ModelConfig(name="fake-model", prompt_format="chat"),
        "dataset": DatasetConfig(name="gsm8k", config="main", split="test", max_examples=2, dataset_seed=9),
        "sampling": SamplingConfig(num_samples=2, generation_seed=7, temperature=0.7, top_p=0.95, max_new_tokens=16, batch_size=2),
        "reward": RewardConfig(),
        "output": OutputConfig(run_id="fake-run", output_dir=tmp_path / "outputs" / "fake-run"),
        "analysis": AnalysisConfig(k_values=(1, 2)),
    }
    fields.update(overrides)
    return RunConfig(**fields)


@pytest.fixture(autouse=True)
def fake_dependencies(monkeypatch):
    FakeSampler.called = False
    FakeSampler.bad_output = False
    monkeypatch.setattr(experiment, "TransformersSampler", FakeSampler)
    monkeypatch.setattr(experiment, "load_dataset_result", lambda config: make_examples())


def test_successful_run_commits_staging_and_writes_complete_manifest(tmp_path):
    artifacts = experiment.run_experiment(make_config(tmp_path))

    assert artifacts.output_dir.is_dir()
    assert artifacts.summary.group_count == 2
    assert len(artifacts.inputs_jsonl.read_text(encoding="utf-8").splitlines()) == 2
    rendered_prompt = json.loads(artifacts.rendered_prompt_json.read_text(encoding="utf-8"))
    assert rendered_prompt["model_input_prompt"] == "<user>Question: 40 + 2</user><assistant>"
    assert len(artifacts.rollouts_jsonl.read_text(encoding="utf-8").splitlines()) == 4
    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert {item["path"] for item in manifest["artifacts"]} >= {
        "inputs.jsonl", "rendered_prompt.json", "rollouts.jsonl", "config_snapshot.json", "provenance.json",
        "analysis/prompt_group_metrics.csv", "analysis/summary.json", "analysis/issues.jsonl",
    }
    assert not list((artifacts.output_dir.parent / ".staging").glob("fake-run-*"))


def test_empty_data_fails_before_model_loading_and_creates_no_final_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "load_dataset_result", lambda config: DatasetLoadResult((), 0, None))
    config = make_config(tmp_path)

    with pytest.raises(ValueError, match="no examples"):
        experiment.run_experiment(config)

    assert FakeSampler.called is False
    assert not config.output.output_dir.exists()


def test_duplicate_prompt_ids_fail_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "load_dataset_result", lambda config: make_examples(duplicate_ids=True))

    with pytest.raises(ValueError, match="duplicate prompt_id"):
        experiment.run_experiment(make_config(tmp_path))

    assert FakeSampler.called is False


def test_nonempty_final_directory_is_rejected(tmp_path):
    config = make_config(tmp_path)
    config.output.output_dir.mkdir(parents=True)
    (config.output.output_dir / "user-file.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="absent or empty"):
        experiment.run_experiment(config)


def test_failed_run_is_retained_with_failure_json_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "analyze_rollouts_jsonl", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("analysis exploded")))
    config = make_config(tmp_path, output=OutputConfig("fake-run", tmp_path / "outputs" / "fake-run", keep_failed_run=True))

    with pytest.raises(RuntimeError, match="analysis exploded"):
        experiment.run_experiment(config)

    failures = list((config.output.output_dir.parent / ".staging").glob("fake-run-*/failure.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text(encoding="utf-8"))["phase"] == "analysis"
    assert not config.output.output_dir.exists()


def test_failed_run_is_cleaned_when_not_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "build_numeric_rollout", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("verifier exploded")))
    config = make_config(tmp_path)

    with pytest.raises(RuntimeError, match="verifier exploded"):
        experiment.run_experiment(config)

    assert not list((config.output.output_dir.parent / ".staging").glob("fake-run-*"))
    assert not config.output.output_dir.exists()


def test_plot_dependency_failure_happens_before_sampler(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "_preflight_plotting", lambda: (_ for _ in ()).throw(RuntimeError("install matplotlib")))
    config = make_config(tmp_path, analysis=AnalysisConfig(k_values=(1, 2), write_plots=True))

    with pytest.raises(RuntimeError, match="install matplotlib"):
        experiment.run_experiment(config)

    assert FakeSampler.called is False


def test_plots_disabled_does_not_call_plot_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "_preflight_plotting", lambda: (_ for _ in ()).throw(AssertionError("plot preflight called")))

    experiment.run_experiment(make_config(tmp_path))


def test_yaml_run_snapshot_preserves_requested_fields_and_resolves_paths(tmp_path):
    output_dir = tmp_path / "outputs" / "yaml-run"
    path = tmp_path / "run.yaml"
    path.write_text(
        f"""model:
  name: fake-model
dataset:
  name: gsm8k
  config: main
  split: test
sampling:
  num_samples: 2
  generation_seed: 7
  temperature: 0.7
  top_p: 0.95
  max_new_tokens: 16
  batch_size: 2
output:
  run_id: yaml-run
  output_dir: {output_dir}
analysis:
  k_values: [1, 2]
""",
        encoding="utf-8",
    )

    artifacts = experiment.run_experiment_from_yaml(path)
    snapshot = json.loads(artifacts.config_snapshot_json.read_text(encoding="utf-8"))

    assert snapshot["requested"]["analysis"] == {"k_values": [1, 2]}
    assert snapshot["resolved"]["analysis"]["write_plots"] is False
    assert snapshot["resolved"]["output"]["output_dir"] == str(output_dir.resolve())


def test_invalid_sampler_contract_prevents_final_commit(tmp_path):
    FakeSampler.bad_output = True
    config = make_config(tmp_path)

    with pytest.raises(ValueError, match="response count"):
        experiment.run_experiment(config)

    assert not config.output.output_dir.exists()


def test_snapshot_provenance_and_core_results_are_deterministic_for_fake_runs(tmp_path):
    first = experiment.run_experiment(make_config(tmp_path, output=OutputConfig("same", tmp_path / "outputs" / "one")))
    second = experiment.run_experiment(make_config(tmp_path, output=OutputConfig("same", tmp_path / "outputs" / "two")))

    snapshot = json.loads(first.config_snapshot_json.read_text(encoding="utf-8"))
    provenance = json.loads(first.provenance_json.read_text(encoding="utf-8"))
    assert snapshot["requested"]["analysis"]["write_plots"] is False
    assert snapshot["resolved"]["output"]["output_dir"] == str(first.output_dir)
    assert provenance["generation"]["generation_seed"] == 7
    assert provenance["dataset"]["selected_source_indices"] == [10, 11]
    assert first.inputs_jsonl.read_text(encoding="utf-8") == second.inputs_jsonl.read_text(encoding="utf-8")
    assert first.rollouts_jsonl.read_text(encoding="utf-8") == second.rollouts_jsonl.read_text(encoding="utf-8")
    assert first.report.summary_json.read_text(encoding="utf-8") == second.report.summary_json.read_text(encoding="utf-8")
