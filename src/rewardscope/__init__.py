"""RewardScope package."""

from rewardscope.schemas import (
    ExtractionCandidate,
    ExtractionResult,
    ExtractionStatus,
    RewardBreakdown,
    RolloutRecord,
    VerificationResult,
)
from rewardscope.extraction import (
    NumericExtractionConfig,
    extract_numeric_answer,
    parse_numeric_value,
)

__all__ = [
    "ExtractionResult",
    "ExtractionCandidate",
    "ExtractionStatus",
    "RewardBreakdown",
    "RolloutRecord",
    "VerificationResult",
    "extract_numeric_answer",
    "parse_numeric_value",
    "NumericExtractionConfig",
]

from rewardscope.verification import (
    verify_extracted_numeric_answer,
    verify_numeric_answer,
)

__all__ += ["verify_extracted_numeric_answer", "verify_numeric_answer"]

from rewardscope.rewards import RewardConfig, compute_reward

__all__ += ["RewardConfig", "compute_reward"]

from rewardscope.rollouts import RolloutInput, build_numeric_rollout

__all__ += ["RolloutInput", "build_numeric_rollout"]

from rewardscope.io import read_rollouts_jsonl, write_rollouts_jsonl

__all__ += ["read_rollouts_jsonl", "write_rollouts_jsonl"]

from rewardscope.metrics import (
    MetricsIssue,
    PromptGroupMetrics,
    PromptGroupMetricsResult,
    PromptGroupSummary,
    compute_prompt_group_metrics,
    summarize_prompt_group_metrics,
)

__all__ += [
    "MetricsIssue",
    "PromptGroupMetrics",
    "PromptGroupMetricsResult",
    "PromptGroupSummary",
    "compute_prompt_group_metrics",
    "summarize_prompt_group_metrics",
]

from rewardscope.reports import (
    AnalysisArtifacts,
    AnalysisPlotArtifacts,
    analyze_rollouts_jsonl,
    write_analysis_report,
    write_analysis_plots,
    OfflineRescoreArtifacts,
    rescore_completed_run,
)

__all__ += [
    "AnalysisArtifacts",
    "AnalysisPlotArtifacts",
    "analyze_rollouts_jsonl",
    "write_analysis_report",
    "write_analysis_plots",
    "OfflineRescoreArtifacts",
    "rescore_completed_run",
]

from rewardscope.config import (
    AnalysisConfig,
    DatasetConfig,
    ModelConfig,
    OutputConfig,
    RunConfig,
    SamplingConfig,
    load_run_config,
)

__all__ += [
    "AnalysisConfig",
    "DatasetConfig",
    "ModelConfig",
    "OutputConfig",
    "RunConfig",
    "SamplingConfig",
    "load_run_config",
]

from rewardscope.datasets import (
    DEFAULT_GSM8K_PROMPT_TEMPLATE,
    GSM8K_COT_4SHOT_PROMPT_TEMPLATE,
    STRICT_GSM8K_PROMPT_TEMPLATE,
    DatasetExample,
    load_dataset_examples,
    load_dataset_result,
    load_gsm8k_examples,
)

__all__ += [
    "DEFAULT_GSM8K_PROMPT_TEMPLATE",
    "GSM8K_COT_4SHOT_PROMPT_TEMPLATE",
    "STRICT_GSM8K_PROMPT_TEMPLATE",
    "DatasetExample",
    "load_dataset_examples",
    "load_dataset_result",
    "load_gsm8k_examples",
]

from rewardscope.runners import ExperimentArtifacts, run_experiment, run_experiment_from_yaml

__all__ += ["ExperimentArtifacts", "run_experiment", "run_experiment_from_yaml"]

from rewardscope.sampling import GeneratedResponse, TransformersSampler

__all__ += ["GeneratedResponse", "TransformersSampler"]
