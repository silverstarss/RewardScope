"""RewardScope package."""

from rewardscope.schemas import (
    ExtractionResult,
    ExtractionStatus,
    RewardBreakdown,
    RolloutRecord,
    VerificationResult,
)
from rewardscope.extraction import extract_numeric_answer, parse_numeric_value

__all__ = [
    "ExtractionResult",
    "ExtractionStatus",
    "RewardBreakdown",
    "RolloutRecord",
    "VerificationResult",
    "extract_numeric_answer",
    "parse_numeric_value",
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
)

__all__ += [
    "AnalysisArtifacts",
    "AnalysisPlotArtifacts",
    "analyze_rollouts_jsonl",
    "write_analysis_report",
    "write_analysis_plots",
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
