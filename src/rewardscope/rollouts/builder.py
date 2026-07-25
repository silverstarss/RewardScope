"""Build complete rollout records from generated model responses."""

from __future__ import annotations

from dataclasses import dataclass

from rewardscope.extraction import NumericExtractionConfig
from rewardscope.rewards import RewardConfig, compute_reward
from rewardscope.schemas import RolloutRecord
from rewardscope.verification import verify_numeric_answer


@dataclass(frozen=True)
class RolloutInput:
    """Generation facts available before verification and reward calculation."""

    run_id: str
    prompt_id: str
    sample_id: int
    model_name: str
    dataset_name: str
    split: str
    generation_seed: int
    temperature: float
    top_p: float
    max_new_tokens: int
    batch_size: int
    prompt: str
    response: str
    ground_truth: str
    prompt_tokens: int
    response_tokens: int
    hit_max_length: bool


def build_numeric_rollout(
    rollout_input: RolloutInput,
    reward_config: RewardConfig = RewardConfig(),
    extraction_config: NumericExtractionConfig = NumericExtractionConfig(),
) -> RolloutRecord:
    """Verify, reward, and package one numeric rollout into a complete record."""
    if not isinstance(rollout_input, RolloutInput):
        raise TypeError("rollout_input must be a RolloutInput.")

    verification = verify_numeric_answer(
        rollout_input.response,
        rollout_input.ground_truth,
        extraction_config=extraction_config,
    )
    reward = compute_reward(
        verification,
        response_tokens=rollout_input.response_tokens,
        config=reward_config,
    )
    return RolloutRecord(
        run_id=rollout_input.run_id,
        prompt_id=rollout_input.prompt_id,
        sample_id=rollout_input.sample_id,
        model_name=rollout_input.model_name,
        dataset_name=rollout_input.dataset_name,
        split=rollout_input.split,
        generation_seed=rollout_input.generation_seed,
        temperature=rollout_input.temperature,
        top_p=rollout_input.top_p,
        max_new_tokens=rollout_input.max_new_tokens,
        batch_size=rollout_input.batch_size,
        prompt=rollout_input.prompt,
        response=rollout_input.response,
        ground_truth=rollout_input.ground_truth,
        verification=verification,
        reward=reward,
        prompt_tokens=rollout_input.prompt_tokens,
        response_tokens=rollout_input.response_tokens,
        hit_max_length=rollout_input.hit_max_length,
    )
