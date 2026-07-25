import pytest

from rewardscope import GeneratedResponse, SamplingConfig
from rewardscope.sampling.transformers import (
    build_generation_kwargs,
    clean_generated_token_ids,
    reshape_generated_responses,
)


def make_sampling_config(**overrides):
    fields = {
        "num_samples": 2,
        "generation_seed": 123,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 16,
        "batch_size": 4,
    }
    fields.update(overrides)
    return SamplingConfig(**fields)


def test_sampling_config_requires_one_greedy_sample():
    with pytest.raises(ValueError, match="num_samples must be 1 when temperature is 0"):
        make_sampling_config(temperature=0, num_samples=2)


def test_generation_kwargs_omit_sampling_only_arguments_for_greedy_decoding():
    kwargs = build_generation_kwargs(
        make_sampling_config(temperature=0, num_samples=1)
    )

    assert kwargs == {
        "max_new_tokens": 16,
        "do_sample": False,
        "num_beams": 1,
    }


def test_generation_kwargs_request_one_sequence_per_sample_for_stochastic_decoding():
    kwargs = build_generation_kwargs(make_sampling_config())

    assert kwargs == {
        "max_new_tokens": 16,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "num_return_sequences": 2,
    }


def test_eos_and_padding_are_classified_from_raw_token_ids_before_decoding():
    clean_ids, finish_reason = clean_generated_token_ids(
        [41, 42, 2, 0, 0], eos_token_id=[2, 3], pad_token_id=0
    )

    assert clean_ids == [41, 42]
    assert finish_reason == "eos"


def test_missing_eos_is_a_length_stop_and_padding_is_not_counted():
    clean_ids, finish_reason = clean_generated_token_ids(
        [41, 0, 42], eos_token_id=2, pad_token_id=0
    )

    assert clean_ids == [41, 42]
    assert finish_reason == "length"


def test_response_reshape_is_prompt_major_and_sample_minor():
    responses = reshape_generated_responses(
        prompt_indices=[5, 7],
        prompt_token_counts=[3, 4],
        continuations=[
            ("a", [11], "eos"),
            ("b", [12, 13], "length"),
            ("c", [], "eos"),
            ("d", [14], "eos"),
        ],
        num_samples=2,
    )

    assert [(item.prompt_index, item.sample_index) for item in responses] == [
        (5, 0),
        (5, 1),
        (7, 0),
        (7, 1),
    ]
    assert responses[1].response_tokens == 2
    assert responses[1].hit_max_length is True


def test_generated_response_validates_and_derives_length_status():
    response = GeneratedResponse(0, 0, "", 1, 0, "eos")

    assert response.hit_max_length is False
    with pytest.raises(ValueError, match="finish_reason"):
        GeneratedResponse(0, 0, "", 1, 0, "stop")  # type: ignore[arg-type]
