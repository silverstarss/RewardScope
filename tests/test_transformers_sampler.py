import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from rewardscope import ModelConfig, SamplingConfig, TransformersSampler


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "</s>"
    chat_template = None

    def __init__(self):
        self.padding_side = "right"
        self.plain_calls = []
        self.chat_calls = []

    def __call__(self, prompts, **kwargs):
        self.plain_calls.append((prompts, kwargs))
        rows = [[0, 11], [12, 13]][: len(prompts)]
        masks = [[0, 1], [1, 1]][: len(prompts)]
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.tensor(masks),
        }

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        return {
            "input_ids": torch.tensor([[0, 11]]),
            "attention_mask": torch.tensor([[0, 1]]),
        }

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return " ".join(str(token_id) for token_id in token_ids)


class FakeModel:
    class Config:
        is_encoder_decoder = False
        max_position_embeddings = 32

    class GenerationConfig:
        eos_token_id = [2, 3]

    config = Config()
    generation_config = GenerationConfig()
    device = torch.device("cpu")

    def __init__(self):
        self.eval_called = False
        self.generate_calls = []

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, *, input_ids, attention_mask, **kwargs):
        self.generate_calls.append((input_ids, attention_mask, kwargs))
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        rows = []
        for prompt_ids in input_ids.tolist():
            for sample_index in range(num_return_sequences):
                rows.append(prompt_ids + [20 + sample_index, 2, 0])
        return torch.tensor(rows)


def make_sampling_config(**overrides):
    fields = {
        "num_samples": 2,
        "seed": 123,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 4,
        "batch_size": 2,
    }
    fields.update(overrides)
    return SamplingConfig(**fields)


def test_sampler_uses_left_padding_padded_width_slicing_and_prompt_major_order():
    model = FakeModel()
    tokenizer = FakeTokenizer()
    sampler = TransformersSampler(model, tokenizer, ModelConfig(name="fake", prompt_format="plain"))

    responses = sampler.generate(["first", "second"], make_sampling_config())

    assert model.eval_called is True
    assert tokenizer.padding_side == "left"
    assert [(item.prompt_index, item.sample_index) for item in responses] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [item.prompt_tokens for item in responses] == [1, 1, 2, 2]
    assert [item.response for item in responses] == ["20", "21", "20", "21"]
    assert all(item.response_tokens == 1 for item in responses)
    assert all(item.finish_reason == "eos" for item in responses)
    assert model.generate_calls[0][2]["num_return_sequences"] == 2
    assert tokenizer.plain_calls[0][1]["truncation"] is False


def test_chat_mode_uses_the_tokenized_generation_template_call():
    model = FakeModel()
    tokenizer = FakeTokenizer()
    sampler = TransformersSampler(model, tokenizer, ModelConfig(name="fake", prompt_format="chat"))

    sampler.generate(["hello"], make_sampling_config(num_samples=1, batch_size=1))

    messages, kwargs = tokenizer.chat_calls[0]
    assert messages == [[{"role": "user", "content": "hello"}]]
    assert kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "padding": True,
        "return_tensors": "pt",
    }


def test_context_window_overflow_is_an_explicit_error_before_generation():
    model = FakeModel()
    model.config.max_position_embeddings = 2
    sampler = TransformersSampler(model, FakeTokenizer(), ModelConfig(name="fake"))

    with pytest.raises(ValueError, match="exceeds the model context window"):
        sampler.generate(["first", "second"], make_sampling_config())

    assert model.generate_calls == []
