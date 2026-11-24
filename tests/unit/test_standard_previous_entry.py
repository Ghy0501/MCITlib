# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Exercise previous-task loading through the standard training entry."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("deepspeed")

from mcitlib.backbones.llava import standard_train, zero3_loading  # noqa: E402


class PromptModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(5, 4)
        self.config = SimpleNamespace()

    def set_num_task(self, num):
        self.num_tasks = num

    def set_comtinual_prompts_tokenizer(self, tokenizer, prefix_len):
        return tokenizer

    def set_cur_task(self, task):
        self.cur_task = task

    def set_clip_tokenizer(self, tokenizer):
        pass

    def set_proto_input_ids(self):
        pass

    def reset_vocab_size(self):
        pass

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_origin_embed_tokens(self):
        self.origin_embed_tokens_weights = self.model.embed_tokens.weight.detach()


class StopAfterPreviousTask(Exception):
    pass


def test_prompt_task2_loads_cleaned_state_through_zero3_helper(tmp_path, monkeypatch):
    model = PromptModel()
    expected = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    previous = tmp_path / "task1"
    previous.mkdir()
    torch.save({"base_model.model.model.embed_tokens.weight": expected}, previous / "non_lora_trainables.bin")
    model_args = standard_train.ModelArguments(
        model_name_or_path="tiny-model", version="v0.5", cur_task=2,
        previous_task_model_path=str(previous), text_tower="tiny-text",
    )
    training_args = SimpleNamespace(
        local_rank=0, fp16=False, bf16=False, bits=16, gradient_checkpointing=False,
        lora_enable=False, pt_enable=True, cache_dir=str(tmp_path), model_max_length=32,
    )
    parser = SimpleNamespace(parse_args_into_dataclasses=lambda: (model_args, SimpleNamespace(), training_args))
    monkeypatch.setattr(standard_train.transformers, "HfArgumentParser", lambda *_: parser)
    monkeypatch.setattr(standard_train.transformers.LlamaForCausalLM, "from_pretrained", lambda *_, **__: model)
    monkeypatch.setattr(
        standard_train.transformers.AutoTokenizer, "from_pretrained",
        lambda *_, **__: SimpleNamespace(unk_token="<unk>"),
    )
    monkeypatch.setenv("MCITLIB_METHOD", "modalprompt")
    monkeypatch.delenv("MCITLIB_MODEL_OVERRIDE", raising=False)
    calls = []
    original_load = zero3_loading.load_zero3_state_dict

    def capture_load(received_model, state, *, strict):
        calls.append((received_model, tuple(state), strict))
        return original_load(received_model, state, strict=strict)

    monkeypatch.setattr(zero3_loading, "load_zero3_state_dict", capture_load)

    def stop_after_load(**kwargs):
        raise StopAfterPreviousTask

    monkeypatch.setattr(standard_train, "make_supervised_data_module", stop_after_load)
    with pytest.raises(StopAfterPreviousTask):
        standard_train.train()
    assert calls == [(model, ("model.embed_tokens.weight",), False)]
    assert torch.equal(model.model.embed_tokens.weight, expected)
    assert torch.equal(model.origin_embed_tokens_weights, expected)
