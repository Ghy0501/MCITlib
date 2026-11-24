# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU checks for the ModalPrompt task and checkpoint lifecycle."""
from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaModel

import mcitlib
from mcitlib.backbones.llava import standard_train, zero3_loading
from mcitlib.methods._modalprompt import trainer as candidate_trainer
from mcitlib.methods._modalprompt.llava_overrides.language_model.modal_prompt import (
    MCITModalPrompt,
)

pytestmark = pytest.mark.env("llava-modalprompt-ds126")


def test_modules_belong_to_the_same_project():
    root = Path(mcitlib.__file__).resolve().parent
    assert Path(standard_train.__file__).resolve().is_relative_to(root)
    assert Path(candidate_trainer.__file__).resolve().is_relative_to(root)


def test_prompt_tokens_transform_freezing_and_save_flags():
    class Tokenizer:
        def __init__(self):
            self.tokens = {}

        def __len__(self):
            return 32000 + len(self.tokens)

        def add_special_tokens(self, value):
            before = len(self.tokens)
            for token in value['additional_special_tokens']:
                if token not in self.tokens:
                    self.tokens[token] = 32000 + len(self.tokens)
            return len(self.tokens) - before

        def convert_tokens_to_ids(self, token):
            return self.tokens[token]

    tokenizer = Tokenizer()
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    core = LlamaModel(config)
    for parameter in core.parameters():
        parameter.requires_grad = False
    modal = SimpleNamespace(model=core, config=config, num_tasks=6, vocab_size=len(tokenizer),
                            hidden_size=8, device=torch.device('cpu'))
    modal.add_prompt_tokens = MethodType(MCITModalPrompt.add_prompt_tokens, modal)
    np.random.seed(7)
    MCITModalPrompt.set_comtinual_prompts_tokenizer(modal, tokenizer, prefix_len=10)
    assert len(tokenizer) == 32060
    assert tokenizer.convert_tokens_to_ids('[PRE1_1]') == 32000
    assert tokenizer.convert_tokens_to_ids('[PRE6_10]') == 32059
    MCITModalPrompt.set_cur_task(modal, 2)
    assert [any(p.requires_grad for p in layer.parameters()) for layer in modal.prompt_transform] == [False, True, False, False, False, False]
    MCITModalPrompt.reset_vocab_size(modal)
    assert config.vocab_size == 32000
    assert torch.nonzero(modal.embed_tokens_mask_1d).flatten().tolist() == list(range(32010, 32020))
    MCITModalPrompt.set_previou_transform_for_save(modal, 2)
    assert [all(p.requires_grad for p in layer.parameters()) for layer in modal.prompt_transform] == [True, True, False, False, False, False]


class PromptModel(torch.nn.Module):
    def __init__(self, trace):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(5, 4)
        self.config = SimpleNamespace()
        self.trace = trace

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_num_task(self, num):
        self.trace.append('tasks')

    def set_comtinual_prompts_tokenizer(self, tokenizer, prefix_len):
        self.trace.append(('prompts', tokenizer.pad_token))
        return tokenizer

    def set_cur_task(self, task):
        self.trace.append(('current', task))

    def set_clip_tokenizer(self, tokenizer):
        pass

    def set_proto_input_ids(self):
        pass

    def reset_vocab_size(self):
        pass

    def set_origin_embed_tokens(self):
        self.trace.append('origin')
        self.origin_embed_tokens_weights = self.model.embed_tokens.weight.detach()


class StopAfterTrainerSelection(Exception):
    pass


def test_previous_checkpoint_load_then_origin_then_modal_trainer(tmp_path, monkeypatch):
    trace = []
    model = PromptModel(trace)
    expected = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    previous = tmp_path / 'task1'
    previous.mkdir()
    torch.save({'base_model.model.model.embed_tokens.weight': expected}, previous / 'non_lora_trainables.bin')
    args = standard_train.ModelArguments(model_name_or_path='tiny-model', version='v0.5', cur_task=2,
                                         previous_task_model_path=str(previous), text_tower='tiny-text')
    training = SimpleNamespace(local_rank=0, fp16=False, bf16=False, bits=16, gradient_checkpointing=False,
                               lora_enable=False, pt_enable=True, cache_dir=str(tmp_path), model_max_length=32)
    parser = SimpleNamespace(parse_args_into_dataclasses=lambda: (args, SimpleNamespace(), training))
    monkeypatch.setattr(standard_train.transformers, 'HfArgumentParser', lambda *_: parser)
    monkeypatch.setattr(standard_train.transformers.LlamaForCausalLM, 'from_pretrained', lambda *_, **__: model)
    monkeypatch.setattr(standard_train.transformers.AutoTokenizer, 'from_pretrained',
                        lambda *_, **__: SimpleNamespace(unk_token='<unk>', pad_token=None))
    monkeypatch.setenv('MCITLIB_METHOD', 'modalprompt')
    monkeypatch.delenv('MCITLIB_MODEL_OVERRIDE', raising=False)
    original_load = zero3_loading.load_zero3_state_dict

    def capture_load(received_model, state, *, strict):
        trace.append('previous_load')
        return original_load(received_model, state, strict=strict)

    monkeypatch.setattr(zero3_loading, 'load_zero3_state_dict', capture_load)
    monkeypatch.setattr(standard_train, 'make_supervised_data_module', lambda **_: {})

    class ProbeTrainer:
        def __init__(self, *, model, origin, **kwargs):
            trace.append('trainer')
            assert torch.equal(origin, expected)
            assert origin.data_ptr() == model.model.embed_tokens.weight.data_ptr()

        def train(self):
            raise StopAfterTrainerSelection

    monkeypatch.setattr(candidate_trainer, 'ModalPromptTrainer', ProbeTrainer)
    with pytest.raises(StopAfterTrainerSelection):
        standard_train.train()
    assert trace == ['tasks', ('prompts', None), ('current', 2), 'previous_load', 'origin', 'trainer']
