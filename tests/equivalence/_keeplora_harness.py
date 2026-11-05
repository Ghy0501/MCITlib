# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Small-model KeepLoRA model, LoRA, and subspace comparisons."""
from __future__ import annotations

import argparse
import importlib.util  # noqa: E402
import json
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from mcitlib.backbones.llava.overrides import allow_t437_llava_registration

if importlib.util.find_spec("deepspeed") is None:  # a stub would shadow (and break) a real install
    deepspeed_stub = ModuleType("deepspeed")
    deepspeed_stub.__spec__ = ModuleSpec("deepspeed", loader=None)
    sys.modules.setdefault("deepspeed", deepspeed_stub)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(), Block()])
        self.config = SimpleNamespace(model_type="llama", to_dict=lambda: {"model_type": "llama"})

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return kwargs

    def forward(self, inputs=None, **kwargs):
        input_ids = inputs if inputs is not None else kwargs["input_ids"]
        hidden = self.model.layers[0].self_attn.q_proj(input_ids.float())
        loss = hidden.square().mean()
        if inputs is not None:
            return loss
        return SimpleNamespace(loss=loss)


def model_comparison(out: Path, implementation: str) -> None:
    with allow_t437_llava_registration():
        if implementation == "v3":
            from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
        else:
            from mcitlib.methods._keeplora.llava_overrides import LlavaLlamaForCausalLM
            from mcitlib.methods._keeplora.llava_overrides.language_model.llava_llama import LlavaConfig
    torch.manual_seed(2026)
    config = LlavaConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=128, vocab_size=320,
    )
    model = LlavaLlamaForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    model.train()
    output = model(input_ids=input_ids, labels=input_ids)
    output.loss.backward()
    gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None}
    model.eval()
    with torch.no_grad():
        eval_loss = model(input_ids=input_ids, labels=input_ids).loss
    torch.save({
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "train_loss": output.loss.detach().cpu(),
        "eval_loss": eval_loss.detach().cpu(),
        "gradients": gradients,
        "requires_grad": {name: p.requires_grad for name, p in model.named_parameters()},
    }, out)


def lora_comparison(out: Path, implementation: str) -> None:
    with allow_t437_llava_registration():
        if implementation == "v3":
            from llava.train.train import freeze_lora_A_matrices, initialize_lora_weights_custom
        else:
            from mcitlib.methods._keeplora.lora import (
                freeze_lora_a_matrices,
                initialize_lora_weights_custom,
            )
            freeze_lora_A_matrices = freeze_lora_a_matrices
    from peft import LoraConfig, get_peft_model
    from peft.utils import get_peft_model_state_dict

    def make_model():
        torch.manual_seed(2026)
        return get_peft_model(Toy(), LoraConfig(
            r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM", bias="none"
        ))

    gradients = {}
    for layer in range(2):
        name = f"model.layers.{layer}.self_attn.q_proj.weight"
        gradients[name] = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
    model = make_model()
    initialize_lora_weights_custom(model, gradients, lora_r=2)
    frozen_count = freeze_lora_A_matrices(model)
    errors = []
    for values in ({"wrong": torch.zeros(2, 4)}, {"model.layers.0.self_attn.q_proj.weight": torch.zeros(3, 4)}):
        try:
            initialize_lora_weights_custom(make_model(), values, lora_r=2)
        except ValueError as error:
            errors.append(str(error))
    torch.save({
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "adapter_keys": sorted(get_peft_model_state_dict(model)),
        "requires_grad": {name: p.requires_grad for name, p in model.named_parameters()},
        "frozen_count": frozen_count,
        "errors": errors,
    }, out)


def _toy_data_module(tokenizer, data_args):
    dataset = [{
        "input_ids": torch.arange(4, dtype=torch.float32) + index,
        "labels": torch.ones(4, dtype=torch.long),
        "attention_mask": torch.ones(4, dtype=torch.long),
    } for index in range(4)]

    def collate(items):
        return {key: torch.stack([item[key] for item in items]) for key in items[0]}

    return {"train_dataset": dataset, "data_collator": collate}


def subspace_comparison(out: Path, implementation: str) -> None:
    with allow_t437_llava_registration():
        if implementation == "v3":
            from llava.train.extract_gradients import extract_gradients, merge_gradient_chunks
            from llava.train.extract_weights import extract_weights
        else:
            from mcitlib.methods._keeplora.extract_gradients import extract_gradients, merge_gradient_chunks
            from mcitlib.methods._keeplora.extract_weights import extract_weights

    root = out.parent
    model_config = root / "model.json"
    data_config = root / "data.json"
    train_config = root / "train.json"
    model_config.write_text(json.dumps({"model_name": "llava"}))
    data_config.write_text(json.dumps({"train_path": "unused", "train_folder": "unused"}))
    space = root / "space"

    def load_model(*args, **kwargs):
        torch.manual_seed(1234)
        return None, Toy(), None, 0

    real_device = torch.device
    with patch("torch.device", side_effect=lambda value: real_device("cpu")), \
         patch("llava.train.extract_weights.load_pretrained_model" if implementation == "v3" else
               "mcitlib.methods._keeplora.extract_weights.load_pretrained_model", side_effect=load_model), \
         patch("llava.train.extract_weights.disable_torch_init" if implementation == "v3" else
               "mcitlib.methods._keeplora.extract_weights.disable_torch_init"):
        extract_weights(str(model_config), str(space), 0.6, "q_proj")
        weight_data = torch.load(space / "gradients_principle_subspace.pt", map_location="cpu")

        train_config.write_text(json.dumps({
            "model_path": "llava", "rank": 2, "energy_threshold": 0.6, "space_path": str(space),
        }))
        gradient_dir = root / "gradients"
        with patch("llava.train.extract_gradients.load_pretrained_model" if implementation == "v3" else
                   "mcitlib.methods._keeplora.extract_gradients.load_pretrained_model", side_effect=load_model), \
             patch("llava.train.extract_gradients.disable_torch_init" if implementation == "v3" else
                   "mcitlib.methods._keeplora.extract_gradients.disable_torch_init"), \
             patch("llava.train.extract_gradients.make_supervised_data_module" if implementation == "v3" else
                   "mcitlib.methods._keeplora.extract_gradients.make_supervised_data_module", side_effect=_toy_data_module):
            for chunk_idx in range(2):
                extract_gradients(SimpleNamespace(
                    data_config=str(data_config), model_config=str(model_config), train_config=str(train_config),
                    output_dir=str(gradient_dir), data_ratio=0.75, batch_size=1, num_workers=0,
                    num_chunks=2, chunk_idx=chunk_idx, target_modules="q_proj",
                ))
            chunks = [torch.load(gradient_dir / f"gradients_chunk_{index}_of_2.pt", map_location="cpu") for index in range(2)]
            torch.manual_seed(2026)
            fixed_path = merge_gradient_chunks(gradient_dir, 2, str(train_config), "fixed_rank")
            fixed_data = torch.load(fixed_path, map_location="cpu")
            for chunk_idx in range(2):
                extract_gradients(SimpleNamespace(
                    data_config=str(data_config), model_config=str(model_config), train_config=str(train_config),
                    output_dir=str(gradient_dir), data_ratio=0.75, batch_size=1, num_workers=0,
                    num_chunks=2, chunk_idx=chunk_idx, target_modules="q_proj",
                ))
            energy_path = merge_gradient_chunks(gradient_dir, 2, str(train_config), "energy")
            energy_data = torch.load(energy_path, map_location="cpu")
    torch.save({"weights": weight_data, "chunks": chunks, "fixed": fixed_data, "energy": energy_data}, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["v3", "v4"], required=True)
    parser.add_argument("--mode", choices=["model", "lora", "subspace"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"model": model_comparison, "lora": lora_comparison, "subspace": subspace_comparison}[args.mode](
        args.out, args.implementation
    )


if __name__ == "__main__":
    main()
