# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Isolated small-model tuner comparison for v3 and v4."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block() for _ in range(32)])
        self.config = SimpleNamespace(model_type="llama", to_dict=lambda: {"model_type": "llama"})
        self.expert_weight = [0.0] * 8

    def forward(self, x):
        return self.model.layers[31].q_proj(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["v3", "v4"], required=True)
    parser.add_argument("--method", choices=["olora", "hide"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(1234)
    model = Toy()
    if args.implementation == "v3":
        if args.method == "olora":
            from CoIN.peft import CoINMOELoraConfig, get_peft_model
            config = CoINMOELoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], cur_task=1, expert_num=4)
        else:
            from HiDe.peft import HiDeMOELoraConfig, get_peft_model
            config = HiDeMOELoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], cur_task=1, expert_num=4)
        model = get_peft_model(model, config)
    else:
        from peft import get_peft_model

        from mcitlib.tuners._peft_registry import tuner_model_context
        if args.method == "olora":
            from mcitlib.tuners.variants.olora import MCITOrthogonalLoraConfig, MCITOrthogonalLoraModel
            config = MCITOrthogonalLoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], cur_task=1, expert_num=4)
            context = tuner_model_context("MOE_LORA_CoIN", MCITOrthogonalLoraModel)
        else:
            from mcitlib.tuners.variants.hide import MCITHiDeLoraConfig, MCITHiDeLoraModel
            config = MCITHiDeLoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], cur_task=1, expert_num=4)
            context = tuner_model_context("MOE_LORA_HiDe", MCITHiDeLoraModel)
        with context:
            model = get_peft_model(model, config)

    x = torch.randn(2, 3, 8)
    model.train()
    train_output = model(x)
    train_output.sum().backward()
    gradients = {name: param.grad.detach().cpu() for name, param in model.named_parameters() if param.grad is not None}
    model.eval()
    if args.method == "hide":
        for layer in model.model.model.layers:
            layer.q_proj.expert_weight = [0.25] * 4
    eval_output = model(x)
    torch.save({
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "train_output": train_output.detach().cpu(),
        "eval_output": eval_output.detach().cpu(),
        "gradients": gradients,
    }, args.out)


if __name__ == "__main__":
    main()
