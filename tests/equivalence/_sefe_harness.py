# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Small SEFE regularization-model comparison in the t437 environment."""
from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["v3", "v4"], required=True)
    parser.add_argument("--regularization", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from mcitlib.backbones.llava.overrides import allow_t437_llava_registration
    with allow_t437_llava_registration():
        if args.implementation == "v3":
            from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
        else:
            from mcitlib.methods._sefe.llava_overrides import LlavaLlamaForCausalLM
            from mcitlib.methods._sefe.llava_overrides.language_model.llava_llama import LlavaConfig

    torch.manual_seed(2026)
    config = LlavaConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, vocab_size=320)
    model = LlavaLlamaForCausalLM(config, regularization_info_path=args.regularization)
    from peft import LoraConfig, get_peft_model
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, target_modules=targets, task_type="CAUSAL_LM", bias="none"
    ))
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
    }, args.out)


if __name__ == "__main__":
    main()
