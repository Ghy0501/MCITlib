# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Synthetic ModalPrompt prompt/embedding comparison without pretrained weights."""
from __future__ import annotations

import argparse
import re
from types import SimpleNamespace

import numpy as np
import torch


class ToyPromptTokenizer:
    def __init__(self):
        self.token_ids = {}
        self.model_max_length = 2048
        self.pad_token_id = 0

    def __len__(self):
        return 320 + len(self.token_ids)

    def add_special_tokens(self, values):
        added = 0
        for token in values["additional_special_tokens"]:
            if token not in self.token_ids:
                self.token_ids[token] = len(self)
                added += 1
        return added

    def __call__(self, text, **kwargs):
        ids = [self.token_ids[token] for token in re.findall(r"\[[^\]]+\]", text)]
        return SimpleNamespace(input_ids=torch.tensor([[1] + ids], dtype=torch.long))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["v3", "v4"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.implementation == "v3":
        from llava.model.language_model.llava_llama import LlavaConfig
        from llava.model.language_model.ModalPrompt import ModalPrompt as ModelClass
    else:
        from mcitlib.methods._modalprompt.llava_overrides import MCITModalPrompt as ModelClass
        from mcitlib.methods._modalprompt.llava_overrides.language_model.llava_llama import LlavaConfig

    np.random.seed(2026)
    torch.manual_seed(2026)
    config = LlavaConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, vocab_size=320)
    model = ModelClass(config)
    model.set_num_task(3)
    model.set_comtinual_prompts_tokenizer(ToyPromptTokenizer(), prefix_len=2)
    # Both implementations undergo v3's float16 conversion above, then use
    # fp32 for this CPU-only linear comparison (half addmm is unavailable).
    model.prompt_transform.to(dtype=torch.float32)
    model.set_cur_task(2)
    model.set_proto_input_ids()
    model.reset_vocab_size()
    prompt_ids = model.get_proto_input_ids(2, torch.device("cpu"))
    prototypes = []
    for index in range(2):
        embedded = model.model.embed_tokens(prompt_ids[index])
        prototypes.append(model.prompt_transform[index](embedded).mean(dim=0))
    prototype_output = torch.stack(prototypes)
    prototype_output.sum().backward()
    gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None}
    before_save = {name: p.requires_grad for name, p in model.named_parameters()}
    model.set_previou_transform_for_save(2)
    after_save = {name: p.requires_grad for name, p in model.named_parameters()}
    input_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    model.train()
    train_loss = model(input_ids=input_ids, labels=input_ids).loss
    model.eval()
    with torch.no_grad():
        eval_loss = model(input_ids=input_ids, labels=input_ids).loss
    torch.save({
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "prompt_ids": prompt_ids.detach().cpu(),
        "embed_tokens_mask": model.embed_tokens_mask_1d.detach().cpu(),
        "prototypes": prototype_output.detach().cpu(),
        "gradients": gradients,
        "requires_grad_before_save": before_save,
        "requires_grad_after_save": after_save,
        "train_loss": train_loss.detach().cpu(),
        "eval_loss": eval_loss.detach().cpu(),
    }, args.out)


if __name__ == "__main__":
    main()
