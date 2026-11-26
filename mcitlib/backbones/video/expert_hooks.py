# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Video worker bridge for the existing expert LoRA method plugins."""
from __future__ import annotations

import re
from contextlib import nullcontext
from types import SimpleNamespace

from ...methods import create_method
from ...tuners._peft_registry import install_method_adapter

METHODS = frozenset({"moelora", "olora", "clmoe", "smolora", "sigma"})
LLM_TARGETS = (
    r"model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
)
ORTHOGONAL_EXPERT = re.compile(r"\.lora_[AB]\.default\.lora[AB]\.(\d+)\.")


def options_for(method, budgets, task, source, backbone):
    if method not in METHODS:
        return {}
    data = dict(expert_num=int(budgets["expert_num"]), task_id=task,
                cur_task=int(budgets.get("cur_task", task - 1)))
    if method == "smolora":
        directory = "Video-LLaVA" if backbone == "videollava" else "VideoLLaMA2"
        asset = source / "preparations/other_backbones/assets" / directory / "smolora" / "ins_emb_multi.pkl"
        data.update(ins_type=int(budgets["ins_type"]), ins_emb=str(asset),
                    ins_emb_sha256="c8a769183da3b2b41e5b08d09e928fd539e14717d490d2a61ed819ed17ff82ef")
    return data


class ExpertHooks:
    def __init__(self, method, options):
        self.method = method
        self.options = options
        self.plugin = None
        if not install_method_adapter(method):
            raise ValueError(f"Missing v4 expert tuner registration for {method}")

    def build(self, model, config):
        cfg = dict(
            r=config.r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            bias=config.bias, target_modules=LLM_TARGETS, **self.options,
        )
        ctx = SimpleNamespace(task_id=cfg["task_id"], output_dir=None)
        self.plugin = create_method(self.method, cfg, ctx)
        self.plugin.before_task(ctx)
        wrapped = self.plugin.build_trainable(model, None)
        if self.method == "olora":
            self.freeze_orthogonal_experts(wrapped)
        return wrapped

    def freeze_orthogonal_experts(self, model):
        selected = int(self.options["cur_task"])
        found = 0
        for name, parameter in model.named_parameters():
            match = ORTHOGONAL_EXPERT.search(name)
            if match:
                active = int(match.group(1)) == selected
                parameter.requires_grad_(active)
                found += int(active)
            elif ".lora_router." in name:
                parameter.requires_grad_(False)
        if not found:
            raise ValueError(f"No trainable OLoRA expert for task slot {selected}")

    def loss(self, model, inputs, outputs, base_loss):
        return self.plugin.extra_loss(model, inputs, outputs, base_loss)

    def checkpoint_scope(self, args):
        if self.method != "olora":
            return nullcontext(None)
        from .checkpointing import video_non_reentrant_scope

        if self.plugin.training_overrides() != {"gradient_checkpointing_kwargs": {"use_reentrant": False}}:
            raise RuntimeError("Unexpected OLoRA gradient checkpointing policy")
        return video_non_reentrant_scope(args, "OLoRA video")
