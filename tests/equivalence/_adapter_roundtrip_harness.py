# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Save an adapter in one process and load it the way an evaluation process does (G8).

``save``  builds a toy PEFT model with the method's tuner (v3 or v4) and writes the adapter.
``load``  starts from nothing but the adapter directory: v3 uses its own ``PeftModel``;
          v4 behaves like ``eval_runners.entry`` (vendored imports, then
          ``install_method_adapter``), then ``PeftModel.from_pretrained`` + ``merge_and_unload``
          exactly as ``llava.model.builder.load_pretrained_model`` does.
"""
from __future__ import annotations

import argparse
import builtins
import io
import sys
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

    def forward(self, x):
        return self.model.layers[31].q_proj(x)


INSTRUCTION_EMBEDDINGS = torch.arange(768, dtype=torch.float32).reshape(2, 384).tolist()


def _patch_clmoe_v3() -> None:
    """Pin the two non-numerical side effects of v3's CL-MoE forward (task.txt, wall clock).

    v3 imports ``datetime`` inside ``forward``; call this after every real import is done.
    """
    real_open = builtins.open

    def reference_open(path, *args, **kwargs):
        if str(path) == "/your_path/MCITlib_v3/LLaVA/CL-MoE/task.txt":
            return io.StringIO("1")
        return real_open(path, *args, **kwargs)

    builtins.open = reference_open
    sys.modules["datetime"] = SimpleNamespace(datetime=SimpleNamespace(now=lambda: SimpleNamespace(second=0)))


def _train_like(model: nn.Module) -> None:
    """Give every adapter tensor non-trivial values (LoRA-B is zero-initialised, which would hide all differences)."""
    generator = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)


def save_v3(method: str, out: str) -> None:
    if method in {"moelora", "mrlora", "olora"}:
        from CoIN.peft import CoINMOELoraConfig, get_peft_model
        extra = {"cur_task": 1} if method == "olora" else {}
        config = CoINMOELoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], expert_num=4, **extra)
    elif method == "clmoe":
        from CLMoE.peft import CLMoEMOELoraConfig, get_peft_model
        config = CLMoEMOELoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], expert_num=4)
    else:
        from CoIN.peft import SMoLoraConfig, TaskType, get_peft_model
        config = SMoLoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj"], expert_num=4,
            ins_type=1, ins_emb=INSTRUCTION_EMBEDDINGS, task_type=TaskType.CAUSAL_LM_SMoLoRA,
        )
    model = get_peft_model(Toy(), config)
    _train_like(model)
    model.save_pretrained(out)


def save_v4(method: str, out: str) -> None:
    import mcitlib  # noqa: F401  (puts vendored llava/peft first, like every real entry point)
    from peft import get_peft_model

    from mcitlib.tuners._peft_registry import METHOD_ADAPTERS, tuner_model_context
    peft_type, config_class, model_class = METHOD_ADAPTERS[method]
    kwargs = dict(r=4, lora_alpha=8, target_modules=["q_proj"], expert_num=4)
    if method == "olora":
        kwargs["cur_task"] = 1
    if method == "smolora":
        kwargs.update(ins_type=1, ins_emb=INSTRUCTION_EMBEDDINGS, task_type="CAUSAL_LM_SMoLoRA")
    config = config_class(**kwargs)
    if method == "clmoe":
        from pathlib import Path

        from mcitlib.utils.router_stats import RouterStats
        config._mcit_router_stats = RouterStats(task_id=1, output_dir=Path("."))
    with tuner_model_context(peft_type, model_class, config_class):
        model = get_peft_model(Toy(), config)
    _train_like(model)
    model.save_pretrained(out)


def load(implementation: str, method: str, adapter: str, out: str) -> None:
    if implementation == "v3":
        if method == "clmoe":
            from CLMoE.peft import PeftModel
        else:
            from CoIN.peft import PeftModel
    else:
        # What an evaluation subprocess does before running a CoIN module.
        import mcitlib  # noqa: F401  (puts vendored llava/peft first)
        from mcitlib.backbones.llava.overrides import import_llava
        from mcitlib.tuners._peft_registry import install_method_adapter

        import_llava()
        install_method_adapter(method)
        from peft import PeftModel
    torch.manual_seed(7)
    model = Toy()
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    x = torch.randn(2, 3, 8)
    if implementation == "v3" and method == "clmoe":
        _patch_clmoe_v3()
    unmerged = model(x).detach().clone()
    model = model.merge_and_unload()
    model.eval()
    merged = model(x).detach().clone()
    torch.save({"unmerged": unmerged, "merged": merged}, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["save", "load"])
    parser.add_argument("--implementation", choices=["v3", "v4"], required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.manual_seed(1234)
    if args.mode == "save":
        (save_v3 if args.implementation == "v3" else save_v4)(args.method, args.adapter)
    else:
        load(args.implementation, args.method, args.adapter, args.out)


if __name__ == "__main__":
    main()
