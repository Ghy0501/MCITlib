# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Method, backbone, benchmark registry and compatibility declarations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class MethodSpec:
    id: str
    v3_dir: str
    aliases: tuple[str, ...] = ()
    env: str = "llava"
    train_entry: str = "standard"
    deepspeed_config: str = "zero2"
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    separate_train_and_infer: bool = False
    extra_stages: tuple[str, ...] = ()
    model_override: str | None = None


@dataclass(frozen=True)
class BackboneSpec:
    id: str
    env: str
    v4_supported: bool
    capabilities: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class BenchmarkSpec:
    id: str
    v3_name: str
    train_script: str
    eval_dir: str
    tasks: tuple[str, ...]


METHODS = (
    MethodSpec("lora_ft", "LoRA-FT", aliases=("LoRA-FT", "lora-ft", "lora")),
    MethodSpec("replay", "Replay", aliases=("Replay",)),
    MethodSpec("moelora", "MoELoRA", aliases=("MoELoRA", "moe-lora"), train_entry="moe"),
    MethodSpec("mrlora", "MR-LoRA", aliases=("MR-LoRA", "mr-lora"), train_entry="standard", required_capabilities=frozenset({"router_data"}), extra_stages=("router",)),
    MethodSpec("olora", "OLoRA", aliases=("OLoRA", "O-LoRA", "o-lora"), train_entry="moe"),
    MethodSpec("clmoe", "CL-MoE", aliases=("CL-MoE", "cl-moe"), train_entry="moe"),
    MethodSpec("disco", "DISCO", aliases=("DISCO",), train_entry="moe", required_capabilities=frozenset({"text_tower", "clip_features"})),
    MethodSpec("hide", "HiDe", aliases=("HiDe", "hide-llava"), train_entry="moe", required_capabilities=frozenset({"text_tower", "clip_features", "layer_index"}), model_override="hide"),
    MethodSpec("smolora", "SMoLoRA", aliases=("SMoLoRA", "smo-lora"), train_entry="moe"),
    MethodSpec("modalprompt", "ModalPrompt", aliases=("ModalPrompt",), deepspeed_config="zero3_offload", separate_train_and_infer=True),
    MethodSpec("sefe", "RegLoRA", aliases=("RegLoRA", "reglora", "SEFE"), env="llava-t437"),
    MethodSpec("keeplora", "KeepLoRA", aliases=("KeepLoRA", "keep-lora"), env="llava-t437"),
)

BACKBONES = (
    BackboneSpec("llava", "llava", True, frozenset({"text_tower", "clip_features", "layer_index", "model_class_override", "router_data"})),
    BackboneSpec("internvl", "internvl", False),
    BackboneSpec("videollava", "videollava", False),
    BackboneSpec("videollama2", "videollama2", False),
)

BENCHMARKS = (
    BenchmarkSpec("ucit", "UCIT", "train_UCIT.sh", "Eval_UCIT", ("ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR-Math", "Flickr30k")),
    BenchmarkSpec("mllm_dcl", "MLLM-DCL", "train_DCL.sh", "Eval_MLLM_DCL", ("RS", "Med", "AD", "Sci", "Fin")),
    BenchmarkSpec("mllm_acl", "MLLM-ACL", "train_ACL.sh", "Eval_MLLM_ACL", ("OCR", "Math", "VP", "APP")),
)


def _key(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def get_method(value: str) -> MethodSpec:
    key = _key(value)
    for spec in METHODS:
        if key == _key(spec.id) or any(key == _key(alias) for alias in spec.aliases) or key == _key(spec.v3_dir):
            return spec
    raise KeyError(f"unknown method: {value}")


def get_backbone(value: str) -> BackboneSpec:
    key = _key(value)
    for spec in BACKBONES:
        if key == _key(spec.id):
            return spec
    raise KeyError(f"unknown backbone: {value}")


def get_benchmark(value: str) -> BenchmarkSpec:
    key = _key(value)
    aliases = {"dcl": "mllm_dcl", "acl": "mllm_acl"}
    key = aliases.get(key, key)
    for spec in BENCHMARKS:
        if key in {_key(spec.id), _key(spec.v3_name)}:
            return spec
    raise KeyError(f"unknown benchmark: {value}")


def compatibility(method: str, backbone: str) -> tuple[bool, str]:
    m, b = get_method(method), get_backbone(backbone)
    if not b.v4_supported:
        return False, "v3-only; v4 migration pending"
    if m.id == "keeplora" and b.id == "videollama2":
        return False, "not present in v3"
    missing = m.required_capabilities - b.capabilities
    if missing:
        return False, f"missing capabilities: {', '.join(sorted(missing))}"
    return True, m.env


def matrix() -> Iterable[dict[str, str | bool]]:
    for method in METHODS:
        for backbone in BACKBONES:
            supported, reason = compatibility(method.id, backbone.id)
            for benchmark in BENCHMARKS:
                yield {
                    "method": method.id,
                    "backbone": backbone.id,
                    "benchmark": benchmark.id,
                    "supported": supported,
                    "environment": method.env if supported else "—",
                    "note": reason,
                }
