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
    script_aliases: tuple[tuple[str, str], ...] = ()
    supported_backbones: tuple[str, ...] | None = None
    supported_benchmarks: tuple[str, ...] | None = None


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
    MethodSpec(
        "mrlora", "MR-LoRA", aliases=("MR-LoRA", "mr-lora"), train_entry="standard",
        required_capabilities=frozenset({"router_data"}), extra_stages=("router",),
        script_aliases=(
            ("scripts/UCIT_extract_router_result.py", "mcitlib/methods/_mrlora/router_results/ucit.py"),
            ("scripts/DCL_extract_router_result.py", "mcitlib/methods/_mrlora/router_results/dcl.py"),
            ("scripts/ACL_extract_router_result.py", "mcitlib/methods/_mrlora/router_results/acl.py"),
        ),
    ),
    MethodSpec("olora", "OLoRA", aliases=("OLoRA", "O-LoRA", "o-lora"), train_entry="moe"),
    MethodSpec("clmoe", "CL-MoE", aliases=("CL-MoE", "cl-moe"), train_entry="moe"),
    MethodSpec("disco", "DISCO", aliases=("DISCO",), train_entry="moe", required_capabilities=frozenset({"text_tower", "clip_features", "layer_index"}), model_override="disco"),
    MethodSpec("hide", "HiDe", aliases=("HiDe", "hide-llava"), train_entry="moe", required_capabilities=frozenset({"text_tower", "clip_features", "layer_index"}), model_override="hide"),
    MethodSpec("smolora", "SMoLoRA", aliases=("SMoLoRA", "smo-lora"), train_entry="moe"),
    MethodSpec("modalprompt", "ModalPrompt", aliases=("ModalPrompt",), env="llava-modalprompt-ds126", deepspeed_config="zero3_offload", required_capabilities=frozenset({"text_tower", "clip_features"}), separate_train_and_infer=True, model_override="modalprompt"),
    MethodSpec(
        "sefe", "RegLoRA", aliases=("RegLoRA", "reglora", "SEFE"), env="llava-t437",
        required_capabilities=frozenset({"model_class_override"}), model_override="sefe",
        script_aliases=(
            ("key_elements_identification.py", "mcitlib/methods/_sefe/key_elements.py"),
            ("scripts/merge_lora_weights.py", "mcitlib/methods/_sefe/merge.py"),
        ),
    ),
    MethodSpec(
        "keeplora", "KeepLoRA", aliases=("KeepLoRA", "keep-lora"), env="llava-t437",
        required_capabilities=frozenset({"model_class_override"}), model_override="keeplora",
        script_aliases=(
            ("llava.train.extract_gradients", "mcitlib.methods._keeplora.extract_gradients"),
            ("llava.train.extract_weights", "mcitlib.methods._keeplora.extract_weights"),
            ("scripts/merge_lora_weights.py", "mcitlib/methods/_keeplora/merge.py"),
        ),
    ),
    MethodSpec("pclr", "PCLR", aliases=("PCLR",), extra_stages=("compression_integration",),
               supported_backbones=("llava",), supported_benchmarks=("ucit", "mllm_dcl")),
    MethodSpec("same", "SAME", aliases=("SAME",), train_entry="moe",
               supported_backbones=("llava",), supported_benchmarks=("ucit", "mllm_dcl")),
    MethodSpec("sigma", "SiGMA", aliases=("SiGMA",), train_entry="moe",
               supported_backbones=("llava",), supported_benchmarks=("ucit", "mllm_dcl")),
    MethodSpec("dgg", "DGG", aliases=("DGG",), separate_train_and_infer=True,
               supported_backbones=("llava",), supported_benchmarks=("ucit", "mllm_dcl")),
    MethodSpec("octopus", "Octopus", aliases=("Octopus",), model_override="octopus", extra_stages=("gpwc", "stage2"),
               supported_backbones=("llava",), supported_benchmarks=("ucit", "mllm_dcl")),
)

BACKBONES = (
    BackboneSpec("llava", "llava", True, frozenset({"text_tower", "clip_features", "layer_index", "model_class_override", "router_data"})),
    BackboneSpec("internvl", "llava", True, frozenset({"text_tower", "clip_features", "layer_index", "model_class_override", "router_data"})),
    BackboneSpec("videollava", "videollava", True, frozenset({"layer_index", "router_data", "text_tower", "clip_features"})),
    BackboneSpec("videollama2", "videollama2", True, frozenset({"layer_index", "router_data", "text_tower", "clip_features"})),
)

BENCHMARKS = (
    BenchmarkSpec("ucit", "UCIT", "train_UCIT.sh", "Eval_UCIT", ("ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR-Math", "Flickr30k")),
    BenchmarkSpec("mllm_dcl", "MLLM-DCL", "train_DCL.sh", "Eval_MLLM_DCL", ("RS", "Med", "AD", "Sci", "Fin")),
    BenchmarkSpec("mllm_acl", "MLLM-ACL", "train_ACL.sh", "Eval_MLLM_ACL", ("OCR", "Math", "VP", "APP")),
)
VIDEO_BACKBONES = frozenset({"videollava", "videollama2"})
VIDEO_METHODS = frozenset({
    "lora_ft", "replay", "moelora", "mrlora", "olora", "clmoe", "smolora", "sigma",
    "disco", "hide", "same", "pclr", "sefe", "keeplora", "dgg", "octopus", "modalprompt",
})
CL_VISTA = BenchmarkSpec(
    "cl_vista", "CL-VISTA", "train_CVU.sh", "Eval_CVU",
    ("counting", "space", "traffic", "movie", "gui", "science", "sports", "star"),
)

GENERAL_BENCHMARK = BenchmarkSpec(
    "general", "General_Benchmarks", "", "Eval_benchmarks", ("POPE", "MME", "MMBench", "SEED-Bench")
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
    key = {"video_llava": "videollava", "video_llama2": "videollama2"}.get(key, key)
    for spec in BACKBONES:
        if key == _key(spec.id):
            return spec
    raise KeyError(f"unknown backbone: {value}")


def get_benchmark(value: str) -> BenchmarkSpec:
    key = _key(value)
    aliases = {"dcl": "mllm_dcl", "acl": "mllm_acl", "cvu": "cl_vista"}
    key = aliases.get(key, key)
    for spec in (*BENCHMARKS, CL_VISTA):
        if key in {_key(spec.id), _key(spec.v3_name)}:
            return spec
    if key in {_key(GENERAL_BENCHMARK.id), _key(GENERAL_BENCHMARK.v3_name)}:
        return GENERAL_BENCHMARK
    raise KeyError(f"unknown benchmark: {value}")


def benchmark_supported(method: str, benchmark: str) -> bool:
    """Check a method's explicitly configured benchmark coverage."""
    spec, benchmark_id = get_method(method), get_benchmark(benchmark).id
    if benchmark_id == CL_VISTA.id:
        return spec.id in VIDEO_METHODS
    return spec.supported_benchmarks is None or benchmark_id in spec.supported_benchmarks


def compatibility(method: str, backbone: str) -> tuple[bool, str]:
    m, b = get_method(method), get_backbone(backbone)
    if b.id in VIDEO_BACKBONES:
        return (True, b.env) if m.id in VIDEO_METHODS else (False, "CL-VISTA video migration pending")
    if m.supported_backbones is not None and b.id not in m.supported_backbones:
        return False, "method is not implemented for this backbone"
    if not b.v4_supported:
        return False, "v3-only; v4 migration pending"
    missing = m.required_capabilities - b.capabilities
    if missing:
        return False, f"missing capabilities: {', '.join(sorted(missing))}"
    return True, m.env


def matrix() -> Iterable[dict[str, str | bool]]:
    for method in METHODS:
        for backbone in BACKBONES:
            supported, reason = compatibility(method.id, backbone.id)
            benchmarks = (CL_VISTA,) if backbone.id in VIDEO_BACKBONES else BENCHMARKS
            for benchmark in benchmarks:
                benchmark_ok = benchmark_supported(method.id, benchmark.id)
                internvl_ok = not (backbone.id == "internvl" and benchmark.id == "mllm_acl")
                available = supported and benchmark_ok and internvl_ok
                note = reason if not supported or available else (
                    "not configured for this benchmark" if not benchmark_ok else "InternVL benchmark not enabled")
                yield {
                    "method": method.id,
                    "backbone": backbone.id,
                    "benchmark": benchmark.id,
                    "supported": available,
                    "environment": reason if available else "—",
                    "note": note,
                }
