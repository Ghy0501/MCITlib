# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""MR-LoRA router prompt and data-selection equivalence tests."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mcitlib.config.schema import DataSource, TaskSpec
from mcitlib.methods._mrlora import build_router_user_message
from mcitlib.methods.mrlora import MRLoRAMethod
from mcitlib.registry import BENCHMARKS

SOURCE_FILES = {
    "ucit": "model_ucit_router.py",
    "mllm_dcl": "model_dcl_router.py",
    "mllm_acl": "model_acl_router.py",
}


def source_constants(v3_root: Path, benchmark: str) -> dict[str, str]:
    path = v3_root / "LLaVA/MR-LoRA/llava/eval/CoIN" / SOURCE_FILES[benchmark]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"ROUTING_PROMPT", "PROMPT_AFTER_QUESTION"}:
                    constants[target.id] = ast.literal_eval(node.value)
    return constants


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=lambda item: item.id)
def test_router_prompt_matches_v3(v3_root: Path, benchmark) -> None:
    constants = source_constants(v3_root, benchmark.id)
    question = "  What is shown here?  "
    expected = f"{constants['ROUTING_PROMPT']}{question.strip()}\n{constants['PROMPT_AFTER_QUESTION']}"
    assert build_router_user_message(benchmark.id, question) == expected
    method = MRLoRAMethod({"benchmark": benchmark.id})
    assert method.build_router_user_message(question) == expected


def test_router_data_selection_is_explicit() -> None:
    source = DataSource(
        train_path="train.json",
        train_folder="train-images",
        router_path="router.json",
        router_folder="router-images",
    )
    ordinary = TaskSpec("ucit", 1, "ImageNet-R", "model", "data", "train", data=source)
    router = TaskSpec("ucit", 1, "ImageNet-R", "model", "data", "train", router=True, data=source)
    method = MRLoRAMethod()
    assert method.select_train_data(ordinary) is source
    selected = method.select_train_data(router)
    assert selected.train_path == "router.json"
    assert selected.train_folder == "router-images"
    assert [stage.name for stage in method.extra_stages()] == ["router"]
