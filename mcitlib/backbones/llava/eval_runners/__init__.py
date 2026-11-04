# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""V3 CoIN evaluation modules and auditable method-specific overrides."""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_eval_module(module: str, method_id: str) -> str:
    """Map a recorded v3 evaluation module to its exact migrated implementation."""
    if module.startswith("llava.eval.ModalPrompt."):
        leaf = module.rsplit(".", 1)[1]
        target = f"mcitlib.methods._modalprompt.eval_runners.{leaf}"
        path = _REPO_ROOT / "mcitlib/methods/_modalprompt/eval_runners" / f"{leaf}.py"
        if not path.is_file():
            raise FileNotFoundError(f"missing ModalPrompt evaluation module: {path}")
        return target
    if not module.startswith("llava.eval.CoIN."):
        raise ValueError(f"unsupported LLaVA evaluation module: {module}")
    leaf = module.rsplit(".", 1)[1]
    private = _REPO_ROOT / "mcitlib/methods" / f"_{method_id}" / "eval_runners" / f"{leaf}.py"
    if private.is_file():
        return f"mcitlib.methods._{method_id}.eval_runners.{leaf}"
    shared = _REPO_ROOT / "mcitlib/backbones/llava/eval_runners/coin" / f"{leaf}.py"
    if not shared.is_file():
        raise FileNotFoundError(f"missing CoIN evaluation module: {shared}")
    return f"mcitlib.backbones.llava.eval_runners.coin.{leaf}"


__all__ = ["resolve_eval_module"]
