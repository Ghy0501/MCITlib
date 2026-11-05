# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""V3 CoIN evaluation modules and auditable method-specific overrides."""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Scorers named in v3 ModalPrompt scripts that exist only as the shared (byte-identical) CoIN copy.
MODALPROMPT_SHARED_SCORERS = frozenset({"eval_deepseek_r1", "eval_caption"})


def resolve_eval_module(module: str, method_id: str) -> str:
    """Map a recorded v3 evaluation module to its exact migrated implementation."""
    if module.startswith("llava.eval.ModalPrompt."):
        leaf = module.rsplit(".", 1)[1]
        private = _REPO_ROOT / "mcitlib/methods/_modalprompt/eval_runners" / f"{leaf}.py"
        if private.is_file():
            return f"mcitlib.methods._modalprompt.eval_runners.{leaf}"
        # v3's ModalPrompt UCIT scripts call llava.eval.ModalPrompt.eval_deepseek_r1 / eval_caption,
        # which do not exist in its tree (ModuleNotFoundError in v3).  Its own llava/eval/eval_*.py
        # and every other method's CoIN copy are byte-identical, so the shared scorer is used (AR-6,
        # verified by tests/equivalence/test_eval_migration.py).
        shared = _REPO_ROOT / "mcitlib/backbones/llava/eval_runners/coin" / f"{leaf}.py"
        if shared.is_file() and leaf in MODALPROMPT_SHARED_SCORERS:
            return f"mcitlib.backbones.llava.eval_runners.coin.{leaf}"
        raise FileNotFoundError(f"missing ModalPrompt evaluation module: {private}")
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
