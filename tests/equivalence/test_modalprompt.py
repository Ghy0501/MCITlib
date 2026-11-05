# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""T1: ModalPrompt prompt initialization, prototypes, gradients, and save flags."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch

from tests.conftest import reference_env


def run(repo_root: Path, v3_root: Path, implementation: str, output: Path) -> None:
    if implementation == "v3":
        pythonpath = os.pathsep.join((str(v3_root / "LLaVA/ModalPrompt"), str(v3_root / "LLaVA/LoRA-FT")))
    else:
        pythonpath = os.pathsep.join((str(repo_root), str(repo_root / "third_party/LLaVA"), str(repo_root / "third_party/peft")))
    env = reference_env(implementation, pythonpath)
    result = subprocess.run(
        [sys.executable, str(repo_root / "tests/equivalence/_modalprompt_harness.py"),
         "--implementation", implementation, "--out", str(output)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(f"{implementation}/modalprompt: {result.stdout}\n{result.stderr}")


def test_modalprompt_prompt_and_model_equivalence(repo_root: Path, v3_root: Path, tmp_path: Path) -> None:
    v3_path, v4_path = tmp_path / "v3.pt", tmp_path / "v4.pt"
    run(repo_root, v3_root, "v3", v3_path)
    run(repo_root, v3_root, "v4", v4_path)
    v3 = torch.load(v3_path, map_location="cpu")
    v4 = torch.load(v4_path, map_location="cpu")
    assert v3["state_dict"].keys() == v4["state_dict"].keys()
    for name in v3["state_dict"]:
        assert torch.equal(v3["state_dict"][name], v4["state_dict"][name]), name
    for key in ("prompt_ids", "embed_tokens_mask", "prototypes", "train_loss", "eval_loss"):
        assert torch.equal(v3[key], v4[key]), key
    assert v3["requires_grad_before_save"] == v4["requires_grad_before_save"]
    assert v3["requires_grad_after_save"] == v4["requires_grad_after_save"]
    assert v3["gradients"].keys() == v4["gradients"].keys()
    for name in v3["gradients"]:
        assert torch.equal(v3["gradients"][name], v4["gradients"][name]), name
