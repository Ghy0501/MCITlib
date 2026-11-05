# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""T1: private LLaVA model classes match their v3 method directories."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.conftest import reference_env


def run(repo_root: Path, v3_root: Path, implementation: str, method: str, output: Path) -> None:
    if implementation == "v3":
        method_dir = "DISCO" if method == "disco" else "HiDe"
        pythonpath = os.pathsep.join((str(v3_root / "LLaVA" / method_dir), str(v3_root / "LLaVA/LoRA-FT")))
    else:
        pythonpath = os.pathsep.join((str(repo_root), str(repo_root / "third_party/LLaVA"), str(repo_root / "third_party/peft")))
    env = reference_env(implementation, pythonpath)
    result = subprocess.run(
        [sys.executable, str(repo_root / "tests/equivalence/_model_override_harness.py"),
         "--implementation", implementation, "--method", method, "--out", str(output)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(f"{implementation}/{method}: {result.stdout}\n{result.stderr}")


@pytest.mark.parametrize("method", ["disco", "hide"])
def test_private_model_state_forward_and_gradients_match(repo_root: Path, v3_root: Path, tmp_path: Path, method: str) -> None:
    v3_path, v4_path = tmp_path / "v3.pt", tmp_path / "v4.pt"
    run(repo_root, v3_root, "v3", method, v3_path)
    run(repo_root, v3_root, "v4", method, v4_path)
    v3 = torch.load(v3_path, map_location="cpu")
    v4 = torch.load(v4_path, map_location="cpu")
    assert v3["state_dict"].keys() == v4["state_dict"].keys()
    for name in v3["state_dict"]:
        assert torch.equal(v3["state_dict"][name], v4["state_dict"][name]), name
    assert torch.equal(v3["train_loss"], v4["train_loss"])
    assert torch.equal(v3["eval_loss"], v4["eval_loss"])
    assert v3["requires_grad"] == v4["requires_grad"]
    assert v3["gradients"].keys() == v4["gradients"].keys()
    for name in v3["gradients"]:
        assert torch.equal(v3["gradients"][name], v4["gradients"][name]), name
