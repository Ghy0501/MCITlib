# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""T1: O-LoRA and HiDe module/forward/gradient equivalence."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def run(repo_root: Path, v3_root: Path, implementation: str, method: str, output: Path) -> None:
    if implementation == "v3":
        method_dir = "OLoRA" if method == "olora" else "HiDe"
        pythonpath = os.pathsep.join((str(v3_root / "LLaVA" / method_dir), str(v3_root / "LLaVA/LoRA-FT")))
    else:
        pythonpath = os.pathsep.join((str(repo_root), str(repo_root / "third_party/LLaVA"), str(repo_root / "third_party/peft")))
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        [sys.executable, str(repo_root / "tests/equivalence/_tuner_harness.py"),
         "--implementation", implementation, "--method", method, "--out", str(output)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(f"{implementation}/{method}: {result.stdout}\n{result.stderr}")


@pytest.mark.parametrize("method", ["olora", "hide"])
def test_tuner_state_forward_and_gradients_match(repo_root: Path, v3_root: Path, tmp_path: Path, method: str) -> None:
    v3_path, v4_path = tmp_path / "v3.pt", tmp_path / "v4.pt"
    run(repo_root, v3_root, "v3", method, v3_path)
    run(repo_root, v3_root, "v4", method, v4_path)
    v3 = torch.load(v3_path, map_location="cpu")
    v4 = torch.load(v4_path, map_location="cpu")
    assert v3["state_dict"].keys() == v4["state_dict"].keys()
    for name in v3["state_dict"]:
        assert torch.equal(v3["state_dict"][name], v4["state_dict"][name]), name
    assert torch.equal(v3["train_output"], v4["train_output"])
    assert torch.equal(v3["eval_output"], v4["eval_output"])
    assert v3["gradients"].keys() == v4["gradients"].keys()
    for name in v3["gradients"]:
        assert torch.equal(v3["gradients"][name], v4["gradients"][name]), name
