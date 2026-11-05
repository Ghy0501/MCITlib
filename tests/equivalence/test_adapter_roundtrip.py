# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""G8: an evaluation process must be able to load what training saved.

v3 trained and evaluated each method with its own peft fork, so the serialized
``peft_type`` always resolved to the right tuner.  v4 has one peft; the evaluation
entry therefore has to register the method's tuner.  These tests start fresh
interpreters exactly like the training/evaluation subprocesses and prove that

* a v4 evaluation process can load adapters written by v3 and by v4, and
* it produces the same outputs as v3's own loader (after the same ``merge_and_unload``
  that ``llava.model.builder.load_pretrained_model`` performs).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.conftest import reference_env

METHODS = ["moelora", "mrlora", "olora", "clmoe", "smolora"]
V3_DIR = {"moelora": "MoELoRA", "mrlora": "MR-LoRA", "olora": "OLoRA", "clmoe": "CL-MoE", "smolora": "SMoLoRA"}


def _run(repo_root: Path, pythonpath: list[str], *args: str) -> None:
    implementation = args[args.index("--implementation") + 1]
    env = reference_env(implementation, os.pathsep.join(pythonpath))
    result = subprocess.run(
        [sys.executable, str(repo_root / "tests/equivalence/_adapter_roundtrip_harness.py"), *args],
        cwd=repo_root, env=env, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        tail = (result.stderr or result.stdout).strip().splitlines()[-6:]
        raise AssertionError(f"{' '.join(args)}\n" + "\n".join(tail))


def _v3_path(v3_root: Path, method: str) -> list[str]:
    return [str(v3_root / "LLaVA" / V3_DIR[method]), str(v3_root / "LLaVA/LoRA-FT")]


def _v4_path(repo_root: Path) -> list[str]:
    # Deliberately *without* extra help: mcitlib's own import must put vendored peft first.
    return [str(repo_root)]


@pytest.mark.parametrize("method", METHODS)
def test_v4_eval_process_loads_a_v3_adapter_like_v3(repo_root: Path, v3_root: Path, tmp_path: Path, method: str) -> None:
    adapter = tmp_path / "adapter_v3"
    _run(repo_root, _v3_path(v3_root, method), "save", "--implementation", "v3", "--method", method, "--adapter", str(adapter))
    expected, actual = tmp_path / "v3_load.pt", tmp_path / "v4_load.pt"
    _run(repo_root, _v3_path(v3_root, method), "load", "--implementation", "v3", "--method", method,
         "--adapter", str(adapter), "--out", str(expected))
    _run(repo_root, _v4_path(repo_root), "load", "--implementation", "v4", "--method", method,
         "--adapter", str(adapter), "--out", str(actual))
    reference, produced = torch.load(expected), torch.load(actual)
    assert torch.equal(reference["unmerged"], produced["unmerged"]), "outputs differ before merge_and_unload"
    assert torch.equal(reference["merged"], produced["merged"]), "outputs differ after merge_and_unload"


@pytest.mark.parametrize("method", METHODS)
def test_v4_eval_process_loads_what_v4_training_saves(repo_root: Path, tmp_path: Path, method: str) -> None:
    adapter = tmp_path / "adapter_v4"
    _run(repo_root, _v4_path(repo_root), "save", "--implementation", "v4", "--method", method, "--adapter", str(adapter))
    produced = tmp_path / "v4_load.pt"
    _run(repo_root, _v4_path(repo_root), "load", "--implementation", "v4", "--method", method,
         "--adapter", str(adapter), "--out", str(produced))
    result = torch.load(produced)
    assert torch.isfinite(result["merged"]).all()
