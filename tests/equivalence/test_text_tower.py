# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""T1: shared CLIP text and projected visual features match v3."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch


def test_shared_clip_capability_matches_v3(repo_root: Path, v3_root: Path, tmp_path: Path) -> None:
    outputs = []
    for implementation in ("v3", "v4"):
        output = tmp_path / f"{implementation}.pt"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((
            str(v3_root / "LLaVA/HiDe") if implementation == "v3" else str(repo_root),
            str(v3_root / "LLaVA/LoRA-FT") if implementation == "v3" else str(repo_root / "third_party/LLaVA"),
            str(repo_root / "third_party/peft"),
        ))
        result = subprocess.run(
            [sys.executable, str(repo_root / "tests/equivalence/_text_tower_harness.py"),
             "--implementation", implementation, "--out", str(output)],
            cwd=repo_root, env=env, capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        outputs.append(torch.load(output, map_location="cpu"))
    v3, v4 = outputs
    assert v3.keys() == v4.keys()
    for key in v3:
        assert torch.equal(v3[key], v4[key]) if isinstance(v3[key], torch.Tensor) else v3[key] == v4[key], key
