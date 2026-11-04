# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""T2: isolated v3/v4 data tensor comparisons on synthetic fixtures."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


def make_fixture(root: Path, multi_image: bool = False) -> tuple[Path, Path]:
    image_dir = root / "images"
    image_dir.mkdir()
    Image.new("RGB", (5, 3), (20, 40, 60)).save(image_dir / "a.png")
    Image.new("RGB", (2, 6), (80, 30, 10)).save(image_dir / "b.png")
    first_image = ["a.png", "b.png"] if multi_image else "a.png"
    records = [
        {"id": "one", "image": first_image, "conversations": [
            {"from": "human", "value": "<image>\nWhat is shown?"},
            {"from": "gpt", "value": "A small colored square."},
        ]},
        {"id": "two", "image": ["a.png", "b.png"] if multi_image else "b.png", "conversations": [
            {"from": "human", "value": "Describe the second image."},
            {"from": "gpt", "value": "Another rectangle."},
        ]},
    ]
    data = root / "data.json"
    data.write_text(json.dumps(records), encoding="utf-8")
    return data, image_dir


def run_harness(repo_root: Path, v3_root: Path, implementation: str, entry: str, data: Path, images: Path, out: Path, stub: Path, method_dir: str | None = None) -> None:
    harness = repo_root / "tests/equivalence/_v3_harness.py"
    if implementation == "v3":
        method_dir = method_dir or ("LoRA-FT" if entry == "standard" else "MoELoRA")
        pythonpath = os.pathsep.join((str(v3_root / "LLaVA" / method_dir), str(v3_root / "LLaVA/LoRA-FT"), str(stub), str(repo_root)))
    else:
        pythonpath = os.pathsep.join((str(repo_root), str(repo_root / "third_party/LLaVA"), str(repo_root / "third_party/peft"), str(stub)))
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    env.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run(
        [sys.executable, str(harness), "--implementation", implementation, "--entry", entry,
         "--method", method_dir or "", "--data", str(data), "--image-folder", str(images), "--out", str(out)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(f"{implementation} harness failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


@pytest.mark.parametrize("entry", ["standard", "moe"])
def test_single_image_data_matches_v3(repo_root: Path, v3_root: Path, tmp_path: Path, entry: str) -> None:
    data, images = make_fixture(tmp_path)
    stub = tmp_path / "stubs"
    stub.mkdir()
    (stub / "deepspeed.py").write_text("# import-only stub for data tests\n", encoding="utf-8")
    v3_output, v4_output = tmp_path / "v3.pt", tmp_path / "v4.pt"
    run_harness(repo_root, v3_root, "v3", entry, data, images, v3_output, stub)
    run_harness(repo_root, v3_root, "v4", entry, data, images, v4_output, stub)
    v3 = torch.load(v3_output, map_location="cpu")
    v4 = torch.load(v4_output, map_location="cpu")
    assert v3.keys() == v4.keys()
    for key in v3:
        assert torch.equal(v3[key], v4[key]), key


def test_standard_multi_image_data_matches_v3(repo_root: Path, v3_root: Path, tmp_path: Path) -> None:
    data, images = make_fixture(tmp_path, multi_image=True)
    stub = tmp_path / "stubs"
    stub.mkdir()
    (stub / "deepspeed.py").write_text("# import-only stub for data tests\n", encoding="utf-8")
    v3_output, v4_output = tmp_path / "v3.pt", tmp_path / "v4.pt"
    run_harness(repo_root, v3_root, "v3", "standard", data, images, v3_output, stub)
    run_harness(repo_root, v3_root, "v4", "standard", data, images, v4_output, stub)
    v3 = torch.load(v3_output, map_location="cpu")
    v4 = torch.load(v4_output, map_location="cpu")
    for key in v3:
        assert torch.equal(v3[key], v4[key]), key


@pytest.mark.parametrize("method_dir", ["OLoRA", "HiDe", "CL-MoE", "SMoLoRA", "DISCO", "ModalPrompt"])
def test_variant_data_matches_their_v3_directories(repo_root: Path, v3_root: Path, tmp_path: Path, method_dir: str) -> None:
    data, images = make_fixture(tmp_path)
    stub = tmp_path / "stubs"
    stub.mkdir()
    (stub / "deepspeed.py").write_text("# import-only stub for data tests\n", encoding="utf-8")
    v3_output, v4_output = tmp_path / "v3.pt", tmp_path / "v4.pt"
    entry = "standard" if method_dir == "ModalPrompt" else "moe"
    run_harness(repo_root, v3_root, "v3", entry, data, images, v3_output, stub, method_dir=method_dir)
    run_harness(repo_root, v3_root, "v4", entry, data, images, v4_output, stub)
    v3 = torch.load(v3_output, map_location="cpu")
    v4 = torch.load(v4_output, map_location="cpu")
    for key in v3:
        assert torch.equal(v3[key], v4[key]), key
