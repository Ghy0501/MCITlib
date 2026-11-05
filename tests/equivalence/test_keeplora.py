# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""T1/T2: KeepLoRA model, LoRA initialization, subspaces, and data."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import transformers

from tests.conftest import reference_env, write_deepspeed_stub

pytestmark = pytest.mark.env("llava-t437")
if transformers.__version__ != "4.37.2":
    pytest.skip("KeepLoRA tests require transformers 4.37.2", allow_module_level=True)

from mcitlib.backbones.llava.overrides import allow_t437_llava_registration, install_model_override  # noqa: E402
from mcitlib.methods.keeplora import KeepLoRAMethod  # noqa: E402


def run(repo_root: Path, v3_root: Path, implementation: str, mode: str, output: Path) -> None:
    if implementation == "v3":
        pythonpath = os.pathsep.join((str(v3_root / "LLaVA/KeepLoRA"), str(v3_root / "LLaVA/LoRA-FT"), str(repo_root)))
    else:
        pythonpath = os.pathsep.join((str(repo_root), str(repo_root / "third_party/LLaVA"), str(repo_root / "third_party/peft")))
    env = reference_env(implementation, pythonpath)
    result = subprocess.run(
        [sys.executable, str(repo_root / "tests/equivalence/_keeplora_harness.py"),
         "--implementation", implementation, "--mode", mode, "--out", str(output)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(f"{implementation}/keeplora/{mode}: {result.stdout}\n{result.stderr}")


def assert_nested_equal(v3, v4, path="root") -> None:
    if isinstance(v3, dict):
        assert v3.keys() == v4.keys(), path
        for key in v3:
            assert_nested_equal(v3[key], v4[key], f"{path}.{key}")
    elif isinstance(v3, (list, tuple)):
        assert len(v3) == len(v4), path
        for index, (left, right) in enumerate(zip(v3, v4)):
            assert_nested_equal(left, right, f"{path}[{index}]")
    elif isinstance(v3, torch.Tensor):
        assert torch.equal(v3, v4), path
    else:
        assert v3 == v4, path


@pytest.mark.parametrize("mode", ["model", "lora", "subspace"])
def test_keeplora_numerics_match_v3(repo_root: Path, v3_root: Path, tmp_path: Path, mode: str) -> None:
    v3_path, v4_path = tmp_path / f"v3-{mode}.pt", tmp_path / f"v4-{mode}.pt"
    run(repo_root, v3_root, "v3", mode, v3_path)
    run(repo_root, v3_root, "v4", mode, v4_path)
    assert_nested_equal(torch.load(v3_path, map_location="cpu"), torch.load(v4_path, map_location="cpu"), mode)


def test_keeplora_override_and_training_defaults(repo_root: Path, monkeypatch) -> None:
    if importlib.util.find_spec("deepspeed") is None:  # never shadow a real install
        deepspeed_stub = ModuleType("deepspeed")
        deepspeed_stub.__spec__ = ModuleSpec("deepspeed", loader=None)
        monkeypatch.setitem(sys.modules, "deepspeed", deepspeed_stub)
    install_model_override("keeplora")
    import llava

    method = KeepLoRAMethod()
    assert llava.LlavaLlamaForCausalLM is method.model_class_override()
    assert method.env == "llava-t437"
    with allow_t437_llava_registration():
        from mcitlib.backbones.llava.standard_train import TrainingArguments
    assert TrainingArguments.__dataclass_fields__["freeze_lora_A"].default is False
    assert TrainingArguments.__dataclass_fields__["init_lora_from_gradients"].default is None


def test_keeplora_data_matches_v3_in_t437(repo_root: Path, v3_root: Path, tmp_path: Path) -> None:
    import runpy

    helpers = runpy.run_path(str(repo_root / "tests/equivalence/test_data_pipeline.py"))
    data, images = helpers["make_fixture"](tmp_path)
    run_harness = helpers["run_harness"]
    stub = tmp_path / "stubs"
    stub.mkdir()
    write_deepspeed_stub(stub)
    v3_output, v4_output = tmp_path / "v3-data.pt", tmp_path / "v4-data.pt"
    run_harness(repo_root, v3_root, "v3", "standard", data, images, v3_output, stub, method_dir="KeepLoRA")
    run_harness(repo_root, v3_root, "v4", "standard", data, images, v4_output, stub, method_dir="KeepLoRA")
    assert_nested_equal(torch.load(v3_output, map_location="cpu"), torch.load(v4_output, map_location="cpu"), "data")


def test_keeplora_merge_script_and_private_builder(repo_root: Path) -> None:
    from unittest.mock import Mock, patch

    from mcitlib.methods._keeplora.merge import merge_lora

    script = repo_root / "mcitlib/methods/_keeplora/merge.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    with allow_t437_llava_registration():
        from mcitlib.methods._keeplora.llava_overrides import builder
    model, tokenizer = Mock(), Mock()
    with patch.object(builder, "load_pretrained_model", return_value=(tokenizer, model, None, 0)) as load:
        merge_lora(SimpleNamespace(model_path="output/checkpoint-1", model_base="llava", save_model_path="merged"))
    load.assert_called_once()
    assert load.call_args.args[2] == "output_checkpoint-1"
    model.save_pretrained.assert_called_once_with("merged")
    tokenizer.save_pretrained.assert_called_once_with("merged")
