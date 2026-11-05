# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""T1/T2: SEFE private model, regularization, data, and extra steps."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers

from tests.conftest import reference_env, write_deepspeed_stub

pytestmark = pytest.mark.env("llava-t437")
if transformers.__version__ != "4.37.2":
    pytest.skip("SEFE tests require transformers 4.37.2", allow_module_level=True)

from mcitlib.config.loader import load_layered_config  # noqa: E402
from mcitlib.methods._sefe.key_elements import identify_key_elements  # noqa: E402
from mcitlib.methods.sefe import SEFEMethod  # noqa: E402


def run(repo_root: Path, v3_root: Path, implementation: str, regularization: Path, output: Path) -> None:
    if implementation == "v3":
        pythonpath = os.pathsep.join((str(v3_root / "LLaVA/RegLoRA"), str(v3_root / "LLaVA/LoRA-FT"), str(repo_root)))
    else:
        pythonpath = os.pathsep.join((str(repo_root), str(repo_root / "third_party/LLaVA"), str(repo_root / "third_party/peft")))
    env = reference_env(implementation, pythonpath)
    result = subprocess.run(
        [sys.executable, str(repo_root / "tests/equivalence/_sefe_harness.py"),
         "--implementation", implementation, "--regularization", str(regularization), "--out", str(output)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(f"{implementation}/sefe: {result.stdout}\n{result.stderr}")


def regularization_fixture(root: Path) -> Path:
    stats = {}
    projections = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for layer in range(2):
        for projection in projections:
            parent = "self_attn" if projection.endswith("_proj") and projection in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
            name = f"base_model.model.model.layers.{layer}.{parent}.{projection}.weight"
            stats[name] = {"top2": torch.tensor([[0, 0], [1, 1]])}
    path = root / "regularization.pth"
    torch.save(stats, path)
    return path


def test_sefe_private_model_and_regularization_match(repo_root: Path, v3_root: Path, tmp_path: Path) -> None:
    regularization = regularization_fixture(tmp_path)
    v3_path, v4_path = tmp_path / "v3.pt", tmp_path / "v4.pt"
    run(repo_root, v3_root, "v3", regularization, v3_path)
    run(repo_root, v3_root, "v4", regularization, v4_path)
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


def test_sefe_key_elements_and_legacy_accumulation(tmp_path: Path) -> None:
    key = "base_model.model.model.layers.0.mlp.gate_proj.weight"
    a = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 100
    b = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 100
    for task in (1, 2):
        directory = tmp_path / f"Task{task}_llava_lora"
        directory.mkdir()
        torch.save({f"{key[:-7]}.lora_A.weight": a, f"{key[:-7]}.lora_B.weight": b}, directory / "adapter_model.bin")
    first = identify_key_elements(0, tmp_path, "llava")
    first_stats = torch.load(first)
    second = identify_key_elements(1, tmp_path, "llava")
    second_stats = torch.load(second)
    assert torch.equal(first_stats[key]["top2"], torch.tensor([[15, 3]]))
    assert torch.equal(second_stats[key]["top2"], torch.cat((first_stats[key]["top2"], first_stats[key]["top2"])))


def test_sefe_step_commands_and_model_override(repo_root: Path, tmp_path: Path) -> None:
    values = load_layered_config(repo_root / "configs", "llava", "ucit", "sefe", task_id=2)
    method = SEFEMethod(values)
    ctx = SimpleNamespace(task_id=2, output_dir=Path(values["train_config"]["output_dir"]))
    before = method.before_task(ctx)[0].command.argv
    after = method.after_task(ctx)[0].command.argv
    assert before == [
        "python", "mcitlib/methods/_sefe/key_elements.py", "0",
        values["train_config"]["key_path"], "llava",
    ]
    assert after[-1].endswith("Task2_llava_lora_merged")
    assert method.env == "llava-t437"
    from mcitlib.backbones.llava.overrides import install_model_override
    install_model_override("sefe")
    import llava
    assert llava.LlavaLlamaForCausalLM is method.model_class_override()


def test_sefe_data_matches_v3_in_t437(repo_root: Path, v3_root: Path, tmp_path: Path) -> None:
    import runpy

    helpers = runpy.run_path(str(repo_root / "tests/equivalence/test_data_pipeline.py"))
    data, images = helpers["make_fixture"](tmp_path)
    run_harness = helpers["run_harness"]
    stub = tmp_path / "stubs"
    stub.mkdir()
    write_deepspeed_stub(stub)
    v3_output, v4_output = tmp_path / "v3.pt", tmp_path / "v4.pt"
    run_harness(repo_root, v3_root, "v3", "standard", data, images, v3_output, stub, method_dir="RegLoRA")
    run_harness(repo_root, v3_root, "v4", "standard", data, images, v4_output, stub, method_dir="RegLoRA")
    v3 = torch.load(v3_output, map_location="cpu")
    v4 = torch.load(v4_output, map_location="cpu")
    for key in v3:
        assert torch.equal(v3[key], v4[key]), key


def test_sefe_merge_script_and_private_builder(repo_root: Path) -> None:
    from unittest.mock import Mock, patch

    from mcitlib.backbones.llava.overrides import allow_t437_llava_registration
    from mcitlib.methods._sefe.merge import merge_lora

    script = repo_root / "mcitlib/methods/_sefe/merge.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    with allow_t437_llava_registration():
        from mcitlib.methods._sefe.llava_overrides import builder
    model, tokenizer = Mock(), Mock()
    with patch.object(builder, "load_pretrained_model", return_value=(tokenizer, model, None, 0)) as load:
        merge_lora(SimpleNamespace(model_path="output/checkpoint-1", model_base="llava", save_model_path="merged"))
    load.assert_called_once()
    assert load.call_args.args[2] == "output_checkpoint-1"
    model.save_pretrained.assert_called_once_with("merged")
    tokenizer.save_pretrained.assert_called_once_with("merged")
