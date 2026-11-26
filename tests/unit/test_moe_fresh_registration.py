# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Each real training entry must install PEFT support in a fresh rank process."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("deepspeed")

METHODS = ("moelora", "olora", "disco", "hide", "smolora", "clmoe")


def _env(repo_root: Path, tmp_path: Path, method: str) -> dict[str, str]:
    result = os.environ.copy()
    for name in ("MCITLIB_NO_VENDORED_PATH", "MCITLIB_MODEL_OVERRIDE", "MCITLIB_SELFCHECK"):
        result.pop(name, None)
    # The CPU harness replaces the attention patch before training; no kernel is executed.
    paths = (repo_root, repo_root / "tests/fixtures/fake_flash_attn")
    result.update(PYTHONPATH=os.pathsep.join(map(str, paths)), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
                  TRITON_CACHE_DIR=str(tmp_path / "triton"), TORCH_EXTENSIONS_DIR=str(tmp_path / "torch"),
                  HF_HOME=str(tmp_path / "hf"), XDG_CACHE_HOME=str(tmp_path / "cache"),
                  MCITLIB_TRAIN_ENTRY="moe", MCITLIB_METHOD=method)
    if method in {"hide", "disco"}:
        result["MCITLIB_MODEL_OVERRIDE"] = method
    return result


@pytest.mark.parametrize("method", METHODS)
def test_fresh_train_entry_registers_and_round_trips_adapter(
    repo_root: Path, tmp_path: Path, method: str,
) -> None:
    harness = repo_root / "tests/unit/_fresh_moe_registration_harness.py"
    completed = subprocess.run([sys.executable, str(harness), str(tmp_path / "adapter")], cwd=repo_root,
                               env=_env(repo_root, tmp_path, method), text=True, capture_output=True,
                               check=False, timeout=90)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert f"fresh {method}: train entry, adapter construction, save and reload OK" in completed.stdout


def test_fresh_public_moelora_plugin_registers_its_tuner(repo_root: Path, tmp_path: Path) -> None:
    code = """
import sys
import transformers
from mcitlib.methods import create_method
from peft.utils import PeftType
assert 'mcitlib.tuners._peft_registry' not in sys.modules
assert not hasattr(PeftType, 'MOE_LORA_CoIN')
config = transformers.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4)
model = transformers.LlamaForCausalLM(config)
method = create_method('moelora', {'r':18, 'lora_alpha':36, 'expert_num':6, 'target_modules':['q_proj']})
adapted = method.build_trainable(model, None)
assert adapted.peft_config['default'].peft_type == 'MOE_LORA_CoIN'
assert any('lora_' in name for name, _ in adapted.named_parameters())
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=repo_root,
                               env=_env(repo_root, tmp_path, "moelora"), text=True, capture_output=True,
                               check=False, timeout=90)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
