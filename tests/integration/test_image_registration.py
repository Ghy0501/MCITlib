# SPDX-License-Identifier: Apache-2.0
"""Image training modules must also work as the first LLaVA import in a process."""
import subprocess
import sys
from importlib.metadata import version

import pytest

from mcitlib.doctor import _pinned


def test_image_environment_has_internvit_dependency(repo_root):
    assert version("timm") == _pinned(repo_root / "envs/internvl-extra.txt")["timm"]


@pytest.mark.parametrize("entry", ["standard_train", "moe_train"])
def test_direct_training_import_registers_vendor_model(repo_root, entry):
    code = """
import importlib, sys
from transformers import AutoConfig, AutoModelForCausalLM
entry = importlib.import_module("mcitlib.backbones.llava." + sys.argv[1])
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
assert entry.LlavaLlamaForCausalLM is LlavaLlamaForCausalLM
assert type(AutoConfig.for_model("llava")) is LlavaConfig
assert AutoModelForCausalLM._model_mapping[LlavaConfig] is LlavaLlamaForCausalLM
"""
    result = subprocess.run([sys.executable, "-c", code, entry], cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
