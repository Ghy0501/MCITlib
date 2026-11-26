# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""HiDe's LLaVA model overrides, kept outside third_party."""

from mcitlib.backbones.llava.overrides import llava_import_context

with llava_import_context():
    from .language_model.llava_llama import LlavaLlamaForCausalLM

__all__ = ["LlavaLlamaForCausalLM"]
