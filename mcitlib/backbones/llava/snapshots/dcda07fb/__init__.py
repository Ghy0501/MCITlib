# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Shared transformers-4.37 LLaVA snapshot for SEFE and KeepLoRA."""

from .language_model.llava_llama import LlavaLlamaForCausalLM

__all__ = ["LlavaLlamaForCausalLM"]
