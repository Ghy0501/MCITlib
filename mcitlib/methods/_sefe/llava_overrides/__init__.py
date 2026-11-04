# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright (c) 2025 Jinpeng Chen
# Copyright (c) 2025-2026 MCITlib authors
"""SEFE's isolated overrides from the newer LLaVA snapshot."""

from .language_model.llava_llama import LlavaLlamaForCausalLM

__all__ = ["LlavaLlamaForCausalLM"]
