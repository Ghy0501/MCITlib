# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Install method-private model classes without editing third_party."""
from __future__ import annotations


def install_model_override(override: str | type | None) -> None:
    if override is None:
        return
    if override == "disco":
        from ...methods._disco.llava_overrides import LlavaLlamaForCausalLM as override_cls
    elif override == "hide":
        from ...methods._hide.llava_overrides import LlavaLlamaForCausalLM as override_cls
    elif isinstance(override, str):
        raise NotImplementedError(f"model override is not registered: {override}")
    else:
        override_cls = override
    import llava.model
    import llava.model.language_model.llava_llama as llama_module
    llama_module.LlavaLlamaForCausalLM = override_cls
    llava.model.LlavaLlamaForCausalLM = override_cls
