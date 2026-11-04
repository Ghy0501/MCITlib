# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Install method-private model classes without editing third_party."""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def allow_t437_llava_registration():
    """Allow the known legacy LLaVA registration only during t437 imports."""
    from transformers import AutoConfig, AutoModelForCausalLM

    original_config = AutoConfig.register
    original_model = AutoModelForCausalLM.register

    def config_register(model_type, config, exist_ok=False):
        return original_config(model_type, config, exist_ok=True)

    def model_register(config, model, exist_ok=False):
        return original_model(config, model, exist_ok=True)

    AutoConfig.register = config_register
    AutoModelForCausalLM.register = model_register
    # transformers 4.37 removed these legacy helpers. The old LLaVA snapshot
    # imports them while loading its unused MPT branch; SEFE computes with Llama.
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    from transformers.models.bloom import modeling_bloom
    from transformers.models.opt import modeling_opt

    added_helpers = []
    for module in (modeling_bloom, modeling_opt):
        for name, value in (
            ("_expand_mask", AttentionMaskConverter._expand_mask),
            ("_make_causal_mask", AttentionMaskConverter._make_causal_mask),
        ):
            if not hasattr(module, name):
                setattr(module, name, value)
                added_helpers.append((module, name))
    try:
        yield
    finally:
        AutoConfig.register = original_config
        AutoModelForCausalLM.register = original_model
        for module, name in added_helpers:
            delattr(module, name)


def install_model_override(override: str | type | None) -> None:
    if override is None:
        return
    if override == "disco":
        from ...methods._disco.llava_overrides import LlavaLlamaForCausalLM as override_cls
    elif override == "modalprompt":
        from ...methods._modalprompt.llava_overrides import MCITModalPrompt as override_cls
    elif override == "hide":
        from ...methods._hide.llava_overrides import LlavaLlamaForCausalLM as override_cls
    elif override == "sefe":
        with allow_t437_llava_registration():
            from ...methods._sefe.llava_overrides import LlavaLlamaForCausalLM as override_cls
    elif isinstance(override, str):
        raise NotImplementedError(f"model override is not registered: {override}")
    else:
        override_cls = override
    import llava.model
    import llava.model.language_model.llava_llama as llama_module

    import llava
    llama_module.LlavaLlamaForCausalLM = override_cls
    llava.model.LlavaLlamaForCausalLM = override_cls
    llava.LlavaLlamaForCausalLM = override_cls
