# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Install method-private model classes without editing third_party."""
from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module


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


@contextmanager
def llava_import_context():
    """Make ``import llava`` valid under every supported transformers version.

    transformers >= 4.36 ships its own ``llava`` model type; the vendored LLaVA
    registers the same name when it is first imported, so that first import has to
    run inside :func:`allow_t437_llava_registration`.  With transformers 4.31 there
    is nothing to do.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    if "llava" in CONFIG_MAPPING_NAMES:
        with allow_t437_llava_registration():
            yield
    else:
        yield


def import_llava():
    """Import the vendored ``llava`` and ``peft`` exactly once and prove their origin.

    Every MCITlib entry point calls this before touching ``llava``.  It fails with a
    clear message when an older v3 installation shadows the vendored packages, which
    would otherwise silently run v3 code.
    """
    import sys
    from pathlib import Path

    from ...paths import repo_root

    with llava_import_context():
        # Keep model registration inside the compatibility context and in this order.
        import llava.model  # noqa: F401, I001
        import peft  # noqa: F401

        import llava  # noqa: F401
    third_party = repo_root() / "third_party"
    for name in ("llava", "peft"):
        origin = Path(sys.modules[name].__file__).resolve()
        if third_party not in origin.parents:
            raise RuntimeError(
                f"`{name}` was imported from {origin}, not from {third_party}. A stale v3 editable install "
                f"is probably shadowing the vendored package; run `pip uninstall -y {name}` (or use a fresh "
                "environment) and make sure MCITlib's PYTHONPATH is set by `mcitlib train/eval`."
            )
    return sys.modules["llava"]


def install_model_override(override: str | type | None) -> None:
    if override is None:
        return
    import_llava()
    if override == "octopus":
        import llava.model.builder as builder_module

        from ...methods._octopus.llava_overrides.builder import load_pretrained_model
        builder_module.load_pretrained_model = load_pretrained_model
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
    elif override == "keeplora":
        with allow_t437_llava_registration():
            from ...methods._keeplora.llava_overrides import LlavaLlamaForCausalLM as override_cls
    elif isinstance(override, str):
        raise NotImplementedError(f"model override is not registered: {override}")
    else:
        override_cls = override
    # Preserve the model import sequence after installing the compatibility override.
    import llava.model  # noqa: I001
    import llava.model.language_model.llava_llama as llama_module

    import llava
    llama_module.LlavaLlamaForCausalLM = override_cls
    llava.model.LlavaLlamaForCausalLM = override_cls
    llava.LlavaLlamaForCausalLM = override_cls
    if isinstance(override, str):
        private_builder = import_module(f"mcitlib.methods._{override}.llava_overrides.builder")
        import llava.model.builder as builder_module
        builder_module.load_pretrained_model = private_builder.load_pretrained_model
