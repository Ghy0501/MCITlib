# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Single registration point for MCITlib tuners and v3 serialized names."""
from __future__ import annotations

from contextlib import contextmanager

from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.utils import PeftType

from .mcit_expert_lora import MCITExpertLoraConfig, MCITExpertLoraModel
from .mcit_separable_expert_lora import MCITSeparableExpertLoraConfig, MCITSeparableExpertLoraModel
from .variants.clmoe import MCITClMoELoraConfig, MCITClMoELoraModel
from .variants.disco import MCITDiscoLoraConfig, MCITDiscoLoraModel
from .variants.hide import MCITHiDeLoraConfig, MCITHiDeLoraModel
from .variants.olora import MCITOrthogonalLoraConfig, MCITOrthogonalLoraModel
from .variants.same import SAMEConfig, SAMELinear, SAMEModel
from .variants.sigma import SiGMAConfig, SiGMAModel

# These aliases are compatibility only. Shared implementation classes use MCIT.
CoINMOELoraConfig = MCITExpertLoraConfig
CoINMOELoraModel = MCITExpertLoraModel
HiDeMOELoraConfig = MCITHiDeLoraConfig
HiDeMOELoraModel = MCITHiDeLoraModel
CLMoEMOELoraConfig = MCITClMoELoraConfig
CLMoEMOELoraModel = MCITClMoELoraModel
SMoLoraConfig = MCITSeparableExpertLoraConfig
SMoLoraModel = MCITSeparableExpertLoraModel


def register_mcit_tuners() -> None:
    """Register the v3 PeftType string without duplicating the tuner."""
    if not hasattr(PeftType, "MOE_LORA_CoIN"):
        setattr(PeftType, "MOE_LORA_CoIN", "MOE_LORA_CoIN")
    if not hasattr(PeftType, "MOE_LORA_HiDe"):
        setattr(PeftType, "MOE_LORA_HiDe", "MOE_LORA_HiDe")
    if not hasattr(PeftType, "MOE_LORA_CLMoE"):
        setattr(PeftType, "MOE_LORA_CLMoE", "MOE_LORA_CLMoE")
    if not hasattr(PeftType, "SMOLORA"):
        setattr(PeftType, "SMOLORA", "SMOLORA")
    for name in ("MOE_LORA_SAME", "MOE_LORA_SIGMA"):
        if not hasattr(PeftType, name):
            setattr(PeftType, name, name)
    PEFT_TYPE_TO_CONFIG_MAPPING["MOE_LORA_SAME"] = SAMEConfig
    PEFT_TYPE_TO_MODEL_MAPPING["MOE_LORA_SAME"] = SAMEModel
    PEFT_TYPE_TO_CONFIG_MAPPING["MOE_LORA_SIGMA"] = SiGMAConfig
    PEFT_TYPE_TO_MODEL_MAPPING["MOE_LORA_SIGMA"] = SiGMAModel
    PEFT_TYPE_TO_CONFIG_MAPPING["MOE_LORA_CoIN"] = MCITExpertLoraConfig
    PEFT_TYPE_TO_MODEL_MAPPING["MOE_LORA_CoIN"] = MCITExpertLoraModel
    PEFT_TYPE_TO_CONFIG_MAPPING["MOE_LORA_HiDe"] = MCITHiDeLoraConfig
    PEFT_TYPE_TO_MODEL_MAPPING["MOE_LORA_HiDe"] = MCITHiDeLoraModel
    PEFT_TYPE_TO_CONFIG_MAPPING["MOE_LORA_CLMoE"] = MCITClMoELoraConfig
    PEFT_TYPE_TO_MODEL_MAPPING["MOE_LORA_CLMoE"] = MCITClMoELoraModel
    PEFT_TYPE_TO_CONFIG_MAPPING["SMOLORA"] = MCITSeparableExpertLoraConfig
    PEFT_TYPE_TO_MODEL_MAPPING["SMOLORA"] = MCITSeparableExpertLoraModel


@contextmanager
def tuner_model_context(peft_type: str, model_class, config_class=None):
    """Temporarily map a serialized PeftType to one method's implementation."""
    previous_model = PEFT_TYPE_TO_MODEL_MAPPING.get(peft_type)
    previous_config = PEFT_TYPE_TO_CONFIG_MAPPING.get(peft_type)
    PEFT_TYPE_TO_MODEL_MAPPING[peft_type] = model_class
    if config_class is not None:
        PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] = config_class
    try:
        yield
    finally:
        if previous_model is None:
            PEFT_TYPE_TO_MODEL_MAPPING.pop(peft_type, None)
        else:
            PEFT_TYPE_TO_MODEL_MAPPING[peft_type] = previous_model
        if config_class is not None:
            if previous_config is None:
                PEFT_TYPE_TO_CONFIG_MAPPING.pop(peft_type, None)
            else:
                PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] = previous_config


# peft 0.4.0 only knows how to read/write adapter weights for its own PeftTypes.  Every v3 method
# fork added its serialized type to the LoRA branch of ``get/set_peft_model_state_dict``
# (``if config.peft_type in (LORA, ADALORA, <its type>)``).  The vendored peft stays pristine, so
# the same behaviour is installed here; without it the first ``save_pretrained`` of every
# expert-LoRA method (and every ``PeftModel.from_pretrained`` in evaluation) raises
# ``NotImplementedError``.
LORA_LIKE_PEFT_TYPES = frozenset({"MOE_LORA_CoIN", "MOE_LORA_HiDe", "MOE_LORA_CLMoE", "SMOLORA", "MOE_LORA_SAME", "MOE_LORA_SIGMA"})


def install_state_dict_support() -> None:
    import peft
    import peft.peft_model as peft_model_module
    import peft.utils as peft_utils
    import peft.utils.save_and_load as save_and_load

    if getattr(save_and_load, "_mcit_lora_like_support", False):
        return
    original_get = save_and_load.get_peft_model_state_dict
    original_set = save_and_load.set_peft_model_state_dict

    def _as_lora(model, adapter_name, call):
        config = model.peft_config[adapter_name]
        if str(getattr(config.peft_type, "value", config.peft_type)) not in LORA_LIKE_PEFT_TYPES:
            return call()
        serialized = config.peft_type
        config.peft_type = PeftType.LORA
        try:
            return call()
        finally:
            config.peft_type = serialized

    def get_peft_model_state_dict(model, state_dict=None, adapter_name="default"):
        return _as_lora(model, adapter_name, lambda: original_get(model, state_dict=state_dict, adapter_name=adapter_name))

    def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
        config = model.peft_config[adapter_name]
        if str(getattr(config.peft_type, "value", config.peft_type)) == "MOE_LORA_SAME":
            # PEFT 0.4 uses a global str.replace to insert the adapter name.
            # SAME has one-character state names (u/s); global replacement
            # corrupts unrelated characters in their full parameter paths.
            # Split only the first LoRA component and preserve the remainder.
            mapped = {}
            for key, value in peft_model_state_dict.items():
                if model.modules_to_save is not None:
                    for name in model.modules_to_save:
                        if name in key:
                            key = key.replace(name, f"{name}.modules_to_save.{adapter_name}")
                            break
                if "lora_" in key:
                    prefix, _, suffix = key.partition("lora_")
                    component, dot, tail = suffix.partition(".")
                    if not component:
                        raise ValueError("invalid SAME adapter parameter name")
                    key = f"{prefix}lora_{component}.{adapter_name}" + (f".{tail}" if dot else "")
                if key in mapped:
                    raise ValueError(f"duplicate SAME adapter parameter after mapping: {key}")
                mapped[key] = value
            # strict=False is necessary because adapter checkpoints exclude the
            # base model and (for SAME) the separately saved projector.  It
            # would also silently accept an entirely missing adapter layer,
            # however, so require every tensor belonging to every injected
            # SAME layer before loading any of the adapter into the model.
            # A separate non-LoRA partial load never enters this PEFT branch.
            expected = set()
            prefixes = tuple(f"{part}.{adapter_name}." for part in
                             ("lora_A", "lora_B", "lora_router", "lora_state"))
            for module_name, module in model.named_modules():
                if isinstance(module, SAMELinear):
                    # Native LLaVA saves model.named_parameters(), which omits
                    # duplicate aliases for tied Expert.weight/mlp.weight.
                    # Requiring state_dict() here would demand both aliases and
                    # reject a valid native checkpoint.
                    expected.update(f"{module_name}.{key}" for key, _ in module.named_parameters()
                                    if key.startswith(prefixes))
            if not expected:
                raise RuntimeError("SAME adapter state is incomplete; no injected SAME adapter tensors")
            missing = expected.difference(mapped)
            if missing:
                raise RuntimeError(f"SAME adapter state is incomplete; missing {min(missing)}")
            return model.load_state_dict(mapped, strict=False)
        return _as_lora(
            model, adapter_name,
            lambda: original_set(model, peft_model_state_dict, adapter_name=adapter_name),
        )

    for module in (save_and_load, peft_utils, peft, peft_model_module):
        if hasattr(module, "get_peft_model_state_dict"):
            module.get_peft_model_state_dict = get_peft_model_state_dict
        if hasattr(module, "set_peft_model_state_dict"):
            module.set_peft_model_state_dict = set_peft_model_state_dict
    save_and_load._mcit_lora_like_support = True


# Which tuner implements the serialized PeftType for each method.  v3 had one ``CoIN`` /
# ``HiDe`` / ``CLMoE`` package per method directory, so "MOE_LORA_CoIN" meant a different
# class depending on the method; v4 selects it explicitly.  Training does this through
# ``tuner_model_context``; evaluation processes call ``install_method_adapter``.
METHOD_ADAPTERS = {
    "same": ("MOE_LORA_SAME", SAMEConfig, SAMEModel),
    "sigma": ("MOE_LORA_SIGMA", SiGMAConfig, SiGMAModel),
    "moelora": ("MOE_LORA_CoIN", MCITExpertLoraConfig, MCITExpertLoraModel),
    "mrlora": ("MOE_LORA_CoIN", MCITExpertLoraConfig, MCITExpertLoraModel),
    "olora": ("MOE_LORA_CoIN", MCITOrthogonalLoraConfig, MCITOrthogonalLoraModel),
    "disco": ("MOE_LORA_CoIN", MCITDiscoLoraConfig, MCITDiscoLoraModel),
    "hide": ("MOE_LORA_HiDe", MCITHiDeLoraConfig, MCITHiDeLoraModel),
    "clmoe": ("MOE_LORA_CLMoE", MCITClMoELoraConfig, MCITClMoELoraModel),
    "smolora": ("SMOLORA", MCITSeparableExpertLoraConfig, MCITSeparableExpertLoraModel),
}


def install_method_adapter(method_id: str | None) -> bool:
    """Make ``PeftModel.from_pretrained`` build ``method_id``'s own tuner for the whole process.

    Without this an evaluation process fails with ``KeyError: 'MOE_LORA_CoIN'`` (nothing has
    registered the serialized type) or, worse, builds the wrong variant.  Methods that use
    plain LoRA (lora_ft, replay, sefe, keeplora, modalprompt) need nothing and return False.
    """
    entry = METHOD_ADAPTERS.get(method_id or "")
    if entry is None:
        return False
    peft_type, config_class, model_class = entry
    PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] = config_class
    PEFT_TYPE_TO_MODEL_MAPPING[peft_type] = model_class
    return True


register_mcit_tuners()
install_state_dict_support()

__all__ = [
    "CoINMOELoraConfig", "CoINMOELoraModel", "HiDeMOELoraConfig", "HiDeMOELoraModel",
    "CLMoEMOELoraConfig", "CLMoEMOELoraModel", "SMoLoraConfig", "SMoLoraModel",
    "register_mcit_tuners", "tuner_model_context", "METHOD_ADAPTERS", "install_method_adapter",
    "LORA_LIKE_PEFT_TYPES", "install_state_dict_support",
]
