# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Single registration point for MCITlib tuners and v3 serialized names."""
from __future__ import annotations

from contextlib import contextmanager

from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.utils import PeftType

from .mcit_expert_lora import MCITExpertLoraConfig, MCITExpertLoraModel
from .variants.clmoe import MCITClMoELoraConfig, MCITClMoELoraModel
from .variants.hide import MCITHiDeLoraConfig, MCITHiDeLoraModel

# These aliases are compatibility only. Shared implementation classes use MCIT.
CoINMOELoraConfig = MCITExpertLoraConfig
CoINMOELoraModel = MCITExpertLoraModel
HiDeMOELoraConfig = MCITHiDeLoraConfig
HiDeMOELoraModel = MCITHiDeLoraModel
CLMoEMOELoraConfig = MCITClMoELoraConfig
CLMoEMOELoraModel = MCITClMoELoraModel


def register_mcit_tuners() -> None:
    """Register the v3 PeftType string without duplicating the tuner."""
    if not hasattr(PeftType, "MOE_LORA_CoIN"):
        setattr(PeftType, "MOE_LORA_CoIN", "MOE_LORA_CoIN")
    if not hasattr(PeftType, "MOE_LORA_HiDe"):
        setattr(PeftType, "MOE_LORA_HiDe", "MOE_LORA_HiDe")
    if not hasattr(PeftType, "MOE_LORA_CLMoE"):
        setattr(PeftType, "MOE_LORA_CLMoE", "MOE_LORA_CLMoE")
    PEFT_TYPE_TO_CONFIG_MAPPING["MOE_LORA_CoIN"] = MCITExpertLoraConfig
    PEFT_TYPE_TO_MODEL_MAPPING["MOE_LORA_CoIN"] = MCITExpertLoraModel
    PEFT_TYPE_TO_CONFIG_MAPPING["MOE_LORA_HiDe"] = MCITHiDeLoraConfig
    PEFT_TYPE_TO_MODEL_MAPPING["MOE_LORA_HiDe"] = MCITHiDeLoraModel
    PEFT_TYPE_TO_CONFIG_MAPPING["MOE_LORA_CLMoE"] = MCITClMoELoraConfig
    PEFT_TYPE_TO_MODEL_MAPPING["MOE_LORA_CLMoE"] = MCITClMoELoraModel


@contextmanager
def tuner_model_context(peft_type: str, model_class):
    """Temporarily map a shared serialized PeftType to one method's model."""
    previous = PEFT_TYPE_TO_MODEL_MAPPING.get(peft_type)
    PEFT_TYPE_TO_MODEL_MAPPING[peft_type] = model_class
    try:
        yield
    finally:
        if previous is None:
            PEFT_TYPE_TO_MODEL_MAPPING.pop(peft_type, None)
        else:
            PEFT_TYPE_TO_MODEL_MAPPING[peft_type] = previous


register_mcit_tuners()

__all__ = [
    "CoINMOELoraConfig", "CoINMOELoraModel", "HiDeMOELoraConfig", "HiDeMOELoraModel",
    "CLMoEMOELoraConfig", "CLMoEMOELoraModel", "register_mcit_tuners", "tuner_model_context",
]
