# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Single registration point for MCITlib tuners and v3 serialized names."""
from __future__ import annotations

from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.utils import PeftType

from .mcit_expert_lora import MCITExpertLoraConfig, MCITExpertLoraModel

# These aliases are compatibility only. Shared implementation classes use MCIT.
CoINMOELoraConfig = MCITExpertLoraConfig
CoINMOELoraModel = MCITExpertLoraModel


def register_mcit_tuners() -> None:
    """Register the v3 PeftType string without duplicating the tuner."""
    if not hasattr(PeftType, "MOE_LORA_CoIN"):
        setattr(PeftType, "MOE_LORA_CoIN", "MOE_LORA_CoIN")
    PEFT_TYPE_TO_CONFIG_MAPPING["MOE_LORA_CoIN"] = MCITExpertLoraConfig
    PEFT_TYPE_TO_MODEL_MAPPING["MOE_LORA_CoIN"] = MCITExpertLoraModel


register_mcit_tuners()

__all__ = ["CoINMOELoraConfig", "CoINMOELoraModel", "register_mcit_tuners"]
