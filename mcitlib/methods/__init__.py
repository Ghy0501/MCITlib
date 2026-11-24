# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Continual-learning method plugin interfaces and lazy factory."""
from __future__ import annotations

from importlib import import_module
from typing import Any

_CLASSES = {
    "pclr": ("pclr", "PCLRMethod"),
    "same": ("same", "SAMEMethod"),
    "sigma": ("sigma", "SiGMAMethod"),
    "octopus": ("octopus", "OctopusMethod"),
    "lora_ft": ("lora_ft", "LoraFTMethod"),
    "replay": ("replay", "ReplayMethod"),
    "dgg": ("dgg", "DGGMethod"),
    "moelora": ("moelora", "MoELoRAMethod"),
    "mrlora": ("mrlora", "MRLoRAMethod"),
    "olora": ("olora", "OLoRAMethod"),
    "clmoe": ("clmoe", "CLMoEMethod"),
    "disco": ("disco", "DISCOMethod"),
    "hide": ("hide", "HiDeMethod"),
    "smolora": ("smolora", "SMoLoRAMethod"),
    "modalprompt": ("modalprompt", "ModalPromptMethod"),
    "sefe": ("sefe", "SEFEMethod"),
    "keeplora": ("keeplora", "KeepLoRAMethod"),
}


def create_method(method_id: str, cfg: dict[str, Any] | None = None, ctx: Any = None):
    """Instantiate only the selected method, keeping optional imports isolated."""
    try:
        module_name, class_name = _CLASSES[method_id]
    except KeyError as error:
        raise ValueError(f"unknown method plugin: {method_id}") from error
    module = import_module(f".{module_name}", __name__)
    return getattr(module, class_name)(cfg, ctx)


__all__ = ["create_method"]
