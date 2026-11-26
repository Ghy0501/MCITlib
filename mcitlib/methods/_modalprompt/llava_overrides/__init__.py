# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""ModalPrompt's private LLaVA model override."""

from mcitlib.backbones.llava.overrides import llava_import_context

with llava_import_context():
    from .language_model.modal_prompt import MCITModalPrompt

__all__ = ["MCITModalPrompt"]
