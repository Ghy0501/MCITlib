# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Environment declarations used by the two LLaVA CI jobs."""
from __future__ import annotations

import pytest

from mcitlib.registry import get_method


@pytest.mark.env("llava-t437")
def test_sefe_and_keeplora_require_t437() -> None:
    assert get_method("sefe").env == "llava-t437"
    assert get_method("keeplora").env == "llava-t437"
    assert get_method("lora_ft").env == "llava"
