# SPDX-License-Identifier: Apache-2.0
"""The isolated ModalPrompt CI environment exposes the real DeepSpeed 0.12.6 APIs."""
from __future__ import annotations

import pytest

from mcitlib.doctor import FAIL, OK, Report, check_environment
from mcitlib.methods.modalprompt import ModalPromptMethod
from mcitlib.registry import get_method

pytestmark = pytest.mark.env("llava-modalprompt-ds126")


def test_real_modal_environment_and_official_safe_api(repo_root) -> None:
    assert get_method("modalprompt").env == "llava-modalprompt-ds126"
    assert ModalPromptMethod.env == get_method("modalprompt").env
    report = Report()
    check_environment(report, get_method("modalprompt").env, repo_root)
    checks = {item.name: item for item in report.checks}
    assert checks["packages for environment `llava-modalprompt-ds126`"].status == OK
    assert checks["ModalPrompt DeepSpeed safe state API"].status == OK
    # CPU CI deliberately uses an import-only flash-attn stand-in for entrypoint tests.
    assert not [item for item in report.checks if item.status == FAIL and item.name != "import flash_attn"]
