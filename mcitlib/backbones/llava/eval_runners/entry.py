# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Install the active method's model hooks before running an evaluation module."""
from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m mcitlib.backbones.llava.eval_runners.entry MODULE [ARGS ...]")
    target = sys.argv[1]
    sys.argv = [target, *sys.argv[2:]]
    method_id = os.environ.get("MCITLIB_METHOD")
    if method_id:
        from ....registry import get_method
        from ..overrides import install_model_override

        install_model_override(get_method(method_id).model_override)
    runpy.run_module(target, run_name="__main__")


if __name__ == "__main__":
    main()
