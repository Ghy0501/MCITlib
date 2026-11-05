# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Install the active method's model hooks before running an evaluation module."""
from __future__ import annotations

import importlib
import json
import os
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m mcitlib.backbones.llava.eval_runners.entry MODULE [ARGS ...]")
    from ..overrides import import_llava, install_model_override

    import_llava()
    target = sys.argv[1]
    sys.argv = [target, *sys.argv[2:]]
    method_id = os.environ.get("MCITLIB_METHOD")
    adapter_installed = False
    if method_id:
        from ....registry import get_method
        from ....tuners._peft_registry import install_method_adapter

        adapter_installed = install_method_adapter(method_id)
        install_model_override(get_method(method_id).model_override)
    if os.environ.get("MCITLIB_SELFCHECK") == "1":
        importlib.import_module(target)
        print(json.dumps({
            "selfcheck": "ok", "target": target, "method": method_id, "adapter_installed": adapter_installed,
        }))
        return
    runpy.run_module(target, run_name="__main__")


if __name__ == "__main__":
    main()
