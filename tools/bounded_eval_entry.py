# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Run an existing evaluation module with a short, explicit generation limit."""
from __future__ import annotations

import importlib
import os
import runpy
import sys


def _target() -> str:
    target = os.environ.get("MCITLIB_SMOKE_EVAL_TARGET", "")
    if not target.startswith("mcitlib.") or ".eval_runners." not in target:
        raise ValueError(f"invalid bounded smoke evaluation target: {target!r}")
    return target


# The usual entrypoint selfcheck imports this module without its original argv.
# Import the actual target too, so doctor checks the method-specific runner.
if os.environ.get("MCITLIB_SELFCHECK") == "1":
    importlib.import_module(_target())


def main() -> None:
    import torch
    from transformers.generation.utils import GenerationMixin

    target = _target()
    limit = int(os.environ.get("MCITLIB_SMOKE_MAX_NEW_TOKENS", "16"))
    if not 1 <= limit <= 32:
        raise ValueError("bounded smoke generation limit must be 1..32 tokens")
    original_generate = GenerationMixin.generate

    def limited_generate(model, *args, **kwargs):
        requested = kwargs.get("max_new_tokens")
        kwargs["max_new_tokens"] = min(requested, limit) if requested is not None else limit
        checked = 0

        def check_logits(_module, _inputs, output):
            nonlocal checked
            logits = getattr(output, "logits", None)
            if not torch.is_tensor(logits):
                raise RuntimeError("bounded generation requires tensor output.logits")
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("bounded generation produced non-finite logits")
            checked += 1

        handle = model.register_forward_hook(check_logits)
        try:
            result = original_generate(model, *args, **kwargs)
            if not checked:
                raise RuntimeError("bounded generation did not produce model logits")
            print(f"[bounded] checked finite logits for {checked} generation step(s)", flush=True)
            return result
        finally:
            handle.remove()

    GenerationMixin.generate = limited_generate
    sys.argv[0] = target
    runpy.run_module(target, run_name="__main__")


if __name__ == "__main__":
    main()
