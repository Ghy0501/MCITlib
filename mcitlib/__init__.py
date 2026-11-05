# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""MCITlib: continual multimodal instruction tuning library."""

__version__ = "0.1.0"


def _prefer_vendored_packages() -> None:
    """Put third_party/{LLaVA,peft} ahead of anything installed in the environment.

    MCITlib is meant to run from a checkout (``pip install -e .``).  The vendored LLaVA and
    peft must win over a stale v3 editable install or a PyPI peft; otherwise v3 code could be
    imported silently.  Outside a checkout this is a no-op.
    """
    import os

    # Set only by the equivalence tests for their *v3 reference* subprocesses, which must import v3's
    # own `llava`/`peft` from PYTHONPATH.  Never set in a real run.
    if os.environ.get("MCITLIB_NO_VENDORED_PATH") == "1":
        return
    try:
        from .paths import bootstrap_sys_path

        bootstrap_sys_path()
    except RuntimeError:
        pass


_prefer_vendored_packages()
