# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""NB-4: bounded replacement for v3's shell checkpoint cleanup."""
from __future__ import annotations

import shutil
from pathlib import Path


def remove_checkpoint_dirs(output_dir: str | Path) -> list[Path]:
    """Remove only direct, non-symlink ``checkpoint-*`` children of output_dir."""
    root = Path(output_dir).expanduser().resolve()
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError(f"training output_dir is not a directory: {root}")
    removed: list[Path] = []
    for child in root.iterdir():
        if child.is_symlink() or not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        resolved = child.resolve()
        if resolved.parent != root:
            raise ValueError(f"refusing to remove path outside output_dir: {resolved}")
        shutil.rmtree(resolved)
        removed.append(resolved)
    return removed
