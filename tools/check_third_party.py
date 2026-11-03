# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Check vendored upstream metadata and patch files.

L1 accepts a not-yet-vendored tree. L2 extends the same command with byte-level
upstream-plus-patches verification for each component.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="MCITlib repository root")
    args = parser.parse_args()
    root = args.root.resolve() / "third_party"
    if not root.exists():
        print("third_party/: not vendored yet (L1 baseline)")
        return 0
    errors: list[str] = []
    for component in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (component / "UPSTREAM.md").is_file():
            errors.append(f"missing UPSTREAM.md: {component}")
        patches = component / "patches"
        if patches.exists() and not patches.is_dir():
            errors.append(f"patches is not a directory: {patches}")
    for error in errors:
        print(error)
    print(f"third-party metadata errors: {len(errors)}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
