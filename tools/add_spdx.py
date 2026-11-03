# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Add or check the MCITlib SPDX header on owned Python source files."""
from __future__ import annotations

import argparse
from pathlib import Path

HEADER = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# Copyright (c) 2025-2026 MCITlib authors\n"
)


def owned_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory in ("mcitlib", "tools", "tests"):
        base = root / directory
        if base.exists():
            files.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(files)


def has_header(text: str) -> bool:
    return "SPDX-License-Identifier:" in "\n".join(text.splitlines()[:8])


def add_header(text: str) -> str:
    if has_header(text):
        return text
    lines = text.splitlines(keepends=True)
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1
    if index < len(lines) and "coding" in lines[index] and lines[index].startswith("#"):
        index += 1
    return "".join(lines[:index]) + HEADER + "".join(lines[index:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="MCITlib repository root")
    parser.add_argument("--check", action="store_true", help="only report files missing the header")
    args = parser.parse_args()
    missing = [p for p in owned_files(args.root.resolve()) if not has_header(p.read_text(encoding="utf-8"))]
    if args.check:
        for path in missing:
            print(path)
        print(f"SPDX missing: {len(missing)}")
        return int(bool(missing))
    for path in missing:
        path.write_text(add_header(path.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"updated {path}")
    print(f"SPDX updated: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
