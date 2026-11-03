# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""P0 analysis helper: measure v3 duplication and pairwise drift.

Read-only. Run from a checkout of the v3 tree (tag ``v3.0-final``).

    python tools/p0/inventory.py --root . --out /tmp/p0/inventory.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path

BACKBONES = {
    "LLaVA": "llava",
    "InternVL": "llava",
    "Video-LLaVA": "videollava",
    "VideoLLaMA2": "videollama2",
}
EXCLUDES = ["__pycache__", ".DS_Store", "._*", "*.pyc", "*.pkl"]
BASE_METHOD = "LoRA-FT"


def _diff_lines_N(a: Path, b: Path) -> tuple[int, int]:
    """(changed lines in files present on both sides, number of such files)."""
    cmd = ["diff", "-ru"]
    for e in EXCLUDES:
        cmd += ["-x", e]
    cmd += [str(a), str(b)]
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace").stdout
    lines = files = 0
    for ln in out.splitlines():
        if ln.startswith("diff -ru"):
            files += 1
        elif ln.startswith(("+++", "---")):
            continue
        elif ln.startswith(("+", "-")):
            lines += 1
    return lines, files


def _only_in(a: Path, b: Path) -> tuple[list[str], list[str]]:
    """Files only in a / only in b (relative paths)."""
    def files(d: Path) -> set[str]:
        res = set()
        for p in d.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store" \
                    and not p.name.startswith("._") and p.suffix not in {".pyc", ".pkl"}:
                res.add(str(p.relative_to(d)))
        return res
    fa, fb = files(a), files(b)
    return sorted(fa - fb), sorted(fb - fa)


def _loc(d: Path) -> int:
    n = 0
    if not d.exists():
        return 0
    for p in d.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        try:
            n += sum(1 for _ in p.open("rb"))
        except OSError:
            pass
    return n


def _changed_files(a: Path, b: Path) -> list[str]:
    cmd = ["diff", "-rq"]
    for e in EXCLUDES:
        cmd += ["-x", e]
    cmd += [str(a), str(b)]
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace").stdout
    res = []
    for ln in out.splitlines():
        if ln.startswith("Files "):
            parts = ln.split(" and ")
            res.append(parts[1].rsplit(" differ", 1)[0].replace(str(b) + "/", ""))
    return sorted(res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=".",
        help="MCITlib 仓库根目录（包含 LLaVA/、InternVL/ 等骨干目录），不是骨干目录",
    )
    ap.add_argument("--out", default="/tmp/p0/inventory.json")
    ap.add_argument("--pairwise", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    result: dict = {}
    for bb, pkg in BACKBONES.items():
        bdir = root / bb
        methods = sorted(p.name for p in bdir.iterdir() if p.is_dir())
        base = bdir / BASE_METHOD
        info = {}
        for m in methods:
            mdir = bdir / m
            tops = sorted(p.name for p in mdir.iterdir())
            vendored = [
                t for t in tops
                if (mdir / t).is_dir() and t not in {pkg, "scripts"}
            ]
            e = {
                "top": tops,
                "pkg_loc": _loc(mdir / pkg),
                "vendored": {v: _loc(mdir / v) for v in vendored},
                "license": (mdir / "LICENSE").exists(),
            }
            if m != BASE_METHOD:
                lines, files = _diff_lines_N(base / pkg, mdir / pkg)
                e["pkg_diff_vs_base"] = {"lines": lines, "files": files}
                miss, extra = _only_in(base / pkg, mdir / pkg)
                e["only_in_base"] = miss
                e["only_in_method"] = extra
                e["changed_files_vs_base"] = _changed_files(base / pkg, mdir / pkg)
            info[m] = e
        if args.pairwise:
            mat = {}
            for a, b in itertools.combinations(methods, 2):
                mat[f"{a}|{b}"] = _diff_lines_N(bdir / a / pkg, bdir / b / pkg)[0]
            info["_pairwise"] = mat
        result[bb] = info
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
