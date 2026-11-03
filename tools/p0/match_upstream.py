# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""P0 analysis helper: find the upstream commit closest to each v3 method tree.

Read-only. For every probe file (``.py`` files that exist in the v3 package of
at least ``--min-methods`` method directories *and* somewhere in the upstream
history), the file at each candidate upstream commit is diffed against the v3
copy; the per-commit score is the sum of changed lines over probe files.

    python tools/p0/match_upstream.py \
        --v3-root . --backbone LLaVA --pkg llava \
        --upstream /tmp/p0/up/LLaVA --upstream-prefix "" \
        --out /tmp/p0/match_LLaVA.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


def git(repo: Path, *a: str, check: bool = True) -> bytes:
    r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True)
    if check and r.returncode:
        raise RuntimeError(r.stderr.decode(errors="replace"))
    return r.stdout


def diff_count(a: bytes, b: bytes) -> int:
    with tempfile.NamedTemporaryFile() as fa, tempfile.NamedTemporaryFile() as fb:
        fa.write(a)
        fa.flush()
        fb.write(b)
        fb.flush()
        out = subprocess.run(["diff", fa.name, fb.name], capture_output=True).stdout
    return sum(1 for ln in out.splitlines() if ln[:2] in (b"< ", b"> "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-root", default=".")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--upstream-prefix", default="")
    ap.add_argument("--min-methods", type=int, default=8)
    ap.add_argument("--max-probe", type=int, default=40)
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = Path(a.v3_root).resolve() / a.backbone
    up = Path(a.upstream)
    methods = sorted(p.name for p in root.iterdir() if p.is_dir())
    pkgdir = lambda m: root / m / a.pkg  # noqa: E731

    # probe files: common to many methods and present at some upstream commit
    cnt: dict[str, int] = defaultdict(int)
    for m in methods:
        for p in pkgdir(m).rglob("*.py"):
            if "__pycache__" in p.parts or "eval" in p.relative_to(pkgdir(m)).parts:
                continue
            cnt[str(p.relative_to(pkgdir(m)))] += 1
    cands = [f for f, c in cnt.items() if c >= a.min_methods]
    # keep files that are present in upstream history
    known = set()
    for f in cands:
        path = f"{a.upstream_prefix}{a.pkg}/{f}"
        if git(up, "log", "--format=%H", "-1", a.rev, "--", path):
            known.add(f)
    # prefer files that differ among methods (informative), then by size
    def informative(f: str) -> int:
        blobs = {pkgdir(m).joinpath(f).read_bytes() for m in methods if pkgdir(m).joinpath(f).exists()}
        return len(blobs)
    probes = sorted(known, key=lambda f: (-informative(f), f))[: a.max_probe]
    print(f"{len(probes)} probe files")

    paths = [f"{a.upstream_prefix}{a.pkg}/{f}" for f in probes]
    commits = git(up, "log", "--format=%H", a.rev, "--", *paths).decode().split()
    print(f"{len(commits)} candidate commits")

    # blob sha per (commit, probe)
    blob_at: dict[str, dict[str, str | None]] = {}
    for c in commits:
        out = git(up, "ls-tree", "-r", c, "--", *paths).decode()
        d: dict[str, str | None] = {p: None for p in probes}
        for ln in out.splitlines():
            meta, path = ln.split("\t", 1)
            d[path[len(a.upstream_prefix) + len(a.pkg) + 1:]] = meta.split()[2]
        blob_at[c] = d

    cache_blob: dict[str, bytes] = {}
    def blob(sha: str) -> bytes:
        if sha not in cache_blob:
            cache_blob[sha] = git(up, "cat-file", "blob", sha)
        return cache_blob[sha]

    score_cache: dict[tuple[str, str, str], int] = {}
    def score(m: str, f: str, sha: str | None) -> int:
        mine = pkgdir(m) / f
        if not mine.exists():
            return 0
        if sha is None:
            return len(mine.read_bytes().splitlines())
        k = (m, f, sha)
        if k not in score_cache:
            score_cache[k] = diff_count(blob(sha), mine.read_bytes())
        return score_cache[k]

    result: dict = {"probes": probes, "methods": {}}
    for m in methods:
        best = []
        for c in commits:
            tot = sum(score(m, f, blob_at[c][f]) for f in probes)
            best.append((tot, c))
        best.sort(key=lambda x: x[0])
        top = best[:5]
        per_file = {f: score(m, f, blob_at[top[0][1]][f]) for f in probes}
        dates = {c: git(up, "log", "-1", "--format=%ad %s", "--date=short", c).decode().strip() for _, c in top}
        result["methods"][m] = {
            "top": [(t, c, dates[c]) for t, c in top],
            "per_file_at_best": {k: v for k, v in per_file.items() if v},
        }
        print(f"{m:12} best={top[0][0]:6} {top[0][1][:10]} {dates[top[0][1]][:60]}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
