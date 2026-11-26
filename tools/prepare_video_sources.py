# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Prepare versioned video runtimes outside MCITlib. Network use requires --fetch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("Video-LLaVA", "VideoLLaMA2")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require_digest(data, expected, name):
    if digest(data) != expected:
        raise ValueError(f"SHA-256 mismatch: {name}")


def recipe(component):
    root = ROOT / "third_party" / component
    metadata = json.loads((root / "MANIFEST.json").read_bytes())
    for name in ("SOURCES.json", "runtime.diff"):
        require_digest((root / name).read_bytes(), metadata["files"][name], f"{component}/{name}")
    spec = json.loads((root / "SOURCES.json").read_bytes())
    if spec["commit"] != metadata["upstream_commit"]:
        raise ValueError(f"inconsistent revision: {component}")
    return spec, root / "runtime.diff"


def upstream_bytes(spec, path, repo):
    if repo is not None:
        # A partial clone must not silently contact its promisor remote.
        return subprocess.check_output(
            ["git", "-C", str(repo), "cat-file", "blob", f"{spec['commit']}:{path}"],
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_ALLOW_PROTOCOL": ""},
        )
    url = f"https://raw.githubusercontent.com/{spec['repository']}/{spec['commit']}/{path}"
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as stream:
                return stream.read()
        except HTTPError as error:
            if attempt == 2 or (error.code not in (408, 429) and error.code < 500):
                raise
        except (URLError, TimeoutError, ConnectionError, IncompleteRead):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def prepare(root, repo):
    for component in COMPONENTS:
        spec, patch = recipe(component)
        target = root / component
        for dest, entry in spec["files"].items():
            data = upstream_bytes(spec, entry["path"], repo)
            require_digest(data, entry["sha256"], entry["path"])
            path = target / dest
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        # Zero context avoids shipping unmodified third-party lines in the recipe.
        # Exact pre- and post-image digests bind each hunk to its input.
        subprocess.run(["git", "apply", "--no-index", "--unidiff-zero", "--whitespace=nowarn", str(patch)],
                       cwd=target, check=True)


def verify(root):
    expected = {}
    for component in COMPONENTS:
        spec, _ = recipe(component)
        for dest, entry in spec["files"].items():
            expected[Path(component) / dest] = entry.get("prepared_sha256", entry["sha256"])
    directories = {parent for p in expected for parent in p.parents}
    seen = set()
    for path in root.rglob("*"):
        name = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"unexpected symbolic link: {name}")
        if name in directories and path.is_dir():
            continue
        if name not in expected or not path.is_file():
            raise ValueError(f"unexpected entry: {name}")
        require_digest(path.read_bytes(), expected[name], name)
        seen.add(name)
    if missing := expected.keys() - seen:
        raise ValueError(f"missing file: {min(missing)}")
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="external source root")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify an existing output, offline")
    mode.add_argument("--from-git", type=Path, help="read from local MCITlib v3 Git objects, offline")
    mode.add_argument("--fetch", action="store_true", help="download only the pinned, allowlisted files")
    args = parser.parse_args()
    try:
        if args.output.is_symlink():
            raise ValueError("output must not be a symbolic link")
        output = args.output.resolve()
        if ROOT == output or ROOT in output.parents:
            raise ValueError("use an external output directory, outside the release checkout")
        if args.check or output.exists():
            count = verify(output)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=output.parent, prefix=".mcit-video-") as tmp:
                stage = Path(tmp) / "sources"
                prepare(stage, args.from_git)
                count = verify(stage)
                stage.rename(output)
        print(f"Verified {count} files in {output}")
    except (OSError, ValueError, IncompleteRead, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"video sources: {exc}\n")


if __name__ == "__main__":
    main()
