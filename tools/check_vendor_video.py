# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Offline acceptance checks for the video source recipes; needs the pinned v3 Git objects."""
import argparse
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import prepare_video_sources as sources


def run_tool(*args, fails=None):
    result = subprocess.run(
        [sys.executable, "-B", sources.__file__, *map(str, args)],
        text=True, capture_output=True,
    )
    if fails is None:
        result.check_returncode()
    elif result.returncode == 0 or fails not in result.stderr:
        raise RuntimeError(f"expected {fails!r}, got {result.returncode}: {result.stderr}")


def check_retries():
    spec, _ = sources.recipe("Video-LLaVA")
    with patch.object(sources.time, "sleep") as sleep:
        with patch.object(sources, "urlopen", side_effect=[URLError("unavailable"), BytesIO(b"ok")]):
            if sources.upstream_bytes(spec, "LICENSE", None) != b"ok":
                raise RuntimeError("download did not recover")
            sleep.assert_called_once_with(1)
        for error, attempts in [(HTTPError("url", 404, "missing", {}, None), 1),
                                (URLError("unavailable"), 3)]:
            with patch.object(sources, "urlopen", side_effect=error) as fetch:
                try:
                    sources.upstream_bytes(spec, "LICENSE", None)
                except URLError:
                    pass
                else:
                    raise RuntimeError("download error did not propagate")
                if fetch.call_count != attempts:
                    raise RuntimeError("unexpected number of acquisition attempts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-git", type=Path, required=True)
    repo = parser.parse_args().from_git.resolve()
    check_retries()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sources"
        run_tool("--output", out, "--from-git", Path(tmp) / "missing-repo", fails="git")
        if list(Path(tmp).iterdir()):
            raise RuntimeError("failed build left partial output")
        run_tool("--output", out, "--from-git", repo)
        run_tool("--output", out, "--check")
        # Repeating on a correct tree needs no download or even an accessible repo.
        run_tool("--output", out, "--from-git", Path(tmp) / "missing-repo")
        for file in out.rglob("*.py"):
            compile(file.read_bytes(), str(file), "exec")
        for extra in (out / ".git", out / "Video-LLaVA/videollava/__pycache__"):
            extra.mkdir()
            run_tool("--output", out, "--check", fails="unexpected entry")
            extra.rmdir()
        link = out / "VideoLLaMA2/other"
        link.symlink_to(out / "Video-LLaVA", target_is_directory=True)
        run_tool("--output", out, "--check", fails="symbolic link")
        link.unlink()
        license_path = out / "Video-LLaVA/LICENSE"
        license_path.write_bytes(b"altered")
        run_tool("--output", out, "--fetch", fails="SHA-256 mismatch")
        license_path.unlink()
        run_tool("--output", out, "--check", fails="missing file")
    print("Video source checks passed: build, repeat, retry, cleanup, syntax and tamper detection.")


if __name__ == "__main__":
    main()
