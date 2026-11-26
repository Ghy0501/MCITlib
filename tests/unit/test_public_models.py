# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tools.public_models import MODEL_GROUPS, assets, install


def test_default_is_offline_and_creates_nothing(tmp_path):
    script = Path(__file__).resolve().parents[2] / "tools/public_models.py"
    root = tmp_path / "no downloads"
    proc = subprocess.run([sys.executable, "-I", "-S", str(script), "llava", "--root", str(root)],
                          check=True, capture_output=True, text=True)
    assert len(proc.stdout.splitlines()) == 2 and "huggingface.co/liuhaotian/" in proc.stdout
    assert not root.exists()


def test_plan_pins_models_and_deduplicates_shared_towers():
    rows = assets(["llava", "internvl", "videollava", "videollama2", "judge"])
    assert len(rows) == 9
    assert all(re.fullmatch(r"[0-9a-f]{40}", rev) for _, rev in rows)
    assert set(MODEL_GROUPS) == {"llava", "internvl", "videollava", "videollama2", "judge"}
    assert not any("Qwen/" in name for name, _ in assets(["videollava"]))


def test_install_links_exact_snapshot_without_overwriting_user_data(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    snapshot = root / ".hub-cache" / "models--org--m" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    install(root, "org/m", "abc", fetch)
    assert (root / "m").resolve() == snapshot
    assert not (root / "m").readlink().is_absolute()
    assert calls[0]["revision"] == "abc"
    install(root, "org/m", "abc", fetch)
    (root / "m").unlink()
    (root / "m").mkdir()
    with pytest.raises(FileExistsError):
        install(root, "org/m", "abc", fetch)
    assert (root / "m").is_dir() and not (root / "m").is_symlink()
