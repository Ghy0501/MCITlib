# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Keep generated physical DeepSpeed slots distinct from inherited CUDA masks."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcitlib.config.schema import Command
from mcitlib.runner import build_sequence_commands, execute_commands


@pytest.fixture
def fake_launcher(tmp_path: Path) -> Path:
    package = tmp_path / "fake" / "deepspeed" / "launcher"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "runner.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps({'visible': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'sentinel': os.environ.get('LAUNCHER_TEST_SENTINEL'), 'args': sys.argv[1:]}))\n"
    )
    return package.parent.parent


def child_record(command: Command, tmp_path: Path, fake_launcher: Path, name: str) -> dict:
    command.env["PYTHONPATH"] = str(fake_launcher)
    before = list(command.argv)
    root = tmp_path / name
    root.mkdir()
    execute_commands([command], root / "logs" / "run", root)
    marker = next((root / "logs" / "run" / ".mcitlib" / "steps").glob("*.json"))
    saved = json.loads(marker.read_text())
    assert command.argv == before == saved["command"]["argv"]
    return json.loads(Path(saved["log"]).read_text())


@pytest.mark.parametrize("method,gpus,inherited", [("dgg", "4", "4"),
                                                   ("dgg", "4", "4,5"),
                                                   ("pclr", "4,5", "4,5")])
def test_generated_physical_launcher_discards_only_inherited_mask(
    repo_root: Path, tmp_path: Path, fake_launcher: Path, monkeypatch: pytest.MonkeyPatch,
    method: str, gpus: str, inherited: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", inherited)
    monkeypatch.setenv("LAUNCHER_TEST_SENTINEL", "preserved")
    commands = build_sequence_commands(repo_root / "configs", "llava", method, "ucit", task_id=1, gpus=gpus)
    train = next(command for command in commands if command.kind == "train")
    assert train.argv[:3] == ["deepspeed", "--include", f"localhost:{gpus}"]
    assert "CUDA_VISIBLE_DEVICES" not in train.env
    record = child_record(train, tmp_path, fake_launcher, "physical")
    assert record["visible"] is None
    assert record["sentinel"] == "preserved"
    assert record["args"][:2] == ["--include", f"localhost:{gpus}"]


@pytest.mark.parametrize("argv,explicit", [
    (["deepspeed", "--module", "training", "--include", "localhost:4"], None),
    (["deepspeed", "--include", "remote:4", "--module", "training"], None),
    (["deepspeed", "--include", "localhost:4@remote:5", "--module", "training"], None),
    (["deepspeed", "--include", "localhost:4,4", "--module", "training"], None),
    (["deepspeed", "--include", "localhost:4", "--include", "localhost:5", "--module", "training"], None),
    (["deepspeed", "--include", "localhost:4", "--module", "training"], "7"),
    (["deepspeed", "--include", "localhost:0,1,2,3", "--module", "training"], "0,1,2,3"),
])
def test_non_generated_or_explicit_mask_is_preserved(
    tmp_path: Path, fake_launcher: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str], explicit: str | None,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    command = Command(argv, env={} if explicit is None else {"CUDA_VISIBLE_DEVICES": explicit})
    record = child_record(command, tmp_path, fake_launcher, "preserve")
    assert record["visible"] == ("4,5" if explicit is None else explicit)


def test_python_inference_retains_its_device_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    script = "import json, os; print(json.dumps({'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}))"
    command = Command(["python", "-c", script], kind="eval", env={"CUDA_VISIBLE_DEVICES": "4"})
    execute_commands([command], tmp_path / "logs/run", tmp_path)
    marker = next((tmp_path / "logs/run/.mcitlib/steps").glob("*.json"))
    record = json.loads(marker.read_text())
    assert json.loads(Path(record["log"]).read_text())["visible"] == "4"
    assert command.env["CUDA_VISIBLE_DEVICES"] == "4"


def test_generated_internvl_profile_keeps_explicit_four_gpu_mask(
    repo_root: Path, tmp_path: Path, fake_launcher: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    commands = build_sequence_commands(repo_root / "configs", "internvl", "lora_ft", "ucit", task_id=1,
                                       execution_profile="four_gpu_validation", gpus="0,1,2,3")
    train = next(command for command in commands if command.kind == "train")
    assert train.argv[:3] == ["deepspeed", "--include", "localhost:0,1,2,3"]
    assert train.env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    record = child_record(train, tmp_path, fake_launcher, "internvl")
    assert record["visible"] == "0,1,2,3"
    chunks = [item for item in commands if "--chunk-idx" in item.argv]
    assert [item.env["CUDA_VISIBLE_DEVICES"] for item in chunks] == ["0", "1", "2", "3"]


def test_deepspeed_095_actual_startup_filter_handles_physical_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    # CUDA may have cached the number of visible devices before DeepSpeed's
    # main() deletes the environment variable. Simulate that cached count while
    # exercising the installed launcher's real argument/resource parsing.
    ds = pytest.importorskip("deepspeed.launcher.runner") if not os.environ.get("MCITLIB_REQUIRE_DEEPSPEED") \
        else __import__("deepspeed.launcher.runner", fromlist=["main"])
    original = ds.parse_inclusion_exclusion

    class Parsed(Exception):
        def __init__(self, resources):
            self.resources = resources

    def run(visible: str | None):
        count = len(visible.split(",")) if visible else 8
        with monkeypatch.context() as patch:
            if visible is None:
                patch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
            else:
                patch.setenv("CUDA_VISIBLE_DEVICES", visible)
            patch.setattr(ds, "fetch_hostfile", lambda _: None)
            patch.setattr(ds, "get_accelerator", lambda: SimpleNamespace(device_count=lambda: count))

            def stop_after_parse(*args):
                raise Parsed(original(*args))

            patch.setattr(ds, "parse_inclusion_exclusion", stop_after_parse)
            ds.main(["--include", "localhost:4", "--module", "example"])

    with pytest.raises(ValueError, match="No slot '4' specified on host 'localhost'"):
        run("4,5")
    with pytest.raises(Parsed) as parsed:
        run(None)
    assert parsed.value.resources == {"localhost": [4]}
