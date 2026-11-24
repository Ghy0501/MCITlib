# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU-only checks for shared configuration, task differences and command records."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mcitlib.config.cli import expand_task, validate_config
from mcitlib.config.loader import read_yaml
from mcitlib.registry import BENCHMARKS, METHODS
from tests.unit.test_config_migration import normalise


def option(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def temporary_config(repo_root: Path, tmp_path: Path) -> Path:
    target = tmp_path / "configs"
    for name in ("common", "methods", "benchmarks", "backbones"):
        shutil.copytree(repo_root / "configs" / name, target / name)
    read_yaml.cache_clear()
    return target


def save(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    read_yaml.cache_clear()


def test_default_and_task_edits_reach_ordered_commands(repo_root: Path, tmp_path: Path) -> None:
    root = temporary_config(repo_root, tmp_path)
    path = root / "methods/lora_ft.yaml"
    method = yaml.safe_load(path.read_text())
    ucit = method["overrides"]["llava"]["ucit"]
    ucit["task_defaults"]["train_config"].update({"rank": 24, "lr": 0.0011, "batch_size": 8})
    ucit["task_2"]["train_config"]["rank"] = 7
    ucit["task_1"]["train_config"]["output_dir"] = "${ckpt_root}/training-custom"
    ucit["task_1"]["eval_config"]["model_path"] = "${ckpt_root}/evaluation-custom"
    save(path, method)
    data_path = root / "common/tasks.yaml"
    datasets = yaml.safe_load(data_path.read_text())
    first = datasets["benchmarks"]["ucit"]["tasks"]["task_1"]["data_config"]
    first.update({"train_path": "${data_root}/custom/train.json", "test_path": "${data_root}/custom/test.json",
                  "test_folder": "${data_root}/custom/images"})
    save(data_path, datasets)
    library_path = root / "common/commands.yaml"
    library = yaml.safe_load(library_path.read_text())
    options = library["fragments"]["options_num_train_epochs_to_report_to"]
    options[options.index("--weight_decay") + 1] = "0.12"
    save(library_path, library)

    one = expand_task(root, "lora_ft", "ucit", 1)
    two = expand_task(root, "lora_ft", "ucit", 2)
    train = one["train_command"]["argv"]
    assert option(train, "--lora_r") == "24"
    assert option(two["train_command"]["argv"], "--lora_r") == "7"
    assert option(train, "--learning_rate") == "0.0011"
    assert option(train, "--per_device_train_batch_size") == "8"
    assert option(train, "--weight_decay") == "0.12"
    assert option(two["train_command"]["argv"], "--weight_decay") == "0.12"
    assert option(train, "--data_path") == "${data_root}/custom/train.json"
    assert option(train, "--output_dir") == "${ckpt_root}/training-custom"
    inference = one["eval_commands"][0]["argv"]
    assert option(inference, "--model-path") == "${ckpt_root}/evaluation-custom"
    assert option(inference, "--question-file") == "${data_root}/custom/test.json"
    assert option(inference, "--image-folder") == "${data_root}/custom/images"
    assert one["train_command"]["args"]["learning_rate"] == "0.0011"


def test_invalid_template_override_and_reference_are_rejected(repo_root: Path, tmp_path: Path) -> None:
    root = temporary_config(repo_root, tmp_path)
    path = root / "methods/lora_ft.yaml"
    method = yaml.safe_load(path.read_text())
    train = method["overrides"]["llava"]["ucit"]["task_1"]["train_command"]
    train.setdefault("set", {})["--not-an-option"] = "x"
    save(path, method)
    with pytest.raises(ValueError, match="unknown option"):
        read_yaml(path)
    del train["set"]["--not-an-option"]
    train["set"]["--lora_r"] = "@{task.train_config.missing}"
    save(path, method)
    with pytest.raises(ValueError, match="unknown configuration reference"):
        read_yaml(path)


@pytest.mark.parametrize("broken", ["--not-a-list", ["--flag", 1]])
def test_invalid_fragment_is_rejected(repo_root: Path, tmp_path: Path, broken: object,
                                      capsys: pytest.CaptureFixture) -> None:
    from mcitlib.config.cli import main

    root = temporary_config(repo_root, tmp_path)
    path = root / "common/commands.yaml"
    library = yaml.safe_load(path.read_text())
    name = "options_num_train_epochs_to_report_to"
    library["fragments"][name] = broken
    save(path, library)
    with pytest.raises(ValueError, match=f"command fragment {name}.*non-empty string list"):
        validate_config(root)
    assert main(["validate", "--config-root", str(root)]) == 2
    error = capsys.readouterr().err
    assert name in error
    assert "Traceback" not in error


def test_template_argv_and_fragment_reference_types_are_validated(repo_root: Path, tmp_path: Path) -> None:
    root = temporary_config(repo_root, tmp_path)
    path = root / "common/commands.yaml"
    library = yaml.safe_load(path.read_text())
    name = "train_train_mem_3"
    argv = library["templates"][name]["argv"]
    library["templates"][name]["argv"] = "not-a-list"
    save(path, library)
    with pytest.raises(ValueError, match=f"command template {name} argv must be a list"):
        validate_config(root)
    library["templates"][name]["argv"] = argv
    fragment = next(item for item in argv if isinstance(item, dict) and "use" in item)
    fragment["use"] = 42
    save(path, library)
    with pytest.raises(ValueError, match=f"unknown command fragment 42 in template {name}"):
        validate_config(root)


def test_all_tasks_validate_without_model_imports(repo_root: Path) -> None:
    result = validate_config(repo_root / "configs")
    assert result == {"methods": 17, "benchmarks": 3, "tasks": 250, "recorded_commands": 2549}


def test_config_validation_needs_no_model_packages(repo_root: Path) -> None:
    code = """
import sys
class RejectModelPackages:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'PIL', 'transformers', 'deepspeed', 'llava', 'peft'}:
            raise AssertionError('unexpected model dependency: ' + fullname)
sys.meta_path.insert(0, RejectModelPackages())
from mcitlib.config.cli import validate_config
assert validate_config(sys.argv[1])['tasks'] == 250
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo_root)}
    subprocess.run([sys.executable, "-c", code, str(repo_root / "configs")], cwd=repo_root,
                   env=env, check=True, capture_output=True, text=True)


def test_config_cli_validate_expand_and_errors(repo_root: Path, capsys: pytest.CaptureFixture) -> None:
    from mcitlib.cli import main

    root = str(repo_root / "configs")
    assert main(["config", "validate", "--config-root", root, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["recorded_commands"] == 2549
    assert main(["config", "expand", "--config-root", root, "--method", "modalprompt",
                 "--benchmark", "ucit", "--task", "2"]) == 0
    task = json.loads(capsys.readouterr().out)
    assert option(task["train_command"]["argv"], "--output_dir").endswith("/Task1_llava")
    assert task["eval_config"]["model_path"].endswith("/Task2_llava")
    assert main(["config", "expand", "--config-root", root, "--method", "lora_ft",
                 "--benchmark", "ucit", "--task", "999"]) == 2
    assert "task must be between" in capsys.readouterr().err


def test_config_cli_relative_root_uses_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture) -> None:
    from mcitlib.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["config", "validate", "--config-root", "configs", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tasks"] == 250


def test_config_cli_reports_malformed_yaml(repo_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from mcitlib.cli import main

    root = temporary_config(repo_root, tmp_path)
    (root / "methods/lora_ft.yaml").write_text("invalid: [\n", encoding="utf-8")
    assert main(["config", "validate", "--config-root", str(root)]) == 2
    assert "mcitlib config:" in capsys.readouterr().err


def test_recorded_argv_environment_and_order_match_fixtures(repo_root: Path) -> None:
    fixture_root = repo_root / "tests/fixtures/v3_commands/llava"
    recorded_count = 0
    for method in METHODS:
        if method.id in {"pclr", "same", "sigma", "dgg", "octopus"}:
            # These new methods have no historical v3 fixture; their plans have dedicated tests.
            continue
        values = read_yaml(repo_root / "configs/methods" / f"{method.id}.yaml")
        for benchmark in BENCHMARKS:
            for router in (False, True) if method.id == "mrlora" else (False,):
                suffix = "_router" if router else ""
                payload = json.loads((fixture_root / method.id / f"{benchmark.id}{suffix}.json").read_text())
                tasks = (values["router"]["benchmarks"][benchmark.id]["tasks"] if router else
                         values["overrides"]["llava"][benchmark.id])
                for index in range(1, len(benchmark.tasks) + 1):
                    actual = tasks[f"task_{index}"]
                    commands = [item for item in payload["commands"] if item["task_id"] == index]
                    for kind, field in (("train", "train_command"), ("eval", "eval_commands"),
                                        ("extra", "extra_commands"), ("other", "other_commands")):
                        expected = [item for item in commands if item["kind"] == kind]
                        records = [actual[field]] if kind == "train" else actual.get(field, [])
                        assert len(records) == len(expected), (method.id, benchmark.id, index, kind)
                        for record, item in zip(records, expected):
                            assert record["argv"] == normalise(item["argv"]), (method.id, benchmark.id, index, kind)
                            assert record["order"] == item["order"]
                            # Some legacy method YAML intentionally did not retain
                            # the recorder's CUDA env; check every retained value.
                            if "env" in record:
                                assert record["env"] == {k: v for k, v in item.get("env", {}).items()
                                                         if k == "CUDA_VISIBLE_DEVICES"}
                            recorded_count += 1
                if not router:
                    expected_pre = [item for item in payload["commands"] if item["task_id"] == 0]
                    actual_pre = tasks.get("pre_commands", [])
                    assert [item["argv"] for item in actual_pre] == [normalise(item["argv"]) for item in expected_pre]
                    assert [item["order"] for item in actual_pre] == [item["order"] for item in expected_pre]
                    recorded_count += len(actual_pre)
    assert recorded_count == 1954
