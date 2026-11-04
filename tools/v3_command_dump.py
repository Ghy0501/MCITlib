# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Record v3 LLaVA shell commands without executing training or installation.

The v3 tree is never modified. For each sequence, its scripts and configs are
copied to a temporary directory, ``/your_path/MCITlib_v3`` is rewritten there,
and shims record deepspeed/python/pip invocations. ``python -c`` is delegated to
the real interpreter because v3 uses it to read JSON configuration values.

Example:
    MCITLIB_V3_ROOT=../MCITlib_v3_ref python tools/v3_command_dump.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from mcitlib.registry import BENCHMARKS, METHODS, get_benchmark, get_method

ROOT_PLACEHOLDER = "__V3_ROOT__"
V3_PLACEHOLDER = "/your_path/MCITlib_v3"
HARD_CODED_PATH_PLACEHOLDERS = {
    "/home/hongbo_zhao/ghy-cl-codebase/clip-vit-large-patch14-336": "/your_model_path/clip-vit-large-patch14-336",
    "/home/hongbo_zhao/ghy-cl-codebase/MCITlib/checkpoints": "/your_ckpts_path",
    "/home/hongbo_zhao/ghy-cl-codebase/MCITlib": "/your_path/MCITlib_v3",
    "/mnt/cache/guohaiyang/miniconda3/envs/coin/bin/python": "/your_python_path",
}

SHIM = r'''#!/usr/bin/env python3
import json
import os
import sys

name = os.path.basename(sys.argv[0])
real = os.environ["MCITLIB_REAL_PYTHON"]
if name in {"python", "python3"} and "-c" in sys.argv[1:]:
    os.execv(real, [real] + sys.argv[1:])

record = {
    "argv": [name] + sys.argv[1:],
    "cwd": os.getcwd(),
    "env": {
        key: os.environ[key]
        for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "HF_HOME", "TRANSFORMERS_CACHE")
        if key in os.environ
    },
}
log = os.environ["MCITLIB_DUMP_LOG"]
line = (json.dumps(record, ensure_ascii=False) + "\n").encode()
try:
    import fcntl
    with open(log, "ab") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(line)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
except ImportError:
    with open(log, "ab") as handle:
        handle.write(line)

# Let shell concatenation steps proceed without running a model. Only relative
# answer files inside the temporary working directory are created.
if name in {"python", "python3"}:
    args = sys.argv[1:]
    for index, arg in enumerate(args[:-1]):
        if arg == "--answers-file":
            output = args[index + 1]
            if output and not os.path.isabs(output):
                parent = os.path.dirname(output)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                open(output, "ab").close()
'''


def _write_shims(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("deepspeed", "pip", "python", "python3"):
        path = directory / name
        # An env-based shebang would resolve python3 back to this shim.
        path.write_text(SHIM.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _copy_tree_with_root(source: Path, destination: Path, temp_root: Path) -> None:
    shutil.copytree(source, destination)
    for path in destination.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rewritten = text.replace(V3_PLACEHOLDER, str(temp_root))
        for source, target in HARD_CODED_PATH_PLACEHOLDERS.items():
            rewritten = rewritten.replace(source, target)
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")


def _repair_commented_tasks(script: Path) -> list[int]:
    """Restore author-approved commented task blocks in the temporary script."""
    lines = script.read_text(encoding="utf-8").splitlines(keepends=True)
    targets = [
        index for index, line in enumerate(lines)
        if "train/task" in line and ".json" in line and line.lstrip().startswith("#")
    ]
    repaired: set[int] = set()
    for target in targets:
        start = target
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        end = target
        while end + 1 < len(lines) and lines[end + 1].lstrip().startswith("#"):
            end += 1
        for index in range(start, end + 1):
            lines[index] = re.sub(r"^(\s*)#\s?", r"\1", lines[index])
        match = re.search(r"train/task(\d+)\.json", lines[target])
        if match:
            repaired.add(int(match.group(1)))
    if repaired:
        script.write_text("".join(lines), encoding="utf-8")
    return sorted(repaired)


def _apply_config_aliases(scripts_root: Path, method_id: str) -> dict[str, str]:
    """Apply author-approved config aliases in the temporary copy only."""
    if method_id != "sefe":
        return {}
    source = "configs/train_configs/SEFE"
    target = "configs/train_configs/RegLoRA"
    for path in scripts_root.rglob("*.sh"):
        text = path.read_text(encoding="utf-8")
        rewritten = text.replace(source, target)
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")
    return {source: target}


def _apply_router_compat(scripts_root: Path, configs_root: Path, benchmark_id: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Fill explicitly authorized router placeholders in the temporary copy."""
    benchmark = get_benchmark(benchmark_id)
    script_aliases = {
        "scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh": "scripts/MCITlib/Eval_UCIT/Eval_ucit.sh",
        "scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh": "scripts/MCITlib/Eval_MLLM_ACL/Eval_acl.sh",
    }
    for path in scripts_root.rglob("*.sh"):
        text = path.read_text(encoding="utf-8")
        rewritten = text
        for source, target in script_aliases.items():
            rewritten = rewritten.replace(source, target)
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")

    placeholders: dict[str, Any] = {}
    data_root = configs_root / "data_configs" / benchmark.v3_name
    for path in sorted(data_root.glob("*.json")):
        values = json.loads(path.read_text(encoding="utf-8"))
        added: dict[str, str] = {}
        if "router_path" not in values:
            # Router data is deliberately independent of train_path. This is a
            # replaceable placeholder following the other v3 router configs.
            values["router_path"] = values["train_path"].rsplit("/", 1)[0] + "/router.json"
            added["router_path"] = values["router_path"]
        if "router_folder" not in values:
            values["router_folder"] = (
                values["train_folder"]
                if benchmark_id == "ucit"
                else values["train_folder"].rsplit("/", 1)[0]
            )
            added["router_folder"] = values["router_folder"]
        if added:
            path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
            placeholders[str(path.relative_to(configs_root))] = added
    return script_aliases, placeholders


def _apply_replay_placeholders(configs_root: Path, benchmark_id: str) -> dict[str, Any]:
    """Give Replay its own data JSON and the router-generated image root."""
    benchmark = get_benchmark(benchmark_id)
    placeholders: dict[str, Any] = {}
    data_root = configs_root / "data_configs" / benchmark.v3_name
    for path in sorted(data_root.glob("*.json")):
        values = json.loads(path.read_text(encoding="utf-8"))
        added: dict[str, str] = {}
        if "replay_path" not in values:
            values["replay_path"] = values["train_path"].rsplit("/", 1)[0] + "/replay.json"
            added["replay_path"] = values["replay_path"]
        if "router_folder" not in values:
            values["router_folder"] = (
                values["train_folder"] if benchmark_id == "ucit" else values["train_folder"].rsplit("/", 1)[0]
            )
            added["router_folder"] = values["router_folder"]
        if added:
            path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
            placeholders[str(path.relative_to(configs_root))] = added
    return placeholders


def _classify(argv: list[str]) -> str:
    executable = argv[0] if argv else ""
    text = " ".join(argv)
    if executable == "pip":
        return "install"
    if executable == "deepspeed":
        return "train"
    if any(token in text for token in ("extract_gradients", "extract_weights", "merge_lora_weights", "key_elements_identification")):
        return "extra"
    if ".eval." in text or "eval_" in text or "convert_result" in text:
        return "eval"
    return "other"


def _normalise(record: dict[str, Any], temp_root: Path, sequence: str, order: int, task_id: int) -> dict[str, Any]:
    root = str(temp_root)

    def clean(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(root, ROOT_PLACEHOLDER)
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return value

    argv = clean(record["argv"])
    return {
        "order": order,
        "argv": argv,
        "cwd": clean(record["cwd"]),
        "env": clean(record.get("env", {})),
        "kind": _classify(argv),
        "task_id": task_id,
        "sequence": sequence,
    }


def _read_commands(log: Path, temp_root: Path, sequence: str, method_id: str = "") -> list[dict[str, Any]]:
    if not log.is_file():
        return []
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = []
    current_task = 0 if sequence == "extract" else 1
    for order, record in enumerate(records):
        text = " ".join(record["argv"])
        cur_task = re.search(r"--cur_task\s+(\d+)", text)
        matches = re.findall(r"[Tt]ask(\d+)", text)
        if method_id == "modalprompt" and cur_task:
            # ModalPrompt uses one-based cur_task; its UCIT task2 JSON has a
            # copy/paste output_dir pointing at Task1, so path inference fails.
            current_task = int(cur_task.group(1))
        elif matches:
            # Taskn argv contains both previous_task_model_path and the
            # current output_dir; the current task is the last occurrence.
            current_task = int(matches[-1])
        result.append(_normalise(record, temp_root, sequence, order, current_task))
    return result


def run_sequence(
    v3_root: Path,
    method_id: str,
    benchmark_id: str,
    output_root: Path,
    router: bool = False,
    keep_temp: bool = False,
) -> dict[str, Any]:
    method = get_method(method_id)
    benchmark = get_benchmark(benchmark_id)
    sequence = "router" if router else "train"
    script_name = benchmark.train_script.replace(".sh", "_router.sh") if router else benchmark.train_script
    source_method = v3_root / "LLaVA" / method.v3_dir
    source_script = source_method / "scripts" / "MCITlib" / "Train" / script_name
    if not source_script.is_file():
        raise FileNotFoundError(f"v3 sequence script not found: {source_script}")

    with tempfile.TemporaryDirectory(prefix=f"mcitlib-{method_id}-{benchmark_id}-") as temporary:
        temp_root = Path(temporary)
        _copy_tree_with_root(source_method / "scripts", temp_root / "scripts", temp_root)
        _copy_tree_with_root(v3_root / "configs", temp_root / "configs", temp_root)
        config_aliases = _apply_config_aliases(temp_root / "scripts", method.id)
        sequence_repairs: list[int] = []
        if not router:
            sequence_repairs = _repair_commented_tasks(
                temp_root / "scripts" / "MCITlib" / "Train" / script_name
            )
        script_aliases: dict[str, str] = {}
        config_placeholders: dict[str, Any] = {}
        if router:
            script_aliases, config_placeholders = _apply_router_compat(
                temp_root / "scripts", temp_root / "configs", benchmark.id
            )
        elif method.id == "replay":
            config_placeholders = _apply_replay_placeholders(temp_root / "configs", benchmark.id)
        shim_bin = temp_root / ".shim-bin"
        _write_shims(shim_bin)
        log = temp_root / "commands.jsonl"
        env = os.environ.copy()
        env["PATH"] = f"{shim_bin}{os.pathsep}{env.get('PATH', '')}"
        env["MCITLIB_REAL_PYTHON"] = sys.executable
        env["MCITLIB_DUMP_LOG"] = str(log)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        completed = subprocess.run(
            ["bash", f"scripts/MCITlib/Train/{script_name}"],
            cwd=temp_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        commands = _read_commands(log, temp_root, sequence, method.id)
        if not commands:
            raise RuntimeError(
                f"no commands recorded for {method_id}/{benchmark_id}: "
                f"exit={completed.returncode}; stderr={completed.stderr[-1000:]}"
            )
        payload = {
            "schema_version": 1,
            "backbone": "llava",
            "method": method.id,
            "v3_method": method.v3_dir,
            "benchmark": benchmark.id,
            "v3_benchmark": benchmark.v3_name,
            "sequence": sequence,
            "source_script": f"scripts/MCITlib/Train/{script_name}",
            "root_placeholder": ROOT_PLACEHOLDER,
            "config_aliases": config_aliases,
            "script_aliases": script_aliases,
            "config_placeholders": config_placeholders,
            "sequence_repairs": sequence_repairs,
            "path_placeholders": HARD_CODED_PATH_PLACEHOLDERS,
            "runner_exit_code": completed.returncode,
            "commands": commands,
        }
        if completed.returncode:
            payload["runner_stderr_tail"] = completed.stderr[-2000:]
        destination = output_root / method.id / f"{benchmark.id}{'_router' if router else ''}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if keep_temp:
            preserved = output_root / method.id / f"_{benchmark.id}{'_router' if router else ''}_work"
            if preserved.exists():
                shutil.rmtree(preserved)
            shutil.copytree(temp_root, preserved)
        print(f"{method.id:12} {benchmark.id:9} {sequence:6} commands={len(commands):4} exit={completed.returncode} -> {destination}")
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-root", type=Path, default=os.environ.get("MCITLIB_V3_ROOT", "../MCITlib_v3_ref"))
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/v3_commands/llava"))
    parser.add_argument("--methods", help="comma-separated method IDs or v3 directory names; default all 12")
    parser.add_argument("--benchmarks", help="comma-separated benchmark IDs; default ucit,mllm_dcl,mllm_acl")
    parser.add_argument("--router", action="store_true", help="also dump the three MR-LoRA router sequences")
    parser.add_argument("--only-router", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()
    v3_root = args.v3_root.expanduser().resolve()
    if not (v3_root / "LLaVA").is_dir():
        parser.error(f"v3 root does not contain LLaVA/: {v3_root}")
    method_ids = [get_method(x.strip()).id for x in args.methods.split(",")] if args.methods else [m.id for m in METHODS]
    benchmark_ids = [get_benchmark(x.strip()).id for x in args.benchmarks.split(",")] if args.benchmarks else [b.id for b in BENCHMARKS]
    count = 0
    if not args.only_router:
        for method_id in method_ids:
            for benchmark_id in benchmark_ids:
                run_sequence(v3_root, method_id, benchmark_id, args.out, keep_temp=args.keep_temp)
                count += 1
    if args.router or args.only_router:
        for benchmark_id in benchmark_ids:
            run_sequence(v3_root, "mrlora", benchmark_id, args.out, router=True, keep_temp=args.keep_temp)
            count += 1
    print(f"wrote {count} sequence files under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
