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
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")


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


def _read_commands(log: Path, temp_root: Path, sequence: str) -> list[dict[str, Any]]:
    if not log.is_file():
        return []
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = []
    current_task = 0 if sequence == "extract" else 1
    for order, record in enumerate(records):
        text = " ".join(record["argv"])
        matches = re.findall(r"[Tt]ask(\d+)", text)
        if matches:
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
        commands = _read_commands(log, temp_root, sequence)
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
