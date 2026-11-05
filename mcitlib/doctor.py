# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""``mcitlib doctor``: find everything that would stop a run *before* it uses GPU time.

It checks, for one method and benchmark, the same things the first hours of a real run would hit:
the Python environment, the GPUs, where ``llava``/``peft`` are really imported from, whether each
distinct entry point of the sequence starts, whether every input file exists, whether every
intermediate file a step reads is produced by an earlier step, and the extra prerequisites of the
benchmark (MathVista table, judge API key, ...).  Every failure comes with the action that fixes it.
"""
from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config.schema import Command

OK, WARN, FAIL = "OK", "WARN", "FAIL"
INPUT_FLAGS = (
    "--model_name_or_path", "--pretrain_mm_mlp_adapter", "--vision_tower", "--text_tower", "--text-tower",
    "--data_path", "--image_folder", "--question-file", "--image-folder", "--annotation-file", "--model-base",
)
MATHVISTA_TABLE = "llava_v1.5_7b_MathVista_MINI.xlsx"
KEEPLORA_MIN_GPU_GB = 70


@dataclass
class Check:
    status: str
    name: str
    detail: str = ""
    fix: str = ""

    def line(self) -> str:
        text = f"[{self.status:4}] {self.name}"
        if self.detail:
            text += f": {self.detail}"
        if self.fix and self.status != OK:
            text += f"\n         -> {self.fix}"
        return text


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, status: str, name: str, detail: str = "", fix: str = "") -> None:
        self.checks.append(Check(status, name, detail, fix))

    @property
    def failed(self) -> bool:
        return any(check.status == FAIL for check in self.checks)

    def render(self) -> str:
        counts = {status: sum(1 for c in self.checks if c.status == status) for status in (OK, WARN, FAIL)}
        lines = [check.line() for check in self.checks]
        lines.append(f"\n{counts[OK]} ok, {counts[WARN]} warnings, {counts[FAIL]} failures")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {"failed": self.failed, "checks": [vars(check) for check in self.checks]}


def _pinned(requirements: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def check_environment(report: Report, env_name: str, repo_root: Path) -> None:
    requirements = repo_root / "envs" / f"{env_name}.txt"
    if not requirements.is_file():
        report.add(FAIL, "environment file", f"{requirements} not found")
        return
    version = sys.version_info
    status = OK if (3, 9) <= version[:2] <= (3, 11) else WARN
    report.add(status, "python", f"{version.major}.{version.minor}.{version.micro}",
               "use Python 3.10 (tested) in a fresh conda environment")
    mismatches, missing = [], []
    for name, wanted in sorted(_pinned(requirements).items()):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if installed.split("+")[0] != wanted:
            mismatches.append(f"{name} {installed} (need {wanted})")
    other = "llava-t437" if env_name == "llava" else "llava"
    if missing or mismatches:
        report.add(FAIL, f"packages for environment `{env_name}`", "; ".join(
            ([f"missing: {', '.join(missing)}"] if missing else []) + mismatches),
            f"pip install -r envs/{env_name}.txt  (this method needs `{env_name}`, not `{other}`)")
    else:
        report.add(OK, f"packages for environment `{env_name}`", "all pinned versions match")
    for package, fix, severity in (
        ("flash_attn", "install the flash-attn wheel for your CUDA/torch, e.g. "
                       "pip install flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl "
                       "(see README, section 3)", FAIL),
        ("openpyxl", "pip install openpyxl  (MLLM-ACL Math evaluation reads/writes .xlsx)", WARN),
        ("xlsxwriter", "pip install xlsxwriter  (MLLM-ACL Math evaluation writes .xlsx)", WARN),
        ("tiktoken", "pip install tiktoken  (token counting in the MLLM-ACL Math judge)", WARN),
        ("dotenv", "pip install python-dotenv  (only needed when a .env file holds the judge API key)", WARN),
    ):
        if importlib.util.find_spec(package) is None:
            report.add(severity, f"import {package}", "not installed", fix)
        else:
            report.add(OK, f"import {package}")


def check_gpus(report: Report, needed: int, method_id: str) -> None:
    try:
        import torch
    except ImportError:
        report.add(FAIL, "torch", "not installed", "install the environment first")
        return
    if not torch.cuda.is_available():
        report.add(FAIL, "CUDA", "torch.cuda.is_available() is False",
                   "run on the GPU node, with a CUDA build of torch (cu118 for torch 2.0.1)")
        return
    count = torch.cuda.device_count()
    if count < needed:
        report.add(FAIL, "GPUs", f"{count} visible, the configuration uses {needed}",
                   "make more GPUs visible, or lower gpu_num in the method YAML only if you also "
                   "adjust the batch size (it changes the effective batch)")
    else:
        report.add(OK, "GPUs", f"{count} visible, {needed} used")
    major, minor = torch.cuda.get_device_capability(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 2**30
    report.add(OK if major >= 8 else WARN, "GPU architecture", f"sm_{major}{minor}, {memory:.0f} GB",
               "flash-attn 2 needs an Ampere (sm_80) or newer GPU")
    if method_id == "keeplora" and memory < KEEPLORA_MIN_GPU_GB:
        report.add(WARN, "KeepLoRA memory", f"{memory:.0f} GB per GPU",
                   f"the authors ran KeepLoRA on 80 GB GPUs (README); expect out-of-memory below ~{KEEPLORA_MIN_GPU_GB} GB")


def check_vendored_imports(report: Report, repo_root: Path) -> None:
    from .paths import process_env

    # import_llava() is what every real entry point calls first: it also copes with the
    # transformers-4.37 `llava` model-type clash and fails loudly if a stale install wins.
    code = ("import sys; from mcitlib.backbones.llava.overrides import import_llava; import_llava(); "
            "print(sys.modules['llava'].__file__); print(sys.modules['peft'].__file__)")
    result = subprocess.run([sys.executable, "-c", code], cwd=repo_root, env=process_env({}, repo_root),
                            capture_output=True, text=True, check=False)
    lines = [line for line in result.stdout.splitlines() if line.endswith(".py")]
    if result.returncode or len(lines) < 2:
        tail = (result.stderr.strip().splitlines() or ["no output"])[-1]
        report.add(FAIL, "vendored llava/peft", tail, "run `pip install -e .` inside the checkout, in this environment")
        return
    third_party = (repo_root / "third_party").resolve()
    bad = [path for path in lines if third_party not in Path(path).resolve().parents]
    if bad:
        report.add(FAIL, "vendored llava/peft", f"imported from {bad[0]}",
                   "a stale install shadows the vendored copy: pip uninstall -y llava peft  (or use a fresh environment)")
    else:
        report.add(OK, "vendored llava/peft", f"{Path(lines[0]).parent.parent} , {Path(lines[1]).parent.parent}")
    try:
        distribution = metadata.distribution("llava")
        report.add(WARN, "installed `llava` distribution", f"{distribution.version} at {distribution.locate_file('')}",
                   "harmless while the vendored copy wins (checked above), but `pip uninstall -y llava` avoids confusion")
    except metadata.PackageNotFoundError:
        pass


def check_entry_points(report: Report, commands: list[Command], repo_root: Path) -> None:
    from .selfcheck import distinct_entry_points, run_selfcheck

    entries = distinct_entry_points(commands)
    failures = 0
    for entry, command in entries.items():
        result = run_selfcheck(entry, command, repo_root)
        if not result.ok:
            failures += 1
            report.add(FAIL, f"entry point {entry.label()}", result.message,
                       "this exact command would fail at start-up; fix the environment before running")
    if not failures:
        report.add(OK, "entry points", f"{len(entries)} distinct commands start without error")


def _under(path: str, roots: tuple[Path, ...]) -> bool:
    candidate = Path(path)
    return any(root == candidate or root in candidate.parents for root in roots)


def check_inputs(report: Report, commands: list[Command], model_root: Path, data_root: Path) -> None:
    missing: dict[str, set[str]] = {}
    checked: set[str] = set()
    for command in commands:
        argv = command.argv
        for position, token in enumerate(argv[:-1]):
            if token in INPUT_FLAGS and _under(argv[position + 1], (model_root, data_root)):
                value = argv[position + 1]
                checked.add(value)
                if not Path(value).exists():
                    missing.setdefault(str(model_root if _under(value, (model_root,)) else data_root), set()).add(value)
    if not missing:
        report.add(OK, "input files", f"{len(checked)} referenced model/data paths exist")
        return
    for root, paths in sorted(missing.items()):
        sample = sorted(paths)[:6]
        report.add(FAIL, f"missing under {root}", f"{len(paths)} of them, e.g. " + "; ".join(sample),
                   "download / place the files as described in README sections 'Benchmarks' and 'Models'")


def check_dataflow(report: Report, commands: list[Command], roots: tuple[str, ...], already: tuple[str, ...]) -> None:
    from .dataflow import find_gaps

    gaps = find_gaps(commands, roots, already)
    if not gaps:
        report.add(OK, "dataflow", "every intermediate file is written by an earlier step")
        return
    flags = sorted({gap.flag for gap in gaps})
    report.add(FAIL, "dataflow", f"{len(gaps)} reads of files no earlier step writes ({', '.join(flags)}); first: {gaps[0]}",
               "known v3 integration gaps are listed in docs/migration/drift.md (G10); ask before running this sequence")


def check_prerequisites(report: Report, method_id: str, benchmark_id: str, commands: list[Command], repo_root: Path,
                        ckpt_root: Path) -> None:
    uses_math = any("eval_math" in " ".join(command.argv) or "model_math" in " ".join(command.argv) for command in commands)
    if benchmark_id == "mllm_acl" and uses_math:
        table = repo_root / MATHVISTA_TABLE
        report.add(OK if table.is_file() else FAIL, MATHVISTA_TABLE,
                   str(table) if table.is_file() else "not found in the repository root",
                   f"copy {MATHVISTA_TABLE} (the MathVista answer table used by v3) into {repo_root}")
        key = os.environ.get("OPENAI_API_KEY", "")
        if os.environ.get("LOCAL_LLM") or key.startswith("sk-"):
            report.add(OK, "Math judge", "LOCAL_LLM set" if os.environ.get("LOCAL_LLM") else "OPENAI_API_KEY set")
        else:
            report.add(FAIL, "Math judge (MLLM-ACL)",
                       "neither OPENAI_API_KEY (sk-...) nor LOCAL_LLM is set",
                       "export OPENAI_API_KEY=sk-... (the scorer asks GPT-4o-mini), or LOCAL_LLM=<model served via an "
                       "OpenAI-compatible endpoint>; without it the MLLM-ACL sequence stops after task 2")
    if method_id == "smolora":
        assets = repo_root / "assets" / "smolora"
        names = [f"ins_emb_multi_{name}.pkl" for name in ("ucit", "dcl", "acl")]
        missing = [name for name in names if not (assets / name).is_file()]
        report.add(FAIL if missing else OK, "SMoLoRA instruction embeddings", ", ".join(missing) or "present",
                   "restore assets/smolora/*.pkl from the repository")
    try:
        ckpt_root.mkdir(parents=True, exist_ok=True)
        probe = ckpt_root / ".mcitlib-write-test"
        probe.write_text("ok")
        probe.unlink()
        free = shutil.disk_usage(ckpt_root).free / 2**30
        report.add(OK if free >= 200 else WARN, "checkpoint directory", f"writable, {free:.0f} GB free",
                   "a full sequence writes several 7B checkpoints; keep at least ~200 GB free")
    except OSError as error:
        report.add(FAIL, "checkpoint directory", f"{ckpt_root}: {error}", "choose a writable ckpt_root in configs/paths.yaml")


def gpus_needed(commands: list[Command]) -> int:
    for command in commands:
        if command.kind == "train" and "--include" in command.argv:
            spec = command.argv[command.argv.index("--include") + 1]
            return len(spec.split(":", 1)[1].split(","))
    return 1


def run_doctor(
    method_id: str, benchmark_id: str, paths_file: Path | None, repo_root: Path,
    skip_gpu: bool = False, skip_entrypoints: bool = False, sequence: str = "train",
) -> Report:
    from .dataflow import produced_paths
    from .paths import PathResolver
    from .registry import get_method
    from .runner import build_sequence_commands

    report = Report()
    method = get_method(method_id)
    resolver = PathResolver.from_file(paths_file, repo_root=repo_root)
    try:
        commands = build_sequence_commands(repo_root / "configs", "llava", method.id, benchmark_id,
                                           paths=resolver, sequence=sequence)
        if method.separate_train_and_infer:
            from .registry import get_benchmark

            for task_id in range(1, len(get_benchmark(benchmark_id).tasks) + 1):
                commands += build_sequence_commands(repo_root / "configs", "llava", method.id, benchmark_id,
                                                    paths=resolver, task_id=task_id, only_eval=True)
    except (KeyError, ValueError, FileNotFoundError) as error:
        report.add(FAIL, "build command sequence", str(error), "check --method/--benchmark and the configs directory")
        return report
    report.add(OK, "sequence", f"{method.id} x {benchmark_id}: {len(commands)} steps, environment `{method.env}`")
    check_environment(report, method.env, repo_root)
    if not skip_gpu:
        check_gpus(report, gpus_needed(commands), method.id)
    check_vendored_imports(report, repo_root)
    check_inputs(report, commands, resolver.paths.model_root, resolver.paths.data_root)
    already: tuple[str, ...] = ()
    if sequence == "router":
        ordinary = build_sequence_commands(repo_root / "configs", "llava", method.id, benchmark_id, paths=resolver)
        already = tuple(produced_paths(ordinary))
    check_dataflow(report, commands, (str(resolver.paths.model_root), str(resolver.paths.data_root)), already)
    check_prerequisites(report, method.id, benchmark_id, commands, repo_root, resolver.paths.ckpt_root)
    if not skip_entrypoints:
        check_entry_points(report, commands, repo_root)
    return report


def main_doctor(args) -> int:
    from .paths import repo_root

    report = run_doctor(
        args.method, args.benchmark, args.paths, repo_root(), skip_gpu=args.skip_gpu,
        skip_entrypoints=args.skip_entrypoints, sequence=args.sequence,
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2) if args.json else report.render())
    return 1 if report.failed else 0
