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
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from .config.schema import Command

OK, WARN, FAIL = "OK", "WARN", "FAIL"
INPUT_FLAGS = (
    "--model_name_or_path", "--pretrain_mm_mlp_adapter", "--vision_tower", "--vision-tower", "--text_tower", "--text-tower",
    "--data_path", "--image_folder", "--question-file", "--image-folder", "--annotation-file", "--annotation-dir",
    "--model-base", "--model-path", "--previous_task_model_path", "--dgg_history_model_path", "--dgg_replay_manifest", "--regularization_info_path",
    "--data-config", "--model-config", "--train-config",
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
    if env_name == "llava-modalprompt-ds126":
        try:
            from importlib import import_module

            from deepspeed.utils import safe_get_full_fp32_param, safe_set_full_fp32_param

            tensor_fragment = import_module("deepspeed.utils.tensor_fragment")

            if (safe_get_full_fp32_param is not tensor_fragment.safe_get_full_fp32_param or
                    safe_set_full_fp32_param is not tensor_fragment.safe_set_full_fp32_param or
                    not callable(safe_get_full_fp32_param) or not callable(safe_set_full_fp32_param) or
                    safe_get_full_fp32_param.__module__ != "deepspeed.utils.tensor_fragment" or
                    safe_set_full_fp32_param.__module__ != "deepspeed.utils.tensor_fragment"):
                raise RuntimeError("DeepSpeed safe state helpers are not the official tensor_fragment functions")
        except Exception as error:
            report.add(FAIL, "ModalPrompt DeepSpeed safe state API", f"{type(error).__name__}: {error}",
                       "use the isolated llava-modalprompt-ds126 environment")
        else:
            report.add(OK, "ModalPrompt DeepSpeed safe state API", "official safe_get_full_fp32_param and safe_set_full_fp32_param available")
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


def internvl_keeplora_memory_lower_bound(model: dict, vision: dict) -> tuple[int, int]:
    """FP16 bytes for selected Llama matrices and frozen InternViT matrices.

    This excludes the other weights, temporaries and activations.  The gradient
    extractor makes the Llama matrices trainable and keeps a zeros_like
    accumulator as well as their ordinary gradients on each GPU.
    """
    hidden, intermediate = int(model["hidden_size"]), int(model["intermediate_size"])
    heads, kv_heads = int(model["num_attention_heads"]), int(model["num_key_value_heads"])
    if min(hidden, intermediate, heads, kv_heads) <= 0 or hidden % heads:
        raise ValueError("invalid Llama dimensions")
    kv_width = hidden // heads * kv_heads
    selected = int(model["num_hidden_layers"]) * (2 * hidden**2 + 2 * hidden * kv_width +
                                                  3 * hidden * intermediate)
    vision_hidden, vision_mlp = int(vision["hidden_size"]), int(vision["intermediate_size"])
    vision_matrices = int(vision["num_hidden_layers"]) * (4 * vision_hidden**2 +
                                                         2 * vision_hidden * vision_mlp)
    if min(selected, vision_matrices) <= 0:
        raise ValueError("invalid layer dimensions")
    return selected * 2, vision_matrices * 2


def check_gpus(report: Report, needed: int, method_id: str, gpu_ids: list[str] | None = None,
               backbone_id: str = "llava", model_root: Path | None = None) -> None:
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
    if gpu_ids is not None and any(int(index) >= count for index in gpu_ids):
        report.add(FAIL, "selected GPUs", f"requested {','.join(gpu_ids)} with {count} visible devices")
        return
    selected = [int(index) for index in gpu_ids] if gpu_ids is not None else [0]
    capabilities = [torch.cuda.get_device_capability(index) for index in selected]
    major, minor = min(capabilities)
    memory = min(torch.cuda.get_device_properties(index).total_memory / 2**30 for index in selected)
    report.add(OK if major >= 8 else WARN, "GPU architecture", f"sm_{major}{minor}, {memory:.0f} GB",
               "flash-attn 2 needs an Ampere (sm_80) or newer GPU")
    if method_id == "keeplora" and backbone_id == "internvl":
        try:
            if model_root is None:
                raise ValueError("model root not provided")
            model_path = model_root / "InternVL-Chat-ViT-6B-Vicuna-7B" / "config.json"
            vision_path = model_root / "InternViT-6B-224px" / "config.json"
            model = json.loads(model_path.read_text(encoding="utf-8"))
            vision = json.loads(vision_path.read_text(encoding="utf-8"))
            selected, frozen_vision = internvl_keeplora_memory_lower_bound(model, vision)
        except (OSError, KeyError, TypeError, ValueError) as error:
            report.add(FAIL, "KeepLoRA memory", f"cannot calculate InternVL lower bound: {error}")
            return
        # The extraction step keeps a weight, a same-dtype accumulated gradient,
        # and a normal gradient for the selected matrices, plus frozen vision.
        lower_bound = (3 * selected + frozen_vision) / 2**30
        detail = (f"{memory:.0f} GiB per GPU; selected Llama q/k/v/o/gate/up/down weights "
                  f"{selected / 2**30:.2f} GiB x 3 (weight, gradient, accumulator), "
                  f"frozen InternViT attention/MLP matrices {frozen_vision / 2**30:.2f} GiB; "
                  f"at least {lower_bound:.2f} GiB excluding other weights and activations. "
                  f"Sources: {model_path}, {vision_path}, "
                  "mcitlib/methods/_keeplora/extract_gradients.py")
        report.add(OK if memory >= KEEPLORA_MIN_GPU_GB else WARN, "KeepLoRA memory", detail,
                   f"use at least {KEEPLORA_MIN_GPU_GB} GiB per GPU for KeepLoRA extraction")
    elif method_id == "keeplora" and memory < KEEPLORA_MIN_GPU_GB:
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


def _keeplora_config_inputs(report: Report, commands: list[Command]) -> tuple[list[Command], tuple[str, ...]]:
    """Expose actual KeepLoRA JSON inputs and outputs at each step for doctor only.

    Gradient workers read a model and data; merge-only reads the prior subspace file.
    Each command's analysis copy records the specific file it creates, so a chunk's
    output directory cannot be mistaken for a completed principal-subspace file.
    """
    from .methods._keeplora.config_paths import resolve_fields

    analyzed: list[Command] = []
    external_subspaces: list[str] = []
    invalid: set[str] = set()
    weight_module = "mcitlib.methods._keeplora.extract_weights"
    gradient_module = "mcitlib.methods._keeplora.extract_gradients"
    for command in commands:
        argv = command.argv
        is_weights = weight_module in argv
        is_gradients = gradient_module in argv
        if "MCITLIB_PATH_CONTEXT" not in command.env or not (is_weights or is_gradients):
            analyzed.append(command)
            continue
        merge_only = is_gradients and "--merge-only" in argv
        config_flags = (("--train-config",) if merge_only else ("--model-config",) if is_weights
                        else ("--data-config", "--model-config", "--train-config"))
        configs: dict[str, dict] = {}
        for flag in config_flags:
            if flag not in argv or argv.index(flag) + 1 >= len(argv):
                invalid.add(f"{flag}: missing from KeepLoRA extra command")
                continue
            path = argv[argv.index(flag) + 1]
            try:
                config = json.loads(Path(path).read_text(encoding="utf-8"))
                if not isinstance(config, dict):
                    raise ValueError("expected a JSON object")
                configs[flag] = config
            except (OSError, ValueError) as error:
                invalid.add(f"{path}: {error}")
        context = command.env["MCITLIB_PATH_CONTEXT"]
        data = resolve_fields(configs.get("--data-config", {}), ("train_path", "train_folder"), context)
        model = resolve_fields(configs.get("--model-config", {}), ("model_name",), context)
        train = resolve_fields(configs.get("--train-config", {}), ("model_path", "space_path"), context)
        for flag, config, fields in (("--data-config", data, ("train_path", "train_folder")),
                                     ("--model-config", model, ("model_name",)),
                                     ("--train-config", train, ("model_path", "space_path"))):
            for name in fields:
                if config.get(name) is not None and not isinstance(config[name], str):
                    invalid.add(f"{flag} field {name}: expected a string or null")
                    config.pop(name)
        values = []
        if merge_only:
            if train.get("space_path"):
                space_file = str(Path(train["space_path"]) / "gradients_principle_subspace.pt")
                values.append(("--regularization_info_path", space_file))
                # Existence is checked for external paths; internal paths must have a producer.
                external_subspaces.append(space_file)
        elif is_weights:
            values.append(("--model-path", model.get("model_name")))
        else:
            values.extend((("--data_path", data.get("train_path")), ("--image_folder", data.get("train_folder")),
                           ("--model-path", train.get("model_path"))))
            if train.get("model_path") and "merged" not in train["model_path"] and train["model_path"] != model.get("model_name"):
                values.append(("--model-base", model.get("model_name")))
        synthetic = [item for flag, value in values if value for item in (flag, value)]
        analysis_argv = list(argv)
        if "--output-dir" in analysis_argv and analysis_argv.index("--output-dir") + 1 < len(analysis_argv):
            output_index = analysis_argv.index("--output-dir")
            output_dir = Path(analysis_argv[output_index + 1])
            if is_weights:
                name = "gradients_principle_subspace.pt"
            elif merge_only:
                mode_index = analysis_argv.index("--svd-mode") + 1 if "--svd-mode" in analysis_argv else None
                mode = analysis_argv[mode_index] if mode_index is not None and mode_index < len(analysis_argv) else "fixed_rank"
                name = "gradients_principle_subspace.pt" if mode == "energy" else "gradients_merged.pt"
            else:
                chunk_index = analysis_argv.index("--chunk-idx") + 1 if "--chunk-idx" in analysis_argv else None
                count_index = analysis_argv.index("--num-chunks") + 1 if "--num-chunks" in analysis_argv else None
                chunk = analysis_argv[chunk_index] if chunk_index is not None and chunk_index < len(analysis_argv) else "0"
                count = analysis_argv[count_index] if count_index is not None and count_index < len(analysis_argv) else "1"
                name = f"gradients_chunk_{chunk}_of_{count}.pt"
            analysis_argv[output_index:output_index + 2] = ["--output", str(output_dir / name)]
        analyzed.append(replace(command, argv=synthetic + analysis_argv))
    if invalid:
        report.add(FAIL, "KeepLoRA input configs", "; ".join(sorted(invalid)), "restore the JSON configuration files")
    return analyzed, tuple(external_subspaces)


def check_inputs(report: Report, commands: list[Command], model_root: Path, data_root: Path,
                 output_roots: tuple[Path, ...] = ()) -> None:
    missing: dict[str, set[str]] = {}
    checked: set[str] = set()
    for command in commands:
        argv = command.argv
        for position, token in enumerate(argv[:-1]):
            if token == "--dgg_history_model_path" and argv[position + 1] in {"none", "None", ""}:
                continue
            if token in INPUT_FLAGS and not _under(argv[position + 1], output_roots):
                value = argv[position + 1]
                checked.add(value)
                if not Path(value).exists():
                    root = (str(model_root) if _under(value, (model_root,)) else
                            str(data_root) if _under(value, (data_root,)) else "external inputs")
                    missing.setdefault(root, set()).add(value)
    if not missing:
        report.add(OK, "input files", f"{len(checked)} referenced model/data paths exist")
    else:
        for root, paths in sorted(missing.items()):
            sample = sorted(paths)[:6]
            report.add(FAIL, f"missing under {root}", f"{len(paths)} of them, e.g. " + "; ".join(sample),
                       "download / place the files as described in README sections 'Benchmarks' and 'Models'")
    check_model_contents(report, commands, output_roots)
    check_sample_images(report, commands)


def check_model_contents(report: Report, commands: list[Command], output_roots: tuple[Path, ...]) -> None:
    models: dict[Path, set[str]] = {}
    projectors: set[Path] = set()
    for command in commands:
        prior_tower = command.env.get("MCITLIB_SAME_CLIP_TOWER")
        if command.env.get("MCITLIB_METHOD") == "same" and prior_tower:
            directory = Path(prior_tower)
            if directory.is_dir() and not _under(str(directory), output_roots):
                models.setdefault(directory, set()).update({"--vision_tower", "--text_tower"})
        for index, flag in enumerate(command.argv[:-1]):
            if flag == "--pretrain_mm_mlp_adapter":
                projectors.add(Path(command.argv[index + 1]))
            if flag not in {"--model_name_or_path", "--model-base", "--model-path", "--vision_tower",
                            "--text_tower", "--text-tower"}:
                continue
            directory = Path(command.argv[index + 1])
            if directory.is_dir() and not _under(str(directory), output_roots):
                models.setdefault(directory, set()).add(flag)
    errors = []
    for directory, flags in sorted(models.items()):
        required = ["config.json"]
        if "--vision_tower" in flags:
            required.append("preprocessor_config.json")
        if flags - {"--vision_tower"}:
            required.append("tokenizer_config.json")
            tokenizer_files = ("tokenizer.json", "tokenizer.model")
            has_tokenizer = any((directory / name).is_file() and (directory / name).stat().st_size
                                for name in tokenizer_files)
            has_vocab = all((directory / name).is_file() and (directory / name).stat().st_size
                            for name in ("vocab.json", "merges.txt"))
            if not has_tokenizer and not has_vocab:
                errors.append(f"{directory}: missing tokenizer")
        missing = [name for name in required if not (directory / name).is_file() or not (directory / name).stat().st_size]
        if missing:
            errors.append(f"{directory}: missing {', '.join(missing)}")
        index_file = next((directory / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
                           if (directory / name).is_file()), None)
        if index_file:
            try:
                weights = set(json.loads(index_file.read_text(encoding="utf-8"))["weight_map"].values())
                missing_shards = sorted(name for name in weights if not (directory / name).is_file()
                                        or not (directory / name).stat().st_size)
                if not weights or missing_shards:
                    errors.append(f"{directory}: missing model shards {', '.join(missing_shards)}")
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(f"{index_file}: invalid weight index ({error})")
        elif not any((directory / name).is_file() and (directory / name).stat().st_size
                     for name in ("model.safetensors", "pytorch_model.bin")):
            errors.append(f"{directory}: missing model weights")
    for projector in sorted(projectors):
        if not projector.is_file() or not projector.stat().st_size:
            errors.append(f"{projector}: missing or empty projector")
    if models or projectors:
        report.add(FAIL if errors else OK, "model contents", "; ".join(errors) if errors else f"{len(models)} directories checked",
                   "restore the model configuration, tokenizer, and all weight shards")


def _sample_records(path: Path, limit: int = 3) -> list[dict]:
    """Read only the first few JSON-array records, without loading a large dataset."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer, position, records, eof = "", 0, [], False
        while len(records) < limit:
            if not eof and len(buffer) - position < 65536:
                buffer = buffer[position:]
                position = 0
                chunk = handle.read(65536)
                buffer += chunk
                eof = not chunk
            while position < len(buffer) and buffer[position] in " \t\r\n,":
                position += 1
            if not records and position < len(buffer) and buffer[position] == "[":
                position += 1
                continue
            if position >= len(buffer):
                if eof:
                    break
                continue
            if buffer[position] == "]":
                break
            try:
                value, position = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = handle.read(65536)
                if not chunk:
                    raise
                buffer = buffer[position:] + chunk
                position = 0
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def check_sample_images(report: Report, commands: list[Command]) -> None:
    pairs: set[tuple[str, str]] = set()
    for command in commands:
        argv = command.argv
        for data_flag, image_flag in (("--data_path", "--image_folder"), ("--question-file", "--image-folder")):
            if data_flag in argv and image_flag in argv:
                pairs.add((argv[argv.index(data_flag) + 1], argv[argv.index(image_flag) + 1]))
    errors, checked = [], 0
    for data, folder in sorted(pairs):
        path = Path(data)
        if not path.is_file() or not Path(folder).is_dir() or path.suffix != ".json":
            continue
        try:
            records = _sample_records(path)
            if not records:
                errors.append(f"{data}: no sample records")
            for record in records:
                images = record.get("image")
                if not images:
                    continue
                for name in images if isinstance(images, list) else [images]:
                    image = Path(folder) / str(name)
                    checked += 1
                    if not image.is_file():
                        errors.append(f"{data}: missing sample image {image}")
        except (OSError, ValueError) as error:
            errors.append(f"{data}: cannot read sample records ({error})")
    if pairs:
        report.add(FAIL if errors else OK, "sample images", "; ".join(errors[:6]) if errors else f"{checked} referenced images checked (up to 3 records per input)",
                   "check the dataset files and their image-folder mappings")


def check_dataflow(report: Report, commands: list[Command], roots: tuple[str, ...], already: tuple[str, ...],
                   internal_roots: tuple[str, ...] = ()) -> None:
    from .dataflow import find_gaps

    gaps = find_gaps(commands, roots, already, internal_roots)
    if not gaps:
        report.add(OK, "dataflow", "every intermediate file is written by an earlier step")
        return
    flags = sorted({gap.flag for gap in gaps})
    report.add(FAIL, "dataflow", f"{len(gaps)} reads of files no earlier step writes ({', '.join(flags)}); first: {gaps[0]}",
               "known v3 integration gaps are listed in docs/migration/drift.md (G10); ask before running this sequence")


def check_dgg_plan(report: Report, commands: list[Command]) -> None:
    """Check DGG's replay, previous anchors, and single-GPU runtime contract."""
    produced = set()
    failures = []
    for command in commands:
        argv = command.argv
        if command.kind == "extra" and "mcitlib.methods._dgg.replay" in argv:
            datasets = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--data_path"]
            roots = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--image_folder"]
            if (len(datasets) != command.task_id or len(roots) != len(datasets) or
                    "--output" not in argv or "--manifest" not in argv or "--samples-per-task" not in argv or "--seed" not in argv or
                    argv[argv.index("--samples-per-task") + 1] != "2000" or argv[argv.index("--seed") + 1] != "1993"):
                failures.append(f"task {command.task_id} has an invalid replay stage")
            else:
                for path in datasets:
                    try:
                        with open(path, encoding="utf-8") as file:
                            rows = json.load(file)
                        if not isinstance(rows, list) or not rows:
                            raise ValueError("expected a nonempty JSON array")
                        if not isinstance(rows[0], dict) or not isinstance(rows[0].get("conversations"), list):
                            raise ValueError("expected original QA conversation rows")
                    except (OSError, ValueError, TypeError) as error:
                        failures.append(f"replay input {path}: {error}")
                produced.add(argv[argv.index("--output") + 1])
                produced.add(argv[argv.index("--manifest") + 1])
        elif command.kind == "train" and command.env.get("MCITLIB_METHOD") == "dgg":
            def one(flag):
                if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
                    raise ValueError(f"missing or repeated {flag}")
                return argv[argv.index(flag) + 1]
            try:
                history, data, output = one("--dgg_history_model_path"), one("--data_path"), one("--output_dir")
                if one("--include").split(":", 1)[1].count(",") != 0 or one("--bf16").lower() != "true":
                    raise ValueError("requires one GPU and bf16")
                if one("--lora_r") != "48" or one("--lora_alpha") != "96":
                    raise ValueError("requires DGG's rank 48 and alpha 96")
                if command.task_id == 1:
                    if history not in {"none", "None", ""}:
                        raise ValueError("first task cannot consume history")
                else:
                    if data not in produced or one("--dgg_replay_manifest") not in produced:
                        raise ValueError("replay data and manifest are not produced by an earlier stage")
                    if history not in produced:
                        directory = Path(history)
                        if not directory.is_dir() or not (directory / "non_lora_trainables.bin").is_file() or not (
                            (directory / "adapter_model.bin").is_file() or (directory / "adapter_model.safetensors").is_file()
                        ):
                            raise ValueError("previous task adapter and non-LoRA state have no producer and are missing")
                produced.add(output)
            except (ValueError, IndexError) as error:
                failures.append(f"task {command.task_id}: {error}")
    if failures:
        report.add(FAIL, "DGG replay and history", "; ".join(failures[:8]),
                   "restore original QA inputs and preceding DGG task checkpoints")
    else:
        report.add(OK, "DGG replay and history", "all replay inputs and history dependencies are present or produced")
    try:
        import inspect

        import deepspeed
        from deepspeed.runtime.engine import DeepSpeedEngine
        backward = DeepSpeedEngine.backward
        if backward.__qualname__.startswith("instrument_w_nvtx.<locals>."):
            candidates = [cell.cell_contents for cell in (backward.__closure__ or ())
                          if inspect.isfunction(cell.cell_contents) and cell.cell_contents.__name__ == "backward"]
            if len(candidates) != 1:
                raise ValueError("cannot inspect the instrumented DeepSpeed backward function")
            backward = candidates[0]
        source = inspect.getsource(backward)
        if deepspeed.__version__.split("+")[0] != "0.9.5":
            raise ValueError(f"unexpected DeepSpeed {deepspeed.__version__}; reviewed target is 0.9.5")
        if (source.count("self.optimizer.backward(") < 1 or "self.allreduce_gradients()" not in source or
                source.index("self.optimizer.backward(") >= source.index("self.allreduce_gradients()")):
            raise ValueError("DeepSpeed backward does not expose the reviewed optimizer/reduction boundary")
    except (ImportError, OSError, TypeError, ValueError) as error:
        report.add(FAIL, "DGG DeepSpeed callback boundary", str(error), "use the pinned llava environment")
    else:
        report.add(OK, "DGG DeepSpeed callback boundary", "DeepSpeed 0.9.5 exposes the reviewed boundary")
        report.add(WARN, "DGG ZeRO-2 numerical validation", "real gradient-buffer and continuation checks are pending",
                   "run the bounded two-task GPU validation before scientific use")


def check_prerequisites(report: Report, method_id: str, benchmark_id: str, commands: list[Command], repo_root: Path,
                        ckpt_root: Path) -> None:
    if method_id == "dgg":
        check_dgg_plan(report, commands)
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
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".mcitlib-write-test-",
                                         dir=ckpt_root) as probe:
            probe.write("ok")
            probe.flush()
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



def check_internvl_model(report: Report, model_root: Path, repo_root: Path | None = None) -> None:
    """Inspect local model metadata and the runtime dependency without loading weights."""
    requirements = (repo_root or Path(__file__).resolve().parents[1]) / "envs/internvl-extra.txt"
    try:
        wanted = _pinned(requirements)["timm"]
        installed = metadata.version("timm")
        if importlib.util.find_spec("timm") is None or installed.split("+")[0] != wanted:
            raise ValueError(f"timm {installed}; need {wanted}")
    except (OSError, KeyError, metadata.PackageNotFoundError, ValueError) as error:
        report.add(FAIL, "InternViT dependency", str(error) or "timm is not installed",
                   "install the additional pins: pip install -r envs/internvl-extra.txt")
    else:
        report.add(OK, "InternViT dependency", f"timm {installed} matches {requirements.name}")
    model = model_root / "InternVL-Chat-ViT-6B-Vicuna-7B" / "config.json"
    vision = model_root / "InternViT-6B-224px" / "config.json"
    try:
        model_config = json.loads(model.read_text(encoding="utf-8"))
        vision_config = json.loads(vision.read_text(encoding="utf-8"))
        if (model_config.get("model_type") != "llava" or
                vision_config.get("model_type") != "intern_vit_6b" or
                model_config.get("mm_hidden_size") != vision_config.get("hidden_size") or
                model_config.get("mm_vision_select_layer") != -4):
            raise ValueError("model/projector/InternViT configuration differs")
    except (OSError, ValueError, TypeError) as error:
        report.add(FAIL, "InternVL model and vision config", str(error))
    else:
        report.add(OK, "InternVL model and vision config",
                   f"LLaVA text={model_config.get('num_hidden_layers')} layers, "
                   f"InternViT={vision_config.get('num_hidden_layers')} layers, "
                   f"vision hidden={vision_config.get('hidden_size')}")

def run_doctor(
    method_id: str, benchmark_id: str, paths_file: Path | None, repo_root: Path,
    skip_gpu: bool = False, skip_entrypoints: bool = False, sequence: str = "train",
    validation_fixes: bool = False,
    execution_profile: str | None = None, gpus: str | None = None, master_port: int | None = None,
    expected_eval_samples: int | None = None,
    expected_eval_hashes: dict[str, str] | None = None,
    backbone_id: str = "llava",
    dgg_sample_profile: bool = False, dgg_sample_manifest: Path | None = None,
) -> Report:
    from .dataflow import produced_paths
    from .paths import PathResolver
    from .registry import get_method
    from .runner import build_sequence_commands

    report = Report()
    method = get_method(method_id)
    resolver = PathResolver.from_file(paths_file, repo_root=repo_root)
    try:
        commands = build_sequence_commands(repo_root / "configs", backbone_id, method.id, benchmark_id,
                                           paths=resolver, sequence=sequence, validation_fixes=validation_fixes,
                                           execution_profile=execution_profile, gpus=gpus, master_port=master_port,
                                           expected_eval_samples=expected_eval_samples, expected_eval_hashes=expected_eval_hashes)
        if dgg_sample_manifest is not None and not dgg_sample_profile:
            raise ValueError("--dgg-sample-manifest requires --dgg-sample-profile")
        if dgg_sample_profile:
            if method.id != "dgg":
                raise ValueError("--dgg-sample-profile requires DGG")
            from .methods._dgg.sample import apply_dgg_sample_profile
            apply_dgg_sample_profile(commands, benchmark_id, repo_root, dgg_sample_manifest)
        if method.separate_train_and_infer:
            from .registry import get_benchmark

            ordinary = commands
            if execution_profile:
                # The validation worker evaluates each task immediately after
                # training it; check the same ordered plan and data dependencies.
                commands = [item for item in ordinary if item.task_id == 0]
            for task_id in range(1, len(get_benchmark(benchmark_id).tasks) + 1):
                if execution_profile:
                    commands.extend(item for item in ordinary if item.task_id == task_id)
                commands += build_sequence_commands(repo_root / "configs", backbone_id, method.id, benchmark_id,
                                                    paths=resolver, task_id=task_id, only_eval=True,
                                                    validation_fixes=validation_fixes, execution_profile=execution_profile,
                                                    gpus=gpus, master_port=master_port,
                                                    expected_eval_samples=expected_eval_samples, expected_eval_hashes=expected_eval_hashes)
    except (KeyError, ValueError, FileNotFoundError) as error:
        report.add(FAIL, "build command sequence", str(error), "check --method/--benchmark and the configs directory")
        return report
    report.add(OK, "sequence", f"{method.id} x {benchmark_id}: {len(commands)} steps, environment `{method.env}`")
    check_environment(report, method.env, repo_root)
    if backbone_id == "internvl":
        check_internvl_model(report, resolver.paths.model_root, repo_root)
    if not skip_gpu:
        if execution_profile == "four_gpu_validation":
            if backbone_id == "internvl":
                check_gpus(report, gpus_needed(commands), method.id, ["0", "1", "2", "3"],
                           backbone_id, resolver.paths.model_root)
            else:
                check_gpus(report, gpus_needed(commands), method.id, ["0", "1", "2", "3"])
        elif backbone_id == "internvl":
            check_gpus(report, gpus_needed(commands), method.id, backbone_id=backbone_id,
                       model_root=resolver.paths.model_root)
        else:
            check_gpus(report, gpus_needed(commands), method.id)
    check_vendored_imports(report, repo_root)
    outputs = (resolver.paths.ckpt_root, resolver.results_root)
    analyzed, extra_inputs = _keeplora_config_inputs(report, commands)
    # Octopus can explicitly put generated stage-2 sample JSON under the
    # project logs root, separate from external checkpoints.  Exempt only the
    # exact outputs of its native sample commands; the independent dataflow
    # check below must still prove that each consumer has an earlier producer.
    generated_samples = ()
    if method.id == "octopus" and resolver.octopus_aux_root is not None:
        generated_samples = tuple(
            Path(command.argv[command.argv.index("--output") + 1])
            for command in analyzed
            if command.metadata.get("octopus_stage") == "sample"
            and command.argv.count("--output") == 1
        )
    check_inputs(report, analyzed, resolver.paths.model_root, resolver.paths.data_root, outputs + generated_samples)
    if method.id == "same" and backbone_id == "llava":
        prior_paths = {c.env.get("MCITLIB_SAME_CLIP_TOWER") for c in analyzed
                       if c.kind == "eval" and "--model-path" in c.argv
                       and c.env.get("MCITLIB_METHOD") == "same"}
        if not prior_paths or None in prior_paths or any(not Path(p).is_dir() for p in prior_paths if p):
            report.add(FAIL, "SAME CLIP prior", f"missing routing/text tower directory: {sorted(str(p) for p in prior_paths)}",
                       "configure the LLaVA CLIP vision tower path in model/paths configuration")
        else:
            report.add(OK, "SAME CLIP prior", f"{len(prior_paths)} configured routing/text tower path(s)")
    already: tuple[str, ...] = ()
    if sequence == "router":
        ordinary = build_sequence_commands(repo_root / "configs", backbone_id, method.id, benchmark_id, paths=resolver,
                                           validation_fixes=validation_fixes, execution_profile=execution_profile,
                                           gpus=gpus, master_port=master_port, expected_eval_samples=expected_eval_samples,
                                           expected_eval_hashes=expected_eval_hashes)
        already = tuple(produced_paths(ordinary))
    check_dataflow(report, analyzed, resolver.external_roots(benchmark_id) + extra_inputs, already, tuple(map(str, outputs)))
    if method.id == "pclr":
        from .methods._pclr.doctor import check_pclr

        check_pclr(report, commands)
    check_prerequisites(report, method.id, benchmark_id, commands, repo_root, resolver.paths.ckpt_root)
    if not skip_entrypoints:
        # A CPU-only doctor must not let recorded per-chunk GPU pins override
        # CUDA_VISIBLE_DEVICES='' in its import-only child processes.
        entry_commands = ([replace(command, env={**command.env, "CUDA_VISIBLE_DEVICES": ""}) for command in commands]
                          if skip_gpu else commands)
        check_entry_points(report, entry_commands, repo_root)
    return report


def main_doctor(args) -> int:
    from .paths import repo_root

    profile = getattr(args, "execution_profile", None)
    gpu_spec = getattr(args, "gpus", None)
    port = getattr(args, "master_port", None)
    if profile:
        gpu_spec = gpu_spec or os.environ.get("MCITLIB_GPUS")
        port = port or (int(os.environ["MCITLIB_MASTER_PORT"]) if os.environ.get("MCITLIB_MASTER_PORT") else None)
    report = run_doctor(
        args.method, args.benchmark, args.paths, repo_root(), skip_gpu=args.skip_gpu,
        skip_entrypoints=args.skip_entrypoints, sequence=args.sequence,
        validation_fixes=getattr(args, "validation_fixes", False),
        execution_profile=profile, gpus=gpu_spec, master_port=port,
        expected_eval_samples=getattr(args, "expected_eval_samples", None),
        expected_eval_hashes=getattr(args, "expected_eval_hashes", None),
        backbone_id=getattr(args, "backbone", "llava"),
        dgg_sample_profile=getattr(args, "dgg_sample_profile", False),
        dgg_sample_manifest=getattr(args, "dgg_sample_manifest", None),
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2) if args.json else report.render())
    return 1 if report.failed else 0
