# SPDX-License-Identifier: Apache-2.0
"""InternVL validation source, sample and four-GPU lease binding."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .paths import PathResolver

GPU_IDS = ("0", "1", "2", "3")
SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".ruff_cache"}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_inventory(project: Path) -> dict[str, str]:
    """Bind all shipped executable source and configurations, not cache output."""
    project = project.resolve()
    result = {}
    for root_name in ("mcitlib", "configs", "third_party", "envs", "assets", "resources"):
        root = project / root_name
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"source root missing or symlinked: {root}")
        for folder, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS)
            for name in dirs:
                if (Path(folder) / name).is_symlink():
                    raise ValueError(f"source directory is a symlink: {Path(folder) / name}")
            for name in sorted(names):
                if name.endswith((".pyc", ".pyo")):
                    continue
                path = Path(folder) / name
                if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                    raise ValueError(f"source is not a regular file: {path}")
                result[str(path.relative_to(project))] = sha256(path)
    project_config = project / "pyproject.toml"
    if project_config.is_symlink() or not project_config.is_file():
        raise ValueError("project configuration is not a regular file")
    result["pyproject.toml"] = sha256(project_config)
    return dict(sorted(result.items()))


def sample_inputs(project: Path, resolver: PathResolver, auxiliary: Path | None = None) -> dict[str, str]:
    """Verify both 3k train and 1k evaluation mappings against the sample manifest."""
    path = project / "configs/validation_3k_1k/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if len(tasks) != 11 or {x["benchmark"] for x in tasks} != {"ucit", "mllm_dcl"}:
        raise ValueError("sample manifest does not contain both complete benchmarks")
    inputs: dict[str, str] = {}
    for task in tasks:
        benchmark = task["benchmark"]
        for split, count in (("train", 3000), ("test", 1000)):
            item = task["splits"][split]
            output = Path(resolver.resolve_templates(item["output"])).resolve(strict=True)
            sources = [item["source"], *(alias["source"] for alias in item.get("source_aliases", []))]
            if any(Path(resolver.resolve_templates(source, benchmark)).resolve() != output for source in sources):
                raise ValueError(f"sample mapping differs for {benchmark}/{task['name']}/{split}")
            raw = output.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["output_sha256"] or item["sample_count"] != count:
                raise ValueError(f"sample hash or budget differs for {benchmark}/{task['name']}/{split}")
            data = json.loads(raw)
            if not isinstance(data, list) or len(data) != count:
                raise ValueError(f"sample row count differs for {benchmark}/{task['name']}/{split}")
            inputs[str(output)] = item["output_sha256"]
        caption = task.get("caption_reference")
        if caption:
            output = Path(resolver.resolve_templates(caption["output"])).resolve(strict=True)
            if (Path(resolver.resolve_templates(caption["source"], benchmark)).resolve() != output or
                    sha256(output) != caption["output_sha256"]):
                raise ValueError(f"caption reference differs for {benchmark}/{task['name']}")
            inputs[str(output)] = caption["output_sha256"]
    if auxiliary:
        inputs.update(verify_auxiliary(project, resolver, auxiliary, manifest))
    return dict(sorted(inputs.items()))


def verify_auxiliary(project: Path, resolver: PathResolver, overlay: Path, sample: dict[str, Any]) -> dict[str, str]:
    """Prove Replay's 3k current plus 20 original QAs per previous task and router inputs."""
    import yaml
    manifest_path = (project / "data/validation_aux/manifest.json").resolve(strict=True)
    manifest_hash = sha256(manifest_path)
    auxiliary = json.loads(manifest_path.read_text())
    sample_hash = sha256(project / "configs/validation_3k_1k/manifest.json")
    if (auxiliary.get("sample_manifest_sha256") != sample_hash or auxiliary.get("pool_size_per_task") != 20 or
            auxiliary.get("sample_count_per_task") != 3000):
        raise ValueError("auxiliary sample identity or replay budget differs")
    samples = {(row["benchmark"], row["name"]): row for row in sample["tasks"]}
    rows = {(row["benchmark"], row["name"]): row for row in auxiliary["tasks"]}
    if set(samples) != set(rows) or len(rows) != 11 or len(auxiliary["tasks"]) != 11:
        raise ValueError("auxiliary task inventory differs")
    mapping = yaml.safe_load(overlay.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or set(mapping) != {"path_mappings"}:
        raise ValueError("auxiliary paths must contain only path_mappings")
    actual_mappings = set()
    for benchmark, items in mapping["path_mappings"].items():
        for entry in items:
            if set(entry) != {"from", "to"}:
                raise ValueError("invalid auxiliary path mapping")
            actual_mappings.add((benchmark, entry["from"], str(Path(resolver.resolve_templates(entry["to"])).resolve(strict=True))))
    expected_mappings = set()
    source_indexes: dict[tuple[str, str], set[int]] = {}
    for key, row in rows.items():
        benchmark, name = key
        stage = row["stage"]
        peers = sorted((item for item in auxiliary["tasks"] if item["benchmark"] == benchmark), key=lambda item: item["stage"])
        if [item["stage"] for item in peers] != list(range(1, len(peers) + 1)) or peers[stage - 1]["name"] != name:
            raise ValueError("auxiliary task order differs")
        previous = row["previous_rows"]
        expected_counts = {item["name"]: 20 for item in peers[:stage - 1]}
        if (row["replay_current_count"] != 3000 or row["replay_previous_count"] != 20 * (stage - 1) or
                row["replay_count"] != 3000 + 20 * (stage - 1) or row["router_count"] != 20 * stage or
                Counter(item["task"] for item in previous) != expected_counts):
            raise ValueError(f"auxiliary Replay/router counts differ: {benchmark}/{name}")
        split = samples[key]["splits"]["train"]
        primary = Path(resolver.resolve_templates(split["output"])).resolve(strict=True)
        if (Path(row["sample_train"]).resolve() != primary or row["sample_train_sha256"] != split["output_sha256"]):
            raise ValueError("auxiliary primary sample differs")
        logical_dir = split["source"].rsplit("/", 1)[0]
        for kind in ("replay", "router"):
            target = (project / "data/validation_aux" / benchmark / name / (kind + ".json")).resolve(strict=True)
            expected_mappings.add((benchmark, logical_dir + "/" + kind + ".json", str(target)))
        for item in previous:
            index = item["source_index_zero_based"]
            if type(index) is not int or index < 0:
                raise ValueError("invalid original Replay row index")
            source_indexes.setdefault((benchmark, item["task"]), set()).add(index)
    if actual_mappings != expected_mappings or sum(len(items) for items in mapping["path_mappings"].values()) != 22:
        raise ValueError("auxiliary mappings differ from the reviewed Replay/router tasks")
    originals = {}
    for key, indexes in source_indexes.items():
        source = rows[key]["source_train"]
        raw = Path(source["source"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["source_sha256"]:
            raise ValueError(f"original Replay source changed: {key}")
        data = json.loads(raw)
        if not isinstance(data, list) or any(index >= len(data) for index in indexes):
            raise ValueError("original Replay source row index is invalid")
        originals[key] = {index: data[index] for index in indexes}
    inputs = {str(manifest_path): manifest_hash}
    for key, row in rows.items():
        benchmark, name = key
        primary = json.loads(Path(row["sample_train"]).read_text())
        replay_path = (project / "data/validation_aux" / benchmark / name / "replay.json").resolve(strict=True)
        router_path = replay_path.with_name("router.json")
        replay_bytes, router_bytes = replay_path.read_bytes(), router_path.read_bytes()
        if (hashlib.sha256(replay_bytes).hexdigest() != row["replay_output_sha256"] or
                hashlib.sha256(router_bytes).hexdigest() != row["router_output_sha256"]):
            raise ValueError("Replay/router sample hash differs")
        replay, router = json.loads(replay_bytes), json.loads(router_bytes)
        if not isinstance(replay, list) or not isinstance(router, list) or len(replay) != row["replay_count"] or len(router) != row["router_count"]:
            raise ValueError("Replay/router sample length differs")
        def expected_row(item, source_task):
            item = copy.deepcopy(item)
            if benchmark == "mllm_dcl":
                item["image"] = source_task + "/" + item["image"]
            return item
        if replay[:3000] != [expected_row(item, name) for item in primary]:
            raise ValueError("Replay primary rows differ from the 3k sample")
        for actual, item in zip(replay[3000:], row["previous_rows"]):
            original = originals[(benchmark, item["task"])][item["source_index_zero_based"]]
            if actual != expected_row(original, item["task"]):
                raise ValueError("Replay previous QA differs from its original row")
        inputs[str(replay_path)] = row["replay_output_sha256"]
        inputs[str(router_path)] = row["router_output_sha256"]
    return inputs


def build_binding(project: Path, profile: Path, auxiliary: Path | None = None) -> dict[str, Any]:
    project, profile = project.resolve(), profile.resolve(strict=True)
    auxiliary = auxiliary.resolve(strict=True) if auxiliary else None
    resolver = PathResolver.from_file(profile, repo_root=project)
    return {"schema_version": 1, "backbone": "internvl", "project": str(project),
            "source_files": source_inventory(project), "profile": {"path": str(profile), "sha256": sha256(profile)},
            "auxiliary": {"path": str(auxiliary), "sha256": sha256(auxiliary)} if auxiliary else None,
            "input_files": sample_inputs(project, resolver, auxiliary),
            "budget": {"train_per_task": 3000, "eval_per_task": 1000, "minimum_free_gib": 32,
                       "gpus": list(GPU_IDS)}}


def verify_binding(path: str | Path, expected_sha: str, project: Path) -> dict[str, Any]:
    path, project = Path(path), project.resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha or "") or sha256(path) != expected_sha:
        raise ValueError("InternVL runtime binding SHA differs")
    binding = json.loads(path.read_text(encoding="utf-8"))
    if (binding.get("schema_version") != 1 or binding.get("backbone") != "internvl" or
            binding.get("project") != str(project) or binding.get("budget") !=
            {"train_per_task": 3000, "eval_per_task": 1000, "minimum_free_gib": 32, "gpus": list(GPU_IDS)}):
        raise ValueError("InternVL runtime binding identity differs")
    actual = source_inventory(project)
    if actual != binding.get("source_files"):
        raise ValueError("InternVL source/config inventory or hashes changed")
    for record in (binding["profile"], binding.get("auxiliary")):
        if record is not None and sha256(record["path"]) != record["sha256"]:
            raise ValueError("InternVL profile changed")
    if not isinstance(binding.get("input_files"), dict) or len(binding["input_files"]) < 22:
        raise ValueError("InternVL sample input evidence is incomplete")
    for filename, digest in binding["input_files"].items():
        if not Path(filename).is_absolute() or sha256(filename) != digest:
            raise ValueError(f"InternVL input changed: {filename}")
    return binding


def gpu_lease_root(project: Path, lease_root: str | Path | None = None) -> Path:
    project = project.resolve()
    root = Path(lease_root or os.environ.get("MCITLIB_INTERNVL_LEASE_ROOT", str(project / "logs/.gpu-leases")))
    if not root.is_absolute() or ".." in root.parts or project not in root.parents:
        raise ValueError("GPU lease root must be an absolute project-owned directory")
    current = project
    for component in root.relative_to(project).parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"GPU lease directory is symlinked: {current}")
    return root


def acquire_gpu_leases(project: Path, lease_root: str | Path | None = None) -> tuple[Path, list[int]]:
    """A public launcher may acquire the four shared project locks itself."""
    root = gpu_lease_root(project, lease_root)
    root.mkdir(parents=True, exist_ok=True)
    fds = []
    try:
        for index in range(4):
            path = root / f"gpu-{index}.lease"
            fd = os.open(str(path), os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
            fds.append(fd)
            actual, observed = os.fstat(fd), path.lstat()
            if not stat.S_ISREG(observed.st_mode) or (actual.st_dev, actual.st_ino) != (observed.st_dev, observed.st_ino):
                raise ValueError(f"GPU lease inode differs: {path}")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_gpu_leases(project, ",".join(map(str, fds)), root)
        return root, fds
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise


def gpu_idle() -> dict[str, str]:
    """Observe only the four devices assigned to this experiment."""
    base = ["nvidia-smi", "-i", "0,1,2,3", "--format=csv,noheader,nounits"]
    raw = subprocess.run(base + ["--query-gpu=index,uuid,memory.used,utilization.gpu"],
                         check=True, text=True, capture_output=True, timeout=18).stdout
    rows = [row.split(",") for row in raw.splitlines() if row.strip()]
    if (len(rows) != 4 or any(len(row) != 4 for row in rows) or
            {int(row[0].strip()) for row in rows} != {0, 1, 2, 3} or
            any(int(row[2].strip()) > 64 or int(row[3].strip()) != 0 for row in rows)):
        raise RuntimeError("GPU 0–3 is not idle: " + raw)
    uuids = {row[1].strip() for row in rows}
    apps = subprocess.run(base + ["--query-compute-apps=gpu_uuid,pid"],
                          check=True, text=True, capture_output=True, timeout=18).stdout
    if any(line.strip() and line.split(",")[0].strip() in uuids for line in apps.splitlines()):
        raise RuntimeError("GPU 0–3 has an active compute application: " + apps)
    return {"inventory": raw, "compute_apps": apps}


def verify_gpu_leases(project: Path, value: str | None = None, lease_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Accept only four already-exclusive file descriptions for GPUs 0–3."""
    text = value if value is not None else os.environ.get("MCITLIB_INTERNVL_LEASE_FDS", "")
    root = gpu_lease_root(project, lease_root)
    try:
        fds = [int(item) for item in text.split(",")]
    except ValueError as exc:
        raise ValueError("missing four inherited GPU lease descriptors") from exc
    if len(fds) != 4 or len(set(fds)) != 4 or any(fd < 0 for fd in fds):
        raise ValueError("missing four distinct inherited GPU lease descriptors")
    evidence = []
    for index, fd in enumerate(fds):
        path = root / f"gpu-{index}.lease"
        if path.is_symlink():
            raise ValueError(f"GPU lease is symlinked: {path}")
        expected, actual = path.stat(), os.fstat(fd)
        if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
            raise ValueError(f"GPU lease descriptor differs: {path}")
        lines = Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines()
        matched = []
        for line in lines:
            fields = line.split()
            if "FLOCK" not in fields:
                continue
            n = fields.index("FLOCK")
            if fields[n:n+3] != ["FLOCK", "ADVISORY", "WRITE"] or fields[n+5:n+7] != ["0", "EOF"]:
                continue
            parts = fields[n+4].split(":")
            if len(parts) == 3 and (int(parts[0], 16), int(parts[1], 16), int(parts[2])) == (
                    os.major(actual.st_dev), os.minor(actual.st_dev), actual.st_ino):
                matched.append(line)
        if not matched:
            raise ValueError(f"GPU lease is not exclusively held by inherited descriptor: {path}")
        evidence.append({"gpu": index, "fd": fd, "path": str(path), "dev": actual.st_dev,
                         "inode": actual.st_ino, "fdinfo": matched})
    return evidence


def audit_commands(commands: list[Any], *, method: str, task_count: int, source_files: dict[str, str]) -> list[dict[str, Any]]:
    """Check the actual launcher and recorded sample budget before any step runs."""
    profile_path = "configs/training_profiles/internvl_sample_unified.yaml"
    profile_sha = source_files[profile_path]
    main_tasks, audit = set(), []
    for index, command in enumerate(commands):
        argv, env = command.argv, command.env
        if "NVIDIA_VISIBLE_DEVICES" in env or env.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID") != "PCI_BUS_ID":
            raise ValueError(f"command {index} has an unreviewed GPU visibility override")
        for key in ("CUDA_VISIBLE_DEVICES",):
            if key in env:
                ids = env[key].split(",")
                if not ids or len(ids) != len(set(ids)) or any(item not in GPU_IDS for item in ids):
                    raise ValueError(f"command {index} targets a GPU outside 0–3")
        if "--include" in argv:
            position = argv.index("--include")
            if argv.count("--include") != 1 or position + 1 >= len(argv) or argv[position + 1] != "localhost:0,1,2,3":
                raise ValueError(f"command {index} has an unreviewed GPU launcher")
        if command.kind != "train":
            continue
        sequence = command.sequence
        key = "training_profile" if sequence == "train" else "auxiliary_training_profile" if sequence == "router" else None
        if key is None:
            raise ValueError(f"unreviewed training sequence in command {index}")
        meta = command.metadata.get(key)
        if not isinstance(meta, dict) or any(meta.get(k) != v for k, v in {
                "id": "internvl_sample_unified", "backbone": "internvl", "path": profile_path,
                "sha256": profile_sha, "world_size": 4,
                "role": "main_train" if sequence == "train" else "auxiliary_preserved"}.items()):
            raise ValueError(f"training profile evidence differs for command {index}")
        values = {}
        for flag in ("--per_device_train_batch_size", "--gradient_accumulation_steps", "--learning_rate", "--num_train_epochs"):
            if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
                raise ValueError(f"command {index} lacks exactly one {flag}")
            values[flag] = argv[argv.index(flag) + 1]
        try:
            micro, accumulation = int(values["--per_device_train_batch_size"]), int(values["--gradient_accumulation_steps"])
            learning_rate, epochs = Decimal(values["--learning_rate"]), Decimal(values["--num_train_epochs"])
            recorded_rate, recorded_epochs = Decimal(str(meta.get("learning_rate"))), Decimal(str(meta.get("epochs")))
        except (ValueError, InvalidOperation) as exc:
            raise ValueError(f"invalid training budget in command {index}") from exc
        if (meta.get("microbatch") != micro or meta.get("gradient_accumulation") != accumulation or
                meta.get("global_batch_size") != 4 * micro * accumulation or
                recorded_rate != learning_rate or recorded_epochs != epochs):
            raise ValueError(f"training argv and profile metadata differ for command {index}")
        if sequence == "train":
            expected_epochs = 1
            if (micro, accumulation, 4 * micro * accumulation, learning_rate, epochs) != (
                    1, 16, 64, Decimal("0.0002"), Decimal(expected_epochs)):
                raise ValueError(f"main training budget differs for command {index}")
            main_tasks.add(command.task_id)
        else:
            original = meta.get("source_values") or {}
            try:
                preserved = (Decimal(original["--learning_rate"]) == learning_rate and
                             Decimal(original["--num_train_epochs"]) == epochs)
            except (KeyError, ValueError, InvalidOperation) as exc:
                raise ValueError("missing original router budget") from exc
            if not preserved or (method == "mrlora" and epochs != Decimal(30)):
                raise ValueError("router training does not preserve its original 30-epoch budget")
        audit.append({"index": index, "task": command.task_id, "sequence": sequence, "profile": dict(meta)})
    if main_tasks != set(range(1, task_count + 1)):
        raise ValueError("primary training commands do not cover every benchmark task")
    if method == "mrlora" and not any(item["sequence"] == "router" for item in audit):
        raise ValueError("MR-LoRA router training is missing")
    return audit
