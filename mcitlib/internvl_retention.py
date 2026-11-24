# SPDX-License-Identifier: Apache-2.0
"""Retention for a *new* SEFE or KeepLoRA validation job; no training logic changes.

Register while its checkpoint root is empty. Call before each command, after
the previous command's event AND durable marker have been written. All paths
are bound to this run and deletions use dirfd-based, no-follow operations.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MERGE = "mcitlib/methods/_sefe/merge.py"
KEYS = "mcitlib/methods/_sefe/key_elements.py"
KEEP_MERGE = "mcitlib/methods/_keeplora/merge.py"
KEEP_MODULES = {"mcitlib.methods._keeplora.extract_gradients", "mcitlib.methods._keeplora.extract_weights"}
NAME = re.compile(r"Task([1-9][0-9]*)_internvl_lora_merged\Z")
TASKS = {"ucit": 6, "mllm_dcl": 5}


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _command(command: Any) -> dict[str, Any]:
    value = command.as_dict() if hasattr(command, "as_dict") else command
    if not isinstance(value, dict) or not isinstance(value.get("argv"), list) or not value["argv"]:
        raise ValueError("invalid command")
    if not all(isinstance(x, str) and x for x in value["argv"]):
        raise ValueError("invalid command argv")
    return json.loads(_json(value))


def _identity(command: dict[str, Any]) -> str:
    # Matches runner._step_identity, including its JSON whitespace.
    return hashlib.sha256(json.dumps(command, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def _absolute(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"expected a non-traversing absolute path: {value}")
    return path


def _within(path: Path, root: Path) -> bool:
    return path != root and root in path.parents


def _flag(argv: list[str], name: str) -> str:
    indexes = [i for i, x in enumerate(argv) if x == name]
    if len(indexes) != 1 or indexes[0] + 1 >= len(argv):
        raise ValueError(f"missing or ambiguous {name}")
    return argv[indexes[0] + 1]


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _known_entry(argv: list[str], method: str) -> bool:
    if argv[0] == "deepspeed":
        return "mcitlib.backbones.llava.train_entry" in argv
    if argv[0] not in {"python", "python3"} or len(argv) < 2:
        return False
    scripts = {MERGE, KEYS} if method == "sefe" else {KEEP_MERGE}
    modules = {"mcitlib.backbones.llava.eval_runners.entry", "mcitlib.eval.chunks"}
    if method == "keeplora":
        modules |= KEEP_MODULES
    return argv[1] in scripts or (len(argv) >= 3 and argv[1] == "-m" and argv[2] in modules)


def build_plan(commands: list[Any], ckpt_root: str | Path, benchmark: str, method: str = "sefe") -> dict[str, Any]:
    """Find producers and all later literal consumers from the exact worker list."""
    root = _absolute(ckpt_root)
    if method not in {"sefe", "keeplora"} or benchmark not in TASKS or root.parts[-2:] != (method, benchmark):
        raise ValueError("expected a new SEFE or KeepLoRA checkpoint root")
    rows = [_command(c) for c in commands]
    ids = [_identity(c) for c in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("empty plan or ambiguous repeated command identities")
    if any(not _known_entry(row["argv"], method) for row in rows):
        raise ValueError("unknown retention command: implicit dependencies cannot be audited")
    merge = MERGE if method == "sefe" else KEEP_MERGE
    references = []
    for row in rows:
        expanded = json.loads(_json(row))
        if method == "keeplora" and len(row["argv"]) >= 3 and row["argv"][1:3] in [
                ["-m", name] for name in KEEP_MODULES]:
            from .paths import PathResolver
            context = row["env"].get("MCITLIB_PATH_CONTEXT")
            if not context:
                raise ValueError("KeepLoRA config command lacks its path context")
            resolver, scope = PathResolver.from_context(context)
            if Path(resolver.paths.ckpt_root) != root:
                raise ValueError("KeepLoRA path context points to a different checkpoint root")
            required = "--train-config" if row["argv"][2].endswith("extract_gradients") else "--model-config"
            if required not in row["argv"]:
                raise ValueError("KeepLoRA command has no audited source configuration")
            config_fields = []
            for flag in ("--train-config", "--model-config", "--data-config"):
                if flag in row["argv"]:
                    path = _absolute(_flag(row["argv"], flag))
                    if (not _within(path, Path(resolver.paths.repo_root)) or path.resolve() != path or
                            path.is_symlink() or not path.is_file()):
                        raise ValueError("KeepLoRA source config is missing or symlinked")
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("KeepLoRA source config is not a mapping")
                    config_fields.extend(resolver.resolve_templates(value, scope) for value in _strings(payload))
            expanded["_resolved_config_references"] = config_fields
        references.append(expanded)
    producers: dict[int, tuple[int, Path]] = {}
    for index, row in enumerate(rows):
        argv = row["argv"]
        if merge not in argv:
            continue
        target = _absolute(_flag(argv, "--save-model-path"))
        match = NAME.fullmatch(target.name)
        if not match or not _within(target, root):
            raise ValueError(f"unexpected SEFE merge output: {target}")
        task = int(match.group(1))
        if task in producers or row.get("task_id") != task:
            raise ValueError(f"ambiguous SEFE task {task}")
        if _absolute(_flag(argv, "--model-path")) != target.with_name(target.name.removesuffix("_merged")):
            raise ValueError("SEFE merge does not consume its task adapter")
        producers[task] = (index, target)
    if set(producers) != set(range(1, TASKS[benchmark] + 1)):
        raise ValueError("SEFE plan must contain every task merge exactly once")
    if len({path.parent for _, path in producers.values()}) != 1:
        raise ValueError("SEFE task merges do not share a checkpoint directory")
    entries = []
    for task in range(1, TASKS[benchmark] + 1):
        index, path = producers[task]
        consumers = []
        for i, row in enumerate(references):
            # Scan all recorded command fields; exclude only this producer's
            # own output flag/value, not any other reference in that command.
            copy = json.loads(_json(row))
            if i == index:
                at = copy["argv"].index("--save-model-path")
                del copy["argv"][at:at + 2]
            if any(path.name in text and str(path) not in text for text in _strings(copy)):
                raise ValueError(f"ambiguous or indirect merged-model reference: {path.name}")
            if str(path) in json.dumps(copy, ensure_ascii=False):
                consumers.append(i)
        if not consumers or min(consumers) <= index:
            raise ValueError(f"invalid producer/consumers for {path}")
        entry = {"task": task, "path": str(path), "producer": index,
                 "consumers": consumers, "last_consumer": max(consumers),
                 "final": task == TASKS[benchmark]}
        if not entry["final"]:
            next_index, next_path = producers[task + 1]
            if next_index <= index or _absolute(_flag(rows[next_index]["argv"], "--model-base")) != path:
                raise ValueError(f"task {task + 1} does not merge from {path}")
            entry.update(next_producer=next_index, next_path=str(next_path))
        entries.append(entry)
    return {"method": method, "benchmark": benchmark, "ckpt_root": str(root), "commands": rows, "ids": ids,
            "commands_sha256": hashlib.sha256(_json(rows)).hexdigest(), "entries": entries}


def _open_dir(path: Path, *, create: bool = False) -> int:
    """Walk absolute directories by fd, refusing every symlink component."""
    path = _absolute(path)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd=fd)
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_beneath(rootfd: int, root: Path, path: Path) -> int:
    if path != root and not _within(path, root):
        raise ValueError(f"checkpoint path escaped its run root: {path}")
    fd = os.dup(rootfd)
    try:
        for part in path.relative_to(root).parts:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_regular(directory: int, name: str) -> bytes:
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"invalid checkpoint file name: {name}")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError(f"not a regular file: {name}")
        return handle.read()


def _nonempty(directory: int, name: str) -> int:
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"invalid checkpoint file name: {name}")
    info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValueError(f"missing, linked, or empty checkpoint file: {name}")
    return info.st_size


def _safetensors(directory: int, name: str) -> set[str]:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size < 10:
            raise ValueError(f"invalid safetensors file: {name}")
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if not 2 <= header_size <= min(info.st_size - 8, 100_000_000):
            raise ValueError(f"invalid safetensors header: {name}")
        header = json.loads(handle.read(header_size))
        if not isinstance(header, dict):
            raise ValueError(f"invalid safetensors metadata: {name}")
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        if not tensors:
            raise ValueError(f"empty safetensors: {name}")
        widths = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
                  "U16": 2, "I16": 2, "F16": 2, "BF16": 2, "U32": 4, "I32": 4,
                  "F32": 4, "U64": 8, "I64": 8, "F64": 8}
        ranges = []
        for key, item in tensors.items():
            if not isinstance(key, str) or not isinstance(item, dict):
                raise ValueError(f"invalid tensor metadata: {name}")
            shape, dtype, offsets = item.get("shape"), item.get("dtype"), item.get("data_offsets")
            if (not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape)
                    or dtype not in widths or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(v) is not int for v in offsets)):
                raise ValueError(f"invalid tensor metadata: {name}")
            elements = 1
            for size in shape:
                elements *= size
            if offsets[1] - offsets[0] != elements * widths[dtype]:
                raise ValueError(f"inconsistent tensor size: {name}")
            ranges.append(tuple(offsets))
        ranges.sort()
        end = 0
        for start, stop in ranges:
            if type(start) is not int or type(stop) is not int or start != end or stop < start:
                raise ValueError(f"invalid safetensors offsets: {name}")
            end = stop
        if end != info.st_size - 8 - header_size:
            raise ValueError(f"truncated safetensors file: {name}")
        return set(tensors)


def _weights(directory: int, name: str) -> set[str] | None:
    _nonempty(directory, name)
    if name.endswith(".safetensors"):
        return _safetensors(directory, name)
    elif name.endswith(".bin"):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        with os.fdopen(fd, "rb") as handle, zipfile.ZipFile(handle) as archive:
            files = archive.infolist()
            if not files or not any(f.filename.endswith("/data.pkl") and f.file_size for f in files):
                raise ValueError(f"invalid PyTorch weight archive: {name}")
        return None
    else:
        raise ValueError(f"unexpected weight format: {name}")


def _check_merged_fd(directory: int) -> dict[str, int]:
    for name in ("config.json", "tokenizer_config.json"):
        value = json.loads(_read_regular(directory, name))
        if not isinstance(value, dict):
            raise ValueError(f"invalid {name}")
    names = set(os.listdir(directory))
    tokenizer = next((x for x in ("tokenizer.model", "tokenizer.json") if x in names), None)
    if tokenizer is None:
        raise ValueError("merged model has no tokenizer")
    _nonempty(directory, tokenizer)
    indexes = [x for x in ("model.safetensors.index.json", "pytorch_model.bin.index.json") if x in names]
    singles = [x for x in ("model.safetensors", "pytorch_model.bin") if x in names]
    if len(indexes) + len(singles) != 1:
        raise ValueError("merged model must have exactly one weight set")
    mapping = None
    if indexes:
        index = json.loads(_read_regular(directory, indexes[0]))
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        if (not isinstance(mapping, dict) or not mapping
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items())):
            raise ValueError("invalid weight index")
        weight_files = sorted(set(mapping.values()))
        suffix = ".safetensors" if indexes[0].startswith("model.") else ".bin"
        if any(not name.endswith(suffix) for name in weight_files):
            raise ValueError("weight index mixes formats")
    else:
        weight_files = singles
    for name in weight_files:
        keys = _weights(directory, name)
        if mapping is not None and keys is not None and keys != {k for k, v in mapping.items() if v == name}:
            raise ValueError(f"weight index does not match shard: {name}")
    return {name: _nonempty(directory, name) for name in weight_files}


def check_merged(path: str | Path) -> dict[str, int]:
    """Validate config, tokenizer and all complete indexed/single weight files."""
    directory = _open_dir(_absolute(path))
    try:
        return _check_merged_fd(directory)
    finally:
        os.close(directory)


def _tree_size(directory: int, device: int | None = None) -> tuple[int, int]:
    device = os.fstat(directory).st_dev if device is None else device
    apparent = allocated = 0
    for name in os.listdir(directory):
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if info.st_dev != device:
            raise ValueError(f"refusing filesystem boundary in merged model: {name}")
        if stat.S_ISDIR(info.st_mode):
            fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            try:
                a, b = _tree_size(fd, device)
            finally:
                os.close(fd)
            apparent += a
            allocated += b
        elif stat.S_ISREG(info.st_mode):
            apparent += info.st_size
            allocated += info.st_blocks * 512
        else:
            raise ValueError(f"refusing linked or special entry in merged model: {name}")
    return apparent, allocated


def _prune(directory: int, device: int) -> None:
    """Delete inside an already opened directory without following symlinks."""
    for name in os.listdir(directory):
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if info.st_dev != device:
            raise ValueError(f"filesystem boundary appeared during deletion: {name}")
        if stat.S_ISDIR(info.st_mode):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            try:
                if (info.st_dev, info.st_ino) != (os.fstat(child).st_dev, os.fstat(child).st_ino):
                    raise ValueError(f"directory changed during deletion: {name}")
                _prune(child, device)
                current = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino):
                    raise ValueError(f"directory changed during deletion: {name}")
                os.rmdir(name, dir_fd=directory)
            finally:
                os.close(child)
        elif stat.S_ISREG(info.st_mode):
            os.unlink(name, dir_fd=directory)
        else:
            raise ValueError(f"linked or special entry appeared during deletion: {name}")


def _store_new(directory: int, name: str, value: dict[str, Any]) -> None:
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    with os.fdopen(fd, "wb") as handle:
        handle.write(_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.fsync(directory)


class FreshMergedRetention:
    def __init__(self, commands: list[Any], *, project: str | Path, ckpt_root: str | Path,
                 job_dir: str | Path, benchmark: str, method: str = "sefe"):
        self.project, self.root, self.job = map(_absolute, (project, ckpt_root, job_dir))
        if not _within(self.root, self.project) or not _within(self.job, self.project):
            raise ValueError("run paths must be inside this project")
        run_id = self.root.parent.parent.name
        if method not in {"sefe", "keeplora"} or self.root.parts[-2:] != (method, benchmark) or self.job.parts[-4:] != (run_id, "jobs", method, benchmark):
            raise ValueError("checkpoint and job paths are not the same fresh run")
        self.commands = commands
        self.plan = build_plan(commands, self.root, benchmark, method)
        jobfd = _open_dir(self.job)
        try:
            if "events.jsonl" in os.listdir(jobfd):
                raise ValueError("SEFE retention only registers before the first event")
        finally:
            os.close(jobfd)
        try:
            markerfd = _open_dir(self.job / ".mcitlib/steps")
        except FileNotFoundError:
            markerfd = None
        if markerfd is not None:
            try:
                if os.listdir(markerfd):
                    raise ValueError("SEFE retention refuses existing markers")
            finally:
                os.close(markerfd)
        rootfd = _open_dir(self.root, create=True)
        try:
            if os.listdir(rootfd):
                raise ValueError("SEFE retention requires an empty fresh checkpoint root")
            info = os.fstat(rootfd)
            self.root_identity = (info.st_dev, info.st_ino)
        finally:
            os.close(rootfd)
        self.receipts = self.job / (".mcitlib/" + method + "-retention")
        receiptfd = _open_dir(self.receipts, create=True)
        try:
            _store_new(receiptfd, "manifest.json", {**self.plan, "project": str(self.project),
                                                    "job_dir": str(self.job), "root_identity": self.root_identity,
                                                    "created_at": datetime.now(timezone.utc).isoformat()})
        finally:
            os.close(receiptfd)

    def _evidence(self, last: int) -> bool:
        jobfd = _open_dir(self.job)
        try:
            try:
                raw = _read_regular(jobfd, "events.jsonl")
            except FileNotFoundError:
                return False
        finally:
            os.close(jobfd)
        try:
            events = [json.loads(line) for line in raw.splitlines() if line.strip()]
        except (ValueError, UnicodeDecodeError):
            return False
        latest = {}
        for event in events:
            if not isinstance(event, dict) or type(event.get("index")) is not int:
                return False
            latest[event["index"]] = event
        try:
            markerfd = _open_dir(self.job / ".mcitlib/steps")
        except FileNotFoundError:
            return False
        try:
            names = os.listdir(markerfd)
            for i in range(last + 1):
                identity, command = self.plan["ids"][i], self.plan["commands"][i]
                event = latest.get(i, {})
                if (event.get("status") != "completed" or event.get("return_code") != 0
                        or event.get("id") != identity or event.get("command") != command or event.get("error")):
                    return False
                matching = [x for x in names if x == f"{identity}.json" or
                            (x.startswith(identity + "-") and x.endswith(".json"))]
                found = False
                for name in matching:
                    try:
                        marker = json.loads(_read_regular(markerfd, name))
                    except (OSError, ValueError, UnicodeDecodeError):
                        return False
                    if (marker.get("id") == identity and marker.get("index") == i
                            and marker.get("command") == command and marker.get("status") == "completed"
                            and marker.get("return_code") == 0 and not marker.get("validation_error")):
                        found = True
                if not found:
                    return False
            return True
        finally:
            os.close(markerfd)

    def before_step(self, index: int) -> list[dict[str, Any]]:
        if type(index) is not int or not 0 <= index <= len(self.commands):
            raise ValueError("invalid next command index")
        if hashlib.sha256(_json([_command(c) for c in self.commands])).hexdigest() != self.plan["commands_sha256"]:
            raise ValueError("worker command list changed after retention registration")
        done = []
        rootfd = _open_dir(self.root)
        try:
            info = os.fstat(rootfd)
            if (info.st_dev, info.st_ino) != self.root_identity:
                raise ValueError("fresh checkpoint root was replaced")
            for item in self.plan["entries"]:
                if item["final"] or index <= item["last_consumer"]:
                    continue
                receiptfd = _open_dir(self.receipts)
                receipt_name = f"task{item['task']}.json"
                try:
                    if receipt_name in os.listdir(receiptfd):
                        old = json.loads(_read_regular(receiptfd, receipt_name))
                        if (old.get("status") != "deleted" or old.get("path") != item["path"]
                                or old.get("commands_sha256") != self.plan["commands_sha256"]):
                            raise RuntimeError(f"unfinished or invalid retention receipt: {receipt_name}")
                        continue
                    if not self._evidence(item["last_consumer"]):
                        continue
                    successorfd = _open_beneath(rootfd, self.root, _absolute(item["next_path"]))
                    try:
                        successor = _check_merged_fd(successorfd)
                    finally:
                        os.close(successorfd)
                    target = _absolute(item["path"])
                    parentfd = _open_beneath(rootfd, self.root, target.parent)
                    try:
                        targetfd = os.open(target.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parentfd)
                        try:
                            apparent, allocated = _tree_size(targetfd)
                            identity = os.fstat(targetfd)
                            check = os.stat(target.name, dir_fd=parentfd, follow_symlinks=False)
                            if (identity.st_dev, identity.st_ino) != (check.st_dev, check.st_ino):
                                raise ValueError("merged directory changed before removal")
                            receipt = {"status": "pending", "path": str(target), "task": item["task"],
                                       "last_consumer": item["last_consumer"],
                                       "last_consumer_id": self.plan["ids"][item["last_consumer"]],
                                       "last_consumer_command": self.plan["commands"][item["last_consumer"]],
                                       "successor": item["next_path"], "successor_weight_bytes": successor,
                                       "apparent_bytes": apparent, "allocated_bytes": allocated,
                                       "prepared_at": datetime.now(timezone.utc).isoformat(),
                                       "commands_sha256": self.plan["commands_sha256"]}
                            _store_new(receiptfd, receipt_name, receipt)
                            _prune(targetfd, identity.st_dev)
                            check = os.stat(target.name, dir_fd=parentfd, follow_symlinks=False)
                            if (identity.st_dev, identity.st_ino) != (check.st_dev, check.st_ino):
                                raise ValueError("merged directory changed during removal")
                            os.rmdir(target.name, dir_fd=parentfd)
                            os.fsync(parentfd)
                        finally:
                            os.close(targetfd)
                    finally:
                        os.close(parentfd)
                    receipt["status"] = "deleted"
                    receipt["deleted_at"] = datetime.now(timezone.utc).isoformat()
                    temp = f"{receipt_name}.done"
                    _store_new(receiptfd, temp, receipt)
                    os.replace(temp, receipt_name, src_dir_fd=receiptfd, dst_dir_fd=receiptfd)
                    os.fsync(receiptfd)
                    done.append(receipt)
                finally:
                    os.close(receiptfd)
        finally:
            os.close(rootfd)
        return done
