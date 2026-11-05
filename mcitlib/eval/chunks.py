# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Merge chunked inference output exactly like v3's ``cat`` loop (G3).

Every v3 evaluation script ran ``CHUNKS`` inference processes, each writing
``<dir>/<CHUNKS>_<IDX>.jsonl``, and then did::

    > <dir>/merge.jsonl
    for IDX in 0..CHUNKS-1: cat <dir>/<CHUNKS>_<IDX>.jsonl >> <dir>/merge.jsonl

before calling the scorer.  Those lines are shell built-ins, so the command recorder
never saw them; this module restores the step as an explicit, testable command.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from ..config.schema import Command

CHUNK_NAME = re.compile(r"^(?P<chunks>\d+)_(?P<index>\d+)\.jsonl$")
MERGE_MODULE = "mcitlib.eval.chunks"


def merge_chunks(output: str | Path, inputs: list[str | Path]) -> int:
    """Truncate ``output`` and append every input byte-for-byte, in the given order."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in inputs if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("chunk output missing (inference failed or was skipped): " + ", ".join(missing))
    written = 0
    with output.open("wb") as handle:
        for path in inputs:
            data = Path(path).read_bytes()
            handle.write(data)
            written += len(data)
    return written


def discover_chunks(directory: str | Path) -> list[Path]:
    """The one complete ``<N>_0..<N>_{N-1}.jsonl`` set in ``directory``; anything else is an error.

    Used when the inference that wrote the chunks is not part of the same command sequence
    (v3's MR-LoRA MLLM-DCL evaluation script has its inference commented out and only merges
    and scores).  v3 would silently ``cat`` whatever happened to exist; here a missing or
    ambiguous set stops the run with an explanation.
    """
    directory = Path(directory)
    found: dict[int, dict[int, Path]] = {}
    if directory.is_dir():
        for path in directory.iterdir():
            match = CHUNK_NAME.match(path.name)
            if match:
                found.setdefault(int(match["chunks"]), {})[int(match["index"])] = path
    complete = [n for n, files in found.items() if sorted(files) == list(range(n))]
    if len(complete) != 1:
        detail = "no chunk files" if not found else f"chunk sets found: {sorted(found)}; complete: {sorted(complete)}"
        raise FileNotFoundError(
            f"cannot merge in {directory}: expected exactly one complete <N>_<i>.jsonl set ({detail}). "
            "Run the inference that produces them first, or remove stale chunk files."
        )
    count = complete[0]
    return [found[count][index] for index in range(count)]


def _answers_file(command: Command) -> str | None:
    argv = command.argv
    return argv[argv.index("--answers-file") + 1] if "--answers-file" in argv else None


def insert_merge_steps(commands: list[Command]) -> list[Command]:
    """Restore v3's ``cat`` merge step in front of every scorer that reads ``merge.jsonl``.

    ``commands`` are in v3 execution order.  Two cases:

    * the chunked inference is part of the sequence: one merge with the explicit chunk list is
      added right after the last chunk command of each group;
    * the scorer reads chunks that no command of this sequence writes: a ``--discover`` merge
      is added in front of the scorer, which fails with an explanation unless exactly one
      complete chunk set is present.
    """
    result: list[Command] = []
    group: list[tuple[str, Command]] = []
    merged: set[tuple[int | None, str]] = set()

    def new_merge(template: Command, directory: str, order: float, *tail: str) -> Command:
        merged.add((template.task_id, directory))
        return Command(
            argv=["python", "-m", MERGE_MODULE, "merge", "--output", str(Path(directory) / "merge.jsonl"), *tail],
            cwd=template.cwd,
            env={key: value for key, value in template.env.items() if key in {"MCITLIB_METHOD", "PYTHONPATH"}},
            kind=template.kind,
            task_id=template.task_id,
            sequence=template.sequence,
            metadata={"order": order, "merge_chunks": True, "source": "v3-cat-loop"},
        )

    def flush() -> None:
        if not group:
            return
        directory = str(Path(group[0][0]).parent)
        counts = {int(CHUNK_NAME.match(Path(path).name)["chunks"]) for path, _ in group}
        if len(counts) != 1:
            raise ValueError(f"inconsistent chunk counts in {directory}: {sorted(counts)}")
        count = counts.pop()
        by_index = {int(CHUNK_NAME.match(Path(path).name)["index"]): path for path, _ in group}
        if sorted(by_index) != list(range(count)):
            raise ValueError(f"{directory}: expected chunks 0..{count - 1}, found {sorted(by_index)}")
        last = max(command.metadata.get("order", -1) for _, command in group)
        result.append(new_merge(
            group[-1][1], directory, last + 0.5, "--inputs", *[by_index[index] for index in range(count)],
        ))
        group.clear()

    for command in commands:
        path = _answers_file(command)
        if path is not None and CHUNK_NAME.match(Path(path).name):
            if group and (
                str(Path(path).parent) != str(Path(group[0][0]).parent) or command.task_id != group[0][1].task_id
            ):
                flush()
            group.append((path, command))
        else:
            flush()
            result_file = _option(command.argv, "--result-file")
            if result_file and Path(result_file).name == "merge.jsonl":
                directory = str(Path(result_file).parent)
                if (command.task_id, directory) not in merged:
                    result.append(new_merge(
                        command, directory, command.metadata.get("order", -1) - 0.5, "--discover", directory,
                    ))
        result.append(command)
    flush()
    return result


def _option(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=MERGE_MODULE)
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--output", required=True)
    source = merge.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", nargs="+")
    source.add_argument("--discover", metavar="DIR", help="merge the single complete chunk set found in DIR")
    args = parser.parse_args(argv)
    try:
        inputs = args.inputs if args.inputs else discover_chunks(args.discover)
        size = merge_chunks(args.output, inputs)
    except FileNotFoundError as error:
        print(f"mcitlib: {error}", file=sys.stderr)
        return 2
    print(f"merged {len(inputs)} chunk files ({size} bytes) into {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
