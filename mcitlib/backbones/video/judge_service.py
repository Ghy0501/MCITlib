# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Serve the CL-VISTA judge on a private Unix socket, one prediction cell at a time."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socketserver
from contextlib import contextmanager
from pathlib import Path

from .native_judge import BATCH_SIZE, load_judge, read_predictions, score_rows


class Judge(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            self.connection.settimeout(30)
            line = self.rfile.readline(16385)
            if len(line) > 16384 or not line.endswith(b"\n"):
                raise ValueError("Expected one newline-terminated judge request (maximum 16384 bytes)")
            request = json.loads(line)
            predictions = Path(request["predictions"]).resolve(strict=True)
            if (self.server.results_root not in predictions.parents or predictions.name != "predictions.jsonl"
                    or not predictions.is_file()):
                raise ValueError("Expected a predictions.jsonl under the configured results root")
            rows = read_predictions(predictions)
            output = predictions.parent / "native-score"
            output.mkdir(mode=0o700)
            result_file = output / "results.txt"
            try:
                count = score_rows(rows, *self.server.judge, output_dir=output / "qwen",
                                   output_json=output / "scores.json", result_path=result_file,
                                   batch_size=self.server.batch_size)
            except Exception:
                # Only remove the fresh output owned by this request. A retry starts clean.
                shutil.rmtree(output)
                raise
            result = {"status": "scored", "count": count, "result": str(result_file)}
        except Exception as exc:
            result = {"status": "failed", "reason": str(exc)}
        self.wfile.write((json.dumps(result) + "\n").encode())


@contextmanager
def open_server(runtime_root, results_root, *, batch_size=BATCH_SIZE):
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    runtime = Path(runtime_root).expanduser().resolve()
    runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
    if runtime.stat().st_uid != os.getuid() or runtime.stat().st_mode & 0o022:
        raise PermissionError("Runtime root must be owned by the current user and not writable by others")
    control = runtime / "control"
    control.mkdir(mode=0o700, exist_ok=True)
    if control.is_symlink() or control.stat().st_uid != os.getuid() or control.stat().st_mode & 0o077:
        raise PermissionError("Use a private (0700), user-owned runtime control directory")
    root = Path(results_root).expanduser().resolve()
    sock = control / "judge.sock"
    # Bind a short relative address, even when the runtime path exceeds sun_path.
    # Do this at startup, before serving requests; bind itself refuses occupied names.
    previous = Path.cwd()
    try:
        os.chdir(control)
        server = socketserver.UnixStreamServer("judge.sock", Judge)
    finally:
        os.chdir(previous)
    with server:
        bound = sock.lstat()
        try:
            sock.chmod(0o600)
            server.results_root = root
            server.batch_size = batch_size
            yield server
        finally:
            try:
                if os.path.samestat(bound, sock.lstat()):
                    sock.unlink()
            except FileNotFoundError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path, help="Parent of control/judge.sock")
    parser.add_argument("--results-root", type=Path, help="Authorized predictions tree (default: RUNTIME_ROOT/results)")
    parser.add_argument("--model-path", required=True, type=Path, help="Trusted local causal LM with a chat template")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--attn-implementation", help="Optional Transformers attention backend, e.g. flash_attention_2")
    args = parser.parse_args()
    results = args.results_root or args.runtime_root / "results"
    with open_server(args.runtime_root, results, batch_size=args.batch_size) as server:
        server.judge = load_judge(args.model_path, attn_implementation=args.attn_implementation)
        print(json.dumps({"event": "judge_ready", "model": str(args.model_path)}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
