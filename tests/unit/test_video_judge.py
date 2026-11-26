# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CL-VISTA native scoring contracts, without third-party checkpoints."""
import hashlib
import json
import socket
import stat
import subprocess
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcitlib.backbones.video import judge_service, native_judge, score_client


def row(key):
    return {"id": key, "question": "Q", "answer": "A", "pred": "P"}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.mark.parametrize("rows", [
    [], [row(1), row("1")], [row(None)], [row(True)], [row("")], [row("../x")],
    [row("..")], [row("a\0b")], [row("a\\b")], [dict(row(1), question_id="1")],
    [{"id": 1, "question": "Q", "answer": "A"}], [42],
])
def test_invalid_rows_fail_before_inference(tmp_path, rows):
    with pytest.raises(ValueError):
        native_judge.read_predictions(write_rows(tmp_path / "p.jsonl", rows))


@pytest.mark.parametrize(("reply", "score"), [
    (" ```\n{'score': 4}\n``` ", 4), ('{"score":"3"}', 3), ("{'score': 3.9}", 3),
    ("{'score': 10}", 5), ("{'score': -1}", 0), ("{}", 0), ("bad output", 0),
])
def test_historical_response_parser(reply, score):
    assert native_judge.parse_score(reply) == score


class Batch(dict):
    def to(self, device):
        assert device == "cpu"
        return self


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        assert [m["role"] for m in messages] == ["system", "user"]
        # Hash the actual service prompts for Q/A/P, including whitespace.
        assert [hashlib.sha256(m["content"].encode()).hexdigest() for m in messages] == [
            "8e50993da59e239755c0974f3387d62e7f2eeb8ce5bf845c4a33161e4e790bf7",
            "d218e171a4dea7b2034ef9fdafb054c0ec90ce8189e282c0663c230f3caf7054",
        ]
        return "chat\udfff"

    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        assert padding and truncation and max_length == 4096 and return_tensors == "pt"
        assert texts == ["chat?"] * len(texts)
        return Batch(input_ids=[[1, 2]] * len(texts))

    def decode(self, tokens, *, skip_special_tokens):
        assert len(tokens) == 1 and skip_special_tokens
        return {3: "{'score': 4}", 4: "invalid"}[tokens[0]]


class Model:
    device = "cpu"

    def generate(self, *, input_ids, max_new_tokens):
        assert max_new_tokens == 128
        return [ids + [3 + i] for i, ids in enumerate(input_ids)]


def test_socket_roundtrip_and_no_clobber(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    runtime = tmp_path / "runtime"
    results = tmp_path / "private experiments"
    preds = write_rows(results / "cell/predictions.jsonl", [row(12), row("abc")])
    outside = write_rows(tmp_path / "outside/predictions.jsonl", [row(0)])
    link = results / "redirect"
    link.symlink_to(outside.parent, target_is_directory=True)
    control = runtime / "control"

    def client(path):
        result = subprocess.run([sys.executable, score_client.__file__, control, path],
                                capture_output=True, text=True, timeout=10)
        reply = json.loads(result.stdout)
        assert result.returncode == int(reply["status"] != "scored")
        return reply

    with judge_service.open_server(runtime, results) as server:
        server.judge = (Model(), Tokenizer())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            sock = control / "judge.sock"
            assert stat.S_IMODE(sock.stat().st_mode) == 0o600
            with pytest.raises(OSError), judge_service.open_server(runtime, results):
                pytest.fail("second service replaced the first")
            with monkeypatch.context() as local:
                local.chdir(control)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as raw:
                    raw.settimeout(3)
                    raw.connect("judge.sock")
                    # Exercise the service's own path resolution, independent of the client.
                    raw.sendall((json.dumps({"predictions": str(link / "predictions.jsonl")}) + "\n").encode())
                    with raw.makefile("rb") as handle:
                        assert json.loads(handle.readline())["status"] == "failed"
            assert client(link / "predictions.jsonl")["status"] == "failed"
            assert not (outside.parent / "native-score").exists()
            assert client(write_rows(results / "other.jsonl", [row(1)]))["status"] == "failed"
            out = preds.parent / "native-score"
            with monkeypatch.context() as local:
                local.setattr(Model, "generate", lambda *a, **kw: [[1, 2, 3]])
                assert client(preds)["status"] == "failed"
            assert not out.exists()
            reply = client(preds)
            assert reply == {"status": "scored", "count": 2, "result": str(out / "results.txt")}
            assert Path(reply["result"]).read_text().splitlines() == [
                "Original Average Score (0-5): 2.0000", "Normalized Average Score (0-100): 40.0000",
                "Number of samples: 2",
            ]
            scores = json.loads((out / "scores.json").read_text())
            qa = {"q": "Q", "a": "A", "pred": "P"}
            assert scores == {"12": [{"score": 4}, qa], "abc": [{"score": 0}, qa]}
            assert json.loads((out / "qwen/12.json").read_text()) == scores["12"]
            assert client(preds)["status"] == "failed"
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert not sock.exists()


def test_refuse_existing_control_entry(tmp_path):
    path = tmp_path / "control/judge.sock"
    path.parent.mkdir(mode=0o700)
    path.write_text("keep")
    with pytest.raises(OSError), judge_service.open_server(tmp_path, tmp_path / "results"):
        pytest.fail("occupied address accepted")
    assert path.read_text() == "keep"


def test_refuse_public_or_redirected_control(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    control.chmod(0o777)
    with pytest.raises(PermissionError), judge_service.open_server(tmp_path, tmp_path / "results"):
        pytest.fail("public control directory accepted")
    control.rmdir()
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    control.symlink_to(target, target_is_directory=True)
    with pytest.raises(PermissionError), judge_service.open_server(tmp_path, tmp_path / "results"):
        pytest.fail("redirected control directory accepted")
    assert list(target.iterdir()) == []


def test_checkpoint_loader(monkeypatch, tmp_path):
    calls = []
    model = SimpleNamespace(generation_config=SimpleNamespace(), eval=lambda: None)
    tokenizer = SimpleNamespace(eos_token="<end>")

    def loader(value):
        def load(path, **kwargs):
            assert Path(path) == tmp_path
            calls.append(kwargs)
            return value
        return SimpleNamespace(from_pretrained=load)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=loader(model), AutoTokenizer=loader(tokenizer)))
    assert native_judge.load_judge(tmp_path) == (model, tokenizer)
    assert all(c["local_files_only"] and not c.get("trust_remote_code") for c in calls)
    assert "attn_implementation" not in calls[1]
    assert tokenizer.padding_side == "left" and tokenizer.pad_token == "<end>"
    cfg = model.generation_config
    assert not cfg.do_sample and cfg.num_beams == 1
    assert cfg.temperature is None and cfg.top_k is None and cfg.top_p is None
