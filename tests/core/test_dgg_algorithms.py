# SPDX-License-Identifier: Apache-2.0
"""CPU checks for DGG's replay, guidance and continuation behavior."""
import random

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from mcitlib.methods._dgg.core import GradientGuidance, image_paths_absolute, replay_rows  # noqa: E402


def test_dgg_guidance_caps_each_tensor_and_skips_zeros():
    first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    zero = torch.nn.Parameter(torch.tensor([9.0]))
    first.grad = torch.tensor([0.6, 0.8])
    second.grad = torch.tensor([3.0])
    zero.grad = torch.zeros(1)
    guidance = GradientGuidance({"first": torch.zeros(2), "second": torch.zeros(1), "zero": torch.zeros(1)}, 1.0)
    assert guidance.apply([("first", first), ("second", second), ("zero", zero)])
    torch.testing.assert_close(first.grad, torch.tensor([1.2, 1.6]))
    torch.testing.assert_close(second.grad, torch.tensor([4.0]))
    torch.testing.assert_close(zero.grad, torch.zeros(1))
    assert not GradientGuidance({"first": first}, 0).apply([("first", first)])
    with pytest.raises(KeyError, match="missing trainable"):
        guidance.apply([("unexpected", second)])

def _guided_steps(parameter, optimizer, guidance, rng, count):
    for _ in range(count):
        optimizer.zero_grad()
        ((parameter - 2) ** 2).sum().backward()
        guidance.apply([("weight", parameter)], rng=rng)
        optimizer.step()

def test_dgg_continuation_roundtrip_includes_anchor_optimizer_and_rng(tmp_path):
    p = torch.nn.Parameter(torch.tensor([0.3, 1.1]))
    optimizer = torch.optim.SGD([p], lr=0.13, momentum=0.8)
    guidance = GradientGuidance({"weight": torch.tensor([-0.4, 0.2])}, 0.5)
    rng = np.random.RandomState(920)
    _guided_steps(p, optimizer, guidance, rng, 4)
    checkpoint = {"p": p.detach().clone(), "optimizer": optimizer.state_dict(),
                  "guidance": guidance.state_dict(), "rng": rng.get_state()}
    torch.save(checkpoint, tmp_path / "dgg.pt")
    _guided_steps(p, optimizer, guidance, rng, 6)
    expected = p.detach().clone()
    saved = torch.load(tmp_path / "dgg.pt", map_location="cpu")
    resumed = torch.nn.Parameter(saved["p"])
    resumed_optimizer = torch.optim.SGD([resumed], lr=0.13, momentum=0.8)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    resumed_rng = np.random.RandomState()
    resumed_rng.set_state(saved["rng"])
    _guided_steps(resumed, resumed_optimizer, GradientGuidance.from_state_dict(saved["guidance"]), resumed_rng, 6)
    torch.testing.assert_close(resumed, expected)

def test_dgg_replay_independent_seed_and_original_qa(tmp_path):
    current = [{"image": "c.jpg", "conversations": ["current"]}]
    one = [{"image": f"one/{i}.jpg", "conversations": [f"QA {i}"]} for i in range(5)]
    two = [{"image": f"two/{i}.jpg", "conversations": [f"answer {i}"]} for i in range(3)]
    rows = replay_rows(current, [one, two], samples_per_task=2)
    assert rows == current + random.Random(1993).sample(one, 2) + random.Random(1993).sample(two, 2)
    resolved = image_paths_absolute(rows, tmp_path)
    assert resolved[0]["image"] == str(tmp_path / "c.jpg")
    assert resolved[1]["conversations"] == rows[1]["conversations"]
    assert current[0]["image"] == "c.jpg"

def test_dgg_compatibility_hook_order_accumulation_and_restoration():
    from mcitlib.methods._dgg.deepspeed_compat import optimizer_backward_hook
    events = []

    class Optimizer:
        def backward(self, microbatch):
            events.append(("backward", microbatch))

    class Engine:
        def __init__(self):
            self.optimizer = Optimizer()

        def backward(self, microbatch):
            self.optimizer.backward(microbatch)
            events.append(("reduce", microbatch))

    engine = Engine()
    with pytest.raises(RuntimeError, match="stop"):
        with optimizer_backward_hook(engine, lambda _: events.append(("guidance", None))):
            for microbatch in range(3):
                engine.backward(microbatch)
            raise RuntimeError("stop")
    assert events == [("backward", 0), ("guidance", None), ("reduce", 0),
                      ("backward", 1), ("guidance", None), ("reduce", 1),
                      ("backward", 2), ("guidance", None), ("reduce", 2)]
    engine.backward(3)
    assert events[-2:] == [("backward", 3), ("reduce", 3)]
    assert "backward" not in vars(engine.optimizer)
