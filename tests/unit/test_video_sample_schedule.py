# SPDX-License-Identifier: Apache-2.0
"""Check the CL-VISTA sample schedule and method-specific training stages on CPU."""
import json
from pathlib import Path
from unittest import TestCase

from mcitlib.backbones.video.workflow import build_cl_vista_plan
from mcitlib.registry import CL_VISTA

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("counting", "space", "traffic", "movie", "gui", "science", "sports", "star")
METHODS = ("lora_ft", "pclr", "octopus")


def sample_plan(method, workers=4):
    tasks = {
        domain: {
            "sample": {"train_path": f"{domain}/train", "test_q_path": f"{domain}/questions",
                       "test_a_path": f"{domain}/answers"},
            "full": {"train_path": f"{domain}/full"},
        }
        for domain in TASKS
    }
    node = {"tasks": tasks, "models": {"videollava": {"model_view": "base", "towers": {"mm_video_tower": "tower"}}},
            "media": "media", "checkpoints": "checkpoints", "results": "results"}
    return build_cl_vista_plan(ROOT / "configs", node, "videollava", method, "sample", "example",
                               gpu_count=workers, judge_directory="judge")


def arg(command, flag):
    return command.argv[command.argv.index("--" + flag) + 1]


class TestVideoSampleSchedule(TestCase):
    def test_generate_then_score_each_seen_domain_after_training(self):
        self.assertEqual(tuple(CL_VISTA.tasks), TASKS)
        for method in METHODS:
            with self.subTest(method=method):
                plan = sample_plan(method)
                task_ids = [c.task_id for c in plan]
                self.assertEqual(task_ids, sorted(task_ids))
                for task in range(1, len(TASKS) + 1):
                    stages = [c for c in plan if c.task_id == task]
                    cells = [c for c in stages if c.kind in {"generate", "score"}]
                    seen = TASKS[:task]
                    self.assertEqual([(c.kind, c.metadata["eval_domain"]) for c in cells],
                                     [(kind, domain) for domain in seen for kind in ("generate", "score")])
                    self.assertLess(max(i for i, c in enumerate(stages) if c.kind == "train"),
                                    stages.index(cells[0]))
                    for domain, generate, score in zip(seen, cells[::2], cells[1::2], strict=True):
                        pred = f"results/sample/videollava/{method}/example/task{task}/{domain}/predictions.jsonl"
                        self.assertEqual(arg(generate, "predictions"), pred)
                        self.assertEqual(score.argv[-1], pred)
                        self.assertEqual(arg(generate, "questions"), f"{domain}/questions")
                        self.assertEqual(arg(generate, "answers"), f"{domain}/answers")

    def test_training_stages_and_parameters(self):
        for method in METHODS:
            train = [c for c in sample_plan(method) if c.kind == "train"]
            expected = []
            for task in range(1, 9):
                expected.append((task, "octopus_stage1" if method == "octopus" else "train"))
                if method == "pclr" and task >= 4:
                    expected.append((task, "pclr_integration"))
                if method == "octopus" and task >= 2:
                    expected.append((task, "octopus_stage2"))
            self.assertEqual([(c.task_id, c.sequence) for c in train], expected)
            for c in train:
                task = c.task_id
                with self.subTest(method=method, task=task, stage=c.sequence):
                    rank = {"lora_ft": 32, "octopus": 48, "pclr": 64 if task <= 6 else 32}[method]
                    self.assertEqual((int(arg(c, "lora_r")), int(arg(c, "lora_alpha"))), (rank, 2 * rank))
                    self.assertEqual(arg(c, "lora_enable"), "False" if method == "pclr" else "True")
                    lr = 1e-4 if method == "octopus" else 5e-5 if c.sequence == "pclr_integration" else 2e-4
                    self.assertEqual(float(arg(c, "learning_rate")), lr)
                    self.assertEqual(float(arg(c, "warmup_ratio")), .05 if method == "octopus" else .03)
                    self.assertEqual(float(arg(c, "num_train_epochs")), 1)
                    self.assertEqual(int(arg(c, "per_device_train_batch_size")), 1)
                    self.assertEqual(int(arg(c, "gradient_accumulation_steps")), 16)
                    if method == "pclr":
                        options = json.loads(arg(c, "method-options"))
                        self.assertEqual(options, {
                            "stage": "integrate" if c.sequence == "pclr_integration" else "learn",
                            "task_id": task,
                            "pool": {"new_rank": rank, "top_rank": 64, "alpha": 128, "share_rank": 32,
                                     "skip_interval": 4 if task < 6 else 8},
                        })
                    if method == "octopus":
                        options = json.loads(arg(c, "method-options"))
                        self.assertEqual(options["stage"], c.sequence.removeprefix("octopus_"))
                        if options["stage"] == "stage2":
                            self.assertEqual((options["lambda_1"], options["lambda_2"]), (.02, .01))
                            self.assertEqual(len(options["history"]), task - 1)

    def test_octopus_execution_profile_and_auxiliary_stages(self):
        plan = sample_plan("octopus", workers=2)
        for task in range(1, 9):
            with self.subTest(task=task):
                cmds = [c for c in plan if c.task_id == task]
                for c in cmds:
                    if c.kind == "train":
                        self.assertEqual(int(arg(c, "gradient_accumulation_steps")), 32)
                        self.assertEqual(int(arg(c, "per_device_train_batch_size")), 1)
                stages = [c.kind for c in cmds if c.kind not in {"generate", "score"}]
                self.assertEqual(stages, ["train", "octopus_first_copy"] if task == 1 else [
                    "train", "octopus_budget", "octopus_budget", "octopus_gpwc", "train", "octopus_cleanup",
                ])
                budgets = [(int(arg(c, "budget")), int(arg(c, "batch"))) for c in cmds if c.kind == "octopus_budget"]
                self.assertEqual(budgets, [] if task == 1 else [(9600, 64), (256, 16)])
