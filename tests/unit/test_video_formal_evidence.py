# SPDX-License-Identifier: Apache-2.0
"""Compare CPU-only release plans with the audited CL-VISTA sample runs."""
import json
from collections import Counter
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
    return build_cl_vista_plan(ROOT / "configs", node, "videollava", method, "sample", "audit",
                               gpu_count=workers, judge_directory="judge")


def arg(command, flag):
    return command.argv[command.argv.index("--" + flag) + 1]


class TestFormalSampleEvidence(TestCase):
    def test_plan_matches_receipt_cells_and_saved_plan_counts(self):
        evidence = json.loads((ROOT / "docs/video-validation.json").read_text())
        self.assertEqual(tuple(evidence["task_order"]), TASKS)
        self.assertEqual(tuple(CL_VISTA.tasks), TASKS)
        self.assertEqual(evidence["native_denominator_per_cell"], 1000)
        self.assertEqual(set(evidence["plan_stage_counts"]), set(METHODS))
        coverage = evidence["coverage_by_training_task"]
        self.assertEqual(coverage, list(range(1, 9)))
        self.assertEqual(len(coverage), evidence["training_tasks_per_run"])
        expected = [(i, domain) for i, n in enumerate(coverage, 1) for domain in TASKS[:n]]
        self.assertEqual(len(expected), evidence["scored_cells_per_run"])
        self.assertEqual(len(expected) * evidence["native_denominator_per_cell"],
                         evidence["scored_predictions_per_run"])
        for method in METHODS:
            with self.subTest(method=method):
                plan = sample_plan(method)
                stages = Counter(c.sequence if c.kind == "train" else c.kind for c in plan)
                self.assertEqual(dict(stages), evidence["plan_stage_counts"][method])
                for kind in ("generate", "score"):
                    cells = [c for c in plan if c.kind == kind]
                    self.assertEqual([(c.task_id, c.metadata["eval_domain"]) for c in cells], expected)
                    for (task, domain), cell in zip(expected, cells):
                        pred = f"results/sample/videollava/{method}/audit/task{task}/{domain}/predictions.jsonl"
                        if kind == "score":
                            self.assertEqual(cell.argv[-1], pred)
                        else:
                            self.assertEqual(arg(cell, "predictions"), pred)
                            self.assertEqual(arg(cell, "questions"), f"{domain}/questions")
                            self.assertEqual(arg(cell, "answers"), f"{domain}/answers")

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
