# MCITlib: Multimodal Continual Instruction Tuning

MCITlib provides unified configuration, training, evaluation, and reporting across continual learning methods. LLaVA-1.5 supports UCIT, MLLM-DCL, MLLM-ACL, and general image evaluation. InternVL-Chat-ViT-6B-Vicuna-7B supports UCIT and MLLM-DCL. Video-LLaVA and VideoLLaMA2 support CL-VISTA training, held-out generation, and local judge scoring.

**New machine?** [Start from a public checkout](docs/public-start.md): choose one of the four backbones, download its pinned public inputs, and run training and inference on a tiny example before preparing a full benchmark.

**Running video?** The [CL-VISTA runtime guide](docs/video-runtime.md) covers model and data layout, training, generation, and scoring for both video backbones.

```text
mcitlib/      Python package and CLI
configs/      Methods, benchmarks, training profiles, and path templates
third_party/  Backbone and adapter source
assets/       Method resources
envs/         Pinned runtime requirements
tests/        CPU and model integration tests
tools/        Configuration, sampling, and validation utilities
docs/         Usage and implementation contracts
```

## Installation

For image backbones, use Python 3.10. Choose the requirements for the method and install each runtime in a separate environment:

| Methods | Requirements |
| --- | --- |
| Most LLaVA and InternVL methods | `envs/llava.txt` |
| SEFE, KeepLoRA | `envs/llava-t437.txt` |
| ModalPrompt | `envs/llava-modalprompt-ds126.txt` |

Install a compatible PyTorch/CUDA and flash-attention build, then run from the project root:

```bash
python -m pip install -r envs/llava.txt
python -m pip install -e .
mcitlib list
```

Each image profile includes [InternVL's pinned vision dependency](envs/internvl-extra.txt). It is not a dependency of the lightweight CLI installation.

## Paths and outputs

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

Set `model_root`, `data_root`, `ckpt_root`, and input mappings for the machine. Relative paths resolve from the project root. `MCITLIB_MODEL_ROOT`, `MCITLIB_DATA_ROOT`, and `MCITLIB_CKPT_ROOT` override these roots. Weights may be stored outside the project.

Logs and resume markers default to `logs/<benchmark>/<backbone>/<method>/`. `logs_root`, `MCITLIB_LOGS_ROOT`, and `--run-dir` must resolve inside the project. LLaVA evaluation results default to `results/<method>/`; InternVL results default to `results/internvl/<method>/`. `results_root` or `MCITLIB_RESULTS_ROOT` can change the results destination. Local paths, datasets, weights, logs, results, and reports are excluded from Git.

For InternVL, place `InternVL-Chat-ViT-6B-Vicuna-7B`, `InternViT-6B-224px`, and `clip-vit-large-patch14-336` under `model_root`. The LLaVA-format language model uses InternViT at 336 pixels; DISCO, HiDe, and ModalPrompt use the separate CLIP tower for guidance.

## Validate and run

```bash
mcitlib config validate --json
mcitlib config expand --method lora_ft --benchmark ucit --task 1

mcitlib doctor --method lora_ft --benchmark ucit --validation-fixes
mcitlib train --method lora_ft --benchmark ucit --validation-fixes --dry-run
mcitlib eval --method lora_ft --benchmark ucit --after-task 1 --validation-fixes --dry-run

mcitlib train --method lora_ft --benchmark ucit --validation-fixes
mcitlib train --method lora_ft --benchmark ucit --validation-fixes --resume
mcitlib metrics --method lora_ft --benchmark ucit
```

Select InternVL explicitly with `--backbone internvl` for doctor, train, eval, and metrics. For example:

```bash
mcitlib doctor --backbone internvl --method lora_ft --benchmark ucit --validation-fixes
mcitlib train --backbone internvl --method lora_ft --benchmark ucit --validation-fixes --dry-run
```

See the [InternVL implementation](docs/internvl.md) for model geometry, method behavior, and prerequisites.

Use `--validation-fixes` consistently with doctor, training, and evaluation to select the complete task and routing plans. Replay requires original task QA for its memory. MR-LoRA requires expert training and a separate router sequence; routed scores use `UCIT_router` or `DCL_router`. Its fixed expert pool is a retrospective evaluation protocol.

OLoRA uses non-reentrant checkpointing. DISCO and HiDe reuse constant expert weights within each generation request. See the [implementation contract](docs/llava_implementation.md) and [training behavior](docs/training_entry_repairs.md).

PCLR, SAME, SiGMA, Octopus, and DGG support the LLaVA backbone on UCIT and MLLM-DCL. Their guides describe method behavior, budgets, and checkpoint dependencies: [PCLR](docs/pclr.md), [SAME and SiGMA](docs/same_sigma_provenance.md), [Octopus](docs/octopus.md), and [DGG](docs/dgg.md). These five methods do not support InternVL or MLLM-ACL.

## Sample validation and reports

The [sample recipe](configs/validation_3k_1k/README.md) selects up to 3,000 training and 1,000 evaluation examples per task without moving images. The queue checks each method's prerequisites, isolates outputs, and continues to the next method after a failed job. See [InternVL sample validation](docs/internvl_sample_validation.md) for its four-GPU queue and hardware checks. A constructed plan or a CPU check does not imply that a GPU run has completed.

```bash
python -m mcitlib.summary \
  --run-dir logs/validation_3k_1k/<run-id> \
  --output-dir reports/<run-id>-summary
```

Reports include the result matrix, MFT, MFN, MAA, BWT, and recorded training budgets. Missing values remain unavailable. See [configuration](docs/configuration.md), [summaries](docs/summary.md), and the [documentation index](DOCUMENTATION.md).

## CPU checks

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/core tests/unit/test_config_templates.py \
  tests/unit/test_registry_and_cli.py tests/unit/test_runner_execution.py \
  tests/unit/test_metrics.py tests/unit/test_four_gpu_validation.py
ruff check mcitlib tools tests
python tools/check_third_party.py
```

CPU checks validate configuration and orchestration. Model integration tests need the matching runtime; GPU execution also requires a passing doctor. See [CI](docs/ci.md).

MCITlib accompanies [Multimodal Continual Instruction Tuning Library and Benchmark](https://arxiv.org/abs/2508.07307) and is distributed under the Apache-2.0 license in [LICENSE](LICENSE).
