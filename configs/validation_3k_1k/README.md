# UCIT and MLLM-DCL fixed validation samples

This recipe makes 3,000 training and 1,000 held-out evaluation records per task: six UCIT tasks and five MLLM-DCL tasks. It samples each source split separately without replacement. The existing **test** split is the source of the held-out evaluation subset; it is not a new dataset split. The source files and sampled annotations are JSON arrays, as required by the existing loaders, not JSONL. All record fields, IDs and image strings are preserved, and images are neither copied nor moved.

From the project root, prepare `configs/paths.yaml` with the original data and model mappings, and use an environment with PyYAML and Pillow installed:

```bash
python tools/sample_validation_data.py
python tools/sample_validation_data.py --verify
```

Generation refuses to overwrite any existing sample directory, manifest or validation paths file. It stages the data and uses Linux `renameat2` to publish the completed directory without replacing an existing one. The expected outputs are:

- `data/validation_3k_1k/`: ignored local sample JSONs and caption references.
- `configs/validation_3k_1k/manifest.json`: reproducible indices, source and output SHA-256 values, source and sample strata, image checks, and caption-reference mappings. Generate it locally with the sampler and keep it with the corresponding dataset snapshot.
- `configs/paths.validation.yaml`: ignored machine-specific configuration for the existing `--paths` option. It inherits original UCIT/DCL image and model mappings and isolates checkpoints, results and logs under `validation_3k_1k`.

`--verify` rebuilds the expected subset in temporary storage and compares the complete manifest, validation profile and sample files. Both generation and verification check that every selected image exists and fully decode the first eight selected rows of each task/split. Use the same source `paths.yaml`, sources and recipe to verify. Existing `MCITLIB_*_ROOT` environment variables override the corresponding roots in path configurations; set them consistently or unset them when using this profile.

The seed is 20261106. A SHA-256-derived seed for each benchmark/task/split ranks zero-based source indices without replacement. Integer proportional quotas preserve each nonempty stratum when the sample size permits. ImageNet-R uses its image class directory; IconQA uses its two question directories; Sci uses the four source prefixes; caption tasks use caption word-count groups; the other tasks use broad answer-kind groups. The manifest records the recipe checksum, derived seeds, counts and selected indices. Within a task, candidate test rows are excluded when their image string and question match a selected training row after removing `<image>` and collapsing whitespace. Reused IDs alone do not cause exclusion, and multiple training captions for an image remain eligible.

For VizWiz and Flickr30k, each sampled test image is matched to the original COCO reference by `file_name`. Sample image IDs follow the selected test order, so the existing caption evaluator pairs predictions and references correctly. Every image retains all five source captions. Some original annotation IDs repeat; the sampled reference assigns unique local annotation IDs for the scorer. The manifest identifies source images and the original annotation **row indices**, alongside the new IDs, rather than assuming original annotation IDs are unique.

After generation, for example, inspect readiness and preview commands without starting training or using a GPU:

```bash
python -m mcitlib.cli doctor --method lora_ft --benchmark ucit --paths configs/paths.validation.yaml --skip-gpu --skip-entrypoints
python -m mcitlib.cli doctor --method lora_ft --benchmark mllm_dcl --paths configs/paths.validation.yaml --skip-gpu --skip-entrypoints
python -m mcitlib.cli train --method lora_ft --benchmark ucit --paths configs/paths.validation.yaml --dry-run
python -m mcitlib.cli train --method lora_ft --benchmark mllm_dcl --paths configs/paths.validation.yaml --dry-run
python -m mcitlib.cli eval --method lora_ft --benchmark ucit --after-task 1 --paths configs/paths.validation.yaml --dry-run
python -m mcitlib.cli eval --method lora_ft --benchmark mllm_dcl --after-task 1 --paths configs/paths.validation.yaml --dry-run
```

Doctor may still report missing model files, prerequisites or other environment issues; the profile does not assert training readiness. The same train-file mappings apply to KeepLoRA's auxiliary data-config reads. Replay's mixed training inputs and MR-LoRA's router inputs require their own construction policies and are **not generated** by this recipe. Their corresponding paths point to absent files in the isolated sample directory, so those sequences fail input checks until independently prepared. MLLM-ACL and general benchmarks are outside this recipe and their data roots point to absent isolated locations. This profile does not imply that every method or benchmark is ready for a complete run.

## Serial validation queue

The queue runs all 12 LLaVA methods on UCIT and MLLM-DCL as 24 independent jobs. Each job runs the full GPU and entry-point doctor in its selected Python environment first; MR-LoRA checks both the expert and router sequences. A failing doctor blocks that job. Training, inference, scoring and metrics run sequentially, and a failed step or missing checkpoint stops that job before any later step. The remaining jobs continue. ModalPrompt runs its separately recorded inference after each task. For validation the queue explicitly repairs ModalPrompt UCIT's task-2 output path, the missing MR-LoRA DCL expert inference and router scoring, and KeepLoRA DCL's missing task-2 Med evaluation. Use `--validation-fixes` consistently when inspecting and executing the complete task plans.

MR-LoRA DCL keeps expert cross-results under `DCL/` and writes selected router predictions and scores under `DCL_router/`; queue metrics use the latter. Its fixed five-expert routing pool can select experts trained in later stages, so this is retrospective rather than a strictly online continual-learning result. Router fallbacks follow the reference implementation and are recorded. The sample profile's DCL test counts and hashes are checked against the manifest, and each router join rechecks its input. See [four-GPU validation](../../docs/four_gpu_validation.md) for the repairs, routing policy and limits.

The optional auxiliary bundle uses original MR-LoRA router files and recovers the corresponding original QA examples for Replay. The new current task contributes 3,000 sample rows; each preceding task contributes 20 original QA rows. It preserves prompts and answers, and checks that each image resolves to its original physical file. The UCIT first router stage uses the first-task rows from the second stage, since the original first file is empty. The source paths, row indices, hashes, counts and image checks are recorded independently of the sample manifest. Create and verify it with:

```bash
/path/to/llava/bin/python tools/generate_validation_aux.py
/path/to/llava/bin/python tools/generate_validation_aux.py --verify
```

Pass `--auxiliary-paths data/validation_aux/paths.yaml` to `preview` or `execute` to add its exact Replay/router file mappings. The base sample profile is unchanged. Without this overlay those inputs remain unavailable. A passing doctor is still required, including all MR-LoRA dependencies.

Preview without launching any process or using a GPU (substitute the real environment paths and a free port):

```bash
/path/to/llava/bin/python -m mcitlib.validation_batch preview \
  --paths configs/paths.validation.yaml --run-id validation-example \
  --python-llava /path/to/llava/bin/python \
  --python-t437 /path/to/llava-t437/bin/python \
  --execution-profile four_gpu_validation --gpus 0,1,2,3 --master-port 29511
```

Use `execute` in place of `preview` to run the queue. `--python-t437` may be omitted; SEFE and KeepLoRA will then be recorded as blocked, while other jobs continue. Each interpreter is used for both its job's doctor and execution, including DeepSpeed. The explicit `four_gpu_validation` profile requires the physical IDs `0,1,2,3` and preserves effective training batch size; KeepLoRA gradient preparation retains its original two shards. Without a profile the queue requires two distinct physical IDs. Select GPUs and a master port that are free before execution. The doctor sees only the selected GPUs; training and evaluation commands use those physical IDs. The default minimum free disk space before each job and each step is 20 GiB; `--min-free-gib` changes this threshold. It is a guard, not an estimate of the space a complete step will consume. No outputs are deleted automatically.

For a persistent background process, launch from the project root in a detached session and write the console log outside the new run directory, for example:

```bash
mkdir -p logs/validation_3k_1k
setsid /path/to/llava/bin/python -m mcitlib.validation_batch execute \
  --paths configs/paths.validation.yaml --run-id validation-example \
  --python-llava /path/to/llava/bin/python \
  --python-t437 /path/to/llava-t437/bin/python \
  --execution-profile four_gpu_validation --gpus 0,1,2,3 --master-port 29511 \
  > logs/validation_3k_1k/console-validation-example.log 2>&1 < /dev/null &
```

The queue uses a lock under `logs/validation_3k_1k` to reject a duplicate launch. A run ID must be new: no previous checkpoint, log or result directory is overwritten. Each method and benchmark has its own checkpoint and result roots below `ckpts/validation_3k_1k/<run-id>` and `results/validation_3k_1k/<run-id>`. The exact resolved roots and input mappings are saved in each job's `paths.effective.yaml`. Logs, JSON doctor reports, step markers, the generated command list, executed arguments, metrics and `events.jsonl` live under `logs/validation_3k_1k/<run-id>/jobs/<method>/<benchmark>`. `training.log` and `evaluation.log` collect completed step logs; while a step is running, use its log path in `state.json` or `worker.stdout.log` for live output.

Inspect or safely request a stop using the run directory:

```bash
/path/to/llava/bin/python -m mcitlib.validation_batch status --run-dir logs/validation_3k_1k/validation-example
/path/to/llava/bin/python -m mcitlib.validation_batch stop --run-dir logs/validation_3k_1k/validation-example
```

`state.json` is atomically replaced as the queue advances and includes the queue PID and Linux start ticks, current job and step, per-job status, exact log paths, failure tails and summary counts. `stop` verifies the queue identity and lock, then asks it to terminate only the currently controlled process session. For a handoff that must let the current job finish, follow [validation queue handoff](../../docs/validation-handoff.md) instead. Jobs not yet started are marked `not_run`. An entirely successful queue exits zero; failures, blocked jobs, an empty success set or a stop return nonzero. This entry point does not resume failed or interrupted training: start a new run ID to avoid silently overwriting partial artifacts. Existing step markers remain available for inspection and for the separate lower-level runner's explicit resume feature; they are not a Trainer checkpoint-resume guarantee.
