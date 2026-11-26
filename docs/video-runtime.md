# CL-VISTA video runtime

Use this guide from a **release checkout** on a Linux x86_64 CUDA host. The two video workers use different versions of Torch and Transformers; a third environment runs the local Qwen judge. You supply the model weights, tower weights, annotations and media. Neither installing MCITlib nor generating a node binding downloads them.

## What has been run?

| Backbone | Official CL-VISTA sample | Other evidence |
| --- | --- | --- |
| Video-LLaVA (`videollava`) | LoRA-FT (`lora_ft`), PCLR (`pclr`), Octopus (`octopus`): each completed 8 training tasks and 36 held-out evaluation cells. | The remaining 14 methods have bounded A100 checks. |
| VideoLLaMA2 (`videollama2`) | No official eight-task sample result. | All 17 methods have bounded A100 checks covering training, saving, reloading, generating and scoring; some checkpoints were temporary. |

The official sample used 3,000 train rows and 1,000 held-out questions per domain. After task N, it evaluated the N domains seen so far: **36,000 scored predictions per method**, not 36,000 distinct questions. PCLR's completed sample included its integration stages at tasks 4-8; Octopus's included GPWC and second-stage training at tasks 2-8. Neither backbone has an official **full-data** result here.

The 17 CLI method IDs for both backbones are `lora_ft`, `replay`, `moelora`, `mrlora`, `olora`, `clmoe`, `disco`, `hide`, `smolora`, `modalprompt`, `sefe`, `keeplora`, `pclr`, `same`, `sigma`, `dgg`, and `octopus`. A bounded check exercises a small amount of real data, not the eight-task schedule or its storage budget. In particular, MR-LoRA has a routing diagnostic, not a validated eight-expert result matrix.

## Install the runtimes

Prepare the versioned native code once before launching a worker. It is intentionally not installed from PyPI. See [source provenance and offline preparation](video-dependencies.md) for the file checksums and the no-network alternative.

Run these Bash examples from the repository root, with Conda initialized. Replace `/absolute/path/to/video-runtime` with a private, user-owned directory (not the checkout). Use a compatible NVIDIA driver, CUDA development toolkit, C++ compiler, Ninja, and sufficient host memory and writable space for compiled extensions, optimizer offload, checkpoints and scores. The Torch wheel supplies runtime libraries, **not** the compiler. Video-LLaVA's doctor actually loads the DeepSpeed CPUAdam extension.

| Environment | Python | Torch / CUDA build | Transformers | FlashAttention in reference env |
| --- | --- | --- | --- | --- |
| [Video-LLaVA](../envs/videollava.txt) | 3.10.19 | 2.0.1 / cu117 | 4.31.0 | 2.6.3 |
| [VideoLLaMA2](../envs/videollama2.txt) | 3.10.18 | 2.2.0 / cu118 | 4.40.0 | 2.5.8 |
| [Judge](../envs/video-judge.txt) | 3.10.18 | 2.6.0 / cu118 | 4.57.6 | 2.7.3 (optional backend) |

The files pin runtime packages from the A100 environments; they are not complete machine images. Before installing, check your checkout for the selected native source: `third_party/Video-LLaVA/videollava` or `third_party/VideoLLaMA2/videollama2`. If missing, obtain the complete release; an unrelated PyPI or upstream package is not a substitute. A worker must load PEFT 0.4.0 from **its own environment**. Do not install the image-backbone requirements or add image PEFT to your shell's `PYTHONPATH`.

```bash
export RUNTIME_ROOT=/absolute/path/to/video-runtime
export RESULTS_ROOT="$RUNTIME_ROOT/results"
export NODE="$RUNTIME_ROOT/node.json"
export JUDGE_MODEL="$RUNTIME_ROOT/models/Qwen3-30B-A3B-Instruct-2507"
mkdir -p -m 700 "$RUNTIME_ROOT" "$RESULTS_ROOT"

# With a full v4-dev Git clone (which includes the original v3 commit):
python3 tools/prepare_video_sources.py --output "$RUNTIME_ROOT/sources" --from-git .
# With a shallow clone or a source archive, use --fetch instead of --from-git .
python3 tools/prepare_video_sources.py --output "$RUNTIME_ROOT/sources" --check

# Each invocation creates and installs one independent environment.
install_video_env() (
  set -e
  case "$1" in
    videollava)   py=3.10.19; cu=cu117; pt=2.0.1; vision=0.15.2; fa=2.6.3 ;;
    videollama2)  py=3.10.18; cu=cu118; pt=2.2.0; vision=0.17.0; fa=2.5.8 ;;
    video-judge)  py=3.10.18; cu=cu118; pt=2.6.0; vision=; fa= ;;
    *) return 2 ;;
  esac
  conda create -y -c conda-forge -p "$RUNTIME_ROOT/envs/$1" "python=$py" pip
  conda activate "$RUNTIME_ROOT/envs/$1"
  python -m pip install -c "envs/$1.txt" numpy 'setuptools>=68,<81' wheel packaging ninja psutil
  wheels=("torch==$pt+$cu")
  if [[ -n "$vision" ]]; then wheels+=("torchvision==$vision+$cu"); fi
  if [[ "$1" == videollama2 ]]; then wheels+=("torchaudio==$pt+$cu"); fi
  python -m pip install --index-url "https://download.pytorch.org/whl/$cu" "${wheels[@]}"
  python -m pip install --no-build-isolation -r "envs/$1.txt"
  if [[ -n "$fa" ]]; then python -m pip install --no-build-isolation "flash-attn==$fa"; fi
  python -m pip install -e .
  python -m pip check
)
install_video_env videollava
install_video_env videollama2
install_video_env video-judge
```

For a single backbone, skip the other worker installation. If FlashAttention is supplied as a binary wheel, install the matching **Python, Torch, CUDA and C++ ABI** build in place of its source install. For a source build, select the compatible toolkit and compiler for that environment before running the function. Limit build parallelism with `MAX_JOBS` on memory-constrained hosts.

## Bind models and data

The worker expects the prepared sources under `RUNTIME_ROOT/sources`. Set `native_sources` in the node JSON or export `MCITLIB_VIDEO_SOURCES` to use a different prepared directory. The public [validation record](video-validation.json) summarizes which stages and denominators were verified.

```bash
conda activate "$RUNTIME_ROOT/envs/videollava"  # either worker can generate the binding
python -m mcitlib.backbones.video.binding --root "$RUNTIME_ROOT" > "$NODE"
```

Edit the generated JSON when a path differs. This is a path template; it does not create model views, check files or subsample full data. Keep it private. The defaults are:

| Under the runtime root | What belongs there |
| --- | --- |
| `models/Video-LLaVA-7B` | Model/tokenizer directory; its `config.json` must point `mm_image_tower` at the local `LanguageBind_Image` and `mm_video_tower` at `LanguageBind_Video_merge`. |
| `models/VideoLLaMA2-7B` | VideoLLaMA2 Mistral 7B model/tokenizer with STC connector; set `mm_vision_tower` to the local `clip-vit-large-patch14-336`. |
| `models/<tower>` | The tower directories above. The auxiliary CLIP directory also needs its text assets for methods such as DISCO, HiDe, SAME and ModalPrompt. |
| `envs/videollava`, `envs/videollama2` | Worker prefixes; the active Python prefix must match the chosen binding. |
| `data/{sample,full}/<domain>/` | `train.json`, `test_questions.json`, `test_answers.json`; add `replay.json` for Replay or `router.json` for MR-LoRA. |
| `media/` | Video files preserving the relative paths in the annotations. |
| `checkpoints/`, `results/`, `logs/runner/` | Writable outputs. Keep method history until later tasks and evaluation finish. |

Point `model_view` at a user-created model view if the original config contains remote IDs or obsolete paths: keep the other model files available in that directory and edit a **copy** of the config. Make its tower paths agree with `models.<backbone>.towers` in the node binding; editing the binding alone does not patch the config.

The domain order is `counting, space, traffic, movie, gui, science, sports, star`. Supply JSON arrays: training rows have `video` (relative filename including extension) and `conversations`; test questions have `video_name` (relative stem), `question_id`, `question`; answers have the matching `question_id` and `answer`. IDs must pair one-to-one without changing type, and must remain unique when converted to strings for scoring. See [offline preparation](../preparations/other_backbones/README.md) to prepare samples and auxiliary inputs. Bind prepared sample and full files separately. Octopus needs the **full training annotation even in sample mode** to scale its auxiliary draws.

## Start the judge first

In a second shell on the **same host**, set the same `RUNTIME_ROOT`, `RESULTS_ROOT` and `JUDGE_MODEL` as above. Allocate GPUs separate from the workers (the IDs below are examples); do not override a scheduler allocation. The historical judge model is **Qwen3-30B-A3B-Instruct-2507**. Provide the complete local checkpoint and tokenizer with chat template.

```bash
conda activate "$RUNTIME_ROOT/envs/video-judge"
CUDA_VISIBLE_DEVICES=2,3 python -m mcitlib.backbones.video.judge_service \
  --runtime-root "$RUNTIME_ROOT" --results-root "$RESULTS_ROOT" \
  --model-path "$JUDGE_MODEL"
```

Wait for **`judge_ready`**, not just the existence of `control/judge.sock`. Leave this terminal running until scoring finishes; Ctrl-C stops it. Use a user-owned runtime root that is not group/world writable and a private `control` directory (0700). The client uses a local Unix socket, not an API key or HTTP endpoint. If using FlashAttention 2, install the compatible `flash-attn==2.7.3` build in the **judge** environment and pass `--attn-implementation flash_attention_2`. Omitting the flag leaves attention selection to Transformers. Record the backend and batch size when comparing scores.

## Train and evaluate

In the worker shell, choose one backbone and activate its matching prefix. Run the block once per desired backbone; the shared run ID below is safe across the two because outputs include the backbone name.

```bash
BACKBONE=videollava       # or: BACKBONE=videollama2
conda activate "$RUNTIME_ROOT/envs/$BACKBONE"
RUN=first-sample
common=(--backbone "$BACKBONE" --method lora_ft --benchmark cl_vista
        --video-node "$NODE" --video-run-id "$RUN" --gpus 0,1)

# Inspect the plan without loading weights. The next two commands DO use GPUs.
mcitlib train "${common[@]}" --video-mode sample --task 1 --dry-run
mcitlib doctor "${common[@]}" --video-mode sample
mcitlib train "${common[@]}" --video-mode sample --task 1
```

`train --task 1` trains, reloads the checkpoint, generates held-out answers, then scores them. For task 2, use the **same** method, backbone, mode and run ID with `--task 2` so the preceding checkpoint is found. Use a fresh run ID when changing inputs or settings. To run all eight tasks, omit `--task`; for example, first inspect the two-GPU full-data schedule (no execution):

```bash
mcitlib train "${common[@]}" --video-mode full --dry-run
```

For a new sample sequence, select `--method pclr` or `--method octopus` in place of `--method lora_ft`. The executed LoRA-FT and PCLR sample training used four worker GPUs with 16 accumulation steps; Octopus used two with 32 steps. All used microbatch 1 and effective batch 64, with separate judge GPUs. Choose 1, 2 or 4 distinct physical worker IDs within your allocation; `--gpus` needs no lease wrapper. Full budgets can require a specific GPU count. MR-LoRA needs **two** GPUs, the whole plan (no `--task` or standalone `eval`), and all expert/router inputs; its expert pool is retrospective.

### Inference from a saved checkpoint

`mcitlib eval --after-task N` generates and scores each seen domain from checkpoint N without training. Use the same run ID and checkpoint root. A successful training run already scored those cells, so for a second evaluation choose a **fresh results directory** under the judge's authorized tree. This example reuses task 1 from the preceding block; run it once per backbone:

```bash
EVAL_NODE="$RUNTIME_ROOT/node-eval.json"
python - "$NODE" "$RESULTS_ROOT/reeval-01" > "$EVAL_NODE" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    node = json.load(f)
node["results"] = sys.argv[2]
print(json.dumps(node, indent=2))
PY
mcitlib eval --backbone "$BACKBONE" --method lora_ft --benchmark cl_vista \
  --video-node "$EVAL_NODE" --video-mode sample --video-run-id "$RUN" \
  --gpus 0,1 --after-task 1
```

### Check or score one cell

For the original task-1 run, inspect:

```bash
CELL="$RESULTS_ROOT/sample/$BACKBONE/lora_ft/$RUN/task1/counting"
wc -l "$CELL/predictions.jsonl"
cat "$CELL/native-score/results.txt"
```

`predictions.jsonl` carries one `id`, `question`, `answer` and `pred` per held-out row. The judge assigns 0-5 per row; the normalized cell mean is 20 times the raw mean (not exact-match accuracy). Check the sample count against the expected held-out denominator. Per-ID judgments live under `native-score/qwen/`, combined judgments in `native-score/scores.json`.

If generation finished but scoring did not start, submit the **unscored** cell without repeating training or inference:

```bash
python -m mcitlib.backbones.video.score_client \
  "$RUNTIME_ROOT/control" "$CELL/predictions.jsonl"
```

A successful reply says `scored` and includes the count and result path. The service only accepts a file named `predictions.jsonl` under its configured results root and refuses to overwrite an existing `native-score` directory. An absent or incomplete judgment is not a zero-score benchmark result.
