# First video score: one native LoRA update

Use this to answer **"can this worker train, save and reload on this host?"**
It trains on a two-second red clip, starts a fresh process to answer a question
about a blue clip, and scores that answer locally. No CL-VISTA data, judge,
socket, API key or other service is needed. A correct or incorrect answer is a
valid smoke result; this is **not a benchmark or a test of learning quality**.

## Prepare once

From the release checkout on Linux x86_64, pick **one** worker and install its
[pinned environment and CUDA build prerequisites](video-runtime.md#install-the-runtimes).
Skip the other worker and the judge. The native stacks are different; don't mix
their Torch, Transformers or PEFT packages. Use a BF16-capable GPU, sufficient host RAM for the 7B model
and a private scratch filesystem with at least 4 GiB free **after** provisioning
models and the environment. Model downloads are much larger than the fixture;
see the [download setup and size estimates](public-start.md#2-get-just-the-models-you-need).

```bash
export BACKBONE=videollava               # or videollama2
export RUNTIME_ROOT=/absolute/path/to/private-video-runtime
export MODELS="$RUNTIME_ROOT/models"
export SOURCES="$RUNTIME_ROOT/sources"
umask 077
mkdir -p "$RUNTIME_ROOT"

# In a download environment, list the pinned inputs; add --download to acquire.
python3 tools/public_models.py "$BACKBONE" --root "$MODELS"
python3 tools/public_models.py "$BACKBONE" --root "$MODELS" --download
# Or use already provisioned weights under the same names. Respect upstream terms.
python3 -B tools/prepare_video_sources.py --fetch --output "$SOURCES"
python3 -B tools/prepare_video_sources.py --check --output "$SOURCES"
```

For offline source provisioning, see [video dependencies](video-dependencies.md).
The runner makes a private model view: it copies only the base config, sets
`mm_image_tower` and `mm_video_tower` (Video-LLaVA), or `mm_vision_tower`
(VideoLLaMA2), and links the other model files. **Do not edit the downloaded
config through a snapshot symlink.** No node binding or manual model-view
override is required for this smoke.

## Plan and run

```bash
conda activate "$RUNTIME_ROOT/envs/$BACKBONE"
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=1
RUN="$(mktemp -d "$RUNTIME_ROOT/first-score.XXXXXXXX")/$BACKBONE"
args=("$BACKBONE" --sources "$SOURCES" --models "$MODELS" --run-dir "$RUN")
python -B tools/first_score_video.py "${args[@]}" --plan
python -B tools/first_score_video.py "${args[@]}"
cat "$RUN/predictions.jsonl" "$RUN/fixture-score.json"
```

The plan needs no GPU or weights. Execution requires a new run directory; a
failed attempt is left intact. The runner uses your CUDA environment. Build tools and the CUDA toolkit must be on the worker's path; set `CUDA_HOME`, `CC` and `CXX` as required on your host.

## What to inspect

| Artifact under `$RUN` | Evidence |
| --- | --- |
| `train-red.mp4`, `eval-blue.mp4`, `train.json`, `questions.json`, `answers.json` | Original, redistributable synthetic fixture generated locally: 16 frames at 8 fps, 128 x 128, a moving white square on a solid background; no external media. |
| `plan.json`, `train.log` | Exact worker arguments; one rank-4 LoRA step, batch size 1, no warmup. Look for `vision_forward` and `parameters_updated`: positive loss, step 1 and actual LoRA and projector changes. |
| `<backbone>-lora/` | Native final `adapter_config.json`, `adapter_model.bin`, model `config.json` and `non_lora_trainables.bin` (projector). No full base-weight copy or optimizer checkpoint is requested. |
| `generate.log`, `predictions.jsonl` | Separate process loads base + saved state; look for `reloaded_for_inference`, `generation_complete` and the one generated `pred`. |
| `fixture-score.json` | `fixture_exact_match`: 0/1 or 1/1 against the independently read answer. Only outer whitespace and case are ignored. Punctuation and extra prose count; missing, extra or blank predictions fail. |

The worker seed is 42; Video-LLaVA's native decoder samples, so don't expect
bitwise identical text across environments. Keep the private model view and
its original snapshots available to reload the adapter later. To reclaim
space, remove only your own run directory after retaining the evidence you need.
For actual CL-VISTA evaluation use the [separate benchmark procedure](video-runtime.md).

## Reference smoke

Each run below used one A100 PCIe 40 GB, the pinned native sources and an
existing local 7B model and towers. Both separate inference processes loaded
the saved states in FP16 and emitted exactly one row for `eval-blue`.

| Worker | Step / logged loss | Changed tensors (LoRA / projector) | Adapter / projector bytes | Prediction / score |
| --- | --- | --- | --- | --- |
| Video-LLaVA | 1 / 0.0799 | 224 / 4 | 20,142,861 / 41,961,191 | `Blue` / 1 of 1 |
| VideoLLaMA2 | 1 / 6.8199 | 224 / 85 | 21,126,346 / 978,416,652 | `Blue` / 1 of 1 |

These are receipts for this fixture, not acceptance thresholds. They verify
the native path with already provisioned environments and weights, not a fresh
environment install, Hub download, or byte-for-byte identity with those snapshots.
