# First image score: train, reload, score

Train LoRA-FT for **one optimizer step**, reload its saved adapter in a fresh process,
and score **one held-out prediction** with the native MLLM-DCL remote-sensing (RS)
evaluator. This is a bounded integration check, not a reproduction of a paper result.

## Run the public demo

Use a source checkout and the image GPU environment in [Public start](public-start.md).
The inputs are local model weights plus a pair of generated 128x128 PNGs. No benchmark
annotations, private server, judge, credential, or dataset download is needed for the
demo. The two original images, prompts and labels are redistributable under the
repository's Apache-2.0 license. The model weights have their own upstream terms.

1. In the **download environment**, populate only the desired model group(s):

   ```bash
   export MODEL_ROOT="$HOME/mcit-runtime/models"
   python tools/public_models.py llava internvl --root "$MODEL_ROOT" --download
   ```

2. In the image training environment, run one of the following from the checkout. Use `--gpu` to choose a device if needed.

   ```bash
   export MCITLIB_MODEL_ROOT="$HOME/mcit-runtime/models"
   mkdir -p logs
   python -m mcitlib.smoke_images --backbone llava --gpu 0 --port 29521 \
     --run-dir logs/first-score-llava
   python -m mcitlib.smoke_images --backbone internvl --gpu 0 --port 29522 \
     --run-dir logs/first-score-internvl
   ```

   These commands **execute**, not preview. Use an unused port and a **new**
   run directory each time. Output stays under the checkout's `logs/` directory.

The trainer sees a red square and its answer. The inference process gets a blue
square and only the question, not the answer. Both use the shared backbone-specific
LoRA-FT path. The single-rank launcher keeps the scheduler's device pin; batch,
accumulation and maximum steps are all one, warmup is disabled, and generation is
greedy with a hard eight-new-token limit. A run-local config points to the downloaded
vision tower when necessary; model files are linked, not copied or modified.

## Read the evidence

In each run directory:

| File | What to check |
| --- | --- |
| `train.log`, `Task1_{llava,internvl}_lora/trainer_state.json` | One completed step. The runner also checks that the saved LoRA B tensors are finite and not all still at their zero initialization. |
| `Task1_{llava,internvl}_lora/` | Nonempty model config, adapter config, adapter weights and additional trainable weights. Keep this directory and the base model for reload. |
| `heldout-question.json`, `inference.log`, `inference.json` | Separate reload; adapter path, question ID, image, prompt, generated text and generation length. |
| `heldout-label.json`, `answers.jsonl`, `Result.text` | One reference, one actual prediction and native accuracy in percent. Zero is a valid measured score. |
| `smoke-result.json` | Exact commands, inputs, run status, verified step count, score, denominator and data scope. Only `completed` is success. |

To independently re-score the saved prediction, without loading a model:

```bash
run=logs/first-score-llava
mkdir "$run/rescore"
python -m mcitlib.backbones.llava.eval_runners.coin.eval_ai2d \
  --annotation-file "$run/heldout-label.json" --result-file "$run/answers.jsonl" \
  --output-dir "$run/rescore"
diff "$run/Result.text" "$run/rescore/Result.text"
```

**Interpretation:** a native scorer on generated examples is still a **non-benchmark**
score. To exercise an authorized RS split instead, add `--dataset rs --paths
/absolute/path/to/paths.yaml` and choose a fresh run directory. The expected layout
is `DATA_ROOT/MLLM-CL/RS/{train.json,test.json}`, with image paths relative to
`RS/`. Configure `data_root` or the normal benchmark path mappings. Only the first
train and first test rows are used; overlapping question IDs or images are rejected.
Even on authentic RS data, this is not full-split accuracy.

## Observed fixture runs

On 2026-11-26, both families ran on a single 40 GB A100 per invocation, using
existing local weights and an offline worker (Python 3.10.15, Torch 2.0.1+cu118,
Transformers 4.31.0, DeepSpeed 0.9.5, FlashAttention 2.6.3, timm 0.6.13).
Neither score below is benchmark
accuracy. In each case the trainer saved step 1, all 224 saved LoRA B tensors
were finite, at least one was nonzero, and the separate inference process
decoded two new tokens on `fixture-test-1`.

| Backbone | Train loss | Reloaded prediction | Native score | Adapter SHA-256 prefix |
| --- | ---: | --- | ---: | --- |
| LLaVA-1.5-7B | 5.7525 | Blue | 1/1 (100.00%) | `1b59e090ff438f3e` |
| InternVL-Chat-ViT-6B-Vicuna-7B | 5.1725 | Blue | 1/1 (100.00%) | `38f0bb2f7427481e` |

Hashes identify these persisted `adapter_model.bin` files; reproducing the
procedure does not require a bit-identical checkpoint. The runs did not exercise
fresh network downloads or audit all existing weight files against the public pins.

**Separate RS data check:** with one authorized RS training row and the first
test question (ID `1027618`, reference `1`), both reloaded models generated
`There are two rectangular retail areas` within the eight-token cap. The native
scorer reported **0/1 (0.00%)** for each. The saved adapter SHA-256 prefixes
were `d163d13cd7746196` (LLaVA) and `653e276b0171244f` (InternVL).
These are measured single-question results, not full RS benchmark estimates.

To test command construction, failure checks, fixture isolation and the native
metric without training, use the image environment with pytest installed:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/unit/test_image_first_score.py \
  tests/unit/test_smoke.py tests/integration/test_internvl_commands.py
```
