# Benchmark inputs for v4

Start with the [public quick start](public-start.md) for installation and a first score. This page is for running the **real benchmarks**. For image inputs, set `data_root` in a private copy of [paths.example.yaml](../configs/paths.example.yaml) (pass `--paths`), or export `MCITLIB_DATA_ROOT`. Below, `$DATA` denotes that absolute directory, **not** a variable the loader reads. Run examples from the checkout root. Video uses a separate node JSON binding.

These instructions compare pinned public file inventories, selected annotation rows and archive headers with [v4 image task definitions](../configs/common/tasks.yaml), [data templates](../configs/data_configs) and the [video binding](../mcitlib/backbones/video/binding.py). They are **not** a full download, media integrity check, or end-to-end reproduction. Use the dataset owners' access terms for each constituent. Save dataset revisions, file hashes, local mappings and any derivation with the run.

LLaVA uses UCIT, DCL, ACL and the general image suite; the supported InternVL image benchmarks here are UCIT and DCL. The video backbones use CL-VISTA. Check the selected method's plan: a data template describes available inputs, not a guarantee that every method schedules every evaluation.

## UCIT: use the original annotations

[MLLM-CL/UCIT at 384446b](https://huggingface.co/datasets/MLLM-CL/UCIT/tree/384446be3b539a6bf9ceed8198b4a886c411700b) contains six train/test Parquet configurations. Its card calls it an **unofficial** training-ready fork of HaiyangGuo/UCIT. The `images` list, `problem`, string `_qid` and `conversations` are not v4 image filenames, evaluation records or multi-reference caption annotations. Renaming a shard or exporting its rows to JSON is not a conversion. No byte- or ID-equivalence to the original split is assumed here.

For the native v4 format, use **[HaiyangGuo/UCIT, UCIT/ subtree at c28c148](https://huggingface.co/datasets/HaiyangGuo/UCIT/tree/c28c148ebdc4e1fc068d193f29f5b310aa65ca23/UCIT)**, not its `CoIN_sample/` subtree. The [authors' media guide](https://github.com/Ghy0501/HiDe-LLaVA#ucit-benchmark) identifies which images must be obtained separately. The following fetches **annotations only**, including the small VizWiz instruction ZIP, with the Hugging Face CLI. Choose a fresh staging directory.

```bash
export STAGE=/absolute/path/to/benchmark-staging
export DATA=/absolute/path/to/benchmark-data
export MCITLIB_DATA_ROOT="$DATA"
hf download HaiyangGuo/UCIT --repo-type dataset \
  --revision c28c148ebdc4e1fc068d193f29f5b310aa65ca23 \
  --include 'UCIT/*/*.json' 'UCIT/VizWiz.zip' --local-dir "$STAGE/original"
unzip -l "$STAGE/original/UCIT/VizWiz.zip"
```

Copy/link the listed JSONs into the **annotation** directories below. For VizWiz extract the three `VizWiz/*.json` members of the ZIP, ignoring `__MACOSX/`. In every row of this table the test filename is `test_3000.json`. The media column is independent: all train/test `image` strings resolve against **`$DATA/UCIT`**, not the directory of the JSON. For instance, the released VizWiz test points to `VizWiz/val/VizWiz_val_00000000.jpg`, not `VizWiz-caption/val/...`.

| Stage | Original under `UCIT/` | Annotation directory under `$DATA/UCIT/` | Train filename | Media under `$DATA/UCIT/` and source |
|---|---|---|---|---|
| 1 | ImageNet-R | ImageNet-R | `train.json` | `ImageNet-R/{train,test}/` from original `ImageNet-R/imagenetr.tar.gz` |
| 2 | ArxivQA | ArxivQA | `train_4w.json` | `ArxivQA/images/` from [MMInstruction/ArxivQA, images.tgz](https://huggingface.co/datasets/MMInstruction/ArxivQA/tree/main) |
| 3 | VizWiz.zip | VizWiz-caption | `train.json` | `VizWiz/{train,val}/` from [VizWiz **Captions** images](https://vizwiz.org/tasks-and-datasets/image-captioning/) |
| 4 | IconQA | IconQA | `train.json` | `IconQA/iconqa_data/iconqa/...` from [IconQA](https://iconqa.github.io/) |
| 5 | CLEVR-Math | CLEVR-Math | `train_4w.json` | `CLEVR/images/{train,val}/` from the [CLEVR-Math image source](https://huggingface.co/datasets/dali-does/clevr-math); media prefix is **CLEVR** |
| 6 | Flickr30k | Flickr30k-cap | `train_brief_4w.json` | `Flickr30k/{train,val}/` from original `Flickr30k/train.tar.gz` and `val.tar.gz` |

Keep `val_coco_type_3000.json` beside the test file for **both** caption tasks; the scorer needs the original image IDs and multiple captions. Preserve all filenames, question ID types (including leading zeros), turns and split membership. Do not use a constituent dataset's own test questions in place of UCIT's. Inspect archive members before extracting. If the authors' layout is already installed with a common `datasets/` directory, expose its media subdirectories directly under `$DATA/UCIT/`, or map the common image root and the individual annotation files separately. Changing the JSON's parent does not change the media root.

**Replay and MR-LoRA:** [UCIT_ReplayData at f55bdd7](https://huggingface.co/datasets/MLLM-CL/UCIT_ReplayData/tree/f55bdd7a7f3c3555b6056acb244601a738f1c03d) and [UCIT_RouterData at 9d5cc41](https://huggingface.co/datasets/MLLM-CL/UCIT_RouterData/tree/9d5cc41bf90e19034aa7defa83ed3ba358db4520) each list stages 2-6 only. Copy `<original task>/replay.json` or `router.json` to that stage's **annotation** directory above (including the two caption aliases); media still resolves from `$DATA/UCIT`. Replay consumes ordinary QA and prior examples; router training consumes expert-selection prompts and labels. The LLaVA method overrides also request `ImageNet-R/replay.json` and `ImageNet-R/router.json`. No corresponding files were listed. For a *derived* first-stage Replay input, the original ImageNet-R train set has no prior task to mix in: link its `train.json` as `replay.json` and record this choice, not an official Replay file. For a full router reproduction obtain the intended nonempty first-stage set or document a separately derived protocol. Do not use normal answers as router labels.

The optional [image sampler](../tools/sample_validation_data.py) is for UCIT + DCL together, not ACL or general evaluation. It also reads logical `train.json` and `test.json`: for UCIT make these aliases to the table's train file and `test_3000.json` wherever the names differ. The sampler compares their hashes. The [sample auxiliary builder](../tools/generate_validation_aux.py) has a specific UCIT stage-one derivation from the stage-two router pool; it additionally requires a sample manifest, original train data and a matching external router-config tree for **both** suites. It is not a downloader or a full-data fallback.

## MLLM-CL: DCL and ACL

Use the [MLLM-CL/MLLM-CL release at be08199](https://huggingface.co/datasets/MLLM-CL/MLLM-CL/tree/be0819923463e34fc67d869d374ebcb9f004a4e4). Fetch individual files/directories from this revision; the full image archives are large. The Hub's `domain/` and `ability/` group the downloads; neither appears between `$DATA/MLLM-CL/` and a v4 task. For example, inspect or fetch just the two RS annotations before obtaining its images:

```bash
hf download MLLM-CL/MLLM-CL domain/RS/train.json domain/RS/test.json \
  --repo-type dataset --revision be0819923463e34fc67d869d374ebcb9f004a4e4 \
  --local-dir "$STAGE/mllm"
```

| Suite (v4 order) | Published annotations | Destination under `$DATA/MLLM-CL/` | Media |
|---|---|---|---|
| DCL: RS, Med, AD, Sci, Fin | `domain/<task>/{train,test}.json` | `<task>/{train,test}.json` | Unpack that task's `images.tar` to give `<task>/images/...`; both splits use `<task>` as image root. |
| ACL: OCR, Math, VP, APP | `ability/<task>/train.json` | `<task>/train.json` | Extract **each** `images-*.tar` shard for that task into the same task directory (OCR: 7, Math: 8, VP: 3, APP: 5). These are tar archives, not numbered fragments of one stream. |
| ACL held-out | `ability/{OCR_test,Math_test,VP_test,APP_test}/test.json` | Respectively `{OCR_test,Math_test,CV-Benchv2,APP_test}/test.json` | Each held-out folder has its own `images.tar`. Expose its extracted `images/` next to its test JSON. |

To finish **RS only**, when ready for a 4.5 GB media download, use the same revision as the RS annotation command. The following assumes a new, empty RS destination and that you have inspected the archive before extraction:

```bash
hf download MLLM-CL/MLLM-CL domain/RS/images.tar --repo-type dataset \
  --revision be0819923463e34fc67d869d374ebcb9f004a4e4 --local-dir "$STAGE/mllm"
mkdir -p "$DATA/MLLM-CL/RS"
cp "$STAGE/mllm/domain/RS/train.json" "$STAGE/mllm/domain/RS/test.json" "$DATA/MLLM-CL/RS/"
tar -xf "$STAGE/mllm/domain/RS/images.tar" -C "$DATA/MLLM-CL/RS"
```

For example, a published VP test row references `images/2D/count/ade20k_365.png`. Either expose the entire extracted `VP_test` directory as `$DATA/MLLM-CL/CV-Benchv2` or map that logical prefix in your paths file. This maps **both** questions and images:

```yaml
path_mappings:
  mllm_acl:
    - from: ${data_root}/MLLM-CL/CV-Benchv2
      to: /absolute/path/to/staged/ability/VP_test
```

**Replay and MR-LoRA:** the authors' [MLLM-CL-ReplayData release at a6b7374](https://huggingface.co/datasets/MLLM-CL/MLLM-CL-ReplayData/tree/a6b7374773969a7f2327536d1ec47574cdc09063) contains both. For DCL, map `replay_DCL/taskN.json` to `$DATA/MLLM-CL/<Nth DCL task>/replay.json` and `router_DCL/taskN.json` to its `router.json`. For ACL do the same with `replay_ACL/`, `router_ACL/` and the Nth **training**, not test, folder. Thus `replay_ACL/task2.json` goes to `Math/replay.json`. Unlike main train JSON, these mixtures use **`$DATA/MLLM-CL`** as the image root: e.g. a first-stage Replay row has `RS/images/8004.png`. Preserve those prefixes and the published ordering. **Both published `router_DCL/task1.json` and `router_ACL/task1.json` contain `[]`.** A plan that trains a first-stage router cannot do so from these files. Obtain the intended input or report the changed protocol; later-stage router files and the single-domain expert data do not fix this gap.

**ACL scoring has extra inputs.** The [original authors' code repository](https://github.com/bjzhb666/MLLM-CL) lists [the MathVista workbook](https://github.com/bjzhb666/MLLM-CL/blob/main/llava_v1.5_7b_MathVista_MINI.xlsx) and [the APP submission TSV template](https://github.com/bjzhb666/MLLM-CL/blob/main/llava_v1.5_7b_MMT-Bench_ALL_openai_submission.tsv). Place the originals under those basenames in the checkout root for plans using those evaluators. Math also needs its configured judge; the APP converter emits a submission TSV, not a local official score. The [benchmark card](https://huggingface.co/datasets/MLLM-CL/MLLM-CL#sample-usage-mr-lora-training-and-evaluation) points to the external scoring service; availability and acceptance were not tested.

## CL-VISTA: raw annotations and videos

Use [CL-VISTA at 835504a](https://huggingface.co/datasets/MLLM-CL/CL-VISTA/tree/835504a8730b1bf8e807121d5bb1a97b0b1f854b). The Hub preview and `train_VISTA_joint.json` are not the eight-task held-out protocol. The small(er) `CL-VISTA_json.tar.gz` contains **`json_row_unified/<Domain>/...`**, including separate Movie train/test subdirectories. Its member names were inspected; equivalence to raw annotations was **not** established. Do not rename it to satisfy the [raw v4 templates](../configs/data_configs/CL-VISTA).

1. Stage authorized archives for the domains needed. Counting, GUI, Movie, Science, Sports, STAR and Traffic each have `<Domain>/<Domain>.tar.part0000` and subsequent parts. Compare the downloaded names and sizes to the pinned listing. These are fragments: join **all** parts for one domain in numeric order, inspect the resulting tar member names, and unpack in a fresh area preserving relative paths. For example, after staging Counting locally: `cat "$STAGE"/Counting/Counting.tar.part[0-9][0-9][0-9][0-9] | tar -tf -` lists the joined stream. The Counting tar starts with `Counting/`. Expose the domain trees under one root, e.g. `$DATA/CL-VISTA`. Confirm the raw annotation files in the table below; if absent from your distribution, obtain the matching raw release from the authors. A partial archive listing does not verify that every referenced clip or raw JSON is present.
2. **Space requires separate access.** Its published `Space/json_row.tar` starts with `json_row/`; extract it under `$DATA/CL-VISTA/Space/`. It is not the video collection. Follow the [ScanNet agreement and download instructions](https://github.com/ScanNet/ScanNet#scannet-data) and the [CL-VISTA card's Space note](https://huggingface.co/datasets/MLLM-CL/CL-VISTA#-benchmarks) for authorized preprocessed frames. The supplied [convert_video.py](https://huggingface.co/datasets/MLLM-CL/CL-VISTA/blob/835504a8730b1bf8e807121d5bb1a97b0b1f854b/Space/convert_video.py) needs its ffmpeg executable, input and output paths set locally. It reads `scene*/color/*.jpg` and writes `Space/scannet_videos_fps1/<scene>.mp4`; the pinned script actually passes `fps=4` despite the directory name. Record the script revision and settings. Access approval, conversion and coverage were not tested.
3. Use the table for the **full** binding. Base paths are relative to `$DATA/CL-VISTA/`. Except for Movie, both held-out files are `test_questions_2k.json` and `test_answers_2k.json` in the listed directory. Except for Counting, Replay/router files are `replay_train.json` and `router_data.json` in the train directory.

| Stage / node key | Raw train directory | Train filename | Exception |
|---|---|---|---|
| 1 counting | Counting/json_row | train_data_20k.json | Replay uses **this train file**; router is `router_data_100.json`. |
| 2 space | Space/json_row | train_30k.json | ScanNet media required. |
| 3 traffic | Traffic/json_row | train_data_30k.json | |
| 4 movie | Movie/train_json_row | movie_train_subtitle_30k.json | `Movie/test_json_row/test_question_2k.json` and `test_answer_2k.json` (singular). |
| 5 gui | GUI/json_row | train_data_30k.json | |
| 6 science | Science/json_row | train_data_30k.json | |
| 7 sports | Sports/json_row | train_data_30k.json | |
| 8 star | STAR/json_row | train_data_25k.json | |

Generate the editable node file as in the [video runtime guide](video-runtime.md). Set its top-level `media` to the common CL-VISTA root: `media / row["video"]` must resolve without dropping the domain prefix. Under `tasks.<lowercase domain>.full` set `train_path`, `test_q_path`, `test_a_path`, `replay_path`, `router_path` to the corresponding **absolute** raw filenames above (or link them under the generated binding's default names). Train rows have `video` with extension and `conversations`. Held-out questions use `video_name` stems and `question_id`; answers must match IDs **and ordering** for the native loaders. Do not conflate expert letters with ordinary Replay answers.

For **sample** mode, on Linux with the raw files and media present, the existing offline tools create separate outputs in fresh directories. This does not load a model:

```bash
python3 preparations/other_backbones/tools/video_sample.py \
  --video-root "$DATA/CL-VISTA" --output "$STAGE/sample" --seed 20261108
# Optional, requires the original raw replay AND router files:
python3 preparations/other_backbones/tools/video_aux.py --video-root "$DATA/CL-VISTA" \
  --sample-root "$STAGE/sample" --output "$STAGE/sample-aux"
```

Set the three main `tasks.<domain>.sample` paths to `$STAGE/sample/<domain>/{train,test_questions,test_answers}.json`; set Replay/router to `$STAGE/sample-aux/<domain>/{replay,router}.json`. Keep `media` unchanged. Inspect the manifests for missing/conflicting media and any shortfall from 3,000 train / 1,000 test pairs per domain. Octopus sample runs also require the full train annotation. See [sample provenance and auxiliary requirements](../preparations/other_backbones/README.md). A partial seven-domain run without Space is not an eight-domain CL-VISTA result.

## General image evaluation

LLaVA's `general` is evaluation only. Use the [upstream LLaVA evaluation guide](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#scripts) for its linked **eval.zip** (custom annotations) and each benchmark's separate media/labels; it is not included with MCITlib. Map its extracted task directories to the following [v4 inputs](../configs/data_configs/General_benchmark), relative to `$DATA`:

| Task | Questions and inputs required by v4 |
|---|---|
| POPE | `pope/llava_pope_test.jsonl`, COCO images in `pope/val2014/`, [POPE](https://github.com/RUCAIBox/POPE) category labels in `pope/coco/coco_pope_*.json`. Preserve the per-category row order for scoring. |
| MME | `MME/llava_mme.jsonl`, official `MME/MME_Benchmark_release_version/` tree with category images **and** yes/no ground-truth text files. |
| MMBench | The English dev `mmbench/mmbench_dev_20230712.tsv` linked in the guide; images, choices and dev labels are in the TSV. V4's offline option score is not an online leaderboard submission. |
| SEED-Bench | `seed_bench/llava-seed-bench-modified.jsonl`, `seed_bench/SEED-Bench.json`, `seed_bench/SEED-Bench-image/` and middle video frames under `seed_bench/SEED-Bench-video-image/`. [LLaVA's script](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/eval/seed.sh) names its input `llava-seed-bench.jsonl` **without** `-modified`. A verified source or transformation for the v4 modified file was not established; obtain the matching questions or disclose and validate an alternative, not just a rename. |

The [general registry](../configs/benchmarks/general.yaml) covers these four tasks; video benchmark JSON templates alone are not an executable general-video suite.

## Check before launch

Inspect the resolved plan for the **same method, task, backbone and paths** you intend to run. Pass an explicit paths file even for a dry run; otherwise paths may be displayed as placeholders. These examples use the shipped file and the exported `MCITLIB_DATA_ROOT`. Substitute your private file when using mappings (no GPU or dataset download):

```bash
mcitlib train --method replay --benchmark ucit --task 1 --paths configs/paths.example.yaml --dry-run
mcitlib train --method mrlora --benchmark mllm_dcl --sequence router --task 2 --paths configs/paths.example.yaml --dry-run
mcitlib eval --method lora_ft --benchmark mllm_acl --after-task 2 --paths configs/paths.example.yaml --dry-run
mcitlib eval --method lora_ft --benchmark general --after-task 5 --paths configs/paths.example.yaml --dry-run
```

A dry run prints arguments; it does **not** open or validate the datasets. Check each resolved annotation and its actual image root together. For example, on a prepared DCL RS task, this read-only check counts missing files without decoding images. Adapt the first two arguments to the other splits or a Replay/router stage (use the **shared** media root there):

```bash
python3 - "$DATA/MLLM-CL/RS/test.json" "$DATA/MLLM-CL/RS" <<'PY'
import json, sys
from pathlib import Path
annotation, media = map(Path, sys.argv[1:])
rows = json.loads(annotation.read_bytes())
if not isinstance(rows, list) or not rows:
    raise SystemExit(f"Expected a nonempty array: {annotation}")
missing = 0
for row in rows:
    name = row["image"]
    if not isinstance(name, str) or Path(name).is_absolute() or not (media / name).is_file():
        if missing < 3:
            print("Missing or invalid relative image:", name)
        missing += 1
print(f"{annotation}: {len(rows)} rows, {missing} unresolved images")
sys.exit(bool(missing))
PY
```

Spot-decode representative media, verify held-out IDs against caption references or paired answers, and check the evaluated denominator against the intended split. Keep missing material explicit; a printed plan, generated sample, or submission file is not a benchmark score.
