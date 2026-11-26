# Other backbones: offline preparation

Backbone implementations follow the [shared method contract](../../docs/llava_implementation.md), including OLoRA non-reentrant checkpoint execution, DISCO/HiDe persistent statistics and request-scoped inference optimization, fresh-rank registration, and target-specific shapes/layers. The executable InternVL integration is described in the [InternVL guide](../../docs/internvl.md) and [sample queue](../../docs/internvl_sample_validation.md). Video backbone execution remains outside this preparation package.

This package prepares data and JSON configuration inventories for InternVL, Video-LLaVA and VideoLLaMA2. Its tools generate offline inputs; they do not load models, start GPU work, install packages or establish reproduction results. For actual video training, inference, scoring and current validation coverage see the [CL-VISTA runtime guide](../../docs/video-runtime.md). The inventory below describes the earlier preparation stage, not the runtime support matrix.

The generated inventory covers 35 backbone/method combinations: 12 InternVL, 12 Video-LLaVA and 11 VideoLLaMA2. InternVL has UCIT and MLLM-DCL variants, so the inventory contains 47 benchmark variants. MLLM-ACL is outside the sample recipe. Video CL-VISTA follows counting, space, traffic, movie, GUI, science, sports and STAR: eight tasks and 36 lower-triangular evaluation cells.

## Video sample

The source is the existing raw `json_row` configuration, mapped to the configured CL-VISTA data root; `json_row_unified` is not substituted. To build a **new** sample directory from the project root:

```bash
python3 -B preparations/other_backbones/tools/video_sample.py \
  --output data/validation_3k_1k_video --seed 20261108
```

The destination must not already exist. The sampler selects up to 3000 train records and 1000 test pairs per task; if fewer usable records exist, it records the shortfall. It preserves original JSON records, typed IDs, questions, answers, subtitles and video strings. It validates one-to-one question/answer IDs and writes the same order to both files because the inference loaders pair them by position. Missing media and cases where the two loaders choose different first extension matches are explicitly excluded and recorded; videos are neither copied nor modified. Source/output hashes, original indices, typed IDs, duplicate pairs, checked media and available train/test video overlap appear in the manifest. Overlap is reported, not silently filtered.

The separate, optional auxiliary builder restores ordinary Replay QA by matching router identities and videos to the original train data and checks the original replay rows. It preserves router records as router data; it never uses router letters as Replay answers:

```bash
python3 -B preparations/other_backbones/tools/video_aux.py \
  --output data/validation_3k_1k_video_aux
```

Each stage contains its selected current task sample and the original 100 QA examples per previous task, in configured Replay order. Earlier-domain memory examples may lie outside the current main-train sample; the separate manifest records each source index and ID and binds the base sample manifest. The base sample is not rewritten. Router training inputs remain distinct from evaluation: the video MR-LoRA extractor has unresolved cross-domain ID/join and denominator behavior and must not be treated as a validated routed score.

## Generate and materialize configurations

After the package is placed under `preparations/other_backbones`, regenerate an inventory into a fresh path and materialize a fresh run namespace:

```bash
python3 -B preparations/other_backbones/tools/generate_configs.py \
  --output preparations/other_backbones/configs/offline-plan-new.json
python3 -B preparations/other_backbones/tools/materialize.py \
  --plan preparations/other_backbones/configs/offline-plan-new.json \
  --output preparations/other_backbones/configs/effective/my-run \
  --run-id my-run --mode sample --gpus 4
```

`--mode full` retains the original per-task budgets; `--gpus 2` selects the two-GPU sample plan. The full profile is not made uniform. The sample main-training plan uses base LR `2e-4`, nominal global batch 64, per-device microbatch 1 with accumulation 16 on four GPUs or 32 on two GPUs, and one epoch except ModalPrompt's explicit 10-epoch sample budget. Video ModalPrompt's original full budget is one epoch; InternVL's is 10. MR router and method-specific auxiliary configuration remain at their original values. These resource choices and effective optimization behavior have not been validated on hardware, and changing microbatch grouping does not guarantee an identical training trajectory.

The materializer emits model/data/train/eval and auxiliary JSON for each model/method/benchmark/task, plus a file-hash and readiness manifest. It expands portable variables and refuses unresolved placeholders or an existing output directory. Checkpoint, result and log roots default under the project and are separated by run ID, backbone, method and benchmark; `--set KEY=VALUE` overrides a declared root. This only creates plans, not checkpoints, scores or a runnable unified CLI.

Absolute and relative checkpoint paths are relocated consistently for outputs, evaluation models, continuation models, gradient space and key paths. The materializer checks fields against each variant's roots, per-stage output uniqueness and references to planned producers. Its inventory retains configuration-source hashes and records missing producers without creating checkpoints. InternVL's executable SEFE merge/key chain, ModalPrompt continuation, and KeepLoRA initialization are provided by the unified runtime; an offline inventory is not its launch manifest.

SMoLoRA image instruction embeddings use existing byte-matched `assets/smolora` files. The two video embedding files under this package's `assets/` are byte-for-byte copies of the corresponding backbone assets, SHA-256 `c8a769183da3b2b41e5b08d09e928fd539e14717d490d2a61ed819ed17ff82ef`. They are checked as files without unpickling. Publish the complete package, including its `assets/` directory, before materializing with the default `PREPARATIONS_ROOT`.

Existing InternVL UCIT/DCL samples and their separately hashed `validation_aux` router/Replay files are reused when the manifests agree. Video sample and auxiliary manifests are independently checked. These checks establish data provenance, not compatibility with a backbone loader. Video-LLaVA needs explicit loader overrides for both `mm_image_tower` and `mm_video_tower`; VideoLLaMA2 needs `mm_vision_tower`. The package records these settings without editing the original model's `config.json`.

Full-data auxiliary paths have separate readiness records. Missing or empty full-data inputs block the affected profiles; sample auxiliary files are not substituted. Regenerate the plan when inputs change.

For video MR-LoRA, the expert pool mapping is `F=counting, G=space, D=traffic, A=movie, C=GUI, E=science, H=sports, B=STAR`; it is not alphabetical task order. The original inference prompt and extractor mapping are checked, but the extractor's cross-domain lookup remains blocked pending an end-to-end correction and validation.

The `reference_evidence/` directory contains byte-identical snapshots of the router and training contracts, stored as `.txt` files with original relative paths and SHA-256 values in its manifest. The generator validates the fixed manifest hash and every snapshot before inspecting them as data; it never imports or executes them. These snapshots support offline contract checks and do not supply model execution entry points. The optional local source inventory is informational and may be absent.

MVBench, LongVideoBench, Video-MME, MMVU and NExT-QA have root placeholders. The inventory does not download, sample or validate these benchmarks.
