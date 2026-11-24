# Configuration

Method configurations use shared task data, ordered command templates, and method/benchmark/task differences. The loader expands these files into the structure used by the runner. Expanded YAML is also accepted. Paths and command order follow the selected configuration. No model, GPU, or private data is needed to inspect or validate the configuration:

```bash
mcitlib config validate --json
mcitlib config expand --method lora_ft --benchmark ucit --task 1
mcitlib config expand --method mrlora --benchmark mllm_dcl --task 3 --sequence router
# Equivalent standalone entry point:
python -m mcitlib.config.cli validate --json
```

Use `--config-root` to inspect another config directory. A relative path is resolved from the project root; an absolute path can point to an external directory. Validation checks every registered method, benchmark and task, the router tasks, template and reference expansion, task coverage, and recorded command fields. It does not execute commands, check the model environment or data, or compare the compatibility JSON files with the YAML. `expand` shows the selected merged task and recorded argv before runner transformations such as script aliases, GPU assignment, path resolution and merge steps. General evaluation uses an MLLM-DCL task's configuration; inspect that task with `--benchmark mllm_dcl`.

## Files and precedence

- `configs/common/tasks.yaml` holds the shared model, source path patterns and benchmark task data. `datasets.task_N` references point to these shared benchmark entries. A method can override fields of its current task separately.
- `configs/common/commands.yaml` holds named ordered argv templates and shared consecutive option fragments. `{use: fragment_name}` inserts a fragment in place. The names describe options; they do not determine the method or execution order.
- `configs/methods/<method>.yaml` holds method metadata, `task_defaults` for each benchmark, individual task differences, and ordered command records. An individual command's `set` mapping changes only named option values in its template. `--flag#2` selects a second occurrence of the same option, if needed. Unknown templates, fragments, references and option overrides fail during loading.
- `configs/common/provenance.yaml` and method `source_metadata` hold configuration annotations and aliases. Runtime path roots and mappings belong in the machine paths file.
- `configs/backbones`, `configs/benchmarks`, `configs/paths*.yaml` and the existing compatibility JSON files keep their roles. Local path roots and mappings are configured through `PathResolver` and the paths file, not through configuration annotations.

The loader still applies defaults, backbone, benchmark, method, task, and explicit API overrides in that order. Within each compact method the common task, benchmark `task_defaults`, and task-specific differences are merged before command references are expanded. Read a method with `mcitlib.config.loader.read_yaml`, or use `load_layered_config`; direct `yaml.safe_load` exposes the compact source, not the expanded method. `read_yaml` is cached in each process; restart it or call `read_yaml.cache_clear()` after editing files.

## Changing parameters

For example, `configs/methods/lora_ft.yaml` has a shared `overrides.llava.ucit.task_defaults.train_config` with `rank`, `lr`, `batch_size`, `epoch`, and `grad_acc`. Editing one changes the corresponding `--lora_r`, `--learning_rate`, `--per_device_train_batch_size`, `--num_train_epochs`, or `--gradient_accumulation_steps` when its template references that field. Put a value under an individual `task_N.train_config` to override it for that task. A command may explicitly override a template option with `set`; inspect `config expand` after editing to see its final recorded argv. Option values which are not linked to a task field are maintained in the shared fragment/template or the command's `set`, not inferred from similarly named values. For example, `--lora_alpha` is an independent value; changing `rank` does not implicitly change it.

References have the form `@{task.train_config.lr}`, `@{task.eval_config.model_path}`, or `@{datasets.task_1.data_config.test_path}`. They interpolate the selected merged task or common benchmark dataset as strings while retaining exact argv positions. The syntax is separate from `${model_root}`, `${data_root}`, `${ckpt_root}` and the other path placeholders, which are resolved later by the existing path resolver. Training output and evaluation model paths are independent even where their initial values match.

The public `load_layered_config(..., cli_overrides=...)` API applies those overrides **after** recorded command references have been expanded. Consequently, an API-only override of `train_config` or `eval_config` does not rewrite a previously expanded argv; edit the compact source or explicitly override the command too. This API is separate from the runtime GPU, model-path and path-resolver options.

Some extra stages, including KeepLoRA gradient/weight extraction, read files such as `configs/data_configs/...`, `configs/model_configs/...` and `configs/train_configs/...` directly through their `--data-config`, `--model-config` or `--train-config` arguments. Editing YAML does not update those compatibility JSON files. If an experiment changes a value consumed by such a stage, inspect and update its referenced JSON as well, and validate the resulting complete plan. The command templates record the JSON arguments and their ordering.

Use `validation_fixes=True` in the runner API, or `--validation-fixes` consistently on train, eval, and doctor, for the complete execution plans. This selects distinct ModalPrompt UCIT task-2 output and continuation paths, all 25 MR-LoRA MLLM-DCL expert inference groups, and KeepLoRA MLLM-DCL task-2 Med evaluation.

MR-LoRA DCL selects an expert answer for the same evaluation domain and question ID, then uses the domain scorer. Router letters are routing decisions, not task answers. Selected predictions and scores use `DCL_router`. See [four-GPU execution](four_gpu_validation.md) for the fixed-five-expert retrospective protocol, fallback behavior, and sample validation.

## Plan verification

Inspect the expanded command after every parameter change. Command inspection fields (`args`, `launcher`, `entry`, `positional`) are derived from the ordered argv and are not independent configuration sources.
