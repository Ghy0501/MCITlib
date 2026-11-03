# LLaVA L2 report

状态：**完成 L2 验收（上游校验、LoRA-FT T2/T3）。**

## 完成项

- `third_party/LLaVA/` — vendored `haotian-liu/LLaVA` @ `31cceb832f2f403b2cbf822d0f8fe689d72fe219`，仅 `llava/` 与原始 LICENSE — `0fb3db9`。
- `third_party/peft/` — vendored `huggingface/peft` @ `v0.4.0`，仅 `src/peft` 与原始 LICENSE — `cff1a98`。
- `third_party/*/UPSTREAM.md`、`MANIFEST.json`、`tools/check_third_party.py` — 上游 provenance、SHA-256 清单和校验 — `58a5d23`。
- `docs/migration/llava_baseline_diff.md` — LoRA-FT 相对上游的训练/模型/CoIN 评测/NB 分类 — `378143c`。
- `mcitlib/backbones/llava/` — `LlavaAdapter`、standard/MoE 两条独立数据管线、`train_entry`、deepspeed argv 构建和 `extra_loss` trainer 钩子 — `b3a7333`。
- `mcitlib/tuners/mcit_expert_lora.py`、`_peft_registry.py` — 唯一基础专家调优器与 v3 PeftType 兼容注册；未迁移 Gate/Router 死代码 — `b3a7333`。
- NB-1：删除训练中的 DoRA 参数、分支和保存条件 — `6ab1363`。
- NB-4：以受限 Python 实现 checkpoint 清理，并附边界测试 — `e76ca89`。
- `tests/equivalence/` — 独立子进程 v3/v4 数据和命令等价性测试 — `adc91d7`。

## 测试结果

| 层级 | 序列/模块数 | 通过 | 失败 | 产物路径 |
|---|---:|---:|---:|---|
| third-party | LLaVA 89 文件、peft 20 文件 | 无缺失/额外/改动 | 0 | `python tools/check_third_party.py` |
| T2 standard 单图 | v3/v4 独立子进程 | `input_ids`、`labels`、`attention_mask`、`images` 逐位一致 | 0 | `tests/equivalence/test_data_pipeline.py` |
| T2 MoE 单图 | v3/v4 独立子进程 | 同上，逐位一致 | 0 | 同上 |
| T2 standard 多图 | v3/v4 独立子进程 | 同上，逐位一致 | 0 | 同上 |
| T3 LoRA-FT | 3 个基准、18 个任务 | v4 argv 与 v3 一致，仅入口模块替换及环境声明 | 0 | `tests/equivalence/test_lora_command.py` |
| MoE 入口选择 | 1 | `MCITLIB_TRAIN_ENTRY=moe` 和源入口正确 | 0 | 同上 |
| NB-4 | 清理边界 | 直接子目录、嵌套、符号链接、缺失目录 | 0 | `tests/unit/test_cleanup.py` |
| 全量 CPU 测试 | 55 | 55 | 0 | `/tmp/p0/l2-pytest.log` |
| lint / SPDX | 自有源文件 | 全通过 / 缺失 0 | 0 | `ruff check mcitlib tools tests`、`tools/add_spdx.py --check` |
| T1、T4、T5 | 后续阶段 | 未执行 | — | L3 及 GPU |

T2 使用合成图片、JSON 和小型自定义 tokenizer/processor，不下载预训练权重。每个 v3/v4 实现均在独立子进程中运行，避免顶层 `llava`/`peft` 包冲突。依赖使用隔离环境，未在训练入口执行 pip。

## 与计划的偏离

- 上游模型文件的 `forward(**kwargs)` 和 `mm_projector strict=False` 仍列为后续 T1 模型加载验证项；本阶段没有为它们修改 `third_party`，也没有宣称完整模型前向已验证。
- `eval_runners/` 的具体 CoIN 脚本迁移和统一评测在后续 L5 完成；本阶段完成其来源分类。
- NB-3 通过使用 pristine peft 0.4.0 避免 v3 调试 print，无额外代码改动；NB-5 在新训练入口中没有运行时 pip，完整白名单随全方法 T3 覆盖。
- `train/eval --dry-run` 的完整 runner 编排仍在后续阶段；本阶段验证的是 adapter 生成的训练命令。

## 新登记的漂移

- `train.py` 与 `train_MOE.py` 的数据和 collator 行为保持两份实现，没有强行合并。
- v3 MoE 导入需要顶层 peft 与 CoIN peft 同时存在；T2 参考子进程显式隔离路径，v4 通过 `_peft_registry` 统一注册。
- `CoINMOEGate`、`CoINMOERouter`、`coinmoelora_new.py` 未迁移，已在差异文档和 tuner 注释中记录。

## 触发的停止条件

- 无。没有修改其他三个骨干、`paper/`、`reviews/`、`design/`，没有推送远端。

## 需要作者执行

- 暂不需要新的决策。L3 后可在 GPU 上运行四个原型方法的 T4；最终仍需 T5。
- 发布前继续补齐相关上游 LICENSE/授权材料。

## 下一阶段

进入 L3，依次完成 `lora_ft` → `replay` → `olora` → `hide`，补齐方法私有模块、T1 模块等价和四个方法 × 三基准的 T0–T3；同时提供 T4/T5 GPU 脚本。L3 结束后暂停一次，等待宏观 Agent 审查接口。
