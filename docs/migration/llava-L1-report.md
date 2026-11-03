# LLaVA L1 report

状态：**SEFE 和 router 的路径问题已按作者指示处理；训练脚本缺失任务触发新的 S1，L1 暂停且未验收。**

## 完成项

以下文件已写入 `v4-dev` 工作树，但未作为完成的 L1 代码提交：

- `pyproject.toml`、`mcitlib/` — 核心包骨架、注册表、配置 schema/loader、路径解析、RunContext、方法/骨干接口和 argparse CLI。
- `LICENSE`、`tools/add_spdx.py`、`configs/paths.example.yaml`、`envs/llava.txt`、`envs/llava-t437.txt` — 基础合规和环境文件。
- `tools/v3_command_dump.py` — 临时副本执行 v3 shell，shim 记录 deepspeed/python/pip；真实执行 JSON 读取用的 `python -c`；支持 SEFE→RegLoRA 临时配置别名、router 独立数据占位和 UCIT/ACL 评测入口别名；所有映射写入 JSON 元数据。
- `tools/migrate_v3_configs.py` — 已实现 JSON/命令到 YAML 的自动迁移原型；在全量运行时因缺失训练任务停止，未生成仓库内 YAML。
- `tests/fixtures/v3_commands/llava/` — 已生成 39 个 JSON；其中普通序列的训练完整性检查发现 3 个异常组合，不能直接作为完整 T0/T3 期望。
- `docs/migration/drift.md` — 登记 SEFE 授权、router 占位规则和所有缺失训练任务。

未实施：仓库内 YAML、T0、CI 和 L1 验收。

## 测试结果

| 层级 | 序列/模块数 | 通过 | 失败 | 产物路径 |
|---|---:|---:|---:|---|
| dump 冒烟 | LoRA-FT × UCIT | 96 条命令，exit 0 | 0 | `/tmp/p0/v3_commands_test/lora_ft/ucit.json` |
| SEFE 授权后 dump | SEFE × UCIT | 140 条命令，参数恢复有效 | 0 | `/tmp/p0/v3_commands_test/sefe/ucit.json` |
| router 占位后 dump | 3 | UCIT 90 / DCL 35 / ACL 26，均 exit 0、无空参数 | 0 | `tests/fixtures/v3_commands/llava/mrlora/*_router.json` |
| 全量 dump | 39 | 2,584 条命令已记录 | 3 个普通序列训练不完整 | `tests/fixtures/v3_commands/llava/` |
| 配置迁移试运行 | 至 SMoLoRA DCL | 前 8 个方法 YAML 写到 `/tmp` | 缺 Task1 停止 | `/tmp/p0/v4_configs_test/` |
| T0 | 36 序列 | 未执行 | — | 因 S1 暂停 |
| T1–T5 | — | 未执行 | — | 后续阶段 |

## 与计划的偏离

- SEFE 按作者确认使用 `RegLoRA` 配置；router 按作者指示使用独立 `router.json` 占位路径，未使用 `train_path`，后续由 GPU 节点替换。
- 训练脚本中被注释的任务不能通过占位符补齐，因此没有继续生成完整 v4 YAML，也没有把缺失任务伪装成通过。
- 涉及 §9：**S1**。

## 新登记的漂移

- SMoLoRA × MLLM-DCL 的 `train_DCL.sh` 注释了 Task1 训练和评测，只有 Task2–5。
- ModalPrompt × UCIT 注释了 Task2 训练，dump 中 Task1 两次、Task2 缺失。
- ModalPrompt × MLLM-ACL 的 Task1–4 训练全部注释，只剩评测。
- ModalPrompt 多处使用 `/home/hongbo_zhao/ghy-cl-codebase/MCITlib/checkpoints/...` 硬编码 checkpoint 路径。

## 触发的停止条件

**S1：v3 普通训练序列缺少计划要求的任务，无法在不改变行为的情况下完成 36 条序列的 T0/T3。**

恢复或补写这些训练段会新增 v3 未执行的步骤，不能由实现方自行决定。未修改 v3 脚本、`paper/`、`reviews/`、`design/` 或其他三个骨干目录。

## 需要作者执行

请提供或确认以下训练参考：

1. SMoLoRA × MLLM-DCL Task1 是否应恢复，及其对应的 v3 脚本/配置版本。
2. ModalPrompt × UCIT Task2、ModalPrompt × MLLM-ACL Task1–4 是否应恢复；若这些基准只做评测，请明确从 36 条训练序列的验收范围中排除，而不是让实现方自行补写。
3. ModalPrompt checkpoint 路径的可替换根目录及实际实验来源。

## 下一阶段

等待训练序列 S1 解除后，重新运行全量 dump 和 `migrate_v3_configs.py`，再完成 T0、CI 和 L1 验收。
