# LLaVA L1 report

状态：**SEFE 配置问题已获作者授权解决；MR-LoRA router 问题触发新的 S1，L1 暂停且未验收。**

## 完成项

以下文件已写入 `v4-dev` 工作树，但未作为完成的 L1 代码提交：

- `pyproject.toml`、`mcitlib/` — 核心包骨架、注册表、配置 schema/loader、路径解析、RunContext、方法/骨干接口和 argparse CLI。
- `LICENSE`、`tools/add_spdx.py`、`configs/paths.example.yaml`、`envs/llava.txt`、`envs/llava-t437.txt` — 基础合规和环境文件。
- `tools/v3_command_dump.py` — 临时副本执行 v3 shell，shim 记录 deepspeed/python/pip；真实执行 JSON 读取用的 `python -c`；支持作者授权的 SEFE→RegLoRA 临时配置别名，并写入 `config_aliases` 元数据。
- `tests/fixtures/v3_commands/llava/` — 已生成 36 条普通序列及 3 条 router 诊断 JSON；普通序列尚未作为 L1 验收产物提交，router 文件不可作为有效期望值。
- `docs/migration/drift.md` — 登记 SEFE 授权映射及 MR-LoRA router 的配置/入口断链。

未实施：YAML 迁移、T0、CI 和 L1 验收。

## 测试结果

| 层级 | 序列/模块数 | 通过 | 失败 | 产物路径 |
|---|---:|---:|---:|---|
| dump 冒烟 | LoRA-FT × UCIT | 96 条命令，exit 0 | 0 | `/tmp/p0/v3_commands_test/lora_ft/ucit.json` |
| SEFE 授权后 dump | SEFE × UCIT | 140 条命令，参数恢复有效 | 0 | `/tmp/p0/v3_commands_test/sefe/ucit.json` |
| 普通序列批量检查 | 36 | 返回码和空参数检查通过 | 0 | `tests/fixtures/v3_commands/llava/*/*.json` |
| router dump | 3 | DCL 记录 35 条但有输出文件噪声 | UCIT/ACL 配置或入口失败 | `tests/fixtures/v3_commands/llava/mrlora/*_router.json`（诊断） |
| T0 | 36 序列 | 未执行 | — | 因 S1 暂停 |
| T1–T5 | — | 未执行 | — | 后续阶段 |

shim 的 shebang 递归问题已修正；Taskn 同时包含 previous/current 路径时，任务编号取最后一个 Task 标记。完整 dump 共生成 39 个 JSON、2,478 条命令；其中 router 的异常不能按普通命令清单验收。

## 与计划的偏离

- SEFE 的配置目录断链已按作者明确“SEFE 就是 RegLoRA”的授权，在 dump 临时副本中映射到 `RegLoRA`，v3 原树未改。
- MR-LoRA router 仍无法生成真实有效的完整命令清单，因此没有继续 YAML 迁移和 T0。
- 涉及 §9：**S1**。

## 新登记的漂移

- `Task1_router.sh` 无条件读取 `router_path`，但 UCIT `ImageNet-R.json`、ACL `OCR.json` 没有该键，导致 `KeyError` 和空训练参数。
- `train_UCIT_router.sh` 调用不存在的 `Eval_UCIT/Eval_finetune1.sh`，目录实际为 `Eval_ucit.sh`；ACL 同样调用 `Eval_MLLM_ACL/Eval_finetune1.sh`，目录实际为 `Eval_acl.sh`。
- DCL router 后续 `cat` 错误来自 shim 不生成模型输出文件，属于 dump 噪声，不与上述真实断链混淆。

## 触发的停止条件

**S1：MR-LoRA router 的 T3 输入无法由 v3 正确生成，原因不是代码路径组织方式不同。**

不能自行推断 `router_path=train_path`，也不能自行把 v3 的评测入口改为同名文件。未修改 v3 脚本、`paper/`、`reviews/`、`design/` 或其他三个骨干目录。

## 需要作者执行

请提供或确认 MR-LoRA router 的参考来源：

1. UCIT/ACL 各任务的 `router_path` 数据配置，尤其是 `ImageNet-R` 和 `OCR`；不能由实现方自行用 `train_path` 替代。
2. UCIT/ACL router 实际应调用的评测脚本（当前训练脚本中的文件名与目录中的文件不一致）。
3. 明确这些差异是否只是公开发布时漏文件，以及论文中的 router 结果对应的代码/配置版本。

## 下一阶段

等待 router S1 解除后，重跑 3 条 router dump，并仅为临时答案文件补齐 shim 的输出占位以消除 DCL 噪声；随后实现 `migrate_v3_configs.py`、T0 和 CI。
