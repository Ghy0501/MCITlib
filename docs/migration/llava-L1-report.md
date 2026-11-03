# LLaVA L1 report

状态：**在命令清单阶段触发 S1，暂停；L1 未验收。**

## 完成项

以下文件已写入 `v4-dev` 工作树，但因 S1 暂停，尚未作为 L1 完成项提交：

- `pyproject.toml`、`mcitlib/` — 核心包骨架、注册表、配置 schema/loader、路径解析、RunContext、方法/骨干接口和 argparse CLI。
- `LICENSE`、`tools/add_spdx.py`、`configs/paths.example.yaml`、`envs/llava.txt`、`envs/llava-t437.txt` — 基础合规和环境文件。
- `tools/v3_command_dump.py` — 在临时副本执行 v3 shell；deepspeed/python/pip shim 只记录，JSON 读取用的 `python -c` 交给真实解释器；输出规范化 argv/cwd/env/task/kind/sequence。
- `docs/migration/drift.md` — 新增 SEFE 配置目录断链及无效 argv 证据（本次文档提交）。

未实施：YAML 迁移、T0、CI、36+3 有效命令清单和 L1 验收。

## 测试结果

| 层级 | 序列/模块数 | 通过 | 失败 | 产物路径 |
|---|---:|---:|---:|---|
| dump 冒烟 | LoRA-FT × UCIT | 96 条命令，exit 0 | 0 | `/tmp/p0/v3_commands_test/lora_ft/ucit.json` |
| dump 诊断 | SEFE × UCIT | 记录到 140 条，但配置读取失败、argv 无效 | **S1** | `/tmp/p0/v3_commands_test/sefe/ucit.json`（仅诊断，不入库） |
| SEFE 配置存在性 | UCIT task1 | `RegLoRA/.../task1.json` 存在 | `SEFE/.../task1.json` 不存在 | `../MCITlib_v3_ref/configs/train_configs/` |
| T0 | 36 序列 | 未执行 | — | 因 S1 暂停 |
| T1–T5 | — | 未执行 | — | 后续阶段 |

shim 初次冒烟时曾出现 shebang 经 PATH 递归调用自身的问题，已改为真实 Python 绝对路径；修正后 LoRA-FT × UCIT 正常完成。另修正了 Taskn 同时包含 previous/current 路径时的任务编号推断，当前任务取最后一个 Task 标记。

## 与计划的偏离

- 计划要求 L1 生成 36 条序列和 3 条 router 序列的有效命令清单；当前只完成 LoRA-FT × UCIT 冒烟及 SEFE 故障诊断，未生成完整清单。
- 原因：SEFE 的 v3 脚本引用了不存在的配置目录，继续生成会把无效命令当成期望值。
- 涉及 §9：**S1**。

## 新登记的漂移

- `LLaVA/RegLoRA/scripts/MCITlib/` 有 61 处 `configs/train_configs/SEFE/...` 引用，但 v3 只有 `configs/train_configs/RegLoRA/`。
- shell 没有 `set -e`，`read_config` 的 `FileNotFoundError` 后仍继续执行，产生空参数和非法 GPU 列表；不能将返回码 0 当作成功。

## 触发的停止条件

**S1：T0/T3 的输入无法由 v3 正确生成，且原因不是代码路径组织方式不同。**

具体证据：

- 缺失：`configs/train_configs/SEFE/LLaVA/UCIT/train/task1.json`。
- 存在但不等于可擅自替代：`configs/train_configs/RegLoRA/LLaVA/UCIT/train/task1.json`。
- SEFE dump 中出现 `--lora_r` 后无值、`--include localhost:0,-1`、空 `--model-path`、空输出路径 `_merged`。

未修改 v3 脚本，未把 SEFE 映射到 RegLoRA，未继续生成 v4 YAML，未修改 `paper/`、`reviews/`、`design/` 或其他三个骨干目录。

## 需要作者执行

需要宏观 Agent/作者确认 SEFE 实验时实际使用的配置来源，并提供可追溯的修复参考：

1. 是否缺少应随 v3 发布的 `configs/train_configs/SEFE/` 目录；若是，请提供该目录及其来源 commit。
2. 若实际应使用 `RegLoRA/` 配置，需要明确授权将这个差异登记为 v3 配置断链，并给出 T0/T3 的期望映射；实现方不会自行假定。
3. 在解决前，L1 的完整 dump、配置迁移和 T0 保持暂停。

## 下一阶段

等待 S1 解除后，从当前工作树继续：先重跑 SEFE × UCIT dump，确认无空参数，再生成 36+3 命令清单；随后实现 `migrate_v3_configs.py`、T0 和 CI。
