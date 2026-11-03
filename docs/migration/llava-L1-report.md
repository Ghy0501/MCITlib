# LLaVA L1 report

状态：**完成。**此前的 S1 诊断项已由作者明确授权处理；本报告不再把它们列为未解决停止条件。

## 完成项

- `pyproject.toml`、`mcitlib/`、`configs/paths.example.yaml`、`envs/`、`LICENSE` — 核心包骨架、注册表、分层 YAML loader、路径解析、RunContext、方法/骨干接口、argparse CLI 和两个 LLaVA 环境声明 — `882d45d`。
- `.github/workflows/ci.yml`、`tools/add_spdx.py`、`tools/check_third_party.py` — lint、两个环境测试、SPDX 和 third-party 检查 — `882d45d`。
- `tools/v3_command_dump.py` — 临时副本执行 v3 shell，shim 记录命令；支持 SEFE→RegLoRA 配置别名、router 独立数据占位、评测入口别名、作者授权的训练段恢复和硬编码路径占位 — `bb592eb`。
- `tools/migrate_v3_configs.py` — 从 v3 JSON 和命令清单自动生成 YAML，保留训练/评测/额外步骤 argv 及解析参数 — `bb592eb`。
- `tests/fixtures/v3_commands/llava/` — 36 条普通序列 + 3 条 MR-LoRA router 序列，共 39 份 JSON、2,584 条命令 — `32bbe67`。
- `tests/unit/` — T0、注册表、路径、CLI、dump 元数据和环境测试 — `32bbe67`。
- `configs/backbones/`、`configs/benchmarks/`、`configs/methods/` — 由迁移工具生成的分层 YAML，未手工转写 — `5cb36d3`。
- `docs/migration/placeholders.md`、`behavior_changes.md`、`drift.md` — 记录占位符用途、作者授权的序列统一及 ModalPrompt UCIT 的特殊 output_dir 漂移 — 本次文档提交。

## 测试结果

| 层级 | 序列/模块数 | 通过 | 失败 | 产物路径 |
|---|---:|---:|---:|---|
| T0 配置等价 | 36 条普通序列 | 36 | 0 | `tests/unit/test_config_migration.py` |
| router 配置/命令检查 | 3 条 router 序列 | 3 | 0 | `tests/unit/test_config_migration.py`、`tests/fixtures/v3_commands/llava/mrlora/` |
| 全量命令完整性 | 39 份 JSON / 2,584 条命令 | 39 | 0 | `tests/unit/test_config_migration.py::test_all_sequence_task_coverage` |
| CPU 单元测试 | 46 | 46 | 0 | `PYTHONPATH=. MCITLIB_V3_ROOT=../MCITlib_v3_ref pytest -q` |
| t437 环境标记测试 | 1 | 1 | 0 | `pytest -q -m env` |
| lint | `mcitlib tools tests` | 通过 | 0 | `ruff check mcitlib tools tests` |
| SPDX | 自有 Python 源文件 | 缺失 0 | 0 | `python tools/add_spdx.py --check` |
| third-party | L1 尚未 vendor | 通过基线检查 | 0 | `python tools/check_third_party.py` |
| CLI | `list`、`train --dry-run` | 通过冒烟 | 0 | `python -m mcitlib.cli ...` |
| T1–T5 | 后续阶段 | 未执行 | — | L2 起 |

T0 直接读取 v3 JSON 和 dump，逐任务比较分层合并后的 YAML 配置值及完整 argv，不是用迁移脚本验证自身。所有训练序列均有且仅有按 benchmark 顺序排列的 Task1…TaskN 训练命令，且无空参数或 `localhost:0,-1`。

## 与计划的偏离

- 按作者后续明确授权，临时参考副本恢复了 SMoLoRA × MLLM-DCL Task1、ModalPrompt × MLLM-ACL Task1–4；恢复值写入 `sequence_repairs`，不属于实现方自行添加的算法变化。
- 按作者确认，SEFE 使用 RegLoRA 配置来源；router 使用独立的新 `router.json` 占位路径，不使用 `train_path`。
- 按作者要求，ModalPrompt 的 checkpoint、CLIP 和旧 Python 硬编码路径统一为占位符。占位符用途和 GPU 替换要求见 `placeholders.md`。
- ModalPrompt × UCIT Task2 的 v3 `output_dir` 仍保留为 Task1，仅任务编号改用一基 `cur_task` 推断；该可疑行为未修复，已登记。
- 当前 `train/eval --dry-run` 只提供 CLI 骨架和状态输出；真正的 LLaVA adapter 命令构建在 L2 实现，因此本报告不宣称 T3 或实际训练已完成。

## 新登记的漂移

- SEFE 配置目录名与脚本引用不一致，按作者确认的 RegLoRA 映射处理。
- MR-LoRA router 缺少首任务数据字段及 UCIT/ACL 评测入口文件名，使用独立占位和显式脚本别名，待 GPU 核对。
- ModalPrompt UCIT Task2 的 `output_dir` 错写为 Task1，保留原值并记录。
- 其余路径、运行时 pip 和序列集成问题已写入 `drift.md` 与 `behavior_changes.md`。

## 触发的停止条件

- 无未解决的停止条件。此前 S1 的各项均获得作者明确处理授权；没有把占位路径或恢复序列声称为 GPU 数值验证。
- 未修改 `paper/`、`reviews/`、`design/`、`InternVL/`、`Video-LLaVA/`、`VideoLLaMA2/`，未推送远端。

## 需要作者执行

1. 在 GPU 节点按 `docs/migration/placeholders.md` 替换模型、数据、router 和 checkpoint 占位路径，尤其确认独立 router 数据内容。
2. 核对 ModalPrompt UCIT Task2 的 `output_dir` 是否应保留 v3 的 Task1 写法。
3. L3 后运行 T4，L4 后运行 T5；当前没有 GPU 数值结果。
4. 发布前补齐相关上游 LICENSE 或授权材料。

## 下一阶段

进入 L2：vendor `third_party/LLaVA` @ `31cceb83` 与 `third_party/peft` @ `v0.4.0`，实现上游校验、LLaVA adapter、standard/moe 两条数据管线和真实 train entry；先让 `lora_ft` 的 T2/T3 通过，再进入 L3 原型。
