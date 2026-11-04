# LLaVA L4 report

状态：**12 个 LLaVA 方法的结构迁移和各自适用的 CPU T0–T3 已完成；按阶段边界在此暂停，等待宏观 Agent 审查。真实 7B/GPU 的 T4/T5 未执行，不宣称完整实验复现。**

## 完成项

| 方法 | 实现 | 测试 | 关键内容 |
|---|---|---|---|
| MoELoRA | `249563f` | `2861e2d` | 共享 `MCITExpertLora*`、PeftType 注册兼容 |
| MR-LoRA | `068501f` | `883a336` | 普通序列与 3 条独立 router 序列、推理 adapter 恢复 |
| CL-MoE | `b42836b` | `6b9f1d1` | 变体、RunContext 任务标识；NB-2 诊断工具为 `44ccd99` |
| SMoLoRA | `6def7c2` | `fe39b0c` | 可分离专家变体、三个带 sha256 的显式指令资产 |
| DISCO | `e8cc10a` | `bf3db2a` | 私有模型/CLIP 覆盖、任务 mask 与合并推理 |
| ModalPrompt | `0c10915` | `e5e4991` | 私有 prompt 模型、原型与 separate train/infer |
| SEFE | `2232a16` | `7a3d47b` | t437 私有正则模型、关键元素累计和 merge |
| KeepLoRA | `17a3ab2` | `4727774` | t437 私有模型、Task0/梯度 SVD、A 初始化与冻结 |

补充修复：

- `7ee68b5`：SEFE/KeepLoRA 的 merge 支持直接文件路径执行，使用各自私有 builder；checkpoint 模型名使用与 v3 CoIN 等价的 `llava.mm_utils.get_model_name_from_path`。
- `76e3312`：GPU 回归工具从注册表读取所有方法的 `model_override`，不再只特判 HiDe。
- `MethodSpec.script_aliases` 和 runner 的别名处理把 v3 脚本/模块名映射到方法私有实现；NB-5 install 命令不进入执行清单。
- runner 支持方法级 `pre_commands`，保留 KeepLoRA 每个 benchmark 一次的 Task0；记录命令保留 `CUDA_VISIBLE_DEVICES` 白名单。
- `tools/migrate_v3_configs.py` 可从 fixture 重新生成 KeepLoRA 的 Task0 和 chunk 环境，不需要手工维护大型 YAML。
- SEFE/KeepLoRA 的私有 LLaVA 文件均位于各自 `_sefe/`、`_keeplora/`，未改写 `third_party/` 公共快照。

## 接口变化

相对 L3 报告中的接口，本阶段增加：

```python
MethodSpec.script_aliases: tuple[tuple[str, str], ...]
MethodSpec.model_override: str | None  # 覆盖 disco/hide/modalprompt/sefe/keeplora

TrainingArguments.freeze_lora_A: bool = False
TrainingArguments.init_lora_from_gradients: str | None = None

allow_t437_llava_registration()  # 仅限已知 t437 导入兼容，退出时清理临时补丁
```

KeepLoRA 的训练入口在 PEFT LoRA 创建后、DeepSpeed 包装前执行 SVD 初始化和 A 冻结；默认参数为 `False`/`None`，其他方法路径不变。SEFE 正则和 KeepLoRA 普通前向使用不同私有模型文件，没有把二者错误合并为同一算法。

## 测试结果

| 层级 | 范围 | 通过 | 失败 | 产物/证据 |
|---|---|---:|---:|---|
| T0 | 36 条普通序列 + 3 条 router；KeepLoRA 另含 Task0 | 全部 | 0 | `tests/unit/test_config_migration.py` |
| T1 模块/插件 | 12 个方法各自适用的小模型、变体或数据选择测试 | 全部 | 0 | `tests/equivalence/test_*`、`tests/unit/test_methods.py` |
| SEFE 数值 | 2 层正则模型、前向/梯度、关键元素累计 | 全部 `torch.equal` | 0 | `test_sefe.py`、`_sefe_harness.py` |
| KeepLoRA 数值 | 普通模型、A 初始化/冻结、选样/在线平均、chunk 合并、旧子空间投影、fixed-rank/energy/权重 SVD | 全部逐张量相等 | 0 | `test_keeplora.py`、`_keeplora_harness.py` |
| T2 数据 | standard/MoE、多图及各私有方法实际 v3 目录；SEFE/KeepLoRA 在 t437 单独验证 | 全部逐张量相等 | 0 | `test_data_pipeline.py`、t437 测试 |
| T3 命令 | 12 方法 × 3 基准 = 36 条普通序列，另 3 条 router | argv、kind、task_id、顺序一致 | 0 | `tests/equivalence/test_lora_command.py` |
| KeepLoRA 顺序/环境 | Task0 仅一次；pre → train → merge → post → eval；chunk GPU 白名单；UCIT Task2 post 仍用 VizWiz | 全部 | 0 | T0/T3 与 `/tmp/p0/l4-keep-task1.json` |
| CLI | `mcitlib list`；KeepLoRA 首任务 dry-run | 12 条命令，task_id 为 0/1，install=0 | 0 | `/tmp/p0/l4-list.out`、`l4-keep-task1.json` |
| transformers 4.31 | 全部非 t437 测试 | **136 passed, 2 skipped, 1 deselected** | 0 | `/tmp/p0/l4-final-431.out` |
| transformers 4.37.2 | SEFE、KeepLoRA、环境声明 | **12 passed** | 0 | `/tmp/p0/l4-t437.out`；merge 修复后同一 CI 路径重跑 |
| lint / SPDX / third-party | 自有源文件和 2 个上游组件 | 全部通过；SPDX 缺失 0；上游错误 0 | 0 | `ruff check mcitlib tools tests` 等 |
| 其他目录不变性 | InternVL、Video-LLaVA、VideoLLaMA2、paper、reviews、design | 相对 `v3.0-final` 无差异 | 0 | `git diff --quiet v3.0-final -- ...` |
| T4 / T5 | 真实模型和 GPU | 未执行 | — | 需作者在 GPU 节点运行 |

CPU 测试合计为两个独立环境的结果，不能把 skipped 项解释为另一环境已验证。T1/T2 使用合成图片、JSON、玩具模型和小矩阵，不下载预训练权重。KeepLoRA 的 v3/v4 子进程将硬编码 CUDA 设备替换为 CPU 以比较公式；GPU SVD 的浮点误差和完整训练仍需 T4/T5。

## 与计划的偏离和验证边界

- 为保持数值语义，保留 standard/MoE 数据入口，并允许 SEFE/KeepLoRA 使用隔离的 t437 私有快照；未强行得到“单一快照/单环境”。
- `train/eval` 目前提供有序 dry-run；真实任务执行、完整 checkpoint 链路、CoIN 评测 runner 和指标实现不属于已完成的 GPU 验证。回归工具的 CPU/mock 测试只证明命令准备和 loss 比较逻辑。
- CL-MoE 的 task 文件读取和默认关闭的统计写文件按 NB-2 改变；SMoLoRA 资产移动、SEFE 配置别名、router 新路径和训练段恢复均按作者授权登记，不作为隐式算法修复。
- NB-1、NB-2、NB-4、NB-5 已有实施与测试记录；NB-6 的通用配置注入仍未完成，不在本报告中宣称完成。
- CL-MoE、HiDe-LLaVA、ModalPrompt 的上游再发布许可证仍待确认；DoRA 仍是发布阻塞项且未被 v4 引入。

## 触发的停止条件

本阶段最终无未解决的 S1/S2。此前 SEFE 配置、router 数据和缺失训练序列问题均按作者明确授权处理并登记。没有修改其他三个骨干或禁改目录，没有推送远端。

## 需要作者在 GPU 节点执行

1. 按 `docs/migration/placeholders.md` 设置模型、数据、checkpoint 根目录；尤其确认独立 router JSON 的真实内容，以及 KeepLoRA 的 Task0/上一任务子空间归属。
2. 分别在 `llava` 和 `llava-t437` 环境运行匹配的方法，禁止在任务之间动态安装依赖。
3. 先用 `tools/regression/run_short.sh --combo <method>:<benchmark>` 预览，再用 `--execute` 运行任务 1、2 的前 20 个 loss，并保存日志与比较产物。`run_full.sh` 当前只扩展现有训练 loss 预检的任务范围，不自动执行 Task0/pre/post/merge/eval，不能单独视为完整 T5。
4. 对 ModalPrompt 分开核对 train/infer；对 SEFE/KeepLoRA 核对 key-element/merge 或 Task0/pre/post/merge 的完整顺序。完整序列执行仍待后续 GPU runner；当前 CPU 结果不能替代这些检查。

## 下一阶段

**L4 在此结束并暂停。** 下一步先由宏观 Agent 审查接口、提交和测试证据；审查通过后再进入后续 CoIN eval runner、指标与实际 GPU runner 工作。本次不提前开始下一阶段。
