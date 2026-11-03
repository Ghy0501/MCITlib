# v3 漂移、可疑行为与兼容性风险登记

日期：2026-11-03  
原则：本文件只登记，不在 P0 修复。任何影响数值的行为都不能以“看起来像 bug”为理由顺手修改。

## 1. 实际训练入口的漂移（LLaVA）

先按 `scripts/MCITlib/Train/Task1.sh` 确定真实入口，再检查对应实现。MoE 系方法走 `train_mem_MOE.py` → `train_MOE.py`，其余方法走 `train_mem.py` → `train.py`。

| 方法 | 实际入口 | 多图输入 | `is_SAT` | `subprocess.run` 清理 checkpoint | `use_reentrant=False` |
|---|---|---:|---:|---:|---:|
| LoRA-FT | `train.py` | 有 | 有 | 有 | 无 |
| Replay | `train.py` | 有 | 有 | 有 | 无 |
| MR-LoRA | `train.py` | 有 | 有 | 有 | 无 |
| KeepLoRA | `train.py` | 有 | 有 | 有 | 无 |
| MoELoRA | `train_MOE.py` | 无 | 无 | 有 | 无 |
| CL-MoE | `train_MOE.py` | 无 | 无 | 有 | 无 |
| DISCO | `train_MOE.py` | 无 | 无 | 有 | 无 |
| HiDe | `train_MOE.py` | 无 | 无 | 有 | 无 |
| OLoRA | `train_MOE.py` | 无 | 无 | 有 | 有 |
| SMoLoRA | `train_MOE.py` | 无 | 无 | 有 | 无 |
| ModalPrompt | `train.py` | 无 | 无 | 有 | 无 |
| RegLoRA/SEFE | `train.py` | 无 | 无 | 有 | 无 |

这比只比较 `train.py` 更准确地描述了运行行为：例如 OLoRA 的普通 `train.py` 没有实际的 checkpoint 删除调用，但它并不是 MoE 训练的实际入口；`train_MOE.py` 中仍然存在该调用。

v4 的统一数据管线若要采用多图/`is_SAT` 超集，必须先通过 T2 证明对无这些分支的方法所用数据逐位一致；否则触发停止条件 1，不得直接合入。

## 2. 其他骨干的训练文件观察

- InternVL 的 `llava/train/train.py` 在 12 个方法目录中均未发现多图/`is_SAT` 分支，但都有 checkpoint 清理相关代码。
- Video-LLaVA 的 `videollava/train/train.py` 在 12 个方法目录中均有多图分支，均未见 `is_SAT`。
- VideoLLaMA2 的 `videollama2/train.py` 在 11 个方法目录中均未见多图/`is_SAT` 分支。

这些是基于文本特征的登记，不等价于数据张量等价证明；统一管线仍以 T2 为准。

## 3. 训练结束删除 checkpoint

多个实际入口使用如下形式：

```python
subprocess.run(
    f"find {remove_dir} -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {{}} +",
    shell=True,
)
```

v4 改为受限的 Python 实现是结构和安全改进，但必须保证删除范围、时机和 v3 一致：仅当前 `output_dir` 的直接子目录、仅匹配 `checkpoint-*`、在相同训练阶段执行。实现前需确认 `remove_dir` 的取值和相对路径解析，不能扩大删除范围。

## 4. CL-MoE 的文件系统副作用

四个骨干的 CL-MoE 调优器都在 forward 中读取硬编码的 `task.txt`，例如：

```text
/your_path/MCITlib_v3/LLaVA/CL-MoE/task.txt
```

并在当前秒数等于 30 时，将 top-2 router 统计追加到 `value_counts_<task>.txt`。训练脚本在训练前写入该 task 文件。

- 这解释了 v3 对硬编码路径和当前工作目录的依赖。
- 文件写入本身通常不改变 router 数值，但读取任务标识、统计时机和异常行为属于运行行为的一部分。
- v4 若将任务 ID 通过 `RunContext` 注入，需保留实际任务值和数值路径；是否保留统计文件输出，需在阶段报告中作为行为差异说明，不能默默声称完全等价。

## 5. SMoLoRA 的 pickle 与指令 embedding

`LLaVA/SMoLoRA/llava/train/train_MOE.py` 使用 `pickle.load` 读取 `model_args.ins_emb` 指定的 `.pkl`，随后转为 list 传入 `SMoLoraConfig`。`ins_type` 控制视觉/指令路由分支，`lora_ins_gate` 的输入维度写死为 384。

v4 需要：

- 保留 v3 读取结果和路由数值；
- 将 `.pkl` 视为不可信输入，禁止 import 时自动加载；
- 由显式命令/配置触发加载，并记录来源、形状和校验信息；
- 按 D5 默认方案考虑迁到 HF 数据资源，作者确认前不在仓库中擅自删除或替换原文件。

## 6. RegLoRA/SEFE 的运行时环境改写

`LLaVA/RegLoRA/scripts/MCITlib/Train/Task1.sh` 和 `Taskn.sh` 含 `pip install transformers==4.37.2`，且训练后调用 `scripts/merge_lora_weights.py`。其 pyproject 仍写 `transformers==4.31.0`，KeepLoRA 的 pyproject 则写 `4.37.2`。

这会造成“声明依赖”和“实际运行环境”不一致。v4 不允许在任务之间安装包；应将该方法的实际版本显式写入环境/方法兼容矩阵，并由作者确认是否需要独立环境。此问题也可能是 LLaVA 两快照无法合并的早期信号。

## 7. 训练脚本中的硬编码与运行时安装

- 训练/评测脚本中大量超参硬编码在 shell，而非 JSON：`mm_projector_lr 2e-5`、`lora_alpha = 2 * rank`、`model_max_length 2048`、`warmup_ratio 0.03`、`save_steps 50000` 等。迁移必须由脚本解析生成 YAML，并通过 T0。
- 按字面匹配统计，共有 **1,634** 处脚本级 `pip install -e .`（Train 426 / Eval 1,208，涉及 170 个文件）；v4 主循环不得在任务之间重复安装。统计命令：

  ```bash
  git grep -hF 'pip install -e .' -- '*.sh' | wc -l
  git grep -hF 'pip install -e .' -- '*/Train/*.sh' | wc -l
  git grep -hF 'pip install -e .' -- '*/Eval*/*.sh' | wc -l
  git grep -lF 'pip install -e .' -- '*.sh' | wc -l
  ```
- 已确认的运行时版本改写至少包括 RegLoRA 的 `pip install transformers==4.37.2`。
- 占位路径统计（`git grep`）：`/your_path/MCITlib_v3` 435 处、`/your_model_path` 372 处、`/your_data_path` 184 处、`/your_ckpts_path` 1,774 处；`sys.path.append('/your_path...')` 149 处。
- 另有大量写死的用户环境路径，例如 `/mnt/cache/guohaiyang/miniconda3/envs/coin/bin/python`，不能只替换 `/your_path` 就宣称路径清理完成。

## 8. 评测脚本中的可疑调用（保持原行为）

LLaVA 多数方法的 `scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh` 对 RS 任务调用 `llava.eval.CoIN.model_ai2d`，ModalPrompt 使用其对应的 ModalPrompt 模块。该名称与 RS 任务直觉不符，但目前没有足够证据证明它影响论文数值，故只登记，不改成看起来“正确”的模块。

v4 的评测任务集合、模块选择和参数顺序应由 `v3_command_dump.py` 解析后用 T3 比对；不能在统一 scorers 时按任务名重新猜测映射。

## 9. 调优器中的调试输出与死代码

- vendored peft 的 `lora.py` 含多处调试 `print`，不同分叉的打印行为不完全一致。
- `CoINMOEGate`、`CoINMOERouter` 等类在基础调优器中定义，但当前检索未发现实际实例化/调用；它们不能直接当作 v4 的有效路由实现。
- 多个 InternVL/Video-LLaVA/VideoLLaMA2 目录存在 `coinmoelora_new.py`，未检索到导入引用，初步判断为死代码，P2 删除或归档前再做一次全仓引用核查。
- OLoRA、DISCO、HiDe 变体在构造时保存/打印 `model.training`，并将训练标志传入层对象。`nn.Module.train()/eval()` 后续可能改写同名属性，T1 必须覆盖构造后切换 train/eval 的路径，不能只测构造时状态。

## 10. 指标与评测复现缺口

对仓库内 `.py` 和 `.sh` 检索 `MFT`、`MFN`、`MAA`、`BWT` 未发现计算脚本命中。论文指标计算需要 v4 新增 `mcitlib/eval/metrics.py`，但这属于复现已报告指标，不是新增算法功能；期望值只能来自论文附录矩阵，并需注明来源和矩阵布局。

## 11. SEFE 配置目录断链（L1 停止条件 S1）

`LLaVA/RegLoRA/scripts/MCITlib/` 中的 Train 和 Eval 脚本引用 `configs/train_configs/SEFE/...`，全仓共 61 处；但 `v3.0-final` 的 `configs/train_configs/` 只有 `RegLoRA/`，没有 `SEFE/`。例如 `SEFE/LLaVA/UCIT/train/task1.json` 不存在，而 `RegLoRA/LLaVA/UCIT/train/task1.json` 存在。

v3 shell 未设置 `set -e`，所以 `read_config` 的 `FileNotFoundError` 不会阻止后续命令。shim dump 因而能返回 0，但会产生无效 argv，例如空的 `--lora_r`、`--include localhost:0,-1`、空的 `--model-path` 和 `_merged`。该产物不能作为 T0/T3 期望值，也不能用“SEFE 就是 RegLoRA 别名”擅自替换路径；需要宏观 Agent/作者确认实际实验配置来源。

诊断产物：`/tmp/p0/v3_commands_test/sefe/ucit.json`（不入库，包含无效参数，仅用于定位）。

## 12. 当前不能宣称的事项

- 尚未完成 T0–T3，因此不能宣称 v4 与 v3 数值等价。
- 尚未在 GPU 上执行 T4/T5。
- 尚未证明所有 LLaVA 方法可以共享一个上游快照或一个 transformers 环境。
- 尚未确定 CL-MoE、HiDe-LLaVA、ModalPrompt 的上游再发布许可证。
- DoRA 文件头是 NVIDIA 专有声明，上游许可证仅限研究或评估用途、不符合 OSI；这是 MLOSS 发布阻塞项。v3 中没有任何脚本启用 DoRA，v4 按 NB-1 不包含它。
