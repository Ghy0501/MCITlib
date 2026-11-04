# LLaVA L5 真实执行、评测与指标报告

日期：2026-11-04。状态：离线逻辑和 CPU 验证完成；真实模型/GPU 的 T4/T5 仍需作者在节点执行。本报告不把 dry-run、玩具模型或合成打分结果描述为实验复现。

## 1. 真实顺序执行与断点续跑

提交：`da5e1bd feat(runner): execute resumable command sequences`。

- `mcitlib train`、`mcitlib eval` 不再返回占位提示，按 dry-run 清单逐条执行。
- 每条命令使用 argv，不经过 shell；Python 使用当前环境解释器，工作目录和路径模板统一解析。
- 每步写独立日志及原子 JSON 标记，标记包含命令哈希、退出码、起止时间和日志路径。
- `--resume` 只跳过同内容哈希且状态为 `completed` 的步骤；修改后的命令不会误跳过。
- 任一步非零退出立即写失败标记并停止后续步骤。
- 离线测试实际运行 Python 小命令，验证顺序、文件副作用、失败即停和修复后 resume。

## 2. B4：CoIN 与方法私有评测

提交：`a8aa558 feat(eval): migrate CoIN and method-specific runners`。

- 公共 CoIN 实现位于 `mcitlib/backbones/llava/eval_runners/coin/`；只有内容不同的评测模块进入各方法 `_<method>/eval_runners/`。
- 统一 eval 入口在执行目标模块前安装方法私有模型和 builder；训练阶段与 continual eval 的参数保持 v3 顺序。
- MR-LoRA 的 `model_dcl_router` 和 UCIT/DCL/ACL 三个结果提取脚本均已迁移；结果根由 `MCITLIB_RESULTS_ROOT` 控制，默认 `./results`。
- T3 白名单只还原登记的模块入口和脚本路径别名；其他 argv、任务号、步骤类型、顺序、GPU 分配仍逐项比较。
- 178 个“方法 × 原始评测模块”组合均与对应 v3 文件比较，只允许新增许可头和登记的 `coin_utils` 导入改写。
- 全部 **1696 条普通序列命令**及 3 条 router 序列的 `-m` 模块、eval 实际目标、直接 Python 脚本和 deepspeed 配置均存在。

## 3. 指标实现

提交：`2aca98b feat(eval): add continual metrics and general benchmarks`。

`mcitlib/eval/metrics.py` 使用论文定义和 `math.fsum`：

- `MFT = mean_i A[i][i]`；
- `MFN = mean_j A[T-1][j]`；
- `MAA = mean_i mean_{j<=i} A[i][j]`；
- `BWT = mean_{j<T-1}(A[T-1][j] - A[j][j])`；单任务时为 0。

输入可以是带 `matrix` 的 JSON、`matrix.json`/`accuracy_matrix.json`，或 CoIN 的 `each_dataset/<任务>/<方法>-taskN/Result.text` 目录。目录模式要求完整下三角，缺失单元和任务顺序不明确都会报错，不做零填充或字母顺序猜测。

```bash
mcitlib metrics --run-dir results/UCIT --benchmark ucit --output metrics.json
```

### 论文源数据核验

fixture：`tests/fixtures/paper_matrices/llava.json`，只读解析 `paper/sample.tex` 的活跃 LLaVA 表 4/6/8，并取表 1 作为参照；共 24 个矩阵、8 个方法，`ModalPrompt*` 不含在内。

以下 7 项是论文表 1 与详细矩阵自身的差异。测试同时固定“矩阵重算值”和“表 1 印刷值”，其他 89 个指标仍要求两位小数一致：

| 基准 | 方法 | 指标 | 矩阵重算 | 表 1 |
|---|---|---|---:|---:|
| UCIT | CL-MoE | MFT | 71.45 | 71.46 |
| MLLM-DCL | O-LoRA | BWT | -12.03 | -12.53 |
| MLLM-DCL | MoELoRA | BWT | -12.70 | -12.71 |
| MLLM-DCL | CL-MoE | BWT | -13.96 | -13.97 |
| MLLM-ACL | LoRA-FT | MFT | 41.27 | 41.28 |
| MLLM-ACL | O-LoRA | MFN | 37.16 | 37.17 |
| MLLM-ACL | SEFE | MFT | 40.98 | 40.99 |

没有修改 `paper/`，也没有为了让测试通过而调整公式或期望。T5 对照论文时应同时保留该说明，优先使用同机 v3 重跑结果。

## 4. 通用基准评测

配置：`configs/benchmarks/general.yaml` 与 `configs/data_configs/General_benchmark/{pope,mme,mm_bench,seed}.json`。

```bash
# 默认读取该方法 MLLM-DCL Task5 的 eval 配置
mcitlib eval --method lora_ft --benchmark general --after-task 5 --paths configs/paths.yaml

# 也可显式指定 checkpoint 和分块数
mcitlib eval --method hide --benchmark general --model-path /path/to/checkpoint --num-chunks 2 --paths configs/paths.yaml
```

- POPE、MME、MMBench、SEED-Bench 各执行分块推理、JSONL 合并和打分，结果写 `./results/General_Benchmarks/`。
- 推理适配自上游 `model_vqa_loader.py` 和 `model_vqa_mmbench.py`，只新增私有 builder 所需的可选参数转发。
- POPE 输出各类 accuracy/F1；MME 按 `(acc + acc_plus) * 100` 汇总 perception/cognition；SEED-Bench 输出 all/image/video 和题型准确率。
- MMBench 使用明确的离线选项提取，报告 missing/unparsed；它不冒充官方在线提交或 LLM 答案抽取，论文表 3 数值由作者在 GPU 节点核对。
- 四个打分器均有 CPU 合成 fixture，general 命令生成、路径解析、私有模型参数和 CLI dry-run 均有测试。

## 5. 最终验证

| 环境/检查 | 结果 |
|---|---|
| Python 3.9 + transformers 4.31，全部非 t437 测试 | **204 passed, 2 skipped, 1 deselected** |
| transformers 4.37.2，SEFE/KeepLoRA/environment | **12 passed** |
| Python 3.10.22 + 实际 `pip install '.[dev]'` | **204 passed, 2 skipped, 1 deselected**；与 3.9 一致 |
| 全命令引用 | 1696 条普通命令 + 3 条 router 通过 |
| ruff / SPDX / third-party / diff check | 全部通过 |
| 其他骨干与只读目录 | 相对 `v3.0-final` 无差异 |

日志：`/tmp/p0/l5-final-431.out`、`/tmp/p0/l5-final-t437.out`、`/tmp/p0/l5-final-py310.out`。

## 6. 剩余边界

- 本机没有执行真实 7B 模型、GPU 训练或论文数值复现；作者仍需运行 T4/T5。
- 普通十个方法与 SEFE/KeepLoRA 分别使用 `llava`、`llava-t437` 环境；NB-5 所需依赖已显式写入两个环境清单。
- ModalPrompt × UCIT Task2 的 output_dir 仍保留 v3 原值，AR-5 尚未授权修改。
- CL-MoE、HiDe-LLaVA、ModalPrompt 的再发布许可和 DoRA 排除仍是发布层面的独立事项。
- 其他三个骨干的迁移不在本次 LLaVA L5 范围内，未提前修改或删除其目录。
