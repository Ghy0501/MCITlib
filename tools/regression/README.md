# LLaVA GPU regression tools

这些脚本只在 GPU 节点执行真实训练。默认 `--dry-run`（只打印 v3/v4 argv），不会在本机启动训练。训练前：

1. 准备 `../MCITlib_v3_ref`（`v3.0-final` worktree），或设置 `MCITLIB_V3_ROOT`。
2. 按 `docs/migration/placeholders.md` 设置 `MCITLIB_MODEL_ROOT`、`MCITLIB_DATA_ROOT`、`MCITLIB_CKPT_ROOT`，并核对 router 数据。
3. 为普通 10 个方法准备 `llava` 环境，为 SEFE/KeepLoRA 准备 `llava-t437` 环境。运行真实命令前不要执行脚本级 `pip install`。

## T4：前 20 个优化步

```bash
# 预览命令
tools/regression/run_short.sh --combo lora_ft:ucit

# GPU 上执行一个组合
tools/regression/run_short.sh --combo lora_ft:ucit --execute

# 全部 36 个组合的 task1/task2（耗时长，建议分组合执行）
tools/regression/run_short.sh --execute
```

每个组合写 `regression_out/<method>/<benchmark>/task<k>/{v3,v4}.log` 和 `comparison.json`。比较器读取日志中的 loss，要求 bf16 下最大相对误差不超过 `1e-3`，且两侧 loss 数都恰好等于 `max_steps`、进程正常退出或由工具主动终止；任一侧提前崩溃都会失败。

任务 2 及之后先用 v3 完整执行之前任务及当前任务所需的训练前依赖，结果缓存到 `regression_out/shared-checkpoints/`。共享准备只在全部依赖成功后写完成标记；v3/v4 当前任务分别写入各自 `v3-checkpoints/`、`v4-checkpoints/`，不会互相覆盖。

## T5：训练 loss 预检与完整序列

```bash
# 旧工具：仅做全任务训练 loss 预检，不执行 extra/eval
# 默认组合：lora_ft:ucit、hide:mllm_dcl、sefe:mllm_acl
tools/regression/run_full.sh --execute

# 新统一 runner：完整执行训练、方法 extra 和迁移后的 CoIN eval
mcitlib train --method lora_ft --benchmark ucit --paths configs/paths.yaml
mcitlib train --method hide --benchmark mllm_dcl --paths configs/paths.yaml
mcitlib train --method sefe --benchmark mllm_acl --paths configs/paths.yaml

# 从 CoIN Result.text 恢复准确率矩阵并计算论文指标
mcitlib metrics --run-dir results/UCIT --benchmark ucit
mcitlib metrics --run-dir results/DCL --benchmark mllm_dcl
mcitlib metrics --run-dir results/ACL --benchmark mllm_acl
```

`run_full.sh --full` 不按 20 步截断，比较器使用 0.005 的默认容差作为日志预检。它仍然只是训练 loss 工具，不替代统一 runner。统一 runner 逐步写日志和完成标记，失败后停止，`--resume` 可继续。最终 MFN/MAA 与 v3 同机重跑结果的差距应不超过 0.5 个百分点；没有 v3 重跑结果时，与论文表 1 对照，并注意 `llava-L5-report.md` 登记的论文源数据差异。

## 仅比较已有日志

```bash
python tools/regression/compare.py compare \
  --v3-log regression_out/lora_ft/ucit/task1/v3.log \
  --v4-log regression_out/lora_ft/ucit/task1/v4.log \
  --rtol 1e-3
```

注意：这些工具不会下载权重，也不会声称本地没有 GPU 时已经完成 T4/T5。
