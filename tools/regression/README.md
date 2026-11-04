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

每个组合写 `regression_out/<method>/<benchmark>/task<k>/{v3,v4}.log` 和 `comparison.json`。比较器读取日志中的 loss，要求 bf16 下最大相对误差不超过 `1e-3`。

## T5：训练 loss 全任务预检

```bash
# 默认组合：lora_ft:ucit、hide:mllm_dcl、sefe:mllm_acl
tools/regression/run_full.sh --execute
```

`--full` 不按 20 步截断，比较器使用 0.005 的默认容差作为日志预检；最终 MFN/MAA 的 0.5 个百分点判定需要另行由 `mcitlib metrics` 的输出计算。若 v3 没有同机重跑结果，按论文表 1 对照。

当前工具逐个运行训练命令，不会自动执行 KeepLoRA 的 Task0/pre/post、SEFE/KeepLoRA 的 merge 或后续 eval，因此它本身不是完整序列 runner，也不能单独完成 T5。完整任务编排和评测结果比较需在后续 GPU runner 中完成。

## 仅比较已有日志

```bash
python tools/regression/compare.py compare \
  --v3-log regression_out/lora_ft/ucit/task1/v3.log \
  --v4-log regression_out/lora_ft/ucit/task1/v4.log \
  --rtol 1e-3
```

注意：这些工具不会下载权重，也不会声称本地没有 GPU 时已经完成 T4/T5。
