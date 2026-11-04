# v4 占位符与参考配置修复说明

本文说明 LLaVA 阶段生成的 YAML、命令清单和 dump 临时副本中占位符的目的。它们不是可直接运行的真实路径，也不表示相应数据或权重已经存在。作者在 GPU 节点运行前，需要把这些值替换为节点上的实际路径，并核对数据内容。

## 1. 路径占位符

| 占位符 | 用途 | 需要替换为 |
|---|---|---|
| `${model_root}` / `/your_model_path` | LLaVA-1.5-7B、CLIP 视觉塔和 projector 的模型文件根目录 | GPU 节点上的模型目录 |
| `${data_root}` / `/your_data_path` | 训练、测试、Replay 和 router 数据根目录 | GPU 节点上的数据目录 |
| `${ckpt_root}` / `/your_ckpts_path` | 每个任务的输出、previous model、梯度/权重提取和 merge 结果 | GPU 节点上的 checkpoint 根目录 |
| `${repo_root}` | dump 中的 v3 临时根或迁移工具需要的参考树位置 | 仅工具运行时使用；不是 v4 训练硬编码路径 |
| `${python_root}` | 旧脚本注释中遗留的 Python 环境路径 | 可选；v4 不依赖该旧环境路径 |
| `__V3_ROOT__` | dump JSON 中对临时目录的规范化标记 | 仅用于测试比较，不需要用户创建 |

`configs/paths.example.yaml` 提供 `model_root`、`data_root`、`ckpt_root` 的用户配置入口，也支持 `MCITLIB_MODEL_ROOT`、`MCITLIB_DATA_ROOT`、`MCITLIB_CKPT_ROOT` 环境变量覆盖。

## 2. router 数据占位符

MR-LoRA router 使用**独立的新数据**，不是 `train_path` 的别名。v3 中缺失的首任务字段按其他任务的目录惯例补为：

- `.../<任务目录>/router.json`；
- UCIT 的 `router_folder` 为 UCIT 数据根目录；
- ACL/DCL 的 `router_folder` 为 `MLLM-CL` 数据根目录。

这些补值记录在每份 router JSON 的 `config_placeholders` 中。GPU 运行前必须确认 router JSON 的真实内容、字段格式和图像目录，不能仅凭文件名开始正式实验。

## 3. 配置和脚本别名

- `config_aliases`：作者已确认 `SEFE` 就是 `RegLoRA`，因此临时参考副本将 `train_configs/SEFE` 映射到 `train_configs/RegLoRA`。这是配置来源映射，不是算法替换。
- `script_aliases`：MR-LoRA router 训练脚本中的 UCIT/ACL 评测文件名与目录实际文件不一致，临时副本分别映射到 `Eval_ucit.sh`、`Eval_acl.sh`。该映射需要 GPU 阶段核对，不能视为已经验证的 v3 原始行为。

v3 原树和 `../MCITlib_v3_ref` 不被这些临时映射修改。

## 4. 训练序列统一

作者授权统一 v3 中因集成不完整而注释掉的训练段，并保持同一 benchmark 的任务顺序：

- SMoLoRA × MLLM-DCL：恢复 Task1；
- ModalPrompt × MLLM-ACL：恢复 Task1–4。

恢复发生在 dump 的临时副本，JSON 中以 `sequence_repairs` 记录任务编号。恢复后的 36 条普通序列都包含完整的 Task1…TaskN 训练命令。

ModalPrompt × UCIT 的 Task2 并非注释缺失：其 v3 JSON 的 `output_dir` 错写为 Task1。dump 使用其一基 `cur_task` 标记任务，但**保留这个 output_dir 原值**，没有擅自修正实验行为。该条已登记在 `drift.md`，GPU 运行前需要作者确认。

## 5. 不作为等价性声明的内容

- 占位路径的存在不等于数据/权重已经验证。
- 序列恢复是作者明确授权的配置集成修复，不是对 v3 原始脚本行为的声称复现。
- 当前通过的是 CPU 的配置/命令检查和小模型/小矩阵单元测试；真实权重、完整 checkpoint 链路以及 T4/T5 仍需在 GPU 上执行。

## 6. SEFE 与 KeepLoRA 的额外产物

- SEFE 的 key-element 输入、LoRA 输出和 `_merged` 模型都由 `${ckpt_root}` 派生；方法脚本别名指向 `mcitlib/methods/_sefe/`，不是指向 v3 机器路径。
- KeepLoRA 的 Task0 权重子空间写入 `${ckpt_root}/.../Task0_llava_lora_gradients/`；每个任务的 pre/post 梯度、fixed-rank 初始化、累计子空间和 merge 输出同样位于 `${ckpt_root}` 下。Task0 在每个 benchmark 完整序列中只执行一次。
- KeepLoRA 的 `CUDA_VISIBLE_DEVICES` 白名单保留 v3 各 chunk 的 GPU 分配；它只是命令环境记录，当前 dry-run 不会启动并行任务。真实执行时仍需按顺序等待 chunk 完成后再 merge/SVD。
- 这些目录在 GPU 节点必须可写且有足够空间；占位符替换后仍需确认旧子空间文件属于正确的上一任务，不能混用不同 benchmark 或方法的 checkpoint。
