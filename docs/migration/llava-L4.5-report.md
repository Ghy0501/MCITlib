# LLaVA L4.5 阻塞修复与去重报告

日期：2026-11-04。范围：宏观 Agent 的 L3/L4 审查意见 B1–B3、S1–S3，以及 GPU T4 前必须完成的训练接线修复。本阶段不宣称已执行真实 GPU 训练。

## 1. 审查阻塞项闭环

| 项 | 修复 | 验证 | 提交 |
|---|---|---|---|
| B1 | 原样迁移 `zero2.json`、`zero3_offload.json` 到 `configs/deepspeed/`；训练 argv 使用绝对路径 | 两个文件与 v3 逐字节一致；全部训练 argv 可被实际 `HfArgumentParser` 解析；新增全命令代码引用检查 | `df0fff5` |
| B2 | T4 任务 2 先用 v3 完整准备并缓存共享 previous checkpoint；v3/v4 当前任务使用隔离输出；严格比较退出状态与 loss 数 | 假训练脚本覆盖 v4 提前崩溃、loss 数不足、工具主动终止和输出目录隔离 | `1936a79` |
| B3 | CI `fetch-depth: 0`，按固定 commit 建立 `../MCITlib_v3_ref`；两个环境均设置 `MCITLIB_REQUIRE_V3=1` | 参考 worktree 缺失且该变量存在时测试直接失败，不再把全部等价测试 skip 成绿色 | `df0fff5` |

## 2. 去重与 NB-6

提交：`a342915 refactor(llava): share CLIP features and complete method training wiring`。

- 新增 `mcitlib/backbones/llava/text_tower.py`，共享 CLIP 文本塔和带 projection 的 CLIP 特征；HiDe/DISCO/ModalPrompt/t437 builder 复用该能力。
- 公共 t437 快照只保留一份：`mcitlib/backbones/llava/snapshots/dcda07fb/`。SEFE 仅保留带正则化的 Llama 类，KeepLoRA 直接复用普通类。
- HiDe/DISCO 的文本塔参数、任务标识仍从各自 eval/train 配置传递；`LlavaAdapter` 向私有 builder 转发 `text_tower`。
- NB-6 已实施：`mm_text_tower`、`mm_text_select_layer` 和 generation config 的默认值处理与 `examples/` 参考配置比较。
- 原 T1/T2 继续通过，新增共享 CLIP 前向等价测试；没有修改既有数值期望。

## 3. 测试强化与训练接线

- HiDe 第 31 层专家权重改为非均匀值，并增加非第 31 层融合路径；DISCO 增加非默认 `mask_signal`。
- `moe_train.py` 使用 `MCITExpertLoraConfig`，不再引用兼容旧类名。
- 12 个方法 × 3 基准的全部 180 条训练命令均通过实际 parser；不是只比较 argv 字符串。
- 补齐训练入口中的方法参数/钩子：SEFE 正则化输入，ModalPrompt prompt tokenizer、prompt 感知数据长度、前一任务加载和保存，KeepLoRA SVD 初始化与冻结。
- 修复 runner 创建真实方法插件；Replay 使用 `replay_path` 和 `router_folder`，不再保留 v3 dump 中缺失值导致的无效 argv。
- Replay 所需字段原不在 v3 数据 JSON 中，现以独立 `replay.json` 和图像根目录占位符显式记录；真实数据仍需 GPU 节点核对。

## 4. 验证结果

- L4.5 完成时，transformers 4.31：**184 passed, 2 skipped, 1 deselected**。
- transformers 4.37.2 的 SEFE/KeepLoRA：**12 passed**。
- ruff、SPDX、`git diff --check` 通过。
- InternVL、Video-LLaVA、VideoLLaMA2、`paper/`、`reviews/`、`design/` 相对基线无修改。
- Python 3.10.22 的 `pip install '.[dev]'` 已在隔离环境实际执行；最终同一非 t437 测试集合为 **204 passed, 2 skipped, 1 deselected**，详见 L5 报告。

日志：`/tmp/p0/l45-431.out`、`/tmp/p0/l45-t437.out`。这些均为临时验证产物，未提交。

## 5. 仍需作者执行/决定

- 可在 GPU 节点开始 T4，推荐先跑 LoRA-FT、O-LoRA、HiDe、SEFE 的代表组合；工具用法见 `tools/regression/README.md`。
- T4/T5 的真实数值结果不在本机报告中伪造。
- ModalPrompt × UCIT Task2 的 `output_dir` 仍保留 v3 的 Task1 值；AR-5 需作者明确授权后才能改为 Task2。
- Replay 与 MR-LoRA 的独立数据路径、真实模型路径和图像目录必须在 GPU 节点设置并检查。
