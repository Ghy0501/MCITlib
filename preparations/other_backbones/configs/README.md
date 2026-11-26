# 骨干模型的离线配置清单

本目录工具只生成数据和配置清单，不启动模型执行。InternVL 的统一入口见[实现说明](../../../docs/internvl.md)和[四卡 sample 队列](../../../docs/internvl_sample_validation.md)；Video-LLaVA、VideoLLaMA2 仍需执行入口接入与 GPU 验证。离线清单与实际实验状态分开记录。

## 可生成的配置

| 版本 | 目录 | 主训练预算 |
| --- | --- | --- |
| sample，计划四卡 | `sample-4gpu` | 每卡 batch 1、累积 16，名义全局 64 |
| sample，计划两卡 | `sample-2gpu` | 每卡 batch 1、累积 32，名义全局 64 |
| 原全量预算 | `full-native` | 保留每个方法、任务原来的 batch、epoch 和辅助参数 |

每套包含 35 个模型/方法组合、47 个 benchmark 变体，共 1050 个 JSON 文件，包含模型、数据、训练、评测、辅助配置和 manifest。sample 普通主训练采用 LR `2e-4`、1 epoch；ModalPrompt 为 10 epochs，router 沿用各模型原辅助预算。资源参数是离线计划。

配置按 `<模型>/<方法>/<benchmark>/{train,eval,data}/taskN.json` 组织。例如：

- InternVL / UCIT：`训练`、`数据`。
- Video-LLaVA / CL-VISTA：`训练`、`数据`。
- VideoLLaMA2 / CL-VISTA：`训练`、`数据`。

InternVL 复用既有 UCIT、MLLM-DCL 的 3000/1000 样本；两个视频模型共用新的 CL-VISTA 样本。VideoLLaMA2 不提供 KeepLoRA 配置。MLLM-ACL 保持低优先级，未新增 sample。

## 数据入口

视频任务顺序为 `counting → space → traffic → movie → gui → science → sports → star`。主样本在项目的 `data/validation_3k_1k_video/<task>/`，包含 `train.json`、`test_questions.json`、`test_answers.json`；独立 Replay/router 数据在 `data/validation_3k_1k_video_aux/<task>/`。原视频地址保留不变。

视频 Replay 每阶段为当前 3000 条原问答加每个历史任务 100 条原问答；router 文件保留原来的累计 100–800 条路由训练记录。视频 MR-LoRA 的跨域结果合并和 ID 处理尚未通过验证，输入数据就绪不代表最终路由评测可运行。

## 通用视频集占位

按用户要求，MVBench、LongVideoBench、Video-MME、MMVU、NExT-QA 仅在 [地址模板](general-video.example.json)保留占位符，未下载或采样。

生成的 plan 和 effective 配置由本目录 `.gitignore` 排除。可用[准备工具](../README.md)指定根路径并生成配置；各目录的 `manifest.json` 记录当前输入文件的就绪状态。
