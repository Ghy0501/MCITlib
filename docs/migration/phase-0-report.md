# Phase 0 report

日期：2026-11-03  
分支：`v4-dev`  
基线 tag：`v3.0-final`（`a3a622c9beb690e84c2923785279f179a6aac07e`）

## 完成项

- 在指定基线打本地 tag `v3.0-final`，创建并切换到 `v4-dev`；未改写历史、未删除分支、未推送远端。
- 复核仓库规模、方法目录数量、LLaVA 重复度和占位路径数量。
- 新增只读分析工具：
  - `tools/p0/inventory.py`：包行数、共有文件差异、单边文件和 vendored 包统计。
  - `tools/p0/match_upstream.py`：基于探针文件的上游 commit 最近候选匹配。
- 完成 `docs/migration/inventory.md`：
  - 47 个组合的逐方法差异表；
  - 四个骨干的上游候选 commit 与建议基线；
  - 方法核心位置、真实训练入口、shell 特定逻辑；
  - vendored peft、版本和许可证证据。
- 完成 `docs/migration/drift.md`：训练入口漂移、checkpoint 删除、CL-MoE 文件副作用、SMoLoRA pickle、运行时 pip、评测可疑调用和指标缺口。
- 完成 `docs/migration/moe_lora_variants.md`：基础调优器、OLoRA、CL-MoE、DISCO、HiDe、SMoLoRA 及 InternVL RegLoRA 额外变体的语义差异和子类化边界。
- 新增 `docs/migration/naming.md`：落实作者补充的 MCITlib 命名原则，区分用户-facing 方法名、内部实现名、`third_party` 原始名和 checkpoint 序列化兼容名。
- 通过实际克隆读取多个上游及方法来源仓库的 LICENSE；未凭记忆填写许可证。

## 测试结果

本阶段是盘点与冻结，没有实现 v4 训练/评测代码，因此没有宣称任何等价性测试通过。

| 层级 | 组合数 | 通过 | 失败 | 产物位置 |
|---|---:|---:|---:|---|
| T0 配置等价 | 未执行 | — | — | P1 `tools/migrate_v3_configs.py` 与测试待实现 |
| T1 模块等价 | 未执行 | — | — | P2 起实现 |
| T2 数据等价 | 未执行 | — | — | P2 起实现 |
| T3 命令等价 | 未执行 | — | — | P1 `tools/v3_command_dump.py` 待实现 |
| T4 短程训练 | 未执行 | — | — | GPU 脚本待提供 |
| T5 全程抽检 | 未执行 | — | — | 作者 GPU 执行 |
| P0 静态盘点 | 47 个组合 | 完成 | 不适用 | 本目录三份盘点文档及 `tools/p0/` |

## 关键盘点结论

1. LLaVA 的 12 份 `llava/` 合计 161,402 行；共有文件差异合计 5,224 行，与委托文档数字一致。
2. 上游最近候选：
   - LLaVA：`31cceb832f2f403b2cbf822d0f8fe689d72fe219` 覆盖 10/12 方法；KeepLoRA/RegLoRA 最近 `dcda07fb143f2faf023474b7237c88d1fb0fcd4b`。
   - InternVL：`f4e5aee1c306cbfd35e23da81d83c5d637ce9109`，路径 `internvl_chat_llava/`。
   - Video-LLaVA：`b6534e7e1b1cdf674811466c5581582a993213dc`。
   - VideoLLaMA2：`1c89aeb9bd55c594c03e960e0934dd90cb0fac7f`。
3. MoE-LoRA 去重不能只按 LLaVA 的 7 份文件处理：InternVL RegLoRA 有额外 MoE 变体；HiDe/DISCO 在 Video-LLaVA（以及 DISCO 在 VideoLLaMA2）有不同推理路径；VideoLLaMA2 多个变体跳过 `vision_tower`。
4. LLaVA 的 KeepLoRA/RegLoRA 使用较晚上游快照，且 RegLoRA 脚本运行时安装 `transformers==4.37.2`，与多数 LLaVA 方法的 `4.31.0` 存在环境风险。
5. `LoRA-FT/peft` 相对 peft 0.4.0 不是只有 DoRA，还含 `lora.py` 调试输出和其他保存/加载改动；InternVL 声明 peft 0.6.0 但 vendored 包版本为 0.4.0，运行时导入路径尚未确定。

## 与设计文档的偏离

- **偏离点：**将“7 个 MoE-LoRA 分叉”扩展为按（方法，骨干）记录的变体矩阵。  
  **原因：**实测发现 InternVL RegLoRA 额外变体及 HiDe/DISCO 的跨骨干推理差异。  
  **是否需要作者确认：**需要，在 P2 接口评审中确认变体注册方式。
- **偏离点：**新增 `docs/migration/naming.md`。  
  **原因：**作者补充要求 v4 自有文件和标识突出 MCITlib，而不是沿用 CoIN/MoELoRA 等上游内部命名。  
  **是否需要作者确认：**建议确认共享调优器的 `MCITExpertLora*` 命名及旧 PeftType 兼容策略。
- **偏离点：**P0 只完成静态候选匹配，没有把上游候选当成已验证的精确快照。  
  **原因：**探针匹配不能替代逐文件和数值验证。  
  **是否需要作者确认：**需要确认建议基线后，P2 才能 vendored 固定。
- **偏离点：**DoRA 暂不处理。  
  **原因：**作者明确指示“可以先不考虑 dora”。  
  **是否需要作者确认：**最终发布前仍需决定许可证和代码去留。

## 发现的 v3 漂移 / 可疑行为（已记入 drift.md）

- 实际 MoE 入口是 `train_MOE.py`，不能只按 `train.py` 判断多图、SAT 或 checkpoint 清理行为。
- 多数 LLaVA 实际训练入口使用 `shell=True` 的 `find ... rm -rf` 删除 checkpoint。
- CL-MoE forward 读取硬编码 task 文件并按秒写统计文件。
- SMoLoRA 使用 `pickle.load` 读取指令 embedding。
- RegLoRA 训练脚本运行时改写 transformers 版本并 merge LoRA。
- 多数 LLaVA DCL RS 评测脚本调用 `model_ai2d`，保持原行为，不顺手修复。
- 存在 1,634 处脚本级 `pip install -e .`（Train 426 / Eval 1,208，涉及 170 个文件），以及大量占位路径和写死的用户环境路径；统计命令已写入 `drift.md`。
- 未发现 MFT/MFN/MAA/BWT 计算脚本；指标复现需后续补齐并用论文矩阵测试。

## 触发的停止条件

- **条件 5：上游许可证无法确定。** 当前实际克隆的 `ECNU-ICALK/CL-MoE`、`Ghy0501/HiDe-LLaVA`、`AuroraZengfh/ModalPrompt` 未发现 LICENSE 文件。对应方法的迁移和再发布暂停，等待作者补充许可证或作出书面决定。
- **条件 5（发布阻塞，v4 按 NB-1 排除）：DoRA。** v3 的 24 份 `tuners/dora.py` 内容相同，文件头是 NVIDIA 专有声明，上游许可证仅限研究或评估用途、不符合 OSI；v3 中没有任何脚本启用 DoRA。v4 不 vendor `dora.py`，并删除相关注册和参数。
- **条件 2 风险（尚未正式触发）：** LLaVA KeepLoRA/RegLoRA 的较晚上游快照和环境差异，可能在 P2 证明无法用“一份上游 + 方法模块”表达；若隔离失败必须停止汇报。
- 未发生删除用户数据、改写 git 历史或远端推送。

## 需要作者执行/确认的事项

1. 确认四个骨干的上游基线候选，尤其是 LLaVA 采用 `31cceb83` 作为共享基线、KeepLoRA/RegLoRA 差异按方法模块隔离的方案。
2. 补充或确认 CL-MoE、HiDe-LLaVA、ModalPrompt 的再发布许可证；在确认前不迁移这三项代码。
3. 确认命名规范：共享调优器采用 `MCITExpertLora*` 等 MCITlib 标识，论文方法名只作为注册表 ID/别名；旧 PeftType 和 state-dict 键通过兼容层保留。
4. 确认 RegLoRA/KeepLoRA 的 transformers 4.37.2 是否需要独立环境；不要依赖训练脚本运行时 pip。
5. 在干净 InternVL 环境记录实际 `peft.__file__` 和 `peft.__version__`，以解决 0.6.0 声明与 vendored 0.4.0 的冲突。
6. 后续 GPU 阶段运行 T4/T5；当前没有 GPU 等价性结果。
7. DoRA 按已定的 NB-1 从 v4 排除，并在 `CHANGELOG.md` 和 `behavior_changes.md` 中记录原因；不再需要作者选择是否 vendor。

## 下一阶段计划

在作者确认上游基线和许可证阻塞项后进入 P1：

- 建立 `mcitlib/` 骨架、注册表、路径解析、分层 YAML 配置和 `argparse` CLI。
- 实现 `tools/migrate_v3_configs.py`、`tools/v3_command_dump.py`，先完成 T0/T3 的静态产物。
- 添加 Apache-2.0、SPDX 工具、第三方通知初稿和 CI。
- 按 `naming.md` 设计共享调优器与旧 checkpoint 兼容层。
- P2 仅先做 LLaVA + LoRA-FT/Replay/O-LoRA/HiDe 原型，完成接口评审后再扩展到其余方法和骨干。
