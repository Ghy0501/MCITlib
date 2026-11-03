# v4 已批准的非数值行为变更（NB 项）

本文件登记 LLaVA 阶段允许的行为变更。除下列项目外，任何行为差异都按 S1 停止并汇报。每项必须独立提交，提交信息为 `behavior(NB-n): ...`，并附对应测试产物。

| ID | 变更 | 数值理由 | 验证方式 | 状态 |
|---|---|---|---|---|
| NB-1 | 不包含 DoRA：不 vendor `tuners/dora.py`，删除 `dora_enable`、`DoraConfig` 及 peft 中 DoRA 相关改动 | v3 没有脚本启用 DoRA，`dora_enable=False`；其 NVIDIA 专有/仅研究许可不符合 MLOSS 开源要求 | 全仓配置/脚本无启用项；T3 白名单只列出 DoRA 参数和命令差异 | 已批准，待实施 |
| NB-2 | CL-MoE 任务标识由 `RunContext` 注入，不读硬编码 `task.txt`；router 统计改为 `diagnostics.router_stats`，默认关闭 | router 数值不依赖该文件；注入的任务 ID 与 v3 文件值相同 | T1 对同一输入比较 router 输出；测试文件读取替换和诊断开关默认值 | 已批准，待实施 |
| NB-3 | 删除 vendored peft 等文件中的调试 `print` | 只改变标准输出，不改变张量、参数或控制流 | T1/T3 数值和 argv 比对；检查没有 print 影响异常路径 | 已批准，待实施 |
| NB-4 | 用 Python 清理训练结束时的 `checkpoint-*`，仅限当前 `output_dir` 直接子目录并校验前缀，不用 `shell=True` | 删除目录集合和时机与 v3 相同 | 单元测试构造嵌套/越界目录，验证删除集合；T3 检查调用时机 | 已批准，待实施 |
| NB-5 | 移除所有运行时 `pip install`，包括 RegLoRA 的 `transformers==4.37.2` | 由 `llava` 或 `llava-t437` 环境预先保证依赖，训练过程不改变环境 | 环境检查测试；T3 白名单列出被移除的安装命令；运行时不调用 pip | 已批准，待实施 |
| NB-6 | 骨干适配器加载时注入 `mm_text_tower`、`mm_text_select_layer` 等配置，替代手改模型文件 | 注入值与 `examples/` 参考配置完全一致 | T2/T3 比较注入后的 config/generation config 与参考文件 | 已批准，待实施 |

## 提交和报告要求

- NB 项不得与方法迁移混交；每个 NB 项独立提交。
- 每个 NB 项完成后，在对应 `llava-L<n>-report.md` 中填写提交哈希、测试命令和产物路径。
- 若验证发现数值或评测协议差异，立即按 S1 停止，不得用“非数值变更”解释带过。

## 后续作者明确授权的配置集成修复

以下不是实现方自行添加的 NB 项，而是作者在 L1 诊断后明确授权的配置/序列处理，单独记录以避免与 NB-1…NB-6 混淆：

| ID | 授权内容 | 实现范围 | 验证/交付说明 |
|---|---|---|---|
| AR-1 | SEFE 与 RegLoRA 使用同一配置来源 | 仅在 dump 临时副本映射 `train_configs/SEFE` → `RegLoRA` | JSON `config_aliases`；v3 原树不变 |
| AR-2 | router 使用独立新数据路径，允许先用占位符 | 补 `router.json` 和数据根目录占位；不套用 `train_path` | `placeholders.md`；GPU 节点替换并核对真实数据 |
| AR-3 | 统一因 v3 集成不完整而注释的训练段 | 临时副本恢复 SMoLoRA DCL Task1、ModalPrompt ACL Task1–4，保持 benchmark 顺序 | JSON `sequence_repairs`；36 条序列训练覆盖测试 |
| AR-4 | 硬编码机器路径统一改为占位符 | checkpoint、CLIP 模型和旧 Python 路径转为根目录占位符 | `placeholders.md`；YAML 无 `/home`/`/mnt` 可执行路径值 |

这些授权修复不代表对应 GPU 数值已经验证，T4/T5 仍是必要条件。
