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
