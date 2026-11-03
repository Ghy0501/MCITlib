# v4 命名与仓库身份规范

版本：P0 草案（2026-11-03）

## 1. 原则

v4 的自有代码必须让维护者一眼看出它属于 **MCITlib**，而不是继续沿用某个上游方法仓库的内部命名。方法论文名可以作为用户-facing 的插件名和别名保留，但不能继续作为共享基础设施的命名来源。

因此：

- `mcitlib/`、`tools/`、`tests/` 中的新文件、类、函数和配置键采用 MCITlib 自己的命名。
- `third_party/` 保持上游原始文件名、类名和版权声明，不为统一命名而改写上游。
- 不在共享调优器中保留 `CoINMOE*`、`CLMoEMOE*`、`HiDeMOE*` 这类把某个方法/上游项目名嵌入基础设施的类名。
- 不通过改名改变任何数值、参数初始化顺序、state-dict 键或 v3 checkpoint 的可加载性。

## 2. 用户-facing 名称与内部名称分离

注册表中的方法 ID 仍使用论文名，保证论文和 v3 配置可对应：

| 用户-facing ID | v3 目录/别名 | 内部实现命名示例 |
|---|---|---|
| `lora_ft` | `LoRA-FT` | `MCITLoraFTMethod` |
| `replay` | `Replay` | `MCITReplayMethod` |
| `moelora` | `MoELoRA` | `MCITExpertLoraMethod` |
| `mrlora` | `MR-LoRA` | `MCITRoutedExpertLoraMethod` |
| `olora` | `OLoRA` | `MCITOrthogonalExpertLoraMethod` |
| `clmoe` | `CL-MoE` | `MCITCLMoEMethod` |
| `disco` | `DISCO` | `MCITDISCOMethod` |
| `hide` | `HiDe` | `MCITHiDeMethod` |
| `smolora` | `SMoLoRA` | `MCITSeparableExpertLoraMethod` |
| `modalprompt` | `ModalPrompt` | `MCITModalPromptMethod` |
| `sefe` | `RegLoRA` | `MCITSEFEMethod` |
| `keeplora` | `KeepLoRA` | `MCITKeepLoRAMethod` |

上表中的“内部实现命名示例”是命名方向，不是已经完成的代码变更。论文名、上游名只应出现在注册表别名、文档、出处和兼容层中。

## 3. 共享调优器命名映射

共享的 MoE/专家 LoRA 基础设施不再叫 `clitmoelora`、`coinmoelora` 或 `ptmoelora`。建议映射如下：

| v3 名称 | v4 建议名称 | 说明 |
|---|---|---|
| `CoIN/peft/tuners/clitmoelora.py` | `mcitlib/tuners/mcit_expert_lora.py` | 唯一共享基础调优器 |
| `coinmoelora.py` | 同上 | 与基础版同义的另一文件名，不另存一份 |
| `ptmoelora.py` | 同上或兼容别名 | 当前四份内容相同；不得复制成第四份实现 |
| `smolora.py` | `mcitlib/tuners/mcit_separable_expert_lora.py` | SMoLoRA 的可分离专家路由实现 |
| `CoINMOELoraConfig` | `MCITExpertLoraConfig` | Python 类名 |
| `CoINMOELoraModel` | `MCITExpertLoraModel` | Python 类名 |
| `CoINMOELoraLayer` | `MCITExpertLoraLayer` | Python 类名 |
| `CoINMOELoraLinear` | `MCITExpertLoraLinear` | Python 类名 |
| `CoINMOELinearA/B` | `MCITExpertBankA/B` | Python 类名 |
| `CoINMOEExpert` | `MCITExpert` | Python 类名 |
| `HiDeMOE*` / `CLMoEMOE*` | `MCIT...` 变体类 | 变体放在 `tuners/variants/`，不复制基础类 |

`MoE` 是通用技术术语，不是本规范要消除的上游项目名；但共享实现的完整标识应以 `MCIT` 开头。方法插件文件可以使用论文方法名，例如 `methods/olora.py`，因为它们是用户选择的方法，而不是共享基础设施。

## 4. 序列化兼容是硬约束

Python 标识符可以改名，以下序列化内容不能未经兼容层直接改动：

1. checkpoint 的 `state_dict` 键；
2. `adapter_config.json` 中的 `peft_type`、配置字段和已有自定义字段；
3. v3 产生的 `non_lora_trainables.bin`、LoRA 权重和其他方法产物的读取格式。

`_peft_registry.py` 必须提供旧名称到新实现的映射，例如 v3 已发现的：

- `MOE_LORA_CoIN`；
- `MOE_LORA_HiDe`；
- `MOE_LORA_CLMoE`；
- `SMOLORA`。

如果 v4 新写出 `MCITLIB_*` PeftType，必须同时验证：旧 v3 checkpoint 可读、新 checkpoint 可回读、相同输入的模块输出一致。若这一兼容性尚未被测试证明，则先保留 v3 的序列化字符串，只改 Python 文件和类名。

## 5. 配置与 CLI

- CLI 统一为 `mcitlib train/eval/metrics/list`。
- 配置文件中的方法 ID 使用论文名小写形式；保留 `reglora -> sefe` 等 v3 别名。
- 骨干 ID 使用 `llava`、`internvl`、`videollava`、`videollama2`，不将上游 Python 包名泄漏到核心配置 schema。
- 核心配置键使用 `mcitlib` 语义命名；只有在生成 v3 等价命令时才映射到上游参数名（如 `expert_num`、`cur_task`）。
- 文档和日志中明确区分“MCITlib 插件名”和“third_party 上游包名”。

## 6. 禁止的命名回退

- 新建 `mcitlib/tuners/coinmoelora.py`、`clitmoelora.py` 或复制一份仅改类名的文件。
- 为了保留旧类名而把 `CoINMOELoraLinear` 作为 v4 的主要公开 API。
- 为了“统一命名”直接修改 `third_party/` 的上游代码。
- 为了改名而删除旧 checkpoint 的读取路径，或把兼容别名当成新的数值实现。
