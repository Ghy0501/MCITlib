# MoE-LoRA 基础调优器及各变体的语义差异

日期：2026-11-03  
基线：`LLaVA/MoELoRA/CoIN/peft/tuners/clitmoelora.py`（547 行）

本文比较的是计算语义、参数和外部依赖，不是把 7 份分叉再做一次行 diff。结论用于设计 `mcitlib/tuners/` 的单一基础实现和子类变体；不授权在 P0 修改算法。

## 1. 基础调优器的实际计算

基础文件中的主要对象（v4 的 MCITlib 命名映射见 `naming.md`）：

- `CoINMOELoraConfig(LoraConfig)`：`task_embedding_dim=64`、`expert_num=4`，并将 `peft_type` 设为 `MOE_LORA_CoIN`。
- `CoINMOELoraModel(LoraModel)`：遍历目标模块，将普通线性层替换为 `CoINMOELoraLinear`。
- `CoINMOELoraLinear`：冻结原始 `weight`，建立 `lora_router`、`lora_A`、`lora_B`。
- `CoINMOELinearA/B`：各含 `expert_num` 个专家；每个专家的 rank 为 `r / expert_num`，要求整除。
- `CoINMOEExpert`：无 bias 的 `nn.Linear`，权重通过 `mlp.weight` 暴露。

在 adapter 有效且 `r > 0` 时，基础 forward 的有效路径为：

1. 计算原始线性层结果 `result = F.linear(x, weight, bias)`。
2. 对 dropout 后的输入分别经过每个专家的 A、B。
3. `lora_router` 产生 `expert_num` 个 logits，再做 softmax。
4. 按 router 权重累加各专家的 `B_i(A_i(x)) * scaling`。
5. 转回输入原 dtype。

A 的每个专家用 `normal_(mean=0, std=0.01)` 初始化，B 的每个专家置零。模块替换、router 创建、reset 和后续初始化的顺序必须保留，否则即使公式相同，随机初始化也可能不同。

基础文件中的 `merge/unmerge` 实际权重合并代码被注释掉，只改变 merged 状态；`CoINMOEGate` 和 `CoINMOERouter` 当前未发现实际调用，不能误认为它们就是基础 forward 使用的 router。

## 2. 文件聚类与跨骨干事实

以下差异行数均以 LLaVA/MoELoRA 基础文件为参照，忽略空白/空行/CRLF，但仍包含类名变化，所以不是“算法改动行数”。`—` 表示该目录没有对应的 vendored MoE 调优器文件。

| 变体 | LLaVA | InternVL | Video-LLaVA | VideoLLaMA2 | 关键结论 |
|---|---:|---:|---:|---:|---|
| MoELoRA 基础 | 0 | 0 | 0 | 3 | VideoLLaMA2 增加跳过 `vision_tower` 的模块过滤 |
| MR-LoRA | 0 | 0 | — | — | LLaVA/InternVL 与基础同文件；路由主要在外部推理脚本 |
| OLoRA | 103 | 103 | 103 | 106 | 前三个骨干同内容；VideoLLaMA2 另有模块过滤 |
| CL-MoE | 72 | 72 | 72 | 72 | 有效 router 计算基本相同；硬编码 task 文件路径不同 |
| DISCO | 126 | 126 | 147 | 150 | 后两个骨干的推理路径有额外 Top-K 逻辑 |
| HiDe | 150 | 118 | 204 | 121 | InternVL 与 LLaVA 语义基本一致；Video-LLaVA 另有 Top-K 逻辑 |
| SMoLoRA 的 `ptmoelora.py` | 0 | 0 | 0 | 0 | 四份文件 md5 相同，实质是基础类改名 |
| SMoLoRA 的 `smolora.py` | 独立实现 | 独立实现 | 独立实现 | 独立实现+3 | VideoLLaMA2 另有模块过滤 |
| RegLoRA/SEFE 额外 MoE 变体 | — | 56 | — | — | 仅 InternVL 目录发现，有专家冻结和累计专家路径 |

完全相同的文件聚类包括：

- 基础 `clitmoelora.py`/`coinmoelora.py`：LLaVA 的 MoELoRA、MR-LoRA，Video-LLaVA 的 MoELoRA，以及 InternVL 的 MoELoRA、MR-LoRA 内容相同。
- OLoRA：LLaVA、InternVL、Video-LLaVA 内容相同；VideoLLaMA2 是带额外过滤的版本。
- `ptmoelora.py`：四个骨干完全相同。
- `smolora.py`：LLaVA、InternVL、Video-LLaVA 相同；VideoLLaMA2 多模块过滤。
- `coinmoelora_new.py`：多个目录存在但未发现导入引用，不能作为有效变体依据。

**重要修正：**“7 个方法共享一个分叉”只描述了部分目录。MR-LoRA 在 Video-LLaVA/VideoLLaMA2 没有同类 vendored 文件；InternVL 的 RegLoRA 另有 MoE 变体。迁移矩阵必须以实际文件和实际入口为准。

## 3. OLoRA

### 3.1 调优器语义

相对基础版：

- 配置增加 `cur_task`（文件中默认值为 4，需原样保留其配置语义）。
- 替换模块时把 `self.model.training` 传入新层；层和 A/B bank 保存 `cur_task` 与训练标志。
- 训练路径不再使用 router 加权求和，而只使用当前任务的专家：

  `result += B_cur(A_cur(dropout(x))) * scaling`。

- 非训练路径将任务 `0..cur_task` 的 A 权重沿输出维拼接、B 权重沿输入维拼接，临时构造线性层后计算合并结果。
- router 参数仍可能在构造时存在，但有效训练/推理路径不再按基础版使用 router。

### 3.2 外部钩子

- `train_MOE.py` 传入 `cur_task`，并调用模型和 trainer 的 `set_cur_task`。
- `llava_trainer.py` 新增 `compute_loss`：当 `cur_task >= 1`，对当前任务 A 与历史任务 A 的矩阵乘积绝对值求和，乘以 **0.05** 加入原 loss；历史参数 `detach()`。
- OLoRA 训练参数使用 `gradient_checkpointing_kwargs={"use_reentrant": False}`。

### 3.3 子类化建议与风险

基础类应把“专家 bank 创建”和“专家前向策略”拆成可覆盖点，OLoRA 只覆盖当前任务选择、合并推理和配置扩展；正交损失放 `ContinualMethod.extra_loss`，不复制 trainer。

风险：构造时保存的 `training` 与后续 `model.train()/eval()` 的属性变化可能不同步；T1 必须分别测试构造、训练切换、推理切换和 checkpoint 加载后的路径。

## 4. CL-MoE

### 4.1 调优器语义

- 类名改为 `CLMoEMOE*`，配置注册为 `MOE_LORA_CLMoE`。
- 新增 `noisy_gating=True`、`topk=2` 属性，但在检查的 LLaVA 文件中，有效专家输出仍是基础版的 softmax router 加权求和。
- forward 额外读取硬编码 `task.txt`，并在特定秒数写入 top-2 router 统计；被注释的 alpha/top-k 试验代码不属于有效计算。
- 四个骨干的有效计算结构相同，但 task 文件路径分别指向各自方法目录。

### 4.2 外部钩子与风险

训练脚本写入 task 标识并传入 `task`。LLaVA 的 trainer 相对 MoELoRA 未发现新的有效 `compute_loss`，主要是注释代码。

v4 应将任务标识由 `RunContext` 提供，将统计输出作为显式可选诊断钩子；不能为了清理副作用而改变 router 计算。若诊断文件不再生成，应在等价性报告中单列这一非数值行为差异。

## 5. DISCO

### 5.1 通用变体语义

- 配置增加 `cur_task`（基础文件中默认值为 0）。
- 线性层保存 `lora_id=cur_task` 和长度 8 的 `mask_signal=[1,0,0,0,0,0,0,0]`。
- 移除基础 router 的有效使用，增加 `lora_AB=Linear(r,r,bias=False)`，初始置零，`diag_size=r/expert_num`。
- forward 根据 `mask_signal` 将 `lora_AB` 的对角块设为单位矩阵乘 flag。
- 训练只走 `lora_id` 对应的专家。

### 5.2 推理的跨骨干差异

- LLaVA/InternVL：A/B bank 将各专家权重拼接，经过 `lora_AB` 后再到 B，形成合并推理路径。
- Video-LLaVA/VideoLLaMA2：另有 `top_k=3`，对 `mask_signal` 做 Top-K、归一化，再按选中的专家分别计算 A/B 并累加；不是 LLaVA 推理路径的简单复制。
- VideoLLaMA2 还在模块替换时跳过 `vision_tower`。

因此 DISCO 至少需要“通用训练/参数结构 + 骨干特定推理策略”，不能只做一份 LLaVA 子类后覆盖所有骨干。

### 5.3 骨干依赖

DISCO 的 `llava_arch.py`、`llava_llama.py`、CLIP encoder/builder 有实质改动，训练脚本还传 `text_tower`。这些改动需要在 P2 逐项归类为通用 backbone capability 或 DISCO 自有模块，不能塞进共享上游树。

## 6. HiDe

### 6.1 调优器语义

- 配置和模型类使用 `MOE_LORA_HiDe`，增加 `cur_task`，模型保存长度 8 的 `expert_weight`。
- peft 的 `_get_submodules` 额外返回 `layer = key.split(".")[2]`，并传入线性层和 A/B bank。
- 训练只走当前任务专家。
- 非第 31 层的推理路径将 `0..cur_task` 的 A/B 权重分别累加融合到临时线性层。
- 第 31 层使用 `expert_weight` 对各专家输出加权求和；`layer == 31` 是代码中的固定判断，不能擅自改成通用“最后一层”而不做等价验证。

### 6.2 跨骨干差异

- LLaVA 与 InternVL 的 HiDe 语义基本一致，主要差异是命名和 `_prepare_*` 名称。
- VideoLLaMA2 在此基础上增加 `vision_tower` 模块过滤。
- Video-LLaVA 额外引入 `topk=3`：在推理、第 31 层且有专家权重时，预计算 Top-K 索引和归一化权重，再走筛选后的专家路径。这是实质的骨干特定变体。

### 6.3 骨干模型钩子

LLaVA HiDe 的模型侧还包含：

- CLIP 文本塔及 `mm_text_tower`/`mm_text_select_layer` 配置；
- 图像/文本 anchor、boundary 的 `ParameterList`（当前代码长度为 10）；
- `set_cur_task(cur_task, expert_num)` 和保存前 boundary 处理；
- 训练时按 batch 更新 anchor；推理时计算图像/文本相似度，经温度 0.1 的 softmax 得到专家权重，并写入最后一层的 q/k/v/o/gate/up/down 投影层。

所以 HiDe 需要 `text_tower`、`clip_features`、`layer_index` 等 capability，且需要一个能保存/恢复 anchor 与 boundary 的方法模块；只迁移 tuner 文件会漏掉方法核心。

## 7. SMoLoRA

### 7.1 `ptmoelora.py`

四个骨干的该文件 md5 相同。相对基础版主要是 `CoINMOE*` 到 `PTMOE*` 的类名替换，配置仍使用 `MOE_LORA_CoIN`。它是重复实现，不应在 v4 再保留一份独立算法文件。

### 7.2 `smolora.py`

这是独立的可分离专家路由实现：

- 配置增加 `ins_type`、`ins_emb`，PeftType 为 `SMOLORA`。
- 线性层增加 `lora_vu_gate`、`lora_ins_gate`、`lora_fc_A`、`lora_fc_B`。
- 视觉/通用路径对 token 平均后的输入做 router，指令路径使用外部 instruction embedding；两者都取 top-1。
- 专家集合分成前后两半，分别计算 `vu_result` 与 `if_result`。
- 两个结果经 fc 得到两类 score，softmax 得到 `attention_weights`，再加权合并到原始线性输出。
- `ins_type > 0` 时冻结相应 gate，否则打开对应 gate 的梯度；`lora_ins_gate` 输入维度写死为 384。
- VideoLLaMA2 增加 `vision_tower` 模块过滤。

外部 `.pkl` 的读取在 `train_MOE.py`，不是 tuner 的 import 副作用。v4 应将其变成显式、可审计的数据输入，同时保持张量内容和路由顺序。

## 8. InternVL RegLoRA/SEFE 的额外 MoE 变体

`InternVL/RegLoRA/CoIN/peft/tuners/coinmoelora.py` 相对 InternVL 基础版另有约 56 行差异：

- 增加 `cur_task` 和训练标志传递；
- `update_layer` 后调用 `set_expert_trainable`，仅当前任务专家的 A/B 权重可训练，其余专家冻结；
- forward 对 `0..cur_task` 的专家累加，且不再使用基础 router 的 softmax 权重；
- 与 `modeling_llama.py` 中的 regularization 信息和 `llava_llama.py` 的 `set_cur_task` 配合。

这不能被当作普通 MoELoRA 或 OLoRA 的别名。LLaVA/Video-LLaVA/VideoLLaMA2 的 RegLoRA 目录并未发现同一 vendored MoE 文件，SEFE 迁移需按骨干分别处理。

## 9. 建议的共享/子类化边界

### 9.1 基础类应暴露的可覆盖点

在不改变默认基础路径的前提下，将以下点设计成小函数或策略对象：

1. 目标模块遍历与跳过规则（例如 VideoLLaMA2 的 `vision_tower` 过滤）。
2. 新模块构造上下文（adapter、training、layer、cur_task、expert_weight）。
3. A/B bank 的创建与初始化。
4. 训练时专家选择策略。
5. 推理时专家合并/路由策略。
6. 配置字段扩展和旧 PeftType 的解析。

### 9.2 变体放置

- 基础：`mcitlib/tuners/mcit_expert_lora.py`。
- OLoRA、CL-MoE、DISCO、HiDe：`mcitlib/tuners/variants/`，只覆盖必要策略。
- DISCO/HiDe 的骨干特定 Top-K：作为明确的 backbone-specific variant 或策略，不能隐藏在共享默认路径里。
- SMoLoRA：`mcitlib/tuners/mcit_separable_expert_lora.py`，因为它是不同的双路由和双分数融合结构，不宜强行套成基础 router 的一个小 if。
- SEFE 的 InternVL 变体：`mcitlib/methods/_sefe/` 或专用 tuner variant，并保留其 MIT 版权信息。

### 9.3 方法/骨干钩子

- OLoRA：`extra_loss`、任务切换、gradient checkpointing 配置。
- CL-MoE：任务上下文和可选诊断输出。
- DISCO/HiDe：文本塔、额外视觉/文本特征、层索引、anchor/expert weight 的骨干 capability。
- MR-LoRA：路由 prompt、router 数据路径和推理期专家选择，不复制基础 tuner。
- SMoLoRA：显式 instruction embedding 数据输入和 `ins_type` 配置。

## 10. 等价性验证重点

T1 不能只比较一个默认 forward。至少覆盖：

- 相同种子下的模块替换、参数名、参数形状和初始化顺序；
- 当前任务与历史任务专家的可训练状态；
- train/eval 切换后的 OLoRA/DISCO/HiDe 路径；
- 各骨干的 `vision_tower` 过滤和 Top-K 分支；
- 旧 PeftType、旧 state-dict 键和 v3 checkpoint 的加载；
- OLoRA 正交损失的梯度及 `use_reentrant=False`；
- HiDe anchor/boundary 的保存、加载和最后一层专家权重注入；
- SMoLoRA instruction embedding 的数值内容和 gate 梯度开关。

在这些测试产物存在之前，不得声称“共享基础调优器已证明等价”。
