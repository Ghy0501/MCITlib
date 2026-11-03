# LLaVA 基线差异分类（L2）

比较对象：

- 上游：`third_party/LLaVA/llava`，`haotian-liu/LLaVA` @ `31cceb832f2f403b2cbf822d0f8fe689d72fe219`。
- v3：`LLaVA/LoRA-FT/llava`。
- 统计：共有文件的空白/CRLF 不敏感 diff，以及两边单边文件。上游单边文件保留在 vendored 树中，不因 v3 某份拷贝缺文件而删除。

分类：

- **(a) 通用钩子**：只有确为所有方法需要、且无法子类化/注入时，才可作为 `third_party/LLaVA/patches/` 的候选。
- **(b) MCITlib 训练/数据/模型逻辑**：迁到 `mcitlib/backbones/llava/` 或方法私有模块，不写回上游。
- **(c) CoIN 评测逻辑**：迁到 `mcitlib/backbones/llava/eval_runners/`，保留原版权头。
- **(d) NB 项/死代码/路径残留**：按批准的 NB 项删除或替换，并登记。

## 1. 共有文件

| 文件 | 差异行数（约） | 分类 | 处理 |
|---|---:|---|---|
| `train/train.py` | 134 | (b)+(d) | 保留 v3 的数据、LoRA 加载、训练参数和行为；DoRA 按 NB-1 删除，调试输出按 NB-3 删除，checkpoint 清理按 NB-4 改为安全 Python，路径按 AR-4 替换 |
| `train/train_mem.py` | 2 | (d) | 删除 `sys.path.append`，由 v4 包导入机制替代 |
| `train/llava_trainer.py` | 2 | (b) | 保留 `is_SAT` 对应的 sampler 分支，在 standard 数据管线中实现 |
| `model/builder.py` | 3 | (d) | 删除 `sys.path.append`；模型加载由 adapter 负责 |
| `model/language_model/llava_llama.py` | 1 | (a) 候选 | `forward(**kwargs)` 是通用兼容性钩子候选；先尝试 adapter/子类，确需上游改动时才制作 patch |
| `model/llava_arch.py` | 2 | (b) | `mm_projector.load_state_dict(..., strict=False)` 属 v3 模型加载行为，放到 adapter 的加载逻辑并用 T1/T2 验证 |
| `eval/eval_gpt_review_visual.py` | 11 | (c)/(d) | 逐项确认是否为 CoIN 评测改动；不将无用调试代码迁入 |

目前没有发现可以直接把方法特定代码作为通用 patch 写入上游的依据。默认优先 adapter、子类和依赖注入。

## 2. v3 单边新增文件

### 2.1 训练与模型入口（(b)）

| 文件 | 处理 |
|---|---|
| `train/train_MOE.py` | 迁移为 `mcitlib/backbones/llava/` 的 MoE 训练逻辑；与 standard `train.py` 保持两条管线，不能强行合并 |
| `train/train_mem_MOE.py` | 迁移为 MoE train entry 的薄封装；保留实际 v3 参数路径 |

### 2.2 CoIN 评测脚本（(c)）

以下目录整体属于 CoIN/CL 评测来源，不放入上游：

- `eval/CoIN/api.py`、`coin_utils.py`、`convert_result_to_submission.py`；
- `eval/CoIN/eval_*.py`（包括 ai2d、caption、deepseek_r1、finvis、gqa、grounding、imagenet、math、ocr、ocrvqa、pope、pvqa、sat、sci、science_qa、textvqa、vizwiz、vqav2）；
- `eval/CoIN/model_*.py`（包括 ai2d、fin、math、ocr、others、pvqa、sat、science_qa）。

v3 根级 `eval/` 中以下上游没有的文件也按同一来源审查并迁移：`eval_ai2d.py`、`eval_caption.py`、`eval_deepseek_r1.py`、`eval_finvis.py`、`eval_gpt_review_visual_cl.py`、`eval_math.py`、`eval_ocr.py`、`eval_pvqa.py`、`eval_sat.py`、`eval_sci.py`、`model_answer.py`、`model_answer_sci.py`、`model_vqa_cl.py`。迁移时按实际调用关系去重，不复制成两套打分实现。

### 2.3 仅在上游存在的文件

`eval/eval_science_qa.py`、`eval/eval_textvqa.py`、`eval/model_vqa_science.py`、上游 `eval/table/` 和 `eval/webpage/` 等仍保留在 `third_party/LLaVA`。它们在某份 v3 方法拷贝中缺失，不构成从上游删除的理由。

## 3. v3 LoRA-FT/peft 相对 peft v0.4.0

上游：`third_party/peft/peft`，`huggingface/peft` @ `v0.4.0`。

| 文件 | 分类 | 处理 |
|---|---|---|
| `tuners/dora.py` | (d) NB-1 | 不 vendor、不迁移；v3 没有脚本启用 DoRA，且 NVIDIA 许可不符合 MLOSS 要求 |
| `__init__.py`、`mapping.py`、`peft_model.py`、`tuners/__init__.py`、`utils/config.py` | (d) NB-1 | 删除 DoRA 导出、PeftType 和模型映射；其余 peft 原逻辑保留 |
| `utils/save_and_load.py` | (d) NB-1 | 删除 DoRA 专用权重保存/加载分支；保留 LoRA 所需逻辑 |
| `utils/other.py` | (d) NB-1/(b) | 分离 DoRA 改动与实际 LoRA 辅助逻辑，逐项用 T1 验证，不整文件复制 |
| `tuners/lora.py` | (d) NB-3 | 删除调试 `print`，保留模块替换、加载和前向逻辑 |
| `utils/__init__.py` | (d) NB-1 | 只保留非 DoRA 导出变化 |

v4 的 `third_party/peft` 当前保持 peft v0.4.0 原始树；MCITlib 自定义调优器放 `mcitlib/tuners/`，通过 `_peft_registry.py` 注册，不向 vendored peft 写方法特定实现。

## 4. 执行约束

1. 先完成 adapter 和 T1/T2，再决定是否需要 `llava_llama.py` 的通用 patch；当前 `patches/` 为空。
2. CoIN 评测脚本只在 `eval_runners/` 保留一份与骨干相关的推理入口，骨干无关打分逻辑后续下沉到 `mcitlib/eval/scorers/`。
3. 所有迁移文件保留原版权头，再追加 MCITlib SPDX 头；MIT 来源按 `THIRD_PARTY_NOTICES.md` 的要求处理。
4. 本分类文档不等于数值等价证明，LoRA-FT 仍需通过 T2/T3，调优器和训练模块需通过 T1。
