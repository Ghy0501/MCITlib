# P0 盘点：v3 代码、上游快照与迁移落点

日期：2026-11-03  
基线：`v3.0-final` = `a3a622c9beb690e84c2923785279f179a6aac07e`  
分析分支：`v4-dev`

## 1. 方法与限制

本盘点只读，未修改任何 v3 方法代码。使用两个辅助脚本：

- `tools/p0/inventory.py`：统计每个方法目录的包行数、相对 LoRA-FT 的共有文件差异、单边文件和 vendored 包。
- `tools/p0/match_upstream.py`：从上游历史中选取探针文件，逐 commit 比较文本差异，找出最近候选。

“最近上游”是**探针差异最小的候选**，不是已经证明的精确快照。它用于选择 P2 的基线；固定 vendored 快照前仍需逐文件核对，并重新读取目标 commit 的 LICENSE。探针主要覆盖模型、训练、编码器和通用工具文件，不覆盖全部评测脚本和数据文件。

差异行数默认忽略空白、空行和 CRLF 差异；“共有文件差异”不把单边新增/删除文件的整文件内容计入，避免把缺文件误算成方法算法改动。

### 1.1 上游克隆记录与复现

本次分析使用的克隆均放在仓库外的 `/tmp/p0/up/`。表中 HEAD 是分析时的本地克隆 HEAD，不是 v4 最终 vendored 快照；最终快照以 §3 的候选 commit 为准。

| 本地目录 | URL | 克隆时 HEAD |
|---|---|---|
| `LLaVA` | `https://github.com/haotian-liu/LLaVA` | `c121f0432da27facab705978f83c4ada465e46fd` |
| `CoIN` | `https://github.com/zackschen/CoIN` | `747b1e52ec29981532d6c1dc5d1a9582ab254f19` |
| `InternVL` | `https://github.com/OpenGVLab/InternVL` | `2410d1dbf208f0e799459aff9376e5747dbf41a2` |
| `Video-LLaVA` | `https://github.com/PKU-YuanGroup/Video-LLaVA` | `984e65bf8e8fd97072258547c4c8061fdd2416b7` |
| `VideoLLaMA2` | `https://github.com/DAMO-NLP-SG/VideoLLaMA2` | `c0bb03abf6b8a6b9a8dccac006fb4db5d4d9e414` |
| `peft` | `https://github.com/huggingface/peft` | `65e7758329ace54785cd3e43dbf727f8ba879016` |
| `MLLM-CL` | `https://github.com/bjzhb666/MLLM-CL` | `a5a6da12a8bba6de453e5b30799cd450a28d6048` |
| `DoRA` | `https://github.com/NVlabs/DoRA` | `7e2f10abbe8efe212c8fca1d983ae1d04ef13a18` |
| `O-LoRA` | `https://github.com/cmnfriend/O-LoRA` | `07117e1fc4a5f5ad9308a815a42cee8f46502dc8` |
| `CL-MoE` | `https://github.com/ECNU-ICALK/CL-MoE` | `a9eff1c35a0cea13aa48e7233f5bbc8dd7b09471` |
| `HiDe-LLaVA` | `https://github.com/Ghy0501/HiDe-LLaVA` | `9d4597570b09c687aeadf3c34db1ff32a2b9ab62` |
| `SMoLoRA` | `https://github.com/Minato-Zackie/SMoLoRA` | `ba50e074f7468e18f0e6d512360acebe8e213424` |
| `KeepLoRA` | `https://github.com/MaolinLuo/KeepLoRA` | `536498eb4affdd1d60d6658c8069688e6f758939` |
| `ModalPrompt` | `https://github.com/AuroraZengfh/ModalPrompt` | `c7bee7c0bdab4be75ce31a6c3d68ce1eafdc08c1` |
| `SEFE` | `https://github.com/jinpeng0528/SEFE` | `2f61efe3a7da0407850399db6bc51f5ddc00f001` |
| `MCITlib` | `https://github.com/Ghy0501/MCITlib` | `a3a622c9beb690e84c2923785279f179a6aac07e` |

复现命令（在仓库根目录执行；方法来源仓库的浅克隆只用于许可证/出处核对）：

```bash
mkdir -p /tmp/p0/up
for repo in haotian-liu/LLaVA zackschen/CoIN OpenGVLab/InternVL \
  PKU-YuanGroup/Video-LLaVA DAMO-NLP-SG/VideoLLaMA2 huggingface/peft \
  bjzhb666/MLLM-CL NVlabs/DoRA cmnfriend/O-LoRA ECNU-ICALK/CL-MoE \
  Ghy0501/HiDe-LLaVA Minato-Zackie/SMoLoRA MaolinLuo/KeepLoRA \
  AuroraZengfh/ModalPrompt jinpeng0528/SEFE Ghy0501/MCITlib; do
  name=$(basename "$repo")
  git clone "https://github.com/$repo" "/tmp/p0/up/$name"
done
python3 tools/p0/inventory.py --root . --out /tmp/p0/inventory.json --pairwise
python3 tools/p0/match_upstream.py --backbone LLaVA --pkg llava \
  --upstream /tmp/p0/up/LLaVA --out /tmp/p0/match_LLaVA.json
python3 tools/p0/match_upstream.py --backbone InternVL --pkg llava \
  --upstream /tmp/p0/up/InternVL --upstream-prefix internvl_chat_llava/ \
  --out /tmp/p0/match_InternVL.json
python3 tools/p0/match_upstream.py --backbone Video-LLaVA --pkg videollava \
  --upstream /tmp/p0/up/Video-LLaVA --out /tmp/p0/match_VideoLLaVA.json
python3 tools/p0/match_upstream.py --backbone VideoLLaMA2 --pkg videollama2 \
  --upstream /tmp/p0/up/VideoLLaMA2 --out /tmp/p0/match_VideoLLaMA2.json
```

`inventory.py --root` 的参数是**仓库根目录**（包含 `LLaVA/`、`InternVL/` 等目录），不是某个骨干目录。

## 2. 总量复核

| 项目 | 实测 |
|---|---:|
| 骨干目录 | 4 |
| 方法目录 | LLaVA 12、InternVL 12、Video-LLaVA 12、VideoLLaMA2 11 |
| 受支持的（骨干，方法）组合 | 47 |
| `.py` 文件 | 5,025 |
| `.sh` 文件 | 1,529 |
| `configs/` 下 JSON | 1,329 |
| LLaVA 的 `llava/` 包合计行数 | 161,402 |
| LLaVA 各方法相对 LoRA-FT 的共有文件差异合计 | 5,224 行 |

VideoLLaMA2 没有 KeepLoRA，v4 继续显式不支持该组合。

## 3. 上游候选与建议基线

| 骨干 | v3 对应上游路径 | 最近候选 commit | 覆盖情况 | 建议 |
|---|---|---|---|---|
| LLaVA | `haotian-liu/LLaVA` 的 `llava/` | `31cceb832f2f403b2cbf822d0f8fe689d72fe219`（2023-11-27） | 10/12 方法最近 | 作为 `third_party/LLaVA` 默认基线 |
| LLaVA（KeepLoRA、RegLoRA） | 同一上游的较晚快照 | `dcda07fb143f2faf023474b7237c88d1fb0fcd4b`（2024-04-18） | 2/12 方法最近 | 差异文件先按方法自有模块隔离；若无法隔离，触发停止条件 2 |
| InternVL | `OpenGVLab/InternVL` 的 `internvl_chat_llava/` | `f4e5aee1c306cbfd35e23da81d83c5d637ce9109`（2024-05-29） | 12/12 方法最近 | 作为 `third_party/InternVL` 默认基线 |
| Video-LLaVA | `PKU-YuanGroup/Video-LLaVA` 的 `videollava/` | `b6534e7e1b1cdf674811466c5581582a993213dc`（2024-01-18） | 12/12 方法最近 | 作为 `third_party/Video-LLaVA` 默认基线 |
| VideoLLaMA2 | `DAMO-NLP-SG/VideoLLaMA2` 的 `videollama2/` | `1c89aeb9bd55c594c03e960e0934dd90cb0fac7f`（2025-01-08） | 11/11 方法最近 | 作为 `third_party/VideoLLaMA2` 默认基线 |

LLaVA 的两个快照差异不能简单视为“方法改动”：KeepLoRA/RegLoRA 多出 `model/language_model/llava_mistral.py`、`modeling_llama.py`，并改动 `conversation.py`、`mm_utils.py`、`model/__init__.py`、`builder.py`、`llava_llama.py`、`llava_arch.py`、投影/编码器构建和训练文件。KeepLoRA 的 `pyproject.toml` 还钉死 `transformers==4.37.2`，而多数 LLaVA 方法是 `4.31.0`；RegLoRA 的训练脚本运行时执行 `pip install transformers==4.37.2`。这与“所有 LLaVA 方法共用一个无差别环境”的假设不符，必须在 P2 用实际入口和 T0/T1/T2 逐项验证。

### 3.1 各方法的最近上游与差异量

“包 LOC”只统计对应骨干 Python 包；“基线缺失/新增”是相对同骨干 LoRA-FT 包的单边文件数，不是算法改动数。

#### LLaVA

| 方法 | 包 LOC | 共有文件差异行/文件 | 基线缺失/新增文件 | 最近上游 | 探针差异 |
|---|---:|---:|---:|---|---:|
| CL-MoE | 10,391 | 101 / 7 | 37 / 1 | `31cceb83` | 135 |
| DISCO | 14,120 | 421 / 23 | 9 / 0 | `31cceb83` | 377 |
| HiDe | 14,140 | 435 / 20 | 9 / 0 | `31cceb83` | 396 |
| KeepLoRA | 9,336 | 1,151 / 26 | 52 / 4 | `dcda07fb` | 445 |
| LoRA-FT | 15,769 | 0 / 0 | 0 / 0 | `31cceb83` | 167 |
| MR-LoRA | 16,400 | 278 / 10 | 0 / 3 | `31cceb83` | 199 |
| MoELoRA | 13,832 | 75 / 7 | 9 / 24 | `31cceb83` | 115 |
| ModalPrompt | 14,512 | 710 / 11 | 39 / 60 | `31cceb83` | 623 |
| OLoRA | 14,951 | 553 / 15 | 12 / 37 | `31cceb83` | 181 |
| RegLoRA | 8,880 | 1,321 / 24 | 51 / 2 | `dcda07fb` | 517 |
| Replay | 15,773 | 14 / 5 | 0 / 0 | `31cceb83` | 167 |
| SMoLoRA | 13,904 | 165 / 13 | 9 / 24 | `31cceb83` | 115 |

#### InternVL

| 方法 | 包 LOC | 共有文件差异行/文件 | 基线缺失/新增文件 | 最近上游 | 探针差异 |
|---|---:|---:|---:|---|---:|
| CL-MoE | 19,455 | 38 / 5 | 0 / 0 | `f4e5aee1` | 152 |
| DISCO | 19,786 | 759 / 18 | 0 / 0 | `f4e5aee1` | 488 |
| HiDe | 19,791 | 764 / 18 | 0 / 0 | `f4e5aee1` | 493 |
| KeepLoRA | 20,336 | 196 / 4 | 0 / 4 | `f4e5aee1` | 239 |
| LoRA-FT | 19,439 | 0 / 0 | 0 / 0 | `f4e5aee1` | 128 |
| MR-LoRA | 19,958 | 4 / 2 | 0 / 3 | `f4e5aee1` | 128 |
| MoELoRA | 19,452 | 39 / 5 | 0 / 0 | `f4e5aee1` | 151 |
| ModalPrompt | 21,418 | 648 / 11 | 29 / 38 | `f4e5aee1` | 730 |
| OLoRA | 19,524 | 113 / 6 | 0 / 0 | `f4e5aee1` | 225 |
| RegLoRA | 20,965 | 116 / 7 | 0 / 1 | `f4e5aee1` | 228 |
| Replay | 19,439 | 6 / 3 | 0 / 0 | `f4e5aee1` | 128 |
| SMoLoRA | 19,517 | 116 / 12 | 0 / 0 | `f4e5aee1` | 180 |

#### Video-LLaVA

| 方法 | 包 LOC | 共有文件差异行/文件 | 基线缺失/新增文件 | 最近上游 | 探针差异 |
|---|---:|---:|---:|---|---:|
| CL-MoE | 21,425 | 67 / 5 | 0 / 0 | `b6534e7e` | 77 |
| DISCO | 21,699 | 371 / 10 | 0 / 0 | `b6534e7e` | 359 |
| HiDe | 21,701 | 371 / 10 | 0 / 0 | `b6534e7e` | 361 |
| KeepLoRA | 28,208 | 388 / 6 | 0 / 21 | `b6534e7e` | 183 |
| LoRA-FT | 21,392 | 0 / 0 | 0 / 0 | `b6534e7e` | 57 |
| MR-LoRA | 22,018 | 46 / 5 | 0 / 2 | `b6534e7e` | 57 |
| MoELoRA | 21,423 | 65 / 6 | 0 / 0 | `b6534e7e` | 74 |
| ModalPrompt | 21,989 | 566 / 12 | 0 / 1 | `b6534e7e` | 535 |
| OLoRA | 21,494 | 142 / 7 | 0 / 0 | `b6534e7e` | 147 |
| RegLoRA | 23,996 | 171 / 7 | 0 / 2 | `b6534e7e` | 184 |
| Replay | 21,416 | 46 / 5 | 0 / 0 | `b6534e7e` | 57 |
| SMoLoRA | 21,494 | 440 / 12 | 0 / 0 | `b6534e7e` | 130 |

#### VideoLLaMA2

| 方法 | 包 LOC | 共有文件差异行/文件 | 基线缺失/新增文件 | 最近上游 | 探针差异 |
|---|---:|---:|---:|---|---:|
| CL-MoE | 10,253 | 60 / 6 | 1 / 0 | `1c89aeb9` | 72 |
| DISCO | 10,277 | 445 / 8 | 2 / 0 | `1c89aeb9` | 414 |
| HiDe | 10,532 | 437 / 9 | 1 / 0 | `1c89aeb9` | 419 |
| LoRA-FT | 10,376 | 0 / 0 | 0 / 0 | `1c89aeb9` | 86 |
| MR-LoRA | 10,539 | 75 / 3 | 0 / 1 | `1c89aeb9` | 86 |
| MoELoRA | 10,229 | 94 / 4 | 1 / 0 | `1c89aeb9` | 99 |
| ModalPrompt | 10,777 | 724 / 11 | 1 / 0 | `1c89aeb9` | 667 |
| OLoRA | 10,393 | 278 / 9 | 1 / 0 | `1c89aeb9` | 272 |
| RegLoRA | 11,867 | 167 / 6 | 0 / 1 | `1c89aeb9` | 176 |
| Replay | 10,246 | 139 / 3 | 1 / 0 | `1c89aeb9` | 117 |
| SMoLoRA | 10,291 | 72 / 8 | 1 / 0 | `1c89aeb9` | 117 |

## 4. 方法核心位置与 shell 逻辑

| 方法 | 真正的核心位置 | shell/流程中的方法特定行为 | v4 落点 |
|---|---|---|---|
| LoRA-FT | vendored `peft/`；普通 LoRA/DoRA 分支 | `train_mem.py`；LoRA 参数由 json 与 shell 合成 | `methods/lora_ft.py` |
| Replay | 基本只有脚本差异 | `train_path -> replay_path`，`train_folder -> router_folder`；任务 1 也走该替换 | `methods/replay.py` 的 `select_train_data` |
| MoELoRA | `CoIN/peft/tuners/clitmoelora.py` | 入口改为 `train_mem_MOE.py`，传 `expert_num` | `tuners/mcit_expert_lora.py` + 方法插件 |
| MR-LoRA | 与 MoELoRA 相同的调优器；路由在 `llava/eval/CoIN/model_*_router.py` | 普通 Task1 与 LoRA-FT 相同；另有 `*_router` 训练和评测脚本 | 共享调优器 + `methods/mrlora.py` |
| O-LoRA | `clitmoelora.py` 变体；trainer 正交损失 | `train_mem_MOE.py`，传 `expert_num`、`cur_task`，gradient checkpointing 可配置；`use_reentrant=False` | 变体 + `extra_loss` |
| CL-MoE | `CLMoE/peft/tuners/clitmoelora.py` | `train_mem_MOE.py`，传 `expert_num`、`task` | 变体 + trainer/方法钩子 |
| DISCO | `clitmoelora.py` 变体；`llava_arch.py`、`llava_llama.py`、CLIP 编码器 | `train_mem_MOE.py`，传 `expert_num`、`cur_task`、`text_tower` | 变体 + 骨干 capability 钩子 |
| HiDe | `HiDe/peft/tuners/clitmoelora.py`；文本塔、anchor 和专家权重在骨干模型中 | `train_mem_MOE.py`，传 `expert_num`、`cur_task`、`text_tower` | 变体 + 骨干钩子；需 `text_tower`、`clip_features`、`layer_index` |
| SMoLoRA | `ptmoelora.py`（与基础同义）和 `smolora.py`；指令 embedding | `train_mem_MOE.py`，传 `expert_num`、`ins_type`、`ins_emb`；`train_MOE.py` 用 `pickle.load` 读 `.pkl` | `mcit_separable_expert_lora.py` + 显式数据加载 |
| ModalPrompt | 主要在 `llava/model/llava_arch.py`、builder、trainer；peft 的 prompt 文件与其他目录相同 | 使用 `zero3_offload.json`，`lora_enable=False`、`pt_enable=True`，传 `prefix_len`、`cur_task`、`num_tasks`；训练/推理流程需分开核对 | `methods/modalprompt.py` + 骨干钩子 |
| SEFE（v3 目录 RegLoRA） | `CoIN-ASD/prompts`、`key_elements_identification.py`；另有较晚 LLaVA 快照文件 | Task1/Taskn 运行时安装 `transformers==4.37.2`，训练后调用 `scripts/merge_lora_weights.py` | `methods/sefe.py` + `methods/_sefe/` |
| KeepLoRA | `extract_gradients.py`、`extract_weights.py` 和较晚 LLaVA 快照文件 | 传 `freeze_lora_A`、`init_lora_from_gradients`，训练后 merge LoRA；需要单独显存说明 | `methods/keeplora.py` + `methods/_keeplora/` |

### 4.1 实际训练入口

不能只比较 `train.py`：LLaVA 的 MoE 系方法实际由 `Task1.sh` 调用 `train_mem_MOE.py`，再进入 `train_MOE.py`。其余 LLaVA 方法、InternVL、Video-LLaVA 和 VideoLLaMA2 大多使用普通 `train_mem.py`/`train.py`。因此 P0 的特性漂移按实际入口重新核对，结果见 `drift.md`。

## 5. vendored 包与 peft 核实

### 5.1 目录形态

- LLaVA/InternVL 的 LoRA-FT、Replay、KeepLoRA 等目录带顶层 `peft/`；MoE 系方法带 `CoIN/`、`HiDe/` 或 `CLMoE/` 下的 peft 分叉。
- Video-LLaVA/VideoLLaMA2 多数方法不带顶层 peft，但 KeepLoRA 在 Video-LLaVA 下仍有一套 `videollava/peft/`，不能按“该骨干没有 vendored peft”处理。
- `SMoLoRA` 还带 `ins_emb_multi_*.pkl` 和 `ins_gen.py`；v4 不得在 import 时自动反序列化这些文件。
- `LLaVA/RegLoRA` 的 `CoIN-ASD/prompts` 包含 8 个任务提示目录，需随 SEFE 方法模块迁移并保留出处。

### 5.2 与 PyPI/GitHub peft 0.4.0 的差异

以 `peft` 仓库 `v0.4.0` 的 `src/peft` 为参照，`LLaVA/LoRA-FT/peft` 的 `__version__` 为 `0.4.0`，但并非“只增加 DoRA”：

| 文件 | 忽略空白后的差异行数 | 已见内容 |
|---|---:|---|
| `__init__.py` | 2 | 导出 DoRA 配置/模型 |
| `mapping.py` | 13 | DoRA 注册及 NVIDIA 头 |
| `peft_model.py` | 6 | DoRA 模型映射 |
| `tuners/__init__.py` | 1 | DoRA 导出 |
| `tuners/lora.py` | 15 | 多处调试 `print`/注释改动 |
| `utils/__init__.py` | 1 | 导出变化 |
| `utils/config.py` | 8 | DoRA 类型/配置 |
| `utils/other.py` | 34 | DoRA 相关辅助逻辑等 |
| `utils/save_and_load.py` | 56 | DoRA 权重保存/加载 |
| `tuners/dora.py` | 新增 805 行 | 24 份内容 md5 相同 |

DoRA 文件头是 NVIDIA 专有声明，上游许可证仅限研究或评估用途、不符合 OSI，是 MLOSS 发布阻塞项；v3 中没有任何脚本启用 DoRA，v4 按 NB-1 不 vendor 相关代码。除 DoRA 外，`lora.py` 的调试输出和其他 peft 改动也必须在 T1/T3 中决定是否保留，不能默认为无关差异。

InternVL 的 pyproject 声明 `peft==0.6.0`，但 vendored peft 的 `__version__` 仍为 `0.4.0`。静态文件无法证明运行时究竟导入哪一个：`train_mem.py`/`builder.py` 通过 `sys.path.append` 指向方法目录，而方法目录同时包含可发现的顶层 `peft` 包。P1 必须在干净 InternVL 环境记录 `peft.__file__`、`peft.__version__` 和实际导入路径；在确认前不得擅自替换版本。

## 6. 已发现的上游/方法许可证证据

以下 LICENSE 是从实际克隆仓库读取的当前文件，不是凭记忆填写；固定到具体 commit 后需再次核对：

| 组件 | 来源 | 当前读取结果 |
|---|---|---|
| LLaVA | `haotian-liu/LLaVA` | Apache-2.0 |
| InternVL 根仓库 | `OpenGVLab/InternVL` | MIT；`internvl_chat_llava/LICENSE` 另为 Apache-2.0 |
| Video-LLaVA | `PKU-YuanGroup/Video-LLaVA` | Apache-2.0 |
| VideoLLaMA2 | `DAMO-NLP-SG/VideoLLaMA2` | Apache-2.0 |
| peft | `huggingface/peft` | Apache-2.0 |
| CoIN | `zackschen/CoIN` | Apache-2.0 |
| MLLM-CL | `bjzhb666/MLLM-CL` | Apache-2.0 |
| O-LoRA | `cmnfriend/O-LoRA` | MIT |
| SMoLoRA | `Minato-Zackie/SMoLoRA` | Apache-2.0 |
| KeepLoRA | `MaolinLuo/KeepLoRA` | MIT |
| SEFE | `jinpeng0528/SEFE` | MIT，Copyright 2025 Jinpeng Chen |
| CL-MoE | `ECNU-ICALK/CL-MoE` | 当前克隆未发现 LICENSE，未确定 |
| HiDe-LLaVA | `Ghy0501/HiDe-LLaVA` | 当前克隆未发现 LICENSE，未确定 |
| ModalPrompt | `AuroraZengfh/ModalPrompt` | 当前克隆未发现 LICENSE，未确定 |

后三项触发设计文档 §2.4 的停止条件 5（上游许可证无法确定），在作者补充许可证或书面决定前，不进入对应代码的迁移和再发布。

## 7. 对 v4 结构的直接结论

1. 最大去重对象确实是 MoE-LoRA 基础调优器，但不能只按 LLaVA 的 7 份文件合并：InternVL/RegLoRA 还有一份带 `cur_task` 和专家冻结逻辑的变体；HiDe、DISCO 在不同骨干上也有实质不同的推理路径。
2. `third_party/` 只能放上游树和通用 patch；所有方法差异（包括上述跨骨干变体）放 `mcitlib/`。
3. 上游包名可以保留在 `third_party/`，但自有共享调优器按 `naming.md` 改为 MCITlib 命名，并通过序列化兼容层读取 v3 checkpoint。
4. 不能把 v3 的“同一论文方法”直接当成“同一份算法实现”：必须按（方法，骨干）建立实现变体矩阵，再用 T1/T2/T3 证明哪些可以共享。
