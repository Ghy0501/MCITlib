# 全部 baselines 集成复核与两个 VideoLLM 的统一接入矩阵

日期：2026-11-17。范围：读取现行 v4 注册表、全部方法入口、LLaVA/InternVL 集成与修复记录，以及两个视频骨干的历史代码和本地模型配置。本轮只更新规则和证据清单，不实施视频适配、不启动训练。前一份 InternVL 五方法复盘保留为历史案例；后续任务不再按“新/旧 baseline”分组。

## 1. 全方法范围与当前真实起点

当前 registry 有 **17 个方法**：LoRA-FT、Replay、MoELoRA、MR-LoRA、OLoRA、CL-MoE、DISCO、HiDe、SMoLoRA、ModalPrompt、SEFE/RegLoRA、KeepLoRA、PCLR、SAME、SiGMA、DGG、Octopus。SEFE 与 RegLoRA 是同一注册方法的别名，不重复计数。

用户的后续目标据此为 **17 × 2 = 34 个方法/视频模型组合**，两个模型为 `videollava`（Video-LLaVA）与 `videollama2`（VideoLLaMA2）。这是目标数量，不是 34 个已适配组合，也不是乘上图像 benchmark 后的任务数。

本次检查的 v4 现状：

- 两个视频 backbone 都是 `v4_supported=False`；`mcitlib/backbones/`、`configs/backbones/` 只有 LLaVA/InternVL，视频专用环境清单也尚未出现。不能仅改注册开关宣称完成。
- 原视频参考树中 Video-LLaVA 有 12 个方法目录，VideoLLaMA2 有 11 个。目录存在只证明参考实现存在，不证明当前环境可运行或已被 v4 接入。
- VideoLLaMA2 缺 KeepLoRA；两个视频骨干都没有本轮找到的 PCLR、SAME、SiGMA、DGG、Octopus 历史方法目录。**这 11 个无历史目录的组合仍在 34 项目标内**，需要新的适配与实证，不沿用旧盘点中的永久排除。
- canonical 中这五种方法当前仍只声明 LLaVA，且 benchmark 限制为 UCIT/DCL；不能只增加视频 backbone 而遗漏 CL-VISTA 方法入口。已验收的 InternVL 增量在 11 月 17 日私有冻结版本，主线与冻结版本须按回执合并所需差异。其余 12 种已有 canonical InternVL 入口，证据强度各不相同。

机器可读清单见 [baseline-porting-matrix.json](baseline-porting-matrix.json)，记录每个组合的目标身份、历史参考有无与 v4 未迁移状态。该文件是本次快照，不能代替未来运行状态；下一轮必须与当时 registry 对账。

## 2. 一个必须先补的缺口：视频 benchmark

两个旧视频骨干的 `train_CVU.sh` 实际指向 **CL-VISTA**，任务顺序为：

`counting → space → traffic → movie → gui → science → sports → star`

现有 v4 只注册 UCIT、MLLM-DCL、MLLM-ACL，未注册 CL-VISTA。普通逐任务下三角结果应有 36 格；MR-LoRA 还要处理独立专家与 router 的额外预测覆盖，不能简单套普通矩阵。

因此后续接入要同时覆盖：视频样本/解码、八任务配置、Replay 原始 QA、router 数据与独立训练、逐任务 checkpoint、预测分片和 ID 合并、原生 judge/评分、矩阵及预算元数据。**不能用 UCIT/DCL 两步图像训练替代视频闭环，也不能把完整历史视频预算自动替换为图像 3k/1k sample。** 如需短程 sample，应明确它只是工程检查及其独立预算。

两套旧视频评估的 counting 入口都调用 `eval_video_qa_qwen.py`，judge 模型由评测配置传入；其环境与视频推理环境分开。新入口必须绑定 judge 的模型身份、提示、生成与解析规则、失败处理、分母及资源需求。仅生成答案不是评估完成；也不应因为 judge 故障重跑已经有效的训练与预测。

参考树使用 CL-VISTA 目录，`MCITlib_video` 部署树使用 CVU 路径。本次选查 16 个关键文件：`MCITlib_v4` 内旧视频树与 `MCITlib_v3_ref` 对应文件均一致；部署树部分不同。进一步查看两份 LoRA-FT train.py 的差异仅为本地导入路径，选查的 train_CVU/router 差异涉及路径本地化与 CVU/CL-VISTA 名称。**不能据这次局部对比声称整棵部署树仅有路径差异，或两套数据已证明相同。** 接入时要以实际样本、配置与哈希确认别名。

## 3. 逐方法统一矩阵

“有参考”仅表示 `MCITlib_v3_ref` 下有对应方法目录。下表全部 34 个视频目标当前均未建立 v4 真实运行验收。最低行为覆盖的详细检查见 [接入规范第 4 节](backbone-baseline-porting.md#4-全部-17-个方法的最低差异覆盖)。

| 方法（registry ID） | 已有共享集成与应复用的依据 | Video-LLaVA 参考 | VideoLLaMA2 参考 | 视频接入不能漏掉的差异 |
|---|---|---|---|---|
| LoRA-FT `lora_ft` | canonical 普通适配/前任务延续；InternVL 两 benchmark 真实存载推及完整矩阵报告 | 有 | 有 | 视频塔与 projector；独立进程重载所有可训练状态；下一任务延续 |
| Replay `replay` | 显式数据替换和历史原始 QA；图像两 benchmark 完整矩阵报告 | 有 | 有 | CVU/CL-VISTA 样本和媒体路径、历史比例；不能混入 router 字母答案 |
| MoELoRA `moelora` | 共享专家 tuner、初始化、router 与序列化；图像完整矩阵报告 | 有 | 有 | 八任务/专家配置、rank 分配；VideoLLaMA2 视觉塔目标过滤 |
| MR-LoRA `mrlora` | 独立专家与 router 入口、逐题专家选择/ID 审计；原始专家分数和 routed 分数分离 | 有 | 有 | 视频有独立 `train_router.sh` / `Eval_CVU_router.sh`；先确定专家池、覆盖与 fallback 协议 |
| OLoRA `olora` | 已合入非重入 checkpoint 与正交损失；Task2 重复梯度归约有已验证修复 | 有 | 有 | 验收 Task2 正交项、历史冻结及实际 checkpoint 调用，不能只跑 Task1 |
| CL-MoE `clmoe` | 任务上下文代替硬编码 task 文件；诊断输出与数值 router 分离 | 有 | 有 | 八任务标签映射；声明的 topk/noisy 属性不等于实际数值分支 |
| DISCO `disco` | prototype/feature/boundary 存载；请求级常量专家缓存已有真实 7B 对照 | 有 | 有 | **两种视频参考均为 Top-K=3 推理**；不能套 LLaVA 全专家融合；保留动态 mask/SSA 语义 |
| HiDe `hide` | 图文 anchor/boundary、分层路由、A/B 分别求和及交叉项、请求缓存 | 有 | 有 | **Video-LLaVA 特殊层 Top-K；VideoLLaMA2 参考不同**；显式特殊层映射与视觉塔过滤 |
| SMoLoRA `smolora` | 独立可分离双路由、instruction embedding 显式哈希加载、gate 状态 | 有 | 有 | 视频八任务 embedding 顺序和 ins_type；不能用普通 MoE 替代双路由 |
| ModalPrompt `modalprompt` | 图文 cosine、inactive FP32 master 行恢复、ZeRO-3 加载；图像训练/推理修复有证据 | 有 | 有 | 两模型 prompt rank 不同；词表+prefix×任务数动态形状；必须保留旧 prompt 和投影器 |
| SEFE / RegLoRA `sefe` | ordinary LoRA、关键元素累计/正则、逐任务 merge；11 月 17 日设备映射/保存修复已合入 | 有（RegLoRA） | 有（RegLoRA） | 各模型真实调用入口、merge 后生成、Mistral/STC 与媒体塔保存；不迁移未调用的历史分叉 |
| KeepLoRA `keeplora` | 梯度/权重提取、SVD 初始化、冻结 A 与 merge；已有 40 GiB 准备阶段 OOM 记录 | 有 | **无，须新增** | 按视频模型目标张量核算辅助峰值；四卡训练不会自动分片 dense 梯度准备 |
| PCLR `pclr` | LLaVA 已合入；InternVL 有界双 benchmark 训存载推与 integration 证据在冻结版本 | 无 | 无 | 视频 query 构建、八任务 bank/压缩计划、独立 LR/batch；媒体梯度与存载 |
| SAME `same` | 修正后的图文全局先验、整数计数、后期谱修正；InternVL 单独 CLIP 预处理 | 无 | 无 | 视频帧先验与时序聚合需明确；八任务 experts/rank、触发谱分支；不可静默 text-only |
| SiGMA `sigma` | 历史专家冻结、活跃专家/projector 更新、顺序合并；记录器幂等修复 | 无 | 无 | 八任务与特殊层关系；Llama/Mistral 目标层、合并 dtype 顺序及后续任务 |
| DGG `dgg` | LLaVA 原生 optimizer/master/scheduler/RNG 恢复与保留政策；InternVL 尚有空间阻塞 | 无 | 无 | 视频 replay/guidance 与真实原生 checkpoint；容量下界先算，不能删唯一恢复副本过关 |
| Octopus `octopus` | Stage1→GPWC→Stage2、两项正则、非重入兼容与临时辅助清理；InternVL 有界 UCIT 证据 | 无 | 无 | 两模型媒体塔/STC/语言侧完整可微梯度、原生辅助预算、跨阶段存载及峰值 |

这里没有把图像的完整矩阵或 CPU 通过当成视频已通过。尤其 KeepLoRA 的已实现代码伴随资源阻塞，ModalPrompt/InternVL UCIT 只有用户指定的流程验收；较早静态进度页也不能用来覆盖后来 SEFE 合入或新五方法冻结交付。

## 4. 旧集成补出的共同根因

### 4.1 训练“正常”也可能缺失方法本身

历史 ModalPrompt 曾缺图文 cosine loss 与 inactive FP32 master prompt 恢复；DISCO/HiDe 曾需要补齐统计量保存和重载优先级。它们可以有正常主 loss、生成文本，甚至产出 checkpoint，但方法语义不完整。SMoLoRA 若仅当普通 MoE 处理，也会遗漏指令侧路由。

**统一规则**：每个方法都有状态/损失/辅助阶段清单，验收同时看“应该更新的发生变化”和“应该保留的没有变化”，跨进程存载才算闭环。缺统计状态不能用默认值补齐后声称恢复。

### 4.2 同一方法名不保证跨骨干同一计算

本次直接读代码确认：

- Video-LLaVA DISCO：`CoIN/peft/tuners/clitmoelora.py` 定义 top_k=3 并在推理分支 topk、归一化和累加。
- VideoLLaMA2 DISCO：同样有 Top-K，且模块替换跳过 vision_tower。
- Video-LLaVA HiDe：第 31 层存在 Top-K 分支；VideoLLaMA2 对应文件没有这条同形 Top-K 路径。
- 现行 v4 的 DISCO/HiDe tuner 与私有 model override 仍服务 LLaVA/InternVL；HiDe 代码仍有 31 的特定判断。两个本地语言模型配置恰好均为 32 层，也不能据此把特殊层语义无条件泛化成“任意最后层”。

**统一规则**：以方法×模型×benchmark 选择已明确的参考语义。共享接口和修复机制，保留已确认的模型变体。若为了统一论文实验而改变历史变体，显式作为科学设计选择处理，而不是技术重构时偷偷改变。

### 4.3 保存权重与保持状态是不同问题

OLoRA Task1 成功而 Task2 ZeRO 重复归约、ModalPrompt 当前 prompt 可更新但历史 master 行漂移、SEFE merge 能写出文件但后续生成/自动设备映射失败，都说明“训存载推”必须跨到真正触发行为的边界。

**统一规则**：两任务/后期分支按方法选择；区分 adapter、non-LoRA projector、统计 sidecar、prompt、辅助矩阵与原生恢复状态。复用 OLoRA 非重入 scope、分区感知加载、SEFE 保存配置修复，不从旧副本重新实现。

### 4.4 验收器把一个 benchmark 的形状或数值当成全局常量

ModalPrompt 的旧验收器曾将 DCL prefix 按 10 而非计划中的 20 检查，错误期待 32050 行，而实际应为 32100；更正验收器后复用了已完成训练，没有为假失败重训。另有“有效学习率相差十倍”的早期判断被撤回，因为应读取实际 DeepSpeed optimizer，而非未生效的 Trainer 分组。

**统一规则**：shape、专家数、词表、prefix、有效 batch/LR 都从绑定计划和实际优化器导出；算法测试与诊断假设分别验证。历史审计更正必须优先于更早 failed/verdict。

### 4.5 评估闭环不是训练的附件

MR-LoRA 曾需补全专家预测再按 question ID 选择，专家成绩不能充当 router 成绩；caption scorer 又出现依赖/Java 资源问题。不同历史恢复记录有不同评分处理，不能混用。视频现有入口还有独立 judge 模型和环境，资源可能与待训练模型竞争。

**统一规则**：评估链在首次大模型运行前做依赖和轻量路径检查；绑定 ID、分母、专家池、judge 和解析口径。评分故障优先从已验证预测恢复，不重训、不默默删指标或丢失败样本。

### 4.6 辅助阶段与主训练不是相同资源问题

KeepLoRA dense 梯度/SVD 准备，SEFE merged 权重及关键元素，后来 Octopus GPWC 和 DGG 原生 checkpoint，属于同一种反复出现的遗漏：只看 LoRA 训练显存/adapter 大小。旧 MoELoRA 任务 1 能跑，任务 2 仍可能 OOM，也不能只测初始任务。

**统一规则**：资源表覆盖所有方法的准备、训练、后期历史增长、保存交接、推理与 judge。可以先完成可运行组合，但资源阻塞仍保留在全方法覆盖清单，不得报告“全部完成”。

这些问题把上一轮复盘进一步扩展了：除了启动与控制层返工，还有**方法语义丢失、跨模型变体被抹平、持久状态不完整、评估协议错认**，它们往往不会立即报错，对后续实验更危险。

## 5. 两个视频骨干应共享什么、分别验证什么

| 层 | 可复用 | 必须分别验证 |
|---|---|---|
| 生命周期与控制 | 已验收 bootstrap、doctor 规范、租约/磁盘/阶段续跑、幂等记录器 | 两环境的实际子进程导入、GPU 映射、完整命令与 scorer/judge |
| 方法计算 | 已确认等价的 tuner/损失/状态处理及测试夹具 | Top-K/目标过滤/特殊层等模型变体；图像到视频特征语义 |
| 视频输入 | 八任务 manifest、样本/分片/ID 核对、结果协议 | LanguageBind image/video 双塔 vs CLIP/STC；帧处理、时序维、padding、token 展开 |
| checkpoint | 完整状态清单、来源绑定、失败回执与兼容边界 | Llama vs Mistral，MLP vs STC，生成入口、目标键与额外状态 |
| 资源 | 核算方法、共享记账和后期检查点 | 实际视频长度、梯度范围、两种模型各自峰值与 judge 占用 |

本地模型配置证据：Video-LLaVA 为 `LlavaLlamaForCausalLM`、LanguageBind 图像/视频塔、`mlp2x_gelu`；VideoLLaMA2 为 `Videollama2MistralForCausalLM`、CLIP 视觉塔、`stc_connector`，历史 Task1 传 `--num_frames 8`。这些是读取到的配置，不是已加载运行验证；路径含占位值，应通过配置映射解决，不能修改原模型权重配置。

## 6. 后续适配的执行顺序与交付条件

1. **统一清单和来源**：检查 34 项覆盖；合入所需已验收修复；区分 canonical、冻结增量、视频参考和部署配置；先核算全部方法的明显资源下界。
2. **共享视频基础闭环**：补两个 backbone 与 CL-VISTA 数据/评分适配。每个模型先用 LoRA-FT 验证最小真实视频训练、完整保存、独立重载、原生生成/评分及下一任务；不在此时要求全部样本结果。
3. **按能力依赖推进方法**：复用基础训练完成 Replay；再完成专家/门控/正交类；独立 router 类；图文引导/先验/prompt 类；最后接通各方法的梯度、压缩、merge 和原生恢复流程。分组只决定复用顺序，不改变 17 方法目标或擅自改正式执行顺序。
4. **第二模型只补真实差异**：复用通过的控制器、夹具和接口；仍需自己的视频媒体路径、变体、存载和容量证据。不要重新写一套调度/冻结系统，也不要声称第一模型的 GPU 验收覆盖第二模型。
5. **按当前授权交付**：每个组合至少记录 implemented、branch_coverage、train/save/reload/infer/score、resource、formal_run 和缺口。全方法完成必须 34 项都有明确结果；blocked/not_run 不能计作通过。用户只要求挂正式队列时，工程通过与排队状态分开交付，不再等所有完整结果才结束。

本轮没有开始上述实现；已完成的是旧集成复核、全方法规则与目标清单。当前资源阻塞不意味着永久无法适配，是否能全部在现有硬件运行要由各模型实际资源核算决定。

## 7. 可追溯来源

代码路径相对于本仓库根目录；视频参考路径相对于另行检出的 v3 源码。

- `mcitlib/registry.py`、`mcitlib/methods/*.py`、`mcitlib/tuners/variants/{disco,hide}.py`：本次读取的注册、入口和现行计算路径。
- `docs/llava_implementation.md`、`docs/training_entry_repairs.md`、`docs/internvl.md`、`docs/troubleshooting.md`：现行集成契约。
- `docs/migration/moe_lora_variants.md`、`docs/migration/inventory.md`：历史视频分叉与覆盖，具体 Top-K/过滤已用实际代码复核；旧“不支持”范围不覆盖用户新的全方法目标。
- `reports/engineering/gpu-fixes-20261107/current-progress-20261107.md`：ModalPrompt 修复与验收器更正、MR 路由/评分、KeepLoRA OOM。
- `reports/engineering/llava-mainline-20261110/README.md`：OLoRA/DISCO/HiDe 已安装修复与真实对照边界。
- `reports/engineering/internvl-integration-20261110/README.md`：12 方法入口与有限 GPU gate；共享基础成功不等于全部方法成功。
- `reports/engineering/modalprompt-ucit-flow-20261113/README.md`、`modalprompt-epoch1-20261113/README.md`：流程验收与历史预算不能混同。
- `reports/engineering/internvl-sefe-promotion-20261117/completion.json`、`reports/engineering/internvl-new-baselines-20261117/delivery.md`：较新修复和五方法真实交付边界。
- 参考树两视频骨干下各方法的 `scripts/MCITlib/Train/`、`Eval/`、`Eval_router/`，以及 `configs/data_configs/CL-VISTA`：视频 benchmark、路由和原生评分入口。

本地保存来源文件与哈希的 `source-evidence.json`。这是一轮代码与记录复核，没有复跑旧模型、重新计算历史指标或将旧 benchmark 的预算强行统一。
