# 新 backbone / VideoLLM 的 baseline 接入约定

本文保留接入时的验收约定；两个视频模型的当前交付状态以 [CL-VISTA 运行指南](video-runtime.md) 为准。

目的：把全部 baselines 历次集成中可预防的失败前移，复用已有实现和证据，避免每个模型重新搭建验证系统。统一覆盖 registry 中全部 17 个方法，不再按新旧分组。当前后续目标是 Video-LLaVA、VideoLLaMA2，共 34 个方法/模型组合；缺实现或资源阻塞仍列入目标。

先看 [全部方法集成复核与迁移矩阵](all-baselines-integration.md) 和 [机器可读目标清单](baseline-porting-matrix.json)。这些是 2026-11-17 的已审查快照；开始新任务时与实际 registry 对账，不把静态清单当作永久完整列表。

本文件是项目工作规则，不代表已新增自动化检查。默认阈值用于发现停滞，不是跳过正确性检查或中止有效训练的理由。用户最新明确指令优先。

## 1. 接入前只建立一份契约

在已有任务计划或状态旁记录以下内容，已有信息直接复用：

| 项目 | 必填内容 |
|---|---|
| 交付边界 | 是接通工程链、启动正式队列，还是等待完整结果；分别列必须证据与可选诊断 |
| 来源 | 当前主线、受保护文件、可信历史修复位置、候选代码身份；历史冻结 source 不作编辑目录 |
| 方法差异 | 阶段、学习率、有效 batch、专家数、冻结状态、辅助预算、保存恢复依赖 |
| backbone 差异 | 训练/生成入口、目标层、视觉特征形状、dtype/device、processor、checkpoint 格式 |
| 数据差异 | 目标 benchmark 的入口、任务顺序、训练与 held-out 身份、原生评分；历史视频使用 CL-VISTA 八任务（CVU 脚本），不能套用 UCIT/DCL 或自动新增其他实验 |
| 资源 | 每阶段 GPU/显存、共享盘峰值及安全余量、并发与输出保留方案；使用当前测量，不搬旧模型数值 |
| 验收计划 | 哪个最小诊断证明哪个分支，哪些历史证据可复用，何种改动会使其失效 |

前 15–20 分钟优先完成来源、环境和资源可行性筛查；这是排查顺序的默认目标，不是无法完成时伪造通过的时限。显然不可运行的方法先标阻塞，不等轮到它才发现。

## 2. 明确可复用的起点

- 全方法的共享生产修复见 `docs/llava_implementation.md`、`docs/training_entry_repairs.md`、`docs/troubleshooting.md`、`docs/mrlora_ucit_evaluation.md`；先保留修复再适配模型，不能从更旧方法副本倒退。图像 benchmark 的指标/预算不是视频协议。
- 视频参考为仓库上级 `MCITlib_v3_ref/Video-LLaVA` 与 `MCITlib_v3_ref/VideoLLaMA2`，另有 `MCITlib_video` 部署树。本次抽查发现后者包含路径本地化与 CVU/CL-VISTA 命名差异；使用前核对数据身份及相关变更，不能仅凭目录较新选代码。
- 当前 v4 没有两个视频骨干的正式 adapter、backbone 配置和环境清单；registry 也没有 CL-VISTA。必须把 backbone 与 benchmark 接入作为共享工作，而不是 34 个组合各复制一份入口。
- LLaVA 五方法：`reports/engineering/llava-new-baselines-20261110/MAINLINE-integration-20261114.md`；SAME 必须结合该目录中较新的 `SAME-mainline-delivery-20261115.md`，不能退回没有完整全局先验的旧版本。
- InternVL：`reports/engineering/internvl-new-baselines-20261117/delivery.md` 记录本轮交付边界；`.workspaces/internvl-new-baselines-20261117-control/independent-review/llava-lessons-for-internvl.md` 汇总旧坑。
- 截至本轮交付，新的五方法适配没有整体合入 canonical 主线；可用实现位于 `.workspaces/internvl-new-baselines-20261117-formal-01/source`，UCIT doctor 后续两行环境修复见 `.workspaces/internvl-new-baselines-20261117-ucit-recovery-01/source`。使用前核对当前交付记录，若已有更新主线，以更新的验收记录为准。
- 这些目录是只读参考。下一模型从当前主线建立开发副本，只提取所需且已审查的差异，不整树覆盖。先核对科学代码与必要控制层修复，再生成自己的运行版本。
- 共享锁文件与记账文件是资源协调状态，不能因复制科学源码而产生第二套互不相见的锁。已有安全复制/冻结工具可复用；不可为了减少复制成本改动运行中的冻结源码。

## 3. 使用固定、渐进的验证顺序

1. **便宜检查**：实际解释器与导入来源、支持的依赖版本、必要环境变量、CLI 参数、spawn、doctor/scorer、阶段参数展开。通过真实执行器走一次轻量入口；其允许的检查应避免加载完整模型。
2. **最小真实分支**：选能够触发目标行为的步数和任务边界，完成真实更新→保存→重载→原生推理/评分。一个普通两步任务不能覆盖所有方法，见下表。
3. **相关容量边界**：语义正确性与最长上下文/后期专家/辅助梯度的容量分别记录。可以用明确的合成压力输入，但不能据此声称真实效果或改写失败语义回执。
4. **正式启动**：使用同一执行路径与正式参数；启动前只核对最终差异、当前资源与首任务接续。不为已验收且身份不变的分支重新整套验收。
5. **按边界交付**：若用户只要求挂队列，确认正式进度增长和必要恢复/交接信息后交付；未覆盖项写清楚，不能私自扩大为“等全矩阵完成”。

新增检查只有在能指出具体未验证风险、最小触发条件和通过标准时才进入关键路径。格式润色、重复清单、可复用的全量审查不阻塞已安全可执行的下一步。

## 4. 全部 17 个方法的最低差异覆盖

以下列的是语义与最小触发点。UCIT/DCL、两步或 12 步等历史例子只说明已验收的覆盖思路；视频以选定 CL-VISTA 协议和真实触发条件为准。代码文件存在、CPU 测试通过、真实闭环通过、完整 sample 完成分别记录。

| 方法 | 必须保留的行为 | 最小证据应覆盖什么 | 已知易错点 |
|---|---|---|---|
| LoRA-FT | 前任务 adapter 与额外可训练参数延续；模型专属目标层和媒体投影 | 当前任务真实更新→保存→独立重载→下一任务与原生评估 | 只保存 adapter 漏投影器；将 Mistral/STC 或视频塔硬套 LLaVA |
| Replay | 当前任务与历史原始 QA 的样本来源、预算和媒体路径 | 样本 ID/答案审计，含历史样本的真实更新、保存重载 | router 字母标签不能充当原任务答案；路径字段与目录名不能代替样本身份 |
| MoELoRA | 专家数、rank 分配、初始化和 softmax 路由、PEFT 保存键 | 当前/后续任务、专家和 router 更新、重载一致性 | 每个新进程须注册 tuner；VideoLLaMA2 目标过滤；不能把六/五专家写死 |
| MR-LoRA | 独立专家训练、独立 router 训练、专家池与逐题选取、分别记录预算 | router 输出→ID 匹配的专家预测→原生评分；记录有效池和 fallback | 不能只看专家矩阵；回顾性专家池不等于严格在线评测；视频有独立 train_router / Eval_CVU_router |
| OLoRA | 当前专家训练、历史冻结、正交损失、非重入 checkpoint | 至少触发后续任务的正交项，检查真实 checkpoint 调用、保存/合并推理 | Task1 成功不能排除 Task2 ZeRO 重复归约；不靠关 checkpoint/正则绕过 |
| CL-MoE | 明确任务标识与原有 router 计算；可选诊断与算法隔离 | 任务切换、专家路由与存载；无硬编码 task.txt 依赖 | 声明 topk/noisy_gating 不代表历史实际启用了相应数值算法 |
| DISCO | mask、lora_AB、特征/prototype/boundary 状态、各骨干推理变体 | 状态存载、请求切换、视频 Top-K 实际路径、缓存开关对照 | 两视频历史实现 Top-K=3；不能直接套 LLaVA 全专家融合；动态路由不可缓存 |
| HiDe | 分层图文路由、anchor/boundary、普通层 A/B 分别求和的交叉项、特殊层策略 | 后续任务统计状态、特殊层注入、存载、两种视频变体的实际推理 | Video-LLaVA 有 Top-K，VideoLLaMA2 参考路径不同；31 是参考语义，不能盲改为最后层 |
| SMoLoRA | 视觉/通用与指令双路由、instruction embedding 内容、gate 冻结和融合 | 明确资产哈希、跨任务门控与独立进程加载；视频任务映射 | 不能简化成基础 MoELoRA；不能套用图像任务 embedding 顺序 |
| ModalPrompt | 图文 cosine loss、活跃 prompt、非活跃 FP32 master 行恢复、投影器状态、ZeRO-3 加载 | 至少两任务、实际生效 optimizer/scheduler、prompt 形状、训练/生成独立重载 | 训练有 loss 不证明 cosine/旧 prompt 保护存在；维度由词表+prefix×任务数导出；只读实际优化器 LR |
| SEFE / RegLoRA | 真实入口的 LoRA、关键元素累计、正则、逐任务 merge 与下一任务加载 | Task2 关键元素/正则触发，merge 保存后原生生成 | 避免迁移未调用的旧 MoE 分叉；generation config、设备映射/dtype 和中间 merged 容量分别检查 |
| KeepLoRA | 梯度/权重提取、SVD 初始化、LoRA A 冻结及子空间/merge 依赖 | 辅助准备的真实参数/梯度覆盖、初始化/更新、存载与后续任务 | 准备阶段可能比训练更耗显存；VideoLLaMA2 无历史实现须新接入，不能永久排除 |
| PCLR | 学习与 integration 的独立 LR/batch；历史 bank/key/query；压缩与蒸馏 | 有真实参数差分的学习阶段、跨任务重载、实际 integration 的 KL/更新、各 benchmark 自己的训练与评分入口 | sample profile/worker 二次覆写；早期诊断强制压缩必须单独标记；重复合成 query 可使 alignment 为零 |
| SiGMA | 活跃专家更新、历史/未来专家和 router 冻结、可训练 projector、任务顺序合并 | 至少能覆盖历史专家关系的继续训练；保存/reload；另一个 benchmark 的专家数与入口差异 | Trainer 可多次调用 create_optimizer；记录器不得因重复回调阻断训练；projector LR 单独保留 |
| SAME | 图文全局先验与局部路由、独立 CLIP 输入、FP32 状态与整数计数、后期谱修正 | 后续任务真实跨过修正触发步（本轮为第 10 步，用 12 步诊断）；prior 保存/重载；两 benchmark 的专家数和原生评分 | 不能复用 InternViT 归一化图像给 CLIP；不能静默退成 text-only；eval hook 不得污染 train 测试 |
| Octopus | Stage1→GPWC→Stage2；视觉/语言完整梯度、两项正则、历史顺序、原生辅助预算 | 一个真实可微 GPWC 分支，Stage2 两项损失参与更新，保存/reload/score；另行声明诊断逻辑 batch 与正式辅助样本总量 | 目录匹配硬编码 LLaVA；辅助大文件与消费后清理；重入 checkpoint 的重复 reduction 已有修复不能丢 |
| DGG | 新任务 adapter、上任务 guidance/replay、原生 optimizer/master/scheduler/RNG 恢复 | 资源可行后证明原生保存及实际恢复；推理副本不能替代训练恢复状态 | 必须同时容纳旧有效 checkpoint 与下一次保存；旧模型 checkpoint 大小不是新模型容量证明 |

表中的语义必须保留；数值遵循当前正式协议，不能把 InternVL 的专家数、步数或 36 GiB 余量硬编码到所有 VideoLLM。未触发的方法分支明确记为未覆盖。PCLR/SAME/SiGMA/DGG/Octopus 尚无本次找到的视频历史方法目录，需要设计模型及八任务协议适配，不能只放开 supported_backbones。

### Octopus 的 sample 预算

2026-11-18 起，新建 Octopus sample 实验时，按任务缩放 Stage2 的训练样本数。令 `N_full` 为该任务对应版本的全量训练集大小，`N_sample` 为其 sample 训练集大小，`B_full` 为该任务完整协议的 Stage2 样本数。使用 `B_sample = min(N_sample, max(1, round_half_up(B_full × N_sample / N_full)))`，从本任务的 sample 训练集按固定种子无放回抽取。完整协议没有 Stage2 优化的任务继续跳过。检查两套训练集的身份和子集关系。全量分母缺失时，先确认该实验的预算口径，不自动套用全量数值。

当前 InternVL MLLM-DCL 的 3k/1k 子集示例（全量 Stage2 预算均为 9,600 条）：

| 任务 | 全量训练集 | 采样训练集 | Stage2 样本预算 |
|---|---:|---:|---:|
| Task2 Med | 22,842 | 3,000 | 1,261 |
| Task3 AD | 60,000 | 3,000 | 480 |
| Task4 Sci | 30,747 | 3,000 | 937 |
| Task5 Fin | 60,000 | 3,000 | 480 |

启动前核对最终命令与数据，记录两套训练集的身份、原预算、实际抽取条数、随机种子、有效 batch 和优化步数。确认 Stage2 有实际优化步。UCIT 和后续视频 sample 各自使用对应任务的全量数据和原预算。GPWC 单独记录，Stage2 的 epoch、学习率、历史 LoRA 和正则沿用所选协议。

这是后续 sample 实验的预算规范。全量复现仍采用对应完整协议。本轮已冻结的 InternVL 队列使用之前的预算，结果按实际运行的 Stage2 样本数标注。比例缩放后的结果应注明预算，便于与旧结果比较。

## 5. 两个 VideoLLM 额外需要确认的契约

本地检查确认 Video-LLaVA 配置为 Llama、LanguageBind 图像/视频塔、MLP projector；VideoLLaMA2 为 Mistral、CLIP 视觉塔、STC connector，其历史训练参数含 8 帧。它们均不是 InternVL 换一个模型路径。模型配置仍有占位路径，使用路径映射/内存配置，不改原始权重配置。

历史视频序列为 counting → space → traffic → movie → gui → science → sports → star；普通持续学习矩阵由此为 36 格，MR-LoRA 的完整专家预测覆盖另算。现有评估还调用独立 Qwen judge，必须绑定 judge 模型、解释器、提示/解析规则和结果分母，并纳入显存/任务调度；不能拿生成退出码代替评分成功。历史 CVU / CL-VISTA 两种目录命名需核对样本与配置一致性。

以下额外预防项尚不代表视频 GPU 验收已完成：

- **媒体输入**：解码后帧的数量、顺序、采样时间与 mask；训练、重载生成和 scoring 使用同一约定。禁止为让 smoke 通过而默默变成单帧图像任务。
- **特征与 token**：核对 patch/token 展开及时间维的位置、projector 输入维度、padding；上下文容量按实际多模态展开长度计算，不能只数文本 token。
- **独立先验/辅助塔**：SAME 的 CLIP 每帧预处理与时序聚合需要明确定义并绑定到训练、推理和保存配置；若原方法未定义视频聚合，作为科学设计选择明确提出，不能静默平均。
- **训练目标**：验证新模型实际 target modules、requires_grad、梯度路径和保存键；不能凭 LLaVA/InternVL 的层数或名称正则判成功。
- **dtype 与传输**：分别检查训练/生成的 BF16/FP16、FP32 norm/统计量与 attention 输入，以及跨卡特征传输是否保留梯度。
- **容量与复用**：第一个 VideoLLM 固定一套可复用执行器；第二个复用通过的控制路径和方法级检查，仅重验模型/processor/数据/资源相关差异。第一模型的成功不等于第二模型的训练或容量已验证。

## 6. 验收脚本也必须遵循真实生命周期

- `create_optimizer`、forward/generate hooks、checkpoint recomputation 等可能多次调用；记录工具须幂等，写不同阶段/事件时带唯一身份，不靠“文件已经存在就失败”判断错误调用。
- 用参数前后差分证明更新；历史参数本来非零，不能据保存后非零作结论。零损失只在该项理论上应被激活的输入条件下构成失败。
- 先用确定性微型输入检查 recorder 和断言，再付完整模型加载成本。需要不同输入才能触发的分支，不允许用重复四行数据替代。
- 模拟 monkeypatch 应还原；真实 train/eval 通过独立进程运行时，相应生命周期测试也应隔离。遇到 baseline 与 candidate 同错，先排环境或夹具，不先改方法。
- CPU 测试隐藏 GPU；双进程 Gloo 需要本地通信，受限环境失败应明确报告并在已授权可用入口核验，不能删除断言。

## 7. 失败、复验与快照失效范围

| 变化 | 应做的最小复验 | 通常不必重做 |
|---|---|---|
| 文档、展示状态字段 | 内容和链接/状态语义检查 | 模型加载、训练和原生评分 |
| observer / 日志工具 | 重复回调、失败路径和不改变算法的检查；必要时一个相关真实事件 | 已证明不受影响的其他方法训练 |
| doctor、环境透传、启动参数 | 同一真实子进程链中的导入/参数/scorer、资源映射与交接 | 科学字节与输入身份未变的全部算法验收 |
| GPU/磁盘协调 | 冲突拒绝、退出后释放、共享资源计账及目标工作流的真实交接 | 每个方法一套新调度器 |
| 算法、processor、checkpoint 或关键协议 | 受影响方法/分支的真实更新、保存/reload，必要的 benchmark 差异与容量 | 无依赖关系的已验收分支 |

每次失败记五件事：最早失败位置、所属层、一个可证伪假设、最小检查、修复影响范围。连续两次同类失败或 20 分钟没有新有效证据时，先重新定位并报告，再继续可独立推进的工作；不要靠多加测试或新建快照掩盖没有进展。

保留原失败记录，续跑从已证明兼容的阶段边界开始。端口冲突需检查存活拥有者、退出收尾与新 rendezvous；没有证据时不能直接归因 TIME_WAIT。DGG 等原生恢复路径必须另证，不能把“跳过训练前 doctor 失败”泛化为任意阶段自动续跑。

限域合并遵循：预备份→预条件检查→受限安装→验证；失败只回滚本次目标文件。已有安装工具及其故障注入证据可复用，工具未改时不重新设计一套事务安装器。禁止覆盖别人的/并行任务的改动。

## 8. 网络和状态是独立故障域

CLI 连续重连默认超过 5 分钟且无新事件时，检查实际长驻进程的路由/代理、隧道存活、DNS/TLS/服务响应及会话事件。新 shell 的 HTTP 探测通过只证明该 shell；恢复必须由原目标会话继续产生新事件确认。不要在日志中打印凭据；网络处理不应停止独立运行的训练。

沿用已有 status 格式，只保留一个当前状态，历史以时间戳追加。至少包含：当前阶段、每方法/benchmark 的证据级别、最近真实训练/输出时间、阻塞原因、下一动作、源码/日志/结果位置。PID 如需用于控制必须结合启动时间和所有权核对。当前字段不能同时表示 preparing 与 running。

## 9. 下一轮怎样证明成本下降

在已有日志中提取，不为了统计再建系统：

- 首个真实优化步的等待时间（将前置合入另计）；
- 完整模型加载前失败 vs 加载后零步失败的次数与原因；
- 控制层故障的重现次数、成功阶段被不必要重跑次数；
- 从用户收紧范围到正式进度出现的时间；
- 网络无新事件时段，和确有工作输出的 GPU/CPU 阶段分别记录；
- 第二个 VideoLLM 新增的控制脚本数量、复用的回执及仍需新证据的差异。

目标是把已有错误在便宜层拦截、把有效训练一次做对、把相同证据复用起来。没有两次可比运行之前，不承诺固定加速倍数或“必定两小时完成”。
