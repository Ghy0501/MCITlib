# LLaVA L3 report

状态：**四个原型方法的结构迁移与 CPU 验证完成，按计划在此暂停，等待宏观 Agent 审查接口。GPU T4/T5 尚未执行，不宣称完整数值等价。**

## 完成项

- `mcitlib/tuners/mcit_expert_lora.py`、`_peft_registry.py` — 暴露专家合并、任务状态等变体覆盖点，保留 v3 PeftType 兼容注册 — `40e93e7`。
- `mcitlib/methods/lora_ft.py` — LoRA-FT 参考插件，沿用标准训练入口与 peft LoRA — `26527ac`。
- `mcitlib/methods/replay.py` — 仅覆盖训练数据选择，使用 `replay_path` 和 `router_folder`，其余数据字段保持不变 — `ecb4a96`。
- `mcitlib/methods/olora.py`、`mcitlib/tuners/variants/olora.py` — O-LoRA 变体、`use_reentrant=False` 和 0.05 正交损失钩子 — `db2254a`。
- `mcitlib/backbones/llava/overrides.py` — 通用模型覆盖安装钩子，方法私有代码不进入 `third_party/` — `6181d27`。
- `mcitlib/methods/hide.py`、`mcitlib/tuners/variants/hide.py`、`mcitlib/methods/_hide/` — HiDe 变体、私有 LLaVA 覆盖及 capability 声明；旧 peft 导入改经 MCITlib 注册层 — `77e443a`。
- `mcitlib/runner.py`、`mcitlib/cli.py` — 按任务顺序生成 train/eval dry-run 清单；训练、评测和额外步骤保留 v3 顺序，NB-5 安装命令不进入任务步骤 — `aeea64f`。
- `tools/regression/` — T4/T5 命令预览、GPU 执行入口和 loss 比较工具，默认不执行训练 — `70b97be`。
- `tests/equivalence/`、`tests/unit/` — 方法插件、O-LoRA/HiDe 模块、数据管线和完整 dry-run 等价测试 — `314f822`。
- `.github/workflows/ci.yml`、`tools/add_spdx.py` — CI 显式设置 vendored 包路径；SPDX 检查兼容上游许可证头 — `0d621ff`。

## 最终接口签名

```python
class ContinualMethod(ABC):
    model_override: ClassVar[str | None] = None
    def select_train_data(self, task: TaskSpec) -> DataSource: ...
    def model_class_override(self) -> type | None: ...
    def build_trainable(self, model: Any, backbone: Any) -> Any: ...
    def training_overrides(self) -> dict[str, Any]: ...
    def extra_loss(self, model: Any, inputs: Any, outputs: Any, base_loss: Any) -> Any: ...
    def before_task(self, ctx: RunContext) -> list[Step]: ...
    def after_task(self, ctx: RunContext) -> list[Step]: ...
    def prepare_inference(self, model: Any, ctx: RunContext) -> Any: ...
    def extra_stages(self) -> list[Stage]: ...

class LlavaAdapter(BackboneAdapter):
    def build_command(self, run_cfg: RunConfig, method: Any, task: TaskSpec) -> Command: ...
    def load_base(self, model_cfg: Any, override_cls: type | None = None) -> Any: ...
    def build_dataset(self, source: DataSource, split: str, entry: str, **kwargs: Any) -> Any: ...
    def make_trainer(self, model: Any, dataset: Any, args: Any, method: Any, **kwargs: Any) -> Any: ...
    def generate(self, model: Any, batch: Any, gen_cfg: Any) -> list[str]: ...

def build_sequence_commands(
    config_root: str | Path,
    backbone_id: str,
    method_id: str,
    benchmark_id: str,
    paths: Any = None,
    task_id: int | None = None,
    only_eval: bool = False,
) -> list[Command]: ...
```

与 §4 的差异：增加 `model_override` 字符串标识，用于在训练子进程导入模型前安装方法私有类；`build_dataset` 显式保留 `entry=standard|moe`，没有合并两条数据管线。`build_sequence_commands` 是当前 dry-run 编排入口，真实训练执行仍属于后续 GPU runner 工作。

## 测试结果

| 层级 | 序列/模块数 | 通过 | 失败 | 产物路径 |
|---|---:|---:|---:|---|
| T0 配置 | 36 条普通序列（L1）；其中原型 12 条 | 36 | 0 | `docs/migration/llava-L1-report.md`、`tests/unit/test_config_migration.py` |
| T1 调优器模块 | O-LoRA、HiDe 两个小模型对照 | 参数名/形状/初始化、state_dict、train/eval 前向及梯度均 `torch.equal` | 0 | `tests/equivalence/test_tuner_variants.py`、`_tuner_harness.py` |
| O-LoRA 损失 | 2 个单元场景 | 首任务零损失；后续任务 0.05 正交损失公式一致 | 0 | `tests/unit/test_methods.py` |
| T2 数据管线 | standard/MoE 单图、standard 多图、OLoRA/HiDe 实际 v3 目录，共 5 组 | `input_ids`、`labels`、`attention_mask`、`images` 逐位一致 | 0 | `tests/equivalence/test_data_pipeline.py` |
| Replay 数据选择 | 1 组 | 仅 `train_path`/`train_folder` 切换，test 字段不变 | 0 | `tests/unit/test_methods.py` |
| T3 原型命令 | 4 方法 × 3 基准 = 12 条序列，60 个训练任务 | 训练 argv 一致；完整 train/eval dry-run 的 argv、kind、task_id、顺序一致；`eval --after-task` 集合一致；仅 NB-5 安装步骤除外 | 0 | `tests/equivalence/test_lora_command.py` |
| GPU 回归工具 | 2 个 CPU 测试 + shell 语法检查 | 相对误差比较、默认 dry-run、`bash -n` 通过 | 0 | `tests/unit/test_regression.py`、`tools/regression/` |
| 全量 CPU 测试 | 75 | 75 | 0 | `/tmp/p0/l3-final.out` |
| lint / SPDX / third-party | 自有源文件、2 个上游组件 | 全部通过；SPDX 缺失 0；上游错误 0 | 0 | `/tmp/p0/l3-ruff.out`、`l3-spdx.out`、`l3-third-party.out` |
| 其他骨干不变性 | InternVL、Video-LLaVA、VideoLLaMA2 | 相对 `v3.0-final` 无差异 | 0 | `git diff --quiet v3.0-final -- InternVL Video-LLaVA VideoLLaMA2` |
| T4 / T5 | GPU | 未执行 | — | 需作者在 GPU 节点运行 |

T1 的 LoRA-FT/Replay 没有新增调优器变体，因此本阶段新增模块对照针对 O-LoRA 和 HiDe。T2 使用合成图片、JSON 和小型自定义 tokenizer/processor，不下载预训练权重；v3/v4 均在独立子进程中运行。上述结果不覆盖真实 7B 模型前向、checkpoint 全流程或 GPU 训练。

## 与计划的偏离

- 保留 `standard` 与 `moe` 两份数据实现，没有为了去重强行统一；这是计划 §4.2 允许的等价性优先选择，不涉及 §9。
- `train/eval` 当前提供 dry-run 编排，真实执行由后续 runner/GPU 工作完成；T3 只验证生成的命令，不声称训练已经运行。
- O-LoRA 的 v3 `train_MOE.py` 存在未导入 `sys` 的缺陷；v4 不依赖该路径。参考子进程仅为导入测试注入 `builtins.sys`，已登记 drift，不宣称 v3 原始入口可直接训练。
- NB-6 的配置注入、其余 8 个方法、CoIN eval runner 和指标实现不在 L3 完成范围内，未提前实施。

## 新登记的漂移

- `docs/migration/drift.md` §14：O-LoRA v3 `train_MOE.py` 使用 `sys.path.append` 但未导入 `sys`；记录参考测试的隔离处理和 GPU 风险。
- `docs/migration/drift.md` §15：更新验证边界，明确当前仅四个原型方法完成其 CPU T1–T3，不代表整体数值等价。

## 触发的停止条件

- 无。没有修改其他三个骨干、`paper/`、`reviews/`、`design/`，没有推送远端，没有引入计划外数值行为变更。

## 需要作者执行

GPU 节点先预览，再按环境分别执行：

```bash
# T4 命令预览；默认不启动训练
tools/regression/run_short.sh --combo lora_ft:ucit

# 在匹配 v3 依赖的 GPU 环境中执行原型组合的任务 1、2 前 20 个 loss
tools/regression/run_short.sh --combo lora_ft:ucit --execute
tools/regression/run_short.sh --combo replay:ucit --execute
tools/regression/run_short.sh --combo olora:ucit --execute
tools/regression/run_short.sh --combo hide:mllm_dcl --execute
```

运行前按 `docs/migration/placeholders.md` 和 `tools/regression/README.md` 设置模型、数据、checkpoint 根目录，并确认 v3/v4 使用相同数据、权重、依赖和 GPU 配置。T4/T5 结果不能由当前 CPU 测试替代。

## 下一阶段

**按作者要求暂停，明天再继续。** 下次从宏观 Agent 的 L3 接口审查开始；审查通过后进入 L4，顺序为 `moelora` → `mrlora` → `clmoe` → `smolora` → `disco` → `modalprompt` → `sefe` → `keeplora`。本次不开始 L4。
