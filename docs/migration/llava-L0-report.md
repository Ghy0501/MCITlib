# LLaVA L0 report

## 完成项

- `docs/migration/inventory.md` — 按审查意见补充上游 URL、克隆 HEAD、复现命令；修正 DoRA 定性 — `a1ee2da`。
- `docs/migration/drift.md` — 将 `pip install -e .` 修正为 1,634 处（Train 426 / Eval 1,208 / 170 个文件），写明统计命令；将 DoRA 标为专有/非 OSI 的 MLOSS 发布阻塞项 — `a1ee2da`。
- `docs/migration/phase-0-report.md` — 同步修正计数和 DoRA 处置 — `a1ee2da`。
- `tools/p0/inventory.py` — `--help` 明确 `--root` 是仓库根目录 — `a1ee2da`。
- `docs/migration/behavior_changes.md` — 建立 NB-1 至 NB-6 登记框架，均未实施 — `a1ee2da`。
- `../MCITlib_v3_ref` — 从 `v3.0-final` 建立独立 v3 参考 worktree，HEAD 为 `a3a622c9`。

## 测试结果

| 层级 | 序列/模块数 | 通过 | 失败 | 产物路径 |
|---|---:|---:|---:|---|
| P0 脚本编译 | 2 | 2 | 0 | `tools/p0/*.py` |
| P0 盘点重复运行 | 47 组合 | 结果一致 | 0 | `/tmp/p0/inventory_recheck.json` |
| pip 计数复核 | 4 项 | 4 | 0 | `drift.md` 中命令 |
| 其他骨干只读检查 | 3 个目录 | 无差异 | 0 | `git diff --quiet v3.0-final -- InternVL Video-LLaVA VideoLLaMA2` |
| T0–T5 | 未执行 | — | — | L1 起实现 |

## 与计划的偏离

- 无范围偏离。DoRA 按 NB-1 排除，不再保留为待决的 vendor 选项。
- P0 产物已按计划独立提交；未推送远端。

## 新登记的漂移

- 无新增算法漂移；本阶段只修正 P0 文档计数和许可证表述。

## 触发的停止条件

- 无新增。DoRA 是已批准 NB-1 的发布合规排除项；CL-MoE/HiDe/ModalPrompt 的许可证文本补充仍是发布前事项，不阻塞本计划编码。

## 需要作者执行

- 暂无阻塞编码的动作。后续在 GPU 上运行 T4/T5；发布前补齐相关上游 LICENSE 或授权材料。

## 下一阶段

进入 L1：建立 `mcitlib` 骨架、注册表、配置/路径/CLI、SPDX 与 CI；实现 `v3_command_dump.py` 和 `migrate_v3_configs.py`，生成 36 条序列及 MR-LoRA router 序列的命令清单，并完成 T0。
