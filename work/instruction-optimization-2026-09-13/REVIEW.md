# AGENTS.md 与 Codex skills 优化候选稿

状态：**用户已确认，13 个文件已全部写回并通过内容一致性校验**。原始快照与批准稿保留。新会话实际加载行为尚未验证。

依据是 OpenAI 于 2026-09-11 发布的 [Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)。本次采用其中的短描述、按需加载、减少固定流程和明确完成边界原则。具体取舍根据本机实际文件与项目科研约束制定，并非照搬通用模板。

## 具体修改集

共 **13 个现有 Markdown 文件**：2 个规则文件、10 个技能入口及 1 个论文输出规范。表中原文件链接指向实际生效位置，候选稿可直接审阅。

| 文件 | 拟修改内容 | 行为与授权影响 |
|---|---|---|
| [project-agents 原文件](D:/PythonProject/retina_rf_SNN/AGENTS.md) · [候选稿](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/proposed/project-agents.md) | 补充完成条件、按需读取、继续已授权验证与新 spike 访问限制；保留全部冻结科研约束。 | 明确现有任务内自主执行；不授权新研究或修改冻结协议。 |
| [hephaestus 原文件](C:/Users/win11-pc/.codex/plugins/cache/sisyphuslabs/omo/4.13.0/components/rules/bundled-rules/hephaestus.md) · [候选稿](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/proposed/hephaestus.md) | 删除旧模型身份和重复通用训诫；取消默认宽检索、固定代理数量和常驻完整工作循环。 | 减少插件自行添加的流程门槛；保留用户/项目明确的审批、独立审阅与安全要求。 |
| [programming 原文件](C:/Users/win11-pc/.codex/plugins/cache/sisyphuslabs/omo/4.13.0/skills/programming/SKILL.md) · [候选稿](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/proposed/programming/SKILL.md) | 855 字符描述缩为 129；入口改为语言参考路由，取消对每个小修改的固定阅读、文件尺寸和工具链处方。 | 允许依项目现有工具完成局部修改；不授权迁移、安装或数值协议变更。 |
| [debugging 原文件](C:/Users/win11-pc/.codex/plugins/cache/sisyphuslabs/omo/4.13.0/skills/debugging/SKILL.md) · [候选稿](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/proposed/debugging/SKILL.md) | 去掉固定三个假设、失败两轮后固定代理审阅及强制专业工具；保留证据驱动诊断。 | 诊断方式按证据选择；只读诊断仍不能修复，缺证据不能宣称已修好。 |
| [ulw-plan 原文件](C:/Users/win11-pc/.codex/plugins/cache/sisyphuslabs/omo/4.13.0/skills/ulw-plan/SKILL.md) · [候选稿](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/proposed/ulw-plan/SKILL.md) | 不再因步数自动进入完整规划流程；完整 .omo 规划按需加载。 | 保留显式 $ulw-plan 的方案批准和实现交接；已授权实现中的内部规划不重复索要启动确认。 |
| [review-work 原文件](C:/Users/win11-pc/.codex/plugins/cache/sisyphuslabs/omo/4.13.0/skills/review-work/SKILL.md) · [候选稿](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/proposed/review-work/SKILL.md) | 保留五个审阅视角，按相关风险选择；删除固定五代理、15–30 场景和所有外部上下文检索。 | 普通审阅不再由插件强制五个独立代理全部通过；用户/项目明确要求的检查与独立性仍必须满足。 |
| [frontend 原文件](C:/Users/win11-pc/.codex/plugins/cache/sisyphuslabs/omo/4.13.0/skills/frontend/SKILL.md) · [候选稿](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/proposed/frontend/SKILL.md) | 从强制 design + perfection 同读改为按设计、性能、样式查询路由。 | 普通局部 UI 修改不触发整站审计；仍需与变更相称的浏览器证据。 |
| [visual-qa 原文件](C:/Users/win11-pc/.codex/plugins/cache/sisyphuslabs/omo/4.13.0/skills/visual-qa/SKILL.md) · [候选稿](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/proposed/visual-qa/SKILL.md) | 普通界面检查用实际截图和交互证据；保留差异工具与 CJK 检查，移除固定两代理。 | 放宽普通 UI 的默认多代理流程；克隆/设计移植保留独立视觉比较和独立代码审阅两条路径。 |
| [ponytail 原文件](C:/Users/win11-pc/.codex/plugins/cache/ponytail/ponytail/4.9.0/skills/ponytail/SKILL.md) · [候选稿](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/proposed/ponytail/SKILL.md) | 收窄触发；删除默认每条回复持续生效和先交付缩水版的措辞。 | 所有强度均保留完整需求；改变明确要求的替代方案先取得用户决定。 |
| [paper-reading-zh 原文件](C:/Users/win11-pc/.codex/skills/paper-reading-zh/SKILL.md) · [候选稿](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/proposed/paper-reading-zh/SKILL.md) | 按完整精读或局部问题路由；仅在请求时展开复现、反例实验和后续研究。 | 保留原文、来源分类和所需深度；阅读分析不授权实验、训练或数据访问。 |
| [study 原文件](C:/Users/win11-pc/.codex/skills/study.skill/SKILL.md) · [候选稿](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/proposed/study/SKILL.md) | 修复入口 BOM/乱码，精简为任务路由；短课无需课程创建、无关旧状态迁移或查看器。 | 保留正式课程路线、研究范围及 Module 00 的确认规则；不改状态脚本、格式、RPG 设置或通知权限。 |
| [paper-full-review 原文件](C:/Users/win11-pc/.codex/skills/paper-reading-zh/references/full-review.md) · [候选稿](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/proposed/paper-reading-zh/references/full-review.md) | 只修改入口范围说明，保留原有 12 节内容作为按请求采用的模板。 | 第 10–12 节改为用户要求研究延展时展开，避免局部分析自动产生新研究方向。 |
| [remove-ai-slops 原文件](C:/Users/win11-pc/.codex/plugins/cache/sisyphuslabs/omo/4.13.0/skills/remove-ai-slops/SKILL.md) · [候选稿](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/proposed/remove-ai-slops/SKILL.md) | 仅将 786 字符的触发描述缩为 97；其正文与清理流程原样保留。 | 不扩大清理授权，也未解除其现有执行门槛；本次没有重写其深层工作流。 |

## 保留的边界

- Canonical V1 的架构、损失、训练协议、参数界限、数据切分和评判标准仍须明确指令才能改变。不得为提高分数而改模型或协议。
- 只读、不训练、不读取新 spike targets 等任务约束继续有效。验证不扩大数据访问权限，也不授权额外实验。
- 区分代码事实、实验结果、模型内部推断及生物学结论；证据不足标为 UNVERIFIED。工程审阅继续区分缺陷与缺失证据；本项目将缺失证据统一报告为 UNVERIFIED（包括技能内部的 INCONCLUSIVE）。
- 研究解释、计划、后续实验由用户决定。任务完成即停止，不因技能流程产生额外基线、消融、重训或科研主张。
- 发布、外部消息、安装、破坏性操作，以及用户/项目明确要求的操作前批准和独立审阅不被本稿豁免。规划技能显式调用的批准环节保留。
- 保留并发修改，不覆盖他人工作。没有修改模型代码、实验数据、检查点、工具脚本、权限配置或模型设置。

## 需要明确接受的流程变化

本稿不是纯字数压缩。以下插件默认要求被有意放宽：

1. 普通实现/审阅不再固定需要五个审阅代理；普通 UI 不再固定需要两个代理。
2. 调试不再强制固定假设数、失败轮数或指定专业工具。
3. 局部改动不再自动要求全库阅读、完整规划、全项目测试或重建工具链。

相应的完成证据仍按请求、变更风险和项目已定验收条件确定。用户或项目明确要求的检查不会因为这些简化而被取消。

## 范围与生效限制

- 全局 C:/Users/win11-pc/.codex/AGENTS.md 的检索纪律已符合按需读取方向，本次不修改。
- scintillating-grid-snn-research 在 config.toml 中明确禁用，保持原状。
- study 当前未出现在本会话可用技能列表，入口有 BOM 和乱码。修复后可能在后续扫描中恢复可发现性；没有修改启用配置，实际加载情况需在新会话核验。
- 官方系统技能和其他专用插件未批量重写。语言、浏览器、课程等深层参考按入口的新范围规则使用，未声称已逐篇审计所有参考。
- 修改对象包括插件缓存中的实际文件，因此会影响使用这些插件的其他项目；插件升级可能覆盖此次修改。此方案不额外创建覆盖插件或改变安装配置。
- 当前会话已经加载的旧指令不会因候选稿自动改变。写回后应在新会话核验加载和行为。

## 检查记录

候选稿总大小从 **179,223 字节降至 58,345 字节，约减少 67%**。这是选中文件的磁盘字节比较，不是实测上下文 token、成本或速度。项目 AGENTS.md 本身略有增加，用于明确完成和实验边界。

- 10 个技能中，9 个通过官方 quick_validate.py 的静态检查。
- Ponytail 保留原有 argument-hint 字段；官方校验器因不支持该字段返回失败。已复测原文件，返回相同错误；没有删除字段来制造通过，其他 YAML 内容已能解析。
- 候选稿中所有 Markdown 本地引用均存在，技能名称和非目标元数据保留。
- 写回前：13 个源文件均与原始快照的 SHA-256 一致。写回后：13 个文件均与批准稿一致，原始备份完整保留。
- 上述为静态结构与内容检查，实际新会话行为尚未验证；没有运行模型训练、科研实验或项目测试。

独立审阅另按七个假设请求检查决策路径：只读冻结模型解释、现有 conda 项目的局部修复、UI 克隆、显式只规划、五分钟短课、带过期语义的最简实现，以及只审阅发现缺陷。发现的三处歧义已修正：克隆入口显式路由到 visual-qa、克隆需独立视觉与代码两条证据、项目缺证据标签统一为 UNVERIFIED。独立审阅者重新读取相关四处候选文本后确认这三项均已解决；此结论仅覆盖文本决策和指定修订，不是实际运行或安装验证。

[完整逐行差异](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/changes.diff) · [源文件清单与哈希](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/manifest.json) · [静态检查记录](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/validation.json)

已按本清单完成写回，无并发冲突。写回记录见 [written.json](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/written.json)；批准时的原始清单见 [approved-manifest.json](D:/PythonProject/retina_rf_SNN/work/instruction-optimization-2026-09-13/approved-manifest.json)。
