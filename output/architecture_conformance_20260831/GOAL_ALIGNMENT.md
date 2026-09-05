# 对既定研究目标的限定判断
日期：2026-08-31。基于本次独立源码／运行时审计与限时原始文献核查；没有优化模型或运行训练。

**判断：总体生理分层可作为研究假设，但当前实现不能称为已满足合同、已辨识真实回路、或已验证错觉应用。需要先解决合同层面的正确性问题；现有证据不足以决定新增哪一种生理模块。**

用户目标保持为：可训练且具有明确生理结构的模型，预测真实 macaque RGC spikes，支持可信的 in-silico circuit analysis；应用结论限定为 retinal response signatures 与 modeled pathway 必要性，不要求预测超过所有黑盒模型。

| 目标层 | 本次已经支持 | 本次不能支持 |
|---|---|---|
| 预测 | 生产 forward、梯度与优化器参数连接可执行 | 没有评估真实 spike 预测，也未比较任何 baseline，预测充分性 UNVERIFIED |
| 模型内干预语义 | 两 BC 视图共享参数；AC 无 stimulus bypass；三种干预精确遵守已测张量关系；RF Jacobian 与 forward 一致 | 空间支持、legacy 入口和结构无效参数仍不符合合同 |
| Pathway 可辨识性 | 已识别参数归属和真实 tensor dependency | 未进行 model-matched recovery；不能从梯度非零、NLL或架构命名推出可辨识性 |
| 生物机制对应 | cone/HC反馈上游、BC分支及内网膜抑制有原始证据支持 | 模型 H1/AC 操作未证明等价于真实细胞沉默或药理阻断 |
| 错觉应用 | 用户提出 response signatures + model interventions + diagnostic variants 的证据范围适当 | 未运行四类 probe；没有验证任何错觉结果，更不能声称模型复现了人类知觉 |

## 生理结构是否合理
外网膜 feedback 先改变 cone 输出与 macaque cone 记录相容，但这不能证明当前 G→delay/LP→Gᵀ→减法算子定量正确。[Verweij et al., 2003](https://pmc.ncbi.nlm.nih.gov/articles/PMC6741006/)

同一 BC 连接 AC 与 RGC 有 macaque DB3 电镜依据，因此共享 BC 来源后分支有解剖合理性。该证据不等同于“一套完全共享线性 encoder 足以覆盖所有细胞类型”；它也记录了 AC→BC 反馈，当前单向分支并不表示完整微回路。[Jacoby & Marshak, 2000](https://pmc.ncbi.nlm.nih.gov/articles/PMC3341735/)

ON/OFF parasol 的空间整合与兴奋性输入整流具有刺激和细胞类型依赖性；当前 BC 段对 stimulus 线性，RGC 汇总后的非线性不能自动等同于汇总前 subunit rectification。这是自然刺激拟合后外推人工 probe 时的明确适用边界。[Turner & Rieke, 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4917290/)

统一负向 AC 支路可以表达一种抑制计算，但 macaque parasol 原始实验还包含 crossover、整流和对比度依赖；因此该支路不能自动解释为所有真实 amacrine 作用。[Crook et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4503220/)

## 为达到目标是否需要优化
**合同层面需要正确性修正。** 当前实际支持超界、非圆盘 geometry 可进入、V1 边界接受 legacy，以及 self-only trainable connection 无功能影响，会削弱对 pathway 空间位置和参数含义的陈述。这些结论来自本次独立源码和反例，不是为了提高预测分数。本轮仅报告，未修正。

**生理层面尚不能决定必须增加模块。** BC subunit 非线性与 AC 的极性／反馈语义有文献依据，属于现有简化需要交代的局限；是否改动取决于既定机制声明与数据证据。仅为让 Mach bands、White 或 Hermann 的输出看起来正确而调整架构，不会建立生物机制证据。

**可辨识性是尚未解决的核心证据缺口。** 从实际源码可见，所有 stimulus pathway 先汇成 total_current，随后才进入同一个 RGC 输出链。由此可推断，真实 spike 拟合良好本身不能验证唯一的 pathway 分解；但本轮也没有证明整个模型必然不可辨识。model-matched synthetic 恢复是用户既定目标的一部分，其成功仍只支持相应模型假设内的恢复能力，不能单独确认真实动物的潜在回路。

## 错觉应用的解释边界
- Mach bands：单一带通响应峰谷不充分，因为也可能错误地在普通 step edge 产生峰谷。该限制来自理论与心理物理对照，不能直接定位 H1 或 AC。[Kingdom, 2014](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2014.00843/full)
- Simultaneous Brightness Contrast：麻醉猫特定远距动态诱导条件下的 RGC 轴突记录没有满足该研究的 brightness-correlate 判据；不是 macaque 的普遍否定。[Rossi & Paradiso, 1999](https://pmc.ncbi.nlm.nih.gov/articles/PMC6783067/)
- White：人类实验中共线与侧邻结构作用不同，不能由简单邻域黑白面积之和解释；这不直接定位视网膜回路。[Blakeslee et al., 2016](https://www.sciencedirect.com/science/article/pii/S0042698916300712)
- Hermann grid：几何变体限制经典中心—周边的充分解释；不等于所有 retina-only 模型均不可能。[Schiller & Carvey, 2005](https://journals.sagepub.com/doi/10.1068/p5447)

本次文献核查有限：Kingdom 已核查全文；部分其他论文只核对可公开获取的原文方法／结果／讨论片段，未完整逐页审阅，标为 PARTIAL_SOURCE_REVIEW。研究目标的生物可辨识性和四类应用结果均保持 UNVERIFIED。详见 physiology.md 和 illusion_literature.md。

