# Brightness illusions：retina-only 解释边界的限定文献对照

2026-08-31。范围：4 篇原始实验／理论论文；没有修改模型、运行训练、重放刺激或选择下一项实验。此附录是文献约束，不是当前 Canonical V1 的生理有效性判定。

结论：用户将应用表述为“与错觉相关的 retinal response signatures，并检验 modeled pathways 的必要性”是合适的证据边界。相同刺激下出现局部峰谷、中心响应差异或交叉点响应差异，不等于模型产生了人类的亮度知觉，也不能单凭效果方向正确识别真实 H1／BC／AC 机制。以下约束不支持为了复现错觉外观而直接扩大模型。

| 应用 | 原始研究、物种与证据层级 | 对 retina-only 模型能支持什么 | 主要限制 |
|---|---|---|---|
| Mach bands | Kingdom (2014)，原始一维多尺度滤波／response normalization 理论；对照既有人类心理物理数据，无新生理记录。 | 早期视觉滤波可以产生梯形亮度坡道端点的峰谷；作为输出 signature 有明确计算意义。 | 单一带通滤波器可在普通 step edge 也产生峰谷，因此“有峰谷”不是完整 Mach-band 解释。作者明确说明没有新增经验数据；不能把 normalization 计算直接定位到 macaque H1 或 AC。[原文全文](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2014.00843/full) |
| Simultaneous Brightness Contrast | Rossi & Paradiso (1999)，麻醉猫，optic tract 中 RGC 轴突、LGN 与 V1 的胞外记录；远距动态 brightness induction。 | RGC 对刺激／周边条件的响应可以作为模型预测研究；论文帮助区分该响应和与亮度知觉对应的信号。 | 在其刺激与判据下，RGC 轴突未显示符合 brightness-correlate 要求的响应；符合全部条件者出现在 V1。结果不能概括为“视网膜没有任何上下文计算”，也不是 macaque 对这四种错觉的直接证据。[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6783067/) |
| White’s illusion | Blakeslee, Padmanabhan & McCourt (2016)，人类 brightness matching；独立操纵与灰块共线的 bars 和侧邻 bars 的亮度。 | 能对照模型是否区分两组邻接结构的作用，而不只比较同一灰度块的响应方向。 | 0.5 cycles/degree 条件下，共线 bars 的 contrast 作用更强，侧邻 bars 也表现为 contrast；简单的邻接黑白面积总量不足以解释方向。该行为证据没有定位 H1、BC、AC，也不能证明所有 retina-only 非线性模型均不可能解释它。[原始论文](https://www.sciencedirect.com/science/article/pii/S0042698916300712) |
| Hermann grid | Schiller & Carvey (2005)，人类知觉演示及理论分析；检验格栅几何变体与经典 RGC 中心—周边解释。 | 交叉点与通道响应差可作为预先定义的模型统计量。 | 某些改变不应显著改变经典中心—周边差异，却削弱／消除知觉效果；作者提出皮层 simple-cell 替代解释。反驳针对该经典解释，不能写成排除了全部 retinal computation。标准格栅通过并不足以解释错觉。[原始论文](https://journals.sagepub.com/doi/10.1068/p5447) |

对研究目标的含义（证据推断，不是新实验决策）：

- 预测层：这些论文没有检验当前 macaque spike 模型的预测能力；这一点仍须由当前数据与既定验证结果决定。
- 机制层：H1-off／direct-BC-off／AC-off 若只在模型中实施，结论限定为该模型、参数与输入条件下的必要性。model-matched synthetic 成功可支持模型内恢复／干预语义，不能据此把真实数据中的潜变量归因为生物通路。
- 应用层：保留用户已提出的 diagnostic variants 是有文献依据的解释边界，不应将标准图形单一效应方向当作生理验证。比较前必须明确 stimulus geometry、视角／模型空间单位、响应时间窗和输出统计量；文献中的知觉亮度并不等同于 RGC rate。
- 关于优化：这 4 篇文献不足以判定 Canonical V1 必须新增某个回路、normalization 或皮层模块。多尺度归一化是 Mach-band 理论的候选计算，但没有给出其在本模型 H1／AC 中的生理归属证据；把它写成已证明必要的架构优化会越过证据。若既定目标保持 retina-only，不能为使 White／Hermann 看起来正确而加入未授权模块。

检索和核查边界：Kingdom 全文已读到模型、结果和 limitations；Rossi 原文的 abstract、introduction 与公开索引的 discussion 片段，以及 White 原文出版社 abstract／方法摘要已核查，但全文直接打开受到站点限制；Schiller 出版社 abstract 与作者论文公开索引的变体图注／正文片段已核查。后 3 篇未完成逐页全文审阅，标记 `PARTIAL_SOURCE_REVIEW`；未报告未经核实的样本量或精细定量结果。以上均为原始论文，未将综述或百科当作主证据。
