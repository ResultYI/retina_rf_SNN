# Canonical V1：限时生理学文献对照

日期：2026-08-31。范围：只读对照与新建本备忘录；未修改模型、未训练、未运行新实验。本对照独立于 Architecture Conformance Audit；文献相容性不能改变实现不合规的判定。

## 证据范围

本轮实际检索并核对以下四篇原始研究的摘要和检索器返回的正文相关段落。PMC 直接打开触发验证码，备用获取未成功；因此没有声称已完整逐页审阅全部正文或补充材料。Verweij 核对到方法、讨论及实验局限；Jacoby 核对到方法、结果及连接证据；Crook 核对到药理方法、图 1/2 图注、结果及讨论；Turner 核对到实验组织来源、讨论及整流结论。以下属于限定文献对照，不是系统综述。其中 Jacoby & Marshak 是电镜解剖研究，明确与生理记录证据区分。

| 当前设计命题 | 原始实验与适用条件 | 支持范围与限制 |
|---|---|---|
| H1 的周边作用位于 BC 上游 | Verweij, Hornstein & Schnapf (2003), *Surround Antagonism in Macaque Cone Photoreceptors*。M. fascicularis / M. mulatta 离体视网膜，cone 全细胞记录、中心/周边光刺激；周边照明产生拮抗反应，空间及光谱结果支持 horizontal-cell 来源。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6741006/) | 支持外网膜反馈先改变 cone 输出这一位置。不能证明当前 G→低通/延迟→Gᵀ→减法就是真实反馈的定量等价物，也未直接识别本模型 H1 参数。论文仅约 20% 记录 cone 显示 surround，且去极化和提高胞内 chloride 有助观察，不能把该比例或幅度直接移植到模型。 |
| 同一 BC 来源可同时进入 direct-RGC 与 AC 支路 | Jacoby & Marshak (2000), *Synaptic Connections of DB3 Diffuse Bipolar Cell Axons in Macaque Retina*。单只 light-adapted M. mulatta，约 4 mm 偏心度，calbindin 标记及连续电镜；DB3 ribbon 输出连接 AC 和 ganglion dendrite，多数突触后 AC 中约 47% 对 DB3 形成反馈突触。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3341735/) | 支持共享 BC 来源后分支这一连接原则。因此不能仅因 direct/AC 来自同一 BC 就判其生理错误。但同一生物 BC 分叉不等于所有 BC 类型共用一套线性权重；论文还显示 AC→BC 反馈。纯 feedforward AC 是简化，不能称完整内网膜微回路。此项是解剖证据，不是功能或可辨识性验证。 |
| AC 统一负向作用代表内网膜抑制 | Crook, Packer & Dacey (2014), *A synaptic signature for ON- and OFF-center parasol ganglion cells of the primate retina*。macaque 离体 ON/OFF parasol，明视觉均值下 spots/annuli 与正弦对比度刺激，突触电流及 receptor-antagonist 记录。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4503220/) | 支持显式抑制成分，但该条件下主要为 rectified glycinergic crossover inhibition，GABAergic feedforward 较小，且抑制随对比度改变。阻断 inhibition 后 center-surround 与 Y-type 性质基本保留。单个负 gain 不足以认定模型涵盖真实 AC 类型、极性来源和药理作用；也不能把这里的 parasol 结果推广到全部 RGC。 |
| 完全线性 BC 编码对自然刺激与未见 probe 足够 | Turner & Rieke (2016), *Synaptic rectification controls nonlinear spatial integration of natural visual inputs*。M. nemestrina / M. mulatta / M. fascicularis 离体 retina，ON/OFF parasol，自然图像及人工 grating。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4917290/) | OFF parasol 对自然输入的非线性空间整合来自强整流兴奋性 subunits；ON 在所测自然刺激下更接近线性，但两者对特制 grating 均可非线性。结果限制统一线性 BC 表述的普适性，尤其涉及从自然刺激外推至人工 probe。论文没有测试本项目 encoder；不能据此断言本模型必然失败，也不能认定在 RGC 汇总后添加非线性就已等价实现 subunit rectification。 |

## 对研究目标的限定判断

**基于上述证据的推断：**外网膜反馈、BC 分支、内网膜抑制的总体分层有生理依据；当前简化的定量充分性以及 pathway 名称对应真实细胞机制，均为 `UNVERIFIED`。生理合理的连接位置与可训练性，本身不能证明从真实 RGC spikes 识别了内部 H1/BC/AC 的计算。

文献没有验证当前模型的 H1-off、direct-BC-off、AC-off 是否等价于生物学干预。电镜中的 AC→BC 反馈和生理实验中的 crossover inhibition，意味着模型删除一个单向分支的结果必须标为 **model intervention**，不能直接等同药理阻断或细胞沉默。

对 Mach bands、Simultaneous Brightness Contrast、White’s illusion、Hermann grid，本轮四篇研究未直接证明该架构能解释其 retinal response signatures，更未证明主观亮度机制。相关结论为 `UNVERIFIED`。ON/OFF 与刺激条件依赖的整流结果说明，人工 probe 的外推不能仅由自然刺激预测良好或 center-surround 外形合理推出。

关于是否需要优化：这些文献足以指出 **BC 级非线性与类型差异、AC 极性/目标/反馈语义** 是当前简化的明确适用边界；尚不足以决定必须新增哪个模块，或声称新增模块即可获得可辨识性。是否改变冻结 Canonical V1 属于单独科学设计决策。本轮不实施优化、不选择下一实验。
