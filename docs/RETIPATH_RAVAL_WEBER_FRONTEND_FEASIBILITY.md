# Raval 2026 Weber-domain local-adaptation front-end feasibility audit

日期：2026-09-16。仅审计外部定义、输入单位、静态数学性质和现有源码。未加载或运行任何 RetiPath checkpoint，未训练、读取 natural movie/spike targets、修改 preprocessing/模型，未运行 FAST SCREEN。

**Verdict：WEBER_EARLY_FRONTEND_NOT_FEASIBLE（在本轮“不猜 absolute light level、不补模型参数”的约束下）。**

Figure 5 确有独立生理动机的简化 local-adaptation 分析，但可核实的定义仍是 **CALIBRATION_DEPENDENT**，不是一个已经公开确定、仅以 Weber contrast 为输入的动态前端。其 half-desensitizing background 是 **2000 R*/cone/s**。改写坐标后，该量与真实背景光强的比值仍存在；“采用 Weber curve”不等于“只需 Weber contrast”。此外，当前公开文本没有完整给出 Figure 5 的 gain→response 算法或时间更新方程，本次未找到可确认归属于该分析的作者代码版本。

本结论不否定 early local adaptation 的生理可能性，也不重新解释既有 prediction/F2 结果；它阻止的是把缺失定义自行补全后，称作零自由度的 Raval 前端。

## 1. 固定来源与证据层级

主来源：Raval V, Oaks-Leaf R, Chen Q, Rieke F. *Origin and functional impact of early nonlinearities in primate retina*。bioRxiv **v1，2026-03-23**，DOI [10.64898/2026.03.19.713068](https://doi.org/10.64898/2026.03.19.713068)。[bioRxiv 版本 API](https://api.biorxiv.org/details/biorxiv/10.64898/2026.03.19.713068) 本次返回 version 1、`published=NA`；按 preprint 解释。

核对了 [bioRxiv 原始 JATS](https://www.biorxiv.org/content/early/2026/03/23/2026.03.19.713068.source.xml)、[PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13041967/)、Figure 5 原图/图注、Methods “Modeling”和 “Linear equivalent stimuli”，以及一页 Supplementary Figures。补充文件仅提供 S1/S2 图示，未补充 Figure 5 的模型参数或算法。

| 证据 | 核实内容 | 不能据此声称 |
|---|---|---|
| Figure 5 正文/图注 | flashed natural images；先局部非线性再按 HC RF 加权；与先整合再非线性比较；D 展示 pixel distribution 与估计 cone-response distribution 的不同 | 完整的时间状态方程已经公布；输出必然没有 DC shift |
| Methods “Modeling” | Figure 5 用只有 gain term 的简化前端，gain 来自 sensitivity–mean-light Weber curve；half-desensitizing background 2000 R*/cone/s | 输入仅需 dimensionless Weber；可不指定真实背景 |
| 引用文献 | Angueyra & Rieke 2013、Cao et al. 2014 支持背景依赖 sensitivity curve | sensitivity curve 唯一决定任意有限 contrast/时间波形的 response |
| Figure 6 / 对应正文 | natural-movie 分析含固定 transduction model 和另一个可选、增减量不同 gain 的 synaptic stage | Figure 6 是同一个 Figure 5 简化 gain-only 前端；或可以搬用该 synaptic 参数来补 Figure 5 |
| 作者代码 | 论文没有给出 Figure 5 专用 repository/commit；本次检索未核实到对应实现 | 作者代码不存在；或 ISETBio、Chen 仓库就是 Figure 5 简化算法 |

Figure 5 正文以 phototransduction/adaptation 描述动机，而 Methods 特别说明此图使用 **simpler gain-only** 近似。本报告据该具体说明限定审计对象；不将 Chen 完整方程、其时间常数或参数移入简化模型。Figure 6 正文和 Methods 对 synaptic nonlinearity 的输入描述亦不能用来填补 Figure 5 的缺项。

代码检索范围：论文链接、精确题名/作者检索、Rieke-Lab 和共同作者 chrischen2 的公开仓库目录。一个表面相关的 `chrischen2/spatialIntegration`，HEAD `c8cdb90d547792170ed3976fe5441b57c5d6e849`，README 描述 mesopic Off-transient alpha RGC homogeneity 模型，**不能归属为此文 Figure 5**，因此未采用。`weber-package` 是以 Alison Weber 命名的 Symphony 包，也不是本任务的 Weber-law 模型证据。

**Figure 5 作者实现 commit/hash：UNVERIFIED。** 检索没有结果不等于证明未公开；没有确证时不任选相关仓库补位。

本次外部原文快照 SHA256：

| 来源文件 | SHA256 |
|---|---|
| bioRxiv v1 JATS | `b92235650699314586e0b49708d3928a2ed259b6f4fcf897c077064599cc4892` |
| Europe PMC v1 XML | `d0e1e000bacc011171f82e6b921b30836fe38f37a6872feb2e6a05f3008ed0a1` |
| `NIHPP2026.03.19.713068v1-supplement-1.pdf` | `de57ad2dbe815772f7a36d0c666abea6c3fffa1faeb565bb923b6ef16824175a` |

下载仅用于查阅外部文献，保存在系统临时目录；项目只新增本报告。

## 2. 精确到哪一步：gain curve 已知，完整 forward 未知

### 可从引用原文核实的关系

Angueyra & Rieke 2013 Eq. 1 给出：

\[
\frac{S(I_a)}{S_D}=\frac{1}{1+I_a/I_{1/2}}.
\tag{1}
\]

\(I_a\) 为 adaptation/background illumination，单位 R*/cone/s；\(S_D\) 为暗处 sensitivity；\(S(I_a)\) 是指定背景下小信号 sensitivity。原文的 gain 测量单位为 pA/R*，不是完整持续刺激的电流输出。Raval Figure 5 Methods 指定 \(I_{1/2}=2000\ \mathrm{R^*/cone/s}\)，并引用这类关系。[Angueyra & Rieke 2013 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3815624/)、[Raval v1 Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC13041967/)

来源数值不能混用：2013 原文的 4500 R*/s 后有 calibration correction；[Angueyra et al. 2022 Figure 4J](https://pmc.ncbi.nlm.nih.gov/articles/PMC8883858/) 明确列 corrected 2250 R*/s，以及按 collecting area 换算的 Cao 数值。**本轮既不重新拟合，也不把 Raval 明写的 2000 擅自换成 2250、3330 或其他值。** 引用原文用于核对曲线语义，而非挑选更有利参数。

### 换成 Weber 坐标不会消除 operating light level

令 \(c=(I-I_b)/I_b\)，其中 \(I_b>0\) 是真实参考背景；令 \(\bar c=(I_a-I_b)/I_b\) 描述局部适应光强相对该背景的位置。由式 (1) 直接得到：

\[
\frac{S(I_a)}{S(I_b)}
=\frac{1+\rho}{1+\rho(1+\bar c)}
=\frac{1}{1+\lambda\bar c},
\quad
\rho=\frac{I_b}{I_{1/2}},\quad
\lambda=\frac{\rho}{1+\rho}.
\tag{2}
\]

这是本项目的**单位/代数推导**，不是新增模型。即使把 background gain 归一化掉，仍留下未知 \(\rho\) 或 \(\lambda\)。相同 Weber trajectory 在不同 absolute backgrounds 下对应不同曲率，不能由现有 contrast tensor 唯一确定。

不能令 \(I_b=I_{1/2}\) 来消除未知量，也不能未经标定宣称 \(I_b\gg I_{1/2}\) 从而取 \(\lambda=1\)。后者还会在理想黑像素 \(c=-1\) 附近改变行为。把 denominator 的常数任意设为 1 同样只是选取了未证实的工作点。

### 完整算法的已知/未知项

| 项目 | Figure 5 可核实定义 | 状态 |
|---|---|---|
| 输入 | 空间变化的 calibrated light/image；图中用 normalized pixel contrast 展示；gain 依赖 mean light intensity | CALIBRATION_DEPENDENT |
| Locality | 各局部输入经过 adaptation-related transform 后，按 HC 的 RF 汇聚 | 已核实操作次序 |
| Gain law | 式 (1) 对应的 Weber sensitivity curve；2000 R*/cone/s half-background | 可核实来源关系；不是已完整公布的 forward |
| \(I_a\) 如何由当前像素/时序产生 | 是局部 flash level、时间平均、因果滤波值，或其他估计，未明确给出算法 | UNVERIFIED |
| Dynamic state、初态 | 简化模型未给出状态更新方程 | UNVERIFIED；不擅自增加 lowpass state |
| 时间常数 | Figure 5 简化模型未公布可直接采用的 τ | UNVERIFIED；不能借用 Chen 或其他图的 τ |
| Gain→response | 没有完整说明乘在哪个量上、是否积分 sensitivity、怎样减 baseline/归一化 | UNVERIFIED |
| Rectification / clipping | 未描述简化前端需要这些操作 | 不能自行补入 |
| Saturation / division | sensitivity curve 有 divisive attenuation；完整 response saturation 不能由此唯一推出 | 只有 gain attenuation 可确认 |
| 输出 | Figure 5 称 estimated cone responses；经过 RF 线性求和，并与 HC response 比较 | 不能从 normalized 图轴直接宣布输出单位为 mV、pA 或 release |
| 下游空间权重 | Gaussian RF，根据每个 HC 实测 area-summation 拟合 | 拟合的 downstream readout，不是 fixed early front-end 参数 |

特别地，固定的 \(S(I_b)\) 乘以 contrast 是线性映射，本身不会产生 F2；要表达局部非线性，gain 必须随局部输入/适应水平变化。**曲线的存在不指定该变化的时间轨迹。** 本轮不能将 Figure 5 flashed-image 的结果直接升级为一个已定义的 4 Hz 动态算子。

## 3. 参数来源与新增自由度

下表列出可识别的参数及完成接口所需但未确定的量。由于作者完整算法未核实，不能宣称已取得一个可执行模型的完整参数表。

| 量 | 来源类别 | 本轮结论 |
|---|---|---|
| \(I_{1/2}=2000\) R*/cone/s | **fixed from literature**：Raval 采用并给出文献依据 | 可固定；不是 dimensionless threshold |
| Weber gain exponent = 1 | **fixed from literature**：所引用式 (1) 的函数形式 | 可作 sensitivity-law 定义；不增加 exponent 拟合 |
| \(I_b\)、局部 \(I_a\) | 实验刺激/标定量，**不是 fitted per-cell/per-dataset 模型参数** | 当前数据缺对应 absolute level；不得改成 RGC NLL 拟合量 |
| \(S_D\) 或额外 output gain | Figure 5 完整实现未核实 | 是否 fitted in Raval / per-cell：UNVERIFIED；自行引入则属 **arbitrary implementation choice** |
| \(\rho\) 或 \(\lambda\) | 由光强与文献半脱敏量导出 | 缺标定时至少留下一个未知无量纲量；分别处理 L/M 时可能各有一个，未必相同 |
| Temporal τ、初态、局部均值估计器 | 无可核实的 Figure 5 参数定义 | 自行增加均属 **arbitrary implementation choice**，不是 published fixed 参数 |
| Baseline subtraction / 输出单位换算 | 无完整接口定义 | 需来源确认；不能当作无影响的编码细节 |
| Gaussian HC RF 权重/尺度 | **fitted per-cell in Raval**，由实测 size tuning 得到 | 属下游模型；不移植至 RetiPath，不重新拟合 |
| Figure 6 的 increment/decrement gain 差异 | **fitted in Raval** 的另一 synaptic-stage 参数 | 不属于 Figure 5 gain-only 前端，不借来补参数或设计第二个 alpha |

**0 trainable parameters 的判断：** 如果完整算子、absolute background、输出规范都已由外部确定，冻结后数学上可以是 0；然而当前缺这些固定值，故不能把“设置 requires_grad=False”当作“参数已有生理来源”。至少一个背景比值未知，另有算法定义缺项；它们需要外部证据，**不是本报告建议新增的可学习参数**。动态版本究竟需要多少参数也无法从原文确定。本轮不提出用当前 RGC NLL 学任何上述量。

保持 current BC alpha 不变与固定早期前端在结构上并不矛盾，但该组合不是 Raval Figure 5 已验证的架构，也不解决输入标定问题。

## 4. 当前 17×17 pooled L+M 输入

现有 [输入源码](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:164) 为 gamma-corrected RGB→相对 L/M→L+M→3×3 pooling→以 blank mean 定义 Weber。绝对标定缺口沿用已完成的[cone feasibility audit](D:/PythonProject/retina_rf_SNN/docs/RETIPATH_CONE_PHOTOTRANSDUCTION_FEASIBILITY.md)，本轮未重新解码 movie 或接触 target。

| 候选输入解释 | 判断 |
|---|---|
| A. 直接作用于 L+M Weber | **不能作为已复现的 cone-local Raval front end**：缺背景比值；L/M 相加后的非线性也不等价于分别计算。若未来另行定义，只能是 effective approximation |
| B. L/M 分别处理后合并 | 对 photoreceptor-local 解释更合适；但分别知道 L/M Weber 仍不提供 \(I_{b,L}/I_{1/2}\)、\(I_{b,M}/I_{1/2}\)。不能假定两者相同 |
| C. 3×3 pooling 的影响 | pooling 已丢失区块内 contrast distribution，不能恢复先非线性再 pooling 的结果；不等于所有空间信息都消失，17×17 之间仍有 variation |

令 \(P\) 是固定线性 pooling，\(f\) 为非线性，一般 \(P[f(c)]\ne f(P[c])\)。例如块内等量 \(+a,-a\) 可有 \(P[c]=0\)，但 \(P[f(c)]\) 不为零。这正是本任务关心的非线性空间整合，不能将 pooling 顺序视为等价。

若要求 cone-local 含义，非线性应先于 L/M 合并及 3×3 空间 pooling；若在当前 17×17 样点执行，须称 **effective local adaptation approximation**。仅做这种逐样点近似不必有 cone mosaic 或绝对 cone density；但真实 cone 数量、谱型 mosaic 与 fine-scale spatial integration 的解释需要额外采样依据。Figure 5 的局部图像计算不能自动赋予本项目格点真实 cone 身份。

本轮不改变任何处理顺序、格点、support 或合并权重。

## 5. 与 raw-input PReLU 的数学比较

### 已有 PReLU 的一阶均值偏移

\[
\phi_\alpha(c)=\frac{1+\alpha}{2}c+\frac{1-\alpha}{2}|c|.
\]

对零均值对称分布 \(c=a u\)，

\[
\mathbb E[\phi_\alpha(au)]
=\frac{1-\alpha}{2}a\,\mathbb E|u|.
\tag{3}
\]

除 \(\alpha=1\) 或零刺激外，DC shift 对正幅度 \(a\) 为 **O(a)**；这里“一阶”指随刺激幅度的阶数。左右导数不同，所以在零点没有一个共同线性切线。该 even term 可使成对反相空间条纹相加后仍有 DC/F2。

### 平滑 gain 能改善什么，不能保证什么

完整 Raval forward 未核实，不能为它直接给出精确 DC/F2。为检查“归一化后是否无需 light level、是否必然无 DC”两项说法，仅作以下**条件性代数分析，不作为作者方程、候选实现或 fixture**：若选择瞬时局部 \(I_a=I_b(1+c)\)，并以式 (2) 的 normalized gain 乘 contrast，则

\[
f_\lambda(c)=\frac{c}{1+\lambda c},\qquad
0<\lambda=\frac{I_b}{I_b+I_{1/2}}<1.
\tag{4，条件性重写}
\]

式 (4) 不是本轮声称核实的 Raval 程序；sensitivity curve 也不唯一要求这一乘法，例如把 sensitivity 解释为导数再积分会得到不同 response。它只用于展示即使采用最直接的乘法解释，未知光强和 DC 仍不会凭空消失。

在 \(c\ge-1\)、\(|\lambda c|<1\) 的小信号区：

\[
f_\lambda(c)=c-\lambda c^2+\lambda^2c^3+O(c^4),\quad
f_\lambda(0)=0,\quad f'_\lambda(0)=1.
\]

它可以有唯一、平滑的小信号线性极限，不具有 PReLU 的 O(a) kink-induced DC；但是

\[
\frac{f_\lambda(c)+f_\lambda(-c)}{2}
=-\frac{\lambda c^2}{1-\lambda^2c^2}.
\tag{5}
\]

因此有限幅度的对称输入仍可产生非零 even response。对 \(c=a\cos\omega t\)，最低阶项为

\[
f_\lambda(c)=a\cos\omega t-
\frac{\lambda a^2}{2}[1+\cos(2\omega t)]+O(a^3).
\tag{6}
\]

即 DC 和 F2 均可从曲率产生，起始为 O(a²)。电流/电压的生理符号转换可能翻转其符号，但不改变其是否存在。这里没有指定 \(\lambda\)、运行数值 forward 或预测当前 H1 的 F2 大小。

| 问题 | 可以支持的结论 |
|---|---|
| 零均值对称输入是否必有一阶 DC？ | 平滑、固定 operating point 的 baseline-subtracted response 不必有 PReLU 式 O(a) DC；完整 Raval 算法缺失，不能无条件保证其具体阶数 |
| 是否完全无 DC/distribution shift？ | **不能保证**；平滑非线性仍可有 O(a²) DC，Figure 5D 本身就强调分布改变甚至均值符号改变 |
| 小 contrast 是否近 identity？ | 一般可近线性；identity 还需要明确且固定的 output normalization，不能从文献自动取得 |
| 是否 contrast-/history-dependent？ | sensitivity 随局部 mean light level 改变；不能把这直接称 temporal contrast adaptation。Figure 5 简化模型未给 history-state 方程，不能声称已具有 history-dependent dynamics |
| 能否产生 F2？ | 非零偶次局部曲率可以；常数 gain 不可以。要量化 4 Hz 需已确定的时间算子 |
| F2 的来源？ | 条件性式 (4) 来自 local light-dependent divisive gain 的静态近似/曲率，不需要 hard rectification；不能据此判定真实 assay 中全部 F2 来自 adaptation |

Raval Supplementary Figure S1 还报告该实验中 HC 对 temporal contrast 的适应很少；不要把 mean-light adaptation 和 contrast-variance adaptation 混称一种已验证机制。

这些数学区别不证明它会避免上一版的 prediction penalty。未知的 operating level、baseline、有限 contrast distribution shift 及与 BC 的组合仍然存在，不能根据平滑性预先宣称 NLL 更好。

## 6. 与现有参数的可识别性

仅检查源码和代数，不运行参数或 checkpoint。当前 [H1](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/h1_pathway.py:71) 为 graph→delay→lowpass→subtraction；[BC feature bank](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/bipolar_subunits.py:162) 对输入线性；[canonical branch](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/retipath_canonical_gain.py:127) 含 common-branch PReLU、AC 线性滤波/固定于刺激的 gate，以及 G_E/G_I；[RGC adaptation](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/retipath_spatial_ei.py:151) 位于 membrane 后。

| 关联量 | 分类 | 原因/边界 |
|---|---|---|
| 绝对 input scale 与 \(I_{1/2}\) | **exact ratio ambiguity** | 若两者同乘正数，式 (1) 的比值不变。固定文献 \(I_{1/2}\) 后不再有两个可学习量的 gauge，但缺 \(I_b\) 仍使有效曲率不确定；不能用 Weber 数据恢复它 |
| 新增自由 overall front-end output gain 与 G_E/G_I | **exact gauge（若引入该 gain）** | 下述正齐次性证明；固定 front end 不新增此可学习 gauge |
| H1 τ/delay | **practical identifiability risk**（若自加可学习适应时程） | 早期时程与既有 H1 filtering 可在有限刺激带宽互相补偿；局部非线性和 H1 graph/subtraction 不一般交换，未证明全局精确 gauge |
| H1 amplitude | 功能相似/实际混淆风险，**不是已证明 gauge** | 改变 surround subtraction 与改变局部适应不是同一算子，可在有限空间输入上近似补偿 |
| BC sustained/transient temporal basis | **practical identifiability risk** | 新增时程可能被后续 basis 的组合补偿；无证据说明所有输入下精确等价 |
| BC alpha | **practical identifiability risk** | 都能改变增减量不对称，但位置在空间汇聚前/后不同，平滑曲率与 PReLU kink 也不同；不能等同或共享同一个 alpha 来消除问题 |
| G_E/G_I 与未知 curvature/background ratio | 小信号或有限数据下的 practical risk | 强度可以补偿一部分平均 gain；不能一般抵消 contrast-dependent 曲率 |
| RGC adaptation | 功能相似，若新增早期动态自由度则有 practical risk | 前端局部适应与 membrane 后减性低通有不同位置/状态，不能仅因都叫 adaptation 就断言精确退化 |

### 新 overall gain 的精确 gauge 证明

假设在某个前端输出后、H1 前另设可学习正标量 \(a_f\)，其他输入/参数固定，沿用当前零输入动态初态和 frozen RMS。H1、BC feature filtering、AC filtering 对输入线性；固定 \(\alpha\) 的 PReLU 满足 \(\phi_\alpha(sc)=s\phi_\alpha(c)\)（\(s>0\)）。因此所有 E/I relative drives 同比例缩放。变换

\[
a_f\to s a_f,\qquad G_E\to G_E/s,\qquad G_I\to G_I/s
\]

保持 effective drives、gE/gI、membrane 及最终 logit 的数学值不变。固定 RMS 不参与重估。该证明针对 **overall output gain**，不针对改变 nonlinear shape 的 \(\lambda\)。这是源码支持的代数结论，未运行数值 replay。

因此，若为补定义而开放 background ratio、output gain 和新的 adaptation τ，应标记 **HIGH_IDENTIFIABILITY_RISK**：其中 output gain 会引入精确 gauge，其他自由度有实际混淆风险。当前没有必要通过学习这些量救回前端；本轮不建议这种做法。

## 7. Fixture gate、verdict 与停止条件

本轮 **不运行 synthetic fixture**。允许 fixture 的条件是模型定义和 published parameters 已足够确定；当前 complete forward、局部适应光强/时程、输出规范和本数据背景比值均未充分确定。选择任意 \(\lambda\) 跑 sinusoid 只能验证自行定义的函数，不能验证 Raval 作者实现。

最终门槛：

| FAST SCREEN 前提 | 结果 |
|---|---|
| 无需未标定 absolute light level | 不满足；式 (2) 保留未知比值 |
| 参数来源和完整 forward 清楚 | 部分满足；gain law/2000 已知，完整映射及时间定义 UNVERIFIED |
| 0 或极少新增 trainables，且无需为 prediction 调参 | 外部定义补齐后可讨论；当前不能以任意固定数代替来源 |
| 当前 L+M pooled 输入可直接承担原语义 | 不满足；只能另行声明 effective approximation |

**WEBER_EARLY_FRONTEND_NOT_FEASIBLE** 是针对当前冻结输入和禁止补参数的直接接入判断。需要解决的外部缺项只有：Figure 5 完整作者算法及输出定义，以及该 gain law 的本实验 background ratio/absolute calibration。若作者另有明确、可验证的严格 contrast-only 极限，则须先取得该定义；本报告没有自行替作者提出此极限。保留原 BC alpha，不启动第二套 early+late PReLU、FAST SCREEN 或新架构。

## 最后八问

1. **精确方程是什么？** 可核实的是 \(S(I_a)/S_D=[1+I_a/(2000\ \mathrm{R^*/cone/s})]^{-1}\) 这一 sensitivity law。Figure 5 的完整 gain→response 与动态更新尚为 UNVERIFIED；本报告的式 (2)、(4)–(6) 是明确标记的代数/条件性分析，不冒充作者 forward。
2. **是否真正只需要 Weber contrast？** **否。CALIBRATION_DEPENDENT。** 改写后仍需 \(I_b/I_{1/2}\)。
3. **能否 0 trainable parameters 接入？** **当前不能形成有完整来源的零参数接入定义。** 完整外部标定和算法冻结后可以为 0；缺失量不能通过不训练而变成已知。
4. **L/M 分开还是 L+M 后？** cone-local 解释应分别处理、再合并，并在空间 pooling 前计算。当前 17×17 L+M 后处理只能称 effective local adaptation approximation。
5. **是否避免必然 DC shift？** **不能保证没有 DC。** 平滑非线性可避免 PReLU 式 O(a) kink-induced shift，但仍可有 O(a²) DC；Figure 5 也没有声称均值不变。
6. **能否原则上产生 outer-retina F2？** 局部 gain 随输入变化而产生偶次曲率时 **可以**；固定 gain 不行。尚不能由未定的时间实现预测当前模型的 4 Hz F2。
7. **最大 identifiability 风险？** 未知背景比值决定非线性强度；若再增加 overall output gain，会与 G_E/G_I 形成精确 gauge。额外 τ/不对称自由度还可能与 H1、BC、RGC 的既有参数混淆。
8. **是否有资格进入 4-cell FAST SCREEN？** **没有。** 当前 verdict 为 **WEBER_EARLY_FRONTEND_NOT_FEASIBLE**。本轮只新增本报告，到此停止。
