# Croner & Kaplan (1995)：正常状态 M/P center–surround 的 Level-2 可行性审计

日期：2026-09-15。结论：**FEASIBLE_PARTIAL**。这是外部生理协议与可观测量的可行性审计，不是模型验证结果；本轮未运行任何 checkpoint、assay、训练或 natural-movie target 分析，未修改模型、FOV 或 external-geometry interface。

## 1. 证据边界与论文实际测量

目标：[Croner LJ, Kaplan E. Vision Research 35:7–24, DOI 10.1016/0042-6989(94)E0066-T](https://doi.org/10.1016/0042-6989(94)E0066-T)。原论文摘要确认：在麻醉、麻痹 macaque 的 LGN 中记录 retinal afferent 的 S-potentials，研究中央约 80° 直径视野。测量对象是 RGC 输出的空间响应，不是 H1 膜电位，也不是 LGN relay-cell 输出。M/P 的 center、surround 尺度随偏心度增加；相邻位置 M center 通常约为 P 的两倍；population 平均 surround/center integrated gain 约 0.55。[原论文摘要](https://pubmed.ncbi.nlm.nih.gov/7839612/)

本轮未能完整读取原文 PDF 正文和所有图表：已定位作者上传的 18 页 PDF，但页面访问/提取受限。因此以下明确区分原文摘要、后续论文中的方法转述、以及公开的数字化/作者提供数据；**不声称已逐项核验原文 Table 1 或完成精确 protocol reconstruction**。

补充证据固定到 ISETBio commit **d1619948425056efa5435c91932bca5b3a30f4a7**：

- [digitizedData.m](https://github.com/isetbio/isetbio/blob/d1619948425056efa5435c91932bca5b3a30f4a7/ganglioncells/%2BRGCmodels/%2BCronerKaplan/digitizedData.m)：P center/surround 散点数字化数组。本轮直接读取数组，未运行其 MATLAB 模型或绘图函数。center 数组 75 点、surround 数组 88 点，数量不同，不能按行配成同一个 cell；函数返回的 Figure 1A/1B 标签与其他源码的 Figure 4 引用不一致，图号对应仍为 UNVERIFIED。
- [M center 数据](https://github.com/isetbio/isetbio/blob/d1619948425056efa5435c91932bca5b3a30f4a7/data/datafiles/rgc/croner_kaplan_parasol_rgc.mat)及[来源/单位说明](https://github.com/isetbio/isetbio/blob/d1619948425056efa5435c91932bca5b3a30f4a7/tutorials/data/underDevelopment/data_rgcEccData.m)：36 个 Figure 4 数字化点，原数组为 eccentricity degree 与 RF radius degree。本轮读取原数组；没有使用该 tutorial 后续的 mm 转换、回归、Dacey dendritic-field 合并或 human scaling。
- [individualMonkeyData.m](https://github.com/isetbio/isetbio/blob/d1619948425056efa5435c91932bca5b3a30f4a7/ganglioncells/%2BRGCmodels/%2BCronerKaplan/individualMonkeyData.m)：文件注明数据由 Lisa Croner 于 2024 年 6 月提供，包含 monkey/cell ID、temporal-equivalent eccentricity、integrated S/C ratio、C/S radius ratio。M 11 行、P 82 行，含缺失值；不是原论文全部实验样本量的证明，亦不包含逐 trial spike response 或这批 cell 的绝对半径。
- [constants.m](https://github.com/isetbio/isetbio/blob/d1619948425056efa5435c91932bca5b3a30f4a7/ganglioncells/%2BRGCmodels/%2BCronerKaplan/constants.m)：明确将合并 M/P 的 surround 趋势归于原文 Figure 4 caption。以下几何预算使用这一公开转录；没有重新拟合本项目数据，也没有将 ISETBio 模型输出作为生理数据。

### Stimulus / RF fitting contract

| 项目 | 可确认内容与未解决部分 |
|---|---|
| 刺激及响应 | Achromatic sinusoidal drifting gratings 的 spatial-frequency response，用 DoG 描述 center/surround。后续原始研究论文明确转述该 surround-strength 结果基于 4 Hz drifting gratings；这是方法转述证据，原文 Methods 的完整时间设置本轮未直接核实。 |
| 拟合量 | Gaussian characteristic radius 与 center/surround contrast sensitivity；空间频率曲线的响应调制幅度，不能替换成一次 flash 的平均 logit。 |
| 仍须核实 | 原文各系列的 contrast levels/contrast-gain 估计步骤、平均 luminance、完整 spatial-frequency sampling、aperture、适应/呈现时长、重复次数、F1/phase 处理、权重/误差模型、optical correction 及排除规则。当前不能填“合理默认值”。 |
| 精确复制状态 | UNVERIFIED；现有 external assay 的 rectangular flash builder 不是这个 grating protocol。不能借用 McMahon 的 2 Hz 或把这里改成单次 flash 后称 protocol-matched。 |

4 Hz 方法转述出处：[McCann, Hayhoe & Geisler (2011), Methods pp. 2, DOI 10.1167/11.10.19](https://pdfs.semanticscholar.org/3a12/e0f7baafa1ad1aabe0256eb65f6f681212ba.pdf)。DoG 定义亦可交叉核对 [Godat et al. (2022), Figures 10–11](https://doi.org/10.1371/journal.pone.0278261)。这些补充来源不替代缺失的原始实验设置。

统一数学定义，令空间距离 r、characteristic radius r_c/r_s 以 degree 表示：

$$D(r)=K_c e^{-(r/r_c)^2}-K_s e^{-(r/r_s)^2}.$$

r_c/r_s 是各 Gaussian 降至其峰值 1/e 的半径；不是有限 support 边界，不是半高宽，也不是 Gaussian 标准差（σ=r/√2）。积分强度为 I_c=πK_c r_c²、I_s=πK_s r_s²，因此

$$\rho_r=r_s/r_c,\qquad \rho_G=I_s/I_c=(K_s r_s^2)/(K_c r_c^2).$$

纯空间、同相位线性 DoG 的频域表达是 I_c exp[−(πr_c f)²]−I_s exp[−(πr_s f)²]；实际 temporal phase/F1 contract 必须另行对齐，不能预设模型空间与时间可分离。K 的响应单位还依赖是否使用 spikes/s、百分比 contrast 和 degree²，绝对 gain 比较需先统一单位。

## 2. 文献尺度、关系与个体/summary 区分

下表只做外部范围整理。0–5、5–10、10–15° 为本次几何审计分箱，不是原文原始分组；数字为公开数字化点的 min–max，**不是置信区间，也不是当前同偏心度 cell 的预测区间**。散点数字化误差没有提供，末位数字无生理精度含义。

| Eccentricity bin | M r_c (°), 点数 | P r_c (°), 点数 | P r_s (°), 点数 |
|---|---:|---:|---:|
| [0,5) | 0.052–0.108, 16 | 0.020–0.070, 18 | 0.081–4.118, 28 |
| [5,10) | 0.069–0.139, 8 | 0.027–0.075, 20 | 0.187–2.265, 26 |
| [10,15) | 0.121–0.195, 5 | 0.034–0.112, 13 | 0.155–1.011, 14 |

P surround 数组确实包含很大的拟合半径点，本轮未裁除、winsorize 或当作“典型值”。这些极端值的原始 fit uncertainty/quality 尚未核实，不能据其最大值决定唯一所需 FOV；也不能删除后声称全部 surround 可容纳。M 的逐 cell 绝对 surround–eccentricity 散点本轮未取得：这一缺口不能由 P 散点或两倍 center 关系补造。

公开转录的**合并 M/P population 趋势**为：

$$\widehat r_s(e)=\max(0.1,0.203e^{0.472})\ \mathrm{deg}.$$

在本 cohort 的 3.49–13° 数值范围内，趋势约 0.366–0.681°；**它不是每个 M/P cell 的真实半径，也不是分别估计的两条 class regression**。原始拟合 CI 本轮未取得。[constants.m，Figure 4 caption 转录](https://github.com/isetbio/isetbio/blob/d1619948425056efa5435c91932bca5b3a30f4a7/ganglioncells/%2BRGCmodels/%2BCronerKaplan/constants.m)

| 关系 | 能使用的证据 | 禁止的替换 |
|---|---|---|
| M/P center vs eccentricity | M/P 两组数字化散点；原文摘要的 matched-location M≈2P 为 population 关系。 | 不能给每个 M cell 固定为其邻近 P 的精确两倍；不能将远外周 regression 无约束外推到本 cohort。 |
| Surround vs eccentricity | P 个体数字化散点 + 合并 M/P 趋势；M 单独绝对半径曲线不完整。 | 不能把合并曲线写成每个 class 的精确条件均值。 |
| r_s/r_c | 作者提供数据在 e∈[3.49,13] 的 M 5 行约 5.88–17.54；P 37 行约 2.76–83.33。包括宽广散布及潜在弱约束 fits。后续文献常用 P 的约 6–7 倍 summary。 | 不能把均值 C/S 的倒数当作平均 S/C；6.7 不是每个 cell 的固定生理比率，也不应同时赋予 M/P。 |
| Integrated S/C gain | 原文 population 平均约 0.55；上述有 eccentricity 的局部原始子集中，M 有效 4 行 0.373–0.614，P 有效 36 行 0.121–0.854。缺失值不补零。公开转录 P 趋势为 0.466+0.007e。 | 不能以 0.55 作每 cell 接受阈值；不能把 K_s/K_c、积分比、annulus response ratio、SSI 或 G_I/G_E 混为一谈。 |

以上 ratio 范围是读取公开个体数组后的描述性计算，不是新的生理实验；数字化点与 2024 年作者提供的 ratio 数据也不能擅自逐行合并。外部样本有 monkey 聚类，不能把所有点当独立动物或把 cell×seed 当生物重复。

## 3. 当前 cohort、FOV 与逐 cell 检查

本地来源：

- [Schottdorf–Lee catalog](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_catalog.py:50)：cell identity、MC/PC class、eccentricity。
- [输入 coordinates](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:220)：原 4.6°/256 pixels，51-pixel crop，3×3 mean pooling。
- [冻结 registration 元数据](D:/PythonProject/retina_rf_SNN/output/evaluations/retipath_h1_center_surround_assay_20260915/stimulus_definition.json)：只读取 cell identity、degree coordinates 和 center_deg；没有读取 assay response 或 checkpoint tensor。

实际 **MC/parasol 9 cells，3.52–13°；PC/midget 13 cells，3.49–8.31°**。全 cohort 最小值是 **70#7 的 3.49°**，不是 3.52°。论文整体覆盖到约 40° 偏心度，公开 center 数字化数据 M 约 1.46–21.94°、P 约 0.16–35.05°，因此本 cohort 数值上落在直接观察范围内部，无需从远外周外推。ratio 原始 M 子集只有 3.62–25.38°，不能据这个小子集否定更完整 center 散点的覆盖，也不能为 69#7 补造一个 3.52° ratio 观测。

**偏心度精确对齐仍有缺口**：论文相关数组使用 temporal-equivalent eccentricity；本地 catalog 只有 eccentricity_deg，未提供这里所需的 nasal/temporal meridian 转换证据。下表代入数值 e 是明确的覆盖预算假设，不能冒充 retinal-location 精确匹配。

输入边长 0.91640625°，边界 ±0.458203125°；17 个采样中心跨 ±0.43125°，间距 0.05390625°。现有 frozen center (c_x,c_y) 不一定在 crop 中央，因此完整圆的最大可容纳半径为

$$r_{available}=0.458203125-\max(|c_x|,|c_y|).$$

所有 radius 比较围绕这个实际冻结中心，不重新居中。以下 r_c 范围来自上一节同 class 的分箱；均只说明 **1/e 轮廓**能容纳。r_s 与“截断”来自公开 pooled trend 情景，逐个真实生理半径及其容纳状态仍有散布/拟合不确定性。

| Cell | Class | e (°) | r_available (°) | 外部 r_c 范围 (°) | r_c 轮廓 | 趋势 r_s (°) | 趋势 r_s 轮廓 | 所需边长下界 (°) |
|---|---|---:|---:|---:|---|---:|---|---:|
| 67#14 | PC | 8.31 | 0.401 | 0.027–0.075 | 内 | 0.552 | 截断 | 1.218 |
| 67#21 | PC | 7.85 | 0.402 | 0.027–0.075 | 内 | 0.537 | 截断 | 1.187 |
| 67#26 | PC | 5.83 | 0.406 | 0.027–0.075 | 内 | 0.467 | 截断 | 1.038 |
| 67#33 | MC | 4.73 | 0.360 | 0.052–0.108 | 内 | 0.423 | 截断 | 1.042 |
| 67#34 | PC | 4.80 | 0.373 | 0.020–0.070 | 内 | 0.426 | 截断 | 1.022 |
| 67#4 | PC | 6.89 | 0.382 | 0.027–0.075 | 内 | 0.505 | 截断 | 1.162 |
| 67#6 | MC | 6.72 | 0.334 | 0.069–0.139 | 内 | 0.499 | 截断 | 1.245 |
| 67#7 | MC | 7.36 | 0.371 | 0.069–0.139 | 内 | 0.521 | 截断 | 1.216 |
| 68#10 | MC | 4.89 | 0.124 | 0.052–0.108 | 内 | 0.429 | 截断 | 1.528 |
| 68#11 | PC | 5.00 | 0.399 | 0.027–0.075 | 内 | 0.434 | 截断 | 0.986 |
| 68#3 | MC | 5.08 | 0.307 | 0.069–0.139 | 内 | 0.437 | 截断 | 1.176 |
| 68#4 | PC | 5.00 | 0.449 | 0.027–0.075 | 内 | 0.434 | 内 | 0.885 |
| 68#7 | PC | 5.72 | 0.353 | 0.027–0.075 | 内 | 0.462 | 截断 | 1.136 |
| 69#21 | PC | 4.40 | 0.213 | 0.020–0.070 | 内 | 0.409 | 截断 | 1.308 |
| 69#3 | PC | 4.59 | 0.424 | 0.020–0.070 | 内 | 0.417 | 内 | 0.901 |
| 69#4 | MC | 13.00 | 0.397 | 0.121–0.195 | 内 | 0.681 | 截断 | 1.484 |
| 69#6 | MC | 3.88 | 0.388 | 0.052–0.108 | 内 | 0.385 | 内 | 0.910 |
| 69#7 | MC | 3.52 | 0.389 | 0.052–0.108 | 内 | 0.368 | 内 | 0.874 |
| 70#1 | PC | 4.56 | 0.154 | 0.020–0.070 | 内 | 0.415 | 截断 | 1.439 |
| 70#15 | PC | 3.78 | 0.353 | 0.020–0.070 | 内 | 0.380 | 截断 | 0.971 |
| 70#34 | MC | 5.66 | 0.383 | 0.069–0.139 | 内 | 0.460 | 截断 | 1.071 |
| 70#7 | PC | 3.49 | 0.442 | 0.020–0.070 | 内 | 0.366 | 内 | 0.766 |

所需边长下界 = 2[趋势 r_s + max(|c_x|,|c_y|)]，保持 frozen center 与 crop origin 不变。表中轮廓“内”不等于积分强度已覆盖。

- 按这一外部趋势情景，**17/22** 个 surround 的 1/e 轮廓不完整：**MC 7/9，PC 10/13**。余下仅 MC 69#6、69#7 与 PC 68#4、69#3、70#7 的该轮廓可容纳；69#6、69#3 的余量尤其小，不能忽视文献误差。
- 即使所有 center 都理想位于 crop 中央，e 约大于 5.61° 的趋势 r_s 也超过半边长：当前 9/22 cells。因此问题不只是局部 registration 偏移。实际中心偏移进一步恶化 68#10、70#1、69#21 等 cell 的覆盖。
- 这些是**系统性截断风险的外部依据**，不是“已测得 17 个当前 cell 的真实 surround 被截断”。P 散点范围同时包含能容纳和不能容纳的尺度；M 的 class-specific surround scatter 仍不完整。两类都不能被认定为整体安全。
- 所有 Gaussian 都有无限尾部；有限 FOV 不可能严格覆盖全部积分。r_s 圆内仅包含该二维 Gaussian 总量的 1−exp(−1)≈63.2%。因此 1/e 轮廓通过，不足以宣告 I_s 可靠。
- 在上述 pooled trend 下，容纳所有 cell 的 **1/e 圆轮廓**至少需约 **1.53°×1.53°**，限制 cell 为 68#10；这只是有条件的几何下界。若仅作几何预算示例，完整包含 95%/99% Gaussian 积分对应的圆，需要 R=r_s√[−ln(1−q)]，当前中心下统一边长约 **2.48°/3.05°**。这些是容纳相应圆的充分方形边长，不是方形积分的最优边长、不是选定新 FOV，也不是 assay effect threshold。真实最小设计仍受散布、光学、stimulus aperture 和误差要求限制。

**分辨率是另一个独立限制。** P center 的 1/e 直径大致只有 0.75–2.8 个现有格点间距；当前空间 Nyquist 约 9.28 cycles/degree。可从有限样本拟合 subpixel radius，并不保证它被数据充分识别。不能把“center 圆落在 FOV 内”写成“center 已能精确验证”。在仍保持 17×17 时单纯扩大 FOV 会加粗采样，反而损害 center 尺度估计；本轮不改任何 preprocessing。

## 4. 正常模型的候选 frozen observable 是否对应论文量

| 候选 | 量纲/定义对应 | 当前可用程度及必须保留的边界 |
|---|---|---|
| 已有 normalized P(x)=Σ_lag J² / Σ_lag,x J² | 非负的离散敏感度质量；丢失符号、temporal phase 和 absolute gain。 | **不直接对应** DoG。其 centroid、second-moment radius、TV/径向 CDF 不能当作 r_c、r_s 或 I_s/I_c。 |
| NORMAL 的 signed effective RF | Jacobian 是固定 stimulus/history 条件下 logit 对 Weber 的局部敏感度；空间轴可标 degree。 | **有条件可建立对应**：保留符号与全部相关 lag，在论文对应背景/频率下形成复数小信号 transfer，再检验同一 DoG 参数是否可识别。现有 movie-context RF 不能直接代入；不假设时间可分离。 |
| External-geometry size tuning | 目前是 rectangular flash 的 baseline-subtracted mean logit 与其归一化曲线/SSI。 | 可做 frozen functional 描述，但**不是原论文 grating spatial-frequency tuning**。无需训练也能未来提取函数量，但本轮现有接口不足以精确复现该协议；flash optimal size 不等于 Gaussian radius。 |
| Fitted center/surround radii | 若从同定义 spatial-frequency response/DoG 提取，r_c/r_s 可同为 degree 与 1/e radius。 | r_c 是最佳候选；需先解决分辨率、temporal/optical/contrast 定义及 eccentricity 对齐。r_s 在当前 FOV 下显著依赖尾部外推，不能作完整 population quantitative primary。拟合模型的观测摘要不等于重训模型，但本轮未做任何拟合。 |
| Center/surround gain | I_s/I_c 是同一响应域、同一 contrast/频率条件下的 Gaussian 积分比；K 是峰值敏感度。 | 比率可能取消共同线性输出单位，但不能取消不同 history、非线性 operating point、光学或 FOV 差异。logit gain、gE/gI、canonical G_E/G_I 与解剖 E/I 都不等于 DoG center/surround gain。绝对 spikes/s/%contrast 对 logit/Weber 不可直接比较。 |

接口代码依据：[geometry 与 rectangular temporal builder](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/h1_external_geometry_assay.py:161)、[logit/SSI 汇总](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/h1_external_geometry_assay.py:203)。Bernoulli occupancy 的 150p 也不是无条件、任意 firing-rate regime 下的真实 spike rate；FIX_HISTORY_ZERO 是条件功能实验，不等于论文动物的 spontaneous/endogenous history。

此外，当前 [support_partition.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/support_partition.py:9) 已设 midget/parasol 不同 BC/AC support 半径（BC 0.06/0.10°，AC 0.13/0.15°）。这些是结构先验，不是本轮生理半径测量。本轮在 models/data/configs 中未发现 Croner/Kaplan 的显式来源引用；这不能证明历史设计从未受其影响。故独立证据应是**未用于调参的正常模型功能量与独立 cell 数据的定量一致程度**，不是重报 support 的大小顺序。

## 5. 三个 external constraints 的预注册可行性

| Constraint | 是否有外部依据 | 本 cohort 可预注册程度 |
|---|---|---|
| A. matched-eccentricity M center > P center | 有；原文观察和数字化散点均支持 population 关系。 | **有条件，优先准备。** Primary 候选为同定义的 r_c(e)，按 class/eccentricity 与外部散点比较；M/P 差异为预先指定的 contrast。共同数值范围约 3.52–8.31°；69#4 的 13°不能和缺少同偏心度 P 的本地均值作 matched 比较，可只与相应外部 M 数据描述比较。考虑 type support 先验，只验证方向不足以作强独立证据。 |
| B. center、surround 随 eccentricity 增大 | 有，涵盖本 cohort 数值范围，无需远外周外推。 | **center 可作为 secondary；surround 暂不可定量锁定。** PC 3.49–8.31°与 MC 一个13°点形成不平衡；不要拟合全22混合斜率当共同生理规律，也不应要求小样本 class 斜率必显著。须按类和可观测范围说明不确定性。 |
| C. surround/center gain relationship | 有，原文约0.55 population summary，且有带 cell ID 的 ratio 子集。 | **当前不宜作为 quantitative primary。** 整体 surround 未充分覆盖，接口/响应域也未对齐。不能把 external SSI、I/E strength、峰值 gain ratio 当替代品，不能以0.55附近定义成功。 |

选择 r_c 是因为其原始定义可明确对齐、两类有独立数据、对不可见 surround 积分尾部的依赖相对较小；不是预测它更容易通过。**当前没有已经准备完毕、可立即运行的 exact quantitative primary。** 可行性为 PARTIAL 而非完全不可行，是因为 center 尺度和 class/eccentricity 关系存在真实、覆盖当前范围的独立参照；接口、采样可识别性和协议缺项解决前，不能升级为已具备严格验证资格。

正常状态 RGC 空间性质即使定量吻合，也只支持整体模型功能一致；不能单独验证 H1 feedback、H1 state 或其生物机制。该文不是 H1 intervention 实验。

## 6. 最终回答

- **哪些 physiological quantities 能真正比较？** 条件对齐后，优先比较 NORMAL 的 DoG center characteristic radius r_c（degree）及 matched-eccentricity M/P 差异；r_s 和 I_s/I_c 仅在充分覆盖 surround、响应与拟合定义对齐后才可定量比较。当前 P(x)、support radius、flash SSI 和 E/I gain 都不能直接替代论文量。
- **当前 FOV 是否足够？** 对已取得文献 center 范围的1/e轮廓通常足够，但 center 采样精度仍受限；对两类完整 surround/gain 验证不够。公开 pooled trend 与当前冻结中心组合下17/22个surround轮廓越界，不能把它当作17个真实cell半径已被测定。
- **哪一个 quantity 最适合作为 primary independent physiology validation？** 按 eccentricity 和 class 对齐、统一 DoG/1/e 定义的 r_c；只做“M>P”方向检验不足，且当前精确 temporal/optical/采样合同尚未完备。
- **是否已经出现扩大 FOV 的独立科学依据？** **是，针对 surround 尺度与积分 gain 的覆盖要求已有外部依据。** 约1.53°是 pooled-trend 1/e轮廓覆盖下界，尾部覆盖预算可到约2.5–3.1°；这些不是已批准的新设计。扩大FOV还必须兼顾原角分辨率，单独放大17×17输入不是解决方案。本轮仅生成本报告，停止于可行性审计。
