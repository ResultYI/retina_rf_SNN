# Primate cone-phototransduction front-end feasibility audit

日期：2026-09-16。范围：外部论文、作者代码、原始公开数据的标定文件及当前输入转换源码。未加载或运行任何 RetiPath checkpoint；未训练、拟合、运行自然电影预测或新增架构；未解码 movie、读取 spike targets；未执行 cone forward。

**Verdict：CONE_FRONTEND_NOT_CALIBRATED。**

这里的 NOT_CALIBRATED 指“当前可追溯数据链尚不能确定 Chen 模型所需的 absolute cone input”，不指原实验从未做标定。原数据保留了 gamma、gun spectra 和分开的相对 L/M/S 转换；关联的 2002 年原论文还报告了 flower-show 时间刺激的 **222.2 td**。但该数值与当前 10 min 空间 movie 的绝对尺度对应关系，以及 photometric cone excitation 到 R*/cone/s 的转换，均未闭合。不能据此给当前 movie 指定一个 mean R*/s。代码可复用与生理输入已标定是两件事。

## 1. 外部模型来源与版本

采用 Chen et al. 2024 的 **Version of Record**，不是早期 reviewed preprint：*Predictably manipulating photoreceptor light responses to reveal their role in downstream visual responses*，eLife 13:RP93795，2024-11-05；DOI [10.7554/eLife.93795.3](https://doi.org/10.7554/eLife.93795.3)。版本由 [eLife article API](https://api.elifesciences.org/articles/93795) 的 `status=vor`、`doiVersion` 核实；正文、Methods 和 Tables 1–2 使用 [VoR 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11537484/) 及其 [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11537484/fullTextXML)。

作者仓库：[chrischen2/photoreceptorLinearization](https://github.com/chrischen2/photoreceptorLinearization)。本审计固定论文自身 Software Heritage 引用所指 revision：

`1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2`

这是本报告公式、接口和参数的代码依据。审计时仓库 HEAD 为 `c1429e9d1850181b44a3fe9178fd4dbed8f0f7bc`，已有后续 Python 实现；不将该较新 HEAD 冒充 2024 发表时的代码。以下均为固定 revision 的文件字节 SHA256：

| 文件 | SHA256 |
|---|---|
| `src/model/BiophysModel.m` | `fb5ff0a6880d5c78421990981dcf31932b0d415d9d216deb73a64753e5f37937` |
| `src/model/initPhotoreceptorParams.m` | `dbff286bc0bda0fbc0aa87775f83285f5dc93d5028559d19efb64651b9ef4de0` |
| `src/model/demoMain.m` | `f448fe4f9c0c1d3aa1a443240ff19fb6ff43e31912f57abb7e3ccee54bf2e88d` |
| `src/calibaration/calcIsomPerWatt.m` | `05a7b9206eb44d192e74aeb1b884a8a6c66ee9831c11446be97124056dce4256` |
| `src/calibaration/IsomerizationConverting.m` | `f7adff7b607fc080e92b63def54cb78b81cfe1b2dda07d71da0750f32f36cb7f` |

`calibaration` 是作者目录原拼写。源文件在系统临时目录读取，未安装或执行作者模型，未向项目加入实现。

## 2. Forward、输入输出与数值接口

以下用 \(F(t)\) 表示非负的 **photoisomerizations/cone/s（R*/cone/s）**，避免与内部 opsin activity \(R(t)\) 混淆。按论文 Methods/Figure 1 和 [固定版 BiophysModel.m](https://github.com/chrischen2/photoreceptorLinearization/blob/1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2/src/model/BiophysModel.m)：

\[
\dot R=\gamma F-\sigma R,\qquad
\dot P=R-\phi P+\eta,
\]
\[
\dot G=S-PG,\qquad I_{\rm mag}=kG^n,
\]
\[
\dot C=qI_{\rm mag}-\beta C,\qquad
S=\frac{S_{\max}}{1+(C/K_{GC})^m}.
\]

其中 \(P\) 为 PDE activity，\(G\) 为 cGMP 浓度，\(C\) 为 Ca²⁺ 浓度；\(R,P\) 是该模型所定义的 activity state，不应再解释为直接计数到的分子数。暗稳态约束为：

\[
G_D=(I_{D,\rm mag}/k)^{1/n},\quad
q=\beta C_D/I_{D,\rm mag},\quad
S_{\max}=G_D(\eta/\phi)[1+(C_D/K_{GC})^m].
\]

作者代码输出 **\(I=-kG^n\)，单位 pA**；`darkCurrent` 存正的暗电流幅值。光照导致通道关闭时，内向电流变得较不负。必须区分论文方程中的正幅值与代码返回的有符号电流。输出是 **outer-segment photocurrent**，不是 cone membrane voltage、glutamate release 或 H1 synaptic drive。

重要接口细节：

- 连续方程输入是 R*/cone/s；但固定 MATLAB 实现的 `params.stm` 是 **每个数值时间步内的 R*/cone**。`BiophysModel.m:59–61` 的光输入增量没有再乘 dt；[demoMain.m:66–68](https://github.com/chrischen2/photoreceptorLinearization/blob/1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2/src/model/demoMain.m#L66) 先生成 rate，再乘 `timeStep`。不能把 R*/s 数组直接当作 `stm`，也不能重复乘 dt。
- 作者默认 `timeStep=10^-4 s`，显式 Euler，初始为暗稳态；示例在正式变化前给予背景光使其稳定。当前 movie 的 1/150 s bin 不能未经数值验证直接替代该 Euler 步长。未来如获授权接入，需另行固定 bin 内光强表示、子步积分与背景适应初态；本轮不设计或实现。
- 当前 RetiPath 的 0 Weber 表示非零背景光，不是 \(F=0\) 的黑暗。把 0 Weber 直接传入此模型会把 operating point 改为黑暗。

### Primate cone 参数：哪些来自哪里

下表值来自 VoR Table 1/Methods 和 [固定版 initPhotoreceptorParams.m](https://github.com/chrischen2/photoreceptorLinearization/blob/1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2/src/model/initPhotoreceptorParams.m#L19)，不是本项目拟合结果。

| 参数 | `peripheralPrimateCone` 值/单位 | 论文中的来源与自由度 |
|---|---|---|
| \(\sigma\) | 22 s⁻¹ | 从 cone recordings 拟合的 consensus 参数；不是先验固定的普适常数 |
| \(\phi\) | 22 s⁻¹ | 约束为 \(\phi=\sigma\)，因两者对响应作用相近 |
| \(\eta\) | 2000，论文表列 s⁻¹ | 从 cone recordings 拟合的 consensus PDE dark activation 参数 |
| \(\beta\) | 9 s⁻¹ | cone consensus 拟合参数；不要误用 rod 中固定 β 的做法 |
| \(K_{GC}\) | 0.5 μM | cone consensus 拟合参数 |
| \(\gamma\) | 10，论文表列 unitless | 作者参考值；论文验证时可随 cone 拟合，primate fitted γ 为 8±3（mean±SD） |
| \(k\) | 0.01 pA·μM⁻³ | 文献依据下固定的 current–cGMP 常数 |
| \(n\) | 3 | 固定 cooperativity |
| \(C_D\) | 1 μM | 假设固定；可与 \(K_{GC}\) 的尺度补偿 |
| \(m\) | 4 | 固定 cooperativity |
| \(G_D\) | 默认 35 μM | 作者示例参考值；论文各 cone 由实测暗电流推导，为 28.7–35 μM |
| \(I_D\) | 默认 −428.75 pA | 由上述默认值计算；论文实测约 −240 至 −428 pA，不是所有 cone 的同一实测值 |
| \(q,S_{\max}\) | 暗稳态公式推导 | 不另设独立自由参数 |
| dt | 0.0001 s | 数值设置，不是生理拟合参数 |

“published fixed physiology front end”可以指**冻结作者已经发表的 consensus/reference 值**，不能写成“这些值在原论文里全都无需拟合”。原文单 cone 模型有五个拟合量：\(\gamma,\eta,\beta,\sigma,K_{GC}\)；consensus 模型共享后四项，同时各 cone 使用自己的暗电流及 sensitivity γ。发布的参考参数足以数值运行一个未记录 cone 的预测模型，**不要求为当前 RGC 数据新增 cone 参数拟合**；但它也不证明当前实验的每个 cone 都拥有该暗电流和 sensitivity。

### 适用范围

Chen 模型不包含 pigment bleaching/regeneration，原文限定 cone 应用光强 **<50,000 R*/s**。这是适用性上界，不代表从零到该上界的所有时间波形都已有验证。作者演示的 5000 R*/s 是示例刺激工作点，不是 Schottdorf–Lee 的标定值。实验 primate cones 位于 **>20°**，而本项目 cohort 约 3.52–13°；因此 peripheral consensus 的跨 eccentricity 适用性仍是外推边界。当前 absolute input 不确定，也就无法确认自然电影各像素是否始终处于该 light-level 范围内。[Chen VoR Methods / Limitations](https://pmc.ncbi.nlm.nih.gov/articles/PMC11537484/)

## 3. 原始 Schottdorf–Lee 标定链

公开数据：[GIN archive DOI 10.12751/g-node.xage77](https://doi.gin.g-node.org/10.12751/g-node.xage77/)，[作者 repository](https://gin.g-node.org/Manuel/Macaque-ganglion-cells)。本地原始 clone：`data/real/schottdorf_lee_2021_repository`，HEAD `cffefb08c760f04c9a951da46061b361d2288e9b`。

检查了原始 README、`stimuli/GunSpectra.txt`、`GunSpectra.jpg`、原始 `retinatools/library.py` 及 calibration 相关 notebook/Python 源码；同时核对 2021 原论文 Methods，以及它明确引用的 2002 原始刺激方法。没有用当前 Weber tensor 代替这一步，也没有读取 notebook 内的模型结果作为标定证据。

| 项目 | 已核实事实 | 对 absolute cone input 的意义 |
|---|---|---|
| 原始 RGB movie | 原始 repository 保存 movie；README 区分 1 min 与 10 min，给出 blank/live frame 结构 | 可回到未合并 RGB 重建相对 L/M；本轮未解码 |
| Gamma | 三枪有各自幂律和黑电平 offset，见下式 | 解决相对数字值→强度非线性，不单独确定光子单位 |
| Gun spectra | 三枪光谱保存在 TXT；图纵轴为 **Intensity [a.u.]**；TXT 无绝对辐射单位/header | 有光谱形状，未找到 W、W/sr/m² 或经溯源绝对功率因子 |
| L/M/S excitation | 原始公式分别存在，并固定用于所有 cells | 支持 separate relative cone signals；不是已注明 R*/s 的 signals |
| 光度校正来源 | README 说明 gun luminance 初标定后，参照 M-cell reverse correlation 调整绿/蓝相对贡献 | 属于作者的有效刺激估计，不能完全当作独立的绝对 radiometric 标定 |
| Background | 2021 原文：equal-energy white，mean luminance 与相应 movie 相同；10 min 文件有 751 个前置 blank frames | 可定义 Weber 分母，但不能由“相同均值”推得绝对数值 |
| 显示系统 | Iiyama Vision Master Pro 410，150 Hz；gamma 用 calibrated photomultiplier，光谱用 Ocean Optics spectrometer | 原实验确实做过光学标定；仪器名称不等于已公开其绝对读数 |
| Pupil | **4 mm artificial pupil**，2021、2002 原文均明确 | 并非缺失项；但瞳孔直径本身不决定 photon catch |
| Absolute photometric clue | 2002 flower-show **时间刺激**平均 **222.2 td**；另一 LED/laboratory stimulus 为 1179 td | 有真实的绝对照度线索；不能将 1179 td 用于当前 movie |
| 当前 10 min movie 的绝对 anchor | 已检查来源中未建立 222.2 td 与当前 `1x10_256.mpg`/blank calibration 数值的逐项对应 | 当前文件的 mean R*/s 仍为 UNVERIFIED，不能简单乘一个猜测因子 |
| Retina photon catch | 未找到当前实验的 R*/cone/s 表、绝对 retinal spectral irradiance，或完整 display→retina→cone 转换 | 尚不能直接运行有标定含义的 Chen front end |
| Ocular conversion | 未找到与这组记录对应的完整透过率、有效 retinal optical geometry/collecting-efficiency 标定 | Chen 离体 retinal-plane 校准不能原样替代在体 CRT 光路 |

2021 原文来源：[Schottdorf & Lee, DOI 10.1113/JP281200](https://doi.org/10.1113/JP281200)，Methods “Stimuli”；本次全文核对使用[作者上传的全文版本](https://www.researchgate.net/publication/351203785_A_quantitative_description_of_macaque_ganglion_cell_responses_to_natural_scenes_the_interplay_of_time_and_space)。

**222.2 td 的来源边界很重要。** van Hateren et al. 2002 的 Methods p.9946 / Figure 1 报告该数值用于 flower-show **1 min、中心区域空间平均后生成的均匀时间刺激**；2021 论文明确另有 10 min 空间 movie。共同的原始场景和显示设备不是当前文件具有同一绝对均值的证明。本审计保留这个来源，不将其删除为“完全没有 absolute luminance”，也不把它升级为当前 10 min 每像素标定。[2002 出版版全文](https://pure.rug.nl/ws/files/3046327/2002JNeuroscivHateren.pdf)

### 可复核的现有变换

原始 [README:98](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/README.md:98) 给出：

\[
r=0.01451+0.9855(I_r/256)^{2.3122},\quad
g=0.005123+0.9949(I_g/256)^{2.2752},\quad
b=0.02612+0.9739(I_b/256)^{2.2818};
\]
\[
L=2.74r+3.4g+1.34b,\quad
M=1.21(1.06r+3.58g+2.07b),\quad
S=0.212r+8.28g+285b.
\]

这些数字并未在该 README 中标注为 R*/cone/s。Smith–Pokorny cone fundamentals 所定义的 photometric L/M contributions 也不能直接当成绝对 opsin isomerization rates。

当前 [data/schottdorf_lee_2021.py:164](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:164) 的实际输入链为：

`RGB → gamma → L/M → L+M → central 51×51 crop / 3×3 pooling → 17×17 → (signal − blank_mean) / blank_mean`。

因此正式输入是 **dimensionless pooled L+M Weber proxy**；没有 retinal optical calibration 或 R*/s 因子。Weber 化会消除整体强度尺度：对任何正 \(\kappa\)，\((\kappa L-\kappa L_0)/(\kappa L_0)=(L-L_0)/L_0\)。仅凭现有 tensor 无法反推光适应水平。

本次直接读取文件的 SHA256：

| 文件 | SHA256 |
|---|---|
| 原始 `README.md` | `aec20c21a4b00d58afaa4144f44c77b8b850f555871e20dc57f1562b4a831ef2` |
| 原始 `stimuli/GunSpectra.txt` | `0c570201ba48f280e1c3916e9b0145225edd804ec7c16586b0db51a82dbfbd6f` |
| 原始 `stimuli/GunSpectra.jpg` | `214bd94f6d79484de89be12ab7971b9fdf44a438efe1ffa0f4fe842c06a69190` |
| 原始 `retinatools/library.py` | `92fc28ac16177bd314910254c41e346334adec374e60b2f3cc369f8ccfb0a50f` |
| 当前 `data/schottdorf_lee_2021.py` | `ae07cb57443ce95fcef2208060638c1cc12168ec9e1a1f3f9e58c82e24ddf764` |

这些 hash 固定的是本次证据文件，不意味着审计或读取了数据集中每个 target/派生产物。

## 4. 从标定到 R*/cone/s 还缺什么

Chen 的 light calibration 是在 **retinal preparation 平面**实测光功率，除以照明面积得到 spectral power density，再按光谱、photon energy、photoreceptor sensitivity 和 collecting area 积分。primate cone collecting area 使用 **0.37 μm²**。抽象写为：

\[
F_c(t)=A_{\mathrm{coll},c}\int
E_{\mathrm{retina}}(\lambda,t)\frac{\lambda}{hc}
q_c(\lambda)\,d\lambda.
\]

这里的 \(E_{\mathrm{retina}}\) 必须是有绝对单位的 retinal spectral irradiance，且与 \(A_{\rm coll}\)、光谱单位一致；\(q_c\) 的有效 spectral sensitivity 定义须与所用 collecting area 一致。作者 `IsomerizationConverting.m` 中的 LED power/voltage 和 spot diameter 是其 setup 的例子，不能移植成 Schottdorf CRT 的标定。[Chen light calibration](https://pmc.ncbi.nlm.nih.gov/articles/PMC11537484/)、[固定版校准代码](https://github.com/chrischen2/photoreceptorLinearization/blob/1bcf7c9a6a6814ef39d2c39d82cf8dd1b5b39db2/src/calibaration/IsomerizationConverting.m)

Trolands 是 photometric retinal illuminance，通常由 luminance 与 pupil area 表示，**不是** R*/cone/s。即使落实当前 movie 的 td，还须给定与光谱、眼内透过和 cone collecting efficiency 一致的转换。Smith–Pokorny fundamentals 的 ocular filtering 不能与另一个光学模型不加检查地重复计算。不能只把 0.37 μm²、4 mm pupil 和未注明绝对单位的谱相乘就宣称转换完成。

按本任务三种输入可行性分类：

| 类别 | 本次判断 |
|---|---|
| A — FEASIBLE_CALIBRATED | **不成立**：无可追溯的当前 movie → absolute L/M photon catch 闭合链 |
| B — FEASIBLE_UP_TO_SCALE | **尚未充分建立**：存在 222.2 td 的关联实验线索，但它与当前文件的关系未确认；也不能保证余下不确定性仅为一个全局公共尺度，而没有相对 L/M 光谱/ocular conversion 问题 |
| C — NOT_CALIBRATED_FOR_BIOPHYSICAL_CONE_FRONTEND | **当前正式输入链所属类别**：可证明的是相对 L/M 或 pooled Weber，工作光强尚不可恢复 |

最小缺失证据是：当前实际 movie/blank 所对应的 absolute display calibration（或可明确关联的 photometric/retinal spectral 标定），以及采用何种有来源的 in-vivo L/M photon-catch 转换及其不确定性。相关 2002 光度数据是可追溯线索，**本轮未将其转换为一个推定 R*/s**。若未来证据补齐，可重新评估类别；当前不以猜测 mean、调 γ 或预测评分补齐。

## 5. L/M 处理与输出语义

**原始 RGB 保留了分别重建相对 L/M 所需的信息；当前单通道 L+M Weber 不保留它。** 作者 cone phototransduction equations 可以对 L 和 M 各自执行相同固定动力学，谱差异通过各自 absolute input 表示。论文没有为本项目提供可直接套用的 L/M subtype-specific fitted parameter sets，也没有提供真实 cone mosaic。

若未来补齐 absolute calibration，正确的候选数据流应为：

```
calibrated RGB/spectra → L absolute drive → fixed cone model → L photocurrent
                       M absolute drive → fixed cone model → M photocurrent
                     → 按当前 luminance-pathway 定义合并 → 后续既有通路
```

合并权重须与 photometric L/M 定义一致；不能把两个“每 cone 的 pA”无说明相加后声称完全保留原 photometric 权重。具体 photocurrent 到当前 effective input 的单位、暗/背景基线、符号也需在未来接口合同中明确，不是把 signed pA 直接解释为 transmitter release。这里仅记录要求，不选取新缩放量或更改模型。

由于 \(f(L+M)\ne f(L)+f(M)\)，**直接对 pooled L+M Weber 运行 cone model 违反 absolute per-photoreceptor input semantics**。同样，非线性与空间 pooling 通常不交换；把模型作用在 17×17 pooled samples 上只能称每个有效样点的前端近似，不能称实际单 cone simulation。这不是本轮修改 support、FOV 或 preprocessing 的理由；此处只记录未来必须声明的近似。

## 6. 参数、可识别性及证据边界

**固定作者参考 cone front end，可新增 0 个 trainable parameters，并保留原 BC alpha。** 这并不使实验 calibration 自动固定。作者参考 \(\gamma=10,G_D=35\) 足够定义一个 frozen forward，不必为运行代码拟合当前 cells；但与论文逐 cone 验证同等级的参数归属需要各 cone 的 dark current/sensitivity 证据，当前 RGC 数据没有提供这些观测。

若反而允许 cone 参数学习，至少有以下问题：

- **精确输入尺度混淆**：\(F=\kappa F_{\rm relative}\) 时，opsin 驱动只含 \(\gamma\kappa\)。\((\gamma,\kappa)\mapsto(\gamma/c,c\kappa)\) 保持该驱动和后续轨迹不变。RGC NLL 不能区分 photon calibration 与 cone sensitivity。
- **已知模型内部不可辨识**：论文因此约束 \(\sigma=\phi\)，并讨论 \(C_D\) 与 \(K_{GC}\) 的补偿。不能把每个 biochemical label 都当作从响应唯一恢复的生理量。
- **下游实际冗余风险**：可学习 cone gain/dark-current scale 可能被 BC/EI/output gains 部分吸收；cone kinetics 与既有 H1/BC/AC 时间滤波可能互相补偿。这些是 practical identifiability 风险，不在本轮无推导地宣称为精确 gauge，也未运行模型检验。

因此不提出从 RGC NLL 拟合 cone light level、γ、time constants 的方案。当前阻碍首先是 **stimulus calibration 和输入语义**，不是缺少可训练参数。

Raval et al. 2026 [preprint](https://doi.org/10.64898/2026.03.19.713068) 的结果为这一方向提供独立动机：固定 phototransduction front end 可不增加自由参数，cone-output synapse 的非线性是另一个过程。其 Methods 使用 20–50° peripheral retina；Figure 5 的简化 local-adaptation front end 使用独立 Weber gain 近似，并非所有分析都运行同一个完整 Chen forward。该文中的 2000 R*/cone/s 是 half-desensitizing background 参数，不能用作本数据的 mean illumination。本审计不由该 preprint 推断 Chen front end 已足以恢复本项目 H1 F2，也不增加 cone-output synapse 模块。

## 7. 最小 dry-run 与完成边界

仅完成了 source-level 单位检查：输入 rate 与 per-step isomerizations 的转换、输出符号、暗稳态参数关系、L/M 信息保留情况、绝对标定缺口。**未执行 synthetic/calibration forward fixture**：用户规定仅在 absolute calibration 完整时才允许，而当前条件不满足。不能用任选 light level 跑出 finite output 充当可行性证据。

仅新增本报告。正式 RetiPath、calibration pipeline、checkpoint、历史 artifacts 均未修改；本轮没有 FAST SCREEN 资格，不启动后续实验。

## 最后六问

1. **Schottdorf–Lee 能否恢复 Chen 所需 absolute cone input？** 目前不能。相对标定充分可追溯，且存在关联实验的 222.2 td 线索；当前 10 min movie 的绝对 anchor 与 L/M photon-catch 转换尚未闭合。
2. **L/M 能否分别输入？** 原始 RGB 可分别恢复相对 L/M；补齐 absolute calibration 后可分别运行同一固定 cone model。当前合并后的 L+M Weber 不能直接承担该输入。
3. **是否有可固定的 primate parameters？** 有作者发布的 peripheral-primate consensus/reference 值；其中一些源自生理拟合，γ/暗电流在论文逐 cone 验证时仍个别确定，不能称普适的全部实测常数。
4. **新增多少 trainable parameters？** 若全部固定，**0**；不要求也不建议用本任务的 RGC targets 再拟合 cone 参数。
5. **最大 blocker？** 当前刺激 absolute operating light level 未确定，且与 γ 形成精确尺度混淆；现有 photometric、光谱与眼内转换证据不足以关闭这条链。
6. **是否可以开始 FAST SCREEN？** **不可以。Verdict：CONE_FRONTEND_NOT_CALIBRATED。** 保留外部 cone-front-end 方向的科学动机，但不能靠猜 mean R*/s 升级。完成本报告后停止。
