# Additional Parameter Evidence for the Primate Retina V1

检索日期：2026-07-10  
适用范围：当前 human/macaque predictive retina-SNN V1，以及 Stage -1 ISETBio 刺激生成。  
与主文档的关系：本文件只补充 `parameter_evidence_human_macaque_v1.md` 尚未明确列出的实验量，不替代主文档中的参数分类与训练边界。

## 1. 结论先行

新增文献主要能约束以下四类量：

1. H1 与 cone 的局部连接数，以及 H1 感受野随偏心度变化的尺度。
2. midget/parasol 的相对空间尺度、中心--周边强度和 bipolar 汇聚数。
3. midget/parasol、ON/OFF 通路的相对时间动力学和 RF 不对称。
4. Stage -1 眼动轨迹的统计范围。

这些结果不能直接给出当前归一化网络中的 `g_AB`、`g_BA`、`g_AG`、阈值、surrogate slope 或 loss weight。论文中的 time-to-peak、RF diameter、contrast gain 也不能分别直接等同于一阶状态的 `tau`、Gaussian `radius_degs` 或代码中的无量纲 gain。

当前最值得立即采用的是轻量约束和诊断：

- 检查每个 cone 的 H1 入度是否主要落在 3--5，而不是新增 H2 或 gap coupling。
- 按训练视野的偏心度选择 profile，并记录该偏心度；不要用一个 profile 覆盖中央凹和外周。
- 将 parasol/midget RF center 尺度比、surround/center integrated gain、ON/OFF 尺度与速度顺序加入训练后 RF 报告。
- 将 bipolar 汇聚数作为 mosaic/pooling 的结构检查，不把解剖突触数当作权重大小。
- 将眼动统计用于 ISETBio 生成器的 sanity check，不作为 SNN 的额外输入。

## 2. 证据等级

- **A**：人类或猕猴的一手定量实验，可直接约束相同测量量。
- **B**：一手实验可靠，但映射到当前抽象需要单位转换、偏心度匹配或分析定义匹配。
- **C**：只支持相对顺序或诊断，不足以固定代码参数。
- **D**：工程量；没有生理实验可直接标定。

## 3. 新增参数证据表

| 参数或可测指标 | 一手实验结果 | 物种与条件 | 对当前模型的有效映射 | 当前处理 | 等级 |
|---|---|---|---|---|---|
| cone 到 H1 的局部覆盖数 | 每个 cone 在全视网膜范围内与约 3--5 个 H1、以及 3--5 个 H2 水平细胞连接 | macaque，形态学重建 | `cone_to_h1` local mask 的每列非零数/有效入度；当前只实现 H1 | **立即加 sanity check；不加入 H2** | A/B |
| H1 combined RF diameter | 4 mm 偏心处约 122 um，11 mm 偏心处约 309 um；约 80% 细胞可由窄中心加宽浅裙边的双指数描述 | macaque，离体 H1 生理记录 | H1 空间 support 的偏心度依赖；宽浅成分涉及 H1 coupling | **用于 profile/诊断；V1 不据此加入 gap coupling** | A/B |
| parasol/midget RF center 尺度比 | M-cell center radius 约为邻近 P-cell 的 2 倍 | macaque，LGN 记录的视网膜输入 | 训练后 parasol/midget Jacobian、STA、GLM center radius 比 | **诊断优先；暂不硬编码为恰好 2** | A/B |
| center--surround integrated gain | M 与 P 的平均 surround/center integrated gain ratio 均约 0.55 | macaque，同上 | RF readout 的 DoG/GLM 指标 | **训练后验收指标；不能作为 H1 `gain=0.55`** | A/B |
| M/P integrated contrast gain | M-cell 平均约为 P-cell 的 6 倍 | macaque，同上 | parasol/midget contrast-response slope 或局部预测敏感性比较 | **诊断；不能映射到归一化 decoder 权重** | A/C |
| 人类 ON/OFF dendritic field 尺度 | ON/OFF 直径比：midget 1.5 +/- 0.3，parasol 1.3 +/- 0.1；midget/parasol 直径比由外周约 1:3 到 3 deg 处约 1:10 | human，形态学 | ON/OFF mosaic/RF 尺度的方向性检查；midget/parasol 密度与尺度随偏心度变化 | **用于 geometry 与 RF 报告；不立即拆分更多 profile 参数** | A/B |
| macaque ON/OFF parasol 功能不对称 | ON RF diameter 约大 20%；time-to-peak、trough、zero-crossing 短 10--20% | macaque，MEA 白噪声 | ON/OFF RF size、TTP、TTZ 的训练后比较 | **轻量 sanity check；第一版不强迫差异** | A/B |
| midget 时间峰值的偏心度依赖 | STA TTP：foveal 67 +/- 6 ms，central 53 +/- 6 ms，peripheral 37 +/- 4 ms；biphasic index 分别 0.33 +/- 0.15、0.30 +/- 0.06、0.51 +/- 0.35 | macaque，约 0--1、2--4、>6 mm 偏心 | 选择与训练 crop 偏心度匹配的时间 profile；RF temporal probe 的目标范围 | **核心修正候选：profile 必须声明 eccentricity；TTP 不等于 `tau`** | A/B |
| midget cone convergence | 研究分组约为 foveal 1 cone/MGC、central 3--4、peripheral 10--20；中央凹形成 private line | macaque | midget mosaic/private-line mask 的偏心度依赖 | **中央凹/旁中央凹 V1 保持近一对一；跨偏心度时再扩展** | A/B |
| foveal midget bipolar 到 RGC 的兴奋性突触数 | 两组 private-line 通路分别报告 28 +/- 4 与 47 +/- 3 个 midget bipolar synapses | macaque fovea，serial EM | private-line 拓扑与较高可靠性的解剖依据 | **只约束连接存在与非发散；不把突触数当作权重** | A/C |
| midget/parasol bipolar 汇聚差异 | 典型记录偏心度下，midget center 收集约 1--3 个 midget bipolar，parasol center 至少约 30 个 bipolar | macaque，生理记录与电路模型 | `parasol_radius_degs`、mosaic spacing 和 pooling mask 的有效输入数 | **加入 pooling/convergence 报告；避免只用 stride 解释类型差异** | B |
| 人类 midget/parasol 功能顺序 | parasol temporal RF 更窄、更 biphasic，spatial RF 更宽、response threshold 更低；ON 相对 OFF 更宽、更快/更 biphasic且阈值更低 | human，外周离体 MEA | cell-type-specific RF 与 contrast-response 方向性验收 | **作为直接人类证据；先诊断，不新增损失** | A/C |
| parasol spike timing precision | 强、高对比、空间均匀刺激下，trial-to-trial spike-time variability 最低可到约 1 ms，且 spike count 为 sub-Poisson | macaque，离体 parasol MEA | spike-history/GLM 分析的时间分辨率与可靠性指标 | **诊断；绝不能设为 `membrane_tau_ms=1`** | A/B |
| natural-viewing microsaccade rate | 动态自然场景观看时平均 0.77 s^-1，受试者范围 0.5--1.1 s^-1；典型 fixation 约 400 ms | human，自然视频 eye tracking | Stage -1 eye trajectory 生成与 manifest 统计 | **数据生成 sanity check；不作为网络输入** | A/B |
| ocular drift 个体差异 | 健康人 fixation drift diffusion constant 约 5--20 arcmin^2/s，个体差异约四倍 | human，高精度眼动记录 | ISETBio drift trajectory 的统计范围与多 seed 覆盖 | **后续 generation profile；当前 smoke 不需扩成消融** | A/B |

## 4. 对当前代码参数的具体解释

### 4.1 H1 spatial profile

当前 `human_macaque_v1()` 使用：

- `h1.radius_degs = 0.16`
- `h1.sigma_degs = 0.10`
- `h1.feedback_radius_degs = 0.21`
- `h1.feedback_sigma_degs = 0.12`
- `h1.h1_spacing_degs = 0.20`

新增文献不能直接证明这些 degree 数值。Packer 与 Dacey 报告的是 4--11 mm 偏心处的 H1 生理 RF diameter，且宽浅成分部分来自电耦合；当前 V1 明确暂缓 gap coupling。因此合理做法是：

1. 保留现有弱 prior 供 smoke test。
2. 在构建 mask 后报告 cone-to-H1 有效入度分布，并以 3--5 为主要解剖参考。
3. 在 profile metadata 中声明适用偏心度，之后再做 mm/deg 与 ISETBio mosaic 的一致转换。

这属于**核心修正的准备工作**，不是要求现在加入 H2 或电耦合。

### 4.2 RGC spatial profile

当前 `parasol_radius_degs = 0.16` 与 midget private-line 的差异主要由连接拓扑产生。Croner 与 Kaplan 的结果支持 parasol/M center 大于 midget/P center，Manookin 等人的结果支持 bipolar 汇聚数存在数量级差异。

当前最小修改方向是先报告每个 RGC 的有效 bipolar 输入数和训练后 RF center radius，不立即把 `parasol_radius_degs` 调到某个唯一值。若 parasol RF 未比 midget 更宽，优先检查 local mask、mosaic spacing 和 pooling 是否实际生效，再调整 profile。

### 4.3 Temporal profile

Sinha 等人的 TTP 表明同一种 midget 通路也会随偏心度显著变慢或变快。它不支持把：

```text
TTP = membrane_tau = bipolar_tau = rate_tau
```

视为同一个量。当前应保留多个有界状态时间常数，并在 RF probe 中计算 TTP、TTZ 和 biphasic index。只有当目标偏心度明确、且生成数据的 `dt_ms` 足够解析这些动力学时，才调整初值与 bounds。

### 4.4 ON/OFF asymmetry

人类形态学与人类/猕猴生理实验都支持 ON 相对 OFF 更宽，并在 parasol 中更快。第一版不建议新增 ON/OFF 专用 loss 或强制比例。最小做法是将以下量加入训练后报告：

- ON/OFF center radius ratio
- ON/OFF TTP 与 TTZ
- biphasic index
- contrast-response threshold/gain（若 probe 刺激允许）

若模型完全不呈现这些方向性差异，也不应立刻增加 A1/A3；先检查 ON/OFF bipolar 非线性和局部连接是否对称得过强。

### 4.5 A2-like amacrine

本轮检索没有找到能安全映射为当前 `A2AmacrineConfig` 中 `radius_degs`、`tau`、`g_BA`、`g_AB` 或 `g_AG` 的人类/猕猴直接数值。灵长类 AII 文献包含密度、树突形态、rod/cone 通路和 gap-junction 结果，但当前模块只是最小 BC--AC recurrent state，并非完整 AII 细胞模型。

因此当前结论仍是：

- `g_AB`、`g_BA`、`g_AG` 弱初始、有界、非负。
- A2 local mask 归一化并限制局部 support。
- 用阻尼恢复、biphasic/multiphasic RF、gain 不塌缩且不振荡作为诊断。
- **不以 AII 解剖密度为由加入 gap coupling、A1 crossover 或额外 amacrine 类型。**

## 5. 当前第一版应做与暂不做

### 应做

1. 为 profile 和每个 HDF5 明确记录 eccentricity。
2. 在 mosaic factory 输出 cone-to-H1 入度、midget/parasol bipolar 汇聚数分布。
3. 在 RF probe 增加 center radius、surround/center gain、TTP、TTZ、biphasic index。
4. 在 Stage -1 manifest 记录 microsaccade rate 或 drift diffusion statistic（生成器支持后再加）。

### 暂不做

1. 不把 `0.55` 写成 H1 feedback gain。
2. 不把 1 ms spike precision 写成 LIF membrane tau。
3. 不把 EM 突触数写成 decoder 或 recurrent weight。
4. 不因 H1 宽浅 RF 成分自动加入 gap coupling。
5. 不因真实 parasol 接受 wide-field amacrine 输入自动加入 A3。
6. 不为每个新增数值建立消融矩阵。

## 6. 参考文献

1. Wässle H, Boycott BB, Röhrenbeck J. Horizontal Cells in the Monkey Retina: Cone connections and dendritic network. *European Journal of Neuroscience*. 1989;1:421--435. https://doi.org/10.1111/j.1460-9568.1989.tb00350.x
2. Packer OS, Dacey DM. Receptive field structure of H1 horizontal cells in macaque monkey retina. *Journal of Vision*. 2002;2:272--292. https://doi.org/10.1167/2.4.1
3. Croner LJ, Kaplan E. Receptive fields of P and M ganglion cells across the primate retina. *Vision Research*. 1995;35:7--24. https://doi.org/10.1016/0042-6989(94)E0066-T
4. Dacey DM, Petersen MR. Dendritic field size and morphology of midget and parasol ganglion cells of the human retina. *PNAS*. 1992;89:9666--9670. https://doi.org/10.1073/pnas.89.20.9666
5. Chichilnisky EJ, Kalmar RS. Functional asymmetries in ON and OFF ganglion cells of primate retina. *Journal of Neuroscience*. 2002;22:2737--2747. https://doi.org/10.1523/JNEUROSCI.22-07-02737.2002
6. Sinha R, Hoon M, Baudin J, Okawa H, Wong ROL, Rieke F. Cellular and circuit mechanisms shaping the perceptual properties of the primate fovea. *Cell*. 2017;168:413--426.e12. https://doi.org/10.1016/j.cell.2017.01.005
7. Calkins DJ, Schein SJ, Tsukamoto Y, Sterling P. M and L cones in macaque fovea connect to midget ganglion cells by different numbers of excitatory synapses. *Nature*. 1994;371:70--72. https://doi.org/10.1038/371070a0
8. Manookin MB, Patterson SS, Linehan CM. Neural mechanisms mediating motion sensitivity in parasol ganglion cells of the primate retina. *Neuron*. 2018;97:1327--1340.e4. https://doi.org/10.1016/j.neuron.2018.02.006
9. Soto F, Hsiang JC, Rajagopal R, et al. Efficient coding by midget and parasol ganglion cells in the human retina. *Neuron*. 2020;107:656--666.e5. https://doi.org/10.1016/j.neuron.2020.05.030
10. Uzzell VJ, Chichilnisky EJ. Precision of spike trains in primate retinal ganglion cells. *Journal of Neurophysiology*. 2004;92:780--789. https://doi.org/10.1152/jn.01171.2003
11. Roberts JA, Wallis G, Breakspear M. Fixational eye movements during viewing of dynamic natural scenes. *Frontiers in Psychology*. 2013;4:797. https://doi.org/10.3389/fpsyg.2013.00797
12. Clark AM, Intoy J, Rucci M, Poletti M. Eye drift during fixation predicts visual acuity. *PNAS*. 2022;119:e2200256119. https://doi.org/10.1073/pnas.2200256119

