# McMahon、Packer & Dacey（2004）：H1 外部生理协议可行性审计

日期：2026-09-15。结论：**NOT_FEASIBLE_EXACT**。

当前冻结的 17×17 输入场无法容纳论文原尺度刺激，现有接口也不能表示其周期调制及电压 Fourier 读出。论文可提供有条件的定性外部约束，但本次审计不构成 independent physiology validation。

本轮仅检查源码、坐标、细胞元数据与外部文献，并生成本报告。未加载或运行 checkpoint，未读取 natural-movie spike target，未训练，未修改模型、已有 assay 或配置。

## 1. 当前输入场与适用 cohort

当前数据前端从 4.6°、256×256 的原电影取中央 51×51 像素，再按 3×3 pooling 得到 17×17。格点间距为 `3×4.6/256 = 0.05390625°`；每轴格点中心约为 `[-0.43125°, +0.43125°]`，中心跨度 0.8625°；包含边缘半个像素的完整输入场为 **0.91640625° × 0.91640625°**，边界约 ±0.458203125°。不能把原电影的 4.6° 当成当前模型的输入宽度。[前端与坐标公式](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:18)

对已保存的 22-cell stimulus-definition 坐标作纯数组核对：全部为相同的 `[289, 2]` degree grid；float32 实际端点为 ±0.4312499762°，与前端公式重建的最大绝对误差为 0。坐标来自刺激定义，不涉及模型响应。[冻结坐标文件](D:/PythonProject/retina_rf_SNN/output/evaluations/retipath_h1_center_surround_assay_20260915/stimulus_definition.json)

原始 Schottdorf–Lee metadata 保存了 cell eccentricity；当前 MC/parasol 子集为 **9 cells：5 ON、4 OFF**，范围 **3.52–13.00°**。另外 13 个 PC/midget 不进入本论文的 primary validation。三 seeds 即使后续全部运行，仍只有 9 个生物学 cell 单位。[原始 metadata](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/README.md:44)、[项目 catalog](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_catalog.py:26)

| MC/parasol cell | Polarity | Eccentricity（°） |
|---|---|---:|
| 67#6 | OFF | 6.72 |
| 67#7 | ON | 7.36 |
| 67#33 | OFF | 4.73 |
| 68#3 | OFF | 5.08 |
| 68#10 | ON | 4.89 |
| 69#4 | ON | 13.00 |
| 69#6 | OFF | 3.88 |
| 69#7 | ON | 3.52 |
| 70#34 | ON | 5.66 |

论文包含 macaque 与 baboon 的离体 parasol intracellular recordings；记录位置为距 fovea **11.4 ± 2.0 mm（mean ± SD，约 53°，n=52）**。因此 cell class 相符，但物种组成、离体条件与 eccentricity 不完全匹配。当前局部 grid 坐标不是 cell 的 retinal eccentricity，不能混用。[McMahon et al., Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6729348/)

## 2. 原尺度几何能否容纳

Macaque 换算随眼球与 eccentricity 改变。Dacey & Petersen 引用 Perry & Cowey 的测量，报告约 **0.223 mm/°（fovea）至 0.17 mm/°（远周边）**；Manookin 等的 primate 方法亦明确采用这些数量级，并在较周边采用 0.20 mm/°。下面使用该文献范围做可容纳性检查；它不是置信区间，也不是当前每只眼的精确标定。[Dacey & Petersen, 1992](https://daceyretinalab.org/wp-content/uploads/2020/12/dacey-petersen-1992.pdf)、[Manookin et al., 2018, Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC5866240/)

换算为 `diameter_deg = diameter_mm / (mm_per_deg)`，表中所有尺寸均为**直径**。

| 论文几何 | 按 0.223–0.17 mm/° 换算 | 0.20 mm/° 参考值 | 能否原尺度放入 0.916406° 输入场 |
|---|---:|---:|---|
| fixed spot：0.3 mm | 1.345–1.765° | 1.5° | 否，连最小 fixed spot 也超出 |
| annulus inner：0.4 mm | 1.794–2.353° | 2.0° | 否，内孔直径已超出输入宽度 |
| annulus outer：2 mm | 8.969–11.765° | 10.0° | 否 |
| area summation：0–2 mm | 从 0 延伸至 8.969–11.765° | 0–10.0° | 否，无法覆盖完整范围 |

上述判断已按输入场中央放置的有利情形计算。最大完整圆形刺激直径仅对应约 **0.156–0.204 mm**；偏离输入中央只会进一步减少可容纳范围。2 mm 刺激需要约当前宽度的 **9.8–12.8 倍**；即使恢复原电影 4.6° 视野也不足。具体眼球换算仍需外部确定，但在所查文献范围内不改变不相容的结论。

当前 external-geometry builder 会拒绝超出输入场边缘的圆，而非静默裁切。本轮不改 radius、不裁切 annulus、不按 support 或 response 缩放，也不把输出 size-tuning normalization 当作几何归一化。[几何检查](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/h1_external_geometry_assay.py:143)

## 3. Contrast、时序与响应读出

**Contrast 定义已核实，跨实验标定仍未建立。** 原文明确定义 contrast 为调制振幅除以平均光强，值为 25%，刺激在平均光强上下调制。若使用固定光谱并以同一平均光强作为零点，则 `L(t)=L̄[1+c(t)]`、`c=(L−L̄)/L̄`：方波对应 Weber 的 ±0.25，正弦对应 0.25 振幅。这是有条件的数学对应，不是本轮填写 stimulus 配置的授权。[McMahon et al., Visual stimuli](https://pmc.ncbi.nlm.nih.gov/articles/PMC6729348/)

当前输入则是 gamma 与 L/M 校准后的 `(L+M−background_LM)/background_LM`，背景来自原电影开头 blank。论文使用另一套显示光谱和 midphotopic 背景（白点 x=0.304、y=0.349，约 1000 trolands）。实验背景、光谱与适应条件到当前 L+M proxy 的严格注册尚未核对，记为 **UNRESOLVED（该标定证据 UNVERIFIED）**；不能把“相同无量纲振幅”写成完整刺激等价。这里缺的是跨实验映射，而非论文没有 contrast 定义。[当前前端公式](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:165)

| 项目 | 论文实际协议 | 当前 external-geometry 接口 | 判断 |
|---|---|---|---|
| fixed spot / annulus | 2 Hz 方波；通常平均 24 cycles | 单次恒定 contrast 的矩形 flash | 不匹配 |
| spatial / area-summation tuning | 2 Hz 正弦调制；每刺激 4 cycles | 不支持周期正弦、频率或相位 | 不匹配 |
| response | intracellular voltage 的 2 Hz Fourier 振幅；空间拟合还使用 phase | baseline-subtracted logit 的窗口均值 | 不匹配 |
| surround strength | 同时拟合多种空间 tuning 的 difference-of-Gaussians；积分得到 center/surround volumes，并将 center volume 归一为 1 | 参考刺激与 surround 刺激的 logit response 差除以参考 response 绝对值 | 不是同一指标 |

[论文 Methods / spatial tuning analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC6729348/)、[当前 stimulus 构建与 response 汇总](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/h1_external_geometry_assay.py:188)

150 Hz 在原则上可采样 2 Hz 正弦：每周期 75 bins。若方波采用对称半周期，每半周期为 37.5 bins，需要明确 phase、采样或 bin-average 规则，不能静默取整；当前只接受整 bin 时长的单 flash 接口没有这项定义。4 cycles 为 2 s，24 cycles 为 12 s，也不能用旧的一次性 150-bin flash 替代。当前固定 history 的 logit 以及有效归一化 membrane 均不能直接称作论文记录的生物膜电压。

## 4. Drug 与 BLOCK_H1_FEEDBACK 的语义边界

| 项目 | 论文 manipulation | 当前模型 operation / 边界 |
|---|---|---|
| 共享的弱假设 | 作者把 carbenoxolone/cobalt 的作用解释为干扰 horizontal-cell feedback，并观察 surround 相对 center 更易衰减 | 外层反馈影响下游空间拮抗，是可检查的功能假设；不证明两种操作等价 |
| 操作对象 | 药物作用于真实视网膜网络；存在剂量、暴露与洗脱过程 | feedback output 精确归零，H1 graph/state 保留；H1-modulated input 及 BC/AC/EI/RGC 重新计算 |
| 选择性 | 不是 H1 subtype 专属干预；论文亦报告较强/较久处理可广泛降低或消除 light response | 理想化的单计算输出 block，不含药物非特异作用、部分阻断或药代过程 |
| 可比边界 | surround/center 相对变化可约束功能方向 | 没有浓度→参数、残余反馈比例、mV 或受体阻断映射；不能称 pharmacological simulation |

[现有 intervention 定义](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/mechanism_observation.py:122)、[McMahon et al., Results / Discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC6729348/)

另外，原始 retinal 实验表明 carbenoxolone 可抑制 cone voltage-gated Ca channels 与 photoreceptor→horizontal-cell transmission，进一步限制“选择性 feedback 开关”的解释。该证据来自 salamander，不能转写为本 macaque cohort 的药效量或直接机制验证。[Vessey et al., 2004](https://pubmed.ncbi.nlm.nih.gov/15028741/)

## 5. 可以保留与不能直接比较的外部证据

**A. 可保留为 qualitative external physiology constraint：** 在论文的光适应、远周边 primate parasol 条件下，被作者解释为干扰 horizontal-cell feedback 的外层扰动，使 surround/annulus-related response 相对 center response 更强衰减。固定 spot/annulus 的观察可写作“药物下 annulus F1 的保留比例低于 spot F1 的保留比例”。这是最安全的下一阶段方向性约束；它允许 center response 也改变，不以 absolute off effect 代替 surround selectivity。[McMahon et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6729348/)

“Parasol surround 在上述操作下显著衰减”也可保留为带有这些条件的文献观察，但不能改写成所有 parasol、所有 eccentricity 均由 H1 主导，更不能由当前 computational block 推出药理机制。论文提供外部约束，不提供当前 H1 state 的直接观测验证。

**B. 不能直接比较的 quantitative results：** 原尺度 preferred diameter、surround radius、2 Hz response amplitude/phase、药物剂量—反应关系及 surround-strength ratio，均受几何、eccentricity、时序、读出或 intervention 不匹配限制。Contrast 公式的可换算性不能消除这些差异。

特别是 carbenoxolone 的 **0.37 ± 0.11（n=10）** 来自论文 DoG 拟合和 center-volume normalization 后的 surround strength 相对 predrug 比例；不是当前 logit SSI，也不是 BLOCK_H1_FEEDBACK 后的通用残余比例。**不得作为拟合目标、接受阈值或本模型应达到的效应量。**[McMahon et al., spatial tuning methods and carbenoxolone results](https://pmc.ncbi.nlm.nih.gov/articles/PMC6729348/)

## 最终回答

- **Exact protocol 是否能运行？** 不能；当前冻结输入与接口下为 **NOT_FEASIBLE_EXACT**，本轮没有运行 checkpoint。
- **最大的 protocol mismatch 是什么？** 空间视野不足：0.3 mm fixed spot 已放不下，2 mm surround 约需当前宽度的 9.8–12.8 倍；另有周期刺激与电压 F1 读出不匹配。
- **哪一个 external observation 可以安全用于下一阶段？** 上述论文条件下，外层扰动 preferentially weakens surround-related response relative to center response；只作为有条件的定性功能约束，不宣称 drug 与 H1 block 等价。
- **是否需要扩展 stimulus interface？** 若要忠实表达该协议，需要支持周期方波/正弦及明确的采样时序，并配套 F1 分析；但仅扩展 builder 不足以解决视野、模型有效电压与药理映射问题。当前冻结输入下仍不能 exact replication，本轮不作任何扩展。
