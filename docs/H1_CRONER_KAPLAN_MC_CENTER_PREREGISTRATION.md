# Croner–Kaplan MC center：measurement / comparison preregistration

日期：2026-09-15。状态：**COMPARISON_FROZEN / MEASUREMENT_STOPPED_BY_USER**。

本轮未加载 checkpoint，未读取模型 RF、拟合 radius、assay response、natural movie 或 spike target；未训练、未修改模型、FOV、support、preprocessing 或已有 assay。已冻结外部 reference 和 comparison metric；analysis-only complex fitter 仅作为通过 synthetic 检查的候选工具保留，未构成生效的 measurement contract。**目前不能把本文件称为已可直接执行的完整 9-cell measurement protocol**：尚缺明确的 Jacobian operating point。下文严格区分已冻结项目与待决定项目；`protocol.json` 的 `ready_for_model_measurement=false`。

## 1. 外部语义和来源等级

| 项目 | 核实到的内容 | 状态 / 边界 |
|---|---|---|
| Croner–Kaplan 1995 原文 Methods | 本轮仍未取得可直接核对 Methods 的原文；出版商 / 原 PDF 路径未提供可读正文 | 原文 temporal frequency、F1 归一化和拟合细节：UNVERIFIED。不能将未获取正文写成论文未说明 |
| 4 Hz | McCann, Hayhoe & Geisler 2011 的 Methods，Simple model of ganglion cell responses，明确将 Croner–Kaplan 的 surround-strength 测量描述为 4 Hz drifting sine-wave gratings | **SECONDARY_SUPPORTED**；该段讨论 P-cell 建模，不能据此声称已逐项核实 1995 所有 M-cell 的实验设置。固定拟议投影频率为 4 Hz，不挑其他频率 |
| Spatial-frequency / F1 / DoG | Kremers et al. 2001 将响应量定义为 PSTH Fourier first-harmonic amplitude，并明确将含 contrast 的 DoG spatial-frequency 方程归于 Croner–Kaplan 1995 | 对 1995 属二手方法支持；不声称与本轮 complex spatial-coefficient objective 完全相同 |
| 未核实刺激设置 | 1995 contrast、aperture、光学矫正及完整 luminance/adaptation 设置 | 均保持 UNVERIFIED，不填推测值；本轮 synthetic fixture 的数值不是论文刺激参数 |

来源：[Croner–Kaplan 1995](https://doi.org/10.1016/0042-6989(94)E0066-T)、[McCann et al. 2011 Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC3214635/#S2)、[Kremers et al. 2001](https://journals.physiology.org/doi/10.1152/jn.2001.85.1.235)。所述二手 DoG 为

\[
R(\nu)=C\{K_c\pi r_c^2 e^{-(\pi r_c\nu)^2}-K_s\pi r_s^2 e^{-(\pi r_s\nu)^2}\}.
\]

这里 spatial frequency \(\nu\) 与 temporal frequency \(f\) 不混用；\(r_c\) 是空间 Gaussian 降至峰值 1/e 的 characteristic radius。没有把 finite-contrast spike F1 与 logit Jacobian 当作同一观测量。

### Eccentricity

- 已直接读取 [Schottdorf–Lee 作者 repository README 的 Cell/Recording list](https://gin.g-node.org/Manuel/Macaque-ganglion-cells/src/master/README.md)，9 个 MC 的 cell ID、class、eccentricity 数值与本地 catalog 及冻结 reference target table 一致。README 字段名是 `Eccentricity`，未定义 nasal/temporal 校正、eye 或二维坐标。两个原始 cell-list 文档也未补齐这一定义；没有读取 spike 文件。
- [原论文 Schottdorf–Lee 2021](https://doi.org/10.1113/JP281200) 及其公开记录确认来源关系；可取得的 paper 内容未建立该 catalog scalar 与 temporal-equivalent 坐标的等价性。不能据此认定二者已对齐。
- [Croner, Purpura & Kaplan 1993 Fig. 3](https://www.cns.nyu.edu/~tony/vns/readings/croner-purpura-kaplan-1993.pdf) 明确描述 nasal 分量乘 0.61 后结合另一分量得到 temporal-equivalent eccentricity。这是同实验室较早论文的直接说明，**不是已经直接核实的 1995 Methods**。
- 固定 ISETBio commit `d1619948425056efa5435c91932bca5b3a30f4a7` 的 [转换函数](https://github.com/isetbio/isetbio/blob/d1619948425056efa5435c91932bca5b3a30f4a7/ganglioncells/@mRGCMosaic/temporalEquivalentEccentricityForEccXYDegs.m) 要求二维坐标和 eye，区分 nasal 侧，WatanabeRodieckBased 分支乘 0.61，另有 ISETBioMosaicsBased 0.70 分支；该函数返回变换后的二维坐标。其 [Croner 原始数据接口](https://github.com/isetbio/isetbio/blob/d1619948425056efa5435c91932bca5b3a30f4a7/ganglioncells/%2BRGCmodels/%2BCronerKaplan/individualMonkeyData.m#L102-L112) 使用 `temporalEquivalentEcc` 名称，但不提供当前 macaque cohort 的转换信息。

**最终状态：UNRESOLVED。** 冻结采用 catalog 数字原样查已有 reference，不转换、不按 hemisphere 猜测、不重新拟合 reference。这允许明确标记为 **numerical eccentricity-conditioned comparison**；不足以支持严格 retinal-location-matched replication。没有二维坐标就无法量化每个 cell 的坐标误差，也不能保证它小于 external scatter；因此未来 systematic bias 不能全部归因于模型。该语义缺口本身不阻止受限的数值比较，但限制其生理结论。

## 2. 现有 Jacobian 的实际边界

[`effective_rf()`](../evaluation/mechanistic_retina/rf_effective.py:9) 求最后一个 logit 对整个 stimulus 的梯度，但在返回前执行 `gradient[:, -model.config.lag_steps:]`。冻结 Phase 2 配置为 `lag_steps=16`、`dt_ms=6.666666666666667`，即返回的 ages 为 15…0，最大 lag 为 100 ms。该接口**不能提供本任务要求的完整 causal-prefix Jacobian**。

这不是“不足一个 4 Hz 周期就数学上不能计算 Fourier 和”：短 FIR 也可以在任何频率计算 transfer。问题是这里丢弃了已存在的更早梯度，且 [conductance recurrence](../models/mechanistic_retina/retipath_spatial_ei.py:29) 与 [adaptation low-pass](../models/mechanistic_retina/state.py:22) 并非天然 16-bin 截断，不能把丢掉的尾部假设为零。增加传入 sequence 长度或补零不会恢复被切掉的梯度；也不得修改 model.config.lag_steps 来绕过，因为它还控制 BC temporal bank。

已有 [Phase 2 `jacobian_modes()`](../work/retipath_phase2_common.py:139) 返回完整、保留符号的最后 logit Jacobian，未作该切片；其 full tensor 才是可复用的候选来源。canonical 模型仍继承相同 forward/integrator 调用结构；本轮仅源码核对，未运行任何正式模型确认其运行结果。本次不改旧 `effective_rf()`、不读历史 RF cache、不运行 Phase 2 分析脚本。

### 候选 measurement 数学定义（未生效）

对一个已固定 operating point 和完整 causal prefix，按实际时间顺序取得
\(J_t=\partial\ell_{T-1}/\partial u_t\)，其中 \(u_t\) 是 17×17 L+M Weber bin 值、\(\ell\) 是 logit。所有 state 对输入的依赖保留，history 作为固定条件。禁止使用 probability 导数、\(J^2\)、normalized \(P\) 或任何 mode/support radius 替代。

\[
H_4(x,y)=\sum_{t=0}^{T-1}J_t(x,y)\exp[-i2\pi\,4\,(T-1-t)/150].
\]

返回 tensor 的时间顺序是 oldest→newest；lag age 顺序相反。\(dt=1/150\) **秒**。离散 bin-value Jacobian 不再乘 dt、不做 FFT 的 1/T normalization、不去掉符号或相位；不窗化、不空间裁剪、不以补零冒充完整时间核。

Fitter 使用原 [synthetic audit 的 pixel-integral Gaussian](../work/croner_kaplan_center_identifiability.py:33)，没有复制另一套近似 operator：

\[
G_{pixel}(r)_{ij}=\int_{x_i-p/2}^{x_i+p/2}\int_{y_j-p/2}^{y_j+p/2}
\frac{e^{-((x-x_0)^2+(y-y_0)^2)/r^2}}{\pi r^2}\,dy\,dx,
\quad p=0.05390625^\circ.
\]

grid centers 为 \((i-8)p\)，edge-to-edge 0.91640625°。固定 center 来自该 cell 的 frozen checkpoint geometry，下一轮只能读取并使用，禁止响应搜索。有限 FOV 内不把 Gaussian 重新归一化为质量 1；完整 17×17 的实部、虚部以相同权重联合拟合

\[
H_4=A_cG_{pixel}(r_c)-A_sG_{pixel}(r_s),\quad A_c,A_s\in\mathbb C,\quad 0<r_c<r_s.
\]

\(A_c,A_s\) 是独立 complex integrated gains，不施加固定相位或正实数约束；1/(πr²) 的规范化吸收于 gains，不改变 1/e radius 定义。使用 variable projection：给定两半径，以 complex linear least squares 精确消去两个 gains，再对实数 `log(rc), log(rs-rc)` 做联合实虚残差最小二乘。

固定 solver 设置沿用前审计的半径范围：rc∈[0.005,0.4]°，rs−rc∈[0.001,8]°；三个 (rc,rs) starts=(0.02,0.15)、(0.06,0.6)、(0.16,2)°；每 start 最多 250 evaluations，ftol=xtol=gtol=1e−11，float64。选全部 starts 中最小 SSE，精确并列保留最先项，不追加 starts 或扩大 bounds。边界不是生理接受阈值。保存所有 start 结果、active bounds、相对残差、条件数和局部 Jacobian singular values；不依据这些结果改定义或择优删除 cells。

Primary 只用 rc；rs、Ac、As 仅作拟合 diagnostic。MC center 的既有 synthetic 可识别性并不保证任意非 DoG 的真实模型 kernel 都能良好拟合；不能将 optimizer success 当作生理有效性。

### Operating point：用户已要求停止 measurement 冻结

用户原请求未指定计算 Jacobian 时的 stimulus operating point、prefix duration、取样时间或 history 值。它们会改变非线性模型的 rc，不能等看到 RF 后再挑。

已提出但未获采纳的方案：0 Weber 均匀背景、FIX_HISTORY_ZERO、150-bin 完整 prefix，取最后 logit，使用原 sequence 初态。这是有限前缀的局部敏感度，不宣称 steady-state 的无限时间 transfer 或 finite-contrast F1。尤其 canonical BC 使用 `pooled >= 0` 选择非对称斜率；在全零输入处，autodiff 采用代码的零点分支，并不保证存在方向无关的真实线性导数。**用户明确选择“停止 measurement 冻结，先解决 operating point”。该方案未采纳，不作为 measurement contract 生效；不以等待时间代替同意。**

若要的是严格 stationary 4 Hz F1-equivalent 量，而非这种局部条件量，目前 protocol 证据不足，必须 STOP。禁止通过修改模型、alpha、时间参数或调用截断接口绕过。

## 3. 已冻结 comparison contract

目标仅为 **67#6、67#7、67#33、68#3、68#10、69#4、69#6、69#7、70#34**。canonical fresh-refit seeds 固定为 **2026091301、2026091302、2026091303**；仅引用 migration manifest 中对应 27 个 checkpoint 的路径和既有 hash，本轮未读取其内容。

外部 reference 仍为已冻结的 ISETBio project radius-linear OLS：

\[
r_{ref}(e)=0.03254444445050388+0.011249762233383602e\quad(^\circ).
\]

**它不是 log-linear fit**；不得将这两个参数代入 log(rc)=a+be。`reference_fit.json` SHA256：`a048e069f0a9a03bc996895f336586d90ffd21f35fb3f71919225fcd3647e5eb`。全部 36 external points 和已有 scatter range 保持原样。

每 seed 单独拟合；cell 内仅作 log 尺度等权平均：

\[
\log r_{model,i}=\tfrac13\sum_{s=1}^3\log r_{c,i,s},\quad
d_i=\log r_{model,i}-\log r_{ref}(e_i),\quad ratio_i=e^{d_i}.
\]

\[
B=\tfrac19\sum_i d_i,\qquad E=\operatorname{median}_i|d_i|,\qquad Q=E/0.140653.
\]

Q 分母严格使用用户固定的 **0.140653**，不悄悄替换为 JSON 中未舍入的 0.1406525852029803。两者差别仅作 provenance 记录。

- 同时保存每 seed rc、cell geometric mean、di、exp(di)，不平均模型参数或概率，不用 arithmetic mean radius 替代。
- Cell bootstrap：NumPy PCG64 seed **19950915**，10000 次；按上述固定 9-cell 顺序，有放回抽 9 个 cell；每个 cell 的 3-seed 汇总值整体保留。B 的 95% percentile CI；E/Q 同样计算，但标记 descriptive CI。quantile 使用 linear interpolation。Q 的 CI 就是 E 的 CI 除固定分母。
- Reference 曲线和 scatter 在该 bootstrap 中固定，不重新拟合 external points、不将 external bootstrap 混为额外 biological cells。此为 cell-level 而非 animal-level CI。
- Secondary count：cell geometric mean radius 是否落入 `reference_fit.json.targets` 中已保存的 `prediction_rc_range95_lower_deg` / `upper_deg`，边界包含，报告 k/9；不是较窄的 reference-mean CI。不得重算另一个更有利 range。
- 必须齐全 9×3 estimates；缺失、nonfinite、zero transfer 或数值失败时保留所有记录，完整 cohort 的 primary summary 标记 UNVERIFIED，不能删除失败 cell 后继续声称 9-cell result。对可算但撞 bounds 的 radius 保留数值并明确其边界依赖，不据此宣称无偏生理估计。

### 解释固定，不设置 PASS 阈值

B 描述 systematic log-scale bias，exp(B) 可辅助读为总体几何比例；B 的 CI 跨零只表示未分辨 systematic bias，不证明等效。E 描述 typical absolute log mismatch。Q<1 仅表示该典型 mismatch 小于冻结 external LOO typical scatter，不自动构成 biological validation。External residual 同时包含生物差异、测量/数字化误差和 reference misspecification；本轮不将其分离成纯 biological scatter。

未来有效测量的 CONSISTENT / MIXED / NOT_SUPPORTED 必须逐项解释 B、E、Q、其区间及 external scatter，不以其他 metric 取代；没有自动 PASS classifier，没有新 effect threshold。Secondary range count 不得推翻 primary B/E/Q。语义或测量失败应报告 UNVERIFIED，不能冒充模型得到 NOT_SUPPORTED。

## 4. 本轮 verification 和交付状态

独立 [complex fitter / temporal projection](../work/croner_kaplan_mc_center_measurement.py) 仅导入 analysis libraries 和已有 synthetic Gaussian operator。三个 synthetic rc=0.052、0.100、0.195°，包括 subpixel center、不同 surround nuisance、非同相 Ac/As。4 Hz complex projection 最大误差 **1.0007415106216804e−16**，rc 最大 relative error **2.1094237467877974e−15**；rs/gains 也通过已知真值检查。Signed delayed impulse 的相位检查通过；16-bin tensor 冒充完整 150-bin prefix 时被拒绝。此为 algebra / data-structure smoke test，不是新 physiology assay，也不是 complex fitter 的完整 noise-identifiability 证明。

Machine-readable record：[protocol.json](../output/evaluations/croner_kaplan_mc_center_preregistration_20260915/protocol.json)，保存 source hashes、拟合设置、synthetic evidence、cohort/seed 清单、comparison formulas 和阻止执行的状态。旧 effective_rf、full-prefix helper、pixel-integral operator 和正式模型均未改动。

## 最终答复

1. **4 Hz 是否得到原文直接支持？** 尚未；目前采用 McCann 2011 的二手方法支持，原文 Methods 为 UNVERIFIED。
2. **Eccentricity 语义是否解决？** 没有，UNRESOLVED；数字逐项匹配作者 metadata，但 temporal-equivalent 等价性未建立。允许受限数值 lookup，不允许严格位置匹配的生理主张。
3. **最终 model-side rc 定义？** 候选为 full signed-Jacobian → 4 Hz complex projection → fixed-center pixel-integral complex DoG；仅完成 synthetic 工具检查。按用户明确决定，measurement 冻结已停止，无已生效的最终 model-side rc 定义。
4. **B/E/Q 合同？** 三 seed 先取 cell 内 geometric mean；di=ln(model/reference)，B=mean(di)，E=median(|di|)，Q=E/0.140653；固定 10000 次 cell bootstrap，external 95% range 只作 secondary。
5. **下一轮还存在阻碍？** 有：用户要求先解决 operating point / prefix / history 及其局部导数解释，并使用完整前缀来源；本轮不自行替换方案。当前 `ready_for_model_measurement=false`；不能直接运行正式 9-cell measurement，也未自动修补或运行任何模型。
