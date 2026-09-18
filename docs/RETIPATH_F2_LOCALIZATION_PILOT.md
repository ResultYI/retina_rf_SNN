# RetiPath frozen contrast-reversing-grating F2 localization pilot

**2026-09-16 · Verdict：F2_FIRST_APPEARS_AT_BC**

仅运行 67#6（OFF parasol）与 67#7（ON parasol），seed 均为 2026091301，使用既有 canonicalized frozen checkpoints 的 float64 inference copy。共 2 个 checkpoint、14 条序列；无训练、checkpoint selection、参数修改、pathway clamp、DoG/radius fitting、FOV/support/preprocessing 修改或新 natural-movie/spike target 读取。

## 冻结刺激、Fourier 定义与可比范围

protocol 和 stimuli 在任何模型 response 前保存。17×17 输入使用竖直 square-bar grating，bar width 固定为 1、2、4 pixels；phase 固定为 0 和 half-bar shift。空间坐标锚定左侧输入列，不按 cell center 或 response 搜索相位。

对列 j=0,…,16，令 `b_j=+1` 当 `floor((j-phase*width)/width)` 为偶数，否则 −1；边界采用半开区间。17 列含有不等数量的正负样本，故先在全部 289 pixels 上减去 b 的均值，再除以最大绝对值。最终两级是 +8/9 与 −1，或 +1 与 −8/9；保留条带宽度和相位位置。无 clipping、额外 aperture、support mask 或响应筛选。**0.5 定义为相对全局背景的最大绝对 Weber amplitude，不是 RMS，也不等同于光暗条带 Michelson contrast。**这项有限网格平衡方式是本轮模型刺激定义，不宣称逐点复现文献光栅。

`X[n,p]=0.5*q[p]*sin(2*pi*4*n/150)`；600 bins 覆盖 16 cycles。前 300 bins（8 cycles / 2 s）pre-roll；后 [300,600) 的 300 bins（8 cycles / 2 s）measurement。单周期为 37.5 bins，不取整、不重复周期端点。保持 FIX_HISTORY_ZERO 与默认动态初态：V1/V2=V0=2/9，其余动态偏差为零。uniform full-field control 为 q=1，具有同一幅度、时间波形和时长。grating 每帧空间均值为零；uniform control 的时间均值为零。

实际全部 pixel contrast 范围为 [-0.499890341737, 0.499890341737] Weber，始终高于 −1。最大 grating 每帧空间均值绝对值为 7.01e-17。1-pixel width 位于空间 Nyquist：本次 rasterization 的两个相位互为符号翻转，不能当成两个独立空间采样。未新增相位补救。

对每个原始 component p，在 measurement window 内先计算复数系数：

`C_h,p = (2/300) sum_n [Y_n,p - mean_n(Y_n,p)] exp(-i 2*pi*(4h)*t_n)`，h=1,2。

然后计算 `F_h,p=|C_h,p|`，固定归约 `A_h=RMS_p(F_h,p)`，primary 为 `A_2(grating)/A_1(uniform)`。H1 使用全部 289 pixels；BC/AC sustained/transient 或 local/transient 分支分别保留，归约 N,K；E/I 与 conductance 归约 N,K；其余为单 cell 标量。绝未在 Fourier 前对 trace 取 abs/norm。分子、分母始终来自相同 cell、trace 和 component 集合；数值大小不是 biological contribution。

`harmonics.csv` 同时包含原始 component 的复数 C1/C2、F1/F2、uniform F1、ratio，以及单独标记的 RMS 汇总行。`traces.npz` 保存全部时间 trace、逐 component 复数 harmonics 和 metadata。BC raw shape 为 [7,600,1,2,2]，末轴 sustained/transient；AC 末轴 local/transient；H1 为 [7,600,289]；E/I、gE/gI 为 [7,600,1,2]；V1/V2、membrane、adaptation、logit、probability 为 [7,600,1]。所有量仍是模型有效量，不校准为 mV/nS。

roundoff reference 在 protocol 中固定为每 component 的 `4096*float64_eps*max_abs(measurement trace)`，group 使用其 RMS。这是数值舍入参照，不是统计显著性或生理效应阈值；uniform F1 未高于自身参照时须标记 ratio 未分辨。本次没有无效分母。固定 8-cycle pre-roll 不自动证明所有下游状态完全达到 steady state，结果属于这一固定 window。

必要检查通过：已知 synthetic F1/F2 的复系数最大误差为 1.56e-14；相反符号 component 不产生人工 F2；实际输入 F2 仅为约 2.42–2.56e-15。所有模型 forward 有限，history state 为零；inference state_dict、checkpoint SHA256、manifest 与模型/trace 源码保持不变。保存后从原始 trace 独立重算 Fourier 和归约，结果一致。

[输出目录](D:/PythonProject/retina_rf_SNN/output/evaluations/retipath_f2_localization_pilot_20260916) · [分析脚本](D:/PythonProject/retina_rf_SNN/work/retipath_f2_localization_pilot.py)。Protocol SHA256：`bf12b7ccf90086eabb3389340f87d22fd3a626e734d9b542c20abbb5d80e2cf7`。

## A. H1-like trace 是否产生可分辨 F2？

没有。在全部 12 个 cell×grating 条件中，H1 graph/state/feedback/modulated input 的原始 components 和固定 RMS 归约均处于 roundoff reference 以下。H1 的 graph、delay、lowpass、feedback subtraction 在固定参数下是线性操作；当前结果与其实现一致。

下表为六个 grating 条件中最大的 `F2 / uniform F1`：

| H1 trace | 67#6 | 67#7 |
|---|---:|---:|
| h1_graph_drive | 2.729e-15 | 2.729e-15 |
| h1_state | 2.040e-15 | 1.401e-15 |
| h1_feedback | 1.473e-15 | 1.010e-15 |
| h1_modulated_input | 5.651e-15 | 5.534e-15 |

H1 state 的最大原生 F2 amplitude 分别为 9.85e-16、6.70e-16；对应最大 group roundoff reference 约为 2.34e-13、2.31e-13。H1 feedback 更小。该“未分辨”限于此模型和本次输入，不等于真实 horizontal cells 不产生 F2。

## B. BC stage 是否产生 F2？

是。两个 cell、六个固定 grating 条件、四条 BC 分支全部出现超过数值 floor 的 F2（48/48 个分支×条件组合）。范围为六个条件的最小–最大 `F2 / uniform F1`，不是置信区间：

| BC branch | 67#6 | 67#7 |
|---|---:|---:|
| bc_direct_effective_drive.sustained | 0.044814–0.316096 | 0.003819–0.064866 |
| bc_direct_effective_drive.transient | 0.044875–0.316011 | 0.003816–0.064904 |
| bc_broad_effective_drive.sustained | 0.008572–0.240983 | 0.003584–0.048096 |
| bc_broad_effective_drive.transient | 0.008780–0.242124 | 0.003581–0.048050 |

源码中的第一项 input-dependent 非线性为 [BC common-branch PReLU](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/retipath_canonical_gain.py:133)。alpha 固定为 67#6=0.200417232、67#7=1.366462539；未做 alpha intervention。整条 BC 分支的正负不对称变换可产生偶次 harmonic；模型没有因本轮实验新增 release state 或真实 bipolar subtype。

## C. Downstream F2 如何传播到 E/I 和 RGC？

BC broad 的 F2 经 AC delay/filter 传播；canonical effective E/I 均保留可分辨 F2。softplus conductance mapping、voltage dynamics 和后续输出变换进一步改变 amplitude 与 phase。membrane、logit、probability 也均有 F2。以下逐条件列出 primary ratio；没有按 response 排除任何 width/phase。

| Cell | Width px | Phase/bar | effective E | effective I | membrane | logit |
|---|---:|---:|---:|---:|---:|---:|
| 67#6 | 1 | 0 | 0.069751 | 0.007371 | 0.094694 | 0.094821 |
| 67#6 | 1 | 0.5 | 0.069751 | 0.007371 | 0.094694 | 0.094821 |
| 67#6 | 2 | 0 | 0.078858 | 0.010767 | 0.102890 | 0.103028 |
| 67#6 | 2 | 0.5 | 0.103762 | 0.025518 | 0.133393 | 0.133572 |
| 67#6 | 4 | 0 | 0.314826 | 0.203060 | 0.462634 | 0.463255 |
| 67#6 | 4 | 0.5 | 0.044707 | 0.013840 | 0.062085 | 0.062169 |
| 67#7 | 1 | 0 | 0.003800 | 0.002981 | 0.006778 | 0.006787 |
| 67#7 | 1 | 0.5 | 0.003800 | 0.002981 | 0.006778 | 0.006787 |
| 67#7 | 2 | 0 | 0.005088 | 0.003822 | 0.008883 | 0.008895 |
| 67#7 | 2 | 0.5 | 0.030748 | 0.008653 | 0.056984 | 0.057060 |
| 67#7 | 4 | 0 | 0.064618 | 0.039996 | 0.146951 | 0.147148 |
| 67#7 | 4 | 0.5 | 0.005043 | 0.004586 | 0.009543 | 0.009556 |

幅度依赖 width/phase；本轮不将最大响应相位用于重新选择条件。各 trace 的 denominator 不同，因此不能把 ratio 的跨层增减解释成某通路的 biological contribution，也不能据此完成因果分解。probability 的 sigmoid 本身还能贡献 harmonic，不能用它替代 logit 或 upstream trace 定位。

## D. First clearly nonzero F2 stage 在哪里？

**BC direct/broad 非对称变换之后；两个 checkpoint 的 verdict 均为 F2_FIRST_APPEARS_AT_BC。** H1-like 层没有超出数值 floor 的 F2，BC 四分支则明确出现。本次 grating 并未针对每个 cell 构造 RF-weighted spatial null；有效分支先空间汇聚、后 rectification 就能形成 F2。因此这一定位不证明真正的细粒度 BC subunits，也不是 phase-invariant Y-cell signature 的正式验证。

外部 observable 对齐仅到“contrast-reversing grating 的 F2，以及相对 uniform F1 的归一化”：primate parasol 的既有实验使用过 4 Hz、50% contrast 的 F2 观测（[Crook et al., 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2778053/)）；grating F2 / uniform spot F1 的量也用于已有 primate 研究（[Yu et al., 2022, Figure 1](https://elifesciences.org/articles/70611/figures)）。本次 grid、balanced rasterization、effective latent state 和 fixed history 均不等同于真实膜电位/突触电流的记录合同。

Raval、Oaks-Leaf、Chen 与 Rieke 的 [2026 preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC13041967/) 报告 horizontal/bipolar 的 nonlinear spatial responses；其 Methods 的 eccentricity 为 20–50°（按 cone density 判断）。本次 catalog eccentricity 为 6.72° 和 7.36°，并且未校准相同 retinal illumination、空间尺度或生理单位。这些差异保留，不设置外部拟合或接受阈值。

**这只定位当前模型的 nonlinear response stage。当前 H1-like stage 的 F2 仅在数值 floor，可以与 Raval et al. 2026 primate horizontal-cell 结果形成一个候选 physiological mismatch；由于该外部工作是 preprint 且 eccentricity 不同，暂不据此自动修改架构。**
