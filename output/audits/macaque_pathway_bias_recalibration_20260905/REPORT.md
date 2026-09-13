# Training-only bias recalibration：六问报告

## 1. Scalar bias fitting 是否正确？Normal sanity 如何？

**全部 88 个 scalar 求解通过**：22 cells × normal/H1-off/direct-BC-off/AC-off，仅使用原 training [0,16) s 的 263040 个 scored bins。确定性二分求根需 34–41 次迭代，最大绝对平均 event-rate residual 为 **9.97036×10⁻¹³**，低于固定 10⁻¹² tolerance；所有 training NLL 均降低。没有 optimizer、其他参数拟合或模型 state_dict 修改。

拟合前，22 cells 的 training tensors/source/trial order 与原产物完全一致；held-out [20,60) s 的 target/mask/order、657600 scored bins 及全部 88 组 normal/off logits 精确重放通过。拟合阶段只读独立的 training-only logits 文件；88 个 b 在任何 recalibrated test NLL 计算前完成 SHA256 冻结。见 [preflight.json](preflight.json)、[bias_fit_per_cell.csv](bias_fit_per_cell.csv)、[bias_fit_lock.json](bias_fit_lock.json)。

| Normal diagnostic | 数值 |
|---|---:|
| Mean / median fitted bias | 0.010307128 / 0.014531747 |
| Bias 范围 | [−0.115766775, 0.118051109] |
| Mean training NLL，before → after | 0.429456352 → 0.429249111 |
| Mean held-out NLL，before → after | 0.429874169 → 0.429830312 |
| Mean / median held-out ΔNLL | −0.000043857 / +0.000042002 |
| Held-out 改善 / 变差 / 平局 cells | 10 / 12 / 0 |

Normal bias 并非精确为零，但这里只观察到很小且 cell 间混合的 held-out calibration change，不存在普遍改善；未新增显著性检验。Normal recalibration 始终只是 diagnostic，所有 pathway 的 primary reference 均为原 frozen normal。完整逐 cell 数据见 [normal_bias_sanity.csv](normal_bias_sanity.csv)。

控制分析的 solver、raw/recalibrated NLL 统一用 float64，输入为同一 frozen float32 logits；上一轮 float32 NLL 另行精确复现。两种计算精度的 raw NLL 最大差为 7.64446×10⁻⁸，不构成模型变化。以下 NLL 单位均为 nats/scored bin，population 按 biological cell 等权。

## 2. H1 raw penalty 校准后还剩多少？

| Raw mean | Recalibrated mean | Recal mean 95% CI | Absolute reduction | Relative reduction |
|---:|---:|---|---:|---:|
| 0.002824355 | 0.002699809 | [0.001196823, 0.004484533] | 0.000124546 | 4.409711% |

Raw / recal median 为 **0.001091924 / 0.000738321**；recal median 95% CI 为 **[0.000219200, 0.004378322]**。Recal positive / negative / zero 为 **16 / 6 / 0**，负 cell 比原先 3 个增加到 6 个。Population residual 仍有正向证据，但 cell 支持未达到预先固定的至少 17/22，故判为 **MIXED**，保留 small / heterogeneous model-internal influence，而不声称普遍一致的 H1 quantitative contribution。

## 3. Direct-BC raw penalty 校准后还剩多少？

| Raw mean | Recalibrated mean | Recal mean 95% CI | Absolute reduction | Relative reduction |
|---:|---:|---|---:|---:|
| 0.305192558 | 0.304053456 | [0.260424642, 0.347131961] | 0.001139102 | 0.373240% |

Raw / recal median 为 **0.280744937 / 0.293847840**；recal median 95% CI 为 **[0.262539067, 0.331477691]**。Recal positive / negative / zero 为 **22 / 0 / 0**。按 population mean 比率，**99.626760%** 的 penalty 保留，判为 **STIMULUS-DEPENDENT PREDICTIVE CONTRIBUTION SUPPORTED**。

## 4. AC raw penalty 校准后还剩多少？

| Raw mean | Recalibrated mean | Recal mean 95% CI | Absolute reduction | Relative reduction |
|---:|---:|---|---:|---:|
| 0.226704781 | 0.195979707 | [0.152843262, 0.241785281] | 0.030725074 | 13.552901% |

Raw / recal median 为 **0.199035254 / 0.160896517**；recal median 95% CI 为 **[0.150379999, 0.201139471]**。Recal positive / negative / zero 为 **22 / 0 / 0**。按 population mean 比率，**86.447099%** 的 penalty 保留，判为 **STIMULUS-DEPENDENT PREDICTIVE CONTRIBUTION SUPPORTED**。

## 5. 三条 pathway 分别属于哪个 verdict？

| Pathway | 预注册 verdict |
|---|---|
| H1 | MIXED |
| direct-BC | STIMULUS-DEPENDENT PREDICTIVE CONTRIBUTION SUPPORTED |
| AC | STIMULUS-DEPENDENT PREDICTIVE CONTRIBUTION SUPPORTED |

没有 pathway 被判为 OPERATING-POINT DOMINATED 或 NO POSITIVE RESIDUAL SUPPORT。所有区间来自固定 seed **2026090502**、**100000** 次 paired-cell percentile bootstrap，无 p-value。至少 17/22 的“广泛支持”及 50% 的多数/少数 penalty removal 规则均在结果前冻结，仅用于标签的操作性判定，不是显著性检验。

| Pathway | Per-cell \|mean Δlogit\| 的中位数 | Per-cell centered Δlogit std 的中位数 |
|---|---:|---:|
| H1 | 0.008585901 | 0.082331009 |
| direct-BC | 0.869794047 | 2.428580915 |
| AC | 0.734208148 | 1.954627657 |

Centered std 使用 ddof=0，并与 centered RMS 一同逐 cell 保存于 [stimulus_dependent_logit_components.csv](stimulus_dependent_logit_components.csv)。这些描述性残差未参与或改变 verdict；它们是在当前 stimulus 与 observed-history convention 下的非均匀 logit variation，不能单独识别 stimulus-only causal effect。

逐 cell R 只在 raw ΔNLL>0 时定义，其他行为留空；未 clip 到 [0,1]。Population relative reduction 是 **1−mean(recal Δ)/mean(raw Δ)**，不是逐 cell R 的平均，更不是 causal variance explained。完整数值见 [pathway_bias_recalibration_per_cell.csv](pathway_bias_recalibration_per_cell.csv) 和 [pathway_bias_recalibration_population.csv](pathway_bias_recalibration_population.csv)。

## 6. 是否改变上一轮 pathway claim 的解释边界？

**细化了模型内解释，但没有提高证据等级。** Direct-BC 和 AC 的大部分 frozen intervention penalty 不能由一个 training-derived global logit/rate shift 解释；AC 存在较 direct-BC 更明显的 calibration component。H1 仍有小幅正向 population residual，但异质性增加，因此应避免把上一轮正向平均效应表述为广泛一致的 cell-level contribution。

本轮是 **predefined follow-up analysis on an already consumed confirmatory set**，不是新的 untouched test。Bias 完全由原 training split 拟合，held-out targets 未参与 estimation、selection 或 calibration。模型、bias 以外所有量、前端、中心、history、reset、mask、frame-zero 均保持原定义；frame-zero acquisition provenance 仍为 UNKNOWN。原 held-out artifacts 没有覆盖。

“不能仅由 global shift 解释”只针对本次一个 scalar、原 training-derived 的补偿控制；不支持 biological necessity、unique causal contribution、retrained structural ablation 或 real-retina lesion effect，也不是对其他可能 compensation 的排除。协议 SHA256 为 `740349691eb4247f2ab1990531b802aca2c57a2e954f35948dbb5913dafecfe2`；源文件、logits、targets/masks、exact source IDs 与拟合锁的追溯信息见 [evidence_manifest.json](evidence_manifest.json)。

论文级总结：在当前 frozen aligned model 中，direct-BC 与 AC removal 的大部分 held-out predictive penalty 无法由 training-only scalar firing-rate recalibration 解释；H1 保留较小且 cell 间异质的 residual evidence。
