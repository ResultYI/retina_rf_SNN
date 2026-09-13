# Held-out natural-movie evaluation：最终报告

## 1. 是否找到 previously-unused 的 natural-movie range？

**是，按本轮 repository 内“未发现模型开发消费证据”的预注册定义，live [20,60) s 合格。** 对应 live frames [3000,9000)、decoded movie frames [3751,9751)，均为零起点、左闭右开。覆盖全部 22 cells、37 recordings、137 recording-local trials，共 5480 个 1 s sequences、657600 个 scored bins。

实际 loader、保存的 train/validation source IDs、88 份 validation target/mask/order 与 checkpoint 元数据确认：既有训练范围为 [0,16) s，development validation 为 [16,20) s。两次互补访问扫描、历史临时证据及两份归档 payload 中未发现 20 s 后的模型开发消费记录；完整时间戳与 onset 检查属于 METADATA_ONLY。该结论不证明 repository 外未记录的操作从未发生。重复 trial 使用同一段一分钟 movie，保持原 parser 的 recording-kind 修正及 trial identity；live origin 751 的 acquisition provenance 仍为 UNKNOWN。

证据见 [DATA_USAGE_REPORT.md](DATA_USAGE_REPORT.md)、[data_usage_inventory.csv](data_usage_inventory.csv)、[production_source_ranges.csv](production_source_ranges.csv)、[test_source_ranges.csv](test_source_ranges.csv)。协议在任何新模型输出前冻结；[preflight.json](preflight.json) 的 88 个 frozen checkpoint strict-load 与 NLL 精确复现全部通过，两个 Canonical 系列的 logits 均逐位相等。生产代码、数据、模型参数和 checkpoint 均未修改；无训练、调参或新模型 seed。

## 2. Aligned vs zero 的 held-out ΔNLL、95% CI、win count 是什么？

ΔNLL = aligned − zero；单位为 nats/scored bin，population 按 22 个 biological cells 等权。

| Mean ΔNLL | Mean 95% CI | Median ΔNLL | Median 95% CI | Aligned wins / zero wins / exact ties |
|---:|---|---:|---|---|
| −0.016701797 | [−0.026637087, −0.008820621] | −0.010585919 | [−0.019634902, −0.005256385] | 19 / 3 / 0 |

区间来自 100000 次 paired-cell percentile bootstrap，固定统计 seed 2026090501。它描述当前 cell 样本的不确定性，不构成独立动物或独立 movie 样本的区间。全部逐 cell 数值及完整精度见 [per_cell_test_nll.csv](per_cell_test_nll.csv) 和 [paired_model_comparisons.csv](paired_model_comparisons.csv)。

## 3. Frozen LN offset 与 held-out improvement 的相关性是什么？

I = zero NLL − aligned NLL，offset 使用先前冻结的 LN radial offset，没有重新拟合 center。

| 预注册分析 | Cells | Mean improvement | Pearson | Spearman |
|---|---:|---:|---:|---:|
| 全部 cells | 22 | 0.016701797 | 0.871005211 | 0.786561265 |
| 仅删除最大 held-out \|I\| 的 68#10 | 21 | 0.012998239 | 0.728782902 | 0.754545455 |

这是唯一一次删除诊断；没有循环排除其他 cells。Scatter-source 数据见 [alignment_heldout_relation.csv](alignment_heldout_relation.csv)。

## 4. Aligned Canonical 与 LN/CNN 在 held-out 上是什么关系？

| 模型 | Equal-cell mean NLL |
|---|---:|
| Zero Canonical | 0.446575975 |
| Aligned Canonical | 0.429874178 |
| Center-surround LN | 0.431272084 |
| Compact causal CNN | 0.436693755 |
| Analytical Constant（同一 test target 拟合，仅作描述参考） | 0.518257059 |

| 配对比较 | Mean ΔNLL | Mean 95% CI | Median ΔNLL | Median 95% CI | Aligned wins / comparator wins / ties |
|---|---:|---|---:|---|---|
| Aligned − LN | −0.001397906 | [−0.004288871, 0.001540345] | −0.003029436 | [−0.004601896, 0.002805144] | 13 / 9 / 0 |
| Aligned − CNN | −0.006819577 | [−0.014805056, 0.001462987] | −0.010786444 | [−0.016153574, 0.000105262] | 15 / 7 / 0 |

两项比较均为 **unresolved / not separated by the current interval**；不作 equivalence、superiority 或 state-of-the-art 声明。Constant 是每个 cell 对同一 scored test target 的解析常数拟合，不是独立冻结参数的泛化结果。

## 5. Spatial-alignment development effect 是否 replicated？

**Prediction verdict：REPLICATED。**

| Split | Mean aligned − zero | Aligned wins | Offset/improvement Pearson | Spearman |
|---|---:|---:|---:|---:|
| Development | −0.014916611 | 18/22 | 0.862291866 | 0.846414455 |
| Held-out | −0.016701797 | 19/22 | 0.871005211 | 0.786561265 |

改善方向保持、量级接近，cell 支持广泛；mean CI 排除 0，单个最大效应 cell 删除后仍有平均改善及明显正向相关，没有出现明显 development-specific collapse。结论限定为当前 22-cell frozen models 在同一 natural movie 后续时间段的预测复现。原 development 数值仅引用先前冻结的 spatial-alignment 结果，见 [development_vs_heldout.csv](development_vs_heldout.csv)。

## 6. H1 / direct-BC / AC clamp 的 held-out ΔNLL 与 predictive consequence 是什么？

ΔNLL = pathway-off − aligned normal。每条 pathway 的 verdict 均为 **HELD-OUT PREDICTIVE CONSEQUENCE SUPPORTED**，依据预注册的正向 mean ΔNLL 及其 95% CI。

| Pathway | Mean ΔNLL | Mean 95% CI | Median ΔNLL | Positive / negative / zero cells |
|---|---:|---|---:|---|
| H1 | 0.002824340 | [0.001522049, 0.004391039] | 0.001091912 | 19 / 3 / 0 |
| direct-BC | 0.305192545 | [0.258092924, 0.352530812] | 0.280744925 | 22 / 0 / 0 |
| AC | 0.226704776 | [0.175476850, 0.281429645] | 0.199035242 | 22 / 0 / 0 |

H1 的 population effect 较小且存在 cell 间异质性：3 个 cell 在 H1-off 后 NLL 降低，逐 cell 负值完整保留于 [pathway_per_cell.csv](pathway_per_cell.csv)。没有隐藏 removal 改善预测的情况。四个 cell-group descriptive means、median CI 见 [pathway_population.csv](pathway_population.csv)。

| Pathway | mean-absolute Δlogit 与 ΔNLL：Pearson | Spearman |
|---|---:|---:|
| H1 | 0.777026961 | 0.888198758 |
| direct-BC | 0.616519474 | 0.531338227 |
| AC | 0.884018483 | 0.776397516 |

这些是描述性相关，不把 logit change 直接等同于 predictive value；所有 signed mean、mean absolute、RMS、完整 scored-vector norm 与 ΔNLL 可追溯到保存的 normal/off logits，完整数值见 [pathway_per_cell.csv](pathway_per_cell.csv)，用于相关分析的列见 [pathway_logit_vs_nll.csv](pathway_logit_vs_nll.csv)。

正向 ΔNLL 支持“从 frozen fitted model 移除此 pathway，会降低对 held-out responses 的 predictive likelihood”。剩余参数没有 retrain，bias 没有 recalibrate；clamp 是依赖当前 model family 的 **model-internal intervention**。本结果不支持直接声称 biological pathway necessity、unique causal contribution、retrained structural ablation 或 real-retina lesion effect。协议中的 AC_LOCAL/AC_TRANSIENT 简写对应生产枚举 AMACRINE_LOCAL/AMACRINE_TRANSIENT，两条分支同时关闭；direct-BC-off 保留 broad BC 到 AC 的输入。

## 7. 这批数据是否正式标记为 consumed？

**是。** [TEST_CONSUMED.md](TEST_CONSUMED.md) 在首个成功 held-out target-based NLL 后、数值被展示前创建，记录了 exact range、cells、recordings/trials、models、clamps、metrics、branch、HEAD、UTC 时间和 protocol SHA256；全部条件完成后追加 completion 记录。

冻结协议 SHA256：`3652c48aa4293f2c3503e15099b40657f6d3b396f4a06f4434c4f0e27167bb08`。此范围不能在根据本次结果修改模型后继续称为 untouched confirmatory test。运行环境检查与纯统计依赖的两次启动修正见 [startup_check_record.md](startup_check_record.md)；没有重复测试推理、修改协议或数值门槛。证据依赖与原始输入不变检查收录于 [evidence_manifest.json](evidence_manifest.json)。

研究决策：当前结果明显加强论文中 fixed spatial alignment 的预测泛化与 frozen pathway intervention 的受限主线，但不增加 biological necessity、唯一机制识别或优于 LN/CNN 的 claim。
