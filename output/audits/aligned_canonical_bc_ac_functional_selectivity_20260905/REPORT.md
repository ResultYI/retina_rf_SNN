# Phase M1 — direct-BC versus downstream AC functional selectivity

## 1. AC predictive consequence是否随 central-vs-broad context mismatch 增大？

Development `[16,20)` 未提供清晰支持：AC的high-S minus low-S mean E接近零，interval跨零，median为负；不是稳定的正向context selectivity。所有primary统计使用22-cell等权、原始scored bins（65,760 bins），E单位nats/bin。

S=|R−C|，C是固定BC support的均匀平均drive，R是固定AC\BC context的均匀平均drive。Δloss_P=loss(P-off,recalibrated)−loss(normal,uncalibrated)。下文“raw E”指尚未归一化的selectivity，仍由recalibrated off logits计算；不是未校准pathway-off penalty。

**证据来源分开记录：**历史/frozen evidence是development raw normal/off logits及training-only biases。本轮按明确授权首次计算raw_off.double()+frozen_bias，产生 **NEW_DETERMINISTIC_DERIVED_ARTIFACT**：development recalibrated logits/NLL，以及后续M1统计。它们不是历史recalibrated replay；确定性派生本身不降低其作为本轮输入的有效性。

22个aligned checkpoints strict-load，normal/BC-off/AC-off共66组raw重放与原artifact逐位一致；44个bias源/lock精确验证，44组新派生加法逐位一致。减法核查最大舍入残差 1.7763568e-15，均在预注册float64舍入界限内。未调用solver、未拟合bias。

## 2. direct-BC 是否也表现相同关系？

direct-BC的raw mismatch E均值同样为正但区间跨零；不能据raw scale宣称AC更selective。Architecture-derived central-magnitude comparison仅为secondary：

| U-Q5−U-Q1 consequence | Mean G | Median | 正/负 cells |
|---|---:|---:|---:|
| direct-BC | +0.056770 | +0.040841 | 14/8 |
| AC | -0.091415 | -0.107168 | 5/17 |

| Per-cell Spearman | Mean | Median | 正/负 cells |
|---|---:|---:|---:|
| S / BC | -0.029820 | -0.071375 | 9/13 |
| S / AC | +0.032570 | +0.030304 | 15/7 |
| U / BC | +0.033427 | +0.021995 | 14/8 |
| U / AC | +0.030559 | +0.061783 | 15/7 |

## 3. AC raw E 是多少？

Mean E **+0.001382**，paired-cell bootstrap mean95% CI **[-0.031096, +0.036378]**；median -0.019407，median95% CI [-0.039165, +0.028967]。正/负/零cells：10/12/0。

唯一E_AC outlier检查：删除最大|E_AC|的 69#7 后，mean为 -0.008288。

## 4. direct-BC raw E 是多少？

Mean E **+0.016127**，paired-cell bootstrap mean95% CI **[-0.035767, +0.068759]**；median +0.002388，median95% CI [-0.050253, +0.075608]。正/负/零cells：11/11/0。

## 5. Normalized AC-vs-BC differentiation D 是多少？

Mean D **-0.147367**，95% CI **[-0.549491, +0.297390]**；median -0.571642，median95% CI [-0.902034, +0.306272]；正/负cells 9/13。不支持AC normalized selectivity更强。

Mean N_AC=+0.002832，mean N_BC=+0.150199。归一化使用各cell/pathway全部analysis bins平均penalty；44个denominator均正且超过预注册数值零界限，无epsilon、clip或排除cell。BC denominator范围 [0.046256, 0.367516]；AC [0.034471, 0.283544]。

唯一D outlier检查：删除 68#10 后，mean D=-0.274322。

## 6. 控制 central |drive| 后 AC effect还剩多少？

| U-matched quantity | Mean | Mean95% CI | Median | 正/负 cells |
|---|---:|---|---:|---:|
| E_AC_U | -0.041460 | [-0.088555, +0.007202] | -0.050890 | 8/14 |
| E_BC_U | -0.046120 | [-0.089958, -0.001526] | -0.078892 | 10/12 |
| D_U | -0.206898 | [-0.725212, +0.392057] | -0.331041 | 9/13 |

AC的raw弱正均值在U-matching后转负。22cells均有overlap；overlap mass/Q1（Q5相同）中位数 43.2%，范围 33.3%–51.0%。连续U的加权Q5−Q1差异中位数 0.112005 Weber drive，范围 [-0.354894, 0.355971]，不是精确连续匹配。Normalized U-control仍使用原无条件denominator。

## 7. 控制 generic spatial heterogeneity 后还剩多少？

| LSC_AC-matched | Mean E | Median | 正/负 cells |
|---|---:|---:|---:|
| E_AC_LSC | -0.059855 | -0.045159 | 9/13 |
| E_BC_LSC | +0.000028 | -0.019930 | 10/12 |

这是预注册secondary描述性控制，没有额外bootstrap；AC正向effect没有保留，因而不能支持特异context解释。Overlap mass/Q1中位数 10.4%，范围 2.1%–16.7%；残余LSC差异中位数 0.113762，范围 [-0.053567, 0.356559]。有限overlap不允许声称LSC已被完全排除或已确定解释全部effect。

## 8. 哪些 cell classes 最明显？

仅描述，不做group tests或更改primary cohort。每格为mean / median；normalized值和D无量纲。Signed Q<0/Q>0的AC描述见signed_context_descriptive.csv，不进入判定。

| Class | N | E_AC | E_BC | N_AC | N_BC | D |
|---|---:|---:|---:|---:|---:|---:|
| MC_ON | 5 | +0.078317 / +0.048390 | +0.121596 / +0.148646 | +0.703245 / +0.170663 | +0.468372 / +0.715062 | +0.234873 / +0.105247 |
| MC_OFF | 4 | +0.009709 / +0.027749 | -0.071099 / -0.044768 | +0.118155 / +0.255792 | -0.229063 / -0.129275 | +0.347218 / +0.702885 |
| PC_ON | 9 | -0.033379 / -0.032446 | +0.031632 / +0.044289 | -0.262638 / -0.224303 | +0.331244 / +0.310633 | -0.593883 / -0.821691 |
| PC_OFF | 4 | -0.024900 / -0.038436 | -0.063371 / -0.050207 | -0.390701 / -0.930272 | -0.275607 / -0.164735 | -0.115094 / -0.596983 |

## 9. Consumed descriptive reuse方向是否一致？

| Quantity | Development mean | Consumed mean | Population方向一致 | Cell符号一致 |
|---|---:|---:|---|---:|
| E_AC | +0.001382 | +0.209551 | 是 | 12/22 |
| E_BC | +0.016127 | +0.144173 | 是 | 10/22 |
| D | -0.147367 | +0.464893 | 否 | 8/22 |

Consumed `[20,60)`使用相同geometry/bias及development冻结S/U/LSC阈值。44组consumed recalibrated NLL与上一轮已记录值exact；没有新inference或bias拟合。D的population方向转为正，但仅8/22逐细胞符号一致。该范围不是新的independent confirmation，不能修订development判定。

## 10. 最终 verdict：GO / MIXED / NO-GO？

**MIXED — FUNCTIONAL SELECTIVITY WEAK OR NONSPECIFIC**

按冻结的有序规则，AC raw mean>0，但CI跨零、单一最大贡献cell删除后转负、U/LSC控制后均值为负，normalized D不支持AC更强，因此为MIXED。该结果没有证明distinct AC context selectivity，也不是对真实生物AC功能的唯一解释。

Development verdict先写入并SHA256锁定，之后才读取consumed prediction payload。历史停止记录已保留；本轮仅解除用户明确授权的确定性派生阻断。训练、拟合、model/front-end/center/support/生产文件修改、新seed、artificial stimuli及M2均为0。

当前没有足够证据进入 Phase M2 的 artificial matched-context circuit prediction experiment。
