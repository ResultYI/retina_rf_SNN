# Phase T1 — temporal/adaptation-dependent residual phenotype

## 1. High recent adaptation-load stimuli 上是否存在更大的 absolute prediction loss？

没有。Development `[16,20)` 的 Canonical equal-cell absolute NLL 在 A-Q5 较 A-Q1 更低。以下是22-cell等权均值；NLL单位为 nats/phenotype bin。

| Development | A-Q1 | A-Q5 | Q5−Q1 |
|---|---:|---:|---:|
| aligned NLL | 0.449542 | 0.440650 | -0.008893 |
| LN NLL | 0.444761 | 0.462533 | +0.017772 |
| CNN NLL | 0.432815 | 0.429421 | -0.003395 |
| Observed event rate | 0.207470 | 0.216669 | +0.009199 |

66/66 strict exact replay 通过，原 scored-mask logits、NLL、输入、target、mask 和 source/trial order 均精确一致。本 phenotype 仅保留 local t45..149：548 sequences、57,540 bins；原65,760-bin模型评价不变。A 使用严格过去45bins，不含当前bin、不跨sequence；C固定k5。

分层分析使用 float64 loss。两个区间全部cell/model的 phenotype-mask float64与原生float32汇总最大数值差为 7.7333527e-08 nats/bin，不改变原NLL exact replay门槛。

## 2. 这种变化是否相对 LN/CNN 特异？

没有发现预注册方向的 comparator-relative deficit：Canonical−LN 的 A high-vs-low E 为负且mean interval位于零以下；Canonical−CNN的点估计也为负，interval跨零。不能将 absolute loss、event rate变化解释为 adaptation 缺失。

## 3. Canonical−LN 的 adaptation high-vs-low excess NLL 是多少？

Mean E **-0.026664**，paired-cell bootstrap 95% CI **[-0.041883, -0.012650]**；median -0.018704，median95% CI [-0.041143, -0.007698]。正/负/零细胞：5/17/0。

唯一预注册删一细胞检查：删除最大|E|的 69#4 后，21-cell mean E 为 -0.022406。

## 4. Canonical−CNN 的 adaptation high-vs-low excess NLL 是多少？

Mean E **-0.005498**，paired-cell bootstrap 95% CI **[-0.026221, +0.015172]**；median -0.011032，median95% CI [-0.028846, +0.006891]。正/负/零细胞：10/12/0。

唯一预注册删一细胞检查：删除最大|E|的 70#34 后，21-cell mean E 为 -0.000584。

## 5. 控制 current |drive| 后 effect 还剩多少？

| Comparator | Raw mean E | U-matched mean E | Mean95% CI | Median | 正/负/零 cells |
|---|---:|---:|---|---:|---:|
| Canonical−LN | -0.026664 | -0.025377 | [-0.039455, -0.012175] | -0.022574 | 4/18/0 |
| Canonical−CNN | -0.005498 | -0.009580 | [-0.028425, +0.009354] | -0.009556 | 8/14/0 |

22个细胞均有U-stratum overlap；overlap mass/Q1（Q5相同）中位数 73.8%，范围 63.1%–85.7%。加权后Q5−Q1的连续U差异中位数 0.371834，范围 [-0.022284, 0.793586] Weber drive；C差异中位数 0.763725。五分位分层没有完全消除连续当前drive或recent-change共变。

Stimulus-only条件触发了预注册U×C二维描述性控制（全22cells，无bootstrap、不参与verdict）：

| Comparator | U×C-matched mean E | Median | 正/负 cells |
|---|---:|---:|---:|
| Canonical−LN | -0.021199 | -0.018698 | 4/18 |
| Canonical−CNN | -0.013796 | -0.016063 | 6/16 |

## 6. Fast temporal change C 是否表现出类似 phenotype？

未发现正向的 population point estimate。C仅为secondary；这些结果不能被自动称为 adaptation deficit。

| Comparator | C mean E | Median | 正/负/零 cells |
|---|---:|---:|---:|
| Canonical−LN | -0.015368 | -0.003428 | 8/14/0 |
| Canonical−CNN | -0.041733 | -0.032982 | 6/16/0 |

| Per-cell Spearman(feature, relative loss) | Mean | Median | 正/负 cells |
|---|---:|---:|---:|
| A / Canonical−LN | -0.033921 | -0.025764 | 8/14 |
| C / Canonical−LN | -0.000114 | -0.011648 | 10/12 |
| A / Canonical−CNN | -0.064237 | -0.063310 | 4/18 |
| C / Canonical−CNN | -0.088022 | -0.102072 | 3/19 |

## 7. 哪些 cell classes 最明显？

相对CNN，MC_OFF的raw/matched均值为正；PC_OFF均值弱正而中位数为负。相对LN四类均值均为负。仅作描述，不据此筛选primary cohort。每格为mean / median adaptation E。

| Class | N | LN raw | LN U-matched | CNN raw | CNN U-matched |
|---|---:|---:|---:|---:|---:|
| MC_ON | 5 | -0.032791 / -0.028951 | -0.034282 / -0.028789 | -0.025180 / -0.054161 | -0.035361 / -0.065326 |
| MC_OFF | 4 | -0.014502 / -0.010801 | -0.013595 / -0.008844 | +0.029537 / +0.036009 | +0.021962 / +0.030047 |
| PC_ON | 9 | -0.028942 / -0.022387 | -0.025213 / -0.025151 | -0.015577 / -0.015626 | -0.017768 / -0.020447 |
| PC_OFF | 4 | -0.026044 / -0.019445 | -0.026393 / -0.018407 | +0.006747 / -0.003633 | +0.009531 / -0.005220 |

## 8. Consumed descriptive reuse 与 development 方向是否一致？

下表使用已consumed `[20,60)`、575,400 phenotype bins，以及development冻结的A/C/U数值阈值。没有重新定义quintiles；仅作描述性复用，不是独立确认。

| Feature / comparison / estimate | Development mean E | Consumed mean E | 均值方向一致 | 逐细胞符号一致 |
|---|---:|---:|---|---:|
| A / LN / raw_E | -0.026664 | -0.024040 | 是 | 18/22 |
| A / LN / matched_E | -0.025377 | -0.030266 | 是 | 18/22 |
| A / CNN / raw_E | -0.005498 | -0.068059 | 是 | 13/22 |
| A / CNN / matched_E | -0.009580 | -0.072049 | 是 | 14/22 |
| C / LN / raw_E | -0.015368 | -0.011657 | 是 | 16/22 |
| C / CNN / raw_E | -0.041733 | -0.065474 | 是 | 18/22 |

## 9. 最终 verdict 是 GO / MIXED / NO-GO？

**NO-GO — NO TEMPORAL/ADAPTATION-SPECIFIC DEFICIT**

**NO-GO FOR TEMPORAL/ADAPTATION FRONT-END**

判定只使用development：两个comparator的raw A mean均不为正，U-matched均值也均为负，因此不满足GO或MIXED的冻结条件。Development verdict在读取consumed phenotype前已写入并SHA256封存，之后未改变。

该结论限于本轮冻结特征、窗口、数据和模型，不能证明真实视网膜没有adaptation机制。A只是stimulus-derived adaptation-load proxy。本轮front-end/core/生产数据/模型参数修改、training、parameter fitting、新seed和Phase T2实施均为0。

目前没有足够证据进入 Phase T2 的 dynamic front-end mechanism experiment。
