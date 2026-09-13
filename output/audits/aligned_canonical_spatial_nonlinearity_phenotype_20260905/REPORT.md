# Phase 1 — spatial-contrast-dependent residual phenotype

## 1. Aligned Canonical 是否在 high-LSC natural stimuli 上出现更大的 prediction residual？

Development live `[16,20)` 的 absolute Bernoulli NLL 在 Q5 较 Q1 增加；这本身不是 BC nonlinear deficit 的证据。以下均为22个细胞等权均值，单位 nats/scored bin。

| Development | Q1 | Q5 | Q5−Q1 |
|---|---:|---:|---:|
| aligned NLL | 0.408577 | 0.424818 | +0.016242 |
| LN NLL | 0.412354 | 0.420290 | +0.007936 |
| CNN NLL | 0.405014 | 0.441276 | +0.036262 |
| Observed event rate | 0.211682 | 0.235198 | +0.023516 |

66/66 cell/model exact replay 通过：strict load、输入、target、mask、顺序、bitwise logits、原始 float32 NLL 一致。LSC 是 aligned checkpoint 固定 BC support 内均匀加权的原始 Weber-drive 空间标准差，未使用输出选择特征。

分层分析使用 float64 per-bin loss；它与原生 float32 NLL 汇总的最大数值差为 6.0818422e-08 nats/bin（两个区间、全部模型/细胞）。这不改变 exact replay 门槛。

## 2. 是普遍变难，还是相对 LN/CNN 特异变差？

三个模型的 absolute loss 均增加。Canonical 相对 LN 的 raw excess loss 增加，但 mean interval 跨零；相对 CNN 的 raw excess loss 减少且 interval 在零以下，因此不支持一致的 comparator-relative deficit。

| Development per-cell Spearman(LSC, relative loss) | mean | median | 正/负 cells |
|---|---:|---:|---:|
| Canonical−LN | +0.001565 | +0.006353 | 12/10 |
| Canonical−CNN | -0.072196 | -0.064552 | 3/19 |

## 3. Canonical−LN 的 high-vs-low excess NLL 是多少？

Mean E **+0.008306**，paired-cell bootstrap 95% CI **[-0.001340, +0.018214]**；median +0.010069，median 95% CI [+0.003391, +0.013889]；正/负/零细胞 17/5/0。

唯一预注册删一细胞检查：删除最大 |E| 的 69#4 后，21-cell mean 为 +0.005122。

## 4. Canonical−CNN 的 high-vs-low excess NLL 是多少？

Mean E **-0.020020**，paired-cell bootstrap 95% CI **[-0.037124, -0.004441]**；median -0.010291，median 95% CI [-0.035263, +0.007685]；正/负/零细胞 8/14/0。

唯一预注册删一细胞检查：删除最大 |E| 的 67#7 后，21-cell mean 为 -0.014988。

## 5. 控制 local mean intensity 后 effect 还剩多少？

| Comparator | Raw mean E | Matched mean E | Matched mean 95% CI | Matched median | 正/负 cells |
|---|---:|---:|---|---:|---:|
| Canonical−LN | +0.008306 | +0.016655 | [-0.000304, +0.034497] | +0.015300 | 16/6 |
| Canonical−CNN | -0.020020 | +0.016566 | [-0.014762, +0.049867] | -0.002063 | 11/11 |

LN 的 matched 点估计未消失；CNN 从 raw 负值变为 matched 正值。但两者 matched mean intervals 均跨零。

控制按预注册 M 五分位进行，同一层内比较全局 Q5/Q1，并按 min(nQ1,nQ5) 加权。仅2–3个 M strata/细胞有共同支持；overlap mass / Q1（Q5相同）的范围为 12.5%–36.5%，中位数 20.8%。匹配后仍有连续 local mean 差异：细胞中位数 0.110150，范围 [0.000886, 0.226965] Weber drive。故不能宣称完全排除了 mean-intensity 共变。

## 6. 哪些 cell classes 最明显？

正向 LN 点估计主要见于 MC_ON、其次 PC_ON；OFF 群体未呈现同样方向。下表仅描述，无 group tests 或 subset selection。每格为 mean / median E。

| Class | N | LN raw | LN matched | CNN raw | CNN matched |
|---|---:|---:|---:|---:|---:|
| MC_ON | 5 | +0.020357 / +0.011732 | +0.043681 / +0.026999 | -0.042759 / -0.046585 | +0.060144 / +0.081905 |
| MC_OFF | 4 | -0.010684 / -0.002702 | -0.004471 / -0.006044 | -0.034535 / -0.028286 | +0.004239 / +0.017776 |
| PC_ON | 9 | +0.014377 / +0.013889 | +0.027285 / +0.022815 | -0.008338 / -0.010329 | +0.011345 / +0.002375 |
| PC_OFF | 4 | -0.001427 / +0.003671 | -0.019919 / -0.012752 | -0.003368 / -0.002596 | -0.013833 / -0.028287 |

## 7. 结果是 GO / MIXED / NO-GO？

**MIXED — WEAK PHENOTYPE**。Development 中没有 comparator 同时满足预注册 raw mean CI >0、删一细胞后 mean >0、matched mean CI >0 与至少50%效应保留。LN 的 raw/matched 均值同为正，因此按冻结规则判为 MIXED。

Development 判定已先写入并封存，之后才计算已 consumed `[20,60)` 的描述性复用：

| Comparator / estimate | Development mean E | Consumed descriptive mean E | 方向一致 |
|---|---:|---:|---|
| LN / raw | +0.008306 | +0.003234 | 是 |
| LN / matched | +0.016655 | +0.003569 | 是 |
| CNN / raw | -0.020020 | -0.023324 | 是 |
| CNN / matched | +0.016566 | -0.003008 | 否 |

该区间沿用 development 的 LSC/M 阈值，未重新分位；不是新的 independent confirmation，不参与或改变判定。

## 8. 是否有足够证据进入 minimal BC pre-pooling nonlinearity 实验？

尚未达到本轮预注册 GO 条件。现有结果是 comparator-dependent 的弱 phenotype，不能据此认定 BC rectification 缺失或已证明具体机制。交回用户/Chat 决定是否进入 Phase 2；本轮停止，未实现 rectification，训练、参数拟合、模型/生产代码修改、新 seed 均为0。
