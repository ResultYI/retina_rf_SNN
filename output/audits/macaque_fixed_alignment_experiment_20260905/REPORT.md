# Canonical V1 fixed-LN-center spatial-alignment experiment

**SUPPORTED: centering materially limits current Canonical。** 这一判断限定为：training-derived fixed spatial-alignment proxy改善了当前22-cell development evaluation；不表示真实生理RF center已恢复。

## 运行身份与范围

| 项目 | 实际记录 |
|---|---|
| Branch / HEAD | `rgc-readout-v2` / `fea28de038821fadee279b93728688b34bcb3bac` |
| Git status | tracked diff为空；保留原有untracked文件及上一轮alignment audit，新写入只在本轮两个目录。完整前后status见`git-state-final.json`；不执行commit/reset/clean。 |
| 新训练次数 | 22个aligned conditions，各一次inner selection和一次fresh full-train refit，共44次优化过程；零次重训、零次zero-reference重新训练。 |
| Reference lineage | `output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/` |
| 唯一改变因素 | 固定`cell_positions_degs`从`[[0,0]]`改为同cell冻结LN final full-train refit center的物理坐标。 |
| 不变合同 | Canonical V1；`h1-shared-bc-direct-broad-ac`；`bc-central-disk_ac-overlapping-full-disk`；33 trainable scalars；同一半径、sigma、H1 graph、loss/history、数据/split、seed、优化器和训练预算。 |
| Validation性质 | development evidence，非untouched test；不用于checkpoint/step selection。 |
| 范围保持 | 没有运行SBC、Mach、White、Hermann；没有trainable center、位置搜索或中心裁剪；没有修改production source/model/data或旧frozen results。 |

新fits位于`output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/`；本报告及CSV位于`output/audits/macaque_fixed_alignment_experiment_20260905/`。训练用时约31.13分钟，CPU torch 2.6.0+cpu、threads=2。

## Correctness preflight与最终核对

源码与数值构造确定的映射为：

\[
(x_{deg},y_{deg})=(x_{LN},-y_{LN})\frac{3\times4.6}{256},\qquad pitch=0.05390625^\circ.
\]

LN的x向右、y沿image row向下；Canonical的x向右、y向上。两者使用同一个中心51-pixel crop和17×17 pooling原点，不加额外half-pixel shift。五个人工坐标在MC/PC两类上的10个方向测试通过。实际使用的float32中心及支持域见`coordinate_mapping.json`、`preflight.json`。

训练前22/22 zero checkpoints strict-load通过；从原movie/spikes重建的target、mask、segment/trial顺序相同；validation logits逐元素精确相同，NLL与reference float精确相等。zero/aligned各33个trainable scalars，初始trainable tensors同时与zero factory及冻结raw checkpoint逐元素一致，optimizer membership一致。

22/22中心均在预定义cone grid坐标闭区间内，BC/AC非空、BC严格包含于AC、支持域精确等于固定半径距离阈值，所有空间tensor finite。`68#10`及`70#1`的AC圆盘触及cone-center网格范围边界，均合法构建；这是相对cone-center包围盒的记录，不等同于证明真实图像域缺失或LN中心错误。没有clip或改radius。

最终22个aligned checkpoints全部保存后strict-load并精确回放；全部inner selected/stopping steps由原`DevelopmentStop`复核，refit轨迹长度等于对应selected step。zero/aligned共44组causal replay检查通过；zero的RF和effective parameters在22/22 cells上与冻结artifacts精确相同，三个结构干预delta也精确一致。RF/干预分析保持模型state、grad和training mode。571项训练前冻结文件哈希在训练及分析完成后全部保持一致；包括旧lineage、旧audit、production源代码及raw sources。

**预注册时间边界：** PROTOCOL.md及其中的公式、seed、分组、RF定义和判定阈值在任何aligned训练前冻结。完整分析wrapper在训练期间完成，代码冻结记录时已有4个cells完成；不能声称完整分析代码已在结果产生前注册。这个时间差已在`analysis-code-freeze.json`、`EXECUTION_NOTES.md`披露，未回写或改动原协议，未改变任何统计公式或判定阈值。

## 1. Fixed-LN alignment是否改善prediction

| 22-cell等权统计 | 结果 |
|---|---:|
| zero mean NLL | 0.438956146 |
| aligned mean NLL | 0.424039535 |
| mean ΔNLL（aligned−zero） | **−0.014916611** |
| median ΔNLL | −0.008142963 |
| aligned / zero wins / ties | **18 / 4 / 0** |
| paired-cell bootstrap 95% CI | **[−0.024541496, −0.007502861]** |
| 去掉预先指定68#10后的mean Δ | −0.011225624 |

Bootstrap为100,000次配对cell重采样，NumPy RNG seed=20260905；不是bin/trial重采样。相对zero mean NLL降低约3.40%，此百分比不表示方差解释率。

| Group | n | mean ΔNLL |
|---|---:|---:|
| MC ON | 5 | −0.027197284 |
| MC OFF | 4 | −0.018430807 |
| PC ON | 9 | −0.010523309 |
| PC OFF | 4 | −0.005936503 |

以上分组只作描述，没有group significance tests。完整22-cell结果见`per_cell_results.csv`及`paired_prediction.csv`。

## 2. 改善是否随预先冻结offset增大

| Hypothesis diagnostic | 结果 |
|---|---:|
| Pearson(offset, improvement) | **0.862292** |
| Spearman(offset, improvement) | **0.846414** |
| 小offset 11 cells的mean improvement | 0.002523043 |
| 大offset 11 cells的mean improvement | 0.027310179 |

offset分组按预先冻结的radial offset排序，未据新NLL选择cell subset。总体CI、小/大offset比较及两种相关系数均符合预注册方向，且改善不只来自68#10。相关性是hypothesis diagnostic，不是因果证明。逐cell数据见`alignment_vs_improvement.csv`；固定分组名单见`summary.json`。

## 3. 原Canonical-vs-LN gap回收描述量

仅对原本zero NLL高于LN的**16 cells**定义`R=(zero−aligned)/(zero−LN)`；其余6 cells留空。

| 描述量 | 结果 |
|---|---:|
| 16-cell mean R | 87.97% |
| 16-cell median R | 91.76% |
| sum(improvement)/sum(original gap)，同16 cells | **94.35%** |
| 全22-cell mean Canonical−LN gap，zero | +0.012958201 |
| 全22-cell mean Canonical−LN gap，aligned | −0.001958409 |

没有截断R>1或R<0。mean R、median R和gap加权比率的权重不同；94.35%不是causal variance explained，也不意味着每个cell都追平LN。LN汇总NLL已逐一与其22个冻结final-refit逐cell结果核对一致。

## 4. RF与pathway quantities是否改变

RF保持当前16-lag合同：validation各sequence末端logit的输入Jacobian，随后跨sequence取均值；`direct_BC=RF(H1-off,AC-off)`，`AC=RF(H1-off)−direct_BC`，`H1=global−RF(H1-off)`。

| RF | median relative L2 | median cosine | median energy-centroid位移（degree） |
|---|---:|---:|---:|
| global | 1.38356 | 0.25727 | 0.08678 |
| H1 ordered | 1.35583 | 0.87026 | 0.08608 |
| direct-BC ordered | 1.32691 | 0.31491 | 0.08648 |
| AC ordered | 0.87245 | 0.66538 | 0.08752 |

这是同一刺激坐标系下的比较，包含固定中心平移的直接影响；不能仅凭RF差异把变化归因于学习参数补偿。

| Learned quantity | zero mean | aligned mean | 上升 / 下降 / 不变 cells |
|---|---:|---:|---:|
| H1 amplitude | 0.04957 | 0.09396 | **22 / 0 / 0** |
| BC gain | 0.72161 | 0.83277 | 17 / 5 / 0 |
| AC gain | 1.08758 | 1.14480 | 16 / 6 / 0 |
| AC local delay（ms） | 13.36989 | 16.21991 | **20 / 2 / 0** |
| AC transient delay（ms） | 10.62577 | 13.86189 | **21 / 1 / 0** |
| BC sustained τ basis 1/2/3（ms） | 26.556 / 56.910 / 106.339 | 26.094 / 53.117 / 97.811 | 下降17 / 17 / 16 cells |
| AC local mixture | 0.29004 | 0.27403 | 9 / 13 / 0 |
| history gate | 0.08021 | 0.04806 | 1 / 2 / 19 |

所有BC/AC τ、delay、mixture、history及其逐cell变化均保存在`mechanism_comparison.csv`、`mechanism_summary.csv`和effective-parameter tensors；上表只列便于判断方向的一部分。H1 amplitude、AC delays及部分BC τ有跨cell方向一致的变化；AC mixture、部分τ及history并非统一移动，不能把所有机制量描述成同一方向的补偿。

结构干预沿用reference的主口径：相同validation所有sequence bins，包括warmup；先逐cell计算mean absolute Δlogit，再等权平均。

| 干预 | zero mean | aligned mean | 上升 / 下降 cells |
|---|---:|---:|---:|
| H1-off | 0.027476 | 0.067166 | **21 / 1** |
| direct-BC-off | 1.270635 | 1.315029 | 13 / 9 |
| AC-off | 1.045035 | 1.042168 | 13 / 9 |

score-mask-only结果另列在CSV，未与以上主口径混合。H1干预效应明显增加；BC/AC效应变化方向较混合。可以据此确认空间配准影响模型内部的pathway/temporal estimates，但不能说aligned参数更接近真实physiology。

## 5. 下一阶段判断

**优先继续寻找独立、真实RF-center evidence。** 本轮已支持fixed centering是当前development prediction的重要限制，而且绝大部分原LN gap已被这一不增加参数的改变回收。仍缺少独立生理中心证据；两个网格边界记录只能说明几何条件，不能单独证明LN proxy错误。现有证据不足以直接把剩余误差归因于需要trainable center。

因此，不把本轮正结果自动转成增加两个center parameters的授权，也不为了提高分数继续调位置或半径。是否另行测试trainable center，须先明确proxy的独立限制及新的参数计数/identifiability合同，再由用户决定。本轮实验至此结束。

## 证据文件

- `PROTOCOL.md`、`manifest.json`、`analysis-code-freeze.json`：训练前协议、源/旧checkpoint/raw数据身份及分析代码时间记录。
- `preflight.json`、`coordinate_mapping.json`：22-cell零中心精确回放、初始化、边界及方向检查。
- `paired_prediction.csv`、`alignment_vs_improvement.csv`、`group_prediction.csv`、`summary.json`：全部预注册prediction结果。
- `mechanism_comparison.csv`、`mechanism_summary.csv`、`rf_comparison.csv`、`verification.json`：机制对照及44组运行验证。
- 新fits目录的`cells/<cell>/`：raw、inner-best、final checkpoints；inner/refit trajectories；selected step；validation predictions；metadata及fixed coordinates；causal replay tensors。
- `EXECUTION_NOTES.md`、`git-state-final.json`：执行偏差披露和工作树记录。最终文件hash清单见`artifact-manifest.json`。
