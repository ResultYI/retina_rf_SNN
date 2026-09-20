# Population RetiPath Stage B：只读诊断

日期：2026-09-20。范围：W01–W05 × seeds 4101/4102/4103 × A/C/E，45 个已完成的 step-400 fits。本文为 post-hoc descriptive diagnosis，不改变 protocol、primary、预算、评价定义或成功标准。

**事实摘要：** C 的内部 state、BC output、d_E 和 direct intervention recovery 改善，但 RGC excess CE 在全部 teacher/paired seeds 上高于 A/E。C 的 RGC microbatch exposure 是 A/E 的 **1/3**。曝光/目标配比、prior 约束与 optimization limitation 的解释与现有证据相容；无法区分各因素的因果贡献。日志存在局部梯度反向，不能由接近零的平均 cosine 宣布没有冲突，也没有持续整体负向的证据。γBR/γAR 最终接近固定 prior 锚点，功能恢复不等于这两个 coupling 参数已恢复。

## 1. 输入与计算边界

本轮仅新建本文。读取既有 trajectory/gradient CSV、15 个 initial student 与 45 个 final checkpoint 张量、已有 summary CSV/JSON及冻结源码文本。checkpoint 使用 `weights_only=True` 在 CPU 上读取；未实例化或调用 student/teacher，未运行 forward/backward/autograd、optimizer、训练、测试或 checkpoint replay。未读取 `train_data/`、`evaluator_only/`、`evaluation/*_raw.pt` 或新的 test target。所有 test 指标引用已有 summary，未重新评分。RF 未计算。

逐 update 分析覆盖 18,000 行 trajectory、288,000 行 gradient norms、216,000 行 gradient cosines。表中的时间窗仅用于展示，不构成新的选择、成功或停止标准。每条件 15 fits 等权，world/seed 数相同；pooled update 均值等于先按 fit、再按 world 等权均值。只分析最终保留的完整轨迹，不把原报告记载的中断尝试并入配对统计。

## 2. Exposure 与目标函数

| 条件 | 每 update 的三个 microbatches | RGC batches / fit | RGC sequence exposures | RGC 项权重总和 |
| --- | --- | --- | --- | --- |
| A RGC_ONLY | base_R + base_R_1 + base_R_2 | 1200 | 4800 | 1 |
| C MULTILEVEL_PARTIAL | base_R + H + BC | 400 | 1600 | 1/3 |
| E EXTRA_RGC_CONTROL | base_R + extra_R_1 + extra_R_2 | 1200 | 4800 | 1 |

均为 400 updates、batch size 4、总 1200 microbatches / 4800 sequence exposures。第一个 base_R 的数据、标签及 schedule 配对相同。A 的另外两个 stream 仍来自同一 72-sequence base pool；E 另外两个来自各 72 条独立 extra-RGC pool。C 的 H/BC 观测不计作 RGC exposure。因此三者是**总 microbatch exposure-matched**，不是 RGC-exposure-matched 或 information-matched。

冻结目标：

- A：`(L_R + L_R1 + L_R2)/3 + P`。
- C：`L_R/3 + 0.4 L_H + 0.4 L_BC + P`。
- E：`(L_R + L_extra1 + L_extra2)/3 + P`。

RGC 为 mean Bernoulli BCE；H/BC 为 `0.5*((prediction-target)/0.03)^2` 的 mean，BC 的 state/output 两端口一并平均。P 是原 hierarchy penalty，系数 1，每步一次。C 相对 P 的 RGC 项总系数也只有 A/E 的 1/3；曝光和目标配比随条件共同变化，现有比较不能将它们拆开。不同条件的 total weighted data loss 不是相同目标，不能直接排序。

## 3. RGC prediction penalty 与 base-RGC trajectory

已有 held-out excess CE（nats/RGC-bin）：

| A | C | E | C−A | C−E |
| --- | --- | --- | --- | --- |
| 0.001028849 | 0.001273956 | 0.000982087 | +0.000245107 | +0.000291870 |

C−A、C−E 在 5/5 teachers、15/15 paired seeds 均为正。相对 A/E 的 **excess CE** 约增加 23.8% / 29.7%，分母不是总 CE；未新增“明显受损”阈值。

以下为同一配对 base_R stream 上、更新前的原始 `loss_base_R`，不是重评全训练集：

| update 窗口 | A | C | E | C−A | C−E |
| --- | --- | --- | --- | --- | --- |
| 1-50 | 0.290673 | 0.290706 | 0.290675 | 3.235692e-5 | 3.040312e-5 |
| 51-100 | 0.290977 | 0.291078 | 0.290980 | 0.000101 | 9.840822e-5 |
| 101-200 | 0.290548 | 0.290727 | 0.290559 | 0.000179 | 0.000168 |
| 201-300 | 0.290312 | 0.290555 | 0.290331 | 0.000244 | 0.000225 |
| 301-400 | 0.290675 | 0.290945 | 0.290697 | 0.000270 | 0.000247 |
| 351-400 | 0.290452 | 0.290727 | 0.290477 | 0.000275 | 0.000250 |

各步 batch 随 schedule 改变，窗口波动不能当作严格单调的学习曲线。同一 update 的横向配对有明确对象。C 的差距在训练中也存在，不能仅归于 test 端的新现象；这不证明收敛，也不排除有限预算限制。

末 50 步逐 teacher 均值：

| teacher | A BCE | C BCE | E BCE | C−A | C−E |
| --- | --- | --- | --- | --- | --- |
| W01 | 0.295702 | 0.295960 | 0.295732 | 0.000259 | 0.000229 |
| W02 | 0.318230 | 0.318772 | 0.318249 | 0.000542 | 0.000523 |
| W03 | 0.299260 | 0.299568 | 0.299280 | 0.000308 | 0.000288 |
| W04 | 0.285711 | 0.285871 | 0.285742 | 0.000160 | 0.000129 |
| W05 | 0.253358 | 0.253465 | 0.253384 | 0.000106 | 8.080035e-5 |

C 的 H loss 从前 50 步 0.517290 到末 50 步 0.511814；BC loss 从 0.532059 到 0.526992。它们是带噪观测训练目标值，不是 held-out latent RMSE，也不是收敛判据。

## 4. 逐 update raw / weighted gradient norms

每个 dataset batch 在同一更新前参数点求 raw gradient，再乘冻结权重。下表为全 400 步的逐 batch L2 norm 均值，每行覆盖 6000 个实际 update；state/output/coupling 列为 **weighted**。

| 条件 / dataset | all raw | all weighted | state weighted | output weighted | coupling weighted |
| --- | --- | --- | --- | --- | --- |
| A / base_R | 0.008318 | 0.002773 | 4.141251e-5 | 0.002769 | 9.759407e-5 |
| A / base_R_1 | 0.008345 | 0.002782 | 4.111840e-5 | 0.002778 | 9.800321e-5 |
| A / base_R_2 | 0.008351 | 0.002784 | 4.127591e-5 | 0.002780 | 9.961395e-5 |
| C / base_R | 0.008762 | 0.002921 | 4.458993e-5 | 0.002918 | 8.644781e-5 |
| C / H | 0.035474 | 0.014189 | 0.014189 | 0.000000 | 0.000000 |
| C / BC | 0.029613 | 0.011845 | 0.011031 | 0.003570 | 0.001787 |
| E / base_R | 0.008244 | 0.002748 | 4.096310e-5 | 0.002745 | 9.810145e-5 |
| E / extra_R_1 | 0.008723 | 0.002908 | 3.907384e-5 | 0.002904 | 0.000109 |
| E / extra_R_2 | 0.009176 | 0.003059 | 4.136581e-5 | 0.003055 | 0.000108 |

组内 raw norm 可由 RGC 各列乘 3、H/BC 各列除以 0.4 得到；原 CSV 同时保存两者。不同 batch 的范数之和不是合计梯度范数；真正合计值使用 trajectory 中的 `preclip_norm`。

参数组为 state 258、output 56、coupling 45，共 359 scalars。H loss 支持 H1 dynamics 的 50 个 state 坐标；BC 支持 250 state + 50 output + 25 coupling，共 325 坐标。BC 的 coupling 支持**仅为 a_H**，不包括 γBR/γAR。新增 H/BC loss 对 γBR、γAR、AC dynamics、b_A、RGC bias 没有数据梯度路径；这些下游量的数据梯度来自 RGC。

C 的 H/BC weighted state norms 均值为 0.014189 / 0.011031，base_R state norm 为 0.000044590。范数尺度差异是日志事实，不等于 Adam 实际步长比例。

### Hierarchy penalty

P 对 unit deviations 收缩，并对指定 family centers 锚定，采用 sum；数据 loss 采用 mean。以下 P 的 raw = weighted：

| 条件 / 窗口 | P all norm | state | output | coupling | P loss |
| --- | --- | --- | --- | --- | --- |
| A / all | 0.224639 | 0.024411 | 0.033502 | 0.214511 | 0.019755 |
| A / 351-400 | 0.088131 | 0.019112 | 0.008886 | 0.081618 | 9.072639e-5 |
| C / all | 0.220860 | 0.038320 | 0.032111 | 0.202259 | 0.020237 |
| C / 351-400 | 0.037956 | 0.028466 | 0.008120 | 0.018177 | 8.654153e-5 |
| E / all | 0.229713 | 0.024545 | 0.033851 | 0.220400 | 0.019689 |
| E / 351-400 | 0.103736 | 0.019858 | 0.009275 | 0.097822 | 0.000110 |

末 50 步 RGC batch 的 weighted all norm，A 每个约 0.00266–0.00268，C 为 0.002906，E 每个约 0.00264–0.00291；P all norm 分别为 0.088131 / 0.037956 / 0.103736。P 标量 loss 很小仍可有较大梯度。日志未保存 prior–data cosine 或原始梯度向量，不能由 norm 判断实际抵消程度。

### Global clip

| 条件 | 触发 / 6000 | 比例 | preclip 全程均值 | 最大值 | 末 50 步均值 | 平均 clip 系数 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 278/6000 | 4.633333% | 0.223759 | 7.035794 | 0.087701 | 0.974101 |
| C | 296/6000 | 4.933333% | 0.221460 | 7.576790 | 0.038845 | 0.972794 |
| E | 276/6000 | 4.600000% | 0.228736 | 6.974226 | 0.103345 | 0.974353 |

触发使用原 norm > 1。所有触发都在前 24 步内，第 25–400 步均未触发。系数用已保存 norm 按 `min(1,1/(norm+1e-6))` 算术核对。C 早期触发略多，日志不支持“C 后期持续被 global clipping 压住”；没有 clip 对照，不能估计早期 clip 的因果贡献。

## 5. Gradient conflict 的证据范围

cosine 在**双方共同 autograd-supported 坐标**上计算，不是全参数补零后的夹角。H 的 output/coupling 无共同坐标，记 N/A，不能填成零。负值表示几何反向，不是新增“强冲突”判定阈值。

| C 内 pair / group | 共同坐标 | 均值 | Q25 / median / Q75 | min / max | mean abs | 负值 / 6000 |
| --- | --- | --- | --- | --- | --- | --- |
| base_R–H / all | 50 | -0.002959 | -0.264191 / 0.001906 / 0.257054 | -0.923943 / 0.924219 | 0.319639 | 2992/6000 |
| base_R–BC / all | 325 | 0.001179 | -0.122298 / -0.001454 / 0.122423 | -0.646215 / 0.806229 | 0.155862 | 3014/6000 |
| base_R–BC / state | 250 | -0.037216 | -0.185073 / -0.032384 / 0.111342 | -0.783747 / 0.792615 | 0.192543 | 3389/6000 |
| base_R–BC / output | 50 | 0.082785 | -0.170592 / 0.049449 / 0.315170 | -0.812539 / 0.877306 | 0.289199 | 2672/6000 |
| base_R–BC / coupling | 25 | -0.003352 | -0.680599 / -0.005973 / 0.687169 | -0.998734 / 0.998963 | 0.623881 | 3016/6000 |
| H–BC / all | 50 | -0.049113 | -0.295777 / -0.049070 / 0.181379 | -0.918080 / 0.903113 | 0.289585 | 3329/6000 |

末 50 步 R–H(all)、R–BC(all)、H–BC(all) 均值为 0.012138、0.011498、−0.018055；R–BC(state) 为 −0.035654，coupling 为 −0.009998。RGC-only 三个 batch pairs 的 all-cosine 均值，A 为 0.679582 / 0.696934 / 0.696030，E 为 0.677715 / 0.684246 / 0.712809。

C 的 R–BC coupling cosine 有较大的正负摆动，但其 25 维共享空间是 a_H，**不是 γBR/γAR**。不能用近零均值宣布无冲突，也不能把局部反向扩展为持续的全网络强冲突。共同坐标内的双方子向量 norms 未独立完整保存，不能用现有 group norms/cosines 精确复原 C 合计 data gradient 的抵消比例。

**诊断：** 曝光/目标系数不对等、data–prior 梯度尺度差异及参数锚点回归都有直接记录，局部冲突也存在。optimization/exposure/prior limitation 与证据相容；没有充分证据把 RGC penalty 单独归为 strong gradient conflict。各因素贡献无法区分，不能证明“优化失败”“达到最优”或“增加 steps 即可解决”。

## 6. Initial → final parameter movement

仅对保存的 initial/final tensors 作差；raw L2/RMS 先在每 fit 内计算，再等权平均。raw norm 不具有生理单位，不同物理单位不混入一个 movement score。

| 参数组 / scalars | A raw L2 / RMS | C raw L2 / RMS | E raw L2 / RMS |
| --- | --- | --- | --- |
| all / 359 | 6.744366 / 0.355954 | 3.785690 / 0.199801 | 6.663749 / 0.351699 |
| state / 258 | 5.041756 / 0.313886 | 2.423821 / 0.150901 | 4.619326 / 0.287587 |
| output / 56 | 4.385989 / 0.586102 | 2.792283 / 0.373135 | 4.616174 / 0.616862 |
| coupling / 45 | 0.769730 / 0.114745 | 0.564688 / 0.084179 | 1.194989 / 0.178138 |

45 fits 均有 359/359 raw scalars 的 final 与 initial 不相等。原 completed.json 另记录 359 trainable / optimizer-listed / 曾出现非零 total gradient；它们与本轮的实际参数移动分别核对。发生移动不等于向 teacher 恢复，尤其从扰动初值回到 prior 锚点时。

关键字段的 **physical movement RMS**（同字段按 unit/coordinate 均方，再对 fits 平均）：

| 字段 | A | C | E |
| --- | --- | --- | --- |
| a_H | 0.085894 | 0.057448 | 0.152662 |
| gamma_BR | 0.094663 | 0.094662 | 0.094663 |
| gamma_AR | 0.037763 | 0.037763 | 0.037763 |
| bias | 0.059610 | 0.059615 | 0.059594 |
| alpha_B | 0.356058 | 0.056128 | 0.338051 |
| b_A | 2.554053 | 2.343203 | 2.811424 |
| tau_A | 26.617174 | 22.964382 | 16.302496 |
| delay_A | 2.995174 | 2.217951 | 2.331849 |

raw→physical 按冻结有界 sigmoid 解码，未运行回路。时间字段的 physical 单位为 ms，其余为模型归一化量，不跨字段比较 RMS 大小。

### 固定 prior 锚点与 RGC 相关参数

γBR、γAR、bias 的 raw center prior 均为 SD=0.3、系数 1：

`P_center = 0.5 * sum(((center-initial_center)/0.3)^2)`；
`dP/dcenter = (center-initial_center)/0.09`。

`initial_center` 是构造器的**固定 prior 锚点**，不是各 student 的扰动初值。physical 锚点分别为 γBR=1.5、γAR=0.375、bias=−2.4。Teacher 和 student center 分别独立扰动，回到锚点不等于恢复 teacher。

| 字段 / 条件 | raw movement L2 | initial 距锚点 raw RMS | final 距锚点 raw RMS | physical initial mean | physical final mean | final 全坐标范围 |
| --- | --- | --- | --- | --- | --- | --- |
| gamma_BR / A | 0.175913 | 0.087955 | 1.984263e-5 | 1.491109 | 1.500005 | 1.499875 … 1.500057 |
| gamma_BR / C | 0.175911 | 0.087955 | 3.040347e-5 | 1.491109 | 1.500006 | 1.499839 … 1.500619 |
| gamma_BR / E | 0.175912 | 0.087955 | 4.370047e-5 | 1.491109 | 1.499991 | 1.499612 … 1.500066 |
| gamma_AR / A | 0.412379 | 0.103095 | 3.475571e-5 | 0.379124 | 0.375003 | 0.374967 … 0.375283 |
| gamma_AR / C | 0.412381 | 0.103095 | 1.242323e-5 | 0.379124 | 0.375000 | 0.374941 … 0.375100 |
| gamma_AR / E | 0.412382 | 0.103095 | 1.954173e-5 | 0.379124 | 0.374999 | 0.374819 … 0.375139 |
| bias / A | 0.097009 | 0.068545 | 0.000453 | -2.389120 | -2.399806 | -2.400598 … -2.398659 |
| bias / C | 0.097018 | 0.068545 | 0.000275 | -2.389120 | -2.399850 | -2.400231 … -2.399236 |
| bias / E | 0.096984 | 0.068545 | 0.000501 | -2.389120 | -2.399853 | -2.400879 … -2.398760 |

RGC bias 的 teacher 两坐标与 student final 均值（后者跨三个 seed、两个 RGC 坐标）：

| teacher | teacher bias 两坐标 | A final mean | C final mean | E final mean |
| --- | --- | --- | --- | --- |
| W01 | -2.186383 / -2.350921 | -2.399760 | -2.399829 | -2.399552 |
| W02 | -2.130644 / -2.151455 | -2.399412 | -2.399755 | -2.399335 |
| W03 | -2.144022 / -2.363813 | -2.399830 | -2.399986 | -2.399976 |
| W04 | -2.371650 / -2.213383 | -2.399751 | -2.399855 | -2.400080 |
| W05 | -2.453405 / -2.633838 | -2.400276 | -2.399824 | -2.400321 |

Teacher bias 不同，而三条件 final bias 都贴近 −2.4。该现象与 γBR/γAR 的锚点回归共存，是 RGC 参数调整受约束的记录，不能单独证明 prediction penalty 的因果来源。

### γBR/γAR gradient magnitude：可计算与缺失证据

没有保存逐参数、逐 dataset 的 γBR/γAR raw gradient。现有 coupling norm 包含 a_H + γBR + γAR，不能冒充单个 γ 字段的数据梯度。本轮未补跑 backward。

首步 RGC 数据的 coupling-group weighted norms **之和**在 A/C/E 平均为 0.000149 / 4.923749e-5 / 0.000138，只是每个 γ 字段合计 RGC gradient norm 的上界。H/BC 对这两个字段的数据梯度为零。同一初始点，由 checkpoint/prior 公式直接算出的 γBR prior-gradient L2=1.954551、γAR=4.582006，均为 15 个初值平均，大于相应 RGC 上界。这是初始梯度数值关系，不是 Adam 更新量比例。

| 字段 / 条件 | final prior-gradient L2 | final Adam exp_avg L2 | sqrt(sum exp_avg_sq) |
| --- | --- | --- | --- |
| gamma_BR / A | 0.000441 | 3.724459e-5 | 0.039239 |
| gamma_BR / C | 0.000676 | 4.776548e-5 | 0.038407 |
| gamma_BR / E | 0.000971 | 6.597312e-5 | 0.039458 |
| gamma_AR / A | 0.001545 | 7.625901e-5 | 0.103462 |
| gamma_AR / C | 0.000552 | 3.444557e-5 | 0.100998 |
| gamma_AR / E | 0.000869 | 5.293345e-5 | 0.104228 |
| bias / A | 0.007125 | 0.000809 | 0.020608 |
| bias / C | 0.004319 | 0.000460 | 0.019763 |
| bias / E | 0.007865 | 0.000764 | 0.020738 |

final prior gradient 是 step-400 更新后 checkpoint 上的解析值。Adam moments 为截至 step 400 的 **clipped total gradient（data+prior）** 平滑记录，不是最后一批 raw gradient，不能拆成 RGC/H/BC 贡献，也不能恢复缺失的逐参数历史。

## 7. Direct pathway：功能改善与 γBR 基本不变同时成立

| 已有指标 | A | C | E | C−A | C−E | 改善 teacher / seeds：对 A；对 E |
| --- | --- | --- | --- | --- | --- | --- |
| observed BC delta_r_B RMSE | 0.020528 | 0.006885 | 0.020464 | -0.013643 | -0.013579 | 5/5, 15/15；5/5, 15/15 |
| unobserved BC delta_r_B RMSE | 0.020417 | 0.008633 | 0.020504 | -0.011784 | -0.011871 | 5/5, 15/15；5/5, 15/15 |
| d_E RMSE | 0.031898 | 0.014500 | 0.031836 | -0.017398 | -0.017336 | 5/5, 15/15；5/5, 15/15 |
| BLOCK_DIRECT_BC_DRIVE delta-logit RMSE | 0.036272 | 0.014094 | 0.036421 | -0.022178 | -0.022326 | 5/5, 15/15；5/5, 15/15 |
| gamma_BR parameter RMSE | 0.296712 | 0.296705 | 0.296709 | -6.959674e-6 | -4.624354e-6 | 4/5, 13/15；2/5, 9/15 |

冻结路径：

`d_E[r,m] = gamma_BR[r,m] * sum_j pi_BR[r,m,j] * delta_r_B[j]`。

BC state/output 的改善与 d_E、direct intervention recovery 改善共存，γBR 的条件间 teacher-error 差仅约 10^-6。d_E 由 upstream BC output 和 coupling 共同决定，并不要求两者误差同步改变。

γBR 从扰动初值移动约 0.09466 physical RMS，最终三条件均接近 1.5。“γBR 基本不改善”指 teacher recovery 和条件间比较，并非没有更新。γBR 不是 H/BC loss 的祖先，新增约束落在 upstream state/output 与 a_H。

### γBR 与 BC output amplitude 的 compensation

LegacyPReLU 的正半轴 slope=1，负半轴 slope=alpha_B。可仅由 checkpoint 算出的静态系数：

`K_minus[r,m] = gamma_BR[r,m] * sum_j pi_BR[r,m,j] * alpha_B[j]`。

这是相关 BC state 同在负半轴、等量微小变化时的 algebraic route coefficient，**不是实际刺激下 delta_r_B 的 RMS/amplitude**，也不是 effective RF。实际幅度还取决于 s_B、正负占比、动态和刺激；本轮未计算响应幅度。

| teacher | teacher γBR mean | final route-weighted alpha_B A / C / E | final K_minus A / C / E |
| --- | --- | --- | --- |
| W01 | 1.329118 | 0.213142 / 0.458135 / 0.184801 | 0.319717 / 0.687203 / 0.277202 |
| W02 | 1.278106 | 0.102157 / 0.514736 / 0.103020 | 0.153237 / 0.772105 / 0.154532 |
| W03 | 1.375279 | 0.404192 / 0.517856 / 0.350385 | 0.606287 / 0.776808 / 0.525560 |
| W04 | 1.976442 | 0.530654 / 0.535985 / 0.448727 | 0.795983 / 0.803975 / 0.673091 |
| W05 | 1.496693 | 0.872503 / 0.551558 / 0.792600 | 1.308753 / 0.827328 / 1.188877 |

共同初值的 route-weighted alpha_B mean=0.502785，K_minus=0.750089；final 全 teacher/seed 均值：

| 条件 | route alpha_B | K_minus |
| --- | --- | --- |
| A | 0.424530 | 0.636795 |
| C | 0.515654 | 0.773484 |
| E | 0.375907 | 0.563852 |

条件间 γBR 都在同一锚点附近，BC 负半轴系数及 output error 却不同；没有观察到 γBR 随 alpha_B 反向大幅调整的条件间模式。state dynamics 与完整 output amplitude 的补偿仍不能由这些静态系数排除。

**UNVERIFIED：γBR 与实际 BC output amplitude 的动态 compensation。** 允许读取的 summary 保存了 RMSE，但没有 initial/final response amplitude、交叉乘积或逐 update amplitude。RMSE 不唯一决定这些量。本文保留参数/局部系数关系及已有 output error，不宣布精确补偿或数学不可辨识；没有精确等价变换证明。

BLOCK_DIRECT_BC_DRIVE 只清零 direct BC 驱动，下游重新计算，BC state/output 与 AC branch 保留。delta-logit 为 blocked−normal，使用相同 conditional observed-event history。RGC 加性 bias 在同一模型干预差分中抵消；conductance 的分支交互仍存在。因此 absolute RGC prediction 和 direct intervention error 是不同对象，不要求排序一致。

## 8. AC pathway：一致改善在哪个接口减弱

路径为 `delta_r_B → u_A → a_A → delta_o_A → gamma_AR × spatial routing → d_I → gI/V/logit`，
其中 `delta_o_A = softplus(b_A+a_A)-softplus(b_A)`。

γBA 固定 1；AC 无直接 observation。C 的新增观测在 H/BC，上游变化经固定 routing 进入 AC；AC dynamics、b_A、γAR 的数据梯度仍只来自 RGC。

| 已有指标 | A | C | E | C−A | C−E | 改善 teacher / seeds：对 A；对 E |
| --- | --- | --- | --- | --- | --- | --- |
| AC input u_A RMSE | 0.018489 | 0.003731 | 0.018580 | -0.014758 | -0.014849 | 5/5, 15/15；5/5, 15/15 |
| AC state a_A RMSE | 0.015538 | 0.004830 | 0.015987 | -0.010708 | -0.011157 | 5/5, 15/15；5/5, 15/15 |
| AC output delta_o_A RMSE | 0.013264 | 0.012285 | 0.013230 | -0.000980 | -0.000945 | 5/5, 15/15；5/5, 14/15 |
| inhibitory d_I RMSE | 0.009802 | 0.009261 | 0.009779 | -0.000541 | -0.000519 | 2/5, 6/15；2/5, 6/15 |
| gamma_AR parameter RMSE | 0.111346 | 0.111345 | 0.111346 | -8.376263e-7 | -6.175648e-7 | 3/5, 7/15；3/5, 8/15 |
| AC-block delta-logit RMSE | 0.008144 | 0.007709 | 0.008124 | -0.000435 | -0.000415 | 2/5, 6/15；2/5, 6/15 |

u_A/a_A 的 C 改善为 5/5 teachers、15/15 paired seeds。delta_o_A 仍有 5/5 teachers 改善，但幅度减弱：相对 A 的平均 RMSE 降幅约为 79.8%（u_A）、68.9%（a_A）、7.4%（delta_o_A）。这是各自已有误差的描述性比例，不跨接口合成 score。到 d_I / AC-block，只剩 2/5 teachers、6/15 seeds 改善；γAR RMSE 基本不变。

| teacher | d_I RMSE A / C / E | d_I：C−A / C−E | AC-block RMSE A / C / E | AC-block：C−A / C−E |
| --- | --- | --- | --- | --- |
| W01 | 0.009470 / 0.009648 / 0.009527 | 0.000178 / 0.000120 | 0.008049 / 0.008209 / 0.008102 | 0.000160 / 0.000107 |
| W02 | 0.007280 / 0.007844 / 0.007115 | 0.000564 / 0.000729 | 0.005721 / 0.006287 / 0.005559 | 0.000566 / 0.000728 |
| W03 | 0.009746 / 0.009839 / 0.009674 | 9.332107e-5 / 0.000165 | 0.008291 / 0.008377 / 0.008222 | 8.670165e-5 / 0.000156 |
| W04 | 0.008893 / 0.007604 / 0.010105 | -0.001289 / -0.002501 | 0.007177 / 0.006151 / 0.008286 | -0.001027 / -0.002135 |
| W05 | 0.013621 / 0.011368 / 0.012477 | -0.002253 / -0.001109 | 0.011481 / 0.009519 / 0.010451 | -0.001962 / -0.000932 |

两个端口均为 W01/W02/W03 变差，W04/W05 改善；各 teacher 内三个 paired seeds 同方向。跨 teacher 均值略降不能覆盖这一模式。

### AC output gain 与 γAR

b_A 为 4 个 family-shared output centers，**没有 center prior**；tau_A/delay_A 也 family-shared。b_A final physical mean A/C/E=−1.750995 / −1.318335 / −1.851129，但总均值掩盖了 teacher 差异。仅用保存参数可解析计算零状态斜率：

`d(delta_o_A)/d(a_A)|a_A=0 = sigmoid(b_A)`。

它不是实际 a_A 轨迹处的斜率，也不是新 RF 计算。

| teacher | teacher baseline slope mean | student slope A / C / E | student gamma_AR × slope A / C / E |
| --- | --- | --- | --- |
| W01 | 0.496242 | 0.053497 / 0.084676 / 0.049351 | 0.020062 / 0.031754 / 0.018507 |
| W02 | 0.497614 | 0.042135 / 0.063255 / 0.041065 | 0.015800 / 0.023720 / 0.015399 |
| W03 | 0.531903 | 0.063127 / 0.081706 / 0.056459 | 0.023673 / 0.030640 / 0.021172 |
| W04 | 0.542707 | 0.121457 / 0.256215 / 0.064122 | 0.045546 / 0.096080 / 0.024046 |
| W05 | 0.531218 | 0.866792 / 0.903039 / 0.916534 | 0.325047 / 0.338640 / 0.343700 |

共同初值 baseline slope mean=0.495529、γAR×slope=0.187879；final A/C/E 分别为 0.229402 / 0.277778 / 0.225506，以及 0.086026 / 0.104167 / 0.084565。乘积对 2 RGC × 2 modes × 4 families 坐标均匀平均，未乘 pi_AR、未按实际活动分布加权，只是参数增益分配诊断。

W01–W04 的 student slope 均值低于相应 teacher family 均值，W05 则更高。γAR 贴近 0.375，b_A 明显移动。连同接口误差，证据将改善明显减弱的位置定位到 **AC state→output**；经过 **output→d_I** 的 coupling/空间汇聚后，改善不再跨 teacher 一致。不能仅由一个斜率或 γAR RMSE 判定独立责任：a_A 分布、family 误差方向与空间求和的贡献不能由 scalar RMSE 分解。

AC-block 清零 postsynaptic inhibitory drive，保留 AC state/output、direct branch 和 tonic conductance。其 delta-logit 与 d_I 的方向均只在 W04/W05 改善。本轮未重新运行干预；γAR 的逐 dataset 梯度同样缺失，适用第 6 节的证据限制。

## 9. 事实结论与不可区分部分

| 问题 | 现有证据支持 | 未得到的结论 |
| --- | --- | --- |
| RGC penalty | C 的训练 base-RGC BCE、已有 held-out excess CE 较高；RGC exposure 为 1/3；目标配比、梯度尺度及锚点约束不同 | 不能将代价单独归因于 exposure、400-step budget、prior 或 conflict；不能证明增加 steps 有效 |
| Gradient conflict | 有逐 batch 反向，a_H 共享 coupling 子空间正负切换较大；整体均值不持续为负 | 不能由均值近零排除局部冲突，也不能宣布全局 strong conflict 是主因 |
| Direct | C 的 BC output、d_E、direct-block recovery 一致改善；γBR final 靠近 1.5，teacher RMSE 几乎相同 | 不能写成 γBR 已恢复，或存在精确 amplitude compensation / 数学不可辨识 |
| AC | u_A/a_A 一致改善，delta_o_A 改善减弱，d_I/AC-block 仅 W04/W05 改善；γAR 接近 prior | 不能将全部 AC 接口描述为一致改善，不能合并 state/coupling 为 mechanism score |
| 参数更新 | 全部 359 raw 参数有 initial→final movement；逐 γ 数据梯度未保存 | optimizer membership、非零 total gradient、实际移动和 teacher recovery 不是同一件事 |

结论仅限这 5 个 synthetic instances 与冻结合同，不是生物 population 显著性或生物因果结论。没有新显著性检验、成功阈值、后续研究决策或修改预算/架构/loss 的建议。完成后停止。

## 10. 来源与核对记录

- 原报告：[RETIPATH_POPULATION_STAGE_B.md](D:/PythonProject/retina_rf_SNN/docs/RETIPATH_POPULATION_STAGE_B.md)；合同：[PROTOCOL.md](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/PROTOCOL.md) / [protocol.json](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/protocol.json)；checkpoint 锁：[CHECKPOINT_LOCK.json](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/CHECKPOINT_LOCK.json)；原完成验证：[VERIFICATION.json](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/VERIFICATION.json)。原报告中的历史 replay 属于原 Stage B，本轮没有 replay。
- 逐步证据：`worlds/W01…W05/fits/{A,C,E}_{4101,4102,4103}/trajectory.csv`、`gradient_norms.csv`、`gradient_cosines.csv`、`completed.json`。示例：[trajectory](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/worlds/W01/fits/C_4101/trajectory.csv) / [norms](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/worlds/W01/fits/C_4101/gradient_norms.csv) / [cosines](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/worlds/W01/fits/C_4101/gradient_cosines.csv)。
- 参数证据：各 world 的 `initial_students/{seed}.pt` 与每 fit 的 `final.pt`。示例：[initial](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/worlds/W01/initial_students/4101.pt) / [final](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/worlds/W01/fits/C_4101/final.pt)。movement 为 final−initial；差值、解码、范数采用 float64 算术汇总，保存的 float32 tensors 未改动。
- 已有结果：[descriptive_summary.csv](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/descriptive_summary.csv) / [teacher_summary.csv](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/teacher_summary.csv) / [coupling_coordinates.csv](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/coupling_coordinates.csv) / [teacher_heterogeneity.csv](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/teacher_heterogeneity.csv)。[per_sequence.csv](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/per_sequence.csv) 仅核对字段结构，未重新评分。
- 冻结源码入口：[loss weights](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/source/experiments/retipath_population_v0_1/stage_b.py:32)；[optimizer、exposure、归约](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/source/experiments/retipath_population_v0_1/stage_b.py:103)；[gradient support](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/source/experiments/retipath_population_v0_1/stage_b.py:247)；[shared-coordinate cosine](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/source/experiments/retipath_population_v0_1/stage_b.py:255)；[gradient 累积与 clip](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/source/experiments/retipath_population_v0_1/stage_b.py:324)；[PooledField 解码](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/source/experiments/retipath_population_v0_1/circuit.py:83)；[prior 公式](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/source/experiments/retipath_population_v0_1/circuit.py:121)；[AC/coupling/bias prior 定义](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/source/experiments/retipath_population_v0_1/circuit.py:184)；[BC output / direct](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/source/experiments/retipath_population_v0_1/circuit.py:287)；[AC output / intervention](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/source/experiments/retipath_population_v0_1/circuit.py:298)；[conductance 与 logit](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/source/experiments/retipath_population_v0_1/circuit.py:308)。
- 主要 251 个归档输入逐一对照 [FILE_MANIFEST.json](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b_20260918/FILE_MANIFEST.json) 的 SHA256：135 个日志 CSV、60 个 checkpoints、45 个完成记录、4 个 summary CSV、7 个协议/来源/完成文件。另核对 `per_sequence.csv` 和原报告。结束前复核所读归档输入未变。
- 本轮新增计算仅为日志描述性聚合、checkpoint 差值、固定 prior 导数和静态代数系数；没有新模型响应或 test 指标。完整性核对不替代科学验证。

| 关键文件 | SHA256 |
| --- | --- |
| FILE_MANIFEST.json | `068a6406b77b265a3933e09d8ed325f98a8578883b1ae626b0210bca49fede0c` |
| protocol.json | `d1cec4171de8721f67ae7e23adb76f9c9c9e3f455cd51384913ffebf1bee1fa7` |
| CHECKPOINT_LOCK.json | `2cb2958237e5f38db5b815864ef40820a664cc5940b9bc8c028d15611cdf0c70` |
| 冻结 circuit.py | `2c269625854b139cad43aace55897de4f9c30454b0c215b3abdac4f127be307d` |
| 冻结 stage_b.py | `757b6c2fa0c63c34c8f1de008e2c42b4cdda07ba3114614e1a091fdcc56c6d16` |


