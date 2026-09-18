# RetiPath 架构与训练设计 v0

日期：2026-09-18。状态：**设计稿已完成，待用户批准实施**。

本文件是 G0 设计交付。下述新方程、数值、接口和日程均为待实现的 synthetic 规格，不是已有实现、已运行实验或生理事实。本轮只阅读文本、合并文档和核对定义；未运行模型、checkpoint、数值试验、pytest、训练或数据生成，未读取 natural-movie/spike payload。

## 1. 推荐设计与本地依据

推荐一个固定几何、12 个可学习标量的有效回路：25 个局部输入/BC 节点、25 个 H1-like 状态节点、9 个 AC-associated 节点、1 个有效 RGC、2 个空间整合 mode。BC 的非线性发生在局部节点、空间汇总之前。H1 的减法反馈输出与状态分开；BC 的同一输出同时进入 direct 和 AC-associated 两条支路。沿用现有 conductance 后端的方程形式及严格过去的 spike-history 语义。

这项候选改动的科学理由是：使局部 BC 观测有明确位置，并让局部非线性、空间汇总及下游耦合分别可检验。它不是为改善已有 NLL、F2 或 RF 外观而改动正式模型。第一版不学习几何、不建立真实 cone mosaic、不加入命名 AC subtype、自由 decoder、dataset embedding 或跨物种参数层级。

### 1.1 已阅读的代码事实与继承范围

| 本地来源 | 文本确认的事实 | 本设计的处置 |
|---|---|---|
| [正式入口](../models/mechanistic_retina/retipath.py)；[基础 forward](../models/mechanistic_retina/model.py) | 正式 RetiPath 选择 conductance 后端；H1 调制输入供共享 BC，BC 分 direct/broad，AC 来自 broad BC | 保留分支组织；不替换正式入口 |
| [H1](../models/mechanistic_retina/h1_pathway.py) | `state=LP(delay(Gx))`，`feedback=amplitude*G^T state`，下游输入为 `x-feedback` | 保留这一有效反馈形式；明确延迟、状态与返回输出 |
| [局部 BC](../models/mechanistic_retina/local_bc_nonlinearity.py)；[空间 E/I](../models/mechanistic_retina/retipath_spatial_ei.py) | 有 pre/post-pool 候选；正式空间路径当前按 pooled 分支选择 slope；K 是有效空间 mode | 新候选明确改为局部非线性，再汇总；K 不称树突区室 |
| [AC](../models/mechanistic_retina/amacrine_pathways.py)；[时序函数](../models/mechanistic_retina/state.py) | AC 为延迟/低通与负号输出；history 先右移一 bin，再低通 | 复用方程；不以 `abs(AC current)` 构造 inhibitory drive |
| [增益坐标](../models/mechanistic_retina/retipath_canonical_gain.py) | 现有坐标把通路总强度和相对组成分开，消除部分精确 gain gauge | 采用每条 E/I 路径一个总强度和一个二分支 log-ratio |
| [观测/干预合同](../evaluation/mechanistic_retina/mechanism_observation.py) | direct block 保留 broad→AC 和 tonic gE；下游重算；history 固定且严格过去 | 继承这些语义，不直接把旧观测函数套到新形状上 |
| [输入坐标构造](../data/schottdorf_lee_2021.py) | `_cone_positions` 用视场/pooling 计算中心，y 方向翻转；现有输入是校准 L+M Weber drive | 新接口显式携带像素边界/面积；不重标旧 checkpoint 的 degree/pixel |

现有观测函数限制正式模型类型和 `[B,T,X]` 输入；本设计的 `[B,T,P,C]`、节点轴和局部输出需要隔离原型接口，不能声称已经支持。方程形式复用也不表示与旧 checkpoint 数值等价或可直接迁移。

已阅读 [S0 结论](RETIPATH_MULTIOBS_SYNTHETIC_S0.md) 和 [S0.5 结论](RETIPATH_MULTIOBS_S05_H1_OBSERVABILITY.md)：前者报告 BC/direct-pathway 的有限支持；后者保留 `MIXED_OR_OPTIMIZATION_LIMITED`，不代表已证明优化故障。这些是旧报告内容，本轮没有独立重算。它们不授权 S0.6 或重跑旧实验。研究计划引用的六份历史报告均存在；其余报告只核对存在性。

## 2. 计算图、更新次序与逐层语义

### 2.1 同一回路的计算图

```text
X(t, pixels) ──面积积分 Q──> x(t, local nodes)
                              ├──G──delay 1 bin──LP_H──> h(t)
                              │                          │
                              │                    a_H G^T
                              │                          ↓
                              └──────────────(+)──Σ──(−) f(t)  [H1 feedback output]
                                                 │
                                              u_B(t)
                                                 ↓
                                 local fast/slow states q_f,q_s
                                                 ↓
                                    s_B=(q_f, q_f−q_s)
                                                 ↓
                                      local φ_alpha → o_B
                                         ┌───────┴────────┐
                            direct pooling D_k       broad pooling A
                                         │                ↓
                                  mix w_E, G_E     delay 1 bin → LP_AC
                                         │                ↓ a_AC
                                         │         pooling I_k, w_I, G_I
                                         ↓                ↓
                                        u_E              u_I
                                         └────gE/gI───────┘
                                                 ↓
                                    conductance state V_k
                                                 ↓
                                      mean_k(V_k)−V0 = m
                                         ├──LP_adapt──> z_adapt
                                         └───────────────┐
 supplied events strictly before t ─LP_history─> q_hist ─┤
                                                         ↓
                                              logit → Bernoulli p

identity observations: HC→h；BC→o_B；RGC→sampled events with likelihood p
```

这里的 H1 feedback 是现有模型的**有效减法返回路径**：H1 由未调制的 `x` 驱动，而不是由 `u_B` 再驱动自身。不能把这幅图声称为已经实现真实 HC–cone 闭环；v0 不额外加入该闭环。低通状态和 history 有时间递推，所有分支始终存在；progressive 只改变 loss 可见性和可更新参数，不增删计算边。

### 2.2 统一时钟、初态与计算顺序

`dt=1000/150 ms`，bin 编号 `t=0,…,299`。定义

`LP_tau(v)_t = rho*s_(t-1) + (1-rho)*v_t`，`rho=exp(-dt/tau)`。

一 bin 延迟严格为 `delay1(v)_t=v_(t-1)`，负时间输入为已声明的零偏差信号。每条序列独立 reset；`h,q_f,q_s,a_AC,z_adapt,q_hist` 初态均为 0，`V_k` 初态为 `V0=2/9`，`events_-1=0`。这是已知 synthetic 初态，不声称生理已平衡。

每 bin 按以下顺序更新：面积积分 → H1 延迟/状态/反馈 → BC 两个状态/局部输出 → direct 汇总与延迟的 AC 状态/输出 → E/I conductance → V → adaptation → 严格过去 history → logit/p。H1 和 AC 均至少一 bin 延迟，无瞬时循环求解。全序列反向传播覆盖 300 bins，不在 warmup 边界 detach；前 60 bins 只形成状态，不计入 loss。

### 2.3 逐层定义表

表中所有幅值单位均为明示的 synthetic effective units；`normalized conductance` 不是 nS，`V` 不是 mV，`o_B` 不是已标定的 release。所有空间权重见第 3 节。

| 层 | input | state 与更新 | pathway output | observation | 局部连接、共享/专有参数与实现状态 |
|---|---|---|---|---|---|
| 输入积分 | 像素 Weber contrast `X[B,T,P,1]` | 无新增状态 | `x_i=Σ_p Q_ip X_p`，`[B,T,25,1]` | 无标签 | 固定面积积分；新接口，不是 cone 生化模型 |
| H1-like | `u_H=Gx` | `h=LP_tauH(delay1(u_H))`，`[B,T,25,1]` | `f=a_H G^T h`；`u_B=x−f` | HC identity 选择 h 的 5 个节点 | 邻接 G 固定；共享 tauH、aH；h 不依赖 aH；方程继承，几何新设计 |
| BC | `u_B[B,T,25,1]` | `q_f=LP_tauf(u_B)`；`q_s=LP_taus(u_B)`；`s_B=stack(q_f,q_f−q_s)` | `o_B=φ_alpha(s_B)`，`φ(v)=v (v≥0), alpha*v (v<0)` | BC identity 选择 o_B 的 5 节点×2 分支；默认不监督 s_B | tau、alpha 全节点共享；先非线性后汇总是候选改动；s_B 为有效分支状态，不命名 voltage |
| direct BC | 同一 o_B | 无独立状态 | `d_i,k=G_E D_ki Σ_d w_E,d o_B,i,d`；`u_E,k=Σ_i d_i,k` | 无训练标签；d 保留作 evaluator 的局部通路读数 | 固定 D；共享 G_E、delta_E；不新增 BC output 总增益 |
| AC-associated | `u_A,j,d=Σ_i A_ji o_B,i,d` | `a_j,d=LP_tauA,d(delay1(u_A,j,d))`，`[B,T,9,2]` | `j_I,j,k=−G_I I_kj Σ_d w_I,d a_j,d`；`u_I,k=−Σ_j j_I,j,k` | 不提供 AC 标签 | 两个 effective filter 分支，不命名真实 subtype；tauA,d、G_I、delta_I 共享 |
| E/I integrator | `u_E,u_I[B,T,1,2]` | g 与 V 方程如下；`V[B,T,1,2]` | `m=mean_k(V_k)−V0`，`[B,T,1]` | 默认不提供 V 标签 | 保留 K=2 方程形式，固定后端常数；候选包装，不修改现有后端 |
| RGC adaptation/history/readout | m、严格过去的 events | `z=LP_120ms(m)`；`q=LP_40ms(shift1(events))` | `ell=8*m−0.5*z−1*q+b`；`p=sigmoid(ell)` | Bernoulli sampled events；身份映射，无自由观测头 | 仅 b 学习，N_R=1；adaptation/history 常数为 synthetic 默认 |

这里 BC 的两分支 `d={sustained,transient}` 只是 `(q_f,q_f−q_s)` 的计算标签；仅选一个正输入极性的有效 BC 家族。它们不代表两种已识别的生理亚型，不宣称覆盖 macaque 全体细胞。

E/I 方程固定为：

```text
b0 = log(exp(1)−1)
gE = softplus(b0 + uE)                  # fixed normalization rE=1
gI = softplus(b0 + uI)                  # fixed normalization rI=1
gL=1; EL=0; EE=1; EI=−1/3; V0=2/9; Cm=3*20 ms
gTot = gL + gE + gI
Vinf = (gL*EL + gE*EE + gI*EI) / gTot
V_t = V_(t−1) + (1−exp(−dt*gTot/Cm))*(Vinf−V_(t−1))
```

同一 bin 内 g 常值时采用上述仿射指数更新；不增加 Euler 步长选择。固定 `rE=rI=1` 定义新 synthetic 坐标单位，不从 teacher 或训练响应估计 RMS，也不改动旧模型冻结 RMS。固定 output scale=1、logit slope=8、threshold=0，避免另外加入与 b 或路径强度重复的精确缩放坐标。正 conductance 不要求 signed effective drive 非负。

## 3. 物理坐标、节点与张量接口

### 3.1 一个明确的 synthetic 空间

采用已知的平面视角坐标 `deg`，x 向右、y 向上，视场 `Ω=[−1,1]×[−1,1] deg²`。这是 synthetic 坐标，不推断视网膜毫米尺度、偏心度或物种。训练/评价默认 32×32 像素，P=1024，像素边长 1/16 deg、面积 1/256 deg²；仅表示一致性验收另用同域 64×64 网格。64×64 不是额外训练条件。

| 对象 | 位置/数量 | 固定局部权重 |
|---|---|---|
| 局部输入与 BC | `r_i∈{−0.30,−0.15,0,0.15,0.30}²`，N_B=25 | 每节点接收以 r_i 为中心、边长 0.10 deg 的方窗平均 |
| H1-like | 与输入节点共位，N_H=25 | `G_ji∝exp(−||r_j−r_i||²/(2*0.18²))`，距离≤0.36 deg，含自边，每行和为 1；返回映射固定为 G^T，不另归一化 |
| AC-associated | `r_A∈{−0.30,0,0.30}²`，N_A=9 | `A_ji∝exp(−dist²/(2*0.20²))`，距离≤0.36 deg，每行和为 1 |
| RGC / K modes | r_R=(0,0)，N_R=1；K=2 | D 与 I 分别在 BC/AC 节点上按 sigma=(0.15,0.30) deg 的 Gaussian 权重、半径≤0.45 deg，每 mode 和为 1 |

G/A/D/I 是事先声明的**有限节点网络**的连接矩阵，不是对未知域外组织积分后截断的替代物。固定有限网络及其边缘效应属于 teacher/student 共有已知条件。所有像素接收方窗均在 Ω 内；没有缺失输入后重归一化。K 是两组并行加权/积分状态，不是显示分辨率、BC 分支或解剖区室。

定义 `W_i(r)=1/0.01 deg⁻²`（r 在节点方窗内），其余为 0。像素 `pixel_p` 内 X 常值时：

`Q_ip = area(pixel_p ∩ window_i) / 0.01 deg²`，`x_i(t)=Σ_p Q_ip X_p(t)`。

Q 无量纲；像素面积乘入恰好一次。用轴向矩形交叠长度的乘积计算积分，不对像素中心采样 W 后仅作任意归一化。方窗是精确可积且小型的工程默认，不是已知 cone/BC aperture。

### 3.2 三类空间变化分开处理

1. **同一物理场换表示**：从同一连续刺激定义分别计算各网格的像素面积平均，使用各自 Q；geometry、参数和尺度标签不变。常量场应精确一致；一般场要求随细化收敛，不声称有限像素平均能恢复被抹掉的细节。禁止把 coarse image 插值后称为新增物理信息。
2. **真实尺寸改变**：只改变第 6 节刺激 Gaussian 的 sigma_deg；Ω、像素坐标、节点、连接和时间轴保持不变。偏心度不随 stimulus resize 改写。
3. **域外/缺失输入**：任何节点方窗未被有效像素完整覆盖，整条序列拒绝送入该原型；不零填未知域，不裁切再归一化。仅在背景被明确提供且已知时，零 contrast 才表示背景。v0 无动态缺像素。

### 3.3 接口规格

| 接口 | 必填字段与形状 | v0 约束 |
|---|---|---|
| `StimulusBatch` | `values[B,T,P,C]`；`pixel_bounds[P,2,2]`，轴为 `[pixel,xy,lower_upper]`；`pixel_area[P]`；`time_ms[T]`、`dt_ms`；`input_valid[B,T,P,C]`；`sequence_id[B]`、`split_group_id[B]`；背景/校准元数据 | C=1，values 为相对背景 contrast；bounds 无重叠、面积正；t 等间隔；`background_relative=1`、`calibration=synthetic_known_contrast`、`absolute_photon_rate=unknown_not_used`；面积须由 bounds 一致导出 |
| `CircuitGeometry` | `input_xy[25,2]`、`h_xy[25,2]`、`bc_xy[25,2]`、`ac_xy[9,2]`、`rgc_xy[1,2]`；Q `[25,P]`、G `[25,25]`、A `[9,25]`、D `[2,25]`、I `[2,9]`；单位与 geometry_id | geometry 与连接不学习；Q 随网格边界重算，G/A/D/I 不变；节点排序按 y 升序后 x 升序；显示栅格的行顺序单独由 bounds 指定 |
| `StateBundle` | h/f/u_B `[B,T,25,1]`；q_f/q_s `[B,T,25,1]`；s_B/o_B `[B,T,25,D=2]`；a_AC `[B,T,9,2]`；d_E `[B,T,25,K=2]`；j_I `[B,T,9,2]`；uE/uI/gE/gI/V `[B,T,N_R=1,K=2]`；m/z/q/ell/p `[B,T,1]` | 状态、输出、观测分键，不混合 P、N、K、D、C；持续状态 carry 仅含最后一步 h/q_f/q_s/a_AC/V/z/q 及延迟所需前一步输入、events；独立序列不得传 carry |
| `ObservationBatch` | `layer`、`variable`、`selector_node_ids[M]`、`selector_branch_ids[J]`；`target[B,T_o,M,J]`、`valid[B,T_o,M,J]`；`time_indices[T_o]`；unit、noise_model、sigma；sequence/split_group_id | identity index selection；无可学习空间插值；时间需命中 stimulus bin；HC `[B,T,5,1]`，BC `[B,T,5,2]`，RGC `[B,T,1,1]`；未知位置/单位拒绝匹配 |
| `InterventionSpec` | 枚举名、target_port、replacement、time_scope、preserved_nodes、recomputed_nodes、history_condition、schema_id | 新 schema 明示这些字段；复用旧语义，不冒充旧类原有字段；非法/含糊端口拒绝；不原地改参数或复用失效 downstream trace |

HC selector 为中心及 `(±0.30,0),(0,±0.30)`；BC selector 为中心及 `(±0.15,±0.15)` 四角，两分支均可见。未选节点只由 evaluator 评价，不作为额外训练标签。真实 HC voltage 到 h、BC 测量到 o_B 的对应关系均为 `UNVERIFIED`；真实接入需另行确定映射、单位、测量滤波及 noise，本轮不设计任意 affine 来补未知标定。

## 4. 参数、观测 loss 与梯度路由

### 4.1 参数分组及固定 synthetic 默认

下表同时给出一组事先指定的 teacher 值和 student 初始化中心。它们由设计选定，不来自旧 checkpoint、真实数据拟合或结果筛选。teacher 的可学习坐标在将来生成器/evaluator 内冻结；student 只获得结构/边界/初始化规则，不载入 teacher 参数文件、latent 全量或 optimizer state。

| 参数组 | 学习标量 | 允许范围 | teacher 值 | student 初值中心 |
|---|---|---|---|---|
| H 状态 H | tauH | [10,100] ms | 35 | 50 |
| H 返回 F | aH | [0,0.8] | 0.20 | 0.30 |
| BC 动力学 B_s | tauf、taus | [8,40]、[40,160] ms | 15、75 | 22、100 |
| BC 输出 B_o | alpha | [0.05,1] | 0.25 | 0.50 |
| AC 动力学 A | tauA,1、tauA,2 | [20,80]、[80,200] ms | 40、110 | 50、140 |
| E 耦合 E | G_E、delta_E | [0.2,8]、[−4,4] | 2.0、0.4 | 1.5、0 |
| I 耦合 I | G_I、delta_I | [0.2,8]、[−4,4] | 1.2、−0.3 | 1.5、0 |
| readout R | b | [−4,−0.5] | −2.0 | −2.4 |

合计 12 个标量。时间常数区间固定快慢顺序；只学习机制所需的少量连续参数。有限区间使用 `lo+(hi−lo)*sigmoid(raw)`；delta 使用 `4*tanh(raw)`，其余同上。两分支权重 `w_E=softmax(delta_E,0)`、`w_I=softmax(delta_I,0)`，避免两个自由 logits 的平移 gauge。student 在各初值中心对应的 raw 坐标上独立加 `N(0,0.15²)`；三个种子下 A/B/C 分别共享同一份初始化。

固定项包括 geometry、pixel integration、两个一 bin 延迟、归一化 rE/rI、E/I 常数、output scale、readout slope/threshold、adaptation/history 时间常数和强度。没有细胞特异学习参数（N_R=1），没有实验 nuisance/观测头参数。固定只是控制第一轮规模，不等于生理已知；其误设风险属于后续明确失配实验，本轮不扩展。

### 4.2 loss 定义

三份 dataset 记为 H（h）、B（o_B）和 R（events）。前 60 bins 的 loss mask 为 false，其余 v0 全有效。每个 loss 先对每条序列的有效 scalar entries 平均，再对 batch 的 4 条序列等权平均；无有效 target 的 batch 报错，不能补零当标签。BC 的两个分支等权，HC/BC 不因可见节点数不同获得额外权重。

`L_H = mean_seq mean_valid 0.5*((h_pred−y_H)/0.03)^2`

`L_B = mean_seq mean_valid 0.5*((o_B_pred−y_B)/0.03)^2`

`L_R = mean_seq mean_valid [softplus(ell) − events*ell]`

前两项是已知独立 Gaussian noise 的 NLL（省去固定常数）；R 为 Bernoulli NLL，单位 nats/bin。无正则项、loss 自适应平衡或 teacher probability 训练目标。H/B noise 与 spike 观测的信息量不等价；等 dataset 权重只定义研究目标，不声称等信息预算。

### 4.3 按计算祖先定义可更新矩阵

`✓` 表示计算图上可能有梯度，不保证当前输入下非零、优化器列入或实际发生更新。冻结阶段还须取本表与 active-parameter 集的交集。

| 观测/目标量 | H:tauH | F:aH | B_s | B_o:alpha | A:tauA | E | I | R:b |
|---|---|---|---|---|---|---|---|---|
| HC h loss（启用） | ✓ | — | — | — | — | — | — | — |
| H feedback f（仅定义，默认无 loss） | ✓ | ✓ | — | — | — | — | — | — |
| BC input u_B（无 loss） | ✓ | ✓ | — | — | — | — | — | — |
| BC state s_B（无 loss） | ✓ | ✓ | ✓ | — | — | — | — | — |
| BC output o_B loss（启用） | ✓ | ✓ | ✓ | ✓ | — | — | — | — |
| direct transmitted d_E（仅评价） | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | — |
| RGC V/m（默认无 loss） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| RGC events likelihood（启用） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

因此，HC state loss 不触及 aH；BC state loss 不触及 alpha；BC output loss 即使准确也不直接触及 G_E/delta_E 或 AC/I 参数。现阶段只有 RGC loss 约束这些未观测 coupling。可更新、可辨识、有限预算下恢复准确是不同命题，v0 不承诺参数唯一性。

## 5. A/B/C 三条件训练日程（待批准运行规格）

### 5.1 共同设置与预算单位

Adam：lr=0.01，betas=(0.9,0.999)，eps=1e−8，weight_decay=0；梯度总范数裁剪 1.0；无 LR schedule、early stop、选 checkpoint 或延长训练。每 fit 固定 180 次 macro-step/optimizer.step，取最终参数。三个 student seeds 为 2026091801/02/03，共 9 fits；一个固定 teacher。CPU float32 forward/backward，评价归约 float64，随机种子与确定性设置在未来合同中记录。180 steps 是初筛预算，不代表收敛。

一个 microbatch = 同一 dataset、同一尺度的 4 条完整序列。选择**多 microbatch 梯度累积，每 macro-step 更新一次**。每个 dataset 使用事先固定的有序 microbatch 流；B/C 消费完全相同的流及其多重集合，只改变流被消费的时刻。A 消费与 B/C 完全相同的 R 流。每次 dataset forward 都从独立序列初态开始，不在条件之间共享可变状态。

每个 H/B microbatch 的 loss 系数 `c_H=c_B=0.4`，每个 R microbatch 的系数 `c_R=1/3`。**不除以本步 microbatch 数量**。全程每个条件的系数之和见后表；B/C 每个 dataset 的累计 loss 系数均为 60。B 的时间平均目标为 `(L_H+L_B+L_R)/3`；C 的阶段目标随课程变化，但累计 dataset 系数与 B 相同。这只匹配曝光和系数，不使两条优化路径等价。

A 使用 `L_R/3`，保持与 B/C 中每个 R microbatch 相同的梯度系数；该常数不改变 RGC-only 的最优点。A/B 只用于比较本规格下增加观测的效果，计算量增加仍须报告；不能据此声称胜过相同成本的额外 RGC 数据。

### 5.2 每阶段具体日程

| 条件/阶段 | macro-step（含端点） | 每步 microbatch | 可更新参数 | 阶段曝光 H/B/R（batch） |
|---|---|---|---|---|
| A RGC-only | 1–180 | R×1 | 全部 12 参数 | 0 / 0 / 180 |
| B joint-from-scratch | 1–180 | 6 步周期：R 每步；H 在周期 1–5；B 在周期 2–6 | 全部 12 参数 | 150 / 150 / 180 |
| C1 HC 预训练 | 1–30 | H×1 | 仅 tauH；其余冻结 | 30 / 0 / 0 |
| C2 加入 BC 拟合 | 31–60 | B×1 | aH、tauf、taus、alpha；tauH 冻结；下游冻结 | 0 / 30 / 0 |
| C3 回放与联合校正 | 61–180 | H×1、B×1；本阶段奇数步 R×1、偶数步 R×2 | 全部解冻，包括 tauH | 120 / 120 / 180 |
| C 总计 | 180 步 | 阶段课程如上 | 无永久冻结的学习参数 | 150 / 150 / 180 |

所有模块在开始时分配同一份初值；“加入”指开始使用相应观测和解冻，不是切换架构。保留单一 Adam 实例及其状态；冻结参数 `grad=None`，不累计动量更新，首次激活时初始化其 optimizer state；解冻已学习参数保留原动量。C1/C2 仍计算相同前向回路，loss 只连相应观测祖先，不缓存旧 latent 作为新模块的输入。

```text
allocate candidate with paired student initialization
create one Adam for the 12 raw parameter coordinates
for macro_step in 1..180:
    active, ordered_dataset_list = frozen_schedule(condition, macro_step)
    set requires_grad from active; zero_grad(set_to_none=True)
    for dataset in ordered_dataset_list:       # H then B then R; repeated R stays last
        batch = next(frozen_stream[dataset])
        trace = full_circuit(batch.stimulus, strictly_past_history(batch))
        loss = masked_dataset_loss(identity_select(trace), batch.target)
        backward(c[dataset] * loss)           # accumulate, no step and no detach here
    clip_norm(active_gradients, 1.0)
    optimizer.step()
    record dataset IDs, masks, loss weights, active parameters, work counters
use final step only; no development-based model selection
```

H/B batch 不携带 RGC target，其 history 输入声明为全零；这不影响 H/BC 上游计算。R batch 携带本序列历史 events，但 bin t 的 likelihood 只能使用 `<t` events。不能把当期 target 当作当前 history 输入。

### 5.3 曝光与计算账本

| 每 fit 项目 | A | B | C |
|---|---:|---:|---:|
| optimizer.step | 180 | 180 | 180 |
| H/B/R microbatch | 0/0/180 | 150/150/180 | 150/150/180 |
| H/B/R 序列曝光（含重复） | 0/0/720 | 600/600/720 | 600/600/720 |
| H/B/R 有效 scalar target 曝光 | 0/0/172800 | 720000/1440000/172800 | 720000/1440000/172800 |
| 全回路 forward / loss backward 调用 | 180/180 | 480/480 | 480/480 |
| H/B/R 累计 loss 系数 | 0/0/60 | 60/60/60 | 60/60/60 |

H/B 各在自己的两个训练尺度交替取 batch，150 次中每尺度 75 次；R 在三个尺度循环，180 次中每尺度 60 次。每尺度的 24 条训练序列按固定 shuffle 循环取样；不足一轮的尾部保留在预先列好的 stream 中，不按结果补齐。9 fits 共 1620 次更新、3420 次 full forward 和 3420 次 loss backward 调用，均为**拟议计数**。

B/C 有相同标签曝光与 full-forward 次数，但冻结、祖先子图、每步梯度混合及裁剪使实际 backward 运算量不同。未来需记录每阶段实际 forward/backward 调用、活跃参数和墙钟时间；本设计未测量时间/FLOPs，不以相同 step 数宣布计算成本完全匹配。

## 6. 一轮小型 synthetic 规格

### 6.1 teacher/student 边界与生成定义

teacher 使用第 2–4 节同一架构和固定表值；student 不从任何已有 RetiPath checkpoint 初始化。共有信息仅为方程、已知 geometry、参数范围、固定常数、坐标和观测 selector/noise。已知 geometry/identity heads 是强假设，结果不叫结构发现或真实跨数据集对齐。

使用一个固定连续空间刺激族：

`X(r,t)=0.4 * exp(−||r−c||²/(2*sigma²)) * [Σ_(f∈{0.5,1.5,3,6}Hz) sin(2π f t_seconds+phase_f)]/4`。

每条 waveform 的 `c_x,c_y` 从 `{−0.1,0,0.1} deg` 独立选取，四个 phase 从 `[0,2π)` 独立均匀选取；contrast 最大绝对值不超过 0.4。像素值为该连续场在像素内的面积平均（Gaussian 可用 erf 积分），时间值取 bin 中点；序列只在 `[0,2 s)` 提供输入。0.5/1.5/3/6 Hz 是载波，不把有限窗口的开启/截断称为严格带限。150 Hz 是数值时钟，不是刺激带宽。

每序列 T=300，前 60 bins warmup，后 240 bins 计分；无重叠切窗，无跨序列 carry。HC/BC 对选定 identity 输出加独立、零均值、标准差 0.03 的 Gaussian noise；噪声在首次生成后固定，不每 epoch 重抽。RGC 按 teacher 的 `Bernoulli(p_t|teacher sampled events_<t)` 顺序生成一份 events 序列；不得用未来事件或正态概率近似替代这一定义。

teacher probabilities、全部 latent、未可见层标签和干预 trace 分离为 evaluator-only 信息。训练器的输入清单只含 stimulus、当前条件允许的 target 和合法过去 history；不把 teacher 参数表读入 student optimizer。当前文档公开 teacher 设计值用于复核，运行时仍应以文件/API 访问边界保证不复制其学习坐标。

### 6.2 尺度×观测层矩阵

尺度指 Gaussian 的 `sigma_deg`，不是像素数、RF 半径或细胞偏心度。`训练` 表示该层在训练集可见；`组合保留` 表示尺度在其他层已见，但本层标签未用于训练；`全尺度保留` 表示任何训练层均未见该尺度。

| sigma_deg | HC h | BC o_B | RGC events | 角色 |
|---:|---|---|---|---|
| 0.15 | 组合保留 | 训练 | 训练 | 小尺度；HC×0.15 留出 |
| 0.225 | 全尺度保留 | 全尺度保留 | 全尺度保留 | 0.15–0.30 内插测试 |
| 0.30 | 训练 | 训练 | 训练 | 跨层共同锚点，同 waveform/初态/坐标 |
| 0.45 | 全尺度保留 | 全尺度保留 | 全尺度保留 | 0.30–0.60 内插测试 |
| 0.60 | 训练 | 组合保留 | 训练 | 大尺度；BC×0.60 留出 |

多尺度不能与层级完全绑定；0.30 是所有层的共同条件，0.15 和 0.60 还有 RGC 重叠。A 只看到其中 RGC 训练标签。第一轮不加入极端外推、其他 stimulus family 或第四个额外 RGC 条件。

### 6.3 数据量与保留规则

| split | 物理序列数 | 可访问标签/用途 |
|---|---:|---|
| train | 每个训练尺度 24 条，共 72 条 | H：48 条；B：48 条；R：72 条；共同尺度可来自同一物理序列 |
| development | 每个训练尺度 8 条，共 24 条 | 只开放与 train 矩阵相同的标签；最终参数完成后作描述报告，不选超参/步数/seed |
| held-out test | 五个尺度各 8 条，共 40 条 | evaluator 可见三层及全节点 clean trace；其中 16 条为全尺度保留；其余区分已见组合与未见组合 |

基础观测数据共 136 条唯一物理序列；另有第 6.6 节预定的 4 条辅助探针变体，只用于 evaluator。训练唯一有效标签数为 H 57600、BC 115200、R 17280 个 scalar entries。重复曝光量见第 5 节，不把重复当独立样本。

先用固定 master seed=2026091810 分配 waveform-family ID 到 split，再派生 center/phase、noise、events 的独立 RNG 流。`split_group_id` 绑定基础 waveform、其所有尺度版本、重复采样、不同观测层和不同网格表示；一个 family 只能属于一个 split。每个 split 的 family 依序与该 split 允许的各尺度配对，形成上表数量。所有条件共享同一 manifest，不能按当前 RGC 结果筛选刺激或剔除低效应序列。

同一训练 waveform 的“不可见层”不能转作独立测试；组合保留必须在另外 40 条 test 序列上评价。test 的 geometry/刺激定义可供 forward 使用，目标和 teacher 参数只由冻结 evaluator 使用。未来先持久化协议/数据可见性与 split manifest，再生成或接触目标；首次评价前记录测试消费，完成一次固定最终评价。若结果促使改设计，须声明原 test 已消费，不把它重新命名为 untouched test。

### 6.4 唯一主要通路问题与 intervention 合同

**主要问题：局部 BC output 观测是否改善未见尺度上，刺激依赖的 direct BC drive 被阻断后 RGC 条件响应变化的预测？** 选择 direct BC 是因为它紧邻已观察的 BC 输出、同时保留不可由该 loss 直接更新的下游耦合；选择发生在任何新结果之前。

主干预使用 `BLOCK_DIRECT_BC_DRIVE`：把两个 direct 分支传输贡献全部置零，故 `d_E=0,u_E=0,gE=softplus(b0)=1`。BC 的 s_B/o_B、broad 输入、AC state/output、gI 均保留；从原始 V0 起对整条独立序列重算 V、m、adaptation、ell、p，计分仍为最后 240 bins。不可只改缓存 logit 或只减去旧 excitatory current。

正常与阻断使用同一条 teacher 正常试次的过去 events 作为固定 conditioning history；student 和 teacher 也使用相同历史。于是主指标是**给定相同历史的条件计算干预效应**，不是模型自由运行后产生不同 spike history 的总体效应，更不是药理或 BC-cell silencing。阻断时 tonic excitation 保留，BC→AC 支路继续存在。

原型最低只需 `NORMAL` 和这个 block 两个枚举。H1/AC/adaptation 等其他干预的扩展延期；没有全通路总分。正常测试中的 h、o_B、a_AC 等读数属于描述性变量评价，不自动构成新增主要通路假设。

### 6.5 连续评价指标与解释规则

主评价集固定为两个全尺度保留 sigma（0.225、0.45）的 16 条 test 序列，两个尺度等权，再对序列等权；其余 test strata 单独列出。所有指标基于最终参数，不挑 seed、尺度或响应窗口。

令 `Δp=p_block−p_normal`，`Δd_i,k=d_block−d_normal=−d_normal`。

| 层次 | 预定指标 | 使用范围 |
|---|---|---|
| 唯一 primary | `RMSE(Δp_student−Δp_teacher)`，单位 probability/bin | 主评价集；每 seed 的 B−A、C−B 配对差与三个 seed 的均值、范围 |
| 局部通路 | o_B clean RMSE；`RMSE(Δd_student−Δd_teacher)` | 所有节点×分支或 mode 等权；单独列出已观察/未观察节点和保留组合，不能与 primary 相加 |
| 相对干预误差 | 上述 Δp、Δd 的绝对 RMSE 各除以对应 teacher effect RMS | 同时报告分母；分母为 0 时 relative=undefined，保留绝对值；不因相对值很大删样本 |
| 保留响应 | sampled Bernoulli NLL；`excess CE=mean[CE(p_teacher,p_student)−H(p_teacher)]` | 给定同一合法 history，单位 nats/bin；全尺度/组合/已见尺度分别报告 |
| 状态/输出 | h、s_B、o_B、a_AC、V 各自 clean RMSE 与 teacher RMS | evaluator 的描述性结果，无 AC 训练标签；不作任意跨单位平均 |
| 跨 seed 分散 | 所有 seed 对之间的 Δp RMS 距离；同时列各 seed 对 teacher 的 primary 误差 | 描述优化解分散，3 seeds 不代表 posterior 或总体显著性 |

`excess CE` 使用 evaluator-only teacher conditional p，训练中不得使用；不继承旧总 CE 的 1% 保持界限。报告绝对 ΔexcessCE 与 ΔNLL，不造 non-inferiority 阈值。relative 指标只作为尺度解释，不替代 primary 的绝对概率误差。

预先规定解释：若 B 比 A 的 primary 误差降低且保留响应没有相反方向的明显代价，属于本 teacher/预算下的支持性证据；若响应误差上升，则并列报告收益与代价，不称“保持”。若只有 o_B 改善而 Δd/Δp 未改善，仅支持被观察变量恢复。C 对 B 的配对改善/恶化/混合分别描述，不先验宣布 progressive 优胜。若三 seed 方向不一致、效应很小或未收敛，不给科学 PASS/FAIL，也不据此区分结构不可辨识与优化不足。不追加预算或改主指标救结果。

该三条件设计检验含多尺度部分观测时的泛化表现；**没有单尺度、等预算对照，因此不能单独归因“多尺度训练带来增量收益”**。同理，没有额外 RGC 数据对照，不能复用旧研究叙事声称多层观测更具成本效益。

### 6.6 dynamic RF 作为有限辅助读数

不训练 RF head。未来只在 test 中第一个 0.30 尺度 waveform 的两个预定上下文（同 center/phase、contrast 历史分别 0.2 与 0.4，前 60 bins）之后施加同一方窗探针：中心 (0,0)、边长 0.10 deg、t=60 的一个 bin，contrast 为 ±0.01。t≥60 的其余位置/时间全部设为已知零 contrast 背景，仍运行完整 300 bins。比较 t=61…90 的条件 logit 中心有限差分 `(ell_plus−ell_minus)/0.02`，history 在两个上下文和扰动间统一固定为零。

该上下文/探针包与所属 test family 共用 split_group_id，只由 evaluator 在最终评价生成；不是额外训练数据，也不由结果选择。它读出状态依赖敏感度，不能恢复 coarse pixels 丢失的细节，不能证明 RF 一定随刺激尺寸扩张，更不能代替保留响应/primary 指标。±扰动也避免把 PReLU 零点单侧分支导数当成唯一导数。

## 7. 后续最小实现落点与必要验收

以下目录/文件是**待批准后才创建的候选路径**，当前均未创建；正式模型、loader、trainer 和旧实验入口不修改。

| 候选路径（相对项目） | 最小职责 | 阶段 |
|---|---|---|
| `experiments/retipath_multiscale_v0/contracts.py` | 五种接口、显式单位/轴、identity selector、intervention 枚举与输入拒绝条件 | G1 |
| `experiments/retipath_multiscale_v0/circuit.py` | Q 与有限 geometry、12 参数回路、state/output trace、direct block；可复用现有纯时序函数和指数更新函数 | G1 |
| `experiments/retipath_multiscale_v0/protocol.json` | 将本稿 teacher、范围、种子、矩阵、日程、指标与访问边界转成冻结运行配置；不要从旧实验继承默认值 | G2 冻结前 |
| `experiments/retipath_multiscale_v0/synthetic_study.py` | 分开的生成、训练、评价函数/显式子命令；训练函数不加载 evaluator-only 文件；固定 stream 与计数 | G2 |
| `experiments/retipath_multiscale_v0/test_contract.py` | 下列 5 项必要验收；G1 使用手写微型输入，涉及生成器/完整 stream 的部分待 G2 授权 | G1/G2 |

复用时以 `state.py` 的 lowpass/delay/history 和 `retipath_spatial_ei.py` 的 `exponential_sequence` 为候选；节点/分支轴可暂时展平后复原，不能改变定义。现有 integrator 构造器带有旧 gain 参数和形状限制，不能直接实例化后把这些额外 gain 留入 optimizer；新 wrapper 只持有本稿参数。旧 `observe_mechanism` 的类型检查保持原样。

未来必要单元验收最多五项；这里仅列规格，**本轮没有创建或执行测试**：

1. **坐标/积分**：像素面积、完整覆盖与 Q 行和；手写常量场/对齐矩形在两网格得到同一积分；缺域拒绝；平滑场细化仅检验数值收敛，不设生理阈值。
2. **state 与端口**：手算两步递推、零输入静息、carry/reset；direct block 保留 BC/AC 和 tonic gE，重算下游；不得把输出 clamp 等价为状态删除。
3. **观测梯度路由**：在含正负 BC 状态、非饱和参数的手写输入上，确认 HC 不触及 aH、BC output 不触及 E/I coupling、RGC 可连接全部组；对“未连接”与“数值零梯度”分别记录。
4. **causal history**：改变 events_t 或未来 events 不改变 p_t；normal/block 同 history；不同序列不得串 state。
5. **日程/划分合同**：仅文本/索引账本验证 B/C stream 多重集合和权重总数相同、A 的 R 流相同、冻结组正确；waveform/尺度/重复/重采样 group 不跨 split，未见标签不进入训练；G2 时再验证实际访问清单。

## 8. 用户决策、证据限制与停止点

推荐保持以上整套设计，不留下工程常数待自动调优。提交用户的实质决策只有两项：

1. 是否接受 **direct BC block 的条件 RGC 效应**作为下一轮唯一主要通路问题？推荐接受；HC/H1 和全通路比较延期，真实生理对应保持 `UNVERIFIED`。
2. 是否批准进入 **G1 隔离原型与上述必要接口/数值验收**？推荐按第 7 节 G1 文件范围实施。该批准不包含 G2 synthetic generator、数据生成或 A/B/C 训练；G2 需另行批准并冻结本稿预算/划分/访问边界。

局限已进入设计：单 teacher、可实现同家族、已知几何与 identity 观测、仅一个有效 RGC/极性家族、有限时间与尺度、条件干预、180-step screen；这些限制不能由良好拟合消除。真实物种、voltage/release 对应、绝对光强和生物机制均未验证。

G0 在文档落盘、规则备份及文本一致性核对后结束。**尚未执行任何实现、模型运行、checkpoint 读取、测试、训练或数据生成；不自动开始下一阶段。**
