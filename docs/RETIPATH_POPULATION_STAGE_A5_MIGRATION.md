# Population RetiPath v0.1 — Stage A.5 migration bridge

2026-09-18。状态：**已完成指定迁移性质检查；未训练、未调参、未进入 Stage B。**

LegacyPReLU 为 migration anchor；BaselineSoftplus 仅为并列候选。所有模型保持构造器默认参数，使用相同确定性物理输入。本轮验证计算性质，不要求新旧模型逐值相同，不评价哪种非线性更优。

## 1. 七项结论

| 项目 | 标记 | 有边界的结论 |
|---|---|---|
| 1. Physical Q | **PRESERVED** | 同一对齐物理场跨网格一致；光滑场积分误差随网格细化下降；真实 Gaussian scale 改变时响应改变 |
| 2. H1 feedback | **PRESERVED** | block 保留 h_H、将 feedback 归零、重新计算下游；G1 无此原生枚举，参考为经 native forward 核对的固定公式重放 |
| 3. BC nonlinear integration / F2 | **CHANGED_BUT_EXPLAINED** | 三个模型首个可分辨 F2 都在 BC output；幅度/相位不相等，BC dynamics 与输出函数的已声明差异保留 |
| 4. Direct / AC split | **PRESERVED** | 同一 BC output 分流；direct block 不改变 BC/AC；AC postsynaptic block 不改变 direct branch |
| 5. E/I conductance | **PRESERVED** | signed drive 经 softplus 成为正 conductance；对应 block 保留 tonic 1；V/logit 的重新递推符合既有后端 |
| 6. Causality | **PRESERVED** | 改未来 stimulus/event 不影响过去；当前 event 不进入当前 logit；history 符合 strictly-past 递推 |
| 7. BLOCK_DIRECT_BC_DRIVE effect | **CHANGED_BUT_EXPLAINED** | 作用点与 AC 保留性质不变；最终效应幅度改变，新增 OFF 输出的符号不与旧 G1 单输出强行对齐 |

本次没有项目标记为 `NOT_PRESERVED`。`PRESERVED` 只指指定 fixture 上的计算性质；不意味着参数可辨识、生理验证或全图数值等价。`CHANGED_BUT_EXPLAINED` 的两项不是通过调参获得的补救结果。

## 2. 来源差异与实际比较对象

依据：[AGENTS](../AGENTS.md)、[v0.1 design](RETIPATH_POPULATION_ARCHITECTURE_V0_1.md)、[Stage A](RETIPATH_POPULATION_STAGE_A.md)、G1 与 Population 源码。既有模型、参数、测试及历史报告均保持原样。

必须先纠正两处历史来源表述：

1. [G1 Intervention](../experiments/retipath_multiscale_v0/contracts.py:122) 只有 NORMAL / BLOCK_DIRECT_BC_DRIVE，没有 H1/AC block 枚举。既有 formal [intervention semantics](../evaluation/mechanistic_retina/mechanism_observation.py:122) 明确规定 H1 feedback output block 保留 h/state、AC postsynaptic block 保留 AC states。此轮对 G1 缺失的两个 API 采用评价专用的代数重放，不能说旧 G1 已有这些原生接口。
2. G1 [test_contract.py](../experiments/retipath_multiscale_v0/test_contract.py:43) 已使用 Gaussian、aligned square、64 列正负交替的零和条纹，但没有专门的 contrast-reversing F2 协议。此前 [F2 localization pilot](RETIPATH_F2_LOCALIZATION_PILOT.md) 属于更早的 Canonical 模型，不能改称 G1 实验。本轮复用 G1 条纹的空间部分，并按用户要求扩展为固定正弦时间波形；另用一个明确标记为新构造的镜面对称 null-sum fixture。

评价重放入口：[old_replay](../experiments/retipath_population_v0_1/migration_bridge.py:85)。它读取 G1 默认参数，以 G1 原式计算各量，仅在指定作用边将输出置零；不修改参数或模型源码。重放的 NORMAL 与 DIRECT block 在 matched fixture 上，对全部 23 个暴露端口与 G1 native forward 的最大误差均为 **0**。H1/AC block 的旧模型结果因此明确标记为“核对过的公式重放”，没有加载或修改任何历史 checkpoint。

G1 保留一个 effective RGC 和 sustained/transient BC 分量；Population 保留 ON/OFF 两个 RGC、ON/OFF BC 及 local/broad×polarity AC。G1 的 sustained/transient 不能重命名为 ON/OFF，G1 单输出也不自动对应任一 Population 细胞。

## 3. 事前固定的输入、数值规则与工件

首次 forward 前已写入 [protocol.json](../output/evaluations/retipath_population_stage_a5_20260918/protocol.json)、[SOURCE_LOCK.json](../output/evaluations/retipath_population_stage_a5_20260918/SOURCE_LOCK.json) 和 [默认参数快照](../output/evaluations/retipath_population_stage_a5_20260918/DEFAULT_PARAMETERS.json)。源码锁时间为 **2026-09-18T07:37:46.046586+00:00**。这些不是训练 checkpoint。

- 环境：CPU、float64、单线程；Python 3.12.7、PyTorch 2.6.0+cpu；固定 150 Hz。
- matched intervention fixture：G1 原有 `gaussian_pixel_average(sigma=0.30 degree)`，32×32 grid，24 bins；原六项 profile `[0.4,0.25,-0.35,0.1,-0.2,0.3]` 重复四次。事件为 G1 `event_fixture`，Population 两 RGC 接收同一条重复的给定事件序列。
- 网格检查：同一半宽 0.25 degree square，32/64 grid；同一 sigma=0.30 degree Gaussian 的物理积分收敛。真正尺度检查固定 grid=64，sigma=0.15 与 0.60 degree，幅度/时间 profile 不变。
- F2：`G1_FINE_STRIPES_CR` 使用原 G1 64 列交替 ±1 空间场，physical bar width=1/32 degree；`MIRROR_NULL_CR` 使用 sign(x) 两个物理半场。两者每帧空间均值为零；不是基于拟合 RF 权重搜索的 null。
- F2 时间波形固定为 `0.5*sin(2*pi*4*n/150)`，1200 bins；前 900 bins 预热，后 `[900,1200)` 的 300 bins 测量，恰为 8 个 4 Hz cycles。F2 fixtures 的给定 history events 为零。预热长度事前固定，没有按结果延长。
- 同一基线重置，V0=2/9，其余偏差/history 为零。所有 forward 位于 `torch.inference_mode()`，无 optimizer。

一般数值不变量的绝对容差为 2e−12；尺度变化的非零检查沿用 G1 的 probability difference >1e−8。它们是工程核对阈值，不是研究成功阈值或显著性检验。

工件目录：[output/evaluations/retipath_population_stage_a5_20260918](../output/evaluations/retipath_population_stage_a5_20260918)。包括：

- [checks.csv](../output/evaluations/retipath_population_stage_a5_20260918/checks.csv)：374 项数值核对与另外 6 项光滑场误差原值；无核对失败。
- [harmonics.csv](../output/evaluations/retipath_population_stage_a5_20260918/harmonics.csv)：2360 行，逐 component 的 C1/C2 实部、虚部、F1/F2、可分辨时的 phase，以及独立标记的 RMS / SPATIAL_MEAN 行。
- [interventions.csv](../output/evaluations/retipath_population_stage_a5_20260918/interventions.csv)：三个 block 的按模型/RGC 汇总。
- [matched_delta_logit.csv](../output/evaluations/retipath_population_stage_a5_20260918/matched_delta_logit.csv)：360 行 normal/block 逐 bin logit、delta-logit、两个 mode 的 d_E/d_I、gE/gI、V。
- [summary.json](../output/evaluations/retipath_population_stage_a5_20260918/summary.json)、原始 `matched__*.npz` / `harmonic__*.npz` / `scale_*.npz` traces，以及输入及其 factorized 数值数组；完整索引见 [FILE_MANIFEST.json](../output/evaluations/retipath_population_stage_a5_20260918/FILE_MANIFEST.json)。约 16.2 MB，属于固定正确性 fixture 的评价工件，不是 synthetic benchmark dataset。

## 4. Physical Q — PRESERVED

入口：[G1 area_integral_weights](../experiments/retipath_multiscale_v0/circuit.py:70)、[Population forward](../experiments/retipath_population_v0_1/circuit.py:261)。两个模型使用同一固定物理 aperture 面积积分；physical scale 与 numerical grid 分开。

| 模型 | 同一 aligned square：max Δq，32/64 | max Δprobability，32/64 | sigma 0.15→0.60：max Δlogit | max Δprobability |
|---|---:|---:|---:|---:|
| G1 | 1.110223025e−16 | 2.775557562e−17 | 0.06140565265 | 0.004700073473 |
| Population LegacyPReLU | 1.110223025e−16 | 0 | 0.07039332381 | 0.005428283099 |
| Population BaselineSoftplus | 1.110223025e−16 | 0 | 0.03629784482 | 0.002695688005 |

两尺度的 max Δq 均为 **0.3067042617**。同一 sigma=0.30 Gaussian 的 Q 与解析 aperture 平均值的最大误差，32 grid 为 **0.01113993734**，64 grid 为 **0.002449495546**，三模型共享相同结果。这是积分收敛，不将任意有限网格光滑场说成逐位等价。

## 5. H1 feedback — PRESERVED

入口：[G1 feedback](../experiments/retipath_multiscale_v0/circuit.py:146)、[Population feedback/block](../experiments/retipath_population_v0_1/circuit.py:279)、[旧 formal semantics](../evaluation/mechanistic_retina/mechanism_observation.py:122)。比较量均为 block−normal，以下为绝对最大值。

| 模型 | max Δh_H | blocked feedback | max Δs_B | max Δlogit |
|---|---:|---:|---:|---:|
| G1 公式重放 | 0 | 0 | 0.02225552039 | 0.01190628647 |
| Population LegacyPReLU | 0 | 0 | 0.01582669834 | 0.02132479799 |
| Population BaselineSoftplus | 0 | 0 | 0.01582669834 | 0.01157427747 |

保留的是“h/state 留下、feedback output 被切断、下游重新前向”的作用点。G1 固定一 bin H1 delay，而 Population 默认分数 delay=5 ms；Population a_H 为层级单元参数。该时间/参数粒度差异已在设计中声明，本结论不要求 normal h_H/feedback 与 G1 数值一致。

## 6. BC nonlinear integration / F2 — CHANGED_BUT_EXPLAINED

入口：[G1 BC state/output](../experiments/retipath_multiscale_v0/circuit.py:148)、[Population BC state/output](../experiments/retipath_population_v0_1/circuit.py:283)、[Fourier 计算](../experiments/retipath_population_v0_1/migration_bridge.py:166)。

对每个原始 component，在固定 measurement window 上先减去时间均值，再计算

`C_h=(2/300) Σ_n y_centered[n] exp(−i 2π·4h·n/150)`，h=1,2。

F_h=|C_h|；表中为 component amplitude 的 RMS，不是在 Fourier 前取 abs/norm。相位参考为 wave 的 n/150 时钟；stimulus metadata 仍是 bin midpoint。每 component 的数值参照为 `4096*eps64*max_abs(trace_window)`，分组参照采用对应 RMS；它不是生理/统计显著性阈值。

### 6.1 G1_FINE_STRIPES_CR

| 模型 / 分组 | BC input F1 | input F2 | state F1 | state F2 | output F1 | output F2 |
|---|---:|---:|---:|---:|---:|---:|
| G1 sustained | 0.08753821800 | 7.89378e−17 | 0.07669699161 | 5.26377e−17 | 0.05752274383 | 0.008136684345 |
| G1 transient | 0.08753821800 | 7.89378e−17 | 0.05549185819 | 5.29221e−17 | 0.04161889375 | 0.005887144191 |
| Population LegacyPReLU ON/OFF | 0.08748204931 | 7.88498e−17 | 0.07125811860 | 5.37239e−17 | 0.05344358914 | 0.007560095776 |
| Population BaselineSoftplus ON/OFF | 0.08748204931 | 7.88498e−17 | 0.07125811860 | 5.37239e−17 | 0.03562905930 | 0.0004068153592 |

### 6.2 MIRROR_NULL_CR

| 模型 / 分组 | BC input F1 | input F2 | state F1 | state F2 | output F1 | output F2 |
|---|---:|---:|---:|---:|---:|---:|
| G1 sustained | 0.4234480728 | 3.88332e−16 | 0.3710058764 | 2.71316e−16 | 0.2782544090 | 0.03936547378 |
| G1 transient | 0.4234480728 | 3.88332e−16 | 0.2684304175 | 2.62258e−16 | 0.2013228144 | 0.02848208187 |
| Population LegacyPReLU ON/OFF | 0.4218308574 | 3.87966e−16 | 0.3436004701 | 2.97475e−16 | 0.2577003542 | 0.03645787628 |
| Population BaselineSoftplus ON/OFF | 0.4218308574 | 3.87966e−16 | 0.3436004701 | 2.97475e−16 | 0.1718002350 | 0.008200783672 |

ON/OFF 行合并仅因为两组 **RMS amplitude** 相同，不能据此合并其逐 node trace 或相位。CSV 保留两组原始 component 与 phase；G1 input 是同一个 u_B，表中为两个时间分量重复列出对应输入，不表示 G1 有两个 BC input family。

### 6.3 首次非线性位置与 null-sum

六个模型×fixture 组合的首个可分辨 F2 都是 **BC output：G1 o_B / Population delta_r_B**。从 q、u_H、h_H、feedback、u_B 到 s_B，最大的分组 F2 为 **3.931688657e−16**；相对其数值参照的最大比值不超过 **0.001075**。不存在由当前结果支持的 H1/BC state 首发 F2。

两个输入每帧空间均值严格为 0；max |mean(q)| 分别为 **4.99600e−18** 和 **3.10862e−17**。BC input/state 的空间平均 F1 最大为 G1 **1.37066e−18**、Population **1.81301e−18**。但在**逐节点非线性之后再做空间平均**，输出 F2 保留下来：

| 模型 / 分组 | 条纹：空间平均 output F2 | 镜面 null：空间平均 output F2 |
|---|---:|---:|
| G1 sustained | 0.006932851958 | 0.03519200583 |
| G1 transient | 0.005016138477 | 0.02546245375 |
| Population LegacyPReLU ON/OFF | 0.006445413777 | 0.03259318645 |
| Population BaselineSoftplus ON/OFF | 0.0003165726847 | 0.007330615475 |

空间平均 output 的 F1 并非一律数值零：LegacyPReLU 的两个 fixture 分别约 **3.66623e−6 / 1.84077e−5**，BaselineSoftplus 约 **2.36e−18 / 3.68e−18**。这些离散测量值原样保留；未通过改 phase、window、alpha 或 bias 将其“修正”为零。

softplus **改变了 BC F2 幅度，没有改变本 fixture 下的 first-nonlinearity locus**。Population 两接口的 BC input/state 相同，输出差异来自接口；与 G1 的比较还同时包含 parallel sustained/transient → cascade+kappa、ON/OFF 节点化与 delay 差异。没有把 F2 变大或变小解释成性能优劣、真实 release 验证、Y-cell signature 或生物 subtype 证据。

## 7. Direct / AC split — PRESERVED

入口：[G1 分流](../experiments/retipath_multiscale_v0/circuit.py:150)、[Population 分流](../experiments/retipath_population_v0_1/circuit.py:294)。新图从同一 delta_r_B 计算 BR 和 BA；从同一 delta_o_A 计算 AR。独立展开 coupling×固定空间权重的最大残差为 **3.469446952e−18**。

matched fixture 上，三模型 direct block 的 h_H、feedback、s_B、BC output、AC input/state/output、d_I/gI、history 最大差值均为 **0**；c_E/d_E 为 **0**。AC postsynaptic block 同样保留 BC/AC state/output，且 direct d_E/gE 最大差值为 **0**；c_I/d_I 为 **0**。

因此保留的是相同输出分流与独立的传输作用点。Population 新增的 ON/OFF、四 AC family 和 AR routing 改变贡献数值，不等于旧 G1 两个时间分量被重新命名，也不把两条路的拟合强度称为已恢复的生物耦合。

## 8. E/I conductance — PRESERVED

入口：[G1 conductance/V](../experiments/retipath_multiscale_v0/circuit.py:162)、[Population conductance/V](../experiments/retipath_population_v0_1/circuit.py:307)。共同作用链为 signed d_E/d_I → softplus(b0+drive) → 正 gE/gI → V，其中 b0=log(expm1(1))。

| 模型 | matched normal 最小 gE | 最小 gI | direct block 的 gE | AC postsynaptic block 的 gI |
|---|---:|---:|---:|---:|
| G1 | 0.9822361691 | 1.0000000000 | 1 | 1 |
| Population LegacyPReLU | 0.9475251310 | 1.0001159012 | 1 | 1 |
| Population BaselineSoftplus | 0.9490293181 | 1.0000028817 | 1 | 1 |

正常及三个 block 的 gE/gI 均为正，tonic 误差在 2e−12 容差内。每条 trace 从原 V0 重新递推 V 与 adaptation，再计算 logit；独立逐 bin 递推与输出的最大误差为 **4.440892099e−16**。没有把 block 实现为 gE/gI=0，也没有冻结 normal V 或对 signed drive 取 abs。

## 9. Causality — PRESERVED

入口：[G1 forward](../experiments/retipath_multiscale_v0/circuit.py:129)、[Population forward](../experiments/retipath_population_v0_1/circuit.py:261)、[共享 causal/history 函数](../models/mechanistic_retina/state.py:21)。

固定 cut=8，同时把 n≥8 的未来刺激乘 −3、反转未来 binary events：三模型全部暴露端口在 n<8 的最大差异均为 **0**。只改变 n≥8 events 时，logit 的 n≤8 前缀最大差异也为 **0**；n>8 出现非零改变，证明 history 既严格过去、又实际连接输出。

保存 trace 的 history 与 `H_t=exp(−dt/40)H_(t−1)+(1−exp(−dt/40))*event_(t−1)` 独立递推相符。该结论限于 independent reset、fixed supplied history；没有自主 spike feedback、流式 carry 或未来信息参与。

## 10. BLOCK_DIRECT_BC_DRIVE effect — CHANGED_BUT_EXPLAINED

入口：[G1 direct mask](../experiments/retipath_multiscale_v0/circuit.py:153)、[Population direct mask](../experiments/retipath_population_v0_1/circuit.py:294)。**delta-logit 固定为 block−normal**。以下为同一 matched Gaussian、同一给定事件的直接保存结果；drive RMS 归约时间与两个 effective modes。

| 模型 / 输出 | normal direct d_E RMS | normal AC d_I RMS | block 后 ΔAC RMS | delta-logit mean | delta-logit RMS | delta-logit [min,max] |
|---|---:|---:|---:|---:|---:|---|
| G1 effective RGC | 0.07780310567 | 0.03195042194 | 0 | −0.06004304437 | 0.06474122917 | [−0.1082847158, −0.02605120425] |
| Population LegacyPReLU ON | 0.07516479279 | 0.004314335082 | 0 | −0.06097611512 | 0.06562621160 | [−0.1046323521, −0.02525496336] |
| Population LegacyPReLU OFF | 0.03953508896 | 0.004314335082 | 0 | +0.02604857065 | 0.02957187339 | [+0.003746866764, +0.04890481363] |
| Population BaselineSoftplus ON | 0.03879753736 | 0.0001705000980 | 0 | −0.02967963565 | 0.03238269753 | [−0.05245823067, −0.01006655298] |
| Population BaselineSoftplus OFF | 0.03716641799 | 0.0001705000980 | 0 | +0.02813078469 | 0.03070235882 | [+0.009262405584, +0.04932882742] |

全部 direct block 后 d_E RMS=0，AC d_I RMS 保持表中 normal 值，gE=1。正常/阻断 logit 均值分别为：G1 −2.469920681 / −2.529963726；Legacy ON −2.448894905 / −2.509871021、OFF −2.535919591 / −2.509871021；Softplus ON −2.477168415 / −2.506848050、OFF −2.534978835 / −2.506848050。

Population OFF 的效应符号与旧 G1 单输出不同，但符合当前作用点语义：matched fixture 下 OFF 的 normal mean d_E 为 Legacy **−0.02106867711**、Softplus **−0.02309135938**，block 将该基线相对负 drive 归零、恢复 tonic excitation，因此 logit 上升。ON 的 mean d_E 分别为 **+0.04972561925 / +0.02410483820**；旧 G1 为 **+0.04956470263**。不能把“只移除 stimulus-dependent direct drive”解释为所有刺激下 logit 都应下降。

这里的 direct contribution 与 AC contribution 分别指 d_E 与 d_I，原始逐节点 c_E/c_I 也已保存；**最终 logit 不能写成两条贡献的简单加和**。conductance 映射、E/I 分母、膜状态和 adaptation 共同决定效应。AC 在 direct block 中数值不变，不表示它对最终 logit 无作用。

## 11. 核验、修改范围与停止

执行入口：[migration_bridge.py](../experiments/retipath_population_v0_1/migration_bridge.py:210)。本轮仅新增该评价脚本、本报告和新的评价目录，未修改 Population circuit/Stage A tests 或 G1/G2/G3 工件；无新增训练参数、optimizer 或机制。三模型参数数目分别保持 **12 / 359 / 359**，所有运行前后的 parameter/buffer 值逐元素相同。

19 个锁定旧源码/规则/报告文件的 SHA256 前后一致，见 [SOURCE_AFTER.json](../output/evaluations/retipath_population_stage_a5_20260918/SOURCE_AFTER.json)。该核对包括 G1 实验目录的旧 Python 文件与 protocol、Population 既有文件、设计/Stage A 文档和 G2/G3 报告；不冒称重新审计了全部旧 checkpoint 或数据。

保存后从 NPZ 使用独立 `numpy.fft.rfft` 重算全部 F1/F2 与复杂系数，在 2360 个 harmonic rows、16120 个标量比较中，FFT 与原 DFT 最大误差为 **2.275892974e−15**。360 行 delta-logit 与保存的 block−normal 减法误差为 **0**；35 个原始工件的 manifest 核对通过。结果见 [ARTIFACT_VERIFICATION.json](../output/evaluations/retipath_population_stage_a5_20260918/ARTIFACT_VERIFICATION.json)。这次核对没有重新运行模型；它是本执行者的独立数值方法复算，不冒称独立人员审计。

没有训练、调参、synthetic benchmark、结果驱动改架构、非线性选择、BC coupling、AC→BC 或 outer nonlinearity。未追加 RF/模型失配实验。**完成后停止；不进入 Stage B，不作下一阶段研究决策。**
