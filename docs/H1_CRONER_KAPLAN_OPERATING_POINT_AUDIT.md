# Croner–Kaplan MC center：operating-point / small-signal audit

日期：2026-09-15。**Measurement 建议：SYMMETRIC_SMALL_SIGNAL_PREFERRED。**

本结论只涉及 measurement 的数值定义。两个预定 checkpoint 的 float64 对称 small-signal 响应通过预先规定的收敛与 harmonic 一致性检查；exact-zero autograd 不适合作为默认 primary。没有执行正式 9-cell physiology measurement，没有读取 external reference target、natural movie 或 spike target，没有计算 B/E/Q，也没有训练或修改正式模型、checkpoint、FOV、support、preprocessing。

**剩余边界：operating point 可以明确，但完整 quantitative radius contract 尚不能直接视为就绪。** 两个诊断的 complex DoG 拟合仍存在较大空间残差和 center/surround 近重合，见后文。数值稳定不等于 fitted radius 已具有可靠的生理含义。

## 1. 预先固定的诊断范围

- Canonical frozen checkpoints：67#6（OFF MC）、67#7（ON MC），各 seed `2026091301`。选择早于本轮 checkpoint response/radius 读取；没有扩展到其余 cells/seeds。
- 从 canonical migration manifest 定位实际文件并逐一核对 SHA256、schema 和 strict state loading。只构建冻结 inference copy，不写入任何 checkpoint。
- epsilon：`1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9` Weber，float32 / float64；CUDA，TF32 关闭。
- 背景为 0 Weber，history 固定为零。保留模型默认初态：H1/AC/adaptation/history 的初始偏离为零；膜电位初态为 **V0 = 2/9**，不是数值 0。
- 4 Hz 是本轮待审计 measurement 的指定频率。本轮不重新裁定论文 protocol，也不宣称复现有限 contrast F1。
- 9 个空间诊断方向：4 个固定 unit pixels、full field、x/y cosine、checkerboard、固定随机符号图。全部在响应前写入配置；不由 support、response 或 fitted radius 选择。

数值判据提前保存在 [protocol.json](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_operating_point_audit_20260915/protocol.json)：配对 response peak 大于 `128 × machine epsilon × max(1, max|logit|)`；相邻 epsilon 的归一化 complex profile L2 差异和 diagnostic radius 相对变化分别不超过 0.001；至少连续 3 点。B 方向一致性、C–B harmonic 一致性、相邻末段 F1 和 150/450-bin transfer 差异各不超过 0.01。**这些都是数值容差，不是 physiology 接受阈值。**

## 2. 零背景的实际 forward 与非光滑点

审计入口为 `CanonicalGainRetiPath`，继承正式 `RetiPath` 的 spatial conductance forward。零输入下，graph drive、H1 state/feedback、BC direct/broad、AC state/drive 均为零；gE = gI = 1。理论膜偏离与 adaptation 为零，logit 保留 threshold/bias 项。

| 源码位置 / 运算 | 零背景实际正向值 | 左 / 右局部行为与当前 autograd 约定 | 对 input→logit spatial derivative 的影响 |
|---|---|---|---|
| [retipath_canonical_gain.py:133](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/retipath_canonical_gain.py:133)：完整 BC 分支 `S=sum(weighted)`；`s=where(S>=0,1,alpha)`；各 mode 乘同一个 s | 所有 S=0，s=1，各 mode 输出=0 | 分支总输出为 S<0 时 alpha×S，S>0 时 S。当前 `>=` 在 0 选择 s=1；比较条件不传导梯度，代码给 mode 路径的局部乘数为 1。左右斜率一般不相同 | **实际命中的关键 kink**；发生在 direct/broad 的 sustained/transient 四个完整分支，均可传播到 E/I、膜和 logit。不能改成逐 mode 独立整流 |
| [h1_pathway.py:78](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/h1_pathway.py:78)：graph、delay、lowpass、subtraction | 全部为 0 | 对输入线性，正负扰动使用相同线性算子 | 没有输入零点 kink |
| [state.py:68](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/state.py:68)：delay 的 floor、整数索引 clamp、valid mask | 输出为 0；索引由时间与冻结 delay 决定，前缀前的采样按零处理 | floor 对 delay 的导数约定不等于对输入的导数；整数索引无输入梯度。冻结 delay 下 gather/interpolation 对输入是线性的 | 不构成 stimulus operating-point 歧义；不得把 padding 的边界误作 input kink |
| [pathway_temporal.py:25](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/pathway_temporal.py:25)：temporal kernel norm / clamp_min | 参数生成的 kernel 非零，norm>0；对零刺激输出为 0 | norm/clamp 的自变量是冻结 kernel，不是刺激。scalar clamp 在下界的 PyTorch 导数为 1，但这里不在该输入边界 | 不构成 input→logit 的零点歧义 |
| [temporal_parameters.py:57](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/temporal_parameters.py:57)：tau bounds 的 minimum/maximum | 由冻结 raw parameters/bounds 决定，与零刺激无关 | min/max 对相等浮点操作数的梯度分配有约定；它们不接收 stimulus。dtype-dependent epsilon 会轻微改变 float64 copy 中的派生 tau | 不构成 stimulus kink；precision 比较中单独保留数值差异 |
| [graph.py:24](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/graph.py:24) 与 BC spatial basis 构造：distance/support mask、归一化 | 冻结 graph/support/空间权重；输入 0 时输出 0 | 距离为零的 norm、距离阈值等属于 geometry 构造；本轮 geometry 不是求导变量。BC 空间归一化分母为冻结正值 | 不对刺激引入 piecewise support，不允许据此改变几何 |
| [shared_subunits.py:99](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/shared_subunits.py:99)：N=1 mixer | connection matrix=[1] | 正式逐 cell checkpoint 直接返回单位矩阵；多 cell 分支中的 clamp normalization 未执行 | 无额外 stimulus kink |
| [amacrine_pathways.py:63](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/amacrine_pathways.py:63)：delay/filter 与 `-gate*state` | state/current=0 | 冻结 gate 下为线性有符号变换；没有 abs、rectification 或 stimulus-dependent gate | 只传递上游 BC 的非光滑性，没有第二个 AC 零点整流 |
| [retipath_canonical_gain.py:92](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/retipath_canonical_gain.py:92)：固定 RMS、effective drive、softplus | effective E/I=0，softplus 自变量 b0=log(e−1)，gE/gI=1 | 两侧导数相同，softplus 对归一化 drive 的导数为 1−exp(−1)。RMS 固定且正；exp/softmax 参数映射也光滑 | 没有 zero-drive kink，未处于 softplus 高值线性分支阈值 |
| [retipath_spatial_ei.py:65](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/retipath_spatial_ei.py:65)：G、V、解析指数递推 | G=3，target=V0，初态 V0；membrane 偏离理论为 0 | 分母 G≥gL=1；expm1 与逐 bin 状态更新在此光滑；保留完整状态依赖 | 没有 abs、max、state detach 或零分母 |
| [retipath_spatial_ei.py:152](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/retipath_spatial_ei.py:152)：adaptation、history、logit/sigmoid | stimulus-driven adaptation 理论 0；zero history=0；logit 一般非零 | adaptation 是线性低通；固定 history 只加到 logit；sigmoid 光滑 | 没有新增 kink。history 可能改变 probability 导数，但不改变本轮 logit 导数 |
| [rgc_state.py:62](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/rgc_state.py:62)：旧 `abs(total_current)` divisive normalization | **正式 conductance forward 不执行此函数** | abs 在 0 的左右导数为 −1/+1，PyTorch 返回 0；只是未执行路径的代码事实 | **不能列为本模型实际零点问题**；本轮未重新叠加旧 normalization/膜低通 |
| alpha/history 参数投影 clamp、旧 local/pre-pool 分支、clamp 枚举 | 非本轮 stimulus forward；无参数更新，无 intervention | 不把训练时参数投影或未执行 variant 的导数混入本轮分析 | 无额外实际输入零点 kink |

scalar fixture 使用安装的 PyTorch 2.10.0 实测：alpha=0.2 时，当前 `where(S>=0)` 的 0 点导数为 1；`where(S>0)` 或 `F.prelu` 返回 0.2，尽管 scalar 前向函数完全相同。两种写法的 symmetric slope 都为 0.6。另测得 abs(0) 的导数为 0、clamp_min(0) 为 1、maximum(x,0) 的 tie 为 0.5、ReLU(0) 为 0。后几项用于区分代码约定，不代表它们均在正式 forward 上执行。自动微分在不可微点需要额外约定，不能把返回值当成唯一数学导数；参见 [PyTorch 2.10 官方说明](https://docs.pytorch.org/docs/2.10/notes/autograd.html#gradients-for-non-differentiable-functions)。

### BC 的对称响应为何不同于 exact-zero autograd

对完整分支，

`phi_alpha(S) = (1+alpha)/2 * S + (1-alpha)/2 * |S|`。

所以标量 odd/symmetric 部分的斜率为 `(1+alpha)/2`。零点后的其余 active 运算光滑时，对一般非退化方向，B 的 small-signal 极限相对于当前 A 应有这个乘数。两诊断的 alpha 为 0.200417 和 1.366463，对应 0.600209 和 1.183231。实测 B 与这个缩放后的 A 相符，epsilon=0.01 时完整 150-lag kernel 的 relative L2 error 为 `1.08e-5`、`2.08e-5`。

因此 **A 的零点 gain 定义任意，不应直接冻结为 primary**。这里所有 BC 分支共用一个 alpha，故 A/B 差异在小信号极限主要是整体 gain；不能反过来声称 exact-zero 一定使 normalized profile 或 fitted radius 发生很大变化。

还有一个需要明确的边界：当前 K=2 实现是各 mode 乘完整分支的共同 slope。如果某方向使 `u1+u2=0` 而两个 mode 都非零，`>=` 与 `>` 会给这两个 mode 不同输出；重加和仍为零，但下游按 mode 处理未必等价。synthetic `[1,-1]` fixture 展示了这个 forward tie 情形。**B 不使用任意 autograd kink 导数，但不能被宣称为整个非光滑网络在任意方向上的唯一 Fréchet Jacobian。** 本轮支持的是下面明确的 pixel-impulse / odd-response 操作性定义，以及对预定方向的 harmonic 检查。

## 3. 三种 measurement 的精确定义

令 z 为 logit，dt=1/150 s，X0 为全零 stimulus prefix。

**A，diagnostic only：** 对 150-bin 前缀末端 `z[149]` 求所有 150×289 输入的 autograd gradient。没有使用 `effective_rf()` 返回的末 16 lag；正式 wrapper 中该截断不能用作本轮完整 kernel。

**B，对称 small-signal：**

`D_sym(U;eps) = [z(X0+eps U)-z(X0-eps U)]/(2 eps)`。

对每个 pixel p，在 bin 0 施加正/负 unit impulse，记录 450 bins 的配对 logit，形成 increasing-lag kernel `k_eps[tau,p]`。零背景、冻结参数、默认静息初态下的 shift covariance 给出有限前缀末端的等价 coefficient；两个预定 pixels × 两个 impulse times 的直接核对在选定精度/epsilon 上 max abs error=0。

`H_eps,L(p) = sum_{tau=0}^{L-1} k_eps[tau,p] exp(-i 2 pi 4 tau /150)`，L=150 或 diagnostic L=450。

这里 J/k 对 **bin stimulus value** 求导，不额外乘 dt，也不套 FFT amplitude normalization。输入 pixel 的单位扰动作用于当前整格 Weber 值。另对 9 个空间方向的 cos 与 negative-sin 时间方向直接算 D_sym，核对 impulse coefficient 的线性组合；没有只凭单 pixel 的收敛假定所有方向自动成立。

**C，harmonic diagnostic：**

`X[t,p] = eps * U[p] * cos(2 pi 4 t/150)`；每次共 450 bins=12 cycles。前 300 bins=8 cycles 为 pre-roll，理由是 2 s 覆盖 8 倍既有 AC tau prior 上限 250 ms；用后两个相邻 4-cycle 窗口判断剩余 transient，不按结果追加周期。

`F1_eps(U) = (2/(150 eps)) sum_{t=300}^{449} [z(X)[t]-z(X0)[t]] exp(-i 2 pi 4 t/150)`。

同时保存 0:150 和 150:300 窗口。比较 `F1_eps(U)` 与 `sum_p H_eps,450(p) U[p]`。这是离散时间的 small-amplitude logit harmonic，不是原论文有限 contrast、spike-rate F1 的复现。带 kink 的采样非线性还可能留下小 harmonic alias；scalar fixture 已显示残差不必在 eps→0 时精确为零，不能据此要求逐位相等。

## 4. 数值结果与 epsilon 决定

两 cells 的共同 float64 稳定区覆盖预定全部网格 `1e-2…1e-9`；最大相邻 normalized-profile 差异约 `1.29e-4`，最大相邻 diagnostic rc 相对变化约 `2.35e-5`，均小于预定 0.001。float32 没有通过连续 3 点共同稳定区要求；低于数值 floor 的全零差分记作不可测，zero-denominator relative error 保存为 null，**不算收敛或 harmonic 一致**。

按事前“共同最宽稳定区内选最大的 epsilon”规则，建议测量数值设置为 **float64 inference copy，epsilon=0.01 Weber**。没有选取更接近外部 physiology 的 epsilon，没有读取该外部目标。

以下都是选定 epsilon 的方法学 diagnostic，不是 population physiology 结果：

| 数值量 | 67#6 | 67#7 |
|---|---:|---:|
| 配对 response peak / 数值 floor | 6.51e10 | 1.07e11 |
| B 直接方向 FD vs pixel-kernel projection，relative L2 | 0.0260% | 0.1053% |
| C 末段 F1 vs B450，relative L2 | 0.0415% | 0.1157% |
| 9 个方向中最大的单方向 C–B relative error | 0.0583% | 0.1243% |
| C 末段 vs 前一 4-cycle 窗口，relative L2 | 8.47e-10 | 5.00e-10 |
| B150 vs B450，relative L2 | 1.89e-8 | 2.15e-8 |
| C 最初 4 cycles vs 末段，relative L2 | 5.363% | 5.210% |
| float64 全 epsilon 网格 diagnostic rc 相对跨度 | 0.00235% | 0.000113% |

150-bin kernel 的 4 Hz transfer 在这两个 checkpoint 上已非常接近 450-bin 结果；**这只支持此有限前缀 measurement 的数值充分性，不把 150-bin 刺激开始后的 F1 称为 steady state。** 最初 4 cycles 的 F1 仍与 pre-roll 后相差约 5%，正是两种量不能混称的实例。不能由两 cells 自动推出其余七个 cells 的时间尾部均合格。

### precision 改变了什么

float64 copy 保留同一 checkpoint 张量值，不做参数更新；graph/support buffers 也保留原值，不重建更高精度几何。但参数非线性重新以 float64 求值，且 ordered tau 的 epsilon 依 dtype 决定，因此不能声称所有派生量逐位不变。

exact-zero A kernel 的 float32/float64 relative L2 差异为 `1.14e-7`、`2.91e-7`。在 eps=0.01，B 的 complex projection 差异为 0.569%、0.0874%，normalized-profile 差异为 0.557%、0.0872%，diagnostic rc 差异为 −0.158%、−0.0148%。float64 主要改善差分消减误差，但确实改变了这些有限精度结果。

float32 零背景膜偏离 max 为 `9.88e-7`、`1.67e-7`，float64 为 `1.84e-15`、`3.12e-16`；gE/gI 均为 1。配对差分使用相同模型/精度的背景，未人为把膜状态清零来修饰结果。

## 5. History 与完整 state dependency

正式 forward 的 logit 为：

`z(X,h) = slope*(membrane(X)-threshold) - adaptation_gain*adaptation(X) - history_gate*history_gain*history_state(h) + response_bias`。

固定 observed history 不进入 H1、BC、AC、E/I、膜或 adaptation，因此数学上 `partial z / partial X` 与 h 无关。相同 h 也在 B 的正负差分中严格抵消。本轮将 zero history 与每 17 bins 一个固定 event 的 history 比较：上游记录的 states 全部逐位相同；gradient max abs difference 在 float32 最大 `7.45e-9`、float64 最大 `2.17e-19`；加性 logit 公式误差分别最大 `8.63e-8`、`1.59e-16`。

最初的逐位 gradient 检查被 GPU 重复计算差异触发；相同 zero history 重复 backward 也得到同样 `7.45e-9` 差异，开启 deterministic 标志仍非逐位一致。该工程问题及数值比较修正已记录，未将实际 history 依赖忽略掉。

**结论：FIX_HISTORY_ZERO 可以作为清楚的条件标签，不是本轮 logit spatial derivative 的未解决 operating-point 参数。** 若改用 probability、free-running history 或 history 随刺激改变，此结论不再自动成立。

## 6. Complex DoG 诊断的实现与解释限制

复用现有 `complex_dog_fit`：完整 17×17 complex coefficients 的 real/imag 联合 residual，固定 checkpoint functional center；pixel-integral Gaussian 的 rc 为 1/e characteristic radius。两个独立 complex gains 通过线性最小二乘求解，实数 rc 与正 gap=rs−rc 通过已有三个固定 starts 优化。Gaussian 采用单位积分归一化；自由 complex gain 吸收其幅度尺度。不使用 J²/P、support radius 或 point approximation。

本轮识别并修复了**仅属于 diagnostic handoff 的坐标顺序错误**：checkpoint 行顺序为 y 从正到负，fitter 为 y 从负到正。现在在传入 fitter 时反转行顺序，并核对 checkpoint 的全部 cone coordinates。没有移动 center、旋转模型、修改像素值或扩展 FOV。已从同一份保存的 transfer coefficients 重算所有拟合项，没有为此重新运行 forward。修正前的 radius/拟合残差无效，不参与最终结论；原始 hash 和修正记录保存在 protocol。

一个独立 synthetic complex DoG fixture（rc=0.08 deg，rs=0.32 deg，非零固定 center）验证了 phase projection、complex gains、pixel integral 和该坐标适配；projection max abs error `2.70e-16`，rc 恢复到浮点精度。未使用任何 external range 或模型响应来生成该 fixture。

**坐标修正后仍有真实的 measurement 限制：** eps=0.01 时两个 model diagnostic 的 relative spatial fit residual 为 **0.470、0.544**，且 rs−rc 都到达既有 gap 下界 0.001 deg。这不是 epsilon 不稳定，也不是 optimizer 未报告 success；它说明这两个 complex profiles 在此固定圆对称 DoG 下拟合质量有限，center/surround 分解可能退化。没有调 bounds、中心、拟合函数或 loss 来改善它。

因此本轮只用 diagnostic rc 的变化量检查数值收敛。**这些 radius 不能直接作为有效 biological center-radius estimate，也不能把稳定拟合当成 physiology 一致。** 下一次完整 measurement preregistration 需要先明确这种 DoG 失配/近重合情况下的可解释范围和结果处理；本轮不新增任意质量阈值，也不修改原 fitter。

## 7. 建议及停止点

建议采用 **SYMMETRIC_SMALL_SIGNAL_PREFERRED**：zero Weber mean、fixed zero observed history、实际默认动态初态、完整 150-bin pixel-impulse symmetric logit coefficients，再投影 4 Hz；采用 float64 inference copy 和事前数值规则选出的 eps=0.01。把结果称为 **finite-prefix symmetric small-signal logit transfer**，不要称 exact-zero Jacobian、唯一线性化或论文 steady finite-contrast F1。

该建议由 B 的数值稳定、直接方向一致性及 C 的 harmonic 结果支持。zero-autograd 分支问题已找到可操作的替代定义；history 不再构成阻碍。但 **尚不建议直接开始正式九-cell radius physiology comparison**：先在最终 contract 中解决上述 DoG measurement 失配的解释/处理，并保留后续 cells 的 numerical-failure 处理规则。没有根据 external agreement 作任何决定。

本轮到此停止：未冻结新的正式 9-cell measurement contract，未计算 B/E/Q，未扩展实验。

## 8. 最小保存结果与可追溯性

- [diagnostic script](D:/PythonProject/retina_rf_SNN/work/croner_kaplan_operating_point_audit.py)：仅本轮分析代码。
- [protocol.json](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_operating_point_audit_20260915/protocol.json)：预定 cells/seed/epsilon/方向/判据、checkpoint 与相关 source SHA256、必要工程修正。
- [results.json](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_operating_point_audit_20260915/results.json)：32 个 precision×cell×epsilon 条目的最终诊断和 zero/history 检查。
- [diagnostic_arrays.npz](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_operating_point_audit_20260915/diagnostic_arrays.npz)：完整 impulse coefficients、A kernels、预定方向的复数响应，约 2.34 MB；没有正式 physiology comparison。
- [fixtures.json](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_operating_point_audit_20260915/fixtures.json) 与 [numerical_summary.json](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_operating_point_audit_20260915/numerical_summary.json)：scalar/坐标 fixtures、稳定区、有限精度差异及结果 hash。

执行环境：PyTorch 2.10.0+cu126，RTX 4070 Laptop GPU。SciPy/MKL 与 torch OpenMP 的冲突仅通过该进程的 `MKL_THREADING_LAYER=SEQUENTIAL` 解决，没有升级依赖、修改环境文件或使用忽略重复 runtime 的开关。一次 JSON 写出因低于数值 floor 的零分母失败；修复 null 表达后仅重放同一冻结 diagnostic 网格，没有增加选择机会。正式 source/checkpoint SHA256 复核通过，inference state_dict 未更新。
