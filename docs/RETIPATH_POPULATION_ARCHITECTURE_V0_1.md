# Population RetiPath architecture v0.1

日期：2026-09-18  
状态：**PROPOSED / 设计稿完成，待审阅；未实现、未训练、未验收。**  
本轮交付仅为本文件。Stage A、A.5、B、C 是后续路线，不构成本轮执行授权。

## 1. 依据、版本关系与证据边界

本稿收紧 [Population 初稿](C:/Users/win11-pc/Downloads/RETIPATH_POPULATION_INITIAL_DESIGN.md)，不覆盖初稿，不修改 G1–G4a、Canonical V1 的代码、实验合同或结果。初稿 SHA256 为 `B5D46A1B3A0662625590763D507FE03E2B59C09A56F11334895E640775E41227`。用户已确认文件名带 `(1)` 与上述文件指同一份初稿；《RetiPath可信度分级.txt》不要求作为本轮本地输入。本稿不声称读取了该缺失文件或复现其中的分级体系。

本稿与初稿冲突时，Population 后续设计以本稿为准；冻结实验仍以各自原合同为准。适用执行边界见 [AGENTS.md](../AGENTS.md)。本稿引用既有报告结论，并对必要 G1 源码做文本核对；未重算实验、重新评价 checkpoint 或进行文献审计。

现有证据需要保留的区别：

- [G4a](RETIPATH_G4A_MULTILEVEL_CONFIRMATORY.md) 中，multi-level joint 的 H1 h、BC s_B、BC o_B 误差，在五个新 teacher 的全部三个 paired seeds 上均低于 RGC-only 和 extra-RGC。
- 同一报告的 direct-BC logit intervention RMSE，C−A 的 teacher 均值为三负两正，C−E 为两负三正。因此，内部状态/输出恢复改善不能替代通路耦合或端到端干预恢复。
- 上述结果属于既有 realizable synthetic 系统；没有验证本稿的 Population 方程、个体层级先验、BC softplus 或真实生理解释。G3 的描述性结果与 G4a 分开引用，不合并 teacher 统计。

以下用“既有代码事实”“既有 synthetic 结果”“设计假设”区分依据。Population 各项运行时正确性和生理对应目前均为 **UNVERIFIED**；不能由设计完整或未来正确性验收通过推导生物可辨识性。

## 2. 推荐设计与本次收紧

推荐保留初稿的物理输入、局部群体回路、H1 feedback、direct BC 与 AC 两条分支、E/I conductance 后端和 joint training。首个实例仍为 ON/OFF parasol-related 有效回路；midget 接口保留为后续范围，不在本轮增加实例或机制。

| 方面 | v0.1 推荐约束 |
|---|---|
| 可辨识对象 | 分别登记 state、output、pathway coupling 参数；分别评价，不能互相代替 |
| H1 | tau/delay 有受收缩的个体偏差；feedback amplitude 采用更强 partial pooling |
| BC | 动力学与启用的输出参数保留较强个体差异，按 family 做 hierarchical partial pooling |
| AC | local/broad × ON/OFF 保留；tau/delay 先 family-shared，输出工作点也先 family-shared |
| BC 输出 | 提供 `LegacyPReLU` / `BaselineSoftplus` 两个互斥接口；迁移参考默认 LegacyPReLU，softplus 为未验证候选 |
| 空间与有效性 | Q 不变；新增按通路祖先支持定义的刺激覆盖、时间与标定有效性合同 |
| 优化 | joint 默认；增加 per-dataset gradient norm/cosine 诊断，不改变 optimizer 或自动调权 |
| 路线 | Stage A correctness → A.5 migration bridge → B heterogeneous population → C mild model mismatch |
| 真实数据 | 提高 RGC E/I current observation 的优先级，用于约束净兴奋/抑制驱动，仍不保证唯一分解各 coupling |

v0.1 不允许从一开始让每个 unit 的所有动力学、输出、连接、观测增益完全自由。拥有参数字段不等于拥有独立可训练自由度。下文列出的共享与固定项是推荐配置的一部分。

## 3. 计算图与实例身份

```text
physical stimulus X + pixel bounds + time/background/calibration
                           |
                    fixed area map Q
                           |
                   q -> identity outer
                     /             |
             G -> h_H              |
                 |                 |
             a_H + F -> f_H -------(-)
                                   |
                        q_modulated -> u_B (ON/OFF)
                                   |
                            v_B, w_B -> s_B
                                   |
                    selected BC output interface
                         r_B, r_B0, Δr_B
                           /             \
              π_BR + γ_BR                 π_BA + γ_BA
                     |                         |
                    d_E                  u_A -> a_A
                     |                         |
                     |                 o_A, o_A0, Δo_A
                     |                         |
                     |                  π_AR + γ_AR
                     |                         |
                     |                        d_I
                     |                         |
                    gE                        gI
                      \                       /
                       effective membrane V
                                |
                     adaptation + past events
                                |
                          logit ell -> p
```

观测端口从其对应量旁路读取，不替换学生自己的上游计算。H1 的 `h_H → a_H,F → f_H` 是有效返回路径；此版本不是完整 cone↔HC 生理突触闭环，h_H 不接收自身 f_H 返回后的输入。该约定决定 H1 state loss 对 a_H 没有直接数据梯度。

`retina_id / preparation_id / cell_id / session_id` 由数据与实例注册层管理。同一已识别生物细胞的重复记录共享个体参数；不同 retina 的细胞不伪装成同一个体或已知突触链。backbone 接收已解析的物理输入、固定图及实例参数，不读取 dataset/session ID 来选择非线性、动力学方程或通路机制。跨样本只共享有声明的 family 参数/先验；不能复制任意 dataset-ID 修正网络。

首个工程小图沿用初稿建议：N_Q=25、N_H=25、ON/OFF BC 各 25、local AC ON/OFF 各 9、broad AC ON/OFF 各 9、N_R=2、K=2 effective modes。2×2 degree 数值域、32×32 基准输入与 64×64 网格核对均为工程默认，不是解剖密度或充分覆盖真实 broad-field 细胞的证据。具体坐标、完整邻接支持与边界有效性必须在后续实施合同中固定，不能由这些数量推断已知生物连接。

## 4. 三类参数与四类变量

### 4.1 参数分类

| 参数类 | 本稿中的例子 | 可辨识目标 | 不应由什么替代 |
|---|---|---|---|
| **state parameters** | H1 tau/delay；BC tau_f/tau_s/delay/kappa；AC family tau/delay | 给定有效输入下的内部时间过程及 state 定义 | RGC prediction 良好或单次 state 轨迹吻合，均不等于所有 state 参数唯一 |
| **output parameters** | BC LegacyPReLU 的 alpha 或 BaselineSoftplus 的 b_B；AC b_A | state 到本地可传输 output 的变换 | 只测 h_H/s_B/a_A，不能直接识别独立的下游 output 变换 |
| **pathway coupling parameters** | a_H、gamma_BR、gamma_BA、gamma_AR | 同一 output 进入哪个接收通路、强度如何 | state/output recovery、跨 seed 一致或固定归一化不能替代 coupling / intervention recovery |

BC kappa 属于 state 组合定义，不重复登记为输出 gain。H1 amplitude a_H 虽产生 feedback output，在本稿统一归为 coupling。gE/gI 是时变 conductance traces，不是 gamma_BR/gamma_AR 参数。RGC bias 属于末端 readout/output 参数；本轮不新增自由的膜参数、conductance transfer gain 或 history gain。固定几何、单位规范和测量 nuisance 单独登记，不能藏在三类参数中重复计数。

参数清单应为每一项记录 `class / owner / family / shared-or-individual / transform / fixed-or-trainable / prior / permitted-loss-ancestors`。同一共享参数仅计数一次。这里定义的是后续接口要求，尚无新的参数注册代码。

### 4.2 input / state / output / observation

令 batch 轴为 B、时间轴为 T，单通道物理像素数为 P_d。节点数 N_Q、N_H、N_B、N_A、N_R 与像素数无关。

| 模块 | input | state | pathway output | observation 及主要形状 |
|---|---|---|---|---|
| Q / outer | X[B,T,P_d]、像素边界、时间 | 无新增动态 state；outer 为 identity | q[B,T,N_Q] | 输入标定不是神经活动观测 |
| H1 | u_H=Gq，[B,T,N_H] | h_H[B,T,N_H] | f_H=F(a_H⊙h_H)，[B,T,N_Q] | HC voltage 仅可经已定义测量映射对应 h_H；f_H 单列 |
| BC | u_B[B,T,N_B] | v_B、w_B、s_B，均 [B,T,N_B] | r_B、r_B0、Δr_B；r_B0[N_B] 广播 | voltage 候选对应 s_B；output 观测须遵守所选接口的量纲与基线 |
| AC | u_A[B,T,N_A] | a_A[B,T,N_A] | o_A、o_A0、Δo_A；o_A0[N_A] 广播 | AC voltage 候选对应 a_A，不自动等于 inhibitory drive |
| E/I | direct BC / AC outputs | 无额外独立 state | d_E、d_I、gE、gI，[B,T,N_R,K] | voltage-clamp current 来自 conductance + holding/reversal 条件 |
| RGC | gE/gI、严格过去的 events | V[B,T,N_R,K]、adaptation/history[B,T,N_R] | ell、p[B,T,N_R] | conditional Bernoulli occupancy；不是平均 PSTH |

所有 state/reset/carry 和初值须显式。首个实施只需独立序列 reset；流式 carry 要单独验收，禁止在 batch 间隐式保留状态。初稿的 150 Hz conditional occupancy 约定保留；同一时刻的 event 不得进入该时刻预测的 history。

## 5. Pathway Coupling Contract

### 5.1 统一强度与几何分解

对于 BR、BA、AR 每个接收端及已声明 route，采用初稿的固定空间权重：

\[
\pi^{P}_{ij}=\frac{M^{P}_{ij}K^{P}_{ij}}
 {\sum_{k\in\mathcal N^{P,full}_i}M^{P}_{ik}K^{P}_{ik}},
\qquad W^{P}_{ij}=\gamma^{P}_i\pi^{P}_{ij},\qquad \gamma^{P}_i\ge0.
\]

M、空间核 K、节点位置、route polarity 与完整邻接集合先固定；gamma 是该 route 的唯一总强度。分母来自完整邻接集合，不能随 batch、FOV 缺失或子图裁剪改变。空支持必须在配置时显式声明该 route 不存在，不能做零分母归一化。H1 的 G/F 沿用既定几何映射，不能为统一记号擅自重新归一化 G1 返回映射。

### 5.2 每条边的作用点

\[
u_{H,i}=\sum_jG_{ij}q_j,\quad
\tau_{H,i}\dot h_{H,i}=-h_{H,i}+u_{H,i}(t-\delta_{H,i}),
\]
\[
\boxed{f_{H,j}=\sum_iF_{ji}a_{H,i}h_{H,i}},\qquad
q^{mod}_j=q_j-f_{H,j}.
\]

H1 合同是 **h_H --a_H--> H1 feedback**，不是把 h_H 直接叫作 feedback。G 为 [N_H,N_Q]，F 为 [N_Q,N_H]，a_H 为 [N_H]。没有额外可自由补偿的 H1 input gain 或 state gain。

对 RGC r、mode k 和 BC i：

\[
c^E_{rki}=\gamma^{BR}_{rk}\pi^{BR}_{rki}\Delta r_{B,i},
\qquad \boxed{d_{E,rk}=\sum_i c^E_{rki}}.
\]

这是 **Δr_B --gamma_BR--> direct excitatory drive**。pi_BR 为 [N_R,K,N_B]，gamma_BR 为 [N_R,K]。c_E 保留局部贡献轴 [B,T,N_R,K,N_B]；d_E 是求和后的 [B,T,N_R,K]。旧 G1 记录的 local `d_E` 具有 BC 节点轴，迁移报告必须说明轴映射，不能仅凭同名字段比较不同量。

对 AC j：

\[
\boxed{u_{A,j}=\gamma^{BA}_j\sum_i\pi^{BA}_{ji}\Delta r_{B,i}}.
\]

这是 **Δr_B --gamma_BA--> AC input**。pi_BA 为 [N_A,N_B]；gamma_BA 为 [N_A] 的注册视图。推荐首版 gamma_BA 固定为 1，作为 AC input 尺度锚，仍在接口中显式登记为 fixed coupling。该约束不表示生理强度已知。后续若要估计它，必须事先说明新增 AC/电流观测与单位锚；不默认同时放开 gamma_BA、AC output gain 和 gamma_AR。

令 f 为 local/broad × ON/OFF 的四个 AC family，每个 family 的 same/crossover 身份由目标 RGC 极性和固定 mask 决定：

\[
c^I_{rkj}=\gamma^{AR}_{rk,f(j)}\pi^{AR}_{rkj}\Delta o_{A,j},
\qquad \boxed{d_{I,rk}=\sum_j c^I_{rkj}}.
\]

pi_AR 在每个已启用 family-route 内按完整支持归一化；gamma_AR 为 [N_R,K,4]，不存在的 route 固定为零。这是 **Δo_A --gamma_AR--> inhibitory drive**。保留 c_I 的 [B,T,N_R,K,N_A] 节点贡献；禁止把 local/broad 与 same/crossover 当作互斥的同一个分类轴。

最后：

\[
\boxed{gE=\operatorname{softplus}(b_0+d_E),\quad
gI=\operatorname{softplus}(b_0+d_I)},\qquad b_0=\log(e^1-1).
\]

因此 **d_E/d_I --> gE/gI --> RGC**；零驱动时 gE=gI=1，沿用 G1 effective normalized conductance baseline。delta output 和 drive 可正可负，conductance 始终为正；输出低于基线可以减少兴奋或减少抑制。禁止对 Δr_B、Δo_A 或 d_I 取绝对值，也不把负 delta 当作负总 conductance。

G1 的参数名 `g_e/g_i` 对应驱动强度角色，不等于 traces `gE/gI`。Population 引入上述 gamma 记号时，不再在同一投射的 output、edge 和接收端各放一份自由 gain。固定/归一化能减少尺度混淆，但不能证明所有耦合唯一可辨识。

### 5.3 BLOCK_DIRECT_BC_DRIVE 的不可变语义

在正常前向与干预前向中使用相同刺激、初值、参数、观测/历史条件：

1. 保留 H1、BC state、所选 BC output，以及送入 AC 的同一 Δr_B。
2. 将全部指定 RGC/mode 的 direct c_E、d_E 置零；不令 Δr_B=0，不切断 BA，不删除 BC 单元。
3. gE 回到 tonic baseline 1，而非 0；AC、d_I、gI 与给定的过去事件 history 保持相同。
4. 从同一初值重新计算 RGC V、adaptation、ell、p，不能冻结正常运行的 V/adaptation 作为干预结果。

干预效应统一登记 `delta_ell = ell_block - ell_normal`；teacher/student intervention RMSE 比较两者的 delta_ell。它是 conditional computational transmission block，不等价药理阻断，不自动代表自主 spike history 下的干预。

## 6. 个体自由度与 partial pooling

采用初稿的 family 层级表达，但不再把“每 unit 有字段”解释为全部独立学习：

\[
z_i=\mu_{f(i)}+\eta_i,\qquad \theta_i=T(z_i),\qquad
\sum_{i\in f}\eta_i=0.
\]

T 维持相应正值/有界/有序约束；BC 使用有序 tau_f<tau_s 的参数化。family center 和个体偏差在同一清单登记，不额外复制一组完全自由的个体参数。层级正则约束 eta，首版不学习自由 covariance、不依据 test 放大 prior 宽度，也不默认相邻单元 tau/delay 平滑。

| 对象 | 首版自由度 | pooling / 固定规则 |
|---|---|---|
| H1 tau_H、delay_H | family center + 受约束个体偏差 | 部分个体化，偏差均值为零；非负 delay、正 tau |
| H1 a_H | family center + 强收缩个体偏差 | **strong partial pooling**；不能与 H1 state/observation gain 任意补偿 |
| BC tau_f、tau_s、delay_B、kappa | ON/OFF family center + 个体偏差 | 较 H1 更宽的工程先验；仍须保持次序、范围与层级收缩 |
| BC output alpha 或 b_B | 所选接口对应的 family center + 个体偏差 | 互斥启用；不新增单位级 release gain |
| AC tau_A、delay_A | 四个 family 各一组 | **family-shared dynamics**；unit 只索引对应共享参数，独立 deviations 固定为零 |
| AC b_A | 四个 family 各一组 | 首版也 family-shared；不同时开放个体输入 gain 和输出 gain |
| gamma_BR、gamma_AR | 按 RGC type / mode / route 的中心及受收缩接收端偏差 | 不放开每条 edge；未启用 route 固定为零 |
| gamma_BA | 显式 fixed coupling | 首版为 1；不计作可训练个体参数 |
| RGC readout bias | 每个 RGC 的受收缩偏差 | beta、膜/漏电/反转电位规范、adaptation/history 系数及时间常数先继承固定后端 |
| 几何、支持、Q、测量单位 | 固定 | 不联合学习坐标、半径、拓扑或任意 observation gain |

“强”与“较宽”需要可执行的先验约定：在已声明的、无量纲标准化 raw 坐标上，推荐 H1 amplitude 的个体 prior SD 不大于 H1 dynamics 的三分之一；BC 的个体 prior SD 可为 H1 dynamics 的两倍。该比例只是事前工程默认，不能直接比较不同物理量的数值 SD。每种参数的标准化尺度、中心、边界、具体 SD 及 hierarchy 权重须随未来 protocol 一次冻结，不按拟合排序调节；本稿不改动历史 loss 或实验预算。

AC dynamics 的个体 prior SD 则为零，是本版有意采用的硬共享。少量 RGC/同类单元不足以估计可靠的 population variance，首版 prior width 固定，不把两个 RGC 的拟合散布当作生物群体分布。未记录 unit 的偏差可能主要由先验决定，不能宣称得到直接观测恢复。

## 7. BC dynamics 与可切换 output 接口

### 7.1 保留的 Population state 提案

\[
u_{B,i}=p_i\sum_jK^{QB}_{ij}q^{mod}_j,\qquad p_i\in\{+1,-1\},
\]
\[
\tau_{f,i}\dot v_i=-v_i+u_{B,i}(t-\delta_{B,i}),\quad
\tau_{s,i}\dot w_i=-w_i+v_i,\quad s_{B,i}=v_i-\kappa_iw_i.
\]

保留初稿的串联 fast/slow 与 `0<=kappa<=1` 设计；不再叠加独立可调 DoG surround。p_i 是 effective ON/OFF contrast 约定，不是完整受体机制。

**迁移边界：**G1 源码中的 fast/slow 是并行接收 u_B，再输出 `[q_f, q_f-q_s]` 两个 sustained/transient 分量；不是上述串联方程，也不是两个 ON/OFF 生物 family。本稿不把它们直接改名后声称数值等价。该差异必须在 A.5 明列，不能通过自动更换方程“修复”桥接。

### 7.2 两种互斥输出

统一概念接口返回 `state=s_B / output_native / baseline / delta / quantity_kind`，并登记启用的 output 参数；两条下游分支只接收同一份 delta。

| 接口 | output_native 与 baseline | delta | 含义和地位 |
|---|---|---|---|
| `LegacyPReLU` | r_B = max(s_B,0)+alpha_B min(s_B,0)；r_B0=0 | Δr_B=r_B | signed effective output；与旧 PReLU 局部算子对照的默认参考，不能叫非负 release |
| `BaselineSoftplus` | r_B=softplus(b_B+s_B)；r_B0=softplus(b_B) | Δr_B=r_B-r_B0 | 总输出非负、变化量有符号；只是 release-like proxy 候选，**未被验证** |

LegacyPReLU 的 r_B 仅为统一接口名，旧名 o_B 可保留为来源映射；不代表新获得了总释放量意义。不能加绝对值、临时 offset 或 clamp 来伪造非负总输出。BaselineSoftplus 固定 slope/temperature 和输入输出尺度，仅 b_B 提供工作点；不额外引入相同作用的 release amplitude。

接口在整次实例/协议开始前固定，不按 dataset ID、held-out 结果或每个 unit 的拟合优劣选择；不增加 learned mixture、自动切换或隐含第三种激活。未启用接口的参数不进入 optimizer。观测映射随 quantity_kind 明确：LegacyPReLU 不能直接当作校准后的 glutamate concentration；softplus 也不因非负就自动等于传感器观测。

推荐 Stage A/A.5 首先以 LegacyPReLU 建立局部算子对照，同时验收两接口的代数合同；这不自动授权训练两组实验。未来选择 softplus 开展研究比较需单独冻结协议。仅选 LegacyPReLU 也不会消除 Population dynamics、ON/OFF 拓扑与 G1 的其他差异。

## 8. AC 与既有 RGC 后端

AC 保留初稿的一阶 dynamics 与 output 提案，但改为 family 共享：

\[
\tau_{A,f(j)}\dot a_{A,j}=-a_{A,j}+u_{A,j}(t-\delta_{A,f(j)}),
\]
\[
o_{A,j}=\operatorname{softplus}(b_{A,f(j)}+a_{A,j}),\quad
o^0_{A,j}=\operatorname{softplus}(b_{A,f(j)}),\quad
\Delta o_{A,j}=o_{A,j}-o^0_{A,j}.
\]

local/broad 是空间支持类别，ON/OFF 是驱动极性；same-polarity/crossover 是该输出相对目标 RGC 的 route。它们不是命名 AC subtype，不能把 arbitrary named-cell recording 直接塞入某个 family。AC softplus 是初稿已有、尚未验证的模型提案；G1 的 AC 输出为线性 lowpass state，故这也是 A.5 需要记录的变化，不沿用 G1 的全图等价结论。

RGC 继承 effective conductance 主体：

\[
C\dot V=g_L(E_L-V)+gE(E_E-V)+gI(E_I-V).
\]

K=2 仅为 effective integration modes，不叫真实树突 compartments。实现时沿用既有固定后端的有效单位、解析更新、初值、adaptation 和严格过去的 event history。膜读数先减去其基线再进入已有 readout；例如 G1 使用 `mean(V)-2/9`，不能因改写通式而无意丢掉该基线。beta 与 V 的单位规范绑定，不同时学习任意 C/conductance 尺度并声称获得 nS/mV。

初稿中的 outer identity 保留，H1 无显式早期非线性的 F2 表达限制公开保留。本版不启用 BC gap coupling、AC→BC、AC→AC、命名 AII/SAC/wiry 回路、显式 AMPA/NMDA、HH 或真实 dendritic tree；也不为桥接添加这些机制。

## 9. Q 不变与 pathway-specific stimulus coverage / validity

### 9.1 Q 的不可变定义

\[
Q^{(d)}_{jp}=\frac{|S_j\cap\Omega^{(d)}_p|}{|S_j|},\qquad
q_j=\sum_p Q^{(d)}_{jp}X_p.
\]

S_j 为固定物理 aperture，Omega_p 是已知像素边界，分母始终是完整 aperture 面积。degree、pixel area、dt、bin 内保持/平均方式、背景/contrast 定义分别登记。物理刺激缩放与 numerical grid 重采样分开，Q 不学习，不以 resize、裁切、degree/pixel 重标或截断后归一化补足缺失输入。

同一常量场或像素边界对齐的分块场可要求面积积分一致；一般光滑连续场在足够分辨率下检查数值收敛，不能要求任意 32/64 网格表示在有限精度下逐位相同。真实物理尺度改变允许响应改变。若 QX1=QX2 且历史/初值相同，后续确定性图不能恢复两刺激在 Q 中丢失的差异。

### 9.2 按完整祖先通路计算支持

新增的是**有效性判定合同，不是新输入模型**。对每个待观测量 y，事先登记其完整依赖支持 S_y：

- h_H,i：G 中所有固定非零输入 aperture 的并集。
- f_H,j：F 中所有固定返回边对应 H1 的输入支持；不能只检查反馈落点 j。
- s_B/r_B：该 BC 局部 K_QB 支持，加上影响这些位置的全部 H1 feedback 祖先。
- direct d_E：BR 的所有固定 BC 支持并集。
- a_A/o_A：BA 的所有固定 BC 支持并集；broad AC 的远端输入不能忽略。
- inhibitory d_I：AR 的所有 AC 支持及其 BC/H1 祖先。
- RGC：相关 E/I 两路完整祖先的并集，以及所需事件历史、动态初值与前史。

支持由固定结构与启用 route 决定，不因拟合后 gamma 很小或暂时为零而缩小。覆盖必须贯穿完整时间窗口及声明的 warmup/reset 条件；IIR 状态不能用未说明的有限 history 当作精确真实前史。

| 有效性维度 | 必须记录 | 失败时的处理 |
|---|---|---|
| spatial coverage | 每条祖先 route 的完整物理支持是否在已知刺激域内 | 对应 observation/loss 无效；不删路径、不把未知外围填零 |
| grid adequacy | 像素界限、面积、足够空间分辨率、Q 合同 | 不把 coarse pooling 丢失的信息当作可恢复细节 |
| temporal coverage | dt、同步、因果延迟、reset 或可信前史 | 无有效初值/前史的目标不作完整通路恢复证据 |
| calibration | 背景/contrast 单位、坐标配准、观测基线与测量滤波 | 映射不明记 UNVERIFIED；不以自由 adapter 吸收 |
| identity / observation | unit/family/recording mode 是否可对应到指定端口 | 不把同名但不同含义的记录合并 |
| excitation coverage | 已有刺激是否覆盖目标通路相关空间、极性、时间和历史变化 | 可判“输入有效但约束不足”；不宣称已辨识 coupling |

例：中心视野可足以监督某个局部 h_H，却不足以预测一个接受 broad AC 输入的 RGC。只排除依赖缺失输入的观测；不能为了保留该 RGC loss 删除 broad 路径或重新归一化剩余连接。完全已知的域外背景仅可按明确记录的实验合同计入，不能猜测。目标相关子图必须含完整祖先，第一版优先完整小图。

normal/block 配对比较采用两种前向所需依赖的并集与共同有效 mask，不能干预后放宽覆盖来改善分数。训练/评价 mask、覆盖分类和排除规则在访问相应结果前固定。共同尺度/共同 probe 可帮助跨观测对齐，但不自动提供 gamma 的独立约束；coverage validity 与 parameter identifiability 分别报告。

## 10. 观测合同与 loss gradient routing

各数据集用自己的物理刺激从同一类回路前向，measurement head 优先 identity、受限 affine 或有依据的测量滤波。观测的 offset/gain 若未知，必须单列 nuisance 和标定依据；不通过任意 affine 让独立 output coupling 消失。有效输入 u_B 仍没有已验证的 EPSC generative mapping，不能直接冒充 BC input current。

下表描述**数据项允许到达的祖先参数**，不是保证非零梯度、实际更新或唯一恢复；hierarchy regularizer 的更新另记。

| observation / data loss | 允许的数据梯度路径 | 没有该直接监督约束的独立下游量 |
|---|---|---|
| H1 h_H | H1 state tau/delay | a_H、BC output、所有 BR/BA/AR coupling |
| H1 f_H（有匹配观测时） | H1 state + a_H | BC/AC/RGC 下游参数 |
| BC s_B | H1 state + a_H + BC state | BC alpha/b_B；BR/BA/AR |
| BC r_B / delta / 对应传感器输出 | 上行全部 + 已启用 BC output 参数 | BR/BA/AR，不因观测了 r_B 就直接观测传输强度 |
| AC a_A | H1、BC state/output、gamma_BA、AC state | AC b_A、gamma_AR；gamma_BR |
| AC o_A / delta | 上行 AC 路径 + AC output b_A | gamma_AR；gamma_BR |
| 已分离 E current，已知 clamp 电位 | H1、BC state/output、gamma_BR、其测量映射 | BA/AC/AR；不经过自由膜 V |
| 已分离 I current，已知 clamp 电位 | H1、BC state/output、BA、AC state/output、AR、其测量映射 | BR；不经过自由膜 V |
| RGC spikes | 两路所有可训练祖先及 RGC readout | 不保证这些参数能由 spike 唯一分解 |

fixed gamma_BA 具有概念依赖但无 optimizer 更新。若电流是自由膜状态下计算的总电流而非分离 voltage-clamp current，上述 E/I 独立梯度结论不适用，必须重新按实际 observation graph 登记。基于先验得到的 coupling 收缩不能记作观测已经识别 coupling。

synthetic 评价继续分开 state trajectory、output trajectory、coupling 参数/功能输出、normal RGC prediction、conditional intervention 和 cross-seed ambiguity。不把低 h_H/s_B RMSE 或低跨 seed 分散度称为通路因果恢复；真实数据无 teacher 时也不能报告未测 latent 的“真值恢复”。

## 11. joint 默认与梯度诊断

保留初稿的联合目标：

\[
J(\theta)=\sum_d w_d L_d(\theta)+R_{hierarchy}(\theta).
\]

数据集采样概率、噪声模型、有效 scalar/sequence 归约、loss 权重、曝光量和 optimizer budget 一并定义。不同数据集的观测频率不等于独立信息量。保持 joint，不自动加入 progressive、中心到外围训练或新的 condition；本稿不调整 optimizer、历史 loss 或冻结训练步数。

新增诊断只读取训练时的梯度。在同一参数快照、optimizer step 与 clipping 之前，对参数类/模块组 G 记录：

\[
g_d^G=\nabla_{\theta_G}L_d,\quad
\|g_d^G\|_2,\quad\|w_d g_d^G\|_2,
\]
\[
\cos(d,e;G)=\frac{\langle w_dg_d^G,w_eg_e^G\rangle}
 {\|w_dg_d^G\|_2\,\|w_eg_e^G\|_2}.
\]

- 同时记录未加权与实际加权 norm，避免混淆测量噪声、归约方式与优化权重；hierarchy gradient 单独记录，不并入某 dataset 的证据。
- cosine 只在双方共享且具有数据依赖的同一参数坐标上解释，并记录坐标数/共享 group。不同 retina 的独立个体参数不因同名字段而强行配对；完全不相交的支持标记 `N/A`。
- 零 norm 的 cosine 标记 `UNDEFINED_ZERO_NORM`，不填 0；非有限梯度报告数值问题，不据此自动改 loss、重启或延长训练。
- 使用原训练 batch 或事先规定的训练诊断 batch、相同参数状态计算；不读取 held-out targets。日志频率与额外诊断开销在未来协议中事先声明，不新增 optimizer updates 或观测曝光。
- 这是诊断，不启用 GradNorm、PCGrad、自适应 loss reweighting、逐数据集 optimizer 或结果驱动 curriculum。负 cosine 仅说明该状态/批次的梯度相反，不直接证明生理矛盾或数据无效。

## 12. Stage A：correctness

目的仅为未来 isolated Population 实施的接口与数值正确性；不把代码通过当成研究有效性。最小范围：

1. 参数 registry 区分三类参数，检查 H1/BC partial pooling、AC family-shared 别名与固定 gamma_BA；未启用 BC output 参数不得进入 optimizer。
2. input/state/output/observation 形状与基线明确，同一 Δr_B 分流；改 output/coupling 参数不会被错误写回对应上游 state。
3. 两个 BC output 接口的零输入基线、符号与梯度合同；不要求 PReLU 与 softplus 响应相等。
4. ON/OFF、local/broad、same/crossover 的 route mask 各有明确作用；signed delta 可造成 disinhibition，gE/gI 始终为正。
5. 按 pathway validity 排除缺失输入目标，不能裁剪后归一化；复用适用的 Q、causality、state/output、gradient routing 检查。
6. BLOCK_DIRECT_BC_DRIVE 与第 5.3 节相符；backbone 不按 dataset/session ID 选择机制。

后续验收可使用固定的手工数值 fixture、前向与导数核对，不需要训练来证明这些代数合同。本轮没有实现这些检查、运行 pytest 或生成 fixture 数据。

## 13. Stage A.5：migration bridge

目的：在异质 Population 训练前，分清哪些既有算子/语义被保留，哪些是初稿已有但尚未验证的改变。G1 和旧 Canonical 参考保持只读；新增 isolated 实现不得修改参考使比较通过。完整 Population 与 G1 未必存在一一对应的参数极限，不预设全图逐位等价。

每项留下“参考来源、张量/参数映射、固定输入与历史、共同观测量、绝对/相对误差、可比范围、差异原因”的记录。数值容差按 dtype/积分误差事前固定；F2/干预不另设为了得到成功结论的研究阈值。

| 必查对象 | 桥接内容 | 验收解释与边界 |
|---|---|---|
| **Q** | 同一 physical field、完整 aperture denominator、不同充分分辨 grid；常量/对齐分块积分、光滑场收敛与 Q 后丢失信息 | 不改 Q 定义；有限网格误差不能靠 resize/renormalize 掩盖 |
| **H1 feedback** | 分别映射 h_H 与 F(a_H⊙h_H)；同幅度/延迟/返回几何的共同算子；G1 一 bin delay 的时间对应 | 固定 h_H 改 a_H 时 feedback 改、h_H 不改；不能只核对 h_H 就宣布 feedback 迁移通过 |
| **BC nonlinear integration / F2** | 比较 state→逐节点非线性→空间汇聚的次序；在预先固定周期输入、窗口和基线下保留 F1/F2 振幅及相位读数 | G1 parallel sustained/transient 与 Population cascade/ON/OFF 非同一模型；Legacy 算子可对照，softplus 只核对自身合同，不继承 F2 已验证结论 |
| **direct / AC split** | 同一 Δr_B 分到 BR 与 BA；逐节点贡献、route mask 和求和轴明确 | 不能将 G1 两个时间分量重命名为 ON/OFF 或 local/broad；不能为了匹配加入独立的第二份 BC output |
| **E/I conductance** | d_E/d_I 与 gamma 名称分离；零驱动 tonic baseline、signed delta、disinhibition、既有后端单位和初值 | 不以 abs 或新 gain 补偿输出改变；Population AC softplus 与 G1 线性 AC 的差异单列 |
| **causality** | 改变未来 stimulus/event 不改变已定时间前缀；同一时刻 event 不进入该时刻 logit；reset/carry、delay 与测量滤波时间约定 | 所有新增接口均保持因果；不得由不透明 batch state 或双向滤波泄漏未来 |
| **BLOCK_DIRECT_BC_DRIVE** | 同初值/历史下 direct drive=0，gE=1，BC/AC/gI 不变，V/adaptation/ell 重新计算 | 对照 delta_ell 的符号、目标轴和作用点；不等价 gE=0、BC silencing 或药理阻断 |

F2 特别限制：旧 [Canonical F2 pilot](RETIPATH_F2_LOCALIZATION_PILOT.md) 中“首个 F2 出现在 BC PReLU”不等于验证了细粒度 BC nonlinear subunits；旧 formal 路径的 pooling/rectification 顺序与 G1 local prototype 也不能混为一谈。A.5 只对声明相同的局部算子作等价要求；不能通过调 alpha、刺激尺度或 loss 放大 F2，再声称迁移成功。

至少三项已知差异须在 bridge ledger 保留：① BC parallel 双时间分量变为每 ON/OFF unit 的 cascade+kappa；② G1 AC 线性 state 输出变为初稿 AC softplus 输出；③ 新群体几何、四个 AC family 与多 RGC route。BC 选择 BaselineSoftplus 时再增加一个明确的输出函数差异。这些是 Population 设计假设，不是本轮新增已验证事实。

桥接结果分别标记“共同算子已核对 / 明确的候选模型变化 / UNVERIFIED”。发现不明差异只暂停依赖它的迁移结论与 Stage B，不自动改架构或补机制；有意模型变化需要单独的解释和后续授权，不能由 G1 正确性或 G4a 结果代替。当前这些运行时比较均未执行。

## 14. Stage B 与 Stage C 的范围

**Stage B：heterogeneous population。**仅在后续明确授权且 A/A.5 所需边界已清楚后进行。teacher 个体差异来自事前固定的 family hierarchy；AC dynamics 仍 family-shared，不能为了“异质”给所有参数加独立扰动。明确 observed/hidden units、未见层×尺度组合、样本身份及支持有效性，不跨动物伪造实测连接。

若比较 multi-level 数据价值，沿用公平性原则：RGC-only 与 multi-level joint 使用相同 base RGC 与相同 exposure；extra-RGC 为独立额外观测，不称 information-matched control。具体 conditions、数量、noise、loss、paired seeds、预算和选择规则由未来协议冻结；本稿不自动复用 180 steps 或启动三条件实验。分开报告 state、output、coupling 功能读数、direct-BC intervention、RGC excess CE 与 ambiguity，不新增综合成功阈值。初版 prior 有效不等于估计了真实 population 分布。

**Stage C：mild model mismatch。**在独立后续授权中仅选一个清楚声明的轻度失配，固定其他合同，检查 prior/共享假设在不完全 realizable 条件下的局限。本稿不指定或启用新的 teacher 生理机制，不自动启动 BC gap、AC→BC、受体或外层非线性实验。不得按 Stage B 排序挑选有利失配、加 condition 或追加训练。

RF 仍为辅助敏感度读数；本稿不安排新 RF 计算，也不把动态 RF 解释为丢失输入恢复。四阶段路线的推进由单独任务决定，设计完成后不自动执行。

## 15. 真实数据阶段：提高 RGC E/I current 优先级

推荐在确认 stimulus/time/background/geometry 与记录身份之后，将**可校准的 RGC E/I current observation**提升为连接 BC/AC outputs 与 RGC spike 的优先中间约束，尽量选择与 spike/上游观测具有可比刺激和 history 的资料。它能更直接约束净 E/I drive；仅靠 H1/BC state 很难单独决定 a_H 以外的全部下游 coupling。

观测合同必须写清 holding potential、reversal potentials、记录电流符号、leak/offset 处理、采样/仪器滤波、E/I 分离依据及对应单位。对 outward-positive voltage clamp：

\[
I_E=gE(V_{hold}-E_E),\qquad I_I=gI(V_{hold}-E_I).
\]

这里 V_hold 是已知实验边界条件，不使用自由膜前向的 V 冒充 clamp voltage。若只能得到混合电流或 separation/calibration 不明，不能直接给 gE/gI 两个独立监督标签；相关 mapping 标记 UNVERIFIED。effective normalized conductance 转为物理 current 需要明确、受限且有依据的标定，不自动升级为 nS/pA。

E current 约束 direct branch 的净驱动；I current 约束 BA→AC→AR 的净结果。即使两者都拟合良好，也不保证分别识别 gamma_BA、AC output 与 gamma_AR，不证明特定 AC subtype 或逐突触连接。HC/BC/AC 观测仍用于其对应 state/output，必要时提供耦合分解的额外锚；不能用更自由 observation head 替代这些证据。

## 16. 本轮交付与未执行范围

本稿完成八项收紧：三类参数、Pathway Coupling Contract、受控个体自由度、双 BC output 接口、Q 不变且按通路判定有效性、joint 梯度诊断、含七项迁移要求的 A/A.5/B/C 路线、真实数据 E/I current 优先级。

仅新增本文件，旧初稿保留。没有修改正式或 isolated 源码、AGENTS、NEXT_TASK、历史结果、数据或 checkpoint；没有执行 Git、模型前向、pytest、训练、synthetic generation、RF 计算或 checkpoint 评价。源文件与既有报告只做必要文本核对。设计稿本身无必需的用户缺项；未来实施协议的参数边界、prior 数值、几何及验收容差需要在对应授权任务中具体冻结，不能由本稿中的路线图自动执行。

主要本地依据：

- [初版 Population 设计](C:/Users/win11-pc/Downloads/RETIPATH_POPULATION_INITIAL_DESIGN.md)：本稿保留的图、Q、state 方程、AC/RGC 主体及未启用机制边界。
- [G1 implementation report](RETIPATH_G1_PROTOTYPE_IMPLEMENTATION.md) 与 [G1 circuit source](../experiments/retipath_multiscale_v0/circuit.py)：既有 state/output、并行 BC、PReLU、线性 AC、conductance 和 block 实际语义。
- [G3 multi-level summary](RETIPATH_G3_MULTILEVEL_SUMMARY.md) 与 [G4a confirmatory report](RETIPATH_G4A_MULTILEVEL_CONFIRMATORY.md)：synthetic 内部恢复与干预恢复必须分别报告的既有结果。
- [Canonical F2 pilot](RETIPATH_F2_LOCALIZATION_PILOT.md)：旧 F2 定位结论的适用范围，不能替代 Population 局部非线性验收。

**停止边界：本轮设计文档完成即停止；不进入 Stage A，不作下一阶段研究决策。**
