# Population RetiPath：真实多层数据接入合同 v0

日期：2026-09-20。状态：**接口设计已完成；未实现 adapter/trainer，未读取新的刺激或 spike payload，未运行模型、测试、训练或 synthetic 实验。**

本合同以 [Population v0.1](RETIPATH_POPULATION_ARCHITECTURE_V0_1.md)、[Stage A](RETIPATH_POPULATION_STAGE_A.md)、[A.5](RETIPATH_POPULATION_STAGE_A5_MIGRATION.md) 和 [Stage B](RETIPATH_POPULATION_STAGE_B.md) 为模型依据；数据事实主要来自 2026-09-17 的 [primate dataset map](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md) 及其 [机器可读记录](../data/research/primate_multilevel_retina_dataset_map.json)，另核对既有 [physiology audit](../output/architecture_conformance_20260831/physiology.md)、[数据可得性边界](../audit/DATA_AVAILABILITY.md) 和 Schottdorf–Lee 时间合同。原审计的可得性是有日期的证据，不是本轮重新下载、重新检查所有 archive 的结果。

**当前没有已闭合接口、可以立即启动本版 Population 真实多层训练的数据组合。** Schottdorf–Lee 和 Karamanlis 有已核实的原始观测级资料，但各自仍有时间、物理几何或实例映射门槛。HC voltage 是 `h_H` 的有条件候选；已审计 Raval BC input current 不是 `s_B` 或 `delta_r_B`。优先补齐可标定的 parasol RGC E/I current 与对应刺激，而不是用更自由的 decoder 消除测量差异。

本轮按用户指令将主线转为 real-data integration。Stage A/A.5/B 的方法学验证对进入这一接口阶段已足够；这不是生理可辨识性通过的结论。B.1/B.2 保留为诊断，不继续扩展，也不把其优化设置自动迁入真实数据。历史数据审计中关于“是否升级主线”的建议由本轮任务取代，原始数据事实和限制保持。

Stage B 已有结果支持在该 synthetic family 内分别考察 state/output 和通路功能：C 的内部状态、BC output 与 direct-intervention recovery 改善，但 gamma_BR/gamma_AR 参数恢复基本未改善，AC 改善不一致，且 RGC prediction 存在代价。因此本合同分别登记 state、output、coupling 与测量尺度，不把增加层级观测当作所有参数都已获得真实监督的依据。

## 1. 接入状态和证据规则

状态针对**当前冻结的 Population v0.1 + 本条具体观测**，不是评价论文质量，也不表示全球不存在更合适的记录。

| 状态 | 本合同含义 |
|---|---|
| READY | 实际刺激与逐试次观测、单位、同步、完整物理支持、细胞/实例身份和端口测量方程全部闭合；无需猜测即可制定冻结训练协议。现有条目没有达到这一完整门槛 |
| NEEDS_CALIBRATION | 已有原始观测级输入—响应资料，端口类别可接，但时间/空间/单位/实例标定仍缺；并不只指光强标定 |
| NEEDS_RAW_DATA | 缺少已核实的逐试次 raw stimulus–response pair 或必要原始日志；现有论文/processed source data 不能补齐。拿到 raw 后仍须通过标定与端口门槛 |
| NOT_DIRECTLY_COMPATIBLE | 当前测量量、预处理、物种/细胞通路或光照机制与冻结模型接口不匹配；仅拿到更多同类记录也不能自动直接监督。不得用高容量 head 或改架构绕过 |

若有多个缺项，条目用最先阻断接入的状态，并另列其余缺项。`UNVERIFIED` 用于尚无证据的具体字段，不是第五种接入状态。`AUTHOR_REQUEST_REQUIRED` 是旧数据地图的来源状态，不等于作者已经同意提供数据；本轮未联系作者。

每个未来可执行记录必须落成一个 manifest：`source/version/hash/license`、`species/animal/retina/preparation/cell/session/trial`、细胞类型及证据、实际刺激与时钟、完整像素边界/面积、光谱和输入单位、基线/适应历史、观测端口及单位、固定测量算子、缺失掩码、参数所有权、数据划分和既有 test 消费记录。未知 ID 保留未知；不能给不同论文填同一个 retina ID。论文 cell 总数、archive 文件数、session 数和可用训练样本数分别登记。

## 2. 冻结的模型接口与测量语义

### 2.1 端口和允许的 observation adapter

本轮保留 LegacyPReLU、现有 Q/H1/BC/AC/E–I/RGC 方程、partial pooling、参数边界和 intervention 作用点。不增加 cone 模型、BC coupling、AC→BC、outer nonlinearity、E/I 自由 decoder 或新的膜参数。

| RetiPath port | 模型含义 | 可接受的真实观测及低容量映射 | 不能直接替代它的量 |
|---|---|---|---|
| `q` | 固定面积积分后的局部相对刺激 | 已标定刺激经原 Q；属于输入合同，无需 trainable observation head | cone photocurrent、cone voltage、ERG；q 不是光感受器生理状态 |
| `h_H` | H1-like effective state | 身份、位置、刺激和状态对应成立时，HC current-clamp voltage 的受限单通道 affine + 已知仪器测量算子 | HC voltage 不直接等于 feedback；H2/未分型 HC 不能自动标成 H1 |
| `H1_feedback` | `F(a_H * h_H)` 返回到输入侧的作用 | 目前没有已闭合的直接观测；需特定反馈作用测量及独立前端标定 | HC 电压、cone surround 或 H1 RF 幅度本身 |
| `u_B` | 去除 H1 feedback 后的 effective BC input | 目前无已验证的真实 EPSC 测量方程 | BC voltage-clamp input current，即使名称同为 input 也不能 identity/affine 接入 |
| `s_B` | BC 动力学的有效状态 | 经验证的 BC current-clamp membrane voltage，按类型/节点选择并作受限 affine；现有直接 raw 合同尚缺 | BC EPSC、RGC EPSC、模型反推 subunit、光电流 |
| `delta_r_B` | LegacyPReLU 的有符号 effective output | 需独立释放/突触输出观测及可验证的基线、传感器映射；当前没有 READY 数据 | 非负绝对 glutamate rate、BC voltage、RGC 汇总 EPSC；不能 abs/clamp 后冒称 release |
| `a_A` | local/broad × ON/OFF AC effective state | 仅在记录类型与该有效 family、位置、测量量对应成立时使用受限 voltage head | 自动把 SAC/PAC/AII voltage 填入任意 AC 节点 |
| `delta_o_A` | AC 基线相减的 effective output | 需该通路的输出测量及已知 sensor operator；目前未闭合 | AC voltage 或 RGC IPSC 本身 |
| `d_E` / `d_I` | conductance 非线性之前的净 effective drives | 可由合格 E/I current 经完整前向间接约束；无直接同单位真实 target | EPSC/IPSC 不是 d_E/d_I，也不直接等于某个 coupling |
| `gE` / `gI` | 正的 normalized conductance，含 tonic 项 | 已分离 voltage-clamp E/I currents，经已知 holding/reversal potentials 与独立 conductance 标定映射 | 电流直接除以任意电压；未经校正的 total current；把归一化值称 nS |
| `V` | 固定 E/I integration 的 effective voltage modes | 需独立证明与 RGC subthreshold current-clamp voltage 的关系；只允许固定聚合后受限 affine | voltage-clamp 命令电位；有 action potential 的完整 Vm 不能直接当本端口 |
| `spike` | logit/probability 定义的条件 Bernoulli 输出 | 单细胞逐事件 timestamps → 固定 150 Hz bin 的 occupancy；identity 概率观测，无自由 rate decoder | 平滑 PSTH、平均 firing rate、任意 count 直接作为 Bernoulli target |

“有条件可直接监督”表示测量方程闭合后可对该端口建立 loss，不表示 raw 已到手或恒等于生理变量。Affine 仅允许每个已登记测量通道一个固定符号的 gain 和一个 baseline；优先独立标定并固定。确有必要估计的 nuisance 只能使用训练集中的标定/重复记录，事先限定自由度与范围；不能以拟合 RGC 的好坏猜未知绝对光强、空间范围或时钟。未标定的 gain 必须保留为 nuisance，不能同时宣称恢复绝对 state/coupling。

仪器滤波仅使用有依据的 acquisition/filter/latency 描述；没有描述则记未知，不拟合任意 temporal kernel 来弥补机制。不能用双向滤波或未来刺激/事件构建过去时点预测。图像归一化、试次平均和 trace 标准化均须保留原始定义，不按每段响应重新 z-score 后称跨数据集同单位。

### 2.2 RGC E/I current：首要的真实中间观测合同

外向电流为正时，独立 voltage-clamp 观测方程为：

\[
\hat I_E(t)=\mathcal M_E\{\kappa_E\,\bar g_E(t)[V_{hold}(t)-E_E]\}+b_E,
\qquad
\hat I_I(t)=\mathcal M_I\{\kappa_I\,\bar g_I(t)[V_{hold}(t)-E_I]\}+b_I.
\]

`bar g` 使用当前两个 effective modes 的**预先声明、固定的聚合**；首版候选为等权平均，与现有 modes 的等权读出约定一致，但这只是待验证的 soma-current 测量假设，不是已证明的空间钳等价。不可引入可学习的逐 mode 电流 decoder。`kappa` 是 normalized conductance→nS 的测量尺度，不是新增膜或通路参数；有独立标定才能赋予物理单位。`M` 只包含已知采样/仪器算子。若无法验证空间钳与该聚合，电流合同不能升级 READY。

必须取得逐记录 holding potential、cation/chloride reversal、液接电位和 series-resistance correction 状态、pA 单位、leak/offset 处理、刺激前基线及药物条件。`V_hold` 来自实验，**不用模型的自由 V 代替**。一个 holding potential 下的混合 current 不能人为拆成两个监督通道；仅在分离依据成立时使用上述独立 E/I loss。

若原数据已扣除 baseline，比较同一定义的 `I(t)-I_baseline`，并保存扣除算子；不得因此删除模型 tonic conductance。没有 absolute current scale/基线时可登记相对 current 合同候选，但不能据此宣称绝对 gE/gI、gamma_BR/gamma_AR 或真实突触强度恢复，也不能把未知 kappa 默认为 1。

Liu2021 的方法记录了通过约 −70 mV 与 0 mV holding 分离兴奋与抑制电流，并声明数据可合理请求；这不是每条试次的实际校正值，也不是本项目已取得 raw 的证明。[原始方法与数据声明](https://pmc.ncbi.nlm.nih.gov/articles/PMC8728393/)

### 2.3 数据项允许更新的参数

| 合格观测 loss | 允许到达的模型祖先参数 | 不应由该数据项直接更新 |
|---|---|---|
| HC `h_H` | H1 state tau/delay | a_H、BC output、BR/BA/AR、RGC readout |
| BC `s_B` | H1 state、a_H、BC state | BC alpha、BR/BA/AR |
| BC `delta_r_B` | 上项 + BC alpha | BR/BA/AR、AC、RGC readout |
| AC `a_A` / `delta_o_A` | 对应 AC 分支祖先；output loss 才包含 AC output 参数 | gamma_BR、gamma_AR；AC state loss 还不含 AC output 参数 |
| 分离 E current | H1/BC state/output、a_H、gamma_BR、已登记测量 nuisance | AC/AR、自由 V 或 RGC bias |
| 分离 I current | H1/BC state/output、a_H、AC state/output、gamma_AR、测量 nuisance | gamma_BR、自由 V 或 RGC bias |
| RGC subthreshold V（若合同成立） | E/I 两支及其上游 | 不自动包含只出现在 logit 的 bias/history |
| RGC spike | 两支祖先与现有 readout | 未观测实例的独立参数 |

gamma_BA 仍固定为 1，有图依赖但无 optimizer 更新。Hierarchy/prior 梯度另列；不能把 prior 收缩当作某 dataset 提供了该参数的数据证据。梯度可达不等于非零、实际更新或唯一可辨识。

## 3. 物理输入、覆盖与实例所有权

### 3.1 Physical stimulus contract

输入保持 `[batch,time,pixel]` 的已知相对刺激，显式登记像素在 deg 中的边界、面积及固定 150 Hz 的时间坐标。当前原型 `Stimulus.validate` 只接受完整已知输入；本轮没有实现缺失像素推断或可变时钟。Q 保持 `area(node aperture ∩ pixel)/area(full node aperture)`，不截断后重归一化。

当前模板的 Q/H1/BC 坐标为两个轴上 `−0.30, −0.15, 0, 0.15, 0.30 deg`，Q aperture 边长 `0.10 deg`；25 个 Q aperture 的包围范围是 `[−0.35,+0.35]² deg`。它们是 effective circuit nodes，不能仅凭坐标宣称为真实 cone/HC/BC mosaic。完整 coverage 按所监督端口的**结构性祖先支持**检查，包括 H1 返回与 broad AC；零耦合、block 或低响应不能缩小所需范围。刺激窗口外只有实测已知背景才可填入，未知区域不能设零。源码入口：[Q/geometry](../experiments/retipath_multiscale_v0/circuit.py)、[Population stimulus/support/forward](../experiments/retipath_population_v0_1/circuit.py)。

μm→deg 必须有相应物种、眼球/retina 的标定或明确适用的实验换算；不直接借用另一物种的常数。原点与旋转只能依据刺激—细胞定位登记，不允许为拟合重新缩放 FOV。真实单细胞电压必须有该细胞与模型节点的对应证据；不能把任意最近节点、可学习空间 mixture 或任意插值当作已识别的同一细胞。若固定模板无法容纳实际位置关系，记录不兼容，不能暗改节点或连线。

对每个端口分别登记四个 validity 条件：空间支持完整、时间/同步完整、单位/基线已定义、身份/端口对应成立。另列 stimulus excitation coverage：空间/极性/频率/历史是否充分刺激目标通路。均匀 spot 即使完整覆盖，也不自动约束所有空间 coupling；validity 通过不等于可辨识性通过。Normal/block 的任何比较使用两者支持的并集。

150 Hz 输入与 10/25 kHz acquisition、60/85/120 Hz display 分开记录。非 150 Hz 的真实刺激必须由实际 onset/offset 生成事前固定的物理时间表示，并声明 150 Hz 离散近似及仪器延迟；不能仅改时间标签，不能把论文 60 Hz response bins 当 150 Hz raw events。当前阶段不声称这样的动态重采样已验证，因此这类条目仍有 calibration gate。禁止通过拟合 NLL 选择帧零点或 temporal shift。

Spike 的当前输出合同是每 bin 至少一次事件的概率。保留原始 event counts 以注明 binarization 丢失的多 spike 信息，不自动改成 Poisson count model。history 只读取严格过去的事件；训练和评价使用同一事前定义的 conditional 方案。当前 forward 在独立 sequence reset；continuous-record 切窗须有真实已知的 prehistory/warm-up 和明确 loss mask，不靠跨 trial 拼接或默认已知的未知初态制造连续性。

### 3.2 跨 dataset 共享与独立 retina instance

下列所有数据卡中的 **S0 / R0** 是逐 dataset 合同的一部分：

- **S0（可共享）**：固定计算图、Q 的物理积分定义、端口意义、LegacyPReLU、参数边界与已登记的 family/prior 形式。只有 species、cell family、偏心度、光照/preparation 可比且谱系无重叠证据时，未来协议才能显式指定某个 family 参数/中心跨实例共享；“都是 primate”不构成数值共享依据。本 v0 不自动建立任何新的跨来源 hard parameter tie。
- **R0（独立所有）**：未知或不同 retina 的 family 实例值、H1/BC 个体偏差、a_H/BR/AR coupling、RGC bias、初态，以及坐标登记、背景、仪器校准与观测 nuisance。AC dynamics 仍按本实例的四个 family 共享，不因每条记录而新增 unit 自由度；gamma_BA 固定。固定 prior 宽度/中心不等于真实 population 分布，B.2 的 anchor-free 诊断不自动更改本合同的 prior。
- 已验证为**同一生物细胞**的重复记录才共享该 cell 参数；同 retina 的不同细胞也不自动是已知连接对。同 cell 改变 clamp/药物/preparation 还须核对记录稳定性及测量条件。跨实验相同名称的字段不能替代 lineage。
- Dataset/session ID 只在外部注册层解析来源、实例和测量元数据；backbone 不读取 ID 选择机制。重复 deposit/mirror 或共享原始样本不能记成独立验证。

这一区分有实际训练后果：若 HC 来自独立 retina，且没有事前许可的共享参数，HC loss **不能**更新另一套 Schottdorf–Lee 实例的 H1 参数。把两个独立 loss 相加并不自动构成约束同一回路的 multi-level supervision。若未来采用同类 family 的统计共享，结果只能称跨实例约束，不能称记录了一条真实 synaptic chain。

## 4. Dataset-by-dataset Observation Contracts

以下 25 个 ID 与既有地图一一对应，另补 physiology audit 的两项 RGC-current 候选。`时间`同时列 acquisition 与 stimulus；`未知`均表示当前证据未闭合，不能由合成数据默认值填充。所有条目的 S0/R0 均执行上一节；表中再写特殊的共享/配对限制。

### 4.1 SL21_10MIN — Schottdorf–Lee macaque RGC

**状态：NEEDS_CALIBRATION。来源：**[冻结 DOI archive](https://doi.gin.g-node.org/10.12751/g-node.xage77/)、[地图条目](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#sl21_10min)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | M. fascicularis，在体麻醉、单细胞 RGC；有 MC/PC/S-on。首版仅纳入已核实 ON/OFF MC/parasol 类别，PC/midget/S-on 不自动迁入本架构 |
| Stimulus / FOV / time / input | 10-min natural movie，256×256，4.6×4.6 deg，150 Hz。逐事件单位 0.1 ms 的本地证据与具体记录绑定。相对 RGB 标定后 L+M Weber contrast；absolute R*/cone/s 未闭合，不猜绝对光强 |
| Measurement / port / direct | 单细胞 spikes → `spike`，端口类别可直接监督；未测任何 H1/BC/AC 状态或 coupling。当前时间 gate 未过，不能立即启动新的 Population fit |
| Adapter / calibration | Event-to-occupancy + 已知时钟与细胞 polarity selector，无自由 head。校正后的 live-relative 时间不再减 VideoStart。旧 loader 起始 frame=751 与解码首内容 frame=750 的绑定争议仍为 UNVERIFIED；禁止从拟合结果择优 |
| Coverage | 原始 4.6 deg 场尺寸已知。旧 51-pixel crop→17×17 pool 的物理宽度是 51×4.6/256=0.91640625 deg，并非 2 deg。居中时尺寸足以包围当前 0.70 deg Q 模板，但实际 RF 原点、完整 ancestor support 与 pooling 丢失的细节仍须登记；不能 resize/重标像素度数代替 |
| Pairing / sharing | 各记录的 animal/retina/cell lineage 按原 metadata；没有与本清单 HC/BC 的同 retina 证据。S0/R0；不同 RGC 不自动配成真实 ON/OFF pair。未观测输出用 mask，不当负样本 |
| Raw / remaining gaps | 冻结 archive 核实 15 个 10-min cells；28 个 raw TXT=15 ten-minute+13 six-repeat，不是论文全体47 cells，也不等于项目其他22-cell cohort。6×1 对应 movie 在该 archive 未闭合。下一步缺项是同步外部证据、cell crosswalk、几何/历史与 split-consumption 清单，不是重训选择时钟 |

时间依据：[frame-zero resolution](../.omo/evidence/schottdorf_lee_frame_zero_resolution.md)、[timing final check](../.omo/evidence/schottdorf_lee_timing_contract_final_check.md)。输入定义只作源码阅读，见 [现有 loader](../data/schottdorf_lee_2021.py)。旧模型的既有拟合与结果保持；其可运行不自动证明新 Population 合同 READY。

### 4.2 RA26_HC — Raval2026 HC

**状态：NEEDS_RAW_DATA。来源：**[RA26_HC](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#ra26_hc)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | Macaque（论文涉及 nemestrina/mulatta/fascicularis）；主要离体 flatmount/RPE，部分 HC/BC slice；20–50 deg 外周，逐 cell species/preparation 未闭合；HC 不自动认作 H1 |
| Stimulus / FOV / time / input | Noise、contrast-reversing gratings、spot；500 μm spot 不等于完整图像 FOV。采集10 kHz/3 kHz filter，显示刷新未知。405 nm/空间显示装置；部分条件1500/5000/15000 R*/cone/s，逐试次光谱和强度未知 |
| Measurement / port / direct | Current-clamp Vm 是 `h_H` 的有条件候选；部分 voltage-clamp input currents 单独分组，不能作 h_H。都不能直接监督 `H1_feedback` |
| Adapter / calibration | H1 身份及状态对应成立后才用单通道受限 voltage affine/已知 acquisition operator；需 mV baseline、polarity、gain、同步、μm→deg、适应历史。HC 非线性若超出冻结 H1 表达范围，原样保留失配，不加非线性 head |
| Coverage | Spot/bar 宽度有描述，完整已呈现区域、cell 原点和 H1 feedback 祖先输入未闭合；不能只用 spot diameter 判足够 |
| Pairing / sharing | 与 RA26_BC/RGC 同论文，不证明同 retina 或连接；S0/R0，原始 ID 取得前均独立 provenance instance |
| Raw / remaining gaps | Raw trials、实际刺激、同步、H1/H2/subtype、retina/cell IDs、标定均需补齐。未核实作者提供承诺；本轮不联系作者 |

### 4.3 RA26_BC — Raval2026 BC excitatory input current

**状态：NOT_DIRECTLY_COMPATIBLE；另缺 raw。来源：**[RA26_BC](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#ra26_bc)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | 与 RA26 来源相同的 macaque 离体实验；BC subtype/ON–OFF/逐 cell preparation 未闭合 |
| Stimulus / FOV / time / input | Spots/gratings/noise、μm 尺寸；完整 FOV 和 display clock 未知。采集10 kHz、3 kHz filter；逐记录 contrast、光强/光谱与背景需原始日志 |
| Measurement / port / direct | 已审计 BC 记录全部是 voltage-clamp **excitatory input current**。当前 `u_B` 没有 EPSC generative mapping；更不能当 `s_B` 或 `delta_r_B`，因此无直接监督端口 |
| Adapter / calibration | 不许可用 affine 把 pA 改名 state/output。即使补齐 holding/reversal、baseline、pA 标定，仍缺对应的物理测量模型；本轮不新增 |
| Coverage | BC input 的空间来源与完整场、μm→deg、节点配准未知，不能声称足够 |
| Pairing / sharing | S0/R0；与 HC/RGC 不构成已验证突触链。仅能保留论文功能现象，不为该 current 开放 BC state/output 梯度 |
| Raw / remaining gaps | 可申请 raw、holding、subtype 和刺激；取得后仍是输入电流。其地位不会因数据量增大自动升级为 direct BC supervision |

### 4.4 RA26_RGC — Raval2026 parasol spikes

**状态：NEEDS_RAW_DATA。来源：**[RA26_RGC](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#ra26_rgc)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | Macaque 离体 parasol；Fig8 为 ON 示例，不能推定所有 trial 为 ON 或所有个体配对 |
| Stimulus / FOV / time / input | Spots/gratings/context protocols；部分 center-masked 条件与 HC 的输入不同。500 μm spot/μm bar width；完整 FOV/refresh/原始帧未知；acquisition10 kHz；条件光强同 RA26 日志范围，不外推至所有 trial |
| Measurement / port / direct | Extracellular glass-electrode spikes → `spike`，可在 raw 合同成立后直接监督；不是 RGC intracellular 或 BC output 数据 |
| Adapter / calibration | Event occupancy/固定 polarity mask；需 spike clock、显示同步、contrast 定义、实际 LM 输入和 μm→deg。示例6 repeats 不当作全库重复数 |
| Coverage | 需真实 mask 外区域和历史；不可用 HC 未遮罩刺激替代 RGC masked stimulus。自然电影训练输入未核实 |
| Pairing / sharing | S0/R0；与 RA26_HC 可构成同源但未配对候选，不能默认共享个体参数 |
| Raw / remaining gaps | Raw events、trial stimuli、metadata、类型、IDs 和标定缺失；提供承诺未核实 |

### 4.5 LI21_HC — Liu2021 HC voltage

**状态：NEEDS_RAW_DATA。来源：**[LI21_HC](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#li21_hc)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | Macaque 三种来源，离体外周约2–8 mm/10–30 deg；HC preparation 去除 RPE，H1/H2 未确认；论文该 HC 证据 n=2 |
| Stimulus / FOV / time / input | Noise/glider motion；60 Hz display，10 kHz acquisition/3 kHz filter。μm 空间定义，完整 FOV/逐 trial pixel geometry 未闭合；Michelson contrast 与 photopic背景有方法描述，不替代逐 cell 校准 |
| Measurement / port / direct | Current-clamp HC Vm → 有条件 `h_H`；不能直接监督 feedback |
| Adapter / calibration | 受限 voltage affine；需 H1 身份、baseline/单位、实际帧/同步、位置与 μm→deg、初态和适应日志 |
| Coverage | 完整刺激范围与相对于 HC 的位置不足；glider 的时空带宽不自动覆盖所有 H1 参数 |
| Pairing / sharing | LI21_RGC 通常保留 RPE，本 HC 去 RPE；不能仅凭同篇论文共享同 retina 参数。S0/R0；与 RA26 独立性须 ID 证明 |
| Raw / remaining gaps | 合理请求的数据声明已核实，尚无本地可执行 raw 合同；n=2 不是支持广泛生物 population 结论的样本 |

### 4.6 LI21_RGC — Liu2021 parasol E/I currents 与 spikes

**状态：NEEDS_RAW_DATA。来源：**[LI21_RGC](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#li21_rgc)、[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8728393/)。真实中间观测的优先接入候选。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | Macaque 离体外周，RPE 保留；parasol 与 smooth monostratified 等分开。首版仅已确认 ON/OFF parasol，不以同为RGC把其余类型纳入 |
| Stimulus / FOV / time / input | Noise/glider、60 Hz；10 kHz acquisition、3 kHz filter；实际像素/完整 FOV/逐 cell 光谱标定未知。方法有 photopic LM 与 Michelson contrast，输入必须从实际刺激建立，不能从离散信息分析图反推 |
| Measurement / port / direct | RGC EPSC/IPSC → 第2.2节的 `gE`/`gI` current observation；extracellular events 另接 `spike`。已知某 cell 测过哪些模式才能建立多模态 loss。不能把 RGC EPSC 当每个 BC 的 delta_r_B |
| Adapter / calibration | 低容量 clamp-current operator；已报道约−70/0 mV 分离 E/I，逐 trial holding、reversal、junction/Rs correction、pA baseline、leak、仪器时程仍须取得。Free V 不进入 clamp 驱动力 |
| Coverage | Glider 的全场范围、retinal magnification、记录 cell 与模型原点、ancestor 支持均需原始日志；不能据“运动刺激”判 broad AC 已被充分约束 |
| Pairing / sharing | 仅同 cell、同 preparation 且稳定的 E/I/spike trials 可共享个体参数；现有审计未闭合同 cell crosswalk。EPSC/IPSC 各组 N 不相加当 unique cells。与 HC 非配对，S0/R0 |
| Raw / remaining gaps | Reasonable-request 来源；缺原始 currents/events/stimuli、模式与 cell 的关系、完整标定。E current 约束净 direct→conductance，I current 约束净 AC→conductance，不单独证明 gamma_AR 或 AC subtype |

### 4.7 CH24_INNER — Chen2024 Figure10 内层记录

**状态：NEEDS_RAW_DATA。来源：**[CH24_INNER](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#ch24_inner)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | Macaque 离体外周>20 deg；HC subtype 未明、3个BC测量模态未闭合、ON parasol |
| Stimulus / FOV / time / input | Step+flash original/modified stimulus，60 Hz OLED；其他 sinusoid assay 分开。完整 FOV、acquisition time、原始输入单位/逐 trial 光强未闭合 |
| Measurement / port / direct | HC mV→候选 h_H；parasol 原始 spikes→spike。BC trajectory-area/图示摘要不能判为 membrane state 或 output，**暂不分配 s_B/delta_r_B loss** |
| Adapter / calibration | HC 受限 voltage head、RGC event head 的候选；需 raw 和测量方法先闭合。平均 rate 不直接当单试次 occupancy，不能拟合 decoder 去解释模态未知的 BC 图 |
| Coverage | 图示刺激不等于可重建完整物理场；位置/范围未知 |
| Pairing / sharing | 同图展示不证明同 retina/cell；S0/R0。不能把 CH24_CONE deposit 当这部分 raw |
| Raw / remaining gaps | Figure10 raw 未进入已审计 Dryad；请求 raw、同步、BC measurement 类型、身份与校准，提供承诺未核实 |

### 4.8 DA00_BC — Dacey2000 cone bipolar intracellular records

**状态：NEEDS_RAW_DATA。来源：**[DA00_BC](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#da00_bc)、[论文记录](https://pubmed.ncbi.nlm.nih.gov/10837827/)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | 离体 macaque fascicularis/nemestrina 和 baboon；形态鉴定 cone BC，逐 cell subtype/原始记录归属待取得，物种不能合并 |
| Stimulus / FOV / time / input | Center/surround spots、annuli；本地审计未闭合完整 FOV、时间采样/刷新、输入单位、背景与 μm→deg |
| Measurement / port / direct | Intracellular light response；仅在 raw 确认 current-clamp Vm、单位、类型与状态对应后，作为 `s_B` 候选；不直接赋予 delta_r_B |
| Adapter / calibration | 受限 Vm affine + 已知 acquisition operator；缺 baseline、单位/同步、位置与 per-cell 类型，禁止把出版图数字化当原始训练 trace |
| Coverage | Center/surround 尺寸与全域输入日志未闭合；不能确认完整图有效 |
| Pairing / sharing | S0/R0；没有与清单 RGC/HC 的真实连接证据；跨 baboon/macaque 不硬共享 family 数值 |
| Raw / remaining gaps | Paper-only，需确认原始资料是否仍存在及可取得；没有已核实的提供承诺。这是 BC membrane 候选，不是已可用 BC output 数据 |

### 4.9 PD02_HC — Packer–Dacey2002 H1

**状态：NEEDS_RAW_DATA。来源：**[PD02_HC](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#pd02_hc)、[论文](https://doi.org/10.1167/2.4.1)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | Macaque 离体 H1，具体 animal/retina/cell metadata 未闭合 |
| Stimulus / FOV / time / input | Spots/annuli、spatial RF；完整 FOV、display/acquisition clocks、输入单位/背景与逐 cell 几何未知 |
| Measurement / port / direct | Intracellular Vm raw 若取得可候选 `h_H`；出版 RF/幅度曲线不是逐时点 h_H target，更不是 H1_feedback |
| Adapter / calibration | 受限 voltage affine，需 mV baseline、刺激日志、空间和时钟标定；RF summary 只保留已有功能证据 |
| Coverage | H1 RF 尺寸描述不自动给出每 trial 的完整已知输入，需原始范围与边界日志 |
| Pairing / sharing | S0/R0；H1 身份比未分型 HC 明确，但不因此与别处 RGC 共用个体 a_H |
| Raw / remaining gaps | Paper-only，原始 trace/元数据可得性未知，未核实作者承诺 |

### 4.10 GR18_HC / GR18_RGC / GR18_AII — Grimes2018 rod-regime 记录

三条分别登记，**均为 NOT_DIRECTLY_COMPATIBLE** 当前 L+M cone-parasol 输入合同；processed/raw 缺项另列。来源：[HC](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#gr18_hc)、[RGC](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#gr18_rgc)、[AII](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#gr18_aii)。

| 字段 | GR18_HC | GR18_RGC | GR18_AII |
|---|---|---|---|
| Species/preparation/cell | Macaque 离体、去RPE、外周≥15 deg；H1，Fig5 n=4 | 同类 preparation；ON parasol，Fig5 current n=2，论文另有 spikes | 同类 preparation；AII |
| Stimulus/FOV/time/input | 500–560 μm uniform disk；LED405/520/640，flash10ms/sinusoids/steps；acquisition/refresh未知；rod scotopic/mesopic，Fig5约10–20 R*/rod/s | Fig5与HC相同背景/刺激，contrast按parasol2Hz反应匹配；不是 photopic LM-only 输入 | 论文 rod-regime protocols；逐记录时钟、完整FOV/绝对输入需日志 |
| Measured / port / direct | H1 Vm，形式上候选h_H，但当前无rod输入，不能直接监督 | Synaptic current / spikes；类型上候选gE/gI或spike，具体current需分离条件；当前rod驱动不兼容 | AII Vm；冻结local/broad AC不含AII机制，无直接端口 |
| Adapter/calibration | Voltage head不能补不存在的rod drive；μm→deg、baseline和raw未知 | Clamp/event head不能把R*/rod/s改名L+M contrast；holding/leak/单位逐记录未知 | 不以affine把AII电压变成a_A；baseline/采样未知 |
| Spatial coverage | Spot直径已知，但全域边界/节点配准未闭合 | 同左，且same spot不能证明所有AC支持有效 | 同左 |
| Pairing/sharing | Fig5明确与parasol同一片retina、同背景/刺激；**不同cell，并非已知单突触链** | 与H1的same-piece证据保留；不能转移到全部论文records | AII与Fig5的配对未证实；S0/R0 |
| Raw/status detail | Source XLSX以modulation/平均为主，非完整raw；S0/R0 | Fig2 MAT示例不等于完整HC–RGC配对数据；S0/R0 | Raw/实际刺激未闭合；named rod pathway不得添加到本模型 |

“同片”是此清单中有价值但有限的身份事实；并不消除 rod/cone 输入差异，也不授权新增 rod/AII 模块。

### 4.11 CR14_RGC / TR16_RGC — physiology audit 的 E/I / intracellular 候选

这两个 ID 为本合同的引用标签，不新增已下载数据集；来源是已有 [physiology audit](../output/architecture_conformance_20260831/physiology.md) 的限定文献核对。**均为 NEEDS_RAW_DATA。**

| 字段 | CR14_RGC：Crook/Packer/Dacey2014 | TR16_RGC：Turner/Rieke2016 |
|---|---|---|
| Species / preparation / cell | Macaque 离体 ON/OFF parasol，photopic | M. nemestrina/mulatta/fascicularis 离体 ON/OFF parasol |
| Stimulus / physical / time / input | Spots/annuli、sinusoidal contrast；本地审计未闭合完整FOV、acquisition/display频率和逐记录单位/背景 | Natural images、gratings；完整FOV、帧时钟、acquisition和光强单位未在本地合同闭合 |
| Measurement / port / direct | Synaptic E/I currents、receptor-antagonist条件；合格baseline E/I current可候选gE/gI；若有spike或Vm raw需分别证明 | 兴奋性synaptic currents与RGC响应；可候选gE，实际spike/Vm模式须raw确认；推断出的rectified BC subunits不是delta_r_B观测 |
| Adapter / calibration | 第2.2节clamp adapter；holding/reversal、pA baseline、junction/Rs、仪器/药物状态均待原始日志 | 同一clamp合同；不能用BC output affine解释RGC汇总EPSC；holding/单位/baseline需取得 |
| Coverage | Annulus/spot描述不足以证明完整祖先支持；需物理尺寸、cell定位、μm→deg | 原始图像/边界、node定位、已知背景与适应历史缺失 |
| Pairing / sharing | S0/R0；同cell药物前后需身份和稳定性证据；pharmacology不等于模型block | S0/R0；ON/OFF差异不允许自动复制为已知BC individual参数；与其他论文独立性未闭合 |
| Raw / remaining gaps | 未有已核实raw包/提供承诺；[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4503220/)仅作来源入口 | 未有已核实raw包/提供承诺；[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4917290/)仅作来源入口 |

没有把“研究测量了 currents”写成“本项目已取得 E/I paired raw”。旧 audit 是有限正文核对，未知数字在此保持未知，不以常见电压或采样率补全。

### 4.12 KA25 — Karamanlis marmoset event-level sessions

**状态：NEEDS_CALIBRATION。来源：**[KA25](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#ka25)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | Callithrix jacchus，3只成年雄性、6个archive sessions；离体外周7–10 mm。功能分型ON/OFF parasol/midget/unclassified，首版仅有证据的parasol类；session不等于animal |
| Stimulus / FOV / time / input | Natural images+fixations、WN/flash/grating；marmoset显示85 Hz，WN按Nblinks变化。800×600、7.5 μm/px或部分2.5 μm/px，分别对应6×4.5 mm或2×1.5 mm，具体session须metadata；event sample index通过expdata.fs换时 |
| Measurement / port / direct | Spike-sorted未平均事件及unit IDs，实际图像/fixations/onset/offset目录已核实；候选spike，非原始MEA电压已验证的声明 |
| Adapter / calibration | Event head；需85→150 Hz物理时钟合同、marmoset μm→deg、类型/实例映射与实际grayscale输入。方法背景约3000 M-cone/6000 rod R*/s不等于每session已标定LM-only |
| Coverage | 像素/FOV必须逐session选用；完整自然图像/眼动边界、ancestor支持和cone主导适用性待登记。不能把鼠75Hz或macaque放大率套用 |
| Pairing / sharing | 同session多unit可有同retina证据，仍不是连接图；S0/R0，独立marmoset auxiliary instances，不与macaque H1/BC硬共享个体参数 |
| Raw / remaining gaps | 原始观测级availability已核实；CC BY-SA 4.0边界保留。缺的是本版物理/时间/类型与光照适用合同，不重复造数据 |

### 4.13 SR26 — Sridhar marmoset processed mirror

**状态：NEEDS_RAW_DATA。来源：**[SR26](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#sr26)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | Marmoset离体；ON/OFF parasol、OFF midget及Large OFF等分开，不将所有功能类都当parasol |
| Stimulus / FOV / time / input | 24Hz源电影经85Hz呈现与saccade/fixations，另有WN/grating；projector800×600、7.5μm，6×4.5mm；processed200×150。原采集25kHz，model bins85Hz、部分grating分析5ms不能混用 |
| Measurement / port / direct | 已核实的是processed mirror spikes/inputs；原GIN raw合同未闭合。事件语义和原始时钟确认后才候选spike，不把processed数组直接当raw |
| Adapter / calibration | Event/occupancy head；需processed count定义、实际frame timing、原始full-resolution输入、marmoset坐标。方法M/S-cone和rod背景有数值，session calibration仍未知；不能忽略movie与WN平均光强差异 |
| Coverage | 全显示FOV已报道，downsample是否保存Q所需局部信息及边界尚未闭合；不可恢复coarse pooling丢失内容 |
| Pairing / sharing | S0/R0；OpenRetina mirror不是独立实验。与KA25即使同物种也需排查animal/retina重用和条件差异，当前不硬共享 |
| Raw / remaining gaps | 原事件/原始刺激、trial/session映射、采样语义、独立性和标定待取得/核实 |

### 4.14 KR23 — Krüppel marmoset saccade/grating

**状态：NEEDS_RAW_DATA。来源：**[KR23](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#kr23)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | Marmoset离体；ON/OFF midget/parasol、Large OFF分开；parasol subset才有当前spike端口候选 |
| Stimulus / FOV / time / input | Grating saccade shifts/WN，fixation约533ms、shift67ms、stripe90μm；完整FOV、display/acquisition频率、绝对背景和输入单位未闭合 |
| Measurement / port / direct | Extracellular spikes → 有条件spike；没有内部层级观测 |
| Adapter / calibration | Event head；需timestamps单位、display sync、原始frames/位移、μm→deg、背景与类型metadata |
| Coverage | 条纹尺度不是完整FOV；祖先支持与偏心度对应未证明 |
| Pairing / sharing | S0/R0；retina/cell lineage未核实时保持独立，不与KA25/SR26拼样本或硬共享 |
| Raw / remaining gaps | GIN来源已列，但原始刺激—响应文件合同未核实；先补raw和完整日志 |

### 4.15 SH25 — Shahidi marmoset temporal RGC subset

**状态：NEEDS_RAW_DATA。来源：**[SH25](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#sh25)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | 仅marmoset离体subset；mouse/axolotl排除。ON/OFF transient/sustained是功能分组，不自动对应parasol |
| Stimulus / FOV / time / input | Full-field Gaussian noise/steps/chirps；60Hz OLED、800×600、7.5μm/px，即6×4.5mm；acquisition未闭合；mesopic–low photopic，方法0.75–2.8mW/m²，不自动等于LM contrast或cone-only regime |
| Measurement / port / direct | RGC spikes → 类型/输入成立后的spike候选；无内部state监督 |
| Adapter / calibration | Event head；需事件时间单位、光谱/rod贡献、实际输入、baseline/同步和μm→deg，不能借统一head消除regime差异 |
| Coverage | Full-field可有输入覆盖，但不能因此识别空间coupling；未知边缘/背景仍需日志 |
| Pairing / sharing | S0/R0；不同物种分开，不将功能cluster直接作为共享family标签 |
| Raw / remaining gaps | GRO来源列出但raw manifest未闭合；Figshare初始化参数表不是响应数据 |

### 4.16 HR21 — HumRet human RGC

**状态：NEEDS_RAW_DATA。来源：**[HR21](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#hr21)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | Human enucleation-derived离体retina、约25°C；16功能clusters不是已确认ON/OFF parasol身份 |
| Stimulus / FOV / time / input | Steps/chirps/gratings/bars，grating周期100–4000μm；25kHz events，显示刷新与完整FOV未知；rod背景8×10^4或8×10^5R*/s，cone输入标定缺失 |
| Measurement / port / direct | Event文件/reader已核实，实际呈现刺激与时钟配对未闭合；spike候选，不是READY |
| Adapter / calibration | Event head；需actual stimulus、timestamp/frame sync、human几何/温度/背景适用性与细胞类型，不能假设human=macaque |
| Coverage | 周期不是FOV，完整输入支持未证明；不可由论文pattern示意生成替代训练刺激 |
| Pairing / sharing | S0/R0，human独立instance；同session细胞可共享retina标签须有ID，不与macaque内部记录组成链 |
| Raw / remaining gaps | 缺实际刺激和对应日志，不只是再取spike文件；临床来源与发布边界保持 |

### 4.17 SH20 — Shah2020 parasol processed data

**状态：NOT_DIRECTLY_COMPATIBLE 当前公开版本。来源：**[SH20](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#sh20)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | Macaque离体外周6–15mm、ON/OFF parasol；RPE条件逐record核实 |
| Stimulus / FOV / time / input | WN/RF-null；CRT120Hz，刺激更新按panel为60或30Hz；采集20kHz，公开bin语义未闭合。像素41.6/20.8μm、完整维度按panel；contrast48/96%，LM约800–2200R*/s等方法值不能跨trial套用 |
| Measurement / port / direct | Public Time×cell spikes，但输入已STA temporal-prefiltered；不是当前Q要求的原始物理movie。原始输入未补齐前不直接监督spike；推断subunit不是BC实测 |
| Adapter / calibration | 不用可学习deconvolution/decoder“还原”刺激。若另取原始frames/events，需独立新版本合同、物理/时间/单位标定 |
| Coverage | 数值tensor尺寸不能证明未滤波物理支持；RF-null刺激与中心/周边关系需原始日志 |
| Pairing / sharing | S0/R0；Fig2–6 deposit不等于Fig7 natural-movie raw；与其他Shah数据须lineage去重 |
| Raw / remaining gaps | 当前processed tier不兼容；原始未预滤波输入/事件、完整FOV/同步才可能解除，不能本轮重造 |

### 4.18 CH24_CONE / BA19 / SA24 — cone 或 S-cone pathway

三条均为 **NOT_DIRECTLY_COMPATIBLE** 当前 q/内层state接口；单位或raw取得不会自动把photoreceptor trace变成q。来源：[CH24_CONE](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#ch24_cone)、[BA19](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#ba19)、[SA24](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#sa24)。

| 字段 | CH24_CONE | BA19 | SA24 |
|---|---|---|---|
| Species/preparation/cell | Macaque离体外周>20deg，L/M cones | Macaque离体foveal/peripheral L/M/S cones及small bistratified RGC | Macaque离体foveal<0.5mm/peripheral>6mm cones |
| Stimulus/FOV/time/input | 600μm均匀disk、405nm；0.1ms archive时间，noise≤60Hz、500ms mean steps；协议约22000R*/cone/s；actual vector待核 | 500μm disk、LED406/515/640；10kHz/3kHzfilter；10ms flash/noise/sinusoids/3s steps；cone约2500R*/s、SBC1000/10000，逐trial未知 | 500μm disk、LED410/505/650；10kHz/3kHzfilter；flash/step，方法0–50000R*/s；display/逐trial日志待取 |
| Measurement/port/direct | Whole-cell photocurrent，4–5 repeats平均并按saturating dark current归一化；不是q，也不是raw trials | Cone current/voltage、SBC spikes；cone非q，S/SBC通路不在LM-parasol图中 | Cone photovoltage（0pA current clamp）；另有HCN voltage commands，均无现有生理端口 |
| Adapter/calibration | 无许可的cone-state adapter；暗电流归一化、光谱/collecting area不等于本项目获得绝对input标定 | VC−60mV/CC0pA为方法值，单位/holding逐trial仍需核实；不用spike head把SBC冒充parasol | Vm baseline、drug/command/light分组需原始日志；不将任意voltage affine接到q |
| Coverage | Disk及cone记录不提供当前完整图的已知物理输入/动态cone模型 | Spot尺寸已知，full domain/μm→deg未闭合；S cone不可collapse进L+M | 同左；foveal/peripheral不能用缩放输入代替类型/条件差异 |
| Pairing/sharing | S0/R0；Figure10配对未知；复用Angueyra模型参数不是独立parameter truth | S0/R0；cone与SBC是否同retina/cell链未知 | S0/R0；部分S-cone资料复用BA19，不作为独立验证 |
| Raw/detail | Dryad为processed，不是内层raw；无需为当前任务扩cone模块 | ZIP存在但内部raw stimulus–response结构未核实 | XLSX processed；full raw需作者请求，不能把Source Data当完整raw |

已有 [Raval/Weber frontend feasibility](RETIPATH_RAVAL_WEBER_FRONTEND_FEASIBILITY.md) 的绝对背景/输出定义缺项保持；本合同不新增前端 gain 或动态参数来接纳这些记录。

### 4.19 KI22_AC — Kim2022 SAC/PAC

**状态：NOT_DIRECTLY_COMPATIBLE；另缺 raw。来源：**[KI22_AC](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#ki22_ac)。

| 字段 | 合同 |
|---|---|
| Species / preparation / cell | Macaque离体flatmount/RPE/choroid、约36°C；ON SAC、ON–OFF A1/PAC等命名类型 |
| Stimulus / FOV / time / input | Spots50–720μm/500ms、radial/drifting stimuli；SAC约0.24cycles/deg、0.5Hz，PAC200μm period/4Hz。10kHz、2/5kHzfilter；display刷新未知，23.4Hz calcium acquisition不是refresh；部分photopic约10^5R*/cone/s |
| Measurement / port / direct | SAC current-clamp Vm/部分inward currents、PAC Vm+spikes；不自动对应local/broad×polarity的a_A/delta_o_A；AC spikes不能当RGC spike |
| Adapter / calibration | 不设命名AC到通用AC的自由decoder；voltage/current/clamp与baseline需分开，测量量对应缺失优先于单位拟合 |
| Coverage | 完整FOV、cell相对位置、actual movies与history未闭合，directional family不能被空间mean替代 |
| Pairing / sharing | S0/R0；不借同species认定AC subtype与模型family相同；是否同retina/其他层配对未知 |
| Raw / remaining gaps | Reasonable-request；Zenodo中的NeuronC代码/形态不是raw physiology。即使取得raw，本轮也不新增SAC/PAC机制 |

### 4.20 AB22 / KW21 / SN04 — 非单细胞光驱动端口的来源

三条均为 **NOT_DIRECTLY_COMPATIBLE**。来源：[AB22](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#ab22)、[KW21](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#kw21)、[SN04](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md#sn04)。

| 字段 | AB22：Abbas2022 | KW21：Kawai2021 | SN04：Shen2004 |
|---|---|---|---|
| Species/preparation/cell | Human donor离体37°C，transretinal tissue | Human isolated BC，SCN2A channel experiments | Cultured human HC，subtype未闭合 |
| Stimulus/FOV/time/input | Flash；6mm punch/2mm aperture，photons/μm²；时间分辨率/完整场背景未知 | Voltage command/TTX，非light；物理FOV不适用，command采样日志未知 | AMPA/kainate化学刺激，非light；FOV不适用，采样/浓度逐记录日志未知 |
| Measurement/port/direct | Transretinal ERG field，不能当单BC s_B/output、q或RGC V | Channel current不是light-driven s_B/delta_r_B/u_B | Receptor current不是h_H/H1_feedback |
| Adapter/calibration | 禁止为ERG新增population decoder；电极baseline/单位与drug分离需raw但仍不闭合端口 | 无允许adapter；holding/units等即使补齐也缺light-driven测量语义 | 无允许adapter；current单位/baseline/holding未闭合，不把药物输入改名q |
| Coverage | Tissue aperture不等于node输入支持，需原始照明日志 | 不适用当前物理刺激图 | 不适用当前物理刺激图 |
| Pairing/sharing | S0/R0；human field不能与macaque单细胞组成链 | S0/R0；不同细胞/培养条件独立 | S0/R0；human culture不与exvivo H1共享个体参数 |
| Raw/detail | GIN processed CSV（b-wave/rod-cone等）不是单细胞raw | Paper-only | Paper-only |

## 5. 其他既有线索：保留来源，不冒充已可训练数据

以下只登记旧地图筛选阶段已出现的线索，不进行新的大审计。所有未列出的 acquisition、FOV、输入单位、holding、baseline、spatial coverage 和 per-cell identity 均为 **UNVERIFIED**；没有字段时不能默认值补齐。各条沿用 S0/R0、非配对边界，暂无任何可直接训练的新增组合。

| 来源 | Species / preparation / actual measurement / stimulus | Port、adapter、状态与缺项 |
|---|---|---|
| Shah2022 individual variability | Macaque大规模RGC，另有人类来源；exvivo，spikes/刺激记录；论文总样本量不是公开raw数量。具体FOV/时钟/单位/类型对应待manifest | parasol子集才候选spike/event head；**NEEDS_RAW_DATA**。完整raw需请求，公开subset合同未闭合；与SH20及其他Shah来源查lineage后才能判独立 |
| Wu2024 | Primate retina RGC、sorted spikes/刺激及模型资源；公开约129MB code/demo与未公开大型voltage分开 | **NEEDS_RAW_DATA**；spike候选需实际刺激与unit identity，prefit model不是观测。不能下载大型原电压来扩大本任务；完整FOV/校准未知 |
| Freeman2015 | Macaque RGC、single-cone-resolution stimuli/反推subunits；不是实测BC trace | **NEEDS_RAW_DATA**（RGC原始pair）；spike head候选，subunit不能接s_B/delta_r_B。FOV/time/units/同retina关系本地未闭合 |
| Angueyra2022 | Macaque cone/HC生理来源；cone trace无q端口，HC模态/身份与raw刺激需逐条确认 | **NEEDS_RAW_DATA**（HC候选h_H）；cone部分仍不直接兼容。电压head须先满足状态合同；Chen复用模型参数不形成独立teacher truth |
| Dunn2007 | Primate cone/HC/BC/parasol层级生理论文；本地仅paper metadata，测量模态、FOV/time/units和具体配对未闭合 | **NEEDS_RAW_DATA**。不得先按细胞名分配state/output；取得逐记录modality后才定h_H/s_B/current/spike候选，不能假定同链 |
| Greschner2014 | Primate PAC与RGC simultaneous MEA，spikes；实际刺激文件/时钟与完整raw可得性未闭合 | RGC部分**NEEDS_RAW_DATA**，parasol身份后候选spike；PAC部分**NOT_DIRECTLY_COMPATIBLE**通用AC state。同步记录是身份线索，不证明连接和state/output映射 |
| Percival2022 | Primate AII glutamate-puff currents与另组RGC light responses；不同assay分开 | AII非光current **NOT_DIRECTLY_COMPATIBLE**；RGC **NEEDS_RAW_DATA**、event/current候选须先确认模态。作者请求不等于已获得；无AII→RGC真实配对证明 |
| Soto2020 | Human离体RGC light responses，raw需请求；具体family、刺激FOV/time/units与本版映射未闭合 | **NEEDS_RAW_DATA**，spike候选；human独立instance，不以跨物种pooling解除校准 |

Gogliettino 的模型代码、Wang 的形态/另一物种 calcium assay、Cowan 的 atlas，以及非primate Grabner/Kuo 数据不升级为本轮 raw Observation Contract。Verweij2003 cone feedback 生理和 Jacoby–Marshak2000 电镜可限定机制解释，但前者未直接测本模型 H1_feedback，后者不是动态观测；均不生成内部state target。完整来源入口保留在 [地图筛选清单](RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md) 与 [physiology audit](../output/architecture_conformance_20260831/physiology.md)。

## 6. A — 当前可以立即用于真实训练的数据组合

严格按本合同，**当前 READY 组合为空，已核实的真实多层联合组合也为空**。这是当前项目证据缺项，非断言这些数据不可获得，也不撤销历史模型结果。

| 组合 | 已有基础 | 当前可以做的范围 / 不能跨过的门槛 |
|---|---|---|
| SL21_10MIN 单来源 RGC | 最完整的macaque事件/电影/相对标定资料，显示150Hz匹配 | 可据已有metadata制定接入清单；frame-zero、cell/type/位置、历史test消耗与有效支持闭合前，不启动新Population真实训练 |
| KA25 单来源 RGC | 原始观测级event/image/fixation资料已核实 | 是独立marmoset auxiliary候选；85Hz时钟、marmoset几何、光照与family映射未过，不与macaque数据拼成READY |
| RA26_HC + RA26_RGC | 同研究的跨层候选 | 缺raw/标定/retina lineage；BC input currents不能补成三层state supervision |
| LI21_RGC E/I + spike，合格时再加HC | currents作用点最接近所需中间约束 | raw和同cell crosswalk未取得；HC与RGC preparation差异须保留，不能称已配对 |
| SL21 + 独立HC/BC/current论文 | 可共享计算图和事前限定的统计假设 | 未有跨来源个体参数共享依据；相加loss不等于观测了同一真实回路 |

无需新增绝对光强就能定义的 relative-contrast RGC prediction，与需要绝对/相对电流标定的 conductance 解释是不同合同。不能因后者缺项把前者强行换成猜测的 R*/s；也不能因前者能运行就宣布内部层级已生理标定。

## 7. B — 仍缺失的关键 observation 与元数据

1. **RGC E/I current + 实际刺激 + 同 cell/retina crosswalk**：优先LI21_RGC，CR14/TR16仅作待核raw候选。需holding/reversal、pA和baseline、Rs/junction、光照历史、空间钳/聚合适用证据；若能与spike同cell配对，才可在一个实例内同时约束中间量与输出。
2. **身份明确的 H1 voltage raw**：RA26/LI21需H1 subtype核实；PD02虽明确H1，raw可得性未闭合。需要刺激、位置和measurement scale；任何HC voltage均不直接提供a_H或feedback真值。
3. **真正 BC membrane state 与 output 的不同观测**：DA00/CH24需先确认modality和原始资料。当前没有闭合的delta_r_B对应输出测量；Raval BC EPSC不能替代这项缺失，也不能因此宣称已经具备20%真实BC监督。
4. **共同的物理/时间/身份合同**：SL帧零点、各数据完整FOV、μm→deg、actual刺激、background、适应/历史、recording lineage和旧test消费表。实验n、平均trace、archive存在均不能补齐这些字段。
5. **真实干预和独立验证**：没有已验证的记录恰好实现模型BLOCK_DIRECT_BC_DRIVE/BLOCK_H1_FEEDBACK/BLOCK_AC_POSTSYNAPTIC_DRIVE。药理条件保留原实际作用范围；不产生伪造intervention targets。

## 8. C — 第一版真实 Population RetiPath 最小训练方案（条件性设计，未授权执行）

### 8.1 范围与准入

推荐范围保持**macaque、已识别ON/OFF parasol-related、与L+M相对输入相容的光照条件**。保留当前固定模板和LegacyPReLU；不纳入midget/S-cone/rod/AII/命名AC机制。先把SL21的时间与实例合同闭合，形成最小RGC-only真实接入；真实多层首选同实例的可标定E/I current+spike，再接入合格H1 voltage。BC只有取得语义匹配的state/output记录后才开放相应端口，不用错误量补层数。

这是单一路线的接入顺序，不是新增A/C/E实验条件或progressive训练日程。取得多模态后仍按joint objective；当前没有READY实例时不启动任何fit。若只能得到互不配对的记录，则各自独立instance；跨来源family共享必须先登记兼容证据及共享字段，不能在训练中自动决定。

保留H1/BC的层级个体约束、a_H强partial pooling、AC family-shared与gamma_BA=1。不依据B.1/B.2诊断改prior、损失或optimizer。现有固定节点不能表示实际cell配准时报告缺项，不把adapter变成新架构。

### 8.2 Observation batches、loss 与可见性

每个batch显式携带 `instance、physical stimulus、port、observed cell/node selector、time mask、units、measurement operator、trial lineage`。未知/未观测节点没有loss，不填零、不生成pseudo-target、不读其他记录的latent。Synthetic的5/25和10/50观测位置只是历史实验合同，**不强加到真实资料，也不代表有20%真实细胞覆盖**。

Joint目标只包括准入观测：

\[
L=\sum_{d\in D_{ready}}w_d\,L_d(\mathcal O_d[\mathrm{RetiPath}_{instance(d)}(X_d)],y_d)+P_{existing}.
\]

Spike用当前conditional Bernoulli occupancy loss。连续voltage/current用原始单位下的观测残差及事前声明的噪声模型；噪声尺度只能由训练侧仪器/重复记录估计，不能从test或synthetic的0.03借用。按有效时间点/观测channel归约，分别保留原单位误差；没有数据支撑时不预设H/BC为0.4。低容量adapter、loss weights、每dataset microbatch/exposure、总updates、final/checkpoint规则、optimizer及clip沿用项必须在**任何训练和test访问之前**单独冻结；本接口稿不以历史400步替代真实数据规模评估，也不授权调参。

Per-dataset raw/weighted gradient norm、state/output/coupling group norm、shared-coordinate cosine及hierarchy gradient分别记录，仅作诊断；不使用GradNorm/PCGrad或诊断驱动的自动权重/optimizer改变。不同实例独立参数之间的cosine为N/A，不能用同名字字段假装共享冲突。

### 8.3 划分与评价

在未来授权的payload访问前登记现有train/dev/test和已消费窗口；不得把已分析SL21片段或重复同一movie的trial重新称独立fresh test。同retina/cell重复、同刺激内容、同来源mirror与派生版本成组处理；任何跨retina验证都要求真实lineage而不是论文不同。没有独立可用记录时明确 `independent real test = NONE`，不自行开新时间段或重造数据。

真实评价限于实测观测：held-out spike CE/NLL、概率/重复响应的相容性，合格voltage/current的原单位误差与残差；如有重复记录可单独说明noise/重复性。**没有teacher时不报告excess CE、未观测h_H/s_B的真值RMSE或coupling parameter recovery。** 不新增成功阈值或显著性检验。模型内部state和model-intervention可保存为模型预测，但不是真实生理恢复；没有匹配干预数据就没有真实intervention recovery score。

本轮到接口文档停止。作者联系、raw下载、metadata/payload核验、adapter/trainer实现及训练均需相应后续授权；不存在自动推进到新synthetic优化或Stage C的步骤。

## 9. D — 暂时不能声称的生理结论

- 不能声称已经拥有或训练了一条来自同一retina、真实配对的HC→BC→AC→RGC synaptic chain；同论文、同物种乃至同片都不自动证明连接。
- 不能把拟合的h_H称已校准H1电压，把H1电压称feedback，把BC input current称membrane state/output，或把RGC EPSC/IPSC称单个BC/AC释放。
- 不能把normalized gE/gI/V、effective modes或LegacyPReLU signed output称nS/mV、真实树突区室、非负绝对释放率。
- 不能仅靠state拟合良好、低seed ambiguity或prior收缩宣称a_H/gamma_BR/gamma_AR被唯一识别；current约束的是对应净通路，不自动分解所有coupling。
- 不能把模型block等同药理阻断/真实细胞沉默；不能因天然图像RGC预测好就推断H1 feedback、BC非线性或AC机制在生物中被确认。
- 不能把synthetic A/A.5/B的方法验证或B.1/B.2诊断推广为真实population因果证据；不能把独立teacher当生物个体统计。
- 不能把macaque/marmoset/human、不同偏心度、RPE/温度、rod/cone光照或药物条件的记录视作同分布，并以自由decoder掩盖差异。
- 不能在缺绝对光强、FOV、时钟、初态或测量标定时通过“能训练”推断这些缺项已被解决；未取得或已消费的保留集也不能改名独立验证。

## 10. 本轮交付与停止记录

只新增本文并更新 [NEXT_TASK](NEXT_TASK.md) 的当前主线，原任务正文收进历史。未修改AGENTS、Population设计/源码、priors、旧模型、旧实验/数据/checkpoints、数据地图或旧审计；没有执行Git、读取新的spike/movie payload、重新评价test、训练、生成数据或联系作者。必要核对限于已有文档、元数据、相关源码文本与原始论文方法的窄范围复核。

下一主线是real-data integration；Stage A/A.5/B按用户决定结束方法学扩展，B.1/B.2仅诊断留档。具体接入实施与真实训练尚未开始，本轮完成后停止。
