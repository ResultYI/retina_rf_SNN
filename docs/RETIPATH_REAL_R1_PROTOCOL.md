# Population RetiPath real-data R1：冻结协议与 trainer 设计

日期：2026-09-20。状态：**FROZEN_PROTOCOL_AND_TRAINER_DESIGN_ONLY / NOT_RUN**。

本文件冻结第一次真实 macaque RGC-only Population 训练的 cohort、接口、预算、选模和评价合同。本轮只写本文；没有实现或运行 trainer，没有执行 optimizer step、生成 checkpoint、解码刺激、读取 spike payload、读取 validation 响应或加载模型权重。核对对象限于源码、R0 合同/清单以及旧运行的身份、split、训练配置元数据。元数据中的 validation bin 数不代表本轮读取或评价了 validation 响应。

推荐且唯一的 R1 方案是：**9 个独立 biological-cell instances；LegacyPReLU；全部当前参数 joint fit；沿用旧 Canonical R4 的 inner-dev 选步数、从相同初值 full-train refit；全部 9 个最终 checkpoint 冻结后统一评价旧 validation。** 不复制 synthetic Stage B 的 400-step 预算，不引入 B.1/B.2 的 loss/prior 变体。

旧 real-data 协议可以在保持数据和选模语义的前提下映射到 Population；没有需要用户另行决定的技术项目。执行训练仍需后续明确授权，本文件不构成启动训练的指令。

## 1. 冻结范围与证据来源

优先依据：

- [AGENTS.md](../AGENTS.md)、[Population v0.1 设计](RETIPATH_POPULATION_ARCHITECTURE_V0_1.md)。
- [真实数据接入合同](RETIPATH_REAL_DATA_INTEGRATION_V0.md)、[R0 preflight](RETIPATH_REAL_R0_PREFLIGHT.md)、[当前任务导航](NEXT_TASK.md)。
- 数据解析/物理预处理：`data/schottdorf_lee_2021.py`、`data/schottdorf_lee_multirecording.py`、`data/schottdorf_lee_spikes.py`、`data/schottdorf_lee_catalog.py`。
- 已通过的适配路径：`experiments/retipath_population_v0_1/real_data.py`；模型：同目录 `circuit.py`。
- split：`evaluation/mechanistic_retina/factorized_ln_split.py::make_inner_dev`。
- 预算与选模：`training/mechanistic_retina/r4_development.py::{fit_r4,select_and_refit_r4}`、`training/mechanistic_retina/center_surround_ln.py::DevelopmentStop`。
- optimizer / NLL：`training/mechanistic_retina/optimizer.py`、`training/mechanistic_retina/losses.py::expected_bernoulli_nll`。
- 旧对照来源：`evaluation/mechanistic_retina/spatial_contrast_source.py`，以及 `.omo/evidence/compact_causal_cnn_baseline/{prepare.py,train.py}` 的数据匹配逻辑。

这些来源用于核对合同；本轮没有调用其中的数据加载、训练或评价入口。保留原模型、数据、所有旧 checkpoint 和旧结果。

## 2. 固定 cohort：9 biological cells / 16 recordings

以下名单从 R0 `per_cell_preflight.csv` 的 MC 记录固定，不按响应幅度、拟合效果、cell 难度或任何 Population 结果重新筛选。MC ON 对应 Population ON port 0，MC OFF 对应 OFF port 1。PC/midget 全部排除，不实现 midget。

`seed` 逐 cell 复用旧 Canonical 22-cell 运行记录中的训练 seed；保留原索引，不把 9 个 cell 重新编号生成 seed。`validation bins` 是既有 split 的元数据计数，未来仍须逐 bin 核对。

| Biological cell | Type | R0 已通过 recording IDs（固定顺序） | Trials | Full-train sequences | Full-train scored bins | Validation scored bins | Seed |
|---|---|---|---:|---:|---:|---:|---:|
| 67#6 | MC OFF | lSS01078; lSS01079 | 7 | 112 | 13440 | 3360 | 20260829 |
| 67#7 | MC ON | lSS01086; lSS01087 | 7 | 112 | 13440 | 3360 | 20260830 |
| 67#33 | MC OFF | lSS01159; lSS01160 | 7 | 112 | 13440 | 3360 | 20260834 |
| 68#3 | MC OFF | lSS01181; lSS01183 | 7 | 112 | 13440 | 3360 | 20260836 |
| 68#10 | MC ON | lSS01221 | 1 | 16 | 1920 | 480 | 20260839 |
| 69#4 | MC ON | lSS01254 | 6 | 96 | 11520 | 2880 | 20260842 |
| 69#6 | MC OFF | lSS01256; lSS01257 | 7 | 112 | 13440 | 3360 | 20260843 |
| 69#7 | MC ON | lSS01258; lSS01259 | 7 | 112 | 13440 | 3360 | 20260844 |
| 70#34 | MC ON | lSS01299; lSS01300 | 7 | 112 | 13440 | 3360 | 20260849 |

合计：MC ON 5 cells / 28 trials，MC OFF 4 cells / 28 trials；16 recordings / 56 trials；896 full-train sequences、107520 full-train scored bins；224 validation sequences、26880 validation scored bins。16 recordings 包括 8 个长 recording 和 8 个六次重复 recording，当前比较只用旧合同指定的 live-relative 前 20 秒。

不得因为某个 cell 训练失败而改成 8-cell 汇总、替换 recording 或换 seed。失败时记录该 cell 的状态；在 9-cell matched 结果齐全前，不发布完整 cohort 比较结论。

## 3. Retina-instance 与参数所有权

1. 实例 ID 为 `SL21::<biological_cell_id>`。每个 cell 拥有一个独立 Population 参数集合和独立 optimizer；该 cell 的多 recording / trial 共用这套参数。
2. 不同 cell 不共享可训练 tensor、实际估计的 family center 或 optimizer state。只共用架构、ON/OFF 路由规则、参数变换、边界和 hierarchy/prior 的规则及固定超参数；不在本轮新增跨动物的共享 population center。
3. 动态 state 不在 sequence、trial、recording 或 cell 之间传递。每个 150-bin sequence 以当前模型的 baseline/reset 语义独立前向；batch 内每个样本有自己的 state。
4. R0 为逐 recording preflight 构造的 `SL21::<cell>::<recording>` 标识保留为 provenance key，**不能直接用作 R1 参数实例的 key**。R1 只在外部 registry 按 biological cell 归并参数所有权，不修改模型计算图。
5. 沿用旧 multi-recording loader 的 recording 顺序、每 recording 内的 segment-major/trial-minor 顺序，以及合并时的 trial-index offset。不能把来自不同 recording 的片段接成连续状态，也不将不同 cell 当作一条被真实同步观测的 synaptic chain。

MC ON/OFF 的 H1、ON/OFF BC、local/broad × polarity AC、direct/AC 分支和 E/I conductance 继续按当前模型存在。它们都是 latent；没有真实 H1、BC、AC 或 E/I 监督。只对该 cell 对应的一个 RGC port 计算观测 loss；另一个 port 既不是额外录到的 cell，也不提供负样本。

## 4. Stimulus、时间与 occupancy 合同

复用现有文件，不下载、替换或重新解释源数据：

- Movie：`data/real/schottdorf_lee_2021_macaque/1x10_256.mpg`；recording repository：`data/real/schottdorf_lee_2021_repository/`。
- 源显示：256 × 256，4.6 × 4.6 deg；中心 51 × 51 crop，3 × 3 pooling 后为 17 × 17。源 degree/pixel 为 0.01796875，pooled degree/pixel 为 0.05390625；实际 crop FOV 为 **0.91640625 × 0.91640625 deg**。
- 沿用原 RGB gamma / `/256` 运算顺序、L/M 权重、pooling 顺序和 baseline。background 来自 live 开始前的 decoded frames `[0,751)`；输入为同一局部 pooled L+M 相对 background 的 Weber drive。绝对光强和独立 cone 光子捕获标定仍未知，不补造 R*/s。
- 使用 R0 的显式像素边界 `[-0.458203125,+0.458203125]` deg 和像素面积进入 physical Q，不能把 17 × 17 index 直接当回路坐标。Population 25 个 Q aperture 的当前结构支持外包围为 `[-0.35,+0.35]` deg，固定模板落在该 FOV 内；此事实不证明所有真实 parasol 的完整生理 RF 都在 crop 内。
- 时间分辨率固定 150 Hz，dt = 1000/150 ms。模型使用原 adapter 的 bin 时间；全局 frame offset 由 metadata 保留。沿用工程 frame alignment：zero-based live start frame **751**。已有 750/751 绝对对齐争议继续标记 `UNVERIFIED`，本轮不利用响应择优选择 offset。
- Spike parser 沿用既有 corrected live-relative 0.1 ms ticks 和 binning，不再次减去 VideoStart。目标 `y_t = 1[count_t > 0]`；同 bin 多 spike 不改成 Poisson count loss。
- 长 recording 只取该旧窗口；六次重复按原 loader 的六个 trial 处理，保留第七维护列的原排除规则，不虚构跨 repeat 的连续时间轴。

所有 sequence 长 150 bins（1 秒）。本地 bins `[0,30)` 是 warmup；`[30,150)` 才是候选 scored bins。没有外部 pre-roll；模型仍在 warmup 内消费真实可用的刺激和 strictly-past spike history。该工程合同不能声称恢复了真实 recording 开始前的未知神经状态。

条件 Bernoulli 输入为真实已观测过去的 binary events，不做生成式 spike rollout。history 在更新顺序上严格延迟，`logit_t` 不可见 `y_t` 或未来 events；另一未观测 RGC port 的 history 输入为 adapter 约定的零占位，不产生 loss、也不宣称观测到零 spike。该路径复用 R0 的因果性合同。

## 5. Train / inner-dev / outer-validation

以每个 trial 的 live-relative bin 为索引；所有集合继续与原 150-bin segmentation / 30-bin warmup mask 相交。

| 用途 | 时间支持或目标区间 | 允许用于什么 |
|---|---|---|
| Full train | `[0,2400)` = `[0,16)` s | 最终 refit；constant baseline 的 occupancy 均值 |
| Inner fit | 输入和目标 `<1860` = `<12.4` s | 选步数阶段的 optimizer updates |
| Guard / inner-dev context | `[1860,1920)` = `[12.4,12.8)` s | 仅为 inner-dev 的过去输入/history；不属于 inner-fit 或 inner-dev scored targets |
| Inner dev targets | `[1920,2400)` = `[12.8,16)` s | 只选本 cell 的步数 K；不是外层 validation |
| Outer validation | `[2400,3000)` = `[16,20)` s | 全部 9 个最终 checkpoint 冻结后的单次统一评价 |
| 其余时段 / 新 test | 不开放 | 不训练、不评价、不补充 cohort |

上述边界来自 `make_inner_dev`：对 2400-bin train 取 4/5 边界 1920，并在前方留出 `CONTEXT_BINS=60`。此 60-bin guard/context 与每段 30-bin warmup 是两个概念；不能把 Population warmup 改成 60，也不能添加连续跨段 state。

严格复用 `_masked_split` 的语义：先按 input-support mask 处理 stimulus/counts/events，再保留至少有一个有效目标的 sequence。Inner-fit 最后一段在 1860 之后无可见输入或目标；inner-dev 第一段的 1860 之前输入/history 为 reset 占位，1860–1919 是真实 context，1920–1949 为可评分目标。其后每段仍排除本段前 30 bins。这个时间选择占位不能延伸为对未知空间区域填零。

因此每 trial 的 scored-bin 数是：inner fit **1470**、inner dev **390**、full train **1920**、outer validation **480**。九 cell 的 inner fit / dev 总计 **82320 / 21840** bins。Guard 的 60 bins 不计入 fit/dev loss；不要把连续时间范围长度当作 scored-bin 数。

Trainer 的数据对象只能包含 full train 和从中派生的 inner fit/dev。不能调用会同时物化 validation 的完整 `load_schottdorf_cell`、`spatial_contrast_source.load_sources`，也不能直接加载含 train+validation 的旧 CNN 输入 `.pt` 再切出 train。实现时应组合现有 train-only adapter 与上述 cell 合并/inner-split 规则，保持既有数值及顺序。

现有原始 spike 文本 parser 可能一次解析整份文件；其原始文件级访问与目标窗口隔离必须分别记账。R1 trainer 不得收到外层 validation tensor、events、targets、预测、NLL 或选模反馈，不能把整份 parser 结果缓存成 trainer 可访问的数据。未来若无法完成该隔离，报告数据接口阻塞，不通过调用旧 eager loader 绕过。本轮没有读取任何原始 spike 文件。

## 6. 模型、参数 hierarchy 与唯一训练 objective

模型保持当前 `PopulationRetipath` 的 Q、H1 feedback、BC input/state/output、AC 路由、conductance、history、intervention 和 parameter boundaries；BC output 固定 **LegacyPReLU**。不新增 adapter 参数、高容量 decoder、机制或 prior。

模型内部仍为 25 H1 / 50 BC / 36 AC / 2 RGC ports（各 2 integration modes），共 **359 个 raw trainable scalars / instance**。全部当前 `state`、`output`、`coupling` parameter groups 一次性 joint 优化，不逐层解冻、不 progressive。没有数据祖先关系的 port 参数可以只有 prior 梯度或零数据梯度；不能据此宣称所有 359 参数被真实数据辨识，也不能为强迫每个参数获得数据梯度而加伪 loss。

对所录 port 的 logit `z` 和 binary occupancy `y`，每个 microbatch 的数据项为：

`L_R = sum(mask * (softplus(z) - y*z)) / sum(mask)`，单位为 nats / scored bin。

唯一训练 objective：**`L = L_R + P_current`**，调用当前 `regularized_objective` 的相同语义。每 update 一个 batch，P 加一次，不按 cell 数、recording 数或参数组数再缩放。不使用 C 的 1/3 RGC 权重，不引入 H/BC/AC auxiliary loss。Inner-dev selection 和最终报告只使用 `L_R`，不把 P 算进预测 NLL。

`P_current` 是各 `PooledField.penalty()` 的和。在 raw bounded-sigmoid 坐标中，存在 deviation 时加 `0.5*sum((contrast/sd)^2)`；存在 constructor-center anchor 时加 `0.5*sum(((center-initial_center)/center_sd)^2)`。保持正交 zero-sum contrast 和当前 family 分组；不改为 physical-coordinate penalty 或均值归约。

| Field | Physical bounds | Constructor value | Group | Deviation sd | Center-anchor sd |
|---|---|---|---|---:|---:|
| tau_H | 10–100 ms | 50 | state | 0.3 | 无 |
| delay_H | 0–20 ms | 5 | state | 0.3 | 无 |
| a_H | 0–0.8 | 0.3 | coupling | 0.1 | 无 |
| tau_f_B | 8–40 ms | 22 | state | 0.6 | 无 |
| tau_gap_B | 20–140 ms | 78 | state | 0.6 | 无 |
| delay_B | 0–20 ms | 2 | state | 0.6 | 无 |
| kappa_B | 0–1 | 0.5 | state | 0.6 | 无 |
| alpha_B | 0.05–1 | 0.5 | output | 0.6 | 无 |
| tau_A | 20–200 ms | local ON/OFF 50；broad ON/OFF 140 | state | 无，family-shared | 无 |
| delay_A | 0–20 ms | 5 | state | 无，family-shared | 无 |
| b_A | −4–4 | 0 | output | 无，family-shared | 无 |
| gamma_BR | 0.2–8 | 1.5 | coupling | 无 | 0.3 |
| gamma_AR | 0–8 | 0.375 | coupling | 无 | 0.3 |
| RGC bias | −4–−0.5 | −2.4 | output | 无 | 0.3 |

`tau_s_B = tau_f_B + tau_gap_B`；`gamma_BA = 1` 继续固定。H1/BC 只在各自实例内 partial pooling；AC dynamics 继续 family-shared。**gamma_BR、gamma_AR、bias 的旧 constructor-center anchors 全部保留**，不采用 B.2 anchor-free；a_H 及其他 prior 也保持原样。表中数值是当前模型的工程合同，不是本次真实数据已测得的生理参数。

初始化使用当前 constructor 原值和零 contrasts；不加载 synthetic 或旧 Canonical 权重，不用 occupancy 估计值覆盖模型 bias/anchor。记录 seed 不意味着额外引入随机参数扰动。Constant baseline 的拟合与模型初始化分开。

## 7. 固定 optimizer、预算与 selection

### 7.1 来源与不可变配置

直接沿用 `r4_development.py` 的真实 Canonical 训练合同，而不是从 synthetic 阶段外推：

| 项目 | R1 固定值 / 语义 | 来源 |
|---|---|---|
| Optimizer | Adam；lr=0.03；betas=(0.9,0.999)，eps=1e−8，weight_decay=0，amsgrad=False | `build_phase1_optimizer` 及其未覆盖的 Adam 默认值 |
| Batch | 4 sequences / update；从该 cell 的对应 split 均匀有放回抽样 | `fit_r4` 的 `torch.randint` |
| Sampling generator | CPU generator；seed = cell seed + 1000003 | `fit_r4` |
| 最大 inner updates | 1000 / cell | `MAX_STEPS` |
| Dev 频率 | step 0 及之后每次 update | `fit_r4` |
| Patience | 200 次连续未达到 plateau-reference 改善的 dev 检查 | `DevelopmentStop` |
| Plateau min_delta | 1e−7 nats/bin；strict inequality | `MIN_DELTA` |
| Gradient clip | **无**；不从 Stage B 导入 clip=1 | 旧 real `fit_r4` 无 clip 调用 |
| LR schedule / sweep | 无；固定单个 lr，不复制 LN/CNN 的候选搜索 | R4 单配置 |
| Population prior | 第 6 节原样，系数 1 | 当前 Population objective |
| Runtime | CPU、float64、2 threads | 延续 R0 已验证 Population 数值路径 |

Population 使用其当前 bounded transforms，不调用 Canonical 专属 `project_mechanism_parameters()`，不增加 parameter-value projection。Optimizer 参数列表来自 Population 的三个当前 groups，去重后必须完整覆盖当前 trainable parameters；不调用假定旧模型类型的 `phase1_parameters()`。

旧 Canonical objective 是数据 NLL；R1 额外保留用户明确要求的当前 Population P。Population 的参数量、float64 路径和 objective 因而与旧模型不同。这里匹配的是既有数据合同及 R4 的更新/选模规则，不宣称所有 baseline 容量、正则、算力或数值实现相同。

### 7.2 每 cell 的选步数与 refit

1. 从固定 constructor 初始化该 cell。Step 0 的 inner-dev NLL 也是候选，因此 **K=0 合法**。
2. 在 inner fit 上 joint 更新，最多 1000 次。每次 update 后在完整 inner dev 上计算不含 P 的 NLL。
3. Best-step 规则：只要新 NLL **严格低于历史 best NLL** 就替换 best；相等时保留较早 step。`min_delta=1e−7` 只控制 plateau/patience，不作为 best-step 更新阈值。
4. 当 NLL 严格低于 `plateau_reference − 1e−7` 时重置 stale count，并更新 reference；否则 stale 加一。达到 200 后停止；若未触发则止于 1000。
5. 得到该 cell 的 K 后，丢弃 inner weights 和 optimizer，从**相同 constructor 初值、相同 cell seed、重置的 sampling generator**，在 full train 上重新 joint fit **恰好 K updates**。
6. Refit 不 early stop、不再选 checkpoint。最终权重是 K 次 full-train 更新后的唯一正式候选；K=0 时为初始权重。不能直接提交 inner 最佳权重，不能复用旧 Canonical 的 best K。
7. 先完成并锁定全部 9 个最终 checkpoint、K、配置与 hash，再开放 outer-validation evaluator。任何 outer NLL 都不能返回 trainer 或触发重跑。

预算上界为每 cell 1000 inner + 至多 1000 refit，九 cell 合计 **至多 18000 optimizer updates / 72000 sampled sequence exposures**。这是固定的上界和确定性停止规则，不是承诺每 cell 跑满 2000；inner-dev forward 和最终评价不计作 optimizer updates。不是本轮执行计数，本轮为 0。

Nonfinite loss/logit/gradient 或非法 mask 是实现/运行失败，应保留诊断并停止受影响 fit；不换 lr、seed、prior、dtype、batch 或延长预算“救回”结果。对无数据祖先的参数，记录数据/总梯度是否为零或缺失，按计算图核对，不照抄旧单输出模型“每个参数必须有非零数据梯度”的假设。

## 8. Trainer 接口设计（尚未实现）

只需在独立 Population experiment 目录提供小型 orchestration；不修改正式 Canonical trainer、数据资产或旧结果。下面是接口合同，不是已存在的新函数。

| 接口 / 对象 | 输入 | 输出与隔离要求 |
|---|---|---|
| `R1FrozenConfig` | 本文常量、9-cell registry、源码与 R0 清单 hash | 验证完全一致；不提供 sweep、自动降 cohort 或改 prior 选项 |
| `load_cell_train_only(cell_id)` | 指定的 R0 recording IDs，原 preprocessing | `CellTrainingBundle`：full-train stimulus / binary events / valid mask / trial & segment provenance；无 validation 字段 |
| `make_inner_dev(bundle)` | 仅 full train | 按既有规则派生 inner fit/dev，边界、顺序和 masks 与旧合同一致 |
| `make_population(cell_id)` | 原 constructor、LegacyPReLU、固定 dtype | 独立实例；不得因 dataset/session ID 改变内部机制；cell ID 仅外部索引 |
| `forward_recorded_port(model,batch,port)` | physical stimulus，严格过去的双 port history 输入 | 全模型 forward 后取 `logit[...,port:port+1]`，shape `[B,150,1]`；target/mask 同 shape |
| `fit_inner_and_select` | 该 cell 模型、inner fit/dev、固定配置 | K、stop step、trajectory；不可访问外层响应 |
| `refit_full_train` | 同初值新模型、full train、K | 唯一 final state；新 optimizer，原 sampling seed；无评价选择 |
| `freeze_all_cells` | 全部 9 个 final artifacts | 含 SHA256 的锁定清单；验证 cohort/recordings 完整后才能开放 evaluator |
| `evaluate_frozen_validation` | 已锁定的 9-cell 清单和匹配 baseline predictions | 只产出第 9 节的固定评价；不写回训练配置或参数 |

`CellTrainingBundle` 的数组合同为 stimulus `[N,150,289]`，events/target/mask `[N,150,1]`，显式 289 个像素物理边界/面积，recording / trial / segment identity。Forward adapter 按当前 `Stimulus` 约定封装；history 两列 `[B,150,2]` 中仅所录 polarity 填真实 events，选择后的 logit/probability `[B,150,1]`。保留既有 singleton observation 维，禁止依赖隐式 broadcasting。观测 loss 在 logit 上计算，不通过人为 clamp probability 改变 objective。

一次 update 的固定顺序：读取一个 train batch → 参数梯度归零 → baseline-reset forward → recorded-port NLL → 加一次原 P → backward → 有限性检查 → Adam step。没有 optimizer step 之外的额外拟合、bias calibration、projection、curriculum 或 path block。

未来实现只做足以验证接口的检查：9-cell/16-recording registry、各 cell 参数所有权与 state reset、source/physical grid、split mask 与计数、recorded port/target shape、严格过去 history、完整无重复参数列表、禁止外层数据访问。现有 R0 结果是接口设计依据，不替代尚未实现 trainer 的验收。本轮没有运行这些新检查或单元测试。

未来执行时的最小证据字段：protocol/source hashes；每 cell 的 recording/trial/segment IDs 与 split-mask hashes；完整 optimizer/constructor/prior 配置；初值 hash；inner K/stop-step 与 dev 曲线；train batch NLL 和 P 分开记录；refit update 数和最终 state hash；requires_grad / optimizer-listed / actually-updated 计数。它们用于证明按合同执行，不新增机制评价指标。

训练与外层评价应为分离入口/数据对象。冻结标记必须在第一笔 outer-validation response/prediction 访问前写出；评价入口拒绝未锁定或缺 cell 的集合。未来新输出应置于独立 R1 run 目录，使用不覆盖写入；这些 artifact 目前**尚未生成**。

## 9. 训练后唯一允许的评价与旧 baseline 固定版本

### 9.1 已固定 lineage

下列每条路径中的 `<cell>` 使用原目录格式，例如 `67_6`。本轮仅确认了 9 × 3 个 checkpoint 文件存在，未打开 checkpoint，也未读取 saved validation predictions 或其中的 NLL。

| 对照 | 固定 artifact 来源 | 使用规则 |
|---|---|---|
| Constant | 同一 full-train scored occupancy 的 cell-specific 均值，clamp 至 `[1e−6,1−1e−6]` | 在外层访问前固定；不从 validation 估计，不覆盖 Population bias |
| 已有 LN | `output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/<cell>/ln-trained.pt` | 复用该目录已有 `validation-predictions.pt`，不重训或重选 lambda |
| 已有 compact causal CNN | `.omo/evidence/compact_causal_cnn_baseline/cells/<cell>/cnn-trained.pt` | 复用已有 `validation-predictions.pt`，不更换模型/候选学习率 |
| 旧 Canonical RetiPath | `output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/<cell>/model-trained.pt` | 复用已有 `validation-predictions.pt`；固定此 lineage，不按 R1 结果换另一旧变体 |
| Population R1 | 未来该 protocol 产生的 9 个 full-train-refit final checkpoints | 全部冻结后只评价一次，无 outer-selection |

这些 LN/CNN/Canonical baseline 已有自己的历史优化预算和选择过程；R1 不重跑它们、不宣称 compute-matched。尤其不复制 LN 的正则搜索或 CNN 的学习率候选，也不引入 NIM、其他 Canonical variants 或容量更大的替代项。

### 9.2 比较合同

冻结后核对每 cell 的 recording IDs、trial 顺序、frame alignment、target occupancy、valid mask、时间 bins 和 shape，与已有预测 artifacts 必须逐项匹配。只限制到本文件固定的 9-cell cohort；不能拿原 22-cell 总均值与 R1 比较。优先从已保存 full-precision logits 依同一 stable Bernoulli 公式重算 NLL，不用四舍五入的旧报告数。

如果已存预测与本协议出现真实不匹配，不择优缩减 bins、改变 cell 或替换 baseline；将相应比较记为 `UNVERIFIED`，报告差异。本轮元数据核对并不声称已经完成这一未来 response-level 匹配。

只报告：

- 每 cell 的 validation Bernoulli NLL（nats / 同一 scored bin），以及所用 recording/bin 数。
- 9-cell cohort **等 cell 权重**的 NLL 均值；所有模型用同一九 cell。可列 MC ON/OFF 的同定义均值，不将 pooled-bin 加权替代 cohort 主汇总。
- Constant、已有 LN、已有 compact causal CNN、旧 Canonical 与 Population 的同表结果；逐 cell 和等 cell 均值的 NLL 差只作描述性比较。

不新增显著性检验、成功阈值、其他训练候选、机制 score、latent recovery、RF、人工刺激或 pathway ablation。真实数据无 synthetic teacher，不报告 synthetic excess CE。模型 history 使用真实 strictly-past validation events 属于 conditional prediction，不能称为无真实 history 的自由 spike 生成预测。

这段 validation 已被旧模型开发和历史报告消费；R1 虽禁止用它选择新 checkpoint，仍只是**复用既有 benchmark validation**，不是全新独立 test，也不是生物机制验证。独立新 real test = **NONE**。不自动开放新的时间段。

## 10. 当前核对结论、限制与停止边界

已通过文本/元数据核对：cohort 来自 R0 的对应 MC 行；MC ON/OFF 数量 5/4；16 个 recording 和每 cell seeds 有旧 lineage；train/dev/validation masks 的定义有现成源代码；R4 budget/selection 可以映射；三个已有 baseline 的九 cell checkpoint 路径存在。

未验证且不在本轮执行：Population 真实训练的收敛性/预测效果、未来 trainer 的运行正确性、所有旧 prediction artifacts 的逐 bin 相等、真实内部状态或 pathway 生理可辨识性。保留 frame-zero、未知绝对光强、截取 FOV 和有限 reset/warmup 的 R0 限制，不通过新标定、搜索或架构调整消除它们。

本轮唯一写入为本文件；不改 `NEXT_TASK.md`、模型、adapter、旧数据/结果，也不生成训练配置工件、checkpoint 或 validation 结果。完成 protocol + trainer design 后停止。

## 附录：冻结源码与 R0 元数据 SHA256

以下为本轮对源码/元数据文件的实际字节哈希；不包含新 payload 或 checkpoint 读取。未来实现与执行应检查漂移，不能静默以新源码替代冻结版本；若确有必要改协议，另行记录版本并获得相应授权。

| 文件 | SHA256 |
|---|---|
| `experiments/retipath_population_v0_1/circuit.py` | `2c269625854b139cad43aace55897de4f9c30454b0c215b3abdac4f127be307d` |
| `experiments/retipath_population_v0_1/real_data.py` | `84d7e0e88e8477b5416d6cc399f28a62d527a0def7e8f9254f96057e3303b2a7` |
| `experiments/retipath_multiscale_v0/circuit.py` | `fe31b115b87ce4d4ab4a69b50f3615e9ef5054959ec434a68a640ed427594f46` |
| `experiments/retipath_multiscale_v0/contracts.py` | `eb7a8bbf04a352fbf0bbfd1520745be14f40abd2539e0023ef739e3cd3c39e85` |
| `data/schottdorf_lee_2021.py` | `ae07cb57443ce95fcef2208060638c1cc12168ec9e1a1f3f9e58c82e24ddf764` |
| `data/schottdorf_lee_multirecording.py` | `3b6b1ae2ddb2c35d1c9ae133017ff5016b79960d56a194f1722b0079ec1716db` |
| `data/schottdorf_lee_spikes.py` | `04993cd009000dab362c352530e5a68d5dfda39403785e25aa6a53dcd9c2b6d4` |
| `data/schottdorf_lee_catalog.py` | `3d1ca53ab11f57f98f452e95af96b51651510e289bf2aef14cff109b9b52a513` |
| `evaluation/mechanistic_retina/factorized_ln_split.py` | `c38761e8471be4118abd54c84258aff28be8a3455e259a9331008b4d8144c327` |
| `training/mechanistic_retina/r4_development.py` | `1bf301b0684a87ad714e51e3ccc886db9ff4b7fe277b2ffacfac60bb69b637ba` |
| `training/mechanistic_retina/center_surround_ln.py` | `d87070f6e87e32d7f7fdbb678525b9fe775a117348d9f92848e0498031719007` |
| `training/mechanistic_retina/optimizer.py` | `06d169ae1ebb6a258d844c7f3038e28ece78a8cdcaa78fcdb6f8c9b3f08aba17` |
| `training/mechanistic_retina/losses.py` | `6e26f7b0ac7d08b681d2b4a291e457d8159e170afc41c6b39aba4166fb9cc91a` |
| `evaluation/mechanistic_retina/spatial_contrast_source.py` | `5aa62ff50c4a3cf1dae55f7439800bd7c8256865f230f7cdd93caca67c9d1cfd` |
| `output/real_data/retipath_population_r0_preflight/dataset_contract.json` | `a36f879ae19583d810a2ee1adc85582808a80f4bf0db200117db95095d621372` |
| `output/real_data/retipath_population_r0_preflight/per_cell_preflight.csv` | `cc9d69f6ba99cc1c558179bcf7e262a54a6e9581fe28c1ecd56db5a6156a00b4` |

Movie SHA256 仅引用 R0 已保存 manifest：`328e229e160eab35028f063468cc2e167045fc8c9fcdbe4a8e7823142a752a67`；本轮未重新读取 movie payload。
