# RetiPath G2 synthetic multilevel/multiscale benchmark

2026-09-18。运行状态：**COMPLETED_FIXED_BUDGET**；工程与数值核验：**VERIFIED**。5 条件 × 3 seeds 的 15 个 fit、固定最终评价和必要核验全部完成。未作研究决策，也未启动后续实验。

本轮用户明确授权 G2 数据生成、训练、checkpoint 与评价，覆盖 G0/G1 当时的阶段限制。G1 架构、loss、参数边界和正式模型保持冻结；G0 设计文件没有改写。用户指定的五条件、曝光匹配、logit 主指标及 scale-history RF 以本轮冻结 protocol 为准。

运行目录：`D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g2_multiscale_20260918/`。

完整合同：[protocol.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g2_multiscale_20260918/protocol.json)。主要结果：[summary.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g2_multiscale_20260918/summary.json)、[per_fit_metrics.csv](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g2_multiscale_20260918/per_fit_metrics.csv)。技术通过不等于机制恢复通过；本轮没有新增成功阈值或生物学结论。

## 1. 冻结合同与数据

同一个 realizable `LocalRetipathV0` teacher；全部条件使用 seeds `2026091801/02/03`，每个 seed 对应完全相同的初始 student 张量。teacher 参数固定为：tau_h=35、a_h=0.20、tau_f=15、tau_s=75、alpha=0.25、tau_a1=40、tau_a2=110、g_e=2、delta_e=0.4、g_i=1.2、delta_i=−0.3、bias=−2。时间常数单位 ms，其他值沿用 G1 effective-unit 定义；这些是 synthetic 设计值。

teacher/student 共用已知 geometry、方程和边界，不从旧实验 checkpoint 初始化。Student 学习 12 个 raw 参数，初始化为 G1 中心加 seed 固定的 raw-space N(0,0.15²) 扰动。训练函数只载入指定 H/BC/R 文件及 student 初值，没有载入 teacher checkpoint、development/test 张量或 RF。公开 protocol 包含 teacher 设计值，但训练更新不使用这些值作为参数或目标。

物理刺激按设计稿生成：2×2 deg 视野、32×32 显示像素；连续 Gaussian 空间场逐像素面积平均，乘以四个频率 0.5/1.5/3/6 Hz 的平均正弦；contrast=0.4，中心坐标各从 {−0.1,0,0.1} deg 取值，相位固定随机。数值采样为 150 Hz、T=300，前 60 bins warmup，后 240 bins 计分；每条独立序列 reset。32×32 是显示网格，回路仍使用 G1 的固定物理节点与面积积分。未以 resize 或 degree/pixel 重标记改变 scale。

| 数据池 | 物理尺度 sigma_deg | 序列数 | 标签用途 |
|---|---|---:|---|
| base multi train | 0.15 / 0.30 / 0.60 | 各 24，共 72 | H 48 条，BC 48 条，R 72 条 |
| paired narrow train | 全部 0.30 | 72 | 与 multi 相同的 H/BC/R 标签槽和数量 |
| extra R train | 0.15 / 0.30 / 0.60 | 各 40，共 120 | 独立新增 RGC 数据，仅 E 使用 |
| development | 0.15 / 0.30 / 0.60 | 各 8，共 24 | 最终描述，按原可见矩阵屏蔽标签，不选模型 |
| test | 0.15 / 0.225 / 0.30 / 0.45 / 0.60 | 各 8，共 40 | 独立 test waveform；其中 16 条用于 primary |

B/C 使用相同的 72 个不同 waveform family、中心、相位及配对随机流。B 将所有 sigma 设为 0.30；C 保留各 24 条的三尺度分配。B 的 72 条 common-scale 波形本身互异，没有将相同刺激的重复计为新增数据；24 条 common anchor 的刺激和观测与 C 逐位一致。E 的 120 个新增 family 与 base family 不重合。

family 在目标生成前分配到 split；同 family 的尺度版本、不同观测层和重复曝光不跨 split。所有条件共用 manifest、teacher 和噪声规则。HC/BC 标签分别是选定 5 个节点的 h 与 o_B（BC 两分支），加一次固定的 iid Gaussian noise，sigma=0.03。RGC events 按 teacher 的严格过去事件条件 Bernoulli 顺序采样；不使用 teacher probability 训练。

| sigma_deg | H: h | BC: o_B | RGC | C/D 的矩阵角色 |
|---:|---|---|---|---|
| 0.15 | 未见组合 | 训练 | 训练 | 小尺度 |
| 0.225 | 保留 | 保留 | 保留 | 全层未见尺度 |
| 0.30 | 训练 | 训练 | 训练 | common anchor |
| 0.45 | 保留 | 保留 | 保留 | 全层未见尺度 |
| 0.60 | 训练 | 未见组合 | 训练 | 大尺度 |

B 的所有训练槽在 0.30；A/E 仅使用 RGC 标签。因此“未见 layer×scale 组合”表中 H@0.15、BC@0.60 的角色是相对 C/D 的可见矩阵定义，不能把所有条件的可见性都称为相同。

## 2. 五条件与预算账本

所有 fit 使用 CPU float32、单进程单计算线程、确定性算法；评价归约 float64。三 seed 进程并行，每个进程依次完成 A–E。运行环境为 Python 3.12.7、PyTorch 2.6.0+cpu。

统一 Adam：lr=0.01，betas=(0.9,0.999)，eps=1e−8，weight_decay=0；总梯度范数裁剪 1.0。每 fit **180 次 optimizer.step、480 个 microbatch、1,920 次序列曝光**；每 microbatch 为同一数据层/同一物理尺度的 4 条完整序列。无 early stopping、checkpoint 选择、调参或追加训练；只用 step 180。

H/BC loss 为 G1 Gaussian NLL（省去固定常数），R 为 Bernoulli NLL；先每条序列有效 scalar 平均，再 batch 等权平均。累计系数保持 H=0.4、BC=0.4、R=1/3；同一步多个 microbatch 的梯度相加，不除以数量。

| 条件 | 名称 | 训练内容 | H / BC / R batch | 有效 scalar target 曝光 | 实测每 fit 平均秒 |
|---|---|---|---:|---:|---:|
| A | RGC_ONLY | base R，补充槽重复 base R | 0 / 0 / 480 | 460,800 | 31.80 |
| B | MULTILEVEL_NARROW | 三层，全为 common scale | 150 / 150 / 180 | 2,332,800 | 23.85 |
| C | MULTILEVEL_MULTISCALE_JOINT | 三层，冻结多尺度集合 | 150 / 150 / 180 | 2,332,800 | 23.93 |
| D | MULTILEVEL_MULTISCALE_PROGRESSIVE | 与 C 相同有序观测流，逐层+rehearsal | 150 / 150 / 180 | 2,332,800 | 23.58 |
| E | EXTRA_RGC_MULTISCALE | 180 base R batch + 300 extra R batch | 0 / 0 / 480 | 460,800 | 31.13 |

B/C 的样本量、标签量、microbatch 索引与曝光量一致；尺度分配不同。C/D 的逐层有序流完全相同，只改变消费时刻和冻结日程。A/E 的 base R 流与 C/D 相同，其余 300 槽分别重复 base 或读取 independent extra。A/E 每尺度各 640 次序列曝光；C/D 中 H、BC 的两个可见尺度各 300 次，R 三尺度各 240 次；B 所有曝光集中于 0.30。

B/C 的 6-step 周期：H 在位置 1–5、BC 在 2–6、R 每步。A/E 使用同样槽数并把 H/BC 槽替换为 R。

| D 阶段 | steps | 每步 microbatch | active 参数 |
|---|---|---|---|
| HC 预训练 | 1–30 | H×1 | tau_h |
| BC 加入 | 31–60 | BC×1 | a_h、tau_f、tau_s、alpha |
| rehearsal + joint | 61–180 | H×1、BC×1，阶段奇数步 R×1、偶数步 R×2 | 全部 12 个 |

D 始终使用同一 Adam 实例；冻结参数 grad=None，解冻保留已有 Adam 状态。实际 Adam state 的 step 计数为：D 的五个 H/BC 祖先参数各 150，其余七个各 120；其他条件全部参数各 180。所有 fit 的 12 个 raw 参数最终均发生变化。

**匹配限制：E 是 exposure-matched control，不是 information-matched control。** 相同序列曝光不等于相同 scalar label 数、信息量、梯度强度或计算量。B/C/D 各层累计 loss 系数为 60/60/60；A/E 只有 R，总系数为 160。保留原 loss 系数，没有为获得公平表象事后重加权。H/BC backward 的祖先子图较短，因此 full-forward 数相同也不表示 FLOPs 相同；墙钟时间受三个并行进程影响。

## 3. Primary：保留尺度的 direct-BC logit intervention error

`BLOCK_DIRECT_BC_DRIVE` 保留既有语义：d_E=uE=0，tonic gE=1；BC state/output、AC-associated branch、gI 与 conditioning event history 不被删除。整条序列从原始初态重算下游状态。

定义 `delta_ell = ell_block − ell_normal`。每条 test 序列最后 240 bins 上，计算 student 与 teacher 的 delta_ell 差的 RMSE；先尺度内等权平均，再对 0.225/0.45 两个 held-out scales 等权平均。teacher/student、normal/block 使用同一条 teacher 正常试次的过去 events。这是给定共同历史的条件计算干预，不是自由运行 spike 总效应或药理干预。

| 条件 | seed 1801 | seed 1802 | seed 1803 | 均值 | 相对误差均值 |
|---|---:|---:|---:|---:|---:|
| A | 0.053342 | 0.053583 | 0.052958 | 0.053294 | 0.515685 |
| B | 0.046994 | 0.058513 | 0.051000 | 0.052169 | 0.500022 |
| C | 0.036336 | 0.047920 | 0.039974 | 0.041410 | 0.397577 |
| D | 0.023936 | 0.044477 | 0.032075 | 0.033496 | 0.322424 |
| E | 0.032714 | 0.030535 | 0.030041 | 0.031097 | 0.302861 |

seed 列省略共同前缀 `202609`。相对量先逐序列除以相应 teacher effect RMS，再按相同规则平均；不是两列均值相除。teacher effect RMS 的 held-out 均值为 0.104428。

| 冻结配对比较（左减右） | primary 差值均值 | 三 seed 差值 | 差值 < 0 的 seed 数 | excess CE 差值均值 |
|---|---:|---|---:|---:|
| B−A | −0.001126 | −0.006348 / +0.004930 / −0.001959 | 2/3 | +2.21656e−5 |
| C−B | −0.010759 | −0.010658 / −0.010592 / −0.011025 | 3/3 | −6.45000e−5 |
| D−C | −0.007914 | −0.012400 / −0.003444 / −0.007899 | 3/3 | −4.02866e−5 |
| E−A | −0.022198 | −0.020628 / −0.023047 / −0.022918 | 3/3 | −1.37477e−4 |
| C−E | +0.010313 | +0.003622 / +0.017385 / +0.009933 | 0/3 | +9.51427e−5 |

这是一个 teacher、一个数据 draw、三个初始化的数值记录，没有据此宣布普遍优越性、显著性或收敛。E 的 primary 均值最低，但不是每个配对 seed 都优于 D。完整五尺度结果及每条序列记录保存在 CSV，未丢弃低效应序列。

## 4. 响应与状态恢复

以下均为 held-out 16 条序列、三 seeds 的等权均值。RGC NLL/excess CE 单位 nats/bin；excess CE 使用 evaluator-only teacher 条件概率，减去其 Bernoulli entropy。h、s_B、o_B 是 clean teacher 的全部节点读数；s_B/o_B 的两个分支等权。训练只见 5 个选定节点，未以全节点 clean latent 训练。

| 条件 | RGC NLL | excess CE | H1 h RMSE | BC s_B RMSE | BC o_B RMSE | local d_E RMSE |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.3503146 | 1.96295e−4 | 0.0224095 | 0.0129220 | 0.0124115 | 0.0025717 |
| B | 0.3503080 | 2.18460e−4 | 0.0002270 | 0.0017191 | 0.0023774 | 0.0025402 |
| C | 0.3502369 | 1.53960e−4 | 0.0002421 | 0.0020147 | 0.0025743 | 0.0020169 |
| D | 0.3501907 | 1.13674e−4 | 0.0003257 | 0.0022685 | 0.0028300 | 0.0016332 |
| E | 0.3501211 | 5.88177e−5 | 0.0213148 | 0.0118285 | 0.0114573 | 0.0015170 |

local d_E 是各 BC 节点的传输贡献；block 后为零，所以 normal d_E 恢复误差也等于该局部 block effect 的误差，不能替代端到端 logit primary。H1 h 恢复同样不等于独立 feedback coupling a_h 已恢复。

未见 layer×scale 组合在独立 test waveform 上评价，三 seed 均值如下：

| 条件 | H h @0.15 RMSE | BC s_B @0.60 RMSE | BC o_B @0.60 RMSE |
|---|---:|---:|---:|
| A | 0.0105144 | 0.0174359 | 0.0167967 |
| B | 0.0001053 | 0.0022987 | 0.0031032 |
| C | 0.0001123 | 0.0026975 | 0.0033606 |
| D | 0.0001511 | 0.0030366 | 0.0036923 |
| E | 0.0099951 | 0.0159260 | 0.0154316 |

Observed-node 与 unobserved-node 的分列结果在 `per_sequence_metrics.csv` / `per_fit_metrics.csv`；development 仅记录原矩阵允许的 noisy observation loss 与 RGC NLL，见 `development_metrics.csv`，没有用于选择结果。

## 5. Cross-seed ambiguity

定义为同一条件三对 seed 在共同 held-out 输入上的两两 RMS 距离，先序列平均再 seed-pair 平均。它描述有限运行的预测分散，不是 posterior uncertainty，也不证明参数可辨识。

| 条件 | direct-block logit effect | RGC probability | H1 h | BC s_B | BC o_B |
|---|---:|---:|---:|---:|---:|
| A | 0.0125262 | 0.0007978 | 0.0024987 | 0.0025546 | 0.0046853 |
| B | 0.0093031 | 0.0007232 | 0.0001455 | 0.0005628 | 0.0009008 |
| C | 0.0095847 | 0.0007507 | 0.0001809 | 0.0005559 | 0.0009136 |
| D | 0.0156711 | 0.0013874 | 0.0002758 | 0.0006385 | 0.0009986 |
| E | 0.0122244 | 0.0007617 | 0.0024262 | 0.0028064 | 0.0051246 |

D 的 primary error 均值低于 C，同时其 logit-effect 跨 seed 距离更大。两种指标分别保留；不把 seed 一致性替代真值恢复误差。完整 600 行 pair×stratum×metric 见 `ambiguity.csv`。

## 6. Secondary：同一 common probe 的 effective RF

在目标生成前冻结 RF 定义。使用首条 common-scale test family 的同一中心/相位，前 60 bins 分别给出五种 sigma 的 scale history，之后背景归零。所有 history 使用同一 probe bank：在 25 个固定 BC 节点位置，各放一个 0.10 deg 方形 aperture，于 bin 60 独立施加 ±0.01 扰动；逐像素面积平均。conditioning events 全为零。

保存 `R[position,lag]=(ell_plus−ell_minus)/0.02`，位置 25 个、lag 1–30 bins，即输出 bins 61–90。各模型/历史都使用相同 probe、窗口与初态；RF 从未进入训练 loss。它是该有限扰动和事件条件下的 effective sensitivity，不是生理 RF 标定。

- Gain：`sqrt(mean(R²))`。
- Spatial profile：先对 lag 求平均，再按空间向量 L2 norm 归一化。
- Temporal profile：先对位置求平均，再按时间向量 L2 norm 归一化。
- profile 是预定 signed projection，允许正负抵消；norm <1e−12 才标为未定义。本次全部 profile 有定义，没有事后替换投影。

各 history 的 gain；student 列为三 seed 均值：

| history sigma_deg | teacher | A | B | C | D | E |
|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 0.00198606 | 0.00331673 | 0.00344783 | 0.00316732 | 0.00295018 | 0.00276212 |
| 0.225 | 0.00204101 | 0.00339764 | 0.00353426 | 0.00324845 | 0.00302303 | 0.00279466 |
| 0.30 | 0.00205348 | 0.00338780 | 0.00357973 | 0.00328878 | 0.00305683 | 0.00280223 |
| 0.45 | 0.00198737 | 0.00335518 | 0.00347954 | 0.00320938 | 0.00298670 | 0.00279509 |
| 0.60 | 0.00198685 | 0.00334873 | 0.00347565 | 0.00320362 | 0.00298246 | 0.00278558 |

以下误差均为五 history × 三 seeds 等权平均；normalized profile 误差为 L2 distance：

| 条件 | gain 绝对误差 | gain 相对误差 | spatial profile L2 | temporal profile L2 | 完整 RF RMSE |
|---|---:|---:|---:|---:|---:|
| A | 0.00135026 | 0.671635 | 0.109757 | 0.309195 | 0.00156532 |
| B | 0.00149245 | 0.742211 | 0.114055 | 0.187785 | 0.00155473 |
| C | 0.00121256 | 0.603047 | 0.101825 | 0.172330 | 0.00127273 |
| D | 0.00098889 | 0.491829 | 0.090586 | 0.156851 | 0.00104730 |
| E | 0.00077698 | 0.386614 | 0.065021 | 0.246629 | 0.00097242 |

将每个 history 的 normalized profile 减去该模型 common-history profile，再比较 student/teacher 的这种变化；下面是坐标 RMS 误差的五 history × 三 seed 均值（包括 common 条目的零值）：

| 条件 | spatial change RMS error | temporal change RMS error |
|---|---:|---:|
| A | 0.0095190 | 0.0039466 |
| B | 0.0041650 | 0.0016554 |
| C | 0.0047362 | 0.0018810 |
| D | 0.0048566 | 0.0019601 |
| E | 0.0087091 | 0.0036337 |

Gain、空间 profile、时间 profile 的排序不同，分别报告。逐 history 的 gain change、profile cosine 和 change error 见 [rf_metrics.csv](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g2_multiscale_20260918/rf_metrics.csv)；全部归一化向量见 [rf_profiles.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g2_multiscale_20260918/rf_profiles.json)，完整 signed RF 矩阵见 `evaluation/*_rf.pt`。

## 7. 核验与一次基础设施修复

协议/source/split/schedule 在 `2026-09-18 02:34:55 UTC` 冻结，数据完成于 `02:35:09 UTC`。所有最终 checkpoint 锁定后，于 `02:39:05 UTC` 写入 `TEST_CONSUMED.json`，随后才进行 student test 评价。后续 replay 明确为已消费 test 的数值核验，不是新的 untouched test。

训练启动时出现一次基础设施错误：Windows 生成的文件哈希键使用反斜杠，运行器查询使用正斜杠。三个 A 进程均在载入训练张量、创建 optimizer 及参数更新前抛出 KeyError。保留了三个空失败目录和原始运行器源码；只修复哈希键读取并记录 source 修订后，使用同一数据、初值、seed、schedule 开始训练。没有重新生成目标、修改研究合同或从结果选择重跑。

修复前/后运行器 SHA256 分别为 `3309286889af88f68c8f142d7872e732afd921953af248abed5ae756404d70b2` / `3993fa9f0bfe53bb3c295ce7e8ac5b818f4ed0daa063defefb1d9b37ab2bb304`。记录见 [RUNTIME_CORRECTIONS.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g2_multiscale_20260918/RUNTIME_CORRECTIONS.json)。原始 `SOURCE_LOCK.json` 未覆盖；verification 对 G1/design 原始指纹和运行器修订指纹分别核对。

| 核验 | 实际证据 |
|---|---|
| 冻结 G1 | 原有 5 组 unittest 全部通过，包含 8 项 G1 要求；G1 三文件、依赖和设计指纹不变 |
| B/C 及 split | 72 个配对 family；common anchor 逐位相同；E extra 独立；无 held-out scale 训练曝光 |
| 因果 Bernoulli sampler | 与未修改 circuit 的条件概率最大差 3.7253e−8；逐个事件决策完全一致 |
| 预算与初值 | 15/15 final checkpoint；2,700 updates、7,200 microbatches、28,800 次序列曝光；配对初值哈希一致 |
| 参数/optimizer | 所有参数有限、12 参数均实际改变；Adam 参数组和各阶段 state-step 计数符合合同 |
| Teacher trace | 全 40 条 test 的正常/干预及保存读数逐位重放一致 |
| Student checkpoint | 每个最终 checkpoint 在 0.225 的 4 条 held-out 序列上重放 normal/block logit/probability，最大误差 0 |
| CSV/JSON | 从保存 trace 重算 test per-fit 与配对/summary；一致 |
| 独立公式复算 | 另用 NumPy 从全部 600 条 sequence trace 重算 primary、NLL/excess CE、h/s_B/o_B；最大差 1.6654e−16 |
| RF | 75 个 student×history 记录从原始矩阵复算；独立 gain/RMSE 复算最大差 4.3369e−19；teacher common-history RF 逐位重放一致 |
| 数据与来源 | 全部生成数据文件、manifest、schedule 的指纹未变；训练载入清单中没有 evaluator/teacher 文件 |

核验文件：[verification.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g2_multiscale_20260918/verification.json)、[INDEPENDENT_METRICS_VERIFICATION.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g2_multiscale_20260918/INDEPENDENT_METRICS_VERIFICATION.json)、[DATA_CONTRACT_VERIFICATION.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g2_multiscale_20260918/DATA_CONTRACT_VERIFICATION.json)、[OPTIMIZER_VERIFICATION.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g2_multiscale_20260918/OPTIMIZER_VERIFICATION.json)。

## 8. 交付与停止边界

新增实现为 [synthetic_study.py](D:/PythonProject/retina_rf_SNN/experiments/retipath_multiscale_v0/synthetic_study.py) 和 [protocol.json](D:/PythonProject/retina_rf_SNN/experiments/retipath_multiscale_v0/protocol.json)。它们生成并消费本轮隔离工件；没有修改 G1、正式模型或旧实验。

运行目录内包括：

- `protocol.json`、`STIMULUS_MANIFEST.json`、`SCHEDULES.json`：完整运行合同、物理序列来源和逐 step 曝光日程。
- `train_data/`、`evaluator_only/`、`initial_students/`：可见训练标签、隔离 teacher/evaluation 真值与配对初值；不要把 evaluator-only 信息接入训练。
- `fits/{A..E}_{seed}/final.pt`：15 份最终 model/optimizer checkpoint，伴随 180 行 `trajectory.csv` 和 `completed.json`。
- `evaluation/`：15 个模型的 normal/block test 与 development trace、teacher 及 student RF 原始矩阵。
- `per_sequence_metrics.csv`（600 行）、`per_fit_metrics.csv`（120 行）、`paired_comparisons.csv`（75 行）、`ambiguity.csv`（600 行）、`development_metrics.csv`（360 行）、`rf_metrics.csv`（75 行），以及 `summary.json`、`rf_profiles.json`。
- source/数据/checkpoint/test-consumption 锁、上述核验记录、原始及修复后的源码快照、[FILE_MANIFEST.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g2_multiscale_20260918/FILE_MANIFEST.json)。该清单列出自身之外的 136 个工件文件，含大小和 SHA256。

本报告及 `docs/NEXT_TASK.md` 已更新。未执行 Git、切分支、真实数据训练、S0/S0.5 重跑或 S0.6；没有改 loss/architecture、新增 condition、延长训练、结果后修改 scale split 或继续下一阶段。180 updates 不代表收敛，单 teacher/有限初始化也不提供真实生理结论。保留集已消费；本轮到此停止。
