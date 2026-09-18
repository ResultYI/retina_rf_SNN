# RetiPath H1→BC fixed-basis capacity diagnosis

本轮为已消费时间区间上的表示能力诊断，不是独立确认，不修改正式 RetiPath 默认入口和历史 artifact。无 type-sharing 梯度，无新 cell type、隐藏层、空间 support、RF loss、Mach/SBC 或新时间 block。

## 科学条件与固定表示

A 是当前正式 `models.mechanistic_retina.retipath.RetiPath`，两个有效空间 mode 的 conductance 架构，37 个可学习参数/cell。由实际 final-model registry 解析 Phase 2 三个 seeds 的 66 个 selected/fresh-refit checkpoint；复用其已完成且完全匹配的 selection/fresh-refit，不重训 A。旧文件名中的 C 只属于历史实验标签。

B/C 仅在 H1.modulated_cones 与现有 BC feature bank 之间接入逐位置 M=4 bank。输入 [batch,time,289]，bank 输出 [batch,time,289,4]；不改变位置、center、support、polarity、H1 surround。原 BC feature bank 对每个 channel 使用完全相同的空间/时间滤波器，输出 [batch,time,N=1,path=4,K=2,time_basis=3,M=4]。对 M 轴使用 softmax 组合。每 cell 有 sustained/transient 各 4 个 mixing logits，初始全零、权重各 1/4；对应 direct/broad 共用该组 mixing。组合后的特征继续原 BC 权重与整条分支非对称变换、AC、spatial E/I conductance、RGC adaptation/bias/strictly-past history。没有到 logit 的 shortcut。

所有 cells/seeds 固定同一斜率 a=[0.5,1,2,4]。C 固定 b=[1,0,-1,-4]，对应归一化输入上的 softplus 转折位置 −b/a=[−2,0,0.5,1]；这些是本轮预先选择的有效突触 basis，不是生理 bipolar subtype，也不是文献拟合常数。C: h_m(u_hat)=softplus(a_m*u_hat+b_m)−softplus(b_m)；B: h_m(u_hat)=a_m*u_hat。h(0)=0，导数非负。a/b 是 buffer，不进入 optimizer；没有 dummy 参数。B/C 均为 45 个可学习参数/cell（37 原参数 + 8 mixing logits），其中 softmax 每组有一个平移不变自由度；66 fits 的参数不跨 cell 或 seed 共享更新。B 的 4 个线性 channels 经凸组合会折叠为正值线性缩放，因此它控制这种参数化/训练方式，不能称作新增空间滤波容量。

u_hat=u/RMS_H1。RMS_H1 是初始 H1 输出在各可用输入 bin 和全部 289 位置上的单一 RMS，包括 warmup。selection 仅用 inner-fit [0,12.4) 输入，排除 guard/inner validation；fresh refit 从同 seed 全新初始化，用完整 [0,16) 重算一次后冻结。B/C 同 cell/阶段/seed 共用 RMS_H1。H1 本身仍学习，但 RMS 不跟随更新。初始 H1 不含随机参数，三个 seeds 的 RMS_H1 应相同。数据、输入和 RMS 均不使用 [16,20) 或之后。

原 conductance 的固定输入 E/I RMS 规则不变：对每个条件的初始表示、相应阶段的训练输入计算 [N=1,K=2] RMS，随后冻结。B/C 的表示不同，故 E/I RMS 可不同；它们不是额外可学习参数。保留 alpha bounds [0.05,2]、反转电位 0/1/−1/3、gL=1、V0=2/9、Cm=3*tau_ref 和原 dt=1000/150 ms。电压是有效归一化尺度，无真实电压/电导恢复声明。

## 训练和选择

22 macaque cells，仍逐 cell、逐 seed、逐条件独立优化。固定 seeds=2026091301,2026091302,2026091303。只用 [0,16) 内既有80/20 split：inner fit [0,12.4)，guard [12.4,12.8)，inner validation [12.8,16)。原150-bin独立sequence、30-bin warmup、120-bin评分，150 Hz Bernoulli occupancy、17×17 L+M Weber、source IDs、history 和 mask 不变。guard/inner validation 输入处理原样调用 Phase 2。

Adam lr=.003，batch4，weight_decay=0，betas=(.9,.999)，eps=1e−8，clip_norm=5，无 scheduler。最多3000 updates，step0和每25步 inner validation，patience400 updates；精确最小 validation NLL 选步，并列取更早；改善超过1e−7才重置 patience，但精确 best 不受此阈值影响。同 cell/seed/阶段 A/B/C 使用 Phase 2 的完全相同 minibatch schedule 前缀。不同条件可有不同 selected step，不依据外部评价增步、重启或追加 seed。

所有 B/C selection 完成并锁定后，丢弃选择阶段权重和 optimizer，按原 seed 从头 fresh refit 完整[0,16)到各自 selected step。全部132个新 refit 冻结后才读外部评价 targets。恢复运行只允许从已保存的同一 optimizer/schedule/step 精确恢复；工程错误修复不能形成新增模型选择机会。记录每25步 sampled train NLL、inner-val NLL及固定诊断 batch 的 train NLL；full train NLL 在阶段起止计算。阶段达到3000只记预算上限，不宣称充分收敛。

## 预测

只评价已消费[16,20)、[20,60)，标 descriptive。A 优先复用实际 checkpoint 对应的冻结 logits，并核对 targets/mask/input/source IDs/hash 及 float64 Bernoulli NLL。B/C 冻结重放。禁止读[60,120)或之后的 targets。

Primary C−B；secondary B−A、C−A，负值有利于前者。保存 per-cell absolute NLL、差值、每 seed 和 seed aggregate。先在 cell 内对三个 seed 的 loss 等权平均，再统计 equal-cell mean/median/wins/losses/ties 与 paired-cell percentile bootstrap95%CI（10000次，固定 seed2026091399）。不平均概率或权重，不将 cell×seed 作为独立样本，不合并时间段。wins/losses 按差值严格正负，精确0为 tie；CI跨0只写未分辨。按原四种 cell type 汇总同样的配对统计，属于描述性分组结果。

## Basis 与 RF secondary

训练前固定诊断规则：在每 cell 原训练数据前4条 sequence、其有效 inner-fit 输入上，每25步保存各 basis 的 RMS、near-zero(|h|<0.01)比例、channel Pearson correlation 及 mixing 权重；这是固定计算子集，不用于选择。阶段起止及 final refit 全训练输入保存激活 mean/std/RMS、1/10/50/90/99%分位数、near-zero 比例、全部 pair correlation、gain 去除后的归一化函数差与最终 mixing 权重。RMS<0.01 或某组 mixing<0.01 仅作为预先固定的诊断标记，correlation>0.995 表示近共线提示；原始数值同时保留。长期标记要求整个 refit 后半段所有预定诊断点均符合，不依据结果追加 channel/改变 bank。B 线性 channels 预期相关系数1，不能把这一设计性质误判为工程失败。

RF 只用[16,20)既有 train-derived HIGH/LOW：原45 strictly-past 帧定义、q20/q80、冻结观测索引，每 context 至多20个，以原 floor(linspace(0,n−1,min(20,n)))确定。无观测时标 UNVERIFIED，不搜索替代 context。复用 A 的合法 Phase2 RF 缓存；B/C 使用 float64 的 logit 对完整可用因果 prefix 的 Jacobian，observed history 固定，新状态对输入不 detach。P(x)=sum_lag J²/sum_lag,x J²；逐观测归一化后 context 内等权平均。gain=||J||2（context median）；报告 centroid、围绕自身 centroid 的 second-moment radius、HIGH/LOW TV、逐观测以自身 centroid 计算后 context 平均的 radial CDF 差，原0..1.5degree步长0.005网格。P/cdf完整数组保存为 CSV JSON 字段。gain、profile、size方向分开报告，无新RF metric/effect threshold，不要求变化或统一收缩，不用 RF 选模。

## 来源和交付

架构、数据、训练合同和 RF 指标来源于本地冻结 final-model/Phase2 manifest 及真实代码；本轮 pointwise固定 basis、a/b数值和mixing设计是用户指定方向下的实验设计，不声称来源论文已验证这一机制。仅新增 work/retipath_h1_bc_capacity 实验代码，正式模型不修改。输出 PROTOCOL.md、correctness.json、training.csv、prediction.csv、basis_diagnostics.csv、rf_secondary.csv、REPORT.md、checkpoints/；日志、原始数组、锁和恢复状态归入 checkpoints/。最终只回答六个指定问题，然后停止。

## 实际冻结记录
