# Population RetiPath v0.1 Stage B：heterogeneous population / partial multilevel observations

状态：**已完成固定 45 fits 与统一评价；停止于 Stage B**。协议冻结于2026-09-18，暂停后的续跑/交付于2026-09-20。

Primary 的 C−A 均值差为 **-0.0221782**，C−E 为 **-0.0223265**。负差值的 teacher 数分别为 **5/5**、**5/5**，配对 seed 数分别为 **15/15**、**15/15**。这是同一 synthetic family 下、固定有限训练预算的描述性结果，不是生物 population 的显著性或因果结论。

本次结果的事实读数：

- **内部state与BC output**：observed/unobserved H1 h_H、BC s_B及delta_r_B的C−A、C−E均在5/5 teacher和15/15 paired seeds中为负。未观测节点的改善不代表各单元参数均已唯一恢复。
- **Direct通路**：primary delta-logit RMSE的A/C/E描述均值为0.0362725 / 0.0140943 / 0.0364208；d_E误差也在全部teacher/paired seeds中降低。AC/inhibitory d_I的均值误差略降，但两个比较都只有2/5 teacher、6/15 paired seeds方向为负，改善不一致。
- **Coupling参数**：a_H RMSE为A=0.0993767、C=0.0350012、E=0.173786，两个比较均为5/5 teacher、15/15 paired seeds降低。gamma_BR和gamma_AR的三条件误差则分别接近0.2967与0.11135，差值仅约10^-6量级且方向不完全一致；本次coupling参数恢复结论应按字段分别保留。固定gamma_BA=1的零误差不属于估计恢复。
- **RGC prediction的代价**：excess CE为A=0.00102885、C=0.00127396、E=0.000982087 nats/RGC-bin；C−A=+0.000245107、C−E=+0.000291870，均在5/5 teacher、15/15 paired seeds中升高。没有用新的“明显受损”阈值将这一代价改写为无影响。
- **Cross-seed ambiguity**：未观测H1/BC state、BC output及direct干预效应的seed-pair距离在两个比较的5/5 teacher中降低。其他量仍逐项列出；较低分散度与较低truth error分开解释。

## 1. 冻结合同与执行边界

完整合同见 [PROTOCOL.md](../output/synthetic/retipath_population_stage_b_20260918/PROTOCOL.md) / [protocol.json](../output/synthetic/retipath_population_stage_b_20260918/protocol.json)，冻结来源见 [SOURCE_LOCK.json](../output/synthetic/retipath_population_stage_b_20260918/SOURCE_LOCK.json)。

- 5 independent teacher instances × 3 paired initializations × A/C/E；每 fit **400 updates**、每步 3 microbatches、每 batch 4 sequences，仅 step 400 final。最终保留轨迹合计 **18,000 optimizer updates / 54,000 microbatches**；中断丢失工作另记。
- **A=RGC_ONLY**：3 个 base-RGC microbatches；**C=MULTILEVEL_PARTIAL**：1 base-RGC + 1 H1 + 1 BC；**E=EXTRA_RGC_CONTROL**：1 base-RGC + 2 独立 extra-RGC microbatches。全程 joint。
- 保留 Population v0.1 的 359 个可训练 scalars、原 partial pooling 和 hierarchy penalty。BC 固定 LegacyPReLU；Q、AC routing、RGC conductance 和 intervention 定义未改。γBA 固定 1，不是待估参数。
- H1 5/25、BC ON 5 + OFF 5：同一物理位置 (0,0)、(±0.30,0)、(0,±0.30) deg，生成前锁定；H1 监督 h_H，BC 同时监督 s_B / delta_r_B，不监督 v_B / w_B；AC 无直接观测。其余 20 H1 / 40 BC nodes 全部 evaluator-only。
- Teacher：原有界 raw family center 高斯扰动；有 center prior 的字段 SD=0.3，其余0.25；H1 dynamics contrast SD=0.3、a_H SD=0.1、BC SD=0.6；AC dynamics 仍 family-shared。不筛选响应或重抽 teacher。初值独立于 teacher，A/C/E 按 world/seed 使用相同初始文件。
- 每 world 72 base、两个各72条的独立 extra-RGC 池；训练尺度0.15/0.30/0.60deg；16个保留序列位于0.225/0.45deg，每尺度8条。32×32 numerical grid、完整2×2deg物理场、300 bins@150Hz；60 warmup + 240 scored bins。节点身份固定，物理尺度与 numerical grid 分离。
- Adam lr0.01、betas(0.9,0.999)、eps1e-8、weight_decay0、合计 clip norm1。RGC mean BCE 权重1/3；H1/BC mean Gaussian loss (sigma0.03) 各0.4；BC 的两个 port 在同一loss中平均；原 hierarchy penalty 系数1，每 update 加一次。数值事前固定，不宣称已为Population调优。

| 条件 | updates | microbatches | sequence exposures | scored scalar targets |
| --- | --- | --- | --- | --- |
| A | 400 | 1200 | 4800 | 2304000 |
| C | 400 | 1200 | 4800 | 10368000 |
| E | 400 | 1200 | 4800 | 2304000 |

**Exposure-matched，不是 information-matched**。三组每 fit 各1,152,000 scored sequence-time bin exposures；C观察scalar更多，A的RGC exposure为C的3倍。C data loss系数和为1.133333…，A/E为1；总梯度clip会改变各项最终作用比例，不声称算力、观测信息或有效base-RGC gradient相等。

## 2. 锁定顺序与验证

[GENERATION_COMPLETED.json](../output/synthetic/retipath_population_stage_b_20260918/GENERATION_COMPLETED.json) 先于 [TRAINING_STARTED.json](../output/synthetic/retipath_population_stage_b_20260918/TRAINING_STARTED.json)；所有45个final checkpoints先写入 [CHECKPOINT_LOCK.json](../output/synthetic/retipath_population_stage_b_20260918/CHECKPOINT_LOCK.json)，随后写 [TEST_CONSUMED.json](../output/synthetic/retipath_population_stage_b_20260918/TEST_CONSUMED.json) 才开始student统一评分。生成器为产生真值会持有teacher，但不做student评价或teacher筛选。训练worker有程序内读取guard，拒绝evaluator_only；只加载许可的可见节点target、RGC事件和初始state。

验证记录：[PREFLIGHT.json](../output/synthetic/retipath_population_stage_b_20260918/PREFLIGHT.json)、[VERIFICATION.json](../output/synthetic/retipath_population_stage_b_20260918/VERIFICATION.json)、[verified_exposures.csv](../output/synthetic/retipath_population_stage_b_20260918/verified_exposures.csv)。逐sequence原数组复算12240项、逐seed-pair复算11520项，独立数值公式最大差8.23994e-17。每world/condition首个seed共15个final students各重放4条已消费test，保存trace逐位一致；5个teacher同样重放核对。重放仅验证数值，不用于选checkpoint或追加训练。

Checkpoint内Adam计数、45个配对初始hash、可见target形状、全部冻结数据/源码hash、family zero-sum deviations、AC family-sharing与固定gamma_BA均通过核对。未修改G1/G2/G3/旧结果或Population circuit，没有early stopping、预算延长或condition追加。中断恢复例外见下文，不能把发生过的丢失训练工作记作零。

用户暂停后，原进程在续跑核对时已不存在；W01的A/C/E seed4103三条内存轨迹不可恢复。经用户明确批准的中断恢复例外，保留原日志与attempt目录，从原初值按原数据/日程各重新执行400步。已有6个final不重跑，其余36个未启动fit首次执行。详见 [INTERRUPTION_RESTART_APPROVED.json](../output/synthetic/retipath_population_stage_b_20260918/INTERRUPTION_RESTART_APPROVED.json) 和 `resume_*_20260920/`。最终45条轨迹仍各400步、合计18,000步；此前丢失工作日志至少确认400步，确切总量不可恢复，不能声称总实际计算只有18,000步。没有根据test或结果选择性重跑。

## 3. Primary：held-out direct-BC intervention recovery

`delta_logit = logit_BLOCK_DIRECT_BC_DRIVE − logit_NORMAL`；teacher/student以及normal/block共用同一teacher-normal event history、baseline初值。每sequence在240 bins×2 RGC上求误差RMSE，再尺度内均值、两个尺度等权均值。block归零direct驱动，保留tonic gE=1，重算下游；是conditional intervention，不等价药理阻断。所有表的负C−A/C−E表示该误差降低，无新增成功阈值。

| Teacher | A | C | E | C−A | 负差seed | C−E | 负差seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W01 | 0.0369016 | 0.0127493 | 0.0398607 | -0.0241523 | 3/3 | -0.0271113 | 3/3 |
| W02 | 0.039261 | 0.0126973 | 0.0474366 | -0.0265637 | 3/3 | -0.0347393 | 3/3 |
| W03 | 0.0358743 | 0.00903879 | 0.038349 | -0.0268356 | 3/3 | -0.0293102 | 3/3 |
| W04 | 0.0392375 | 0.0272542 | 0.0358895 | -0.0119833 | 3/3 | -0.00863531 | 3/3 |
| W05 | 0.0300879 | 0.00873179 | 0.0205681 | -0.0213562 | 3/3 | -0.0118363 | 3/3 |


完整primary paired seeds：

| Teacher | Seed | A | C | E | C−A | C−E |
| --- | --- | --- | --- | --- | --- | --- |
| W01 | 4101 | 0.0368151 | 0.0127433 | 0.0396943 | -0.0240718 | -0.026951 |
| W01 | 4102 | 0.0367353 | 0.0127509 | 0.039385 | -0.0239844 | -0.0266341 |
| W01 | 4103 | 0.0371544 | 0.0127538 | 0.0405027 | -0.0244006 | -0.0277488 |
| W02 | 4101 | 0.0393935 | 0.0126914 | 0.0475973 | -0.0267021 | -0.0349059 |
| W02 | 4102 | 0.039199 | 0.012702 | 0.0476366 | -0.026497 | -0.0349345 |
| W02 | 4103 | 0.0391905 | 0.0126984 | 0.0470759 | -0.0264921 | -0.0343776 |
| W03 | 4101 | 0.0362814 | 0.00903249 | 0.0383983 | -0.0272489 | -0.0293658 |
| W03 | 4102 | 0.0356005 | 0.00906178 | 0.0379231 | -0.0265387 | -0.0288613 |
| W03 | 4103 | 0.0357411 | 0.00902209 | 0.0387256 | -0.026719 | -0.0297035 |
| W04 | 4101 | 0.0393155 | 0.0272637 | 0.0358529 | -0.0120518 | -0.00858915 |
| W04 | 4102 | 0.0387033 | 0.0272505 | 0.0354897 | -0.0114528 | -0.00823918 |
| W04 | 4103 | 0.0396937 | 0.0272482 | 0.0363258 | -0.0124454 | -0.00907759 |
| W05 | 4101 | 0.0299399 | 0.00874645 | 0.02051 | -0.0211935 | -0.0117636 |
| W05 | 4102 | 0.0305554 | 0.00871125 | 0.0204318 | -0.0218441 | -0.0117205 |
| W05 | 4103 | 0.0297685 | 0.00873767 | 0.0207624 | -0.0210308 | -0.0120247 |


## 4. 各项独立secondary与跨teacher描述性均值

下表A/C/E为各teacher先平均3个seed，再对5个teacher等权平均。方向计数为teacher均值 / paired-seed原差值；不计算p值，不把五个teacher当成生物样本。state/output/coupling/effect保持分开。BC在Legacy下baseline=0，delta_r_B是signed effective output，不声明为校准释放率。

| 量 | A | C | E | C−A | 负差world;seed | C−E | 负差world;seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RGC excess CE (nats / RGC bin) | 0.00102885 | 0.00127396 | 0.000982087 | 0.000245107 | 0/5; 0/15 | 0.00029187 | 0/5; 0/15 |
| Observed H1 h_H RMSE | 0.0214247 | 0.00382997 | 0.0216964 | -0.0175948 | 5/5; 15/15 | -0.0178665 | 5/5; 15/15 |
| Unobserved H1 h_H RMSE | 0.0220894 | 0.00516118 | 0.0223876 | -0.0169282 | 5/5; 15/15 | -0.0172265 | 5/5; 15/15 |
| Observed BC s_B RMSE | 0.0208013 | 0.00682002 | 0.0227174 | -0.0139813 | 5/5; 15/15 | -0.0158974 | 5/5; 15/15 |
| Unobserved BC s_B RMSE | 0.0189049 | 0.00845402 | 0.0213006 | -0.0104509 | 5/5; 15/15 | -0.0128466 | 5/5; 15/15 |
| Observed BC delta_r_B RMSE | 0.0205283 | 0.00688527 | 0.0204639 | -0.013643 | 5/5; 15/15 | -0.0135786 | 5/5; 15/15 |
| Unobserved BC delta_r_B RMSE | 0.0204171 | 0.008633 | 0.0205035 | -0.0117841 | 5/5; 15/15 | -0.0118705 | 5/5; 15/15 |
| Direct d_E RMSE | 0.0318983 | 0.0145 | 0.0318364 | -0.0173983 | 5/5; 15/15 | -0.0173364 | 5/5; 15/15 |
| AC/inhibitory d_I RMSE | 0.00980187 | 0.00926058 | 0.00977948 | -0.000541284 | 2/5; 6/15 | -0.000518893 | 2/5; 6/15 |
| a_H parameter RMSE | 0.0993767 | 0.0350012 | 0.173786 | -0.0643755 | 5/5; 15/15 | -0.138785 | 5/5; 15/15 |
| gamma_BR parameter RMSE | 0.296712 | 0.296705 | 0.296709 | -6.95967e-06 | 4/5; 13/15 | -4.62435e-06 | 2/5; 9/15 |
| gamma_BA fixed-contract RMSE | 0 | 0 | 0 | 0 | 0/5; 0/15 | 0 | 0/5; 0/15 |
| gamma_AR parameter RMSE | 0.111346 | 0.111345 | 0.111346 | -8.37626e-07 | 3/5; 7/15 | -6.17565e-07 | 3/5; 8/15 |
| H1 feedback RMSE | 0.0102833 | 0.00241477 | 0.0141598 | -0.00786851 | 5/5; 15/15 | -0.0117451 | 5/5; 15/15 |
| BC-to-AC input u_A RMSE | 0.0184886 | 0.00373087 | 0.0185796 | -0.0147578 | 5/5; 15/15 | -0.0148487 | 5/5; 15/15 |
| H1-block delta-logit RMSE | 0.0133973 | 0.0049902 | 0.0156687 | -0.00840713 | 5/5; 15/15 | -0.0106785 | 5/5; 15/15 |
| AC-postsynaptic-block delta-logit RMSE | 0.00814384 | 0.0077086 | 0.00812386 | -0.000435242 | 2/5; 6/15 | -0.000415265 | 2/5; 6/15 |
| AC state a_A RMSE | 0.0155383 | 0.00482986 | 0.0159867 | -0.0107084 | 5/5; 15/15 | -0.0111568 | 5/5; 15/15 |
| AC output delta_o_A RMSE | 0.0132644 | 0.0122848 | 0.0132296 | -0.000979552 | 5/5; 15/15 | -0.000944783 | 5/5; 14/15 |
| RGC probability RMSE | 0.0121569 | 0.0134907 | 0.0118287 | 0.00133376 | 0/5; 0/15 | 0.00166204 | 0/5; 0/15 |


gamma_BA的零误差/零分散度来自teacher与student均固定为1，不能称作已识别。a_H、gamma_BR、gamma_AR的参数误差不替代干预效应误差；不同通路误差没有合成mechanism score。RGC excess CE使用evaluator-only teacher conditional probability计算，不使用抽样NLL差替代。

全部secondary的逐teacher、逐paired seed数值见 [teacher_summary.csv](../output/synthetic/retipath_population_stage_b_20260918/teacher_summary.csv)、[paired_differences.csv](../output/synthetic/retipath_population_stage_b_20260918/paired_differences.csv)、[per_fit.csv](../output/synthetic/retipath_population_stage_b_20260918/per_fit.csv)；逐node/route coupling估计见 [coupling_coordinates.csv](../output/synthetic/retipath_population_stage_b_20260918/coupling_coordinates.csv)。下列为每teacher的关键未观测量、通路量与prediction：

| Teacher | 量 | A | C | E | C−A | C−E |
| --- | --- | --- | --- | --- | --- | --- |
| W01 | Unobserved H1 h_H RMSE | 0.0173229 | 0.00485711 | 0.0161321 | -0.0124658 | -0.011275 |
| W01 | Unobserved BC s_B RMSE | 0.0233417 | 0.00912618 | 0.0246462 | -0.0142155 | -0.0155201 |
| W01 | Unobserved BC delta_r_B RMSE | 0.0200526 | 0.00881692 | 0.0207515 | -0.0112357 | -0.0119345 |
| W01 | Direct d_E RMSE | 0.0317261 | 0.0110979 | 0.0336593 | -0.0206282 | -0.0225614 |
| W01 | AC/inhibitory d_I RMSE | 0.00946978 | 0.00964759 | 0.00952722 | 0.000177812 | 0.000120376 |
| W01 | a_H parameter RMSE | 0.15334 | 0.0366091 | 0.202011 | -0.116731 | -0.165402 |
| W01 | gamma_BR parameter RMSE | 0.209255 | 0.209244 | 0.209243 | -1.16161e-05 | 9.6782e-07 |
| W01 | gamma_AR parameter RMSE | 0.141469 | 0.141472 | 0.141472 | 2.92663e-06 | -2.10751e-07 |
| W01 | RGC excess CE (nats / RGC bin) | 0.000652083 | 0.000873124 | 0.000632072 | 0.000221041 | 0.000241052 |
| W02 | Unobserved H1 h_H RMSE | 0.022754 | 0.00566771 | 0.0230468 | -0.0170862 | -0.0173791 |
| W02 | Unobserved BC s_B RMSE | 0.0210969 | 0.00792079 | 0.0250995 | -0.0131761 | -0.0171787 |
| W02 | Unobserved BC delta_r_B RMSE | 0.0205647 | 0.0081299 | 0.02247 | -0.0124348 | -0.0143401 |
| W02 | Direct d_E RMSE | 0.0338559 | 0.0109021 | 0.0400031 | -0.0229538 | -0.0291009 |
| W02 | AC/inhibitory d_I RMSE | 0.00727956 | 0.00784387 | 0.00711499 | 0.000564312 | 0.000728888 |
| W02 | a_H parameter RMSE | 0.0555323 | 0.0152863 | 0.254173 | -0.040246 | -0.238887 |
| W02 | gamma_BR parameter RMSE | 0.228338 | 0.228328 | 0.228348 | -1.03499e-05 | -2.05868e-05 |
| W02 | gamma_AR parameter RMSE | 0.114596 | 0.114596 | 0.114597 | -8.95685e-07 | -1.05721e-06 |
| W02 | RGC excess CE (nats / RGC bin) | 0.00208795 | 0.00260416 | 0.00200343 | 0.000516214 | 0.000600734 |
| W03 | Unobserved H1 h_H RMSE | 0.0286596 | 0.00513964 | 0.0302081 | -0.02352 | -0.0250685 |
| W03 | Unobserved BC s_B RMSE | 0.0176533 | 0.00849999 | 0.0200407 | -0.00915331 | -0.0115407 |
| W03 | Unobserved BC delta_r_B RMSE | 0.0183704 | 0.00818909 | 0.0195989 | -0.0101813 | -0.0114098 |
| W03 | Direct d_E RMSE | 0.0318208 | 0.0130045 | 0.0338674 | -0.0188163 | -0.0208629 |
| W03 | AC/inhibitory d_I RMSE | 0.00974572 | 0.00983904 | 0.00967366 | 9.33211e-05 | 0.000165384 |
| W03 | a_H parameter RMSE | 0.159452 | 0.0738307 | 0.191576 | -0.0856211 | -0.117745 |
| W03 | gamma_BR parameter RMSE | 0.292002 | 0.291987 | 0.292001 | -1.45845e-05 | -1.38647e-05 |
| W03 | gamma_AR parameter RMSE | 0.129573 | 0.12957 | 0.129568 | -2.89446e-06 | 1.28931e-06 |
| W03 | RGC excess CE (nats / RGC bin) | 0.000963195 | 0.00125563 | 0.000966097 | 0.00029244 | 0.000289538 |
| W04 | Unobserved H1 h_H RMSE | 0.0314645 | 0.00567714 | 0.0316553 | -0.0257874 | -0.0259781 |
| W04 | Unobserved BC s_B RMSE | 0.0182076 | 0.00863673 | 0.0235549 | -0.00957088 | -0.0149182 |
| W04 | Unobserved BC delta_r_B RMSE | 0.0233548 | 0.00862298 | 0.0249155 | -0.0147318 | -0.0162926 |
| W04 | Direct d_E RMSE | 0.0343523 | 0.0246126 | 0.0313955 | -0.00973968 | -0.00678293 |
| W04 | AC/inhibitory d_I RMSE | 0.00889326 | 0.00760426 | 0.0101048 | -0.001289 | -0.00250051 |
| W04 | a_H parameter RMSE | 0.0629529 | 0.0319289 | 0.12937 | -0.031024 | -0.0974413 |
| W04 | gamma_BR parameter RMSE | 0.526891 | 0.526899 | 0.52689 | 7.39496e-06 | 8.7173e-06 |
| W04 | gamma_AR parameter RMSE | 0.0967575 | 0.0967541 | 0.0967573 | -3.3347e-06 | -3.141e-06 |
| W04 | RGC excess CE (nats / RGC bin) | 0.000617339 | 0.000764876 | 0.0005158 | 0.000147536 | 0.000249075 |
| W05 | Unobserved H1 h_H RMSE | 0.0102461 | 0.00446428 | 0.0108959 | -0.00578185 | -0.00643165 |
| W05 | Unobserved BC s_B RMSE | 0.0142249 | 0.00808643 | 0.0131617 | -0.00613852 | -0.00507532 |
| W05 | Unobserved BC delta_r_B RMSE | 0.0197429 | 0.00940613 | 0.0147818 | -0.0103368 | -0.00537568 |
| W05 | Direct d_E RMSE | 0.0277367 | 0.0128829 | 0.0202568 | -0.0148538 | -0.00737392 |
| W05 | AC/inhibitory d_I RMSE | 0.013621 | 0.0113681 | 0.0124768 | -0.00225287 | -0.0011086 |
| W05 | a_H parameter RMSE | 0.0656066 | 0.0173509 | 0.0918002 | -0.0482557 | -0.0744493 |
| W05 | gamma_BR parameter RMSE | 0.227072 | 0.227066 | 0.227065 | -5.64283e-06 | 1.64456e-06 |
| W05 | gamma_AR parameter RMSE | 0.0743348 | 0.0743348 | 0.0743348 | 1.00856e-08 | 3.18203e-08 |
| W05 | RGC excess CE (nats / RGC bin) | 0.000823683 | 0.000871986 | 0.000793038 | 4.83032e-05 | 7.89482e-05 |


## 5. Cross-seed ambiguity

每condition同teacher的3个seed pairs，在相同test上计算trajectory或单个coupling字段的距离；先sequence/scale平均再pair平均。分散度降低不自动代表更接近teacher，更不是posterior不确定性。以下为5teacher描述均值，所有逐pair/teacher值另存。

| 量的seed-pair距离 | A | C | E | C−A | C−E | 负差teacher C−A;C−E |
| --- | --- | --- | --- | --- | --- | --- |
| Direct-BC delta-logit RMSE (primary) | 0.00170522 | 8.48715e-05 | 0.00157834 | -0.00162035 | -0.00149347 | 5/5; 5/5 |
| Observed H1 h_H RMSE | 0.000867106 | 4.42692e-05 | 0.000916854 | -0.000822837 | -0.000872585 | 5/5; 5/5 |
| Unobserved H1 h_H RMSE | 0.000837355 | 4.23286e-05 | 0.00088609 | -0.000795026 | -0.000843761 | 5/5; 5/5 |
| Observed BC s_B RMSE | 0.00106019 | 6.81712e-05 | 0.00103871 | -0.000992015 | -0.000970535 | 5/5; 5/5 |
| Unobserved BC s_B RMSE | 0.00100016 | 6.51487e-05 | 0.00097817 | -0.000935007 | -0.000913021 | 5/5; 5/5 |
| Observed BC delta_r_B RMSE | 0.000962324 | 5.44709e-05 | 0.000883206 | -0.000907854 | -0.000828736 | 5/5; 5/5 |
| Unobserved BC delta_r_B RMSE | 0.000904806 | 5.20923e-05 | 0.000829136 | -0.000852714 | -0.000777044 | 5/5; 5/5 |
| Direct d_E RMSE | 0.00147846 | 7.77704e-05 | 0.00135843 | -0.00140069 | -0.00128066 | 5/5; 5/5 |
| AC/inhibitory d_I RMSE | 0.000346823 | 0.000309947 | 0.00029143 | -3.68758e-05 | 1.8517e-05 | 2/5; 1/5 |
| a_H parameter RMSE | 0.0165965 | 0.00237203 | 0.0137865 | -0.0142245 | -0.0114144 | 5/5; 5/5 |
| gamma_BR parameter RMSE | 3.01617e-05 | 6.08063e-05 | 7.35805e-05 | 3.06446e-05 | -1.27742e-05 | 3/5; 4/5 |
| gamma_BA fixed-contract RMSE | 0 | 0 | 0 | 0 | 0 | 0/5; 0/5 |
| gamma_AR parameter RMSE | 2.24812e-05 | 8.40654e-06 | 1.3471e-05 | -1.40747e-05 | -5.06448e-06 | 3/5; 2/5 |
| H1 feedback RMSE | 0.00116663 | 0.000179816 | 0.000989487 | -0.000986813 | -0.000809672 | 4/5; 5/5 |
| BC-to-AC input u_A RMSE | 0.000877738 | 4.74803e-05 | 0.000805778 | -0.000830258 | -0.000758298 | 5/5; 5/5 |
| H1-block delta-logit RMSE | 0.00153517 | 0.000247527 | 0.00128711 | -0.00128764 | -0.00103958 | 4/5; 5/5 |
| AC-postsynaptic-block delta-logit RMSE | 0.000300863 | 0.000266672 | 0.000254428 | -3.41903e-05 | 1.22441e-05 | 2/5; 1/5 |
| AC state a_A RMSE | 0.000983284 | 0.000732101 | 0.00115502 | -0.000251183 | -0.000422923 | 5/5; 5/5 |
| AC output delta_o_A RMSE | 0.000412299 | 0.000412816 | 0.000341534 | 5.16983e-07 | 7.12819e-05 | 1/5; 1/5 |
| RGC probability RMSE | 0.0001247 | 3.45695e-05 | 0.000123662 | -9.01303e-05 | -8.90927e-05 | 5/5; 5/5 |


详见 [ambiguity_per_seed_pair.csv](../output/synthetic/retipath_population_stage_b_20260918/ambiguity_per_seed_pair.csv)、[ambiguity_teacher_summary.csv](../output/synthetic/retipath_population_stage_b_20260918/ambiguity_teacher_summary.csv)、[ambiguity_paired_differences.csv](../output/synthetic/retipath_population_stage_b_20260918/ambiguity_paired_differences.csv)。

## 6. 梯度诊断（未用于优化调整）

| 条件 | batch streams | mean cosine(all共同坐标) | negative | defined | N/A | zero norm |
| --- | --- | --- | --- | --- | --- | --- |
| A | base_R/base_R_1 | 0.679582 | 607 | 6000 | 0 | 0 |
| A | base_R/base_R_2 | 0.696934 | 532 | 6000 | 0 | 0 |
| A | base_R_1/base_R_2 | 0.69603 | 524 | 6000 | 0 | 0 |
| C | H/BC | -0.0491131 | 3329 | 6000 | 0 | 0 |
| C | base_R/BC | 0.00117868 | 3014 | 6000 | 0 | 0 |
| C | base_R/H | -0.0029593 | 2992 | 6000 | 0 | 0 |
| E | base_R/extra_R_1 | 0.677715 | 628 | 6000 | 0 | 0 |
| E | base_R/extra_R_2 | 0.684246 | 513 | 6000 | 0 | 0 |
| E | extra_R_1/extra_R_2 | 0.712809 | 385 | 6000 | 0 | 0 |


保存 **288000** 行norm记录（含单列hierarchy）和 **216000** 行cosine记录；每个update、每个batch在同一参数状态、clip/step前计算。raw/weighted norm及all/state/output/coupling分类见各fit日志与 [gradient_diagnostic_summary.csv](../output/synthetic/retipath_population_stage_b_20260918/gradient_diagnostic_summary.csv)。H1 data loss对output/coupling梯度为零的路由已核对。A的三个stream是同一base RGC数据的batch；E是base及两个独立RGC池。负cosine只描述当前梯度，不证明生理矛盾。没有PCGrad/GradNorm/重加权或增加optimizer steps。

## 7. 原始工件与限制

主目录：`output/synthetic/retipath_population_stage_b_20260918/`。汇总JSON：[results.json](../output/synthetic/retipath_population_stage_b_20260918/results.json)；逐sequence：[per_sequence.csv](../output/synthetic/retipath_population_stage_b_20260918/per_sequence.csv)；跨teacher汇总：[descriptive_summary.csv](../output/synthetic/retipath_population_stage_b_20260918/descriptive_summary.csv)。

- `worlds/Wxx/evaluator_only/teacher.pt`：teacher state/physical parameters；`test.pt`：冻结刺激、events、全节点truth与各block的delta-logit。
- `worlds/Wxx/train_data/*.pt`：base/extra RGC及切片后的H1/BC训练观测；`initial_students/*.pt`：15个配对初始state。
- `worlds/Wxx/fits/{A,C,E}_{seed}/`：45个step400 `final.pt`（含optimizer计数）、`completed.json`、trajectory、逐batch gradient norm/cosine。
- `worlds/Wxx/evaluation/*_raw.pt`：45个student全节点/通路预测数组与coupling参数；`teacher_heterogeneity.csv`：实际teacher字段分布。
- `source/`：冻结来源快照；`verify_results.py`：已消费test的数值验证；`FILE_MANIFEST.json`：最终文件hash清单。

结果仅覆盖本次5个realizable synthetic instances、所列刺激分布、20%固定节点及400-step预算；没有证明唯一可辨识、全局收敛或生理正确。强partial pooling、噪声、有效观测量及有限优化均可能限制恢复，不能将未观测节点改善直接等同于单细胞参数完全恢复。保留状态/输出、coupling参数、通路输出和端到端干预的不同结论。未比较softplus、未改变architecture或loss、未新增RF或进入Stage C。完成后停止。
