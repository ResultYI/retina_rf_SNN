# Synthetic S0.5 — H1 Observation–Mechanism Alignment

2026-09-17。一个已知synthetic RetiPath teacher的固定预算实验；不是真实HC voltage或真实视网膜机制验证。

**Verdict：MIXED_OR_OPTIMIZATION_LIMITED**。四条件×三个seed均完成3000次macro-updates，共36,000次有效optimizer updates；最终数值重算通过。

## 1. 八个问题

1. **State-only监督没有改善state平均恢复；联合监督改善了state。** B的state nRMSE为0.01497，高于A的0.00902，三个seed均更差；D降至0.00504，较A降低44.2%，三个seed均改善。State观测在代码中直接约束状态轨迹及H1 tau/delay，不直接依赖amplitude；但本预算下B的tau和delay平均绝对误差也都高于A，因此不能把其作用称为成功恢复了tau/delay。B最明确的收益是跨seed状态一致性。

2. **Feedback监督额外提供了amplitude与返回映射输出的直接约束，但没有实现更准确的feedback恢复。** C的feedback nRMSE为0.03490，D为0.03483，均高于A的0.01941与B的0.01358，三个seed方向一致。反而B在没有直接feedback loss时将此误差较A降低30.1%。须区分“计算图中增加可见约束”与“固定联合训练后获得更准确恢复”。

3. **没有任何监督条件达到预定义的H1 intervention明确改善标准。** E_H1均值为A0.15225、B0.15957、C0.15067、D0.15822。C数值最低，但相对A仅下降1.03%、仅1/3 seeds改善；相对B下降5.58%、2/3 seeds改善，均未达到10%且3/3同向。D比C高5.012%，略超5%保持界限，且C本身并无合格的feedback/intervention收益。结论不依赖这个接近5%的单一边界。

4. **Amplitude恢复与intervention恢复没有形成支持性的对应关系。** Teacher amplitude=0.185228；A/B/C/D的平均绝对误差分别为0.003234/0.001225/0.007263/0.006606。B的amplitude最接近teacher，却有最大的平均E_H1；C的amplitude误差最大，但E_H1均值略低。因此不能用amplitude恢复解释本轮的intervention改善，更不能声称已证明因果中介。D同时给出“更准确state不保证更准确feedback”的模型内例子：state误差较A下降44.2%，feedback误差反而增加79.5%。

5. **State和feedback跨seed一致性提高，但没有同等程度的干预一致性收益。** 相对A，B/C/D的A_state分别下降约84.2%/89.7%/85.2%，A_feedback分别下降约32.8%/86.9%/85.6%。A_intervention为A0.08663、B0.09595、C0.08541、D0.08164：B变差，C/D仅小幅下降。C/D更一致的feedback同时更偏离teacher，不能把ambiguity下降称为正确恢复。

6. **RGC prediction保持。** A/B/C/D的expected CE分别为0.450734/0.450743/0.450731/0.450741 nats/bin，三个H1监督条件均通过沿用的1%描述界限；最大均值增幅约0.00214%。Sampled NLL分别约0.456860/0.456850/0.456850/0.456864。此处不是统计non-inferiority结论。

7. **Verdict：MIXED_OR_OPTIMIZATION_LIMITED。** STATE_SUPERVISION_SUFFICIENT不成立，因为B未改善state与intervention。FEEDBACK_OBSERVATION_REQUIRED也不成立，因为C未改善feedback，H1 intervention变化未达幅度与seed一致性要求，D也未保留所需改善。该名称不表示已经证实优化故障；固定有限预算无法单独区分优化、联合目标权衡与可辨识性限制。12个fit全部完成，未调整判据、权重、架构或预算，未启动S1。

8. **在当前model family内，只有理想H1 state观测仍不足以单独确定H1→downstream amplitude coupling。** 原因是state对amplitude没有直接依赖；即使state准确，也未直接测得amplitude乘返回映射后的feedback。RGC loss可间接提供约束，但本轮未建立唯一coupling恢复，也没有证明feedback监督是充分或必要的成功解决方案。真实HC voltage是否对应此state、是否能约束真实反馈机制均未测试，不能外推。

## 2. 冻结身份、数据与公平性

Teacher沿用S0：cell `67#7`、seed `2026091301`、checkpoint SHA256 `5283d88100f7b44c6e9f9d711489514767426480ce6c9713bb6bc22c5f09293d`。Teacher全冻结，学习参数仅代表此synthetic dynamical system。

S0.5 protocol SHA256：`028fbd6f2c60df25a2084e7edddf69b3e03c4159befdcf77a765fdbc0fbabe1e`，冻结时间 `2026-09-17T10:30:28.300298+00:00`；来源S0 protocol SHA256：`f41192f06eb3240ea6f1e4bfca6e8d281998eb652eeebe371ee87c2aa784ebae`。

学生seeds为2026091701/02/03，逐字节复用S0保存的fresh canonical initial states、固定geometry/RMS与33个可学习标量。未从S0 final或teacher学习参数warm-start。A/B重新按本轮预算训练，三个A与三个B的final model state hashes分别与S0 RGC_ONLY/RGC_H1完全一致。

复用D_H train64/held-out32、D_R train64/held-out32，以及同一个96条mechanism bank。没有重新生成任何stimulus或spike。Mechanism bank仍为原H/B/R各32条；其中B-family只是被冻结评价库的一部分，本轮未读取D_B观测数据、D_R_EXTRA或真实数据。

这些held-out stimuli已用于S0评价。本轮按用户指定复用，属于既有synthetic bank上的限定比较，independent new confirmation test = NONE。协议与checkpoint核对中的文件哈希读取只用于身份验证；held-out response指标在12个final完成后统一计算。

每条150bins，150Hz，独立state reset，前30bins warmup、后120bins计分。时间相关参数原封不动：H的AR rho=.85（相关tau约41.02ms），R的rho=.92（约79.95ms）；机制库也逐字节复用，包括原B片段。没有应用后来提出的50/20/40ms方案。

Adam lr=.003、betas=(.9,.999)、eps1e-8、weight decay0、batch4、clip5、原project_mechanism_parameters与全部参数约束。每fit固定3000步，final-only，无early stopping或checkpoint selection。CPU float32、每fit单线程、deterministic algorithms；统计归约float64。

所有active loss系数为1：A仅RGC Bernoulli NLL；B增加state MSE；C增加feedback MSE；D增加两者。B/C/D每步使用完全相同的S0 H batch序列；D的两个loss来自同一个H forward和同一个batch，求和后backward，再累积RGC梯度，只调用一次optimizer.step。RGC batch序列也与S0一致。

B/C均监督一个scalar trace；D同时监督两个scalar traces，因此D与B/C不是观测信息量完全相同的比较。相同optimizer steps也不意味着A与其余条件计算量相同。

## 3. State与feedback的正式定义

正式计算：`graph_drive = G(x)`；`state = lowpass(delay(graph_drive))`；`feedback = A_H × G.transpose_apply(state)`；下游输入为`x − feedback`。实现直接复用`H1Pathway.forward`，没有重写反馈量或更改架构。

State observable固定为node146。State由H1 tau/delay和固定输入图决定，没有对H1 amplitude的直接依赖；RGC loss仍可通过下游间接约束amplitude。

Feedback observable仅取pixel `146`，位置 `[0.10781249403953552, -0.0]` deg。功能中心 `[0.08704902231693268, 0.026340056210756302]` deg。规则为输入geometry到冻结中心的平方距离argmin，平局取首索引；已在任何student response之前写入protocol。

虽然两者索引均为146，feedback并不是简单的node146 state乘amplitude：它先聚合返回映射所覆盖的多个state。Feedback监督同时约束动态状态的返回映射结果与amplitude，不是amplitude-only监督，也不代表测得的feedback current。

State normalization完全复用S0 TRAIN常数：mean=-0.0187365193201，std=0.343995609979。Feedback仅使用teacher D_H TRAIN的7680个scored bins冻结：mean=-0.00364571240028，std=0.0591065961503。Population std，identity observation head，无noise、可学习scale/offset/filter。

正式forward逐项对齐检查通过；H1 block使feedback归零而保留state。预训练autograd检查也确认state loss路径对amplitude无直接梯度、feedback路径具有非零amplitude梯度。检查未执行optimizer updates。

## 4. 评价与预先冻结判据

以下表格为三个seed均值及[min,max]。只有三个seed，不作显著性或population推断。State/feedback分别在相同D_H held-out评价，nRMSE分母为对应teacher TRAIN std，correlation为计分段展平后的Pearson correlation。

唯一primary intervention为BLOCK_H1_FEEDBACK。NORMAL与block均使用FIX_HISTORY_ZERO，Delta=normal logit−blocked logit；E_H1=RMS(student Delta−teacher Delta)/(RMS(teacher Delta)+1e-8)。未计算其他干预，也未使用任何intervention loss。

A_state、A_feedback和A_intervention全部在同一个96条mechanism bank计分段上计算，分别用state TRAIN std、feedback TRAIN std和teacher Delta RMS+1e-8标准化；保留三对seed原始值并报告其均值。三个pair共享seed，不能当作三个独立重复实验。

用户确认并在训练前冻结：明确改善=平均归一化误差至少下降10%，且3/3配对seed严格同向改善。D保持改善=相对A/B均保留上述改善，且feedback与H1 intervention平均误差各不超过C的105%。

- STATE_SUPERVISION_SUFFICIENT：B相对A同时明确改善state、feedback和H1 intervention。
- FEEDBACK_OBSERVATION_REQUIRED：B明确改善state recovery，或A_state下降至少10%；C相对A和B均明确改善feedback与H1 intervention；D保持上述改善。
- MIXED_OR_OPTIMIZATION_LIMITED：不满足前两者，包括配对seed方向不稳定。按上述顺序裁决，无结果后改判据。

RGC prediction在共同D_R held-out评价，使用相同teacher sampled past；teacher true probabilities只用于评价。沿用S0的1% expected CE均值描述界限，不是统计non-inferiority证明。Teacher history gate实际为0，正确因果采样语义保留。

## 5. Held-out state、feedback与primary intervention

| Condition | State nRMSE | State correlation | Feedback nRMSE | Feedback correlation | H1 intervention E |
|---|---:|---:|---:|---:|---:|
| A RGC_ONLY | 0.009021 [0.008232, 0.010466] | 0.999954 [0.999938, 0.999963] | 0.019409 [0.016777, 0.021644] | 0.999958 [0.999942, 0.999966] | 0.152245 [0.144209, 0.157450] |
| B RGC_STATE | 0.014971 [0.014774, 0.015264] | 0.999890 [0.999886, 0.999893] | 0.013576 [0.013385, 0.013758] | 0.999899 [0.999895, 0.999902] | 0.159574 [0.154940, 0.167666] |
| C RGC_FEEDBACK | 0.013817 [0.013653, 0.013931] | 0.999904 [0.999903, 0.999906] | 0.034896 [0.034322, 0.035475] | 0.999911 [0.999909, 0.999913] | 0.150670 [0.132866, 0.160364] |
| D RGC_STATE_FEEDBACK | 0.005036 [0.004849, 0.005374] | 0.999986 [0.999985, 0.999987] | 0.034831 [0.034330, 0.035412] | 0.999988 [0.999986, 0.999988] | 0.158222 [0.140531, 0.167959] |

Teacher H1 Delta RMS=0.178207014402 logit units；CSV同时保存absolute RMS error。

## 6. 三个H1参数（secondary）

| Condition | Tau ms | Delay ms | Amplitude |
|---|---:|---:|---:|
| Teacher | 12.097323418 | 19.658266068 | 0.185228452 |
| A RGC_ONLY | 12.110062 [12.034096, 12.254300] | 19.903866 [19.902143, 19.904921] | 0.188462 [0.187886, 0.188944] |
| B RGC_STATE | 12.748462 [12.741087, 12.756605] | 19.292955 [19.281460, 19.299118] | 0.186454 [0.185757, 0.187079] |
| C RGC_FEEDBACK | 11.460472 [11.454073, 11.468122] | 19.732154 [19.728962, 19.737221] | 0.177965 [0.177848, 0.178086] |
| D RGC_STATE_FEEDBACK | 12.186100 [12.179873, 12.196259] | 19.504189 [19.493729, 19.509956] | 0.178622 [0.178495, 0.178749] |

CSV保存每个seed的teacher/student值、带符号误差、绝对误差和相对绝对误差。未扩展全参数audit。参数误差与intervention误差的共同变化只能描述对应关系；本轮未做amplitude单独替换或因果中介实验，不能把所有改善归因于amplitude。

## 7. Cross-seed ambiguity与RGC prediction

| Condition | A_state | A_feedback | A_intervention | Sampled NLL | Expected CE |
|---|---:|---:|---:|---:|---:|
| A RGC_ONLY | 0.003671 | 0.005740 | 0.086627 | 0.456860 [0.456235, 0.457237] | 0.450734 [0.450068, 0.451114] |
| B RGC_STATE | 0.000578 | 0.003859 | 0.095952 | 0.456850 [0.456238, 0.457249] | 0.450743 [0.450082, 0.451135] |
| C RGC_FEEDBACK | 0.000380 | 0.000754 | 0.085410 | 0.456850 [0.456227, 0.457246] | 0.450731 [0.450063, 0.451122] |
| D RGC_STATE_FEEDBACK | 0.000544 | 0.000828 | 0.081636 | 0.456864 [0.456237, 0.457255] | 0.450741 [0.450073, 0.451131] |

Teacher sampled NLL=0.455695605，expected CE/entropy=0.449382611 nats/bin。

## 8. 判据逐项结果

| Gate | Result |
|---|---|
| state_sufficient | False |
| B_state_constraint | True |
| C_clearly_improves_feedback_and_intervention | False |
| D_retains_improvements_within_5percent_of_C | False |

每项mean ratio、相对变化和三个seed的paired differences保存在summary.json；不使用correlation或ambiguity替代ground-truth recovery裁决。

## 9. 验证与解释边界

- 7项预训练检查通过；12个final checkpoints均为3000步，每个optimizer参数的step均核实。A/B六个模型与S0对应结果逐字节相同。
- 从保存数组使用NumPy另写公式重算state/feedback、H1干预、预测、三参数和pairwise ambiguity，并核对loss总和、summary与固定verdict；全部通过。这是执行者的独立公式重算，不冒称独立人员审计。
- Source与输入hash保持冻结，S0 artifacts只读。实现仅位于work/retipath_multiobs_s05；新结果位于本轮独立目录。
- State loss不直接依赖amplitude是代码与数学性质；有限预算训练结果不能单独区分有限样本、优化限制与结构不可辨识，也不能证明唯一参数恢复。
- HC voltage到此H1 state的真实观测对应关系未建立。即使synthetic feedback监督有效，也不等于真实HC voltage或可测feedback量能提供相同约束。
- 未改architecture、未用D_B或D_R_EXTRA观测、未加BC/AC监督、未调loss weights、未增加seed或steps、未进入S1。完成后停止。

## 10. 文件

结果目录：`output/experiments/retipath_multiobs_s05_h1_observability_20260917/`。

- protocol.json、normalization.json、feedback_train.pt、preparation_complete.json
- training_curves.csv、heldout_state_feedback.csv、h1_parameter_recovery.csv
- intervention_recovery.csv、ambiguity.csv、prediction.csv、summary.json
- checkpoints/、curves/、evaluation_arrays/、verification/、evaluation_manifest.json
- 运行实现：work/retipath_multiobs_s05/run.py、launch.py、evaluate.py、verify_results.py；报告生成器report.py。
