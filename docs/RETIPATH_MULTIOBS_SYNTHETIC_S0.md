# RetiPath Synthetic MultiObs Stage S0

核查日期：2026-09-17。单个 frozen canonical teacher 的 realizable synthetic proof-of-principle；不是真实视网膜机制验证。

## 1. 八个问题的回答

1. **RGC_ONLY：观察到预测接近teacher，但部分内部功能恢复有偏差。** A的expected CE为0.450734 nats/bin，teacher为0.449383，excess CE约0.001351；BC mean4 nRMSE仍为0.3001，四项intervention error为0.1294–0.1568。H1 nRMSE已低至0.0090，不能把所有latent均称为恢复失败。这是单teacher、有限数据和固定优化预算下的现象，不是结构不可辨识的数学证明。

2. **加入H1监督主要提高跨seed一致性，没有提高H1平均恢复准确度。** B相对A的H1 latent ambiguity从0.003671降至0.000578，约下降84.2%；但held-out H1 nRMSE从0.0090升至0.0150，四项intervention平均误差也未改善。因此不能把更一致的共同偏差解读为更正确的机制。

3. **加入BC监督明显改善BC状态和direct-BC intervention恢复。** C相对A的BC mean4 nRMSE从0.3001降至0.0104，约下降96.5%；direct-BC intervention error从0.1366降至0.0737，AC postsynaptic error从0.1568降至0.1380。H1 feedback与adaptation intervention没有同步改善；该收益具有pathway选择性。

4. **H1+BC没有在BC-only之上带来全面的intervention增益。** D相对A改善direct BC与AC两项，H1 feedback与adaptation两项反而更差。四项误差算术均值仅作描述汇总：D为0.1355，C为0.1341，A为0.1437；D较C略差，不能宣称两个中间层监督具有进一步的整体协同收益。

5. **EXTRA_RGC_CONTROL未复制BC恢复收益，但整体intervention表现更好。** E的BC nRMSE为0.3187，明显高于D的0.0106；direct-BC intervention error为0.1176，也高于D的0.0741。然而E在另外三项intervention上的平均误差均低于D，四项均值为0.1213，低于D的0.1355。因此目前只支持BC相关的额外收益，不支持multi-level在整体mechanism recovery上超过更多RGC数据。

6. **multi-level相对RGC_ONLY降低跨seed ambiguity，但没有胜过更多RGC数据对照。** D相对A的四项intervention ambiguity均降低，四项均值从0.0780降至0.0600，下降约23.0%；H1和BC latent ambiguity也降低。但E四项intervention ambiguity均更低，均值0.0195，D约为E的3.07倍。降低ambiguity不等于正确恢复，也不能据三个seed声称消除全部歧义。

7. **没有观察到明显的RGC prediction牺牲。** D的expected CE为0.450293，低于A的0.450734；比E的0.450233仅高0.0000598 nats/bin，约0.0133%。这符合预先冻结的1%描述界限，但不是统计non-inferiority证明。Sampled NLL同样为D略优于A、E略优于D。

8. **当前为部分支持，暂不以S0整体成功为依据进入Synthetic S1。** BC latent/direct-pathway的proof-of-principle得到支持；H1的准确度增益、两个中间层的额外intervention收益以及超过更多RGC数据的整体优势尚未建立。15个有效fit与最终评价全部完成，验证通过，未追加训练或启动S1。结论仅限这个realizable synthetic teacher，不外推真实生理机制。

## 2. 运行身份与边界

有效运行完成 **5 conditions × 3 student seeds × 3000 macro-updates = 45,000 optimizer updates**。仅使用每次训练的 final checkpoint，无 early stopping、checkpoint selection、延长或结果驱动条件扩展。正式模型与默认训练入口未修改。

Teacher：cell `67#7`，seed `2026091301`，`CanonicalGainRetiPath`；checkpoint SHA256 `5283d88100f7b44c6e9f9d711489514767426480ce6c9713bb6bc22c5f09293d`。全部参数冻结。其学习参数仅定义已知 synthetic dynamical system，不称为真实生理 ground truth。

Students：seeds `2026091701/02/03`。复用标准 fresh RetiPath 初始化，再通过现有 exact canonical coordinate migration 得到各自33个可学习标量；未从教师学习参数或 optimizer state warm-start。所有条件在同一seed共享完全相同初值与相同dataset minibatch schedule。几何、配置、固定RMS沿用教师，以保持同一个可实现的坐标系统；未读取真实刺激或spike targets。

协议 SHA256：`f41192f06eb3240ea6f1e4bfca6e8d281998eb652eeebe371ee87c2aa784ebae`。有效运行协议锁定时间 `2026-09-17T09:22:59.774283+00:00`。运行设备 CPU、float32、每fit1线程、deterministic algorithms；统计归约使用float64。

### 作废尝试的完整披露

首版固定光栅仅有少量phase/sign组合。逐条stimulus哈希审计发现D_B held-out中7个、mechanism_B中10个唯一波形与train重复，因此在任何held-out指标查看之前终止了首批5个部分训练。没有final checkpoint，所有结果排除；未根据拟合效果修改实验。

该尝试保存在 `output/experiments/retipath_multiobs_synthetic_s0_20260917_invalid_stimulus_overlap/`。末次日志更新数：A2500、B1750、C1250、D1250、E2500；日志每250步记录，精确中断步数为UNVERIFIED。**45,000只计有效benchmark，不能当成本轮全部计算量**。源码、旧协议、日志和`INVALIDATION.json`、`interruption_accounting.json`均保留。

修正仅为固定空间光栅引入独立连续phase及每15bin独立contrast amplitude，保持原定刺激家族。全部有效数据和protocol重新锁定后才开始有效学生训练。576条序列及其计分段逐条哈希唯一，无train/held-out/mechanism完全重复。

## 3. Synthetic 数据与观测合同

全部sequence为150bins、150Hz、独立state reset；前30bins只作warmup，后120bins评分。Weber输入固定在[-0.8,+0.8]内，不进行结果后幅度调整。

**150Hz是simulation/data clock（dt≈6.67ms），不是刺激以150Hz明暗振荡。** 当前被冻结的noise components均为AR(1)：H rho=.85（相关tau≈41.02ms），BC flicker rho=.30（tau≈5.54ms），R rho=.92（tau≈79.95ms）。对应线性AR分量的3dB频率约3.89/33.10/1.99Hz；tanh、steps和reversals会改变最终频谱，不能把这些数当作完整刺激的硬cutoff或生理参数。

运行中收到的50/20/40ms建议不同于上述已冻结合同，尤其BC flicker原设计更宽带。用户随后明确选择完成当前冻结S0并报告差异，因此未替换刺激或预算。本报告不把现有结果描述为已经检验了该新参数版本，也不据此声称biological plausibility或real transfer有效性。

| Dataset | Train / held-out | 预先固定generator |
|---|---:|---|
| D_H | 64 / 32 | Gaussian空间平滑noise，sigma2 grid pixels；ARrho=.85；30bin contrast steps；有界tanh |
| D_B | 64 / 32 | 一半full-field ARrho=.3 flicker；一半8pixel周期固定空间光栅、0/45/90/135度、每15bin reversal；独立phase Uniform[0,2pi)、block amplitude Uniform[.2,.6] |
| D_R | 64 / 32 | sigma1.25空间相关、rho=.92时间相关local+global Gaussian过程，再有界tanh |
| D_R_EXTRA | 192 / 无专属held-out | 同D_R generator，独立seed；E仅用该192条，评价使用共同D_R held-out |
| Mechanism bank | 0 / 96 | 全新独立seeds，H/B/R families各32条；只用于interventions和cross-seed ambiguity |

H1：固定functional center `[0.08704902231693268, 0.026340056210756302]` 附近最近graph node `146`，位置 `[0.10781249403953552, -0.0]`；不依据student或响应挑选。

BC：canonical `parts.direct` 和 `parts.broad` 在K=2空间模式轴求和，得到direct sustained/transient、broad sustained/transient四值，位于composition/gain之前。与正式forward的`bc_direct_presynaptic`/`bc_broad_presynaptic`逐项核对。不是BC膜电位或释放的生理声明。

Observation head为identity，没有measurement noise、scale/offset学习或temporal decoder。H1及每个BC channel使用各自teacher TRAIN计分段的mean和population std；所有student使用相同固定常数。`L_H`为标准化MSE，`L_B`对time×4channels平均，`L_R`为项目现有Bernoulli NLL；active dataset losses直接相加，系数全为1。

Spikes复用项目`clean_sampled_data._sample_spikes`，逐时刻从teacher probability采样，过去sampled spikes进入下一时刻history。Teacher true conditional probabilities与training targets分文件保存，训练函数不读取它们。评价时teacher和student都以同一已采样过去历史为条件。教师history gate实际为0：因果采样语义保留，但该teacher的history contribution为零；另用标准初值的非零history gate验证采样重放。

## 4. 固定训练与评价

Adam lr=.003、betas=(.9,.999)、eps=1e-8、weight decay0、batch4/dataset、global gradient clip5。采用当前RetiPath Phase2的optimizer/batch/上限；旧400-step/.03 synthetic合同对应较早的pre-spatial模型，不用于本次canonical gain optimizer。H→B→R依次累积梯度，再一次optimizer.step；只执行该condition存在的数据集。沿用原参数约束，包括alpha[.05,2]和history gate[0,1]。

H1 recovery在D_H held-out，BC recovery在D_B held-out；nRMSE分母为各自teacher TRAIN std。BC_mean4为四个channel指标的算术平均。相关系数为scored bins展平后的Pearson correlation，详见CSV。

四项正式interventions均在同一独立96-sequence mechanism bank上运行，NORMAL和intervened两边都FIX_HISTORY_ZERO。Delta=normal logit−intervention logit；E=RMS(student Delta−teacher Delta)/(RMS(teacher Delta)+1e-8)。Intervention targets从不进入loss。

A_latent也使用这同一个96-sequence bank，H1/BC差异用teacher TRAIN std标准化；A_intervention以同intervention的teacher Delta RMS+eps标准化。每condition保留三对seed的原始值，并报告其平均。不能用raw parameter variance替代这些指标。

下列表格为三个seed均值及[min,max]；不进行3-seed显著性或population推断。预先固定的“prediction保持”描述界限是expected CE相对比较条件均值上升不超过1%，不是统计non-inferiority结论。

## 5. 预测与latent recovery

| Condition | Sampled NLL | Expected CE | H1 nRMSE | BC mean4 nRMSE |
|---|---:|---:|---:|---:|
| A RGC_ONLY | 0.456860 [0.456235, 0.457237] | 0.450734 [0.450068, 0.451114] | 0.009021 [0.008232, 0.010466] | 0.300113 [0.276509, 0.316577] |
| B RGC_H1 | 0.456850 [0.456238, 0.457249] | 0.450743 [0.450082, 0.451135] | 0.014971 [0.014774, 0.015264] | 0.299169 [0.273135, 0.317282] |
| C RGC_BC | 0.456548 [0.455966, 0.456841] | 0.450294 [0.449698, 0.450604] | 0.020478 [0.018642, 0.023150] | 0.010414 [0.010117, 0.010991] |
| D RGC_H1_BC | 0.456546 [0.455963, 0.456838] | 0.450293 [0.449694, 0.450606] | 0.014350 [0.014148, 0.014656] | 0.010560 [0.010272, 0.011075] |
| E EXTRA_RGC_CONTROL | 0.456406 [0.456334, 0.456494] | 0.450233 [0.450215, 0.450265] | 0.010617 [0.009257, 0.012174] | 0.318686 [0.315327, 0.325057] |

Teacher sampled NLL=0.455695605，teacher expected CE/entropy=0.449382611 nats/bin。后者是已知true probability的参考；完整excess CE在`heldout_prediction.csv`。

## 6. Primary：held-out intervention recovery

| Condition | H1 feedback block | Direct BC drive block | AC postsynaptic drive block | Adaptation term removal |
|---|---:|---:|---:|---:|
| A RGC_ONLY | 0.152245 [0.144209, 0.157450] | 0.136602 [0.125519, 0.147338] | 0.156766 [0.133311, 0.169702] | 0.129364 [0.076152, 0.160210] |
| B RGC_H1 | 0.159574 [0.154940, 0.167666] | 0.137494 [0.124491, 0.149366] | 0.158634 [0.132801, 0.172652] | 0.130768 [0.076918, 0.160646] |
| C RGC_BC | 0.174193 [0.142730, 0.191003] | 0.073749 [0.063881, 0.093231] | 0.137970 [0.121097, 0.153628] | 0.150555 [0.079447, 0.195800] |
| D RGC_H1_BC | 0.178540 [0.145730, 0.195485] | 0.074073 [0.063609, 0.094988] | 0.140142 [0.122914, 0.155481] | 0.149128 [0.077794, 0.194514] |
| E EXTRA_RGC_CONTROL | 0.145616 [0.141082, 0.152065] | 0.117560 [0.115208, 0.120624] | 0.130269 [0.125585, 0.132688] | 0.091784 [0.090474, 0.094004] |

Teacher Delta RMS（logit units）：BLOCK_H1_FEEDBACK=0.178207014；BLOCK_DIRECT_BC_DRIVE=1.82775173；BLOCK_AC_POSTSYNAPTIC_DRIVE=1.30025406；REMOVE_ADAPTATION_TERM=0.00454006932。同时报告absolute RMS error，防止小分母的相对误差被误读。

## 7. Cross-seed ambiguity

| Condition | H1 latent | BC mean4 latent | H1 intervention | BC intervention | AC intervention | Adaptation intervention |
|---|---:|---:|---:|---:|---:|---:|
| A RGC_ONLY | 0.003671 | 0.051981 | 0.086627 | 0.070809 | 0.037021 | 0.117386 |
| B RGC_H1 | 0.000578 | 0.057289 | 0.095952 | 0.072866 | 0.039695 | 0.120222 |
| C RGC_BC | 0.003321 | 0.005955 | 0.057029 | 0.053125 | 0.030352 | 0.101270 |
| D RGC_H1_BC | 0.000646 | 0.006051 | 0.055353 | 0.052927 | 0.030505 | 0.101223 |
| E EXTRA_RGC_CONTROL | 0.003628 | 0.015663 | 0.027057 | 0.008043 | 0.015964 | 0.026995 |

## 8. Secondary：canonical parameter recovery

`parameter_recovery.csv`提供20个对应量×15fits=300行：H1 tau/delay/amplitude、六个BC basis taus及两类delay、alpha、canonical G_E/G_I、两项独立composition weights、两类AC tau/delay。保存teacher值、student值和绝对/相对误差。未比较旧raw gain/gauge坐标，也未把这些参数误差作为主结果或唯一可识别性的证明。

## 9. 解释边界与验证

- A/B/C/D的RGC训练集相同；E使用192条独立RGC序列。相同optimizer steps和batch不等于相同计算量：D每步三个dataset batches，E每步一个。E也不是information-matched control；无噪声连续latent观测与随机binary spikes的信息量不同。
- 初始化随机性沿用项目标准的小幅BC perturbation，另有各seed minibatch差异。只有一个teacher、三个seeds，不能据此断言消除了全部机制歧义或发现了所有等价解。
- 固定有限训练预算下的恢复差异，不能单独区分有限样本、优化不足与结构不可辨识。Output-close/mechanism-far是本benchmark的观测现象，不是RGC-only在无限数据/充分优化下必然不可辨识的数学证明。
- H1 state与反馈amplitude是不同量；H1 identity监督不直接观测amplitude。BC观测是四个K-summed branch values，不是两个K modes分别的全状态。未监督AC或RGC内部变量。
- 任何正结果仅支持当前realizable synthetic teacher下的latent/intervention约束作用。真实HC/BC observation correspondence、measurement nuisance和model mismatch均未测试。
- 11项预训练检查通过；15个final checkpoint与optimizer step数核实；teacher与正式源码hash未变；保存数组对prediction/latent/intervention/ambiguity CSV逐行公式重算通过。这是执行者独立公式重算，不冒称独立人员审计。
- 未读取真实physiology数据、未做population training、未改architecture、未调loss weights、未运行S1。完成此S0后停止。

## 10. 文件

结果根目录：`output/experiments/retipath_multiobs_synthetic_s0_20260917/`。

- `protocol.json`、`teacher_metadata.json`、`normalization.json`、`dataset_manifest.json`
- `datasets/`：train/held-out inputs、synthetic observations、分离的teacher probabilities和独立mechanism stimuli
- `training_curves.csv`、`heldout_prediction.csv`、`latent_recovery.csv`、`parameter_recovery.csv`
- `intervention_recovery.csv`、`ambiguity.csv`、`summary.json`
- `initial_states/`、`checkpoints/`、`evaluation_arrays/`、`verification/`、`evaluation_manifest.json`
- 实现：`work/retipath_multiobs_s0/`；`run.py prepare`→`test_contract.py`→`launch.py`→`evaluate.py`→`verify_results.py`。现有输出拒绝自动覆盖或重新训练。
