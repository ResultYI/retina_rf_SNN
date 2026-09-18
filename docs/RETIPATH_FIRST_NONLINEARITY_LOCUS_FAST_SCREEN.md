# RetiPath first-nonlinearity locus FAST SCREEN

1. **Relocation 是否成功产生 H1-like F2？成功；四个 cell 的机制位置检查均通过。**

M1 是 **cone-related effective input nonlinearity candidate**：将原有同一个 cell-specific `alpha=exp(theta)` 移到原始 stimulus proxy 的逐位置输入端，`phi(x)=x (x>=0), alpha*x (x<0)`；原 BC common-branch 非对称变换位置为 identity。H1 算子本身仍然线性，H1 trace 的 F2 来自它接收的上游非线性输入，不是证明 H1 本身新增了非线性。不能称为真实 cone phototransduction。

使用既有冻结 F2 刺激文件与投影代码：4 Hz、最大绝对 Weber contrast 0.5、width 1/2/4 pixels、phase 0/0.5，8 cycles pre-roll + 8 cycles measurement，150 Hz 连续相位生成，FIX_HISTORY_ZERO、默认动态初态。刺激文件 hash 不变，像素 Weber 最小值约 -0.49989034。每个原始 signed component 先投影到 4/8 Hz，再做固定 RMS 归约；归一化分母为对应 checkpoint、对应 trace 的 uniform F1。所有分母有效。下表是六种 grating 的范围，不是生物贡献量。

| Cell | Model | H1 feedback F2/F1 | Direct BC sustained F2/F1 | Direct BC transient F2/F1 | Membrane F2/F1 | Logit F2/F1 |
|---|---|---:|---:|---:|---:|---:|
| 67#6 | M0 | 1.88e-16–1.52e-15 | 0.0461–0.3243 | 0.0461–0.3244 | 0.0641–0.4766 | 0.0641–0.4772 |
| 67#6 | M1 | 0.1799–0.1801 | 0.1276–0.1375 | 0.1977–0.2130 | 0.3128–0.3363 | 0.3132–0.3367 |
| 67#7 | M0 | 1.25e-16–1.01e-15 | 0.0033–0.0560 | 0.0033–0.0560 | 0.0058–0.1389 | 0.0058–0.1391 |
| 67#7 | M1 | 0.0298–0.0298 | 0.0223–0.0242 | 0.0331–0.0360 | 0.0602–0.1064 | 0.0603–0.1065 |
| 68#3 | M0 | 1.28e-16–1.04e-15 | 0.0229–0.2788 | 0.0295–0.2789 | 0.0407–0.3586 | 0.0408–0.3590 |
| 68#3 | M1 | 0.1446–0.1448 | 0.1128–0.1210 | 0.1679–0.1797 | 0.2325–0.2478 | 0.2328–0.2481 |
| 68#10 | M0 | 1.20e-16–9.74e-16 | 0.0019–0.0832 | 0.0019–0.0832 | 0.0052–0.1356 | 0.0052–0.1358 |
| 68#10 | M1 | 0.0678–0.0679 | 0.0528–0.0570 | 0.0767–0.0829 | 0.1213–0.1406 | 0.1215–0.1407 |

H1 graph/state/feedback 的全部 24 个 cell×grating 条件在 M0 均低于既有 roundoff reference，在 M1 均超过它；这里的 reference 是 `4096*eps64*max_abs(raw component)` 再按 RMS 归约，保持原 F2 合同。M0 的 direct/broad BC 与 M1 的下游 BC 均有 F2。H1-modulated input、AC、effective E/I、gE/gI、V1/V2、adaptation、probability 及全部 width/phase 结果均在 `f2_comparison.csv`，不得把跨层幅值比解释为 biological contribution。

2. **Natural-movie prediction 相对 matched control 如何？平均恶化，主要代价出现在两个 OFF cells。**

单 seed `2026091301`；从每个 cell 同一 canonical checkpoint strict warm start（包括现有 alpha、fixed RMS 与 geometry），M0/M1 使用完全相同 minibatch schedule。原可学习参数继续学习、原固定参数继续固定。CPU float32，Adam lr=0.003、batch=4、weight decay=0、clip norm=5、200 updates，无 scheduler、early stopping 或 restart；step0 及每20步评价。训练仅 [0,16)，development [16,20)，150 Hz Bernoulli occupancy，150-bin 独立序列、30 warmup/120 scored、相同 strictly-past observed history 和 mask。

Primary 是各模型预定 development 评估点中最低 NLL checkpoint（完全相等时取最早 step）。全部八个 checkpoint 在任何 F2 推理前已锁定；F2 不参与选择。下表 NLL 单位为 nats/scored bin，Delta=M1−M0，正值不利于 M1。

| Cell | Type | M0 selected step | M1 selected step | M0 NLL | M1 NLL | Delta selected | Delta step200 |
|---|---|---:|---:|---:|---:|---:|---:|
| 67#6 | OFF parasol | 200 | 200 | 0.403958 | 0.444450 | +0.040492 | +0.040492 |
| 67#7 | ON parasol | 200 | 200 | 0.365538 | 0.367725 | +0.002187 | +0.002187 |
| 68#3 | OFF parasol | 0 | 200 | 0.468653 | 0.504805 | +0.036152 | +0.035027 |
| 68#10 | ON parasol | 140 | 0 | 0.296350 | 0.294343 | -0.002006 | -0.001457 |
| Equal-cell mean | 4 cells | — | — | 0.383625 | 0.402831 | +0.019206 | +0.019062 |

Selected Delta median=+0.019170；wins/losses/ties=1/3/0。固定 step200 也是 1 win/3 losses，mean=+0.019062、median=+0.018607。两个 OFF cells 的 selected 平均 Delta=+0.038322；两个 ON cells 为 +0.000090（一好一差）；每类仅两个 cell，不能作 population/type 结论。

M1 改位置后 step0 功能本来就不同，不要求与 M0 输出相等；所有共享坐标起点仍完全一致。三个 M1 的 dev 最优点落在 step200，其中两个 OFF cells 在最后阶段仍改善，不能称充分收敛。68#10 M1 的 dev 最优点是 step0，其训练没有带来进一步 dev 改善。起点优势/劣势与额外优化分别记录，没有用 F2 或结果更换预算。

| Cell / model | Train start | Train step200 | Best train (all planned points) | Dev start | Dev step200 | Best dev |
|---|---:|---:|---:|---:|---:|---:|
| 67#6 / M0 | 0.417285 | 0.416893 | 0.416893 | 0.405999 | 0.403958 | 0.403958 |
| 67#6 / M1 | 0.622882 | 0.472934 | 0.472934 | 0.605122 | 0.444450 | 0.444450 |
| 67#7 / M0 | 0.356275 | 0.356180 | 0.356001 | 0.367064 | 0.365538 | 0.365538 |
| 67#7 / M1 | 0.357866 | 0.357162 | 0.356970 | 0.370376 | 0.367725 | 0.367725 |
| 68#3 / M0 | 0.451892 | 0.450209 | 0.450209 | 0.468653 | 0.469778 | 0.468653 |
| 68#3 / M1 | 0.553952 | 0.479080 | 0.479080 | 0.595412 | 0.504805 | 0.504805 |
| 68#10 / M0 | 0.304702 | 0.304214 | 0.304214 | 0.297955 | 0.296882 | 0.296350 |
| 68#10 / M1 | 0.306390 | 0.305498 | 0.305498 | 0.294343 | 0.295425 | 0.294343 |

完整 train/dev/alpha 的 0,20,…,200 trajectories 保存在 `summary.json` 的 `fits[].curves` 及各 checkpoint 目录的 `training_record.json`。同 step 的预测差异如下：

| Step | 67#6 Delta | 67#7 Delta | 68#3 Delta | 68#10 Delta |
|---:|---:|---:|---:|---:|
| 0 | +0.199123 | +0.003313 | +0.126759 | -0.003611 |
| 20 | +0.155077 | +0.003092 | +0.102084 | -0.003951 |
| 40 | +0.123617 | +0.002950 | +0.083789 | -0.003293 |
| 60 | +0.099304 | +0.002756 | +0.071872 | -0.002867 |
| 80 | +0.083255 | +0.002718 | +0.062746 | -0.002202 |
| 100 | +0.072108 | +0.002609 | +0.056285 | -0.001898 |
| 120 | +0.062152 | +0.002486 | +0.050003 | -0.001708 |
| 140 | +0.054855 | +0.002470 | +0.045039 | -0.001438 |
| 160 | +0.049664 | +0.002297 | +0.041599 | -0.001317 |
| 180 | +0.045053 | +0.002200 | +0.038126 | -0.001432 |
| 200 | +0.040492 | +0.002187 | +0.035027 | -0.001457 |

3. **Alpha 是否稳定远离 identity，或回到 1 附近？四个 M1 的 step200 alpha 都朝 1 移动，但均未到 1，不能称已达到稳定平台。**

| Cell | Shared initial alpha | M0 selected alpha | M0 step200 alpha | M1 selected alpha | M1 step200 alpha | M1 step200 abs(alpha−1) |
|---|---:|---:|---:|---:|---:|---:|
| 67#6 | 0.200417 | 0.189058 | 0.189058 | 0.339472 | 0.339472 | 0.660528 |
| 67#7 | 1.366463 | 1.308151 | 1.308151 | 1.181698 | 1.181698 | 0.181698 |
| 68#3 | 0.254063 | 0.254063 | 0.227417 | 0.433511 | 0.433511 | 0.566489 |
| 68#10 | 1.483303 | 1.437961 | 1.444794 | 1.483303 | 1.281731 | 0.281731 |

68#10 的 F2 使用选定 step0 alpha=1.483303，而非 step200 的1.281731。M1 的两个 OFF alpha 仍明显低于1；两个 ON alpha 接近1的方向移动，但本轮不另设“等同 identity”阈值。全部八次训练均无 alpha bounds hit，仍用 [0.05,2]；全部参数/梯度/推理 trace 有限，所有33个预期可学习标量进入 optimizer，并观测到非零梯度。无 gradient clipping 触发。

4. **四 cell 结果是否一致？机制位置结果一致，prediction 的方向为三差一好。**

67#6 与 68#3 两个 OFF cell 的 NLL 代价较大；67#7 ON 有较小代价，68#10 ON 有较小收益。三个劣势 cell 在全部预定同-step 比较均为正 Delta，68#10 在全部预定同-step 比较均为负 Delta，因此方向并非仅某个 checkpoint 偶然决定。

M0/M1 都有33个 trainable scalar，theta 只有一个；M1 旧 BC transform identity 的张量重构检查通过。theta=0（alpha=1）的必要 fixture 中两者 logits 逐位一致。正半轴斜率固定1，负半轴斜率为 alpha：alpha 改的是斜率比例而非任意总体乘法尺度，没有增加第二个 alpha/强度坐标或新的精确 scale gauge；这不等于宣称全部参数可辨识。现有 canonical G_E/G_I 与 relative composition 不变。

CPU float32 preflight 的 trace/forward logits 与 probability 逐位一致。F2 float64 GPU 检查最初被过严的跨执行逐位相等检查中断；仅将该检查修为记录实际误差并验证64*eps64的舍入界限。最终最大 logit error=3.7747583e-15，最大 probability error=9.9920072e-16，远低于冻结 F2 定位 floor。该接口修复记录于 `checkpoints/inference_check_note.json`；原 protocol 和 selection lock 未改，未重训或重选，投影、F2 floor、刺激不变。

所有正式模型源码与四个来源 checkpoint hashes 保持不变，F2 推理 state 前后相同。实验实现只在 `work/retipath_first_nonlinearity_locus/`，未修改正式默认入口；initial/final/best 共24个新实验 checkpoint 均可 strict load，best NLL 在保存后精确重放。

5. **Verdict：REJECT_EARLY_NONLINEARITY。**

早期非线性成功使 H1-like trace 出现 F2，满足机制位置 gate；但 matched prediction 平均明显变差且3/4 cells受损，selected 与固定 step200 两种读法一致。因此本轮记录该 trade-off 并停止。拒绝的是这一具体 relocation 候选在冻结 FAST SCREEN 合同下的升级，不是证明所有早期视网膜非线性无用；本轮没有足够证据称训练充分或作总体结论。

这是四-cell、单-seed development 架构筛选与模型内部 F2 定位，不是 population confirmation、独立生理验证或真实 cone/H1 机制验证。没有读取 [20,60) 或新的独立 test interval，没有新增 cells/seeds/updates、RF/DoG/radius、pathway clamps、Mach/SBC 或 population long training。

产物：`output/experiments/retipath_first_nonlinearity_locus_fast_screen_20260916/`。`protocol.json` 记录来源 hash、全部 schedule 与预注册规则；`per_cell_prediction.csv` 记录起点/selected/final结果；`f2_comparison.csv` 记录每个刺激/trace group；`summary.json` 记录完整曲线和 correctness；`checkpoints/selection_lock.json` 记录 F2 前冻结选择。
