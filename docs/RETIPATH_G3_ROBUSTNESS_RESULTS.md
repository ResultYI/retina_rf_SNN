# RetiPath G3 synthetic robustness benchmark

2026-09-18。运行状态：**COMPLETED_FIXED_BUDGET**；必要工程与数值核验：**VERIFIED**。5 个 world × 5 条件 × 3 seeds 的 75 个 fit 全部完成，共 13,500 updates、36,000 microbatches、144,000 次序列曝光。没有失败重试、追加训练或结果后改合同。

主要方向记录：**C−B 在 2/5 个 world 为负、3/5 为正；D−C 在 5/5 个 world 为正**。负值表示左侧条件的 primary error 更低；本轮没有据此作研究决策。

运行目录：`D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g3_robustness_20260918/`。

本轮只复核两个 primary 比较：**C−B（multiscale vs narrow）**、**D−C（progressive vs joint）**。D/E 的响应、latent 和 direct-BC recovery 单独描述；RF 保持 secondary。没有设立其他 primary、恢复阈值或研究决策。

完整合同：[protocol.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g3_robustness_20260918/protocol.json)。沿用依据：[G2 结果报告](D:/PythonProject/retina_rf_SNN/docs/RETIPATH_G2_MULTISCALE_RESULTS.md)与其中的冻结 protocol。

## 1. 五个 synthetic world 的事前生成规则

用户确认采用 5 个新 world。每个 world 的 12 个 teacher raw 坐标由 G2 teacher raw 坐标加独立 `N(0,0.5²)` 扰动获得，再经 G1 原有 sigmoid/tanh 映射回原参数范围。标准差 0.5 是事前固定的工程采样选择，不是生物先验或调参结果。没有根据 teacher 响应、训练难度或评价结果筛选、拒绝采样或重抽。

所有 teacher 参数、数据 family manifest 和训练 batch 日程，在任何 G3 目标生成之前一起冻结。W01–W05 使用互异的 `G3:Wxx` 命名空间，沿用 G2 的 SHA256 派生随机种子方式；teacher draw、刺激中心/相位、Gaussian noise 和 Bernoulli event uniform 使用分离的随机流。B/C 的配对随机数仅在同一个 world 内共享。

每个 world 都使用 G2 的三个 student seeds `2026091801/02/03`，对应初始张量与 G2 逐位一致，并在 A–E 间配对。跨 world 也复用这些初值，以固定优化起点；“独立 world”指在这些共同初值条件下独立生成的 teacher/data draw，不指生物个体或完全独立的初始化样本。原 G2 world 不计入 G3 的五 world 汇总。

以下为事前 teacher physical 参数（显示四位小数，完整精度及 raw draw 见 protocol）：

| 参数 | W01 | W02 | W03 | W04 | W05 |
|---|---:|---:|---:|---:|---:|
| tau_h | 26.7689 | 29.9920 | 35.5035 | 43.4544 | 35.0220 |
| a_h | 0.2408 | 0.2231 | 0.2402 | 0.0607 | 0.2323 |
| tau_f | 19.9543 | 13.7014 | 14.8010 | 17.4159 | 15.0874 |
| tau_s | 68.4283 | 73.5642 | 67.2935 | 69.7383 | 80.7140 |
| alpha | 0.2978 | 0.3065 | 0.1866 | 0.3066 | 0.2313 |
| tau_a1 | 38.6713 | 31.1694 | 44.2979 | 31.3728 | 35.1888 |
| tau_a2 | 107.2467 | 107.8283 | 118.4400 | 109.7248 | 124.1441 |
| g_e | 2.6958 | 0.9954 | 1.1519 | 3.1352 | 1.6205 |
| delta_e | −0.8361 | 2.2456 | −1.2749 | 2.0236 | 2.6701 |
| g_i | 0.8201 | 0.8140 | 1.5649 | 0.9376 | 1.5381 |
| delta_i | −1.5912 | −0.3723 | −2.4017 | −2.1525 | −1.8033 |
| bias | −1.8391 | −2.8453 | −1.2252 | −2.7641 | −2.0289 |

时间常数单位 ms，其余沿用 G1 effective-unit 含义。全部 teacher 均在同一 realizable 模型家族内；本轮不检验结构失配或生理真实性。

## 2. 完全沿用的 G2 合同

G1/G2 architecture、geometry、state/output/observation 语义、direct BC/AC/EI/RGC branches、参数范围、loss、优化器、训练日程和 RF 数值函数保持不变。新增 [robustness_study.py](D:/PythonProject/retina_rf_SNN/experiments/retipath_multiscale_v0/robustness_study.py) 只组织 world、来源锁、执行队列和汇总，直接调用 G2 已有函数；报告汇总入口改为仅汇总两个指定 primary 对比，不改指标公式。

每个 world 的设置相同：

- 物理视野 2×2 deg，32×32 像素面积平均，Gaussian 空间场 × 0.5/1.5/3/6 Hz 平均正弦；contrast=0.4，150 Hz，T=300，前 60 bins warmup，最后 240 bins 计分。
- 训练 scales `{0.15,0.30,0.60}`，common scale 0.30；held-out scales `{0.225,0.45}`。H 在 0.30/0.60 可见，BC output 在 0.15/0.30 可见，RGC 在三个训练尺度可见。
- Base multi 72 条（24/scale）；B 的 paired narrow 同样为 72 个不同 waveform family，但全部 sigma=0.30；E 独立 extra RGC 120 条（40/scale）。Development 24 条；test 40 条（五尺度各 8），其中 primary 用 16 条全尺度保留序列。
- H/BC 每层固定观察 5 个节点，BC 两分支；Gaussian sigma=0.03，首次生成后不重抽。RGC 使用严格过去 teacher events 条件下的 Bernoulli 样本。
- 每 fit 180 updates、480 microbatches、1,920 次序列曝光；batch=4。Adam lr=0.01、betas=(0.9,0.999)、eps=1e−8、weight_decay=0，梯度范数裁剪 1.0；只取 step 180，无选择或延长训练。
- H/BC/R loss 系数保持 0.4/0.4/1⁄3，按 G2 有效 scalar→序列→batch 归约，多 microbatch 梯度累加不再除以数量。CPU float32、归约 float64、每进程 1 个计算线程；最多 6 个训练进程并行。

| 条件 | 内容 | 每 fit H / BC / R microbatches |
|---|---|---:|
| A RGC_ONLY | 原 base RGC，重复补足曝光 | 0 / 0 / 480 |
| B MULTILEVEL_NARROW | 三层、全部 common scale | 150 / 150 / 180 |
| C MULTILEVEL_MULTISCALE_JOINT | 三层、冻结多尺度集合 | 150 / 150 / 180 |
| D MULTILEVEL_MULTISCALE_PROGRESSIVE | 与 C 相同有序数据流，progressive + rehearsal | 150 / 150 / 180 |
| E EXTRA_RGC_MULTISCALE | 180 base R + 300 独立 extra R batches | 0 / 0 / 480 |

B/C 的标签量和有序 batch 索引相同，只改变尺度分配。D：1–30 步只更新 tau_h；31–60 步更新 a_h/tau_f/tau_s/alpha；61–180 步全部解冻，H、BC 各一次，阶段奇数步 R 一次、偶数步两次。使用同一 Adam，冻结时 grad=None，解冻保留已有状态。A/B/C/E 保持 G2 原日程。

E 仍为曝光匹配、**非 information-matched control**；B/C/D 每 fit 有效 scalar label 曝光 2,332,800，A/E 为 460,800。累计 loss 系数分别为三层各 60 与 R-only 160。等 steps/batches 不代表信息量、梯度强度或 FLOPs 相等。

## 3. 评价与汇总规则

全部 75 份最终 checkpoint 锁定后才开始任何 student 保留集评价。每 world 在读取 test targets 前记录 `TEST_CONSUMED.json`；一次最终评价后仅允许本轮数值核验，不作结果后重跑。

Primary 为 G2 原定义：在同一 teacher 过去事件历史下，对 `BLOCK_DIRECT_BC_DRIVE` 的 `delta_ell = ell_block−ell_normal` 计算 student−teacher 的最后 240 bins RMSE；先尺度内序列等权，再对两个 held-out scales 等权。干预保留 tonic gE=1 和 BC→AC 支路，从初态重算下游，不能视为 BC-cell silencing。

对每个 world 分别计算三个配对 seed 的 C−B、D−C，再取该 world 的三 seed 等权平均。负值表示左侧条件误差更低，正值表示更高，精确零记为 tie；没有事后容差或成功门槛。方向一致性按 **world 均值的符号** 计数，同时报告各 world 内 seed 符号。World 等权均值、范围只是描述；不提供 p-value/生物 population CI，不把 15 个 seed pair 当作 15 个独立 world。

D/E 报告 held-out RGC excess CE、H1 h、BC s_B/o_B、local d_E 和端到端 direct-BC logit intervention recovery。Latent 使用全节点 clean teacher trace，仅 evaluator 可见；绝对与相对干预误差分别保留。

RF evaluator 完全沿用 G2：各 world 的第一条 common-scale test family 提供相同中心/相位的五尺度历史（前 60 bins），随后对所有模型施加同一 bank 的 25 个 0.10 deg 方形 probe，bin 60 独立 ±0.01，读取 lag 1–30 的 logit 中心差分；events 全零。保存完整 25×30 signed RF。Gain=RMS(R)，空间/时间 profile 分别对 lag/位置平均再 L2 归一化，固定 norm<1e−12 规则；不调整 probe、窗口或投影。RF 不进入训练 loss，跨 world 的 anchor waveform 按相同规则独立生成。

## 4. 逐 world 结果

### 4.1 各条件的 held-out direct-BC logit intervention error

每格是该 world 的三个配对 seeds 均值，保持 G2 的逐序列→逐尺度归约。全部 A–E 均完整运行；主要复核只使用下节的 C−B、D−C。

| World | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| W01 | 0.04407519 | 0.02785756 | 0.03417151 | 0.03958425 | 0.02104495 |
| W02 | 0.01688931 | 0.00858525 | 0.00938835 | 0.01280468 | 0.01252315 |
| W03 | 0.11114803 | 0.11318280 | 0.10146071 | 0.11750512 | 0.11495744 |
| W04 | 0.06478438 | 0.04397366 | 0.04880080 | 0.07662963 | 0.06698291 |
| W05 | 0.03263946 | 0.01899595 | 0.01586359 | 0.01883359 | 0.02892905 |

### 4.2 两个 primary 配对比较与方向一致性

Seed 差值依次对应 `2026091801/02/03`。数值为左侧 error 减右侧 error；“负值 seed 数”只描述该 world 的三个配对起点。

| World | 比较 | world 均值差 | 三个配对 seed 差值 | 负值 seed 数 |
|---|---|---:|---|---:|
| W01 | C−B | +0.00631394 | +0.00593103 / +0.00665537 / +0.00635543 | 0/3 |
| W01 | D−C | +0.00541274 | +0.00053813 / +0.01172613 / +0.00397395 | 0/3 |
| W02 | C−B | +0.00080310 | +0.00709902 / −0.00627592 / +0.00158618 | 1/3 |
| W02 | D−C | +0.00341633 | +0.00425874 / +0.00111153 / +0.00487872 | 0/3 |
| W03 | C−B | −0.01172210 | −0.01241654 / −0.01145225 / −0.01129750 | 3/3 |
| W03 | D−C | +0.01604441 | +0.01906332 / +0.00521084 / +0.02385908 | 0/3 |
| W04 | C−B | +0.00482714 | +0.00737994 / +0.00212676 / +0.00497474 | 0/3 |
| W04 | D−C | +0.02782882 | +0.03214712 / +0.02346375 / +0.02787560 | 0/3 |
| W05 | C−B | −0.00313235 | −0.00115981 / −0.00497542 / −0.00326184 | 3/3 |
| W05 | D−C | +0.00297000 | +0.00391475 / +0.00391636 / +0.00107888 | 0/3 |

| 比较 | world 均值为负 / 正 / 零 | world 等权均值差 | world 均值差范围 | 三个 seeds 全为负的 world 数 |
|---|---|---:|---|---:|
| C−B | 2 / 3 / 0 | −0.00058205 | [−0.01172210, +0.00631394] | 2/5 |
| D−C | 0 / 5 / 0 | +0.01113446 | [+0.00297000, +0.02782882] | 0/5 |

C−B 的跨 world 均值略负，但五个 world 的方向混合；W02 的 seed 方向也混合。D−C 在五个 world 和各自全部三个 seeds 上均为正，即本批固定运行中 D 的 primary error 高于 C。G2 中两个比较均为负的方向没有在这五个新 world 中共同保持。这是事前指定采样规则下的数值观察，不是生物 population 显著性推断，也不证明其他 teacher 分布或预算下的结论。

来源：[world_primary.csv](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g3_robustness_20260918/world_primary.csv)、[primary_paired_seeds.csv](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g3_robustness_20260918/primary_paired_seeds.csv)、[summary.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g3_robustness_20260918/summary.json)。没有将 G2 world 加入这些方向计数。

### 4.3 D/E 的响应、latent 和干预恢复

以下为每个 world 的 held-out scales、三 seeds 均值。Excess CE 单位 nats/bin；h、s_B、o_B 和 d_E 是全部节点 clean trace 的 RMSE，s_B/o_B 的两个 BC 分支等权。它们不与训练中有限节点 noisy observation loss 混同。

| World | 条件 | RGC excess CE | H1 h RMSE | BC s_B RMSE | BC o_B RMSE | local d_E RMSE | logit intervention RMSE | 相对 intervention error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| W01 | D | 0.000156029 | 0.00344256 | 0.00284575 | 0.00254834 | 0.00194215 | 0.03958425 | 0.399224 |
| W01 | E | 0.000029154 | 0.02670484 | 0.01197824 | 0.01139658 | 0.00102438 | 0.02104495 | 0.212120 |
| W02 | D | 0.000085330 | 0.00168538 | 0.00278726 | 0.00230543 | 0.00060573 | 0.01280468 | 0.194280 |
| W02 | E | 0.000004129 | 0.00487735 | 0.01469736 | 0.01061153 | 0.00063783 | 0.01252315 | 0.189967 |
| W03 | D | 0.009379383 | 0.00014080 | 0.00474006 | 0.00533368 | 0.00576468 | 0.11750512 | 2.595990 |
| W03 | E | 0.002757419 | 0.01262300 | 0.01131151 | 0.00817544 | 0.00554182 | 0.11495744 | 2.533443 |
| W04 | D | 0.000187206 | 0.00004371 | 0.00362393 | 0.00269240 | 0.00370160 | 0.07662963 | 0.346346 |
| W04 | E | 0.000047719 | 0.01024667 | 0.00997481 | 0.01694051 | 0.00322664 | 0.06698291 | 0.309708 |
| W05 | D | 0.000048033 | 0.00029353 | 0.00298559 | 0.00354935 | 0.00092580 | 0.01883359 | 0.182375 |
| W05 | E | 0.000043615 | 0.01359711 | 0.00732292 | 0.01099427 | 0.00140505 | 0.02892905 | 0.280329 |

相对 intervention error 按逐序列 RMSE / teacher effect RMS 后再平均；不是两个跨 world 均值相除。W03 的较大误差原样保留，没有剔除或延长预算。H1 state、BC state/output、local transmission 和端到端 intervention 分别记录，不用 latent 恢复代替主要干预结论。

完整精度见 [D_E_metrics.csv](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g3_robustness_20260918/D_E_metrics.csv) 及同名 JSON。所有 A–E 的分尺度、observed/unobserved-node、未见 layer×scale 和跨 seed ambiguity 也保留在完整 CSV/JSON 中；这些沿用 G2 的描述性输出，不增加 G3 primary。

### 4.4 RF secondary

以下是 D/E 的每 world 结果，各格对 5 个固定 scale histories × 3 seeds 等权平均。Gain error 先逐 model/history 取绝对值后平均，因此不一定等于表中两个 gain 均值之差的绝对值。

| World | 条件 | teacher gain | student gain | gain 绝对误差 | gain 相对误差 | normalized spatial L2 | normalized temporal L2 | 完整 RF RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| W01 | D | 0.00168854 | 0.00132248 | 0.00036606 | 0.217508 | 1.455679 | 0.324818 | 0.00071978 |
| W01 | E | 0.00168854 | 0.00157497 | 0.00011357 | 0.067449 | 0.652053 | 0.169925 | 0.00034770 |
| W02 | D | 0.00058660 | 0.00060551 | 0.00007377 | 0.119374 | 1.265735 | 0.556654 | 0.00029177 |
| W02 | E | 0.00058660 | 0.00106205 | 0.00047545 | 0.892827 | 0.112855 | 0.183815 | 0.00053676 |
| W03 | D | 0.00077361 | 0.00161434 | 0.00084073 | 1.086237 | 1.895995 | 0.676766 | 0.00118535 |
| W03 | E | 0.00077361 | 0.00108580 | 0.00031219 | 0.403080 | 1.902481 | 0.863861 | 0.00084574 |
| W04 | D | 0.00373546 | 0.00324344 | 0.00051653 | 0.139031 | 0.087199 | 0.129351 | 0.00075156 |
| W04 | E | 0.00373546 | 0.00353405 | 0.00025256 | 0.065399 | 0.052990 | 0.119076 | 0.00056995 |
| W05 | D | 0.00220981 | 0.00205640 | 0.00023304 | 0.105286 | 0.135241 | 0.146256 | 0.00035806 |
| W05 | E | 0.00220981 | 0.00225874 | 0.00007981 | 0.036921 | 0.109849 | 0.184068 | 0.00043855 |

全部 profiles 均有定义，未触发 norm<1e−12；固定 signed projection 下的较大 L2 距离也原样保留。没有根据 RF 结果更换投影、探针、时间窗或阈值。Gain、空间 profile、时间 profile 分别报告，不合并为新胜负指标。

[world_rf_secondary.csv](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g3_robustness_20260918/world_rf_secondary.csv) 包含所有 A–E 的 25 个 world×condition 汇总；[rf_metrics.csv](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g3_robustness_20260918/rf_metrics.csv) 保留 375 条逐 seed/history 结果，包括 cosine 与相对 common-history 的 gain/profile change。完整 normalized profiles、teacher/student 25×30 signed RF 分别位于每个 world 的 `rf_profiles.json` 与 `evaluation/*_rf.pt`。

## 5. 核验、交付与停止边界

五 world 全部合同冻结于 `2026-09-18 03:05:59 UTC`；75 个最终 checkpoint 统一锁定于 `03:26:36 UTC`；首次 student 评价始于 `03:26:45 UTC`。每 world 的 test consumption marker 均先于其 test target 载入；全轮评价完成于 `03:35:27 UTC`。后续 forward replay 仅为同一批已消费结果的核验，没有重新训练或重新选择样本。

| 核验 | 实际结果 |
|---|---|
| 冻结来源 | G1 三文件、依赖、G2 runner、G2 protocol 指纹保持；G3 没有修复后重试 |
| G1 验收 | 复用 G2 已通过的 5 组验收及相同源码指纹；本轮未重复运行这套未改动的测试 |
| 多 world 合同 | teacher sets、manifest、schedule 先于所有目标固定；跨 world family/波形互异；B/C common anchor 逐位一致；E extra family 不重合 |
| 训练日程 | 所有 world 的 schedule 索引与 G2 完全一致；三个 student 初值哈希与 G2 相同 |
| 采样因果性 | 五 world sampler 对 frozen circuit 的概率最大误差分别为 3.7253e−8 / 2.2352e−8 / 5.9605e−8 / 3.7253e−8 / 3.7253e−8；每个事件决策完全一致 |
| 完整预算 | 75/75 checkpoints；13,500 updates；36,000 microbatches；144,000 次序列曝光；15/15 worker 正常退出，仅一次尝试 |
| 优化器与参数 | 每 fit 的 Adam 设置、冻结/解冻 state-step 计数符合 G2；参数有限，全部 12 raw 参数实际改变 |
| Teacher replay | 各 world 全 40 条 test 的保存 trace 与正常/干预前向逐位一致 |
| Student replay | 每个最终 checkpoint 的 4 条 0.225 held-out 序列 normal/block logit/probability 重放最大误差为 0 |
| CSV/JSON 复算 | 逐序列 trace→per-fit、配对比较、world mean 和方向计数一致 |
| 独立数值核对 | NumPy 独立重算全部 3,000 条 sequence 的 primary、NLL/excess CE、h/s_B/o_B；最大差 2.2205e−16 |
| RF 核验 | 全部 375 条 RF metric 从保存 raw RF 重算一致；各 world teacher common-history RF 逐位重放一致 |
| 工件保护 | 已生成数据、manifest、schedule、checkpoint 与来源指纹保持；训练未载入 evaluator-only test/latent/teacher checkpoint |

证据：[verification.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g3_robustness_20260918/verification.json)、[DATA_CONTRACT_VERIFICATION.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g3_robustness_20260918/DATA_CONTRACT_VERIFICATION.json)、[TRAINING_OUTCOMES.json](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g3_robustness_20260918/TRAINING_OUTCOMES.json)，以及 `worlds/W01`–`W05` 各自的 `verification.json` / `INDEPENDENT_VERIFICATION.json`。技术状态 VERIFIED 不意味着科学方向通过；不根据本轮结果补设恢复门槛。

交付工件均在运行目录内：

| 工件 | 内容/行数 |
|---|---|
| `protocol.json`、`SOURCE_LOCK.json`、`PREFLIGHT.json` | 完整五 world 合同、teacher 全精度 draw、来源指纹、目标生成前检查 |
| `worlds/Wxx/` | 独立 protocol、manifest、训练/评价数据、初值、15 个 final model/Adam checkpoints、训练轨迹、原始 trace/RF、核验与哈希清单 |
| `per_sequence_metrics.csv/json` | 3,000 行，各 world/条件/seed/test sequence |
| `per_fit_metrics.csv/json` | 600 行，各 world/条件/seed/stratum |
| `primary_paired_seeds.csv/json` | 30 行，仅两个 primary 对比 × 五 world × 三 seeds |
| `world_primary.csv/json`、`world_metrics.csv/json` | 10 行主比较、25 行 A–E 的 world 汇总 |
| `D_E_metrics.csv/json` | 10 行 D/E 的逐 world 恢复结果 |
| `paired_comparisons.csv/json` | 150 行，两个对比的 G2 指标明细；primary 由单独的 30 行文件明确限定 |
| `ambiguity.csv/json`、`development_metrics.csv/json` | 3,000 / 1,800 行 G2 兼容的描述结果 |
| `rf_metrics.csv/json`、`world_rf_secondary.csv/json` | 375 行逐 seed/history、25 行逐 world/condition |
| `summary.json` | 两个主比较的 world 方向一致性及 D/E 完整均值 |
| `FILE_MANIFEST.json` | 自身以外 675 个工件文件的大小与 SHA256；包含源码快照和各 world 的独立清单 |

[完整文件清单](D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g3_robustness_20260918/FILE_MANIFEST.json)。CSV 的同名 JSON 镜像保留列值文本；`world_*.json`、`D_E_metrics.json`、`summary.json` 和 protocol 提供数值类型的汇总/合同。

本轮新增隔离运行器、本报告和上述 G3 工件，并更新 `docs/NEXT_TASK.md` 保留 G2/G1/G0 历史。未修改 G1/G2 源码或旧结果、正式模型、loss、budget、scale split、RF evaluator；未执行 Git、切分支、真实数据训练或额外 condition。所有 teacher/world 是预定工程分布下的 synthetic 实例，且同时改变 teacher 与数据 draw；本轮不拆分二者的贡献，不作生物 population 推断，也不把 180 updates 称为收敛。完成后停止，没有下一阶段研究决策或自动任务。
