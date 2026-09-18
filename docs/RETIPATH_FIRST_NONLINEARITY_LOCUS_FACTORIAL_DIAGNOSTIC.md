# First-nonlinearity locus 2×2 frozen diagnostic

结论：**STRONG_INTERACTION_OR_MIXED**。四-cell 平均即时 relocation penalty 的主要数值来源是 BC nonlinearity removal，尤其在两个 OFF cells；但 ON/OFF 个例方向不同，且两个位置存在非加性交互，不能概括为跨四个 cell 一致的单因素解释。

本轮仅 inference：67#6、67#7、68#3、68#10，seed=2026091301。全部从 canonical migration manifest 解析原始 checkpoint，使用原始冻结 alpha（0.200417、1.366463、0.254063、1.483303），未加载上一轮训练后的 M0/M1 checkpoint。新 checkpoint=0、optimizer updates=0、checkpoint selection=0。A/B/C/D 的全部参数/state_dict 起点相同。

| Condition | pre-H1 | BC common branch | Status |
|---|---|---|---|
| A CURRENT | identity | phi_alpha | 当前模型 |
| B RELOCATED | phi_alpha | identity | 完整 relocation 的冻结诊断 |
| C BOTH_DIAGNOSTIC | phi_alpha | phi_alpha | 仅 diagnostic |
| D LINEARIZED_DIAGNOSTIC | identity | identity | 仅 diagnostic |

phi_alpha(x)=x（x>=0）或 alpha*x（x<0）。C 两处复用同一个冻结 alpha，未增加第二个坐标。模型正式源码、checkpoint、固定 RMS、geometry/support、后端与 history 定义均未改。实验 runner 复用上一轮独立 candidate 与 trace 接口：C 仅给 H1 注册输入变换 hook，D 仅令实验对象的 effective_input 为 identity。

1. **Early addition 本身对 prediction 的即时影响。**

只评价已消费 development [16,20)：150 Hz Bernoulli occupancy，150-bin 独立 sequence、30 warmup/120 scored、原 mask、严格过去 observed history。前向 float32、NLL reduction float64；缓存以 mmap 访问 development 张量，未访问 train 或其他时间块。development stimulus/target/mask/source-ID hashes 与上一轮锁定合同完全一致。NLL 单位为 nats/scored bin，所有差值正值表示损失变差。

| Cell | NLL A | NLL B | NLL C | NLL D |
|---|---:|---:|---:|---:|
| 67#6 | 0.405999 | 0.605122 | 0.428191 | 0.618844 |
| 67#7 | 0.367064 | 0.370376 | 0.375387 | 0.369742 |
| 68#3 | 0.468653 | 0.595412 | 0.497237 | 0.593722 |
| 68#10 | 0.297955 | 0.294343 | 0.300018 | 0.296951 |
| Equal-cell mean | 0.384918 | 0.466313 | 0.400208 | 0.469815 |

保留 BC 非线性时，early addition=C−A：mean **+0.015290**，四个 cells 均变差。没有 BC 非线性时，early addition=B−D：mean **-0.003501**，两个改善、两个变差。因而 early addition 不是一个与 BC 状态无关的固定 penalty。

2. **BC nonlinearity removal 的即时影响。**

| Cell | C−A early addition | B−C BC removal given early | D−A BC removal without early | B−D early addition without BC | Interaction | B−A total |
|---|---:|---:|---:|---:|---:|---:|
| 67#6 | +0.022191 | +0.176931 | +0.212845 | -0.013722 | +0.035913 | +0.199123 |
| 67#7 | +0.008323 | -0.005010 | +0.002678 | +0.000634 | +0.007689 | +0.003313 |
| 68#3 | +0.028584 | +0.098175 | +0.125070 | +0.001689 | +0.026895 | +0.126759 |
| 68#10 | +0.002063 | -0.005674 | -0.001004 | -0.002608 | +0.004671 | -0.003611 |
| Equal-cell mean | +0.015290 | +0.066105 | +0.084897 | -0.003501 | +0.018792 | +0.081396 |

平均 BC removal 为：已有 early 时 **+0.066105**；没有 early 时 **+0.084897**。两者均大于相应平均 early-addition 效应的绝对值，因此平均 penalty 的较大数值部分来自 BC removal。

该结论主要由两个 OFF cells 支撑：67#6 的 B−C/D−A 为 +0.176931/+0.212845，68#3 为 +0.098175/+0.125070。两个 ON cells 的 B−C 反而为负；67#7 的 BC removal 随 early 有无而变号，68#10 的两种 removal 均略有收益。因此不能称四个 cell 一致证明 BC removal 是唯一原因，更不能由 D 表现差验证 BC 生物机制。

3. **是否存在 interaction？存在清楚的模型内非加性交互。**

预定 interaction=(C−A)−(B−D)，四个 cells 全为正，equal-cell mean=**+0.018792**。两个 OFF cells 分别为 +0.035913 和 +0.026895；两个 ON cells 为 +0.007689 和 +0.004671。其平均值大于 C−A 平均值，且 early addition/BC removal 在部分 cells 的方向随另一处状态改变，不能忽略。这里没有显著性检验或 biological interaction 推断。

两条精确分解都保留：B−A=(C−A)+(B−C)=(D−A)+(B−D)。本轮原始权重的平均 B−A=+0.081396；这不是上一轮200-step后约+0.01921 penalty 的数值归因，训练已改变过其他坐标。本轮只能解释固定原始权重下 architecture trade-off 的即时结构。

归类规则在本轮 protocol 中于 response 前固定：只有同一因素在另一因素的两种状态下平均都为正、其较小均值仍超过另一因素的两个均值绝对值，且至少3/4 cells 的两种效应均为正，才称 MAINLY。BC removal 在均值大小上占优，但仅两个 cells 在两种状态下均受损，故最终为 **STRONG_INTERACTION_OR_MIXED**。这是描述性归类，不是效应阈值或显著性判断。

最小 F2 sanity 只使用冻结 bank 中 width=2 pixel、phase=0 和 matched uniform：4 Hz、max Weber amplitude=0.5、8 pre-roll+8 measurement cycles、连续 bin phase、FIX_HISTORY_ZERO。每个 signed component 先 Fourier projection 再 RMS，除以同 trace uniform F1；floor 仍为既有4096*eps64尺度。没有重跑6-condition grid。

| Cell | Condition | H1 feedback F2/uniform F1 | BC direct sustained | BC direct transient | Membrane | Logit |
|---|---|---:|---:|---:|---:|---:|
| 67#6 | A | 3.63e-16 | 0.079077 | 0.079155 | 0.10289 | 0.10303 |
| 67#6 | B | 0.24285 | 0.18046 | 0.2748 | 0.43215 | 0.43273 |
| 67#6 | C | 0.24285 | 0.11023 | 0.16374 | 0.25844 | 0.25879 |
| 67#6 | D | 3.63e-16 | 1.1329e-15 | 8.1051e-16 | 0.00043507 | 0.00043566 |
| 67#7 | A | 2.4944e-16 | 0.0051293 | 0.0051102 | 0.0088831 | 0.0088951 |
| 67#7 | B | 0.055444 | 0.04334 | 0.065304 | 0.11699 | 0.11715 |
| 67#7 | C | 0.055444 | 0.048881 | 0.073702 | 0.13751 | 0.13769 |
| 67#7 | D | 2.4954e-16 | 3.7193e-16 | 1.8482e-16 | 0.00070748 | 0.00070843 |
| 68#3 | A | 2.5619e-16 | 0.04253 | 0.03507 | 0.045304 | 0.045365 |
| 68#3 | B | 0.21719 | 0.1701 | 0.25276 | 0.34432 | 0.34478 |
| 68#3 | C | 0.21719 | 0.10673 | 0.15725 | 0.21451 | 0.2148 |
| 68#3 | D | 2.5621e-16 | 6.9085e-16 | 3.7723e-16 | 0.0001457 | 0.00014589 |
| 68#10 | A | 2.4015e-16 | 0.012889 | 0.0129 | 0.019834 | 0.01986 |
| 68#10 | B | 0.067835 | 0.054104 | 0.07869 | 0.13999 | 0.14018 |
| 68#10 | C | 0.067835 | 0.062288 | 0.090236 | 0.17517 | 0.1754 |
| 68#10 | D | 2.4005e-16 | 7.7428e-16 | 3.4291e-16 | 0.0013802 | 0.001382 |

A：H1 graph/state/feedback 均在 floor，direct/broad BC 均有明确 F2；B/C：H1 均有 F2；D：H1 与 direct/broad BC 均在 floor。D 的膜与 logit 仍有约1.5e-4–1.4e-3的归一化 F2；D 保留 conductance mapping/integration 等下游非线性，不能称整个网络线性，也不能将该残余归因于已移除的 alpha。本轮不进一步定位这些下游贡献。

4. **Natural movie 输入是否发生 distribution shift？有明确 mean/DC shift。**

预定67#6、仅 scored development bins：pre-H1 input mean 从 **0.530575→0.716444**，增加 **0.185869**（相对原均值 +35.03%）；RMS 从 1.803275→1.767607（-1.98%），centered SD 从 1.723453→1.615903（-6.24%）。原始 natural movie 的该窗口均值本来就不是0；此处测到的是额外偏移。alpha>0 保留符号，负值比例不变，但负半轴幅值被缩小。

这不是输入 RMS 暴涨，而是非对称压缩带来的 DC 与分布形状变化；H1 feedback mean 也从0.088690升至0.120377。没有对结果重新 center、normalize 或修正 bias。下表每项为 mean / RMS，归约固定为相同 scored bins 与原始 components：

| Trace | A | B | C | D |
|---|---:|---:|---:|---:|
| pre_h1_input | +0.530575 / 1.803275 | +0.716444 / 1.767607 | +0.716444 / 1.767607 | +0.530575 / 1.803275 |
| h1_feedback | +0.088690 / 0.231267 | +0.120376 / 0.231083 | +0.120376 / 0.231083 | +0.088690 / 0.231267 |
| bc_direct_effective_drive | +0.006956 / 0.447444 | -0.798487 / 1.682449 | -0.135978 / 0.343473 | -0.608796 / 1.678228 |
| bc_broad_effective_drive | -0.006049 / 0.420860 | -0.790144 / 1.628779 | -0.140066 / 0.331050 | -0.603015 / 1.620661 |
| effective_drive_E | +0.064340 / 0.688777 | -1.059053 / 2.512968 | -0.161866 / 0.517326 | -0.801463 / 2.541367 |
| effective_drive_I | +0.018966 / 0.318063 | -0.610360 / 1.215248 | -0.099306 / 0.244681 | -0.469544 / 1.212124 |
| membrane_readout | +0.091107 / 1.908306 | -0.444018 / 4.894285 | -0.022657 / 1.452765 | -0.331642 / 5.244428 |
| logit | -1.465354 / 2.402212 | -1.994577 / 5.259840 | -1.577965 / 2.143355 | -1.883331 / 5.555537 |

B 与 C 的 pre-H1 input/H1 feedback 完全相同；A 与 D 也相同。后续差异来自 BC common-branch transform 是否存在及下游重新计算。B/D 在67#6的 BC/E/I 与 membrane RMS 大幅高于A，C则没有出现同样的幅值扩张；这支持固定权重下的计算分解，不能据此把 penalty 唯一归因于 DC。原始条件之间仍同时存在时空分布与非线性传播差异。

5. **下一步是否还有科学依据设计 distributed early+late mechanism？有有限的研究动机，尚无采用依据。**

早期位置能使 H1-like trace 产生 F2，保留 late BC 非线性又避免了两个 OFF cells 上较大的即时损失，因此“early 的功能位置”和“late 的预测作用”不能通过完整 relocation 互相替代。这个分解为继续思考二者如何分工提供了模型内依据。

但本轮 C−A 在四个 cells 全为正：同一个原始 alpha 同时用于两处并未保持当前 prediction，不能接受 C 为正式 architecture，不能宣称 distributed mechanism 已被支持或生理验证。任何后续设计都仍需独立授权和新的预注册检验，本轮不提出新参数方案、不训练、不启动后续实验。

核查：四个 canonical checkpoint 与全部引用模型源码 hashes 不变；16个 inference 条件 state 不变、所有参数冻结；BC分支公式与 pre-H1 输入语义检查精确通过；A/D、B/C 的 H1 state/feedback 配对精确一致；67#6 trace logit 与正常前向匹配；F2 所有输出有限、history=0、所有 uniform 分母有效。只保存指定CSV/JSON与本文，没有新checkpoint。

输出目录：`output/evaluations/retipath_first_nonlinearity_locus_factorial_20260916/`。`protocol.json` 保存冻结来源与定义；`per_cell_nll.csv` 保存全部预定 contrasts；`minimal_f2.csv` 保存两个刺激的分层谐波；`trace_summary.csv` 保存67#6分布统计；`summary.json` 保存equal-cell mean、sanity和归类。
