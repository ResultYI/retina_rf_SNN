# RetiPath frozen nonlinear spatial-integration pilot

**日期：2026-09-16；结论：MODEL_NONLINEAR_INTEGRATION_PRESENT。**

仅使用 67#6（OFF parasol）和 67#7（ON parasol），各 seed 2026091301。canonical checkpoint 在 float64 inference copy 中运行。无训练、参数更新、其他 cells/seeds、DoG/radius fitting、external reference、natural movie 或 spike target 访问；未运行 pathway clamp。

## 冻结方法与数据

150 bins / 150 Hz；[0,30) 零 Weber 背景，[30,90) 空间 pattern 恒定，[90,150) recovery；FIX_HISTORY_ZERO。保持原动态初态：V1/V2 从 V0=2/9 开始，其余动态偏差从零开始。R 是 [75,90) logit 均值减去同时间窗的零刺激 logit 均值。不是 unconditional spike prediction。

`protocol.json` 在任何本轮模型响应前写入；两个 cell 的 `stimuli.npz` 在任何本轮 finite-contrast response 前写入。

- Protocol SHA256：`156f7aac614b2404daeaea264bafed81b9fa0a4877605e7aa42c029d8a91aa18`。
- Stimuli SHA256：`de0d3e36c982ae7354a054b892e448a6a9c3947d966b3d327993a3cf18314a8c`。
- [原始数据目录](D:/PythonProject/retina_rf_SNN/output/evaluations/retipath_nonlinear_spatial_integration_pilot_20260916)；[分析脚本](D:/PythonProject/retina_rf_SNN/work/retipath_nonlinear_spatial_integration_pilot.py)。

| Cell | Canonical checkpoint SHA256 |
|---|---|
| 67#6 | `e55e82f64fcaa3fa78f9478d738b2623be2d907b23ef1a024445125d63e42c31` |
| 67#7 | `5283d88100f7b44c6e9f9d711489514767426480ce6c9713bb6bc22c5f09293d` |

对 289 个 pixel 分别施加 ±0.01 Weber，时间 envelope 与主响应完全相同，计算 `k_p=[R(+epsilon e_p)-R(-epsilon e_p)]/(2 epsilon)`。这是固定 epsilon、固定 operating point 的逐 pixel 对称差分线性代理；不假定 kink 上存在唯一常规 Jacobian。

每个 cell 使用独立 PCG64 随机序列（67#6：2026091606；67#7：2026091607），标准正态候选投影两次到 span{k,ones} 的正交补，再归一化为 RMS=1。事前拒绝条件仅为投影 RMS<1e-12 或归一化 max|u|>4；最多 1000 候选。两 cell 均直接接受候选 0–5，无拒绝。未用 finite-contrast response 筛 pattern。aligned control 为 k/RMS(k)，也测正负符号。

固定 C0=0.9340826124 Weber；幅度为 0.2335206531、0.4670413062、0.9340826124 Weber spatial RMS。幅度不是最大 pixel contrast。每 cell 共 43 个 sequence：36 null、6 aligned、1 zero；两个 cell 共 86 行 responses。

`R_lin=k^T x`，`N=R_full-R_lin`；primary 为每 cell、每 amplitude 的 12 个 null 刺激平均 |N|。Odd/Even 先按每个 ± pair 计算；responses.csv 中同一 pair 的两行重复保存 pair-level Odd/Even，而不是随行符号重新取反。

trace 先对 [75,90) 求时间均值并减去匹配零刺激，保留所有 pixel/mode/pathway 分量，再逐分量计算 Even/Odd。最后才计算有符号平均值与 component RMS。因此不会把 unsigned norm 错当成线性 H1 的 Even response，也不会因 spatial sum 抵消而漏掉 BC 非线性。

`traces.npz` 保存 full trace、`late_R`、`late_even`、`late_odd`、case 索引和单位/定义 metadata。H1 full trace 为 [43,150,289]；BC direct/broad 和 AC 为 [43,150,1,2,2]（末轴 sustained/transient 或 local/transient）；E/I 与 conductance 为 [43,150,1,2]；V1/V2、membrane、adaptation、logit 为 [43,150,1]。不使用 legacy uE/uI。

## 必要 sanity 与实际 contrast 范围

| Cell | max abs(kᵀu) / abs(kᵀv) | max abs(mean(u)) | ± RMS 差 | 最小 amplitude aligned 方向 |
|---|---:|---:|---:|---|
| 67#6 | 7.627e-17 | 4.303e-17 | 0.0 | 正负均符合 |
| 67#7 | 2.083e-17 | 1.844e-17 | 0.0 | 正负均符合 |

unit RMS 最大误差为 1.11e-16；所有 forward 有限，history state 为零。checkpoint SHA256、manifest、模型/trace 源码均未改变；同一刺激重放前后的 inference state_dict 逐 tensor 相同。

工程记录：首次 forward 已完成并保存所有刺激、response、full trace，summary 导出被 NumPy bool 的 JSON 序列化阻断。随后只重放相同已保存刺激以补存状态检查；第二次导出遇到数组布局导致的均值归约差（≤2.3e-16），按 float64 误差处理后完成导出。没有新 kernel、pattern、amplitude 或独立 checkpoint；共加载同两个 checkpoint 6 次（原推理 2 次，导出修复中的相同刺激重放 4 次）。重放 trace 与初次保存值最大差分别为 3.69e-15、8.89e-16。所有定量结果沿用初次保存 response/full trace，原始文件未覆盖；protocol 原文与 hash 未改。

| Cell | C/C0 | null pixel Weber 范围 | aligned pixel Weber 范围 | null pixels < −1 |
|---|---:|---|---|---:|
| 67#6 | 0.25 | [-0.887236, 0.887236] | [-1.167957, 1.167957] | 0.00% |
| 67#6 | 0.50 | [-1.774473, 1.774473] | [-2.335913, 2.335913] | 1.67% |
| 67#6 | 1.00 | [-3.548945, 3.548945] | [-4.671827, 4.671827] | 14.07% |
| 67#7 | 0.25 | [-0.891512, 0.891512] | [-1.009392, 1.009392] | 0.00% |
| 67#7 | 0.50 | [-1.783025, 1.783025] | [-2.018785, 2.018785] | 1.70% |
| 67#7 | 1.00 | [-3.566050, 3.566050] | [-4.037570, 4.037570] | 13.90% |

**范围限制：**0.25 C0 的 null patterns 均在物理 Weber 下限 −1 以上；0.5/1 C0 的部分 null pixels，以及 aligned controls 的部分 pixels 低于 −1，不能对应非负亮度。这些条件是代数模型输入，不可当作可直接呈现的亮度刺激。按冻结要求未 clip、重新缩放或筛除。低 amplitude 已有可分辨残差；高 amplitude 结果仍受此限制。

## 1. Null-space stimuli 是否产生可分辨的 finite response？

是，数值上可分辨。以下均为 baseline-subtracted late logit，单位为 log-odds；不是生理效应量或统计显著性阈值。

| Cell | C/C0 | mean abs(N) | mean Even | mean abs(Odd) | max abs(R_lin) |
|---|---:|---:|---:|---:|---:|
| 67#6 | 0.25 | 0.000203298 | -0.000203298 | 0.000186112 | 1.32e-16 |
| 67#6 | 0.50 | 0.000803045 | -0.000803045 | 0.000736065 | 2.64e-16 |
| 67#6 | 1.00 | 0.003108597 | -0.003108597 | 0.002842805 | 5.27e-16 |
| 67#7 | 0.25 | 0.003311362 | -0.003311362 | 0.001364527 | 1.11e-16 |
| 67#7 | 0.50 | 0.014084545 | -0.014084545 | 0.007204131 | 2.22e-16 |
| 67#7 | 1.00 | 0.062063835 | -0.062063835 | 0.040703059 | 4.44e-16 |

最小单个 |N| 为 4.55e-7 logit，远高于本次 float64 重放差；最大线性残留仅 5.28e-16。零刺激 baseline subtraction 为零。aligned control 在最低固定 amplitude 下，67#6 的响应为 +4.35994/−0.91248（线性预测 ±2.71955），67#7 为 +2.81066/−3.41594（线性预测 ±4.35539）；方向检查通过，不把有限 amplitude 下的幅度差异当作线性恢复。

## 2. Nonlinear residual 是否随 contrast 系统变化？

是。两 cell 的 mean |N| 随 0.25→0.5→1 C0 严格增加；每个 cell 的 12/12 pattern/sign 序列也都严格增加。67#6 相邻倍数约 3.95、3.87；67#7 约 4.25、4.41。这里只描述冻结三点，不额外拟合幂律或新增 contrast。Even 与 Odd 均非零，不能把输出概括成纯 sign-symmetric response。

## 3. 67#6 与 67#7 的方向是否一致？

一致：两 cell 每个 amplitude 均为 12/12 negative late-logit responses（共 72/72）。增大 contrast 后，残差绝对值增大。67#7 的幅度明显较大；本轮只有两个各自带独立 null bank 的 checkpoint，不能据此推断 ON/OFF 或 parasol population 差异。

## 4. Nonlinear component 首先出现在哪个 model stage？

首先出现在 BC direct/broad 的共同分支 PReLU 非对称变换之后。H1 graph/state/feedback/modulated input 的 Even 仅在舍入量级。下表为 1 C0、六个 null pairs 的 mean component RMS(Even)；不同 trace 单位不同，不能按列内数值大小推导 biological importance。

| Trace | 67#6 | 67#7 |
|---|---:|---:|
| h1_feedback | 4.94319e-19 | 7.03271e-19 |
| h1_modulated_input | 1.79323e-18 | 7.55754e-19 |
| bc_direct_effective_drive | 0.0828826 | 0.0351066 |
| bc_broad_effective_drive | 0.0900521 | 0.0333252 |
| ac_states | 0.0900515 | 0.0333234 |
| ac_inhibitory_drive | 0.045123 | 0.115964 |
| effective_drive_E | 0.105877 | 0.259053 |
| effective_drive_I | 0.0652343 | 0.164242 |
| gE | 0.0213724 | 0.0153242 |
| gI | 0.0299135 | 0.0300349 |
| V1 | 0.000388173 | 0.0056721 |
| V2 | 0.000397291 | 0.0056057 |
| membrane_readout | 0.00310364 | 0.0626888 |
| adaptation_state | 0.00052402 | 0.0625004 |
| logit | 0.0031086 | 0.0620638 |

BC 各分支单独保留后也可见这一结果：

| BC branch | 67#6 RMS(Even) | 67#7 RMS(Even) |
|---|---:|---:|
| bc_direct_effective_drive / sustained | 0.104427 | 0.0436327 |
| bc_direct_effective_drive / transient | 0.0532332 | 0.0236882 |
| bc_broad_effective_drive / sustained | 0.113504 | 0.0414197 |
| bc_broad_effective_drive / transient | 0.0577529 | 0.0224842 |

源码依据：[retipath_canonical_gain.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/retipath_canonical_gain.py:128) 中，线性 feature bank 与固定权重先形成整条 BC 分支 S，再由 `s=1 if S>=0 else alpha` 共同作用于该分支各 spatial mode。当前 alpha 为 67#6=0.200417232、67#7=1.366462539，均未修改。分支求和后的 `Even(phi(S))=(1-alpha)|S|/2` 非零；没有逐 mode 独立整流。

BC Even 的有符号方向在两 cell 中相反，但最终 logit 均为负。这说明“最早出现非线性”与“最终 residual 的唯一来源”不同：AC 传播、E/I 相互抵消、softplus conductance mapping 和 voltage dynamics 共同形成后续输出。本轮未做新 intervention，不能把最终 N 全部归因于 BC，更不能将其称为真实 BC subunit 的证据。

## 5. Verdict

**MODEL_NONLINEAR_INTEGRATION_PRESENT。** 两个冻结 RetiPath 在自身局部线性 logit proxy 近似为零、且空间均值为零的刺激上，产生超过数值误差的有限响应；效应随预定 contrast 增大。该结论仅限本轮 model-conditioned functional probe，证明超出自身 local linearization 的 nonlinear spatial integration，不构成真实 BC subunit 或真实视网膜机制验证。高 amplitude 的非物理 Weber 范围限制保留。完成后停止，没有追加 cells/seeds、训练或架构修改。
