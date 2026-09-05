# Canonical V1 illusion history-semantics audit

日期：2026-08-31。范围：当前冻结的 22 个 N=1 final checkpoints；仅推断审计，无训练、无生产代码修改、无 checkpoint 修改。

## 判定

**PASS：在同一模型、对齐的时间轴和相同完整外部 spike history 条件下，paired logit 及其 H1/direct-BC/AC clamp 差分具有 history 不变性。**

这不是所有 novel-stimulus inference 语义均通过。逐刺激 logit、paired probability 幅度以及由模型自行生成不同 history 的响应不具有相同保证。既有 float32、1e-9 分类标签存在近零数值敏感性，见下文；未修改任何报告阈值。

本结论不改变全局 Architecture Audit 的 FAIL，也不撤销当前 22-cell applicability 的通过状态。不证明不同训练解具有稳定机制结论，不复现真实数据 NLL，不作生物通路真实性判断。

## 源码推导与解释边界

在固定 checkpoint 和 clamp 模式 c 下，生产实现可写成：

\[
z_t^c(S;y)=U_t^c(S)-\gamma H_t(y),\qquad
\gamma=g_{\mathrm{history}}\,\mathrm{history\_gain}.
\]

其中 U 包含 membrane、adaptation 和 response bias；H 是 observed_counts 右移一 bin、零起始后得到的因果低通状态。observed_counts 不进入 H1、BC、AC、total current、divisive state、membrane 或 adaptation。H1/direct-BC/AC clamp 不改变 history gate。

源码依据：

- [model.py:141](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/model.py:141)：刺激通路与 RGC 调用；observed_counts 只在 RGC 调用处传入。
- [rgc_state.py:62](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/rgc_state.py:62)：刺激状态与 history 分开计算，history 作为 final logit 的加性负项。
- [state.py:96](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/state.py:96)：history 的一 bin 移位和低通。
- [pathway_gates.py:87](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/pathway_gates.py:87)：三个目标 clamp 均保留相同 history gate。

对 A、B 施加相同完整 history，包括相同序列起点和前缀，则在实数运算中：

\[
\Delta z_t^c(A,B;y)=z_t^c(A;y)-z_t^c(B;y)
=U_t^c(A)-U_t^c(B).
\]

因此 paired clamp-minus-normal 差分也不依赖共享 history。单刺激 clamp-minus-normal 在 clamp 与 normal 使用相同 history 时同样相消。对这些差分作固定时间窗平均或积分，代数性质不变。这个对任意共享 history 的结论来自源码结构；有限个 runtime histories 只验证实现与推导一致。

这给当前 zero-history replay 一个明确解释：**固定外部 history 条件下的刺激或通路干预对比**。它不是对未知 spike history 边缘化后的平均响应，也不是自由生成 spike 的模拟。固定 history 的 clamp 对比不包含 clamp 改变 spikes 后再改变未来 history 的反馈效应。

边界条件：

1. 逐刺激 z 随 history 平移。不能把单条 zero-history logit 当成与 history 无关的绝对响应。
2. p=sigmoid(z)。共享平移通常改变 p_A−p_B 的幅度；精确算术下单 bin 的 A/B 大小顺序由 sigmoid 单调性保留，但跨时间平均的概率差没有普遍不变性保证。probability 层的 clamp 差分也无相消保证。
3. 自回归条件下 A、B 通常具有不同的输出 history，出现额外项 −γ[H(y_A)−H(y_B)]。本轮未实施自回归推断，相关响应 **UNVERIFIED**。
4. Mach 的空间 profile/plateau 对比要求全部位置使用相同 history；response-minus-blank 要求刺激与 blank 使用相同 history。不能只在单个配对内相同，却在组合 profile 的位置间不同。

## 冻结对象与探针

直接读取原始 checkpoint 的 model_config，确认 architecture_mode=mechanism_identifiable，并 strict-load 原始 state；不使用旧 application loader 的 mode override。

- Checkpoints：[22-cell lineage](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830)。每个是独立的 N=1 模型。
- 输入：[原有 inputs.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_frozen_applications_20260830/illusion/inputs.pt)，cone_drive 为 72×150×289，history 为 72×150×1；保存的 history 全零。未重新生成视觉刺激。
- 35 个原有配对：25 个 Mach 位置配对；SBC、Hermann、White 及 White/Hermann diagnostic 的目标与匹配控制配对，共 10 个。另包含原有 blank traces。
- 四个模式：normal、H1_off、direct_BC_off、AC_off。
- 三种外部 history：saved_zero；索引 t mod 11=3 时为 1 的 periodic_11；每 bin 为 1 的 dense_one。各 history 在全部 72 条刺激间相同。dense_one 是有界二值压力输入，不是生理放电率估计。
- 主指标：逐 bin paired logit、paired clamp-minus-normal 差分；原有 300≤t<400 ms 窗口均值。
- float32 覆盖全部 22 个模型。float64 使用四组中保存的 |history gate×gain| 最大者，平手按路径顺序选首个：67#33 parasol_OFF、67#4 midget_OFF、67#7 parasol_ON、70#7 midget_ON。仅在内存中提升同一已保存参数的精度。

在观察输出对比之前，float64 代表选择从每组首个调整为每组最大 history 系数，以免所有代表都落在 history 关闭的模型；这是 metadata-only 调整，不改变全 22-cell 主分析。规则保存在 [PROTOCOL.md](D:/PythonProject/retina_rf_SNN/output/audits/illusion_history_semantics_20260831/PROTOCOL.md)。

当前 18/22 的 history gate 精确为零，另外四个为：

| cell | group | saved history gate |
| --- | --- | ---: |
| 67#26 | midget_ON | 0.00002204498741775751 |
| 67#4 | midget_OFF | 0.1554955542087555 |
| 70#15 | midget_ON | 0.6091353893280029 |
| 70#7 | midget_ON | 1.0 |

四者的 saved history_gain 均约为 0.02，history_decay 约为 0.80073738。因此通过不只是由所有 history gate 为零造成。

## 独立运行结果

预设数值容差为 B=32×eps(dtype)×(1+max|z_zero|+max|z_changed|+max|history term|+max|bias|)，每个 cell/mode/history 分别计算。clamp-minus-normal 使用 B_clamp+B_normal。它是保守审计容差，不是严格最紧浮点误差定理，也不是生物效应阈值。

| 观测量 | float32，22 cells | float64，4 组代表 |
| --- | ---: | ---: |
| paired logit 最大 history 残差 | 2.384185791015625e-7 | 4.440892098500626e-16 |
| paired clamp-minus-normal 最大残差 | 4.76837158203125e-7 | 6.661338147750939e-16 |
| paired logit 最大残差/对应容差 | 0.011111 | 0.012626 |
| paired logit mean_on 最大变化 | 8.754432201385498e-8 | 7.401486831696023e-17 |
| 超出数值容差、两端均可区分于零的 mean_on 符号反转 | 0 | 0 |
| 单刺激 logit 最大变化 | 0.020000100135803223 | 0.019999999552965164 |
| paired probability 逐 bin 最大变化 | 0.00043682754039764404 | 0.00043682997005242696 |
| paired probability mean_on 最大变化 | 0.00017153937369585037 | 0.00017153396679116598 |

176 组 float32 和 32 组 float64 共享 history 对比全部通过；每组检查所有 35 个配对、全部 150 bins。156 组 paired clamp-minus-normal 从保存的 NPZ 独立复算后全部通过。10 个上游 current/state tensors 在 history 改变前后逐位相同；相同 history 下匹配控制配对保持精确零。logit 的绝对平移与显式 history 项一致。

**数值报告限制：** 在 6,160 条 float32 logit 配对比较记录中，有 3 条 mean_on 原始符号反转、25 条按既有 ±1e-9 规则得到不同分类；这里的记录单位是 cell×mode×非零 history×pair，并不是独立样本或细胞数。上述受影响均值的绝对值最大仅 3.1789145538141383e-8。不能将这些近零标签变化解释为 pathway necessity 的变化。93 条 post-onset 绝对峰值索引改变；argmax 对近乎并列的值敏感，不能据此把峰时的所有数值标签也宣布稳定。四个 float64 代表无 mean_on 分类改变，仍有 1 条 logit 峰值索引变化。

这些观察不否定加性相消公式；它们说明 float32 下先相减再分类不等于精确算术。现有 1e-9 报告阈值与未来 mechanism-stability 的科学判据尚需区分，本轮未改阈值、未替用户确定新的 effect-size 标准。

**非共享 history 对照：** 最大 history 系数模型 70#7，normal 模式，原有 SBC 配对，偶数 stimulus 索引用零 history，奇数用 dense_one。paired logit 改变约 0.02，明显超过该对照自己的容差；与预期 −γ[H_A−H_B] 的误差为 float32 1.0058283805847168e-7、float64 2.0469737016526324e-16。该对照说明共同 history 的要求不能无条件省略；非零 γ 时，不同 history 可以改变 signature。γ=0 的 18 个模型不受该项影响，共同 history 并非所有个例中数学上的必要条件。该对照不作为新推断协议。

原有指标实现依据：[metrics.py:15](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_r4_dev_visual_illusions_20260830/metrics.py:15)（时间窗和峰值）、[metrics.py:41](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_r4_dev_visual_illusions_20260830/metrics.py:41)（paired 及 paired clamp-minus-normal）、[metrics.py:52](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_r4_dev_visual_illusions_20260830/metrics.py:52)（Mach profile）、[metrics.py:88](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_r4_dev_visual_illusions_20260830/metrics.py:88)（1e-9 分类）。本轮复算指定主指标及附属数值检查，未逐项重跑原有全部报告指标；其他线性指标与 profile 平移性质的结论来自上述代数推导。

## 完整性与证据

194 个相关生产/application 源文件、冻结输入、22 个 checkpoint 及原有 frozen replay 文件的前后 SHA256 一致。每次精度运行前后 model state 逐位相同；parameter gradients 均不存在。没有 optimizer、训练、重训或新增训练 checkpoint。float64 临时模型未保存回 lineage。

- [SUMMARY.json](D:/PythonProject/retina_rf_SNN/output/audits/illusion_history_semantics_20260831/SUMMARY.json)：汇总判定及全部精确数值。
- [checkpoint_identity.json](D:/PythonProject/retina_rf_SNN/output/audits/illusion_history_semantics_20260831/checkpoint_identity.json)：22 个 checkpoint 的 SHA256、组别与 history 参数。
- [dod_recomputed.json](D:/PythonProject/retina_rf_SNN/output/audits/illusion_history_semantics_20260831/dod_recomputed.json)：156 组差分独立复算。
- [numerical_category_changes.json](D:/PythonProject/retina_rf_SNN/output/audits/illusion_history_semantics_20260831/numerical_category_changes.json)：近零分类变化及原始均值。
- [integrity_before.json](D:/PythonProject/retina_rf_SNN/output/audits/illusion_history_semantics_20260831/integrity_before.json) / [integrity_after.json](D:/PythonProject/retina_rf_SNN/output/audits/illusion_history_semantics_20260831/integrity_after.json)：冻结对象完整性。
- 每个 cell 的 *_checks.json、*_pairs.csv、*_responses.npz 保存分项检查、配对指标以及全部 raw logits/probabilities。
- [audit_runtime.py](D:/PythonProject/retina_rf_SNN/output/audits/illusion_history_semantics_20260831/audit_runtime.py)、[history_metrics.py](D:/PythonProject/retina_rf_SNN/output/audits/illusion_history_semantics_20260831/history_metrics.py)、[summarize.py](D:/PythonProject/retina_rf_SNN/output/audits/illusion_history_semantics_20260831/summarize.py)：独立探针与汇总逻辑。

独立只读审阅修正了数值容差的绑定：DoD 用 normal 与 clamp 容差之和，非共享 history 对照使用其自身容差并要求 observed/predicted change 都超过它。所有 DoD 均从已存响应按修正后的公式复算，非共享 history 产物包含并通过修正后的判据。没有改变主分析的共享 history、视觉输入或模型。一次有界运行在尚未保存下一 cell 时超时；只补跑该未完成 cell，其余已完成证据保留。

复核命令（在 repository root；START/STOP 为排序后 checkpoint 索引，半开区间，建议每次不超过 3 个）：

```text
PYTHONDONTWRITEBYTECODE=1 D:/anaconda/python.exe -B output/audits/illusion_history_semantics_20260831/audit_runtime.py START STOP
PYTHONDONTWRITEBYTECODE=1 D:/anaconda/python.exe -B output/audits/illusion_history_semantics_20260831/summarize.py
```

## 对冻结研究顺序的影响

history-semantics 这一关在 **shared-history paired-logit estimand** 上通过；zero-history 可以继续作为这一条件化对比的固定约定。现阶段没有由 history 加性项引出的架构修改或重训理由。

real-data mechanism stability、独立 NLL/result reproduction 和 benchmark closure 本轮均未启动。predictive-equivalent fits 是否保持相同 pathway 结论仍为 **UNVERIFIED**；下一阶段的 representative cells、fresh seeds、prediction-equivalence 和 mechanism-stability 标准仍需在查看新结果之前锁定。H1/BC/AC 是否对应真实生物通路仍为 **UNVERIFIED**。
