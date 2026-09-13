# Aligned Canonical V1 — RF / pathway / temporal identifiability main audit

日期：2026-09-05。**只读审计；新增 training runs=0，optimizer steps=0。**

## 1. Executive conclusion

当前证据最支持的是：**在固定 LN functional alignment、固定空间与动力学模型族、给定 stimulus 和 observed history 的条件下，拟合模型提供可复现的 global logit RF 与非零 H1/direct-BC/AC functional intervention effects。Functional RF 的形状比 pathway magnitude 和 individual temporal parameters 更稳定。**

这允许把 conditional global RF 作为模型主结果展示，但尚不能写成“aligned 22-cell RF 已被独立 seed 验证为唯一恢复”。现有跨 seed 证据只有 **4 个 zero-center cells × 3 fits**：global RF cosine 0.983164–0.997655，relative L2 0.0700–0.1863。固定中心、Gaussian family 与支持圆盘本身限制了 RF 形状；zero-center RF centroid 的稳定尤其不能视为数据独立恢复了中心。

Global RF 的 16-bin 截断在本次条件下不是主要障碍：完整原始 150-bin Jacobian 中保留能量为 **99.1660%–99.9898%**，4 个固定 600-bin 长窗口代表模型为 **99.4468%–99.9895%**。然而 H1 ordered RF 最低只保留约 **74.41%**，不能用同一结论保证各 pathway 的完整 temporal profile。

H1/direct-BC/AC 的 signed-mean 方向在现有真实 fit 比较中全部保持，但大小并非同等稳定。H1 的 seed magnitude CV 最高 28.11%；67#6 两个 fresh fits 的 H1-off mean-|Δlogit| 相差 75.35%，而 validation NLL 只差 0.001322。个体 tau/delay 没有可靠的生理参数恢复证据：存在 exact parameterization redundancy、重叠 basis、边界聚集和 alignment-induced shifts。**“模型含有 pathway”“固定拟合中的 pathway 有影响”“spikes 必须依赖该 pathway”“真实视网膜具有相同贡献”是四种不同主张。**

另一个重要证据边界：索引中的 2026-08-30 shared-BC synthetic 是 N=8；2026-08-31 correctness patch 改变了它实际进入的 multi-cell mixing 路径。因此其保存结果不能冒充当前实现的精确 full-forward replay。本轮精确重建了其历史输出，保留为历史 shared-BC same-family evidence；没有混入旧 independent-AC 结果。

## 2. Frozen object and evidence boundary

| 项目 | 本轮锁定对象 |
|---|---|
| branch / HEAD | `rgc-readout-v2` / `fea28de038821fadee279b93728688b34bcb3bac` |
| git status | 开始时已有 untracked 审计文件及用户文件；完整原文保存在 `evidence_manifest.initial.json:git_status_before`，未清理或覆盖 |
| public model name | Canonical V1；revision=4；mechanism_identifiable |
| causal contract | `h1-shared-bc-direct-broad-ac` |
| spatial contract | `bc-central-disk_ac-overlapping-full-disk` |
| primary aligned | `output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/` |
| zero-center comparator | `output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/`，仅 nuisance comparator |
| indexed shared-BC synthetic | `output/synthetic_canonical_v1_shared_bc_noise_free_3seeds_20260830/`；须附后述实现版本限制 |
| independent seeds | `.omo/evidence/real_data_independent_seed_sanity/`，67#4、67#6、68#4、69#4 |
| real-data identity | 22 cells、37 recordings；每个 checkpoint 单独 N=1、289 cone inputs、33 trainable scalars |
| fixed center | full-training-refit LN center，经 `(x,-y) × 3 × 4.6/256` 转成 degree，作为 Canonical 固定 buffer |
| final stage | `trained`；fresh full-train refit；refit steps 等于原 inner selection best step |
| original validation | 未用于 checkpoint selection；本轮只用于 replay 与诊断 |

**强制门槛：22/22 strict-load 和 original validation 精确重放通过。** Target、mask、source-image order、trial order、logits 均逐位一致；NLL 与原保存值精确相等；无 checkpoint conversion、无参数梯度残留或 state 修改。证据为 [primary_replay.json](D:/PythonProject/retina_rf_SNN/output/audits/aligned_canonical_identifiability_main_20260905/primary_replay.json)。因此满足继续机制审计的条件。

原 zero-center mean NLL=0.4389561455，aligned=0.4240395345，18/22 改善。该比较同时包含改变固定中心后重新拟合其他参数的影响，不能解释为保持参数不变的纯坐标变换，也不能把 aligned RF 当 ground truth。

审计开始冻结 1,278 个相关文件，包括所有使用的 lineage checkpoints、原始 movie/spike files、RF/intervention/parameter artifacts、生产代码与来源审计。最终 [evidence_manifest.json](D:/PythonProject/retina_rf_SNN/output/audits/aligned_canonical_identifiability_main_20260905/evidence_manifest.json) 给出完整 SHA256、source drift、新增产物哈希及前后完整性结果。初始清单保留原始 git status，不用新哈希覆盖历史 producer manifest。

已读取 indexed correctness 与 causal/applicability audits：较早 applicability 报告的“全局 FAIL”是历史状态；随后 correctness patch 对四项问题的结论为 PASS，并验证原 N=1 lineage 的 1,188 个冻结张量逐位不变。本轮 primary replay 提供 aligned N=1 的当前执行证据。该保证不能外推至 N=8 synthetic。

未取得/不存在的证据：aligned independent-seed fits；独立生物 RF center/参数真值；current implementation 下的 temporal parameter recovery；在其他参数可重新优化后检验 pathway necessity 的结果。全部列为 **UNVERIFIED**，不补训练、不继续中心 provenance 搜索。

### Synthetic 的实现版本边界

该 synthetic producer 是 shared-BC causal family，不是旧 independent-AC。Teacher/student 使用相同 topology、fixed spatial family、零 observed history，以及 noise-free teacher probabilities；3 个学生各 400 步，teacher 不训练。原结果明确 `parameter_recovery_audit_run=false`。

但历史 producer hash 与当前代码在 `bipolar_subunits.py`、`model.py`、`pathway_rf.py`、`pathway_spatial_geometry.py`、`shared_subunits.py`、`spatial_contract.py`、`support_partition.py` 七处不同，对应已记录的 correctness patch。当前 teacher strict-load 后，H1 state 与保存值精确相等，direct BC 最大误差 0.0455299，最终 probability 最大误差 0.0119702。CPU 1/2 threads 均如此，不能解释为线程舍入。该 full-model replay 不通过，不能称为当前实现已验证 recovery。

为完成既有 prediction–mechanism 对照，本轮只把保存的四路 post-gain currents 按原加法顺序相加，通过源哈希未变化的 RGC 模块重建输出。Teacher probability、各 student 原 float32 KL、四个模型各三种 Δlogit 均精确匹配。**历史重建通过与当前 full-model replay 不通过同时成立。** 新 raw 结果分别保存，未覆盖旧产物或转换 checkpoint。

## 3. Quantity taxonomy

[quantity_taxonomy.csv](D:/PythonProject/retina_rf_SNN/output/audits/aligned_canonical_identifiability_main_20260905/quantity_taxonomy.csv) 逐项列出 30 个量、实际代码来源、trainability、fixed component、effective definition、冗余、真实/synthetic/seed 证据与 biological ceiling。

- **FIXED PRIOR**：Canonical 中冻结的 LN center、BC/AC support、Gaussian scales、RGC adaptation/history constants。LN center 在 LN 拟合中学习过，但在 primary Canonical 中不是 trainable 参数。
- **LEARNED BUT GAUGE/REDUNDANT**：BC raw softmax weights、AC local/transient raw logits，以及存在联合 mode-label permutation 的 individual BC tau parameterization。
- **LEARNED EFFECTIVE QUANTITY**：有 forward 意义的 bounded tau/delay、H1 amplitude、BC/AC gains、history gate、response bias；“能读出有效值”不等于“唯一恢复”。
- **FUNCTIONAL OUTPUT QUANTITY**：global/ordered pathway RF、energy centroid/extent、signed temporal marginal、clamped-minus-normal effects。
- **BIOLOGICAL QUANTITY UNVALIDATED**：把上述量解释成真实 circuit center、pathway strength 或生理 time constant 的身份关系。

原始 softmax logits 的共同平移在数学上不改变 normalized weights；BC 同时置换三个 sustained/transient mode pairs 及对应 weights 也不改变表示的函数。这里说的是 parameterization 的代数等价，不声称未执行的浮点 perturbation 已获得逐位相等。每个 sustained mode 只被要求大于其 paired transient mode，代码未强制三个 modes 在各自 family 内按 index 排序。

## 4. RF identifiability

### Definition 与 intervention 的区别

实际 producer 调用 [rf_effective.py](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/rf_effective.py) 与 [clean_sampled_reporting.py](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/clean_sampled_reporting.py)。Target 是 **final logit**，不是 probability；输入是原 validation cone drive，observed spike history 保持固定；每条原序列在开始处 reset，endpoint 为第 149 个 bin。先对每个 context 求 Jacobian，再对 context 取算术平均。保存的 16 lags 按 oldest→current 排列，包含 lag 0；dt=6.6667 ms，因此最老保存 lag 的年龄为 100 ms，16 个 bin 宽度合计 106.6667 ms。

令 J 为 normal Jacobian，J_H 为 H1-off，J_HA 为 H1+AC-off，则：

`direct-BC ordered RF = J_HA; AC ordered RF = J_H − J_HA; H1 ordered RF = J − J_H`。

三者 telescoping 相加得到 global RF。H1 引起的下游 BC/AC 与非线性 RGC 变化分配给 H1；AC 项是在 H1-off 条件下定义。因此 ordered RF 不是各 pathway 独立的生物 receptive field，也不等同于 normal 条件下分别 clamp 每一路得到的效应。

[pathway_decomposition.py](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/pathway_decomposition.py) 另有 `effective_pathway_rf`：通过 `∂final_logit/∂total_current`，对 BC sustained/transient、AC local/transient currents 作 chain-rule 分解。它不是这些保存 RF 的 producer，不应混用命名或 H1 attribution。

RGC 的 observed-history 项与 bias 在最终 logit 中加性进入。固定 observed history 时，它们不参与对 stimulus 的这个 logit Jacobian；这不能证明它们对 probability、prediction 或闭环 spike feedback 没有影响。

### 16-bin truncation

本轮保存完整 true-forward autograd Jacobians。能量为 `sum(J²)`，先分别平方各 context 再汇总，避免 context 平均抵消产生错误 retained fraction。16-bin context mean 与 22 个 aligned 原 RF **全部精确一致**。

| RF | 原150 bins：retained energy 最小 / 中位 / 最大 | 固定600 bins四代表：最小 / 最大 |
|---|---:|---:|
| global | 99.1660 / 99.8702 / 99.9898% | 99.4468 / 99.9895% |
| H1 ordered | 75.1167 / 95.3834 / 99.3788% | 74.4120 / 99.3397% |
| direct-BC ordered | 99.4585 / 99.9928 / 99.9997% | 99.5318 / 99.9989% |
| AC ordered | 92.1014 / 99.1758 / 99.8907% | 96.3923 / 99.9293% |

600-bin 诊断使用既有 seed protocol 的四个代表 cells，各取第一个 recording/trial 的四段连续 validation，source frame continuity 已断言；只在四秒窗口开始 reset。它有不同的 conditioning/reset，不能替换 production RF。最大配置单 state tau=250 ms，4 秒覆盖其16倍，同时远长于 finite BC kernel 与串联 delays。实测最老100 bins 能量占比不超过 5.5e-40；这是有限精度执行下的残余检查，不是递归系统严格有限 memory 的证明。

**Global 与 direct-BC 的本次截断问题可关闭。H1 完整 temporal profile 必须降级；AC 个别模型也需注明约8%的窗口外能量。** 无 lag-length 扫描。详见 [rf_truncation.csv](D:/PythonProject/retina_rf_SNN/output/audits/aligned_canonical_identifiability_main_20260905/rf_truncation.csv)。

### Alignment sensitivity 与 existing seeds

| RF | zero→aligned cosine 中位数 | 4-cell zero-center seed cosine 范围 | seed relative L2 范围 |
|---|---:|---:|---:|
| global | 0.257272 | 0.983164–0.997655 | 0.0700–0.1863 |
| H1 | 0.870265 | 0.982649–0.998998 | 0.0473–0.8192 |
| direct-BC | 0.314911 | 0.987989–0.999793 | 0.0212–0.1698 |
| AC | 0.665379 | 0.996058–0.999649 | 0.0265–0.2018 |

这些 cosine 使用同一绝对 cone coordinate frame，不先平移配准 RF。Global energy-center displacement 的 zero→aligned 中位数为 0.086778°、范围0.013569–0.417310°；signed temporal marginal cosine 中位数0.992966、最低0.660786。前者大变而后者常保持相近，说明空间与时间 marginal 必须分开报告。RF norms、relative L2、extent 和两套 temporal marginals 均见 [rf_identifiability.csv](D:/PythonProject/retina_rf_SNN/output/audits/aligned_canonical_identifiability_main_20260905/rf_identifiability.csv)。

Seed RF 的最大 centroid displacement 仅约1.16e-10°，但 zero-center 对称 geometry 已把 energy centroid 限制在中心；不能把这个结果当作 recovered biological center。Extent 最大变化0.002344°，global temporal marginal cosine≥0.985672。所有原4-cell fits 被纳入，未追加或选择 aligned seeds。**Aligned 22-cell independent-seed RF stability：UNVERIFIED。**

### Historical shared-BC synthetic RF

| RF | raw cosine 中位数 | trained cosine 中位数 | raw→trained relative L2 中位数 |
|---|---:|---:|---:|
| global | 0.996776 | 0.999967 | 0.08778→0.00817 |
| H1 | 0.991300 | 0.999846 | 0.93474→0.04782 |
| direct-BC | 0.998710 | 0.999971 | 0.05541→0.00761 |
| AC | 0.975352 | 0.999722 | 0.25539→0.02370 |

Raw RF cosine 已很高，尤其 global/direct-BC；它来自同 topology、空间 basis family 与初始化，不是训练恢复的全部功劳。Relative L2 显示训练确实改进了功能量幅度。该证据只支持**历史指定实现、正确 same family、noise-free teacher 的条件性 mapping recovery**；不证明 misspecification 下的 identifiability、真实 RF ground truth 或 real-data uniqueness，也不替代修后 multi-cell 的当前实现验证。

## 5. Pathway intervention identifiability

统一 primary definition 为同一实际 validation score mask 上的 `off−normal Δlogit`。报告 signed mean、mean absolute、RMS、full-vector cosine 和 nonzero-bin sign agreement。H1-off 关闭 upstream surround 后重算全部 downstream；direct-BC-off 仅关闭 direct currents，保留 broad BC→AC；AC-off 关闭 local/transient gates。RGC 非线性状态重算，因此单独 gain 不能代替 functional contribution。

| Pathway | alignment full-vector cosine 中位/最低 | alignment mean absolute Δlogit ratio 中位/范围 | seed cosine 最低 | 4 cells magnitude CV 范围 |
|---|---:|---:|---:|---:|
| H1 | 0.98519 / 0.58046 | 1.8553 / 0.9658–32.1351 | 0.98719 | 9.32–28.11% |
| direct-BC | 0.98074 / 0.66892 | 1.0378 / 0.7474–1.3822 | 0.99108 | 0.61–10.51% |
| AC | 0.98983 / 0.84156 | 1.0149 / 0.7395–1.3380 | 0.99666 | 2.17–14.14% |

22 个 alignment pairs 及12个 seed pairs，各 pathway 的 signed-mean sign 均一致；这不是每个 bin 方向完全不变。Alignment nonzero-bin sign agreement 最低 H1=65.0%、direct-BC=77.29%、AC=84.38%；seed 最低分别95.63%、96.46%、97.92%。Signed-mean 的 ON/OFF 解释不能简化成“某一路在所有刺激都兴奋/抑制”。

H1 对 spatial nuisance 的相对 magnitude 最敏感，且一个小 reference effect 会放大 ratio；CSV同时保存绝对值，不单凭32倍作结论。Direct-BC/AC 在已测试 seeds 中较稳定，但仍有10–14% CV和约20%以上 pair amplitude changes；三路均应限制为 **directionally stable, quantitatively weakly identifiable**，其中 H1 定量证据最弱。Aligned 单 fit 中非零 effect 证明条件性 influence，不证明在允许其他参数补偿后 spikes 必须依赖该 pathway。

Historical synthetic 的训练后 intervention Δlogit cosine 中位数：H1=0.998616、direct-BC=0.999957、AC=0.999747；relative L2 约0.09、0.0094、0.028。相应 RF Δ recovery 已保存，H1 相对 L2约0.048。尽管 float32 prediction KL仅2.9802e-8，H1 mean-|Δlogit| 仍比 teacher 低约7.5%。**极相近 prediction 并未使干预幅度相同。** 这是一项有实现版本限制的功能反例，不是 biological intervention 验证。

逐 cell、fit、pathway 的 signed/RMS/full-vector结果见 [pathway_identifiability.csv](D:/PythonProject/retina_rf_SNN/output/audits/aligned_canonical_identifiability_main_20260905/pathway_identifiability.csv)。

## 6. Temporal parameter identifiability

### Actual forward 中的参数含义与补偿路径

H1 使用 bounded tau 转成 `exp(-dt/tau)` 的 lowpass；delay 先作用于 graph drive，再进入 H1 state，amplitude 经图回投后从 cone drive 扣除。故 H1 tau/delay/amplitude 改变所有 downstream input，能与 spatial alignment、BC weights/gains及AC分支相互补偿。

BC 使用三对 sustained/transient gamma-shaped modes，在16-bin固定网格上 **先 L2 normalize，再 fractional delay，再截断**。改变 tau/delay既可改变形状，也可改变 delay 后保留的 norm，后者可被 gain/weights部分吸收。Aligned within-family basis cosine 中位数为 sustained=0.973198、transient=0.996340，存在非常相近的模式；并非所有 pairs 都相近，完整范围分别0.7731–0.999994、0.4141–1.000000。不能把三个 tau 当作三个可独立定位的真实细胞动力学。

AC 接收与 direct BC 共享 weights/temporal bank 的 broad BC drive，然后做 downstream delay和lowpass；local/transient gates与AC gain控制输出。最终 RGC 使用固定divisive/membrane/adaptation动力学，再减去固定核过滤 observed past spikes 后乘 learned history gate 的项，最后加 response bias。BC与AC串联 delays/taus、mixture、gain及bias有明确的补偿渠道。这里是 actual forward 给出的结构可能性；本轮 observed parameter shifts 不能证明某个唯一 trade-off 方向或估计 Fisher rank。

所有 ordering 是变换约束：sustained tau/delay>paired transient；AC local tau/delay>transient。具体上支 lower=`max(fixed lower, paired transient+epsilon)`；CSV同时列 fixed-bound u 与 active conditional lower/u。Hard-bound proximity 不等于边界因果决定了参数，亦不代表统计显著性。

### Aligned bound proximity

描述性阈值固定为 u≤0.05 / u≥0.95；共22 cells，每个 mode 分别计数。

| Parameter | 固定范围 ms | 靠近 lower | 靠近 upper |
|---|---:|---:|---:|
| H1 tau | 10–200 | 15 | 0 |
| H1 delay | 0–20 | 1 | 5 |
| BC sustained tau modes 0/1/2 | 20–200 | 21 / 2 / 2 | 0 / 0 / 0 |
| BC transient tau modes 0/1/2 | 5–120 | 19 / 17 / 11 | 0 / 0 / 0 |
| BC sustained delay | 0–30 | 0 | 0 |
| BC transient delay | 0–20 | 0 | 1 |
| AC local tau | 20–250 | 20 | 0 |
| AC transient tau | 15–180 | 21 | 0 |
| AC local / transient delay | 0–40 / 0–25 | 0 / 0 | 0 / 0 |

尤其 AC taus 和 BC 第一个 sustained mode 聚集在低端。即便 seed SD小，也只能说当前约束优化得到相似值；没有改变 bounds 或独立 truth 的证据，不能进一步声称这些值由 spikes 精确决定。

### Alignment-induced shifts

下表绝对/相对变化均为 **逐 cell paired change 的中位数**；不等于两列总体中位数之差。增加/减少计数和 Spearman rho 均按22 cells计算，不作显著性检验。

| Parameter | median absolute change (ms) | median absolute relative change | 增加/减少 | zero-aligned rank rho |
|---|---:|---:|---:|---:|
| H1 tau | 1.9622 | 12.50% | 2/20 | 0.6669 |
| H1 delay | 0.9347 | 8.77% | 17/5 | 0.3473 |
| BC sustained tau 0 | 0.2720 | 0.92% | 5/17 | 0.7425 |
| BC sustained tau 1 | 4.5032 | 8.56% | 5/17 | 0.7967 |
| BC sustained tau 2 | 5.2162 | 4.68% | 6/16 | 0.8317 |
| BC transient tau 0 | 0.7177 | 8.88% | 10/12 | 0.6567 |
| BC transient tau 1 | 0.9157 | 9.74% | 5/17 | 0.7222 |
| BC transient tau 2 | 0.9210 | 8.87% | 5/17 | 0.7391 |
| BC sustained delay | 0.7795 | 4.95% | 12/10 | 0.7854 |
| BC transient delay | 0.6930 | 4.80% | 13/9 | 0.6115 |
| AC local tau | 0.4704 | 1.85% | 12/10 | 0.7866 |
| AC transient tau | 0.3002 | 1.78% | 6/16 | 0.8792 |
| AC local delay | 1.5212 | 13.19% | 20/2 | 0.6985 |
| AC transient delay | 1.6442 | 18.33% | 21/1 | 0.7233 |

H1 amplitude **22/22 上升**，median paired absolute change=0.029207、median paired relative change=70.05%；BC gain17/22、AC gain16/22上升。空间 nuisance 修正后的重新拟合系统改变了 H1 amplitude/tau/delay、AC delays，且一部分 BC basis taus 明显移动。这支持**原 zero-center estimates 具有 cross-domain compensation / nuisance dependence**；不能把其时间参数直接当作 invariant physiology。

### Existing real seed values 与 recovery gap

4-cell NLL spread 为0.001238–0.004851。67#4 H1 tau 在28.483–38.641 ms之间；67#6 BC sustained mode1 在54.822–81.139 ms之间（CV20.18%）。69#4对应多数 temporal spreads较小，不能只挑较差或较好的一组代表全部22 cells。

全部 tau/delay 的逐 fit values、range、sample SD、CV、bounds、NLL spread 均在 [temporal_identifiability.csv](D:/PythonProject/retina_rf_SNN/output/audits/aligned_canonical_identifiability_main_20260905/temporal_identifiability.csv)。4 cells 两个 BC families 的 within-family mode ranks在现有3fits保持0<1<2；这不排除联合 permutation gauge。Sustained>transient及AC local>transient的跨family稳定是代码强制，不是独立 recovered ordering。详见 `temporal_order_stability.csv`。

沿 AUDIT_INDEX 定位并检索的 current shared-BC artifacts 未提供另一份 teacher/student effective tau/delay recovery audit。**CURRENT-CONTRACT TEMPORAL PARAMETER RECOVERY NOT TESTED**。本轮不以读取可用的 synthetic parameters替代一个尚未执行的 recovery protocol，更不把 RF/counterfactual recovery改称 temporal recovery。修后N=8 full-forward差异进一步限制了历史 synthetic 对当前实现的适用性。

## 7. Prediction-equivalent vs mechanism-equivalent

[prediction_vs_mechanism_equivalence.csv](D:/PythonProject/retina_rf_SNN/output/audits/aligned_canonical_identifiability_main_20260905/prediction_vs_mechanism_equivalence.csv) 包含22个 zero/aligned nuisance pairs、4cells内每3fits的全部12个 pairs、3个 historical teacher/student population comparisons，共37行，不合并总体统计。Synthetic logits来自已验证的 archived-current reconstruction，temporal distance留空并标明 recovery gap；真实 temporal distance为14个有效tau/delay值按各自bound width归一化后的RMS，保留BCmode labels。

“Prediction 接近”仅为下列连续指标的描述；没有预注册的 statistical equivalence margin、置信区间或等效性检验。Logit cosine还可能受相同负bias支配，所以同时列RMSE。

| 既有 pair | absolute ΔNLL | logit cosine / RMSE | global RF cosine | 实际机制差异 |
|---|---:|---:|---:|---|
| 67#6 fresh_1→fresh_2 | 0.001322 | 0.995817 / 0.173809 | 0.985642 | H1 magnitude ×1.7535；direct-BC ×0.8095；AC ×0.7734；BC sustained tau1 54.822→81.139 ms |
| 68#4 fresh_1→fresh_2 | 0.000529 | 0.999882 / 0.042885 | 0.997213 | H1 magnitude ×0.7937；H1 tau55.506→49.108 ms；H1 delay7.571→6.224 ms |
| 67#4 primary→fresh_2 | 0.003007 | 0.999028 / 0.161669 | 0.992521 | H1 tau28.483→38.641 ms；AC magnitude ×0.7924 |
| historical teacher→53001 | 2.9802e-8 | 0.999999899 / 0.000920 | 0.999967 | H1 magnitude ×0.9252，Δlogit relative L2约0.0906 |

这些例子来自全量对照表：分别呈现最大 real seed H1 magnitude pair ratio、最小 real seed |ΔNLL| pair、H1 tau spread较大的cell，以及固定第一个synthetic seed；不是隐藏其他 fits 后只报有利结果。它们反驳“相近 prediction 自动意味着相同机制量”的实际使用方式；不宣称已证明真实数据存在完全相等的预测函数而不同生物机制。

## 8. Biological interpretation boundary

[claim_verdicts.csv](D:/PythonProject/retina_rf_SNN/output/audits/aligned_canonical_identifiability_main_20260905/claim_verdicts.csv) 给出全部请求项目的四级判定：

| 级别 | 本轮可归入的窄范围 |
|---|---|
| SUPPORTED | 代码包含H1/shared-BC/direct-BC/AC；固定模型具有sustained/transient family、强制AC ordering。此处仅为implementation/prior事实 |
| FUNCTIONALLY SUPPORTED | 条件global RF空间及短窗时间形状；限定的direct-BC ordered pattern；三个pathway在冻结fit中的非零functional influence |
| WEAKLY IDENTIFIABLE / MODEL-DEPENDENT | H1/AC ordered RF的完整定量解释；所有pathway magnitudes；H1tau/delay、BCdelay、individualACtau/delay有效数值 |
| UNIDENTIFIED | biological RFcenter；唯一individualBCmode identity；spikes在允许补偿后必须依赖某pathway；真实retina的精确贡献；learnedtau=biologicaltimeconstant |

“FUNCTIONALLY SUPPORTED”不把 aligned22缺少seed复核的事实抹掉；“WEAKLY”也不表示该参数有已证明的统计识别区间。当前没有任何 individual tau/delay 可作为已恢复的生理常数。

## 9. Main text / supplement / remove

**可保留主文的表述：**“在固定的 training-derived functional alignment 和生理约束模型族下，我们拟合真实 macaque RGC responses，并用 final-logit Jacobian 和冻结模型干预刻画条件性 RF 与 pathway influence。现有证据显示 functional RF 形状的稳定性强于 pathway contribution magnitude 和 individual temporal parameters；良好预测不能单独确立 circuit parameter identity。”

主文可展示aligned fitted global RF、同条件的三路intervention及其证据边界，明确RF为模型conditional sensitivity，并指出aligned多seed缺口。不要把跨seed稳定性写成22-cell已验证结论。

**Supplement：**历史shared-BCsame-family recovery及其currentimplementation不兼容说明；4-cellzero-centerseed sanity；zero-vs-aligned nuisance sensitivity；bound/ordering/basis-overlap表；所有per-cell数值；RF truncation及完整replay证据。

**应避免或改写：**`recovered biological circuit parameters`；`identified H1/BC/AC dynamics`；`pathway contribution equals biological contribution`；`high RF cosine proves circuit identifiability`；`similar prediction proves same mechanism`。分别改成“给定family内的有效拟合参数/功能描述”“冻结模型中的条件干预效应”“RF与参数识别分开检验”。

## 10. Do we need another experiment?

**对上述窄主张，无需新增训练：STOP ADDITIONAL IDENTIFIABILITY EXPERIMENTS。** 本轮已经识别出可报告的功能量和必须降级的参数主张，不为完整性继续堆叠seeds或audits。

仅当论文仍要求升级到“individual tau/delay可以恢复”时，最高价值的一个条件性候选是 **current-implementation N=1 synthetic effective-parameter recovery**；本轮不执行。其可证伪最小设计如下，实施前应作为单独协议冻结：

| 项目 | 唯一条件性候选 |
|---|---|
| 具体uncertainty | 在当前实际N=1实现与固定family内，prediction恢复是否同时恢复teacher的effective tau/delay |
| 为什么现有证据不够 | parameter_recovery_audit_run=false；现有real seeds无真值；历史N=8synthetic不能通过当前full-forward replay |
| Teacher | 直接冻结已有aligned67#6 final model作为known-effective-value teacher，不再训练teacher；不把其参数视为生物真值 |
| 最小新训练数量 | 两个独立fresh students；各沿既有noise-free400-step预算，总800optimizer updates；不得失败后自动加seed或延长训练 |
| 固定合同 | 当前revision/causal/spatial实现、N=1、teacher原289-conegeometry/fixedcenter、所有bounds/order/gains/RGCconstants、原noise-freeloss/optimizer/steps；沿原syntheticstimulus生成器构造teacher几何下32train/12validation、64bin序列，零history；所有随机种子在训练前固定；无architecture sweep |
| Primary metric | teacher/student14个effective tau/delay按boundwidth归一化的误差；BC同时置换成对taus及对应weights后匹配；每个student报告max及RMS，不能用RFcosine代替 |
| 成功条件（待单独预注册） | 两个students均predictionKL≤1e-6且全部matched tau/delay normalized error≤0.05；同时公开RF及干预向量误差，防止只报参数阈值；阈值是操作定义而非显著性 |
| 失败如何改变claim | 若prediction达标而temporalerror不达标，直接拒绝该teacher条件下的参数恢复主张；若prediction也未达标，结果为optimization/recovery INCONCLUSIVE，不能反推结构不可辨识，也不自动追加训练 |
| 成功如何改变claim | 只增加“当前N=1实现、一个已知teacher、正确family、noise-free下的effective parameter recovery”证据；仍不升级为real-datauniqueness或biologicaltimeconstant |
| 成本与服务主问题 | 两个400-stepCPUfits加现有diagnostics；未测wall-clock，不承诺时长；直接针对tau/delayrecovery缺口 |

该候选是条件性设计，不是本轮执行承诺。若接受窄论文主张，就按STOP结束。

## Reproduction and artifacts

[PROTOCOL.md](D:/PythonProject/retina_rf_SNN/output/audits/aligned_canonical_identifiability_main_20260905/PROTOCOL.md) 在对应诊断前冻结了指标与窗口，并明确记录发现syntheticsource drift后的证据处理规则。`validation-replay.pt`保存原validationinputs/targets/masks/order/logits；`inference-results.pt`保存完整150/600binJacobians、seedRF/effectiveparameters、synthetic历史重建logits及当前full-forward差异。`inference-source-hashes.json`记录实际执行诊断及protocol的SHA256；最终manifest另覆盖全部新source和数值产物。复现入口为本目录的`gate_replay.py`、`inference_diagnostics.py`、`summarize_evidence.py`；文件已存在时脚本保护性退出，避免覆盖已完成审计。

所有本轮写入都限于本目录；production source/model/data和旧artifacts不修改，不执行training、illusion或baseline。加载枚举元数据与synthetic版本差异的处理记录在`EXECUTION_NOTES.md`。
