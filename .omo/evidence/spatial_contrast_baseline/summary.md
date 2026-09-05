# Spatial Contrast baseline — definition gate

STATUS: **STOP_DEFINITION_UNRESOLVED**

日期：2026-08-31。仅完成来源核对与证据保存。实现数、训练数、模型加载数均为 0；未修改 Canonical V1、baseline、数据流程、split、target 或 loss。未使用 original validation 进行任何选择。

## 已锁定的原始定义

来源：Sridhar et al. (2026), *Modeling spatial contrast sensitivity in responses of primate retinal ganglion cells to natural movies*, DOI [10.1371/journal.pcbi.1014157](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014157)。其 Methods 的 SC 模型、拟合部分以及 Discussion 的模型限制均已核对。论文明确将 difference-of-Gaussians center/surround 扩展列为后续可能方向，而非本文已有模型。

官方代码：[gollischlab/SpatiotemporalSCModel](https://github.com/gollischlab/SpatiotemporalSCModel)，锁定 commit `76b733421cc16131c2229e66a7714d8892de39d7`。本目录 `sources/` 保存 6 份相关源码的文本快照及 MIT LICENSE；未安装或执行该代码。文件、Git blob SHA 和固定链接见 [source_manifest.json](D:/PythonProject/retina_rf_SNN/.omo/evidence/spatial_contrast_baseline/source_manifest.json)。

无额外 spatial smoothing 的原始特征计算为：

\[
h_{s,t}=\sum_{\ell}v_{\ell}X_{s,t-\ell},\qquad
\mu_t=\frac{\sum_s u_s h_{s,t}}{\sum_s u_s},\qquad
q_t=\sqrt{\frac{\sum_s u_s(h_{s,t}-\mu_t)^2}{\sum_s u_s}}.
\]

其中 `u` 为单个 Gaussian spatial filter，`v` 为单个 temporal filter。训练数据的均值/标准差分别用于标准化 `mu` 与 `q`；测试数据复用训练统计量。官方代码的参数坐标为：

\[
\lambda_t=a\log\left(1+\exp\left[b+w_1\widetilde\mu_t+w_2\widetilde q_t\right]\right),
\quad a\ge0.
\]

`a, b, w1, w2` 共 4 个拟合参数；代码另外输出 `w=w2/w1`。这里的 `lambda` 是期望 spike count，不是 Bernoulli probability。**原始 SC prediction 不含 spike-history 输入或 history 参数**：spikes 仅用于 RF 估计/拟合 target，不进入给定参数后的 prediction 函数。

直接源码依据（下列均为固定 commit 中的原行号）：

| 事实 | 官方源码快照及行号 |
|---|---|
| 从 STA 分离 spatial/temporal filter，再拟合单 Gaussian | `sources/sc_model__models__sc_model.py.txt:124–155` |
| 训练统计量标准化；拟合仅输入两列 stimulus features | `sources/sc_model__models__sc_model.py.txt:158–179` |
| temporal convolution、加权均值、加权标准差 | `sources/sc_model__utils__convolutions.py.txt:159–227` |
| 四参数 softplus 输出，无 spike history | `sources/sc_model__utils__nonlinearities.py.txt:6–13` |
| Poisson NLL 与 L-BFGS-B | `sources/sc_model__utils__minimization.py.txt:8–18,63–101,104–136` |
| RF filter 的 unit-L2 normalization | `sources/sc_model__utils__receptive_fields.py.txt:128–133` |

## 原始设置与项目适配边界

以下是作者官方入口中实际设置，**未擅自当成 Schottdorf 150 Hz 的适配超参数**：

| 项目 | 官方设置 | 本项目状态 |
|---|---|---|
| filter 来源 | 已提供的 cell STA，`sigpix` 方法；再做 Gaussian fit | 未找到已冻结的、可从项目现有 LN 唯一得到同等 `u,v` 的规则 |
| temporal crop | CLI 默认 30 bins；原数据 85 Hz | 项目原生 150 Hz；不能把 30 bins 自动当成现有 60-bin LN 合同 |
| spatial crop | CLI 默认每侧 20 pixels，即 40×40 | 当前固定输入为 17×17；作者示例 crop 不是通用适配规范 |
| Gaussian window / sigpix threshold | `sigma_window=3.0` / `6.0` | 未进行新的 STA/RF 估计 |
| stimulus smoothing | 默认 0，即 None | 未新增 smoothing search |
| parameter initialization | `[max(train_counts), -2, 1, 0]` | 未初始化或拟合模型 |
| bounds | `a>=0`；其余三参数无界 | 已锁定原代码语义 |
| optimizer / selection | L-BFGS-B；原 SC 入口无 LN 式 lambda grid 或 inner-dev refit | 用户要求 inner-dev/refit；SC 的适配候选配置尚无记录 |
| likelihood / output | Poisson count likelihood / nonnegative expected count | 用户指定 Bernoulli events；尚无冻结的输出概率绑定实现 |
| spike history | 不含 | 不会自动继承项目 LN/Canonical 的 history term |

以上入口设置见 `sources/sc_model__scripts__run_sc_model.py.txt`。本轮允许的 Bernoulli 与 inner-dev 约束不会被改回作者的 Poisson/test 协议；作者协议也不能直接冒充已满足本项目要求。

## 触发 STOP 的具体缺口

**唯一阻塞项：本项目 faithful SC adaptation 的完整数学合同尚未冻结。**

此前项目要求从 frozen center-surround LN 的 training-only fit 定义 linear RF；当前实现见 [center_surround_ln.py](D:/PythonProject/retina_rf_SNN/baselines/center_surround_ln.py:36)。它使用两个独立 60-bin temporal filters，实际 kernel 为

\[
K(s,\ell)=a_cG_c(s)v_c(\ell)-a_sG_s(s)v_s(\ell).
\]

一般情况下这个 kernel 的 space×time rank 可为 2，不能无损唯一转换成原 SC 要求的单个 `u(s)v(lag)`；空间差分也不是定义 weighted variance 所需的非负 Gaussian 权重。代码在第 82 行明确做 center-minus-surround，不能凭名称假定它等于论文的 single-Gaussian LN。

项目记录未规定这里如何获得单一 `u,v`；自行取 center、合并两个 temporal filters、取绝对值、做 rank-1 近似或新增 center/surround contrast terms，均会新增模型定义。本轮未做上述任何选择。Bernoulli output link 与 SC-specific inner-dev 候选配置也未被此前 LN 的 lambda grid 自动决定。

因此，原始 SC 公式已确认，但 requested project model 的参数量、selected hyperparameters 及 faithful implementation 均保持 **UNVERIFIED**。依照“定义不明确立即 STOP”，未进入实现、contract tests 或训练阶段。不是因模型表现或任何 validation 结果停止。

## NLL：已有记录参考，非本轮 SC 比较结果

未运行 SC，所以没有它的 overall / group / per-cell NLL、selected hyperparameters 或拟合 checkpoint。下表仅摘录/等权汇总现有结果，未重放模型、未重新验证逐元素 target/mask identity，不把它描述为本轮新完成的同合同 benchmark。

| 范围 | Constant | Center-surround LN | 当前 shared-BC Canonical V1 | Spatial Contrast |
|---|---:|---:|---:|---|
| overall（22 cells 等权） | 0.509817266 | 0.425997944 | 0.438956146 | NOT RUN |
| MC ON（5） | 0.496057206 | 0.395894259 | 0.428506756 | NOT RUN |
| MC OFF（4） | 0.533821441 | 0.430339150 | 0.441669881 | NOT RUN |
| PC ON（9） | 0.494986362 | 0.418741528 | 0.427831286 | NOT RUN |
| PC OFF（4） | 0.536382698 | 0.475613281 | 0.474335082 | NOT RUN |

已有 per-cell 数值及参考来源：

- [LN/Constant results.json](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/results.json)
- [LN/Constant per-cell.csv](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/per-cell.csv)
- [shared-BC Canonical results.json](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/results.json)

## 验证与未执行项

已核对：官方 commit、源码 blob identity、公式/参数/无-history 路径，以及项目 LN 的两个 temporal filters 与 center-minus-surround 公式。官方源码快照的实际 Git blob 校验结果保存在 `snapshot_verification.json`。

未执行：implementation、model contract tests、smoke、22-cell training/refit、checkpoint inference、任何新的 baseline metric。没有用既有 NLL 选择模型定义。

先前 inventory 中“可直接层叠在已完成 center-surround LN 上”的建议不足以锁定 faithful SC 定义；本轮来源检查发现该适配缺口，故该建议不能当成现成实现合同。
