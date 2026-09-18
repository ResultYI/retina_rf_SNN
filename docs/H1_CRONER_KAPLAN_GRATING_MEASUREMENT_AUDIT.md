# Croner–Kaplan-aligned grating-domain center-radius measurement audit

2026-09-15。**Method verdict：GRATING_RC_NOT_READY。**

Grating-domain amplitude DoG 对两个预定 model diagnostic 的曲线确实给出比 full-2D complex DoG 更紧凑的 summary，完整频率网格 residual 为 1.91%–3.05%。但这没有解决 radius measurement：当前有限视场/像素观测下的 synthetic ground-truth recovery 未通过既有标准；四个完整网格 model fit 全部触碰 gap bounds，且预定 frequency subset 可使 rc 漂移约 25%–39%。当前不足以冻结正式九-cell quantitative contract。

这是 measurement-method audit，不是 physiology failure、prediction benchmark 或 population validation。没有读取冻结 external rc target/reference，没有计算 B/E/Q，没有加载或运行任何 checkpoint，也没有训练或修改模型、geometry、FOV、support、preprocessing。使用的两个 cached transfer 固定为 67#6 / 67#7、seed 2026091301、float64、epsilon=0.01、完整150-bin kernel 的4 Hz投影。

## 1. 文献语义与 source level

| 项目 | 可核实内容 | 本轮状态 |
|---|---|---|
| Croner & Kaplan 1995 原论文 | 原始 bibliographic record/abstract 确认通过 macaque LGN 中记录的 S-potentials 研究 RGC；本轮未取得可直接核对的原文 Methods | 原文 grating/F1 细节 **UNVERIFIED**。不能写成论文没有说明 |
| Temporal frequency | McCann, Hayhoe & Geisler 2011 Methods 明确将其引用的 Croner–Kaplan surround-strength 测量描述为 4 Hz drifting sine-wave gratings；该段讨论 P-cell model | **SECONDARY_SOURCE_SUPPORTED**。固定4 Hz，不能声称逐项核实1995全部M-cell设置 |
| Observable | Kremers, Silveira & Kilavik 2001 Visual stimuli 将其 response amplitude/phase 定义为 spike Fourier fundamental（4 Hz）的幅度/相位 | 对Kremers是原文直接证据；作为1995方法支持则是二手来源。本轮不声称复现1995的F1归一化 |
| Frequency-domain DoG | Kremers 2001 Eq.1 将含 contrast 的 center-minus-surround Gaussian frequency response 归于 Croner–Kaplan，并说明约4 Hz时两部分近似拮抗 | 支持实数、相反符号的 center/surround，而非两套自由 complex phases |
| 原始 orientation / aperture / frequency sampling | 未核实1995的具体列表或选择规则 | **UNRESOLVED**；本轮不冻结正式 orientation aggregation，不声称 protocol-exact |

来源：[Croner & Kaplan 1995 原始记录](https://pubmed.ncbi.nlm.nih.gov/7839612/)、[McCann et al. 2011 Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC3214635/#S2)、[Kremers et al. 2001](https://journals.physiology.org/doi/10.1152/jn.2001.85.1.235)。出版商直接正文请求受限；公开可检索的 Kremers Methods/Eq.1 提供了上述方法文本。没有读取或使用本项目已冻结的 external radius 数据。Kremers 的物种、contrast、时长等参数没有移植成1995 protocol。

按用户指定，本轮拟合：

`R(nu) = A * |Ic exp[-(pi rc nu)^2] - Is exp[-(pi rs nu)^2]|`。

Kremers 展示的 signed expression 含 `Kc*pi*rc^2` / `Ks*pi*rs^2`；这里将它们写成 integrated strengths Ic/Is，并按本轮 amplitude observable 取模。rc/rs 均为1/e characteristic radius，单位degree，nu为cycles/degree。没有引入独立 complex gains、额外Gaussian或相位拟合。

## 2. 事前冻结的 measurement

本轮 [protocol.json](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_grating_measurement_audit_20260915/protocol.json) 在产生 grating response / synthetic recovery 前写入。输入沿用 zero Weber mean、FIX_HISTORY_ZERO、原默认动态初态（V0=2/9）、symmetric small-signal、float64、epsilon=0.01 和150-bin完整 impulse kernel。该量仍是 **finite-prefix symmetric small-signal logit transfer**，不是steady finite-contrast spike F1。

17×17 physical pixels 的 pitch 为0.05390625°，edge-to-edge FOV为0.91640625°，轴向Nyquist为9.27536 cpd。本轮 grating 使用全部既有方形FOV，域外不贡献响应；没有扩大视场、根据 support 定义 aperture 或补出缺失 surround。

对像素中心(xp,yp)，qx=nu cos(theta)、qy=nu sin(theta)，复数pixel平均为：

`Pp = sinc(pitch*qx) sinc(pitch*qy) exp[-i2pi(qx(xp-cx)+qy(yp-cy))]`，其中 `sinc(v)=sin(pi v)/(pi v)`。

它是每个physical pixel内 sinusoid 的解析面积平均，不是point sample。现实时间形式为 `epsilon*Re[Pp exp(i2pi*4*t)]`。由已冻结transfer得到 `Z=sum_p H4(p)*Pp`，**不取共轭**；observable为`|Z|`（logit/Weber），乘固定epsilon即相应小振幅logit F1。整体scale自由，因此是否乘这个全局常数不改变radius fit。

frozen functional center只作为空间phase origin，不搜索/移动它；phase origin整体平移只给Z乘单位模复数，不改变amplitude。保存的模型行按y递减排列，grating数组按y递增排列，交接时明确反转行序；没有改变物理坐标。

**固定诊断方向是wavevector的0°和90°**，不能与grating条纹方向混淆。两个都完整报告，不选择fit较好者、不平均成正式primary。由于原论文orientation规则未核实，正式aggregation仍留空。

Frequency sampling：33个log-spaced点，0.05–8 cpd。选择仅依据事前已知的MC test range和grid Nyquist。nu不是FFT整数frequency bin；低于1/FOV的grating片段可以呈现，但不代表低频分量在有限视场上正交或易于分离。两个预定subset为：

- `sparse9`：full33的每4个点取1个，保留两端，共9点；
- `upper4`：保留所有nu≤4，共28点，实际最大点为3.61990 cpd。

这两个subset在任何response前固定，所有结果保留；不是事后删掉不利频率。

拟合保持上述解析amplitude DoG，unweighted L2。用Ic=1消除A与Ic/Is的纯scale gauge，rho=Is/Ic>0，A为自由非负输出scale、条件于形状解析最小二乘求解。rc范围[0.005,0.4]°，gap=rs−rc范围[0.001,8]°，rho范围[1e−6,100]；固定三个starts和每start最多250次evaluation。除去整体conditioning scale外没有改变loss权重，没有给fit加pixel-MTF校正、deconvolution或额外Gaussian。

## 3. Synthetic recovery：结果不支持直接继承旧MC可识别性

Bank为rc=0.035、0.045、0.052、0.060、0.075、0.100、0.139、0.195、0.240°；0.052–0.195°是预先允许的MC测试范围，其他为边界。复用旧六组 `(rs/rc, Is/Ic)`：`(3,.2),(3,.85),(7,.2),(7,.55),(7,.85),(20,.55)`，以及三个subpixel phase=0、1/4、1/2格。没有根据模型响应设计nuisance。

Main operator：同心圆单位积分DoG → 当前方形FOV上的Gaussian pixel integral → 同一pixel-area-averaged grating → complex response → amplitude。合成center=(phase*pitch,phase*pitch)固定。圆对称DoG、方形FOV和相同x/y offset下，0°/90°对称，synthetic只拟合0°并用fixture核对90°一致，避免重复计算。

Noise设为clean complex response peak的0.5%或2% RMS，在取幅度前对real/imag加入独立Gaussian噪声；固定seed、每级5个draw；所有subset复用同一full-frequency noise realization。**这是本轮frequency response的数值误差模型，和旧pixel-coefficient noise的注入位置不同，不是生物噪声估计。** 沿用旧成功标准，没有放宽：无噪rc相对误差≤0.1%；带噪≤10%，同时solver成功且rc不触界；每个phase成功率≥90%、median relative bias绝对值≤5%才算该radius通过。gap/gain bounds另列，不隐藏。

共162个pixel-field cases，产生5346次固定subset/noise fit；另有162次无噪analytic oracle fit。这里的fit是analysis parameter estimation，**不是RetiPath训练**。

Full33结果如下。MARE为median absolute relative error。无噪每radius18条；每个非零noise90条。

| true rc (deg) | 无噪 median bias | 无噪通过率 | 0.5% noise：MARE / 通过率 | 2% noise：MARE / 通过率 | solver失败 0% /18 | 0.5% /90 | 2% /90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.035 | +33.54% | 0.0% | 33.40% / 4.4% | 33.77% / 4.4% | 0 | 0 | 0 |
| 0.045 | +23.01% | 0.0% | 23.11% / 0.0% | 23.58% / 0.0% | 0 | 0 | 0 |
| 0.052 | +18.58% | 0.0% | 18.57% / 0.0% | 18.57% / 0.0% | 0 | 0 | 0 |
| 0.060 | +14.18% | 0.0% | 14.14% / 0.0% | 14.81% / 1.1% | 0 | 0 | 0 |
| 0.075 | +9.53% | 0.0% | 9.61% / 61.1% | 10.06% / 47.8% | 0 | 0 | 0 |
| 0.100 | +7.38% | 0.0% | 7.32% / 83.3% | 7.15% / 78.9% | 0 | 0 | 0 |
| 0.139 | +6.20% | 0.0% | 6.22% / 81.1% | 5.76% / 72.2% | 0 | 0 | 0 |
| 0.195 | +16.72% | 0.0% | 16.21% / 31.1% | 6.12% / 63.3% | 12 | 54 | 28 |
| 0.240 | +12.02% | 0.0% | 11.96% / 26.7% | 4.82% / 56.7% | 12 | 66 | 36 |

MC六个半径在full33下：无噪成功0/108，0.5% noise成功231/540（42.78%），2% noise成功237/540（43.89%）。相应MARE为12.19%、12.16%、11.10%。solver失败分别12、54、28条，主要集中在较大radius；**solver失败与已知真值偏差分开记录，不能将所有失败归为不可识别的数学证明。** 噪声越大个别solver状态反而改善，不代表噪声提升信息量。

**三个frequency subsets、三个noise levels均未建立满足既定逐phase标准的measurement floor。** 不能把结果写成“floor一定高于0.195°”；这里不是只在小radius失效的单调sampling floor，而是当前operator+解析fit的系统偏差和部分病态共同作用。旧signed-pixel fit的floor不能转移到这个新measurement。

预定subsets的MC汇总（不据此选更好subset）：

| subset | 0.5% noise MARE / 通过率 | 2% noise MARE / 通过率 | 正式floor |
|---|---:|---:|---|
| full33 | 12.16% / 42.78% | 11.10% / 43.89% | 未建立 |
| sparse9 | 9.36% / 54.07% | 9.99% / 50.19% | 未建立 |
| upper4 | 13.22% / 31.85% | 12.34% / 37.22% | 未建立 |

### Frequency sampling 与观测错配如何区分

同样的frequency grids，若直接生成无限视场、连续正弦的解析DoG频响，MC每subset的36个noiseless oracle均恢复成功；最大rc相对误差分别3.11e−15、2.14e−13、5.33e−15。它证明当前solver/grid对理想解析函数有基本恢复能力，**不能替代physical-pixel主结果**。

Main synthetic即使没有噪声也偏离真值。这说明有限FOV、pixel integration/hold之后的响应与拟合的无限连续解析DoG之间存在实际观测定义错配；frequency point数量不足并非唯一解释。减少采样或截掉高频没有建立合格floor。本轮没有进一步单独分解pixel与FOV贡献，也没有给analytic oracle添加noise，因此不能排除frequency coverage在噪声条件下增加病态性。尤其不允许用更低的subset残差救回正式measurement。

### 相邻radius是否collapse

Full33在三个noise水平分别检查相邻true radii、相同phase/nuisance的fitted median；0.240°边界不进入floor/collapse主计数。全部126对中各有2对≤5% collapse、2对次序反转、8对log间距压缩至少一半；这些发生在MC下方的边界探针。仅MC的90对在各noise下均没有≤5% median collapse或次序反转；2% noise时有1对经验10–90%区间重叠。

**没有MC median collapse不等于定量恢复成功**：此次主要问题是系统radius bias，而不只是不同true radius全挤成同一个值。每个经验区间只有5个draw，不能称confidence interval。

## 4. 两个cached model diagnostic

| cell | 方向° | frequency subset | rc (deg) | rs (deg，仅诊断) | normalized residual | gap触界 | gain ratio触界 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 67#6 | 0 | full33 | 0.069649 | 8.069649 | 2.649% | 上界 | 边界 |
| 67#6 | 0 | sparse9 | 0.064966 | 8.064966 | 4.419% | 上界 | 边界 |
| 67#6 | 0 | upper4 | 0.071421 | 0.072421 | 0.477% | 下界 | 边界 |
| 67#6 | 90 | full33 | 0.079325 | 0.080325 | 3.053% | 下界 | 否 |
| 67#6 | 90 | sparse9 | 0.054569 | 0.111289 | 0.215% | 否 | 否 |
| 67#6 | 90 | upper4 | 0.048378 | 0.280600 | 0.032% | 否 | 否 |
| 67#7 | 0 | full33 | 0.089901 | 0.090901 | 1.908% | 下界 | 否 |
| 67#7 | 0 | sparse9 | 0.064459 | 0.615608 | 0.278% | 否 | 否 |
| 67#7 | 0 | upper4 | 0.063675 | 0.779022 | 0.064% | 否 | 否 |
| 67#7 | 90 | full33 | 0.088121 | 0.089121 | 1.986% | 下界 | 否 |
| 67#7 | 90 | sparse9 | 0.065198 | 8.065198 | 0.509% | 上界 | 边界 |
| 67#7 | 90 | upper4 | 0.065498 | 0.066498 | 0.272% | 下界 | 边界 |

所有12个best fits均报告solver success，rc自身均未触搜索上下界。**但完整full33的4/4 fits均触gap界：**

- 67#6，0°：rho达到1e−6下界，gap达到8°上界；surround几乎退出，rs没有可用解释。
- 其余三个full33：gap达到0.001°下界，rho约0.959–0.971，两个几乎同半径的Gaussian配合很大的overall scale。低residual并未证明center/surround可分。

rs只作为measurement diagnostic，不能写成真实surround radius。没有解除bounds、增加starts、删除frequency或更换DoG来降低残差。

Orientation和frequency subset稳定性：

| cell | full33：rc90/rc0−1 | sparse9：rc90/rc0−1 | upper4：rc90/rc0−1 |
|---|---:|---:|---:|
| 67#6 | +13.89% | −16.00% | −32.26% |
| 67#7 | −1.98% | +1.15% | +2.86% |

67#6的90°，sparse9/upper4相对full33的rc变化为−31.21%/−39.01%；0°为−6.72%/+2.54%。67#7两个方向随subset改变均有约−26%至−29%的漂移。因此67#6有明显方向与frequency-domain形状相互作用；67#7方向差异较小，却仍没有稳定的radius measurement。不能因后者方向一致就宣告方法就绪。

完整complex responses和三套fit curves保存在NPZ。以下仅选固定frequency点展示full33观测/拟合，均除以各自观测curve peak；这只是展示，没有删除fit中的其他点：

| cpd | 67#6 0° 观测/拟合 | 67#6 90° | 67#7 0° | 67#7 90° |
| --- | --- | --- | --- | --- |
| 0.0500 | 1.0000 / 0.9925 | 0.9995 / 0.9990 | 0.9999 / 0.9987 | 1.0000 / 0.9967 |
| 0.6325 | 0.9712 / 0.9738 | 0.9975 / 0.9953 | 0.9881 / 0.9865 | 0.9825 / 0.9825 |
| 2.2494 | 0.7660 / 0.7790 | 0.9113 / 0.9255 | 0.8185 / 0.8327 | 0.8007 / 0.8158 |
| 4.2420 | 0.4311 / 0.4194 | 0.6657 / 0.6355 | 0.4771 / 0.4453 | 0.4672 / 0.4330 |
| 8.0000 | 0.1682 / 0.0463 | 0.1764 / 0.0796 | 0.0789 / 0.0241 | 0.0787 / 0.0257 |

## 5. 与旧full-2D complex fit的关系

旧full-2D complex spatial DoG residual约0.470/0.544；本轮full33 grating amplitude residual为0.0191–0.0305。两个residual的空间、相位信息、权重和自由度不同，不能当成相同loss下的模型性能改进。

可以支持的有限表述是：**Croner–Kaplan-style measurement domain 提供了更合适的amplitude summary。** 不支持“完整RetiPath RF已经成为圆对称DoG”，也不支持“拟合rc已可与外部目标直接比较”。丢弃phase、只取两条orientation剖面确实减少了拟合所需解释的信息；gap collapse和subset漂移仍表明radius分解不稳。

## 6. Correctness、文件与停止边界

- Gaussian pixel integral复用既有analysis实现；grating pixel平均另由16×16 Gauss–Legendre quadrature核对，max abs error **3.20e−15**。
- synthetic global spatial-phase shift保持amplitude，圆对称fixture的正交orientation一致。
- 纯synthetic signed delayed impulse fixture核对traveling-grating的复数符号、无共轭合成与F1提取，max abs error **2.48e−16**；没有运行checkpoint。
- 相关正式source、既有assay/preprocessing和cached operating-point artifacts的SHA256前后相同；analysis module不导入torch/model/training模块。
- 本次固定bank运行约140.6 s，无restart、追加noise seed、放宽拟合预算或结果驱动的subset选择。

输出：

- [protocol.json](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_grating_measurement_audit_20260915/protocol.json)
- [synthetic_recovery.csv](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_grating_measurement_audit_20260915/synthetic_recovery.csv)：全部true/fitted参数、bias/error、solver/bounds、noise seed和subset；全部失败保留。
- [diagnostic_model_fits.csv](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_grating_measurement_audit_20260915/diagnostic_model_fits.csv)：12个预定fit，无external comparison。
- [diagnostic_curves.npz](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_grating_measurement_audit_20260915/diagnostic_curves.npz)：synthetic clean complex curves、模型complex curves、所有frequency与predicted fit curves；noise按CSV种子可重建。
- [analysis-only implementation](D:/PythonProject/retina_rf_SNN/work/croner_kaplan_grating_audit.py)

文件hash：

| 文件 | SHA256 |
| --- | --- |
| protocol.json | de101999b4cf66c852783741b98203f99e588be89e207c2ae8b5ec40c3e22d63 |
| synthetic_recovery.csv | ea4976b358917394391fb402d79fb5111abb00c26ba95bf12576d038e86143ac |
| diagnostic_model_fits.csv | de3688a32e517a07818f42524fcc333b4c6eb626a8fd6ce03465612cc64a3ae5 |
| diagnostic_curves.npz | 74bf1b8b64383d1ccbe68c02185cc6fe91a2c11313bc29828a56e7e8b981d220 |

## 最后回答

1. **MC范围是否synthetically identifiable？** 当前physical-pixel/FOV→解析amplitude DoG measurement没有通过既有recovery标准；理想解析oracle可恢复，二者必须区分。
2. **两个model diagnostic能否合理summary？** amplitude曲线可被低residual summary，但center/surround参数分解触界，rc随frequency subset显著改变；不能认定为稳定radius measurement。
3. **Orientation是否严重？** 对67#6是实质问题；67#7较小。未确定原论文正式orientation规则，也未选择或聚合成一个更有利的方向。
4. **能否冻结正式九-cell contract？** 不能；**GRATING_RC_NOT_READY**。
5. **具体失败是什么？** measurement侧是当前离散有限观测与连续解析DoG的systematic recovery mismatch；model-shape侧是surround退出/center-surround近重合，以及方向和frequency subset依赖。它们不等于RetiPath physiology不成立，也没有通过改模型救结果。

完成后停止。未读取external target rc，未运行正式九-cell comparison。
