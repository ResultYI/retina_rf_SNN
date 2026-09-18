# Croner–Kaplan center radius：当前空间采样的可识别性

## 运行前冻结的 measurement contract（2026-09-15）

本轮只使用 synthetic ground truth，禁止加载 RetiPath checkpoint、读取 movie/spike target、训练或修改模型、FOV、preprocessing、现有 assay。下面的参数与标准在 recovery bank 运行前冻结；结果随后追加。

**审计量**：D(r)=K_c exp[−(r/r_c)²]−K_s exp[−(r/r_s)²] 中的 1/e characteristic radius r_c，单位 degree。不是 support、σ 或 FWHM。令 I_c=πK_c r_c²=1、I_s=πK_s r_s²=ρ，固定总体生成尺度只是消除无关单位；拟合时两个 gain 均未知。

**来源范围**：[上一轮可行性报告](D:/PythonProject/retina_rf_SNN/docs/H1_CRONER_KAPLAN1995_FEASIBILITY.md) 的 PC 约0.020–0.075°、MC约0.052–0.195°，为当前偏心度相应文献分箱中的数字化范围，不是原始细胞概率分布。本轮 bank：0.015、0.020、0.027、0.035、0.045、0.052、0.060、0.075、0.100、0.139、0.195、0.240°；首尾为额外边界探针。

**观测与拟合定义**：当前项目尚无最终 Croner–Kaplan DoG fitter。本轮建立一个独立、可重复的候选定义：拟合完整、有符号的 spatial linear-response coefficients，采用未经归一化、未平方的普通 L2。这个目标通过 Parseval 等价于拟合完整的复数 discrete Fourier response；不是仅拟合幅度、少数 spatial frequencies、flash size tuning，亦不是现有 P(x)。未来若使用此 measurement，必须保持这个定义与离散观测算子；本轮不声称已复现论文的实际 grating/F1/optical protocol，也不把结果推广到信息更少的测量。

主观测算子把每个输入格点解释为一块 pitch×pitch 的常值 Weber pixel：其单位刺激响应为 Gaussian 在该 pixel 中的精确积分（erf，含 aperture 边界，不在 FOV 内重新归一化）。它是未来从 signed pixel Jacobian 测量连续空间尺度的一种明确 reconstruction convention，不是当前模型已经定义了唯一的连续 RF。另以点采样乘 pixel area 的算子作 sampling 对照，且始终先匹配生成/拟合算子。两者均不能由插值等同。

**采样 A**：17×17、pitch=0.05390625°、edge-to-edge=0.91640625°。无 surround 的 center-only 控制，比较 matched point 与 matched pixel-integral；加入相同 FOV、51×51、pitch/3 的纯 synthetic center-only 参照。此参照不修改任何真实输入或模型。Subpixel center 分别为(0,0)、(p/4,p/4)、(p/2,p/2)，51-grid 相位相对于自己的 pitch 定义。

**FOV / nuisance B**：在当前17×17上使用六组(r_s/r_c, I_s/I_c)：(3,.2)、(3,.85)、(7,.2)、(7,.55)、(7,.85)、(20,.55)。这些是从文献广泛范围选出的有限 nuisance stress bank，不是六组生理 cell 或概率分布；r_s 覆盖0.045–4.8°，包含完整、小幅以及严重截断。每个 r_c 运行三个内部相位；另外在(p/4,6.25p)放置近边界中心，与内部(p/4,p/4)配对，以相同相位单独检查 aperture 位置。所有中心均为预设合成位置，不来自模型响应。

主 DoG fit 自由估计 I_c、I_s、r_c、r_s，约束 r_s>r_c；同时以已知真实 surround 的 oracle subtraction 重拟合同一带噪数据，分开记录 center sampling 与截断 surround nuisance 的影响。Oracle 不是可用于真实 assay 的方法。没有把 FOV 外真实信号补回主 fit，也没有使用拓展模型 FOV 的结果。近边界条件只跑 noiseless 与2% noise。

**拟合条件**：本审计固定正确 center 坐标，允许估计半径/gain。结论是已知 registration、真实径向 DoG、完整 signed response 的条件性上界；没有验证自由拟合 center、真实非DoG RF 或实际 model Jacobian 的误差。SciPy bounded least_squares，float64，三个固定初始化，不用 true radius 初始化，不按结果重启。每个初始化最多250次函数评估，ftol/xtol/gtol=1e−11。r_c范围[0.005,0.4]°，r_s−r_c范围[0.001,8]°，两 gain 范围[1e−6,100]；不把文献 class 标签输入 fitter。选最低残差解，不依据 radius recovery 选解。Noiseless 另记录用 point fitter 错配 pixel-integrated response 的后果。

**最小噪声**：每系数独立 Gaussian error，SD=该 noiseless sampled response 最大绝对值的0、0.5%、2%；有噪声每case 5次，固定seed19950107。噪声对不同geometry的意义是相同峰值相对精度，不是相同总光子数、刺激重复次数或生理噪声。此 bank 不估计真实assay noise，也不按当前RetiPath表现校准误差。

**预先固定的标准**：noiseless成功需r_c相对误差≤0.1%；带噪成功需≤10%，并且优化器返回成功、r_c不在搜索bounds。分别报告数值失败与准确度失败，不把两者混为一谈。某radius的带噪“可恢复”要求：三个内部相位各自成功率≥90%，各相位median relative bias绝对值≤5%；针对主DoG fit，各相位在六组nuisance×5次噪声上汇总。Measurement floor定义为测试bank中最低、且到0.195°为止所有更大测试radius均满足标准的值；不在测试点之间声称精确阈值。0.240°只作边界探针。

**区分/塌缩**：对每个相位和nuisance，按true radius排序，比较相邻值。报告median fitted radius的log间距与true log间距之比；≤0为顺序反转，(0,.5)为至少一半差异被压缩。若true相差≥20%但median fitted相差≤5%，标记collapse。经验噪声分布重叠另行报告；仅凭两个均值不同不能宣告可区分。这些是synthetic measurement标准，不是生理effect threshold。

可重复实现：[analysis-only fitter](D:/PythonProject/retina_rf_SNN/work/croner_kaplan_center_identifiability.py)。冻结输入、生成规则、源码hash及受保护文件hash保存于output/evaluations/croner_kaplan_center_identifiability_20260915/protocol.json。原始结果保存所有true/fitted参数、noise seed、误差及失败状态；synthetic_bank.npz保存无噪观测，噪声可按seed精确重建。本轮不保存任何模型checkpoint。

## 结果：RC_PARTIALLY_IDENTIFIABLE

### 完成与正确性

固定396个 synthetic case，产生3996条带true/fitted参数的recovery记录；其中主DoG内部位置2376条、近边界432条、center-only三种观测共1188条。主结果之外的surround oracle及noiseless错配fit在同一行记录，没有追加seed或响应条件。运行约86秒，不是训练时间。

独立Gauss–Legendre积分对erf pixel积分的最大绝对误差为1.39×10⁻¹⁶；Parseval检查通过；基本center-only和DoG recovery均通过。受保护的canonical模型、H1、前端、external assay、template和前一轮报告的hash与lock一致。未加载任何RetiPath或training模块。

完成后从保存的bank、noise seed与fitted参数独立重建全部3996条观测及残差：SSE与原记录的最大差异6.94×10⁻¹⁷；396个case的重复数全部符合lock。该检查没有重新拟合。

全部3996行有66行solver未返回成功，其中37行半径误差>10%，29行半径误差≤10%；按预先标准均标为失败。**solver失败不是结构不可识别的证明**；也没有通过增加迭代或按ground truth选取另一解“救回”这些记录。

### Noiseless：不存在仅由一个pixel pitch决定的绝对半径下限

匹配pixel-integral且已知surround/center-only时，当前17-grid的全部测试半径，包括0.015°，都能在此有限bank中恢复。精确参数模型能从subpixel敏感度中获得信息；不能据“直径不到两个pixels”直接证明严格不可识别。

自由估计surround的主DoG结果如下；每行6 nuisance×3 phases。0.027°及以上均恢复到约浮点精度，0.015/0.020°存在例外。

| true r_c (°) | 最大绝对相对误差 (%) | recovery失败 / 18 | solver失败 / 18 |
|---|---:|---:|---:|
| 0.015 | 6.62e+1 | 2 | 0 |
| 0.020 | 2.80e+1 | 1 | 1 |
| 0.027 | 3.80e-12 | 0 | 0 |
| 0.035 | 1.78e-13 | 0 | 0 |
| 0.045 | 5.09e-13 | 0 | 0 |
| 0.052 | 1.79e-11 | 0 | 0 |
| 0.060 | 9.25e-14 | 0 | 0 |
| 0.075 | 4.39e-12 | 0 | 0 |
| 0.100 | 1.53e-13 | 0 | 0 |
| 0.139 | 4.99e-13 | 0 | 0 |
| 0.195 | 8.54e-14 | 0 | 0 |
| 0.240 | 1.39e-13 | 0 | 0 |

例外分解：

- 0.015°、phase=(p/2,p/2)、r_s/r_c=3：两个gain条件得到约0.00507/0.01178°，normalized SSE却仅1.60×10⁻¹⁶/1.27×10⁻¹⁵。不同半径与略变的surround/gain能产生极相近离散系数，体现强烈病态性，而非单纯查表分辨率。
- 0.020°、相同phase、r_s=0.060°、ρ=.85：一个fit达到固定预算但未收敛，r̂_c≈0.01440°；该失败不能被写成0.020°在数学上绝不可能恢复。真实参数的残差为零，oracle subtraction可恢复0.020°。
- 上述small-radius例外的surround均几乎100%落在FOV内，**不能归因于surround截断**。它们暴露的是小center、subpixel相位、自由nuisance和有限拟合精度共同造成的脆弱性。

因此表中“noiseless bank门槛”仅是已冻结fitter通过整个bank的边界，不是不可突破的sampling theorem，也不是直接套用于float32 RetiPath的界限。本次计算为float64，未验证未来有效RF提取/连续重建的数值误差。

### 0.5% noise：主 DoG、当前17×17、三个内部相位

每行90次（6 nuisance×3 phase×5 draws）。Bias为mean(r̂_c−r_c)，相对bias为mean relative error；MARE为mean absolute relative error。失败率包含半径错误、bound及solver未成功；全部结果保留，不删失败fit。

| true r_c (°) | Bias (°) | 相对bias (%) | MARE (%) | P90绝对相对误差 (%) | 失败率 (%) | solver失败数 | 逐phase标准 |
|---|---:|---:|---:|---:|---:|---:|---|
| 0.015 | 7.14e-4 | 4.76 | 27.24 | 75.83 | 50.00 | 6 | 未通过 |
| 0.020 | -1.32e-4 | -0.66 | 16.26 | 49.11 | 32.22 | 0 | 未通过 |
| 0.027 | -8.06e-4 | -2.98 | 4.55 | 6.82 | 8.89 | 0 | 未通过 |
| 0.035 | -2.88e-5 | -0.08 | 0.74 | 1.34 | 0.00 | 0 | 通过 |
| 0.045 | -2.95e-5 | -0.07 | 0.53 | 1.12 | 0.00 | 0 | 通过 |
| 0.052 | 1.79e-5 | 0.03 | 0.37 | 0.74 | 0.00 | 0 | 通过 |
| 0.060 | -4.64e-5 | -0.08 | 0.34 | 0.70 | 0.00 | 0 | 通过 |
| 0.075 | 1.96e-5 | 0.03 | 0.27 | 0.52 | 0.00 | 0 | 通过 |
| 0.100 | 4.70e-6 | 0.00 | 0.22 | 0.42 | 0.00 | 0 | 通过 |
| 0.139 | -4.59e-6 | -0.00 | 0.18 | 0.32 | 0.00 | 0 | 通过 |
| 0.195 | 9.58e-5 | 0.05 | 0.19 | 0.45 | 1.11 | 1 | 通过 |
| 0.240 | 3.66e-4 | 0.15 | 0.29 | 0.56 | 2.22 | 2 | 通过 |

### 2% noise：主 DoG、当前17×17、三个内部相位

每行90次（6 nuisance×3 phase×5 draws）。Bias为mean(r̂_c−r_c)，相对bias为mean relative error；MARE为mean absolute relative error。失败率包含半径错误、bound及solver未成功；全部结果保留，不删失败fit。

| true r_c (°) | Bias (°) | 相对bias (%) | MARE (%) | P90绝对相对误差 (%) | 失败率 (%) | solver失败数 | 逐phase标准 |
|---|---:|---:|---:|---:|---:|---:|---|
| 0.015 | 1.99e-3 | 13.26 | 40.81 | 102.02 | 73.33 | 3 | 未通过 |
| 0.020 | 9.72e-4 | 4.86 | 24.52 | 57.67 | 52.22 | 5 | 未通过 |
| 0.027 | -7.60e-4 | -2.81 | 15.48 | 56.75 | 33.33 | 3 | 未通过 |
| 0.035 | -1.56e-4 | -0.45 | 5.52 | 12.27 | 15.56 | 1 | 未通过 |
| 0.045 | 2.37e-4 | 0.53 | 2.21 | 5.00 | 1.11 | 0 | 通过 |
| 0.052 | -1.81e-6 | -0.00 | 1.48 | 3.42 | 1.11 | 1 | 通过 |
| 0.060 | 2.39e-4 | 0.40 | 1.62 | 3.16 | 2.22 | 1 | 通过 |
| 0.075 | 2.40e-4 | 0.32 | 1.32 | 2.13 | 2.22 | 2 | 通过 |
| 0.100 | 5.28e-4 | 0.53 | 1.01 | 1.81 | 2.22 | 1 | 通过 |
| 0.139 | 6.81e-4 | 0.49 | 1.13 | 1.82 | 4.44 | 4 | 通过 |
| 0.195 | 1.08e-3 | 0.55 | 0.97 | 1.89 | 4.44 | 4 | 通过 |
| 0.240 | 1.76e-3 | 0.73 | 1.20 | 2.80 | 5.56 | 5 | 未通过 |

### 采样floor及PC/MC范围

| 测量/拟合条件 | noiseless bank门槛 (°) | 0.5% noise (°) | 2% noise (°) |
|---|---:|---:|---:|
| Center-only，17点采样匹配 | 0.020 | 0.035 | 0.045 |
| Center-only，17 pixel-integral匹配 | 0.015 | 0.027 | 0.045 |
| Center-only，51 pixel-integral，同FOV | 0.015 | 0.015 | 0.015 |
| 主DoG，17 pixel-integral，自由surround | 0.027 | 0.035 | 0.045 |

主DoG的保守floor为：0.5% noise约 **0.035°**；2% noise约 **0.045°**，即约0.65/0.83个当前pixel pitch。只能把过渡定位在已测试的 **(0.027,0.035]°** 与 **(0.035,0.045]°**，不把未测试的中间半径认作已通过。

Phase依赖很强：2% noise下，r_c=0.020°的phase=0、1/4、1/2分别失败11/30、6/30、30/30；0.035°分别2/30、0/30、12/30；到0.045°为1/30、0/30、0/30。平均bias接近零并不表示小半径可靠：正负误差相消时MARE和失败率仍高。

- **PC范围0.020–0.075°：部分可恢复。** 七个PC范围内测试值中，0.5% noise有2/7未通过（0.020、0.027）；2% noise有3/7未通过（再加0.035）。以首个通过值作为保守边界，分别约27%/45%的**线性radius区间长度**处于边界以下；这是区间占比，不是13个PC cells中的比例，更不是论文population概率。真实过渡区未加密测试。
- **MC范围0.052–0.195°：本合同下全部六个测试值通过逐phase标准。** 2% noise、等权测试半径/条件汇总的MARE约1.25%，失败率2.78%；主要失败包含solver状态，不能直接当生理测量错误率。没有证明所有未测试半径或所有未知真实noise条件均通过。
- 同FOV的51×51 center-only参照在全部三个noise水平上可恢复到bank下界0.015°；证明提高采样密度能改善这个问题。**本轮未运行51-grid的完整DoG nuisance bank**，因此它不是“51×51正式测量已验证”的结论，也不包含模型FOV/预处理升级。

### FOV与surround nuisance：不能和center sampling混为一谈

自由surround fit与oracle在同一观测/同一噪声上比较：2% noise时，r_c=0.027°的MARE从主fit 15.48%降至oracle 10.19%；0.035°由5.52%降至2.30%；0.052°由1.48%降至1.20%。Oracle改善反映未知surround带来的额外不确定性；它既包括component分离，也包括截断导致的信息缺失，**不是单独FOV效应的无偏估计**。

直接的aperture位置对照保留同样pitch与1/4相位，仅把中心移至(p/4,6.25p)。Noiseless近边界DoG全部恢复成功；2% noise下较小半径仍不稳定，但未观察到一个会将整个MC range系统性推向错误r_c的截断偏差。对照中的noise independently seeded，比较是有限bank的描述，不作显著性检验。知道正确有限aperture算子的fit，不必仅因surround尾部不可见就把center估错。

不过 **r_s本身仍高度不可靠**。在0.020–0.195°测试范围、2% noise下：

| 位置 | r_s/r_c | FOV内surround质量范围 | r_s median绝对相对误差 | r_c median绝对相对误差 |
|---|---:|---:|---:|---:|
| 内部 | 3 | 53.4–100% | 9.35% | 1.59% |
| 内部 | 7 | 13.3–100% | 23.79% | 1.26% |
| 内部 | 20 | 1.7–80.1% | 94.36% | 1.13% |
| 近边界 | 3 | 43.0–99.8% | 8.32% | 1.53% |
| 近边界 | 7 | 12.6–89.0% | 18.73% | 0.91% |
| 近边界 | 20 | 1.7–59.3% | 87.47% | 1.32% |

这些汇总不平衡于真实生理分布；高r_s误差也包含gain–radius trade-off与solver上界。主要结论是：**center可恢复不代表surround/gain已可验证；surround被截断也不自动意味着center无法恢复。** 本轮没有推翻前一轮surround coverage问题，也未使用模型support替代外部RF radius。

### 不同true radius的塌缩与M>P差异

每个noise水平比较180个相邻radius×phase×nuisance组合（上限0.195°）。Noiseless无预定义collapse；0.5% noise有2个collapse、6个至少一半log间距压缩；2% noise有11个至少一半压缩，虽未落入更窄的“5%同值”collapse判据，但24/180对的经验10–90%区间发生重叠（0.5%时13/180）。这类经验区间仅5次draw，不是置信区间；无collapse也不代表可区分。

具体地，0.5% noise、半pixel相位、r_s/r_c=7时，true 0.020°和0.027°的median fitted radius可分别约0.02556°和0.02677°；35%的真实差异被压到约4.7%。这直接展示small-PC半径会合并为相似估计。

还存在一个完全不同的**观测算子错误**：把pixel-integrated系数直接当作point samples来拟合，即使无噪，也会抬高小center估计：

| true r_c (°) | 错用point fitter得到的r̂_c范围 (°)，3 phases |
|---|---:|
| 0.015 | 0.02251–0.02584 |
| 0.020 | 0.02557–0.03028 |
| 0.027 | 0.03298–0.03570 |
| 0.035 | 0.04076–0.04194 |
| 0.045 | 0.05007–0.05049 |
| 0.052 | 0.05655–0.05675 |
| 0.075 | 0.07824–0.07824 |
| 0.100 | 0.10243–0.10243 |
| 0.195 | 0.19624–0.19624 |

例如true 0.027/0.052°的真实比约1.926，错误point fit可将比率压到约1.59–1.72。这里出现的约0.023–0.030°小半径偏移是算子错配，不是使用正确pixel-integral fit后仍必然存在的生理尺度。

正确matched fit对接近两倍的PC/MC差异通常保留方向。例如2% noise时，true 0.035/0.075°（比2.143）的18个匹配phase/nuisance组合，median fitted比的中位数约2.135，范围1.819–2.464，方向18/18保持；但小PC的绝对radius仍未达到全phase标准。对于true 0.027/0.052°，比值范围可扩大到1.727–6.171，说明“仍然M>P”与“定量比率正确”是两件事。这里只描述固定bank已有点的对比，不把M/P标签输入fitter。

**Sampling floor本身不会在相同真值、相同noise/phase/nuisance分布下凭空知道哪个是M或P。** 它可以压缩已有差异；若两个class的subpixel位置、测量误差或筛选条件不同，也可能产生差异偏差。当前synthetic结果不证明未来模型的M>P是生理证据，更不排除模型已有type-specific support先验的影响。

### 适用范围与原始证据

这是候选measurement的conditioned recovery test，**不是模型validation**。假设已知准确center、径向DoG、完整signed linear coefficients、匹配pixel reconstruction与指定iid噪声；真实model RF可能非DoG，背景、history、temporal frequency和光学的对齐均未解决。使用稀疏grating amplitude/F1、自由center或现有normalized P(x)时，不能继承本floor。噪声floor来自预设误差水平，未测量实际assay的noise或model mismatch。

- [冻结合同与case定义](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_center_identifiability_20260915/protocol.json)
- [逐次synthetic recovery raw results](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_center_identifiability_20260915/recovery.csv)：bias可由r_c_hat−rc重算，所有失败行保留；oracle及错配结果单列。
- [无噪signed response bank](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_center_identifiability_20260915/synthetic_bank.npz)：396个array；有噪观测由对应case、noise_sd与noise_seed精确重建，未存重复大数组。
- [独立实现](D:/PythonProject/retina_rf_SNN/work/croner_kaplan_center_identifiability.py)：仅NumPy/SciPy/stdlib；使用既有隔离环境，未安装或升级依赖。拟合行为依据[SciPy least_squares文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)。

## 最终回答

- **PC range是否可定量恢复？** 只能部分。当前matched、已知center的合同下，0.5% noise保守边界约0.035°，2%约0.045°；PC低端不能整体认定为可靠，noiseless理想恢复也不能保证真实measurement稳健。
- **MC range是否可定量恢复？** 在此次噪声、nuisance及已知center条件下，0.052–0.195°的测试值均通过；这是条件性可行，不能升级为正式RetiPath验证通过。
- **当前grid的measurement floor在哪里？** 带噪实用floor约0.035–0.045°，依赖算子、noise、phase、surround估计与fitter。没有“pitch=0.05390625°，所以所有更小radius绝对不可识别”的结论。
- **是否需要在正式physiology validation前提高spatial resolution？** 若目标是覆盖完整PC范围的稳健定量验证，已有优先提高采样密度的依据；MC或PC较大radius的限定验证不必因此一律停止。但本轮不能证明提高resolution是唯一必要措施，也未验证升级后的完整测量协议。不能直接沿用现有grid宣称整个PC范围都可靠。本轮完成后停止。
