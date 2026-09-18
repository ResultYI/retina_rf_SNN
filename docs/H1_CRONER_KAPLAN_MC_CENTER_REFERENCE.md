# Croner–Kaplan 1995 M-cell center：external reference freeze

## 拟合前冻结的选择（2026-09-15）

本轮仅处理公开外部physiology数据和9个目标cell的eccentricity catalog。禁止加载或运行RetiPath checkpoint、RF、assay response、natural movie或spike target；不训练、不改模型、FOV、前端或support。

固定ISETBio commit：**d1619948425056efa5435c91932bca5b3a30f4a7**。数据：[croner_kaplan_parasol_rgc.mat](https://github.com/isetbio/isetbio/blob/d1619948425056efa5435c91932bca5b3a30f4a7/data/datafiles/rgc/croner_kaplan_parasol_rgc.mat)，Git blob **c850d74d663ff59118ccc9ade935331cdb4a846d**，使用完整d1数组，保留原始顺序及所有点，不重新数字化或按模型表现删改。

**找到了可直接重现的ISETBio项目拟合定义。** [data_rgcEccData.m:79–98](https://github.com/isetbio/isetbio/blob/d1619948425056efa5435c91932bca5b3a30f4a7/tutorials/data/underDevelopment/data_rgcEccData.m#L79-L98)加载同一d1数组，对全部M-center点做radius–eccentricity线性OLS。它位于underDevelopment tutorial，不能称为Croner/Kaplan作者发表或推荐的公式。本轮未核实到原论文作者单独给出的M-center fit；[constants.m](https://github.com/isetbio/isetbio/blob/d1619948425056efa5435c91932bca5b3a30f4a7/ganglioncells/%2BRGCmodels/%2BCronerKaplan/constants.m)中也没有M-center–eccentricity公式，不能借用其中P或surround关系。

根据本任务“已有作者或项目明确给出的定义优先”的规则，**采用这个已有项目定义，不触发后备log-linear，也不拟合其他函数进行择优**：

**r_ref(e)=a+b·e，OLS在radius尺度上进行。**

原代码先令x_mm=0.2253·e，y_diameter_um=2·1000·0.2253·r_c，再做y=A+B·x的OLS。因此转换回degree的a=A/450.6、b=B/2000，与直接以全部原始degree radius对eccentricity做OLS完全等价；不把Dacey dendritic-field数据混入。源码中“Microns to degrees”注释与实际运算/坐标标签方向不一致，本轮依据实际变换并做数值等价核对；该比例在恢复degree拟合时抵消。

r_c仍为1/e Gaussian characteristic radius（degree），不是diameter、sigma、support或FWHM。保存的expected_r_c为参考曲线值a+b·e；expected_log_r_c=ln(a+b·e)，明确是**reference radius的log**，不声称它等于E[ln r_c|e]。这与后备log-linear的参数语义不同，JSON将显式标记。

## 拟合前冻结的不确定性描述

- 使用全部原始M-center点，等点权重、含截距普通最小二乘。保存radius residual及natural-log residual、两种RMSE、median absolute log residual、每点leave-one-out residual；LOO保留同一函数形式，不选模型。
- Bootstrap固定NumPy PCG64 seed **19950115**，**10000次**，按完整(e,r_c)点对有放回重采样，每次仍做同一OLS；不按target eccentricity重采样或裁点。保存全部(a,b) draws及95% percentile reference bands。
- 另描述external scatter：保存全部LOO log residual及其2.5/97.5 percentiles。逐target的reference-plus-scatter range由每个bootstrap reference log值加一个有放回抽取的LOO log residual得到，仍为10000 draws；不把它作为PASS/FAIL范围。该描述假设log residual散布可跨所观测eccentricity使用，不检查其他函数以改善拟合。
- 点数据缺少animal ID、逐点测量与digitization误差。本轮的point bootstrap不是animal-level uncertainty；残差包含biological variability、digitization/measurement error和reference misspecification，不能宣称已分离出纯biological scatter。
- Target cohort仅按用户列出的9个cell读取catalog eccentricity，绝不读取其模型radius。95%区间是描述性reference/预测范围，不是正式comparison metric或接受阈值。

## 冻结结果

状态：**FROZEN_EXTERNAL_ONLY**。保留全部36点，删除0点、手工修正0点，原始数组中重复行0；CSV中source_index保持MAT原顺序。数据eccentricity为 **1.455989–21.943729°**，r_c为 **0.051728–0.291636°**。这是digitized点集的覆盖，不是原论文总细胞数或动物数。

原MAT SHA256：

**2b28209d07ccd3d5a413da56ba4f93851a92e9b2d4ec27d2cfb581f50f97834c**

原MAT的Git blob校验通过。reference_fit.json中嵌入原MAT base64，保存source commit/blob、数组hash、原始源码片段和全部bootstrap系数draws；无需再次下载或重新数字化即可核对。外部点CSV保留每点source URI/hash、原值、residuals和LOO结果。

### 已有项目定义的外部拟合

$$r_{ref}(e)=0.03254444445050388+0.011249762233383602\,e\quad (\mathrm{degree}).$$

| 参数 | 冻结值 | 95% point-bootstrap percentile interval |
|---|---:|---:|
| a，intercept (°) | 0.03254444445050388 | 0.02106717–0.04296790 |
| b，radius degree / eccentricity degree | 0.011249762233383602 | 0.00980586–0.01251740 |

**没有拟合后备的ln(r_c)=a+b·e。** 上述a/b是radius尺度参数，不能作为log-linear参数引用。原tutorial单位下的OLS与此degree形式预测最大差异5.55×10⁻¹⁷°；保留的是同一个函数定义，并非拟合后另选形式。该项目定义的存在已在本轮计算任何拟合结果前记录。

log均指ln(r_c / 1 degree)。本表reference radius是OLS曲线位置；expected_log只是它的log，既不是另一个拟合，也不构成对真实条件log均值的无偏估计。

### 外部散布与LOO

| 描述量 | 值 |
|---|---:|
| Radius residual RMSE | 0.022689° |
| Natural-log residual RMSE | 0.210977 |
| Median absolute log residual | 0.134780 |
| LOO radius residual RMSE | 0.024059° |
| LOO log residual RMSE | 0.221521 |
| LOO median absolute log residual | 0.140653 |
| Log residual 2.5 / 50 / 97.5 percentiles | -0.459070 / -0.006498 / 0.377921 |
| LOO log residual 2.5 / 50 / 97.5 percentiles | -0.470666 / -0.005362 / 0.405902 |
| LOO multiplicative residual 2.5 / 50 / 97.5 percentiles | 0.624586 / 0.994652 / 1.500655 |

Median absolute log residual约0.135，对应约1.14倍的典型乘性偏差大小。LOO散布的中间95%约为参考位置的 **0.625–1.501倍**，即约−37.5%至+50.1%。这说明外部点本身有明显散布，不能将单cell偏离参考线直接当作模型失败。

这里的scatter并未分解为纯biological variability；digitization、原实验measurement uncertainty、monkey间差异和reference形式误差都混在其中。未按真实动物进行bootstrap，未建立测量误差模型，未删异常点。10000次重采样只描述这36个点对当前reference的敏感性，不能把很窄的reference band解释成同等精度的生理定律。

**Reference band与prediction range不同**：前者只包含bootstrap曲线不确定性；后者再加入原始LOO log散布，明显更宽。其描述公式为
ln r_draw(e)=ln[a_boot+b_boot·e]+ε_LOO,draw。
LOO residual未中心化、截断或筛除。log散布被合并用于全部观测eccentricity，是预先声明的简单描述假设；不宣称具有独立动物层面的95%预测覆盖保证。

### 九个目标eccentricity：未读取任何model radius

仅读取[data/schottdorf_lee_catalog.py](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_catalog.py)中的class与eccentricity，重复recordings只用于核对同cell数值一致。9/9都是MC；3.52–13.00°全部落在外部点集的直接数值覆盖范围内。以下数值是纯外部reference lookup，**不是RetiPath测量或预测的RF半径**。

| Cell | e (°) | expected log r_c | reference r_c (°) | reference 95% point-bootstrap band (°) | reference + scatter 95% range (°) |
|---|---:|---:|---:|---:|---:|
| 67#6 | 6.72 | -2.22430 | 0.10814 | 0.10091–0.11503 | 0.06617–0.16677 |
| 67#7 | 7.36 | -2.15985 | 0.11534 | 0.10793–0.12238 | 0.07066–0.17805 |
| 67#33 | 4.73 | -2.45625 | 0.08576 | 0.07810–0.09276 | 0.05242–0.13245 |
| 68#3 | 5.08 | -2.41136 | 0.08969 | 0.08216–0.09658 | 0.05484–0.13834 |
| 68#10 | 4.89 | -2.43548 | 0.08756 | 0.07996–0.09449 | 0.05352–0.13508 |
| 69#4 | 13.00 | -1.72154 | 0.17879 | 0.16625–0.18947 | 0.10968–0.27667 |
| 69#6 | 3.88 | -2.57448 | 0.07619 | 0.06801–0.08354 | 0.04655–0.11745 |
| 69#7 | 3.52 | -2.62910 | 0.07214 | 0.06368–0.07969 | 0.04402–0.11121 |
| 70#34 | 5.66 | -2.34114 | 0.09622 | 0.08887–0.10299 | 0.05889–0.14849 |

CSV另保存两类区间的log端点及metadata hash，避免依赖此表显示精度。没有把外部范围裁到当前模型FOV、support或synthetic测试范围。

仍有一个坐标语义缺口：当前catalog只有scalar eccentricity；前序可行性审计指出论文使用temporal-equivalent eccentricity。**数值范围覆盖已确认，两个eccentricity坐标定义的精确等同尚未确认。** 本轮按catalog数值原样lookup，不猜测nasal/temporal转换或改写metadata。这一限定记录在每个target行中，必须在正式comparison解释前处理。

### 图与完整冻结文件

![External M-center data and frozen reference](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_mc_center_reference_20260915/external_reference.png)

橙点为全部36个外部digitized点，蓝线为已有ISETBio定义的OLS reference，蓝带为point-bootstrap reference uncertainty，灰带为reference加LOO scatter。图上没有任何RetiPath输出，区间均不是接受阈值。

- [external_points.csv](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_mc_center_reference_20260915/external_points.csv)
- [reference_fit.json](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_mc_center_reference_20260915/reference_fit.json)
- [mc_target_eccentricities.csv](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_mc_center_reference_20260915/mc_target_eccentricities.csv)
- [external_reference.png](D:/PythonProject/retina_rf_SNN/output/evaluations/croner_kaplan_mc_center_reference_20260915/external_reference.png)

reference_fit.json SHA256：**a048e069f0a9a03bc996895f336586d90ffd21f35fb3f71919225fcd3647e5eb**。该JSON保存另外三个artifact的hash。36点CSV读回与原数组逐值完全一致；逐点LOO重新拟合与hat-matrix恒等式最大差异9.71×10⁻¹⁷°。固定bootstrap中无秩退化或非正reference draws。

完成后从冻结CSV与seed重新计算全部10000组bootstrap系数，最大差异为0；9个target的reference-plus-scatter区间亦全部精确重现，三个artifact的hash核对通过。该验证只读取本任务外部artifact。

## 最后回答

1. **是否存在可直接采用的published/author-provided M-center fit？** 本轮未核实到Croner/Kaplan作者发布的M-center公式；但固定ISETBio项目源码存在明确、可重现的radius-linear OLS定义。按任务的项目定义优先规则采用，并保留其tutorial来源等级。
2. **冻结参数是什么？** 已有项目定义为r_c=0.03254444445+0.01124976223·e，均按degree计。后备log-linear未触发、未尝试；不能将此a/b解释成log-linear系数。
3. **9个MC eccentricity是否全部在直接覆盖范围内？** 数值上9/9均在1.455989–21.943729°内，无需外推；但eccentricity坐标语义仍需统一。
4. **External biological scatter大约多大？** 观测log RMSE≈0.211，median absolute log residual≈0.135；LOO中间95%约为reference的0.625–1.501倍。这是包含biological、measurement、digitization与reference误差的外部散布，不能称纯生物变异或动物层面CI。
5. **是否存在阻止下一步正式测量的理由？** 外部数据完整性没有新增阻碍。**目前仍不能直接进入带正式判读的测量**：按用户顺序，须先在查看任何RetiPath r_c前冻结comparison metric，并明确eccentricity坐标对应及正式r_c measurement的响应/拟合定义。当前reference已为这一步准备好；本轮没有设置PASS/FAIL threshold，也没有开始模型测量。
