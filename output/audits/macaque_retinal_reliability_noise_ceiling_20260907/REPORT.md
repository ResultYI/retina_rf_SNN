# Phase R1 — retinal reliability / noise-ceiling audit

## 1. 当前22 cells中有多少能进入repeatability/noise-ceiling分析？

**20/22 有可读取且结构验证通过的 6×1 min raw recording，共120个 live repeats。** 分组为 MC ON 4、MC OFF 4、PC ON 9、PC OFF 3。`67#4`、`68#10` 没有本地公开 raw repeat 文件，不能进入分析；这不表示实验中从未记录过这些细胞的 repeats。

`68#4 / lSS01184` 的 README 表格写作10 min，但 raw header、六列重复数据及作者详细记录清单均支持6×1 min；保留并记录这个差异。时间精度0.1 ms，每个live repeat为60 s，末尾第七列不作为repeat。`usable_for_reliability` 仅表示原始数据可用，**不是可靠性方法已经验证**；当前实际完成noise-ceiling评分的细胞数为0。

## 2. 作者官方 retinal reliability方法是否成功复现？

**UNVERIFIED，未达到 REPRODUCED 或 PARTIAL。** 一手论文确认1 ms bins、八级低通滤波、MC 2 ms / PC 4 ms，以及60 s重复响应之间的相关分析。但必要的离散滤波实现、初始化/边缘处理、具体trial配对及汇总规则未能从已检查的一手来源中恢复。[原论文 Methods 与 Results](https://pmc.ncbi.nlm.nih.gov/articles/PMC8998785/)

官方仓库 `SimpleLowpass` 是模型函数；没有证据表明它就是产生原始spike reliability数据的处理程序。`Piecharts.ipynb` 使用来自作者、已舍入的常数，并非raw-repeat估计程序。缺失实现选择不能通过挑选一个结果接近论文的算法来补齐。按R1第4节，已在此停止。[官方仓库](https://gin.g-node.org/Manuel/Macaque-ganglion-cells)

## 3. 这些cells的retinal reliability是多少？

**未计算，不能给出当前cohort的可靠性均值、中位数或分组分布。** 原论文Table 2的更大样本中，类别均值范围0.58–0.81；不能移植成当前20 cells的结果，也不能据此宣称量级复现。每个repeat的原始live spike times和计数已保存；未生成未验证的smoothed response。

## 4. Aligned Canonical的stimulus-only prediction捕获了多少reliable response？

**UNVERIFIED / NOT RUN。** 已从指定fixed-alignment audit的真实manifest解析22个checkpoint，并交叉核对hash。未执行 `STIMULUS_ONLY_ZERO_HISTORY` inference；捕获比例未知，不等于零。

## 5. LN和CNN分别捕获多少？

**均为UNVERIFIED / NOT RUN。** 既有LN与CNN各22个frozen checkpoint文件hash匹配；没有缺失模型触发训练，也没有加载checkpoint张量。它们的response correlation和合法normalization尚未计算。

## 6. Canonical离noise ceiling还有多大gap？

**未知。** 没有通过验证的repeatability estimator和对应repeat-mean ceiling，不定义或填写gap。未把model与repeat mean的相关直接除以任意repeat-pair相关，也未裁剪归一化数值。

## 7. Canonical与CNN/LN的差距相比ceiling gap有多大？

**未知。** 模型分数、模型差值和paired-cell bootstrap均未运行。输出表中的空值和 `NOT_RUN_OFFICIAL_PROTOCOL_UNVERIFIED` 是停止状态，不是模型测量值。

## 8. stimulus-only与formal conditional prediction差多少？

**未知，两种contract均未评分。** 前者须全零observed history；后者允许各repeat自己的strictly-past history，不能用于primary retinal-reliability normalization。没有把既有正式conditional NLL移植成stimulus-only reliability分数。

## 9. remaining ceiling gap是否与multi-spike burden / event rate有关？

**尚不能判断。** gap尚未合法定义；没有计算相关系数、history contribution或reliability分层。原始repeat spike counts仅用于记录完整性核验，不作为新的observation-model结论。

## 10. 最终research-navigation verdict是什么？

**UNRESOLVED。** 原因是official reliability protocol未达到可复现标准；不是因模型已接近ceiling，也不是Canonical特异失败。20个可用repeat recordings均出现在当前aligned训练合同的recording IDs中；这批数据不得称为untouched predictive test。已消费的 `[20,60)` 没有被当作新test。

本轮training、parameter fitting、model inference、new seed、checkpoint/model/front-end/center/production修改、pathway/illusion/synthetic分析均为0。所有本轮derived记录和输入hash见 `evidence_manifest.json`。后续数值阶段因明确STOP条件未执行；不进入count likelihood或新architecture。

**我们目前仍不知道是否存在足够大的、值得继续优化模型的预测空间，也不能把剩余限制归因于core representation、front-end/data或observation likelihood/history。** 本轮证据不足以作出这些研究方向判断。
