# Independent RF-center recovery for the frozen 22-cell lineage

日期：2026-09-05。结论：**B — FUNCTIONAL ALIGNMENT SUPPORTED, BIOLOGICAL CENTER UNRESOLVED**。

本次从可读取的官方资料中恢复到当前 22 cells 的合格 author-derived RF center 为 **0/22**。这不等于证明作者从未保存或公开过这些中心：统计补充 workbook 的实体仍未能取得，其内容为 `UNVERIFIED`。本轮训练次数为 **0**，RF-center 重新计算次数也为 **0**。

## 1. 原实验是否假定 RF 完全位于 movie center？

实验主动进行过对准，但作者没有把实际误差视为严格为零。[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8998785/) Methods 的 Preparation and recording（本地 `paper_blocks.json` index 19）描述：先在距眼 114 cm 的 tangent screen 上定位 RF，再用 1 Hz flashing spots 将 movie center 对准 RF；spot 逐渐缩小至约 2 arcmin，调整显示位置以获得最大响应。作者明确承认有时仍有数 arcmin 的定位误差。

| 原论文位置 | 已重新核实的事实 | 适用范围 |
|---|---|---|
| Quantitative analysis of cell responses；P37、P41；blocks 78、85 | 对 peak response frame 做同心、圆对称二维 DoG 最小二乘拟合；拟合 x/y 表示相对 video center 的偏移；报告径向对称误差及 mean positional error 5.6 arcmin | 作者该段分析样本；没有当前 22-cell 的逐 cell 数据 |
| Reliability and correlation；P50；block 107 | Reverse correlation 可估计相对 movie center 的 RF 位置；PC mean 5.50、SD 4.81 arcmin；MC mean 8.15、SD 4.78 arcmin | 原论文 population statistics |
| Figure 6 caption；block 110 | cell separation 由 RF centers 估计；spatial/full-field 同 cell 比较使用 RC peak 与 central pixel 的距离 | 原论文图示分析 |

5.6 arcmin 与后文 PC/MC 分组均值来自不同段落；可读资料未给出足够的逐 cell 成员关系来统一重算它们，本报告保留原文各自语境。均值、SD、eccentricity 都不能生成或替代当前 22 个 x/y offsets。Retinal eccentricity 是相对视网膜中心的偏心度，也不等于相对 movie center 的 mis-centering。

## 2. 能否恢复当前 22 cells 的作者中心？

**当前可恢复的合格中心：experiment-time 0/22，author reverse-correlation/DoG 0/22。**

检索对象为 [官方数据 DOI](https://doi.gin.g-node.org/10.12751/g-node.xage77/) 所指向的 [Manuel repository](https://gin.g-node.org/Manuel/Macaque-ganglion-cells) 与本地官方 snapshot。在线主页 HEAD 与本地 HEAD 一致：`cffefb08c760f04c9a951da46061b361d2288e9b`；共清点 **116 个 tracked files**。Published DOI archive 的根目录也已检查，archive HEAD 为 `c779cb99368d958c37f5734a905e2946667993be`；未把根目录检查说成完整 archive payload 检索。

| 资料 | 实际检查 | 结果及排除理由 |
|---|---|---|
| README、Cell List.docx、CellsList.docx、raw spike `.txt` | 完整文本和字段；cell key、recording、duration、eccentricity、timing、spike counts、采集备注 | 能建立 recording/cell 身份；未找到 flashing-spot 的逐 cell 屏幕 x/y 坐标。`Repeat centered` 只是采集备注，没有坐标 |
| Fig2publication.xlsx | 下载 annex 实体；全部 4 sheets、文本、公式示例、defined names、chart XML、外部链接缓存 | 仅有标记为 122（MC ON）和 272（PC OFF）的 RF 示例；对应旧采集目录 ER67U18、ER69U22，均不属于当前 22 cells |
| Fig3publication.xlsx | 全部 7 sheets，以及同样的 workbook 内部证据 | RF 示例标记 101、292、231；101 对应旧目录 ER67U9，292 标记 ER70U25（原备注有问号），231 为 68#12 S-cone；均不属于当前 MC/PC 22 cells |
| Workbook 外部链接缓存 | `Fig3PCLum.xlsx`、`Fig5BlOn.xlsx` 的 sheet names 与所有缓存 cell 记录 | 前者只在 off272 缓存数值，`RF fit` sheet 缓存数为 0；后者缓存 231 map 与 time fits，178 sheet 无缓存。没有新增目标 MC/PC 中心。外部文件是作者原机器路径，并非可访问下载地址 |
| retinatools、所有 notebooks（含 checkpoint notebooks）、HPC scripts | 全部 Python/code/markdown cells 与文本输出 | 公开的是前向 prediction model 和模型影片生成；未找到 spike-triggered map、去卷积和二维 DoG RF-estimation 的完整实现 |
| run_model 的 CSV、firing-rate txt、5 个 PDF、HPC `.out` | 完整文本或逐页提取 | 响应时序、模型比较、贡献率、模拟输出；不是独立 RF-center table |
| `.npy`、`.npz`、`.pkl` | 完整文件名 inventory | 当前官方 snapshot 中不存在这些扩展名的文件 |
| Statistical_Summary_Document.xlsx / tjp14666-sup-0006-s08.xlsx | PMC 与 Wiley 官方链接、Europe PMC supplementary API、普通浏览器访问 | **实体内容 `UNVERIFIED`**：PMC 返回 HTML 下载验证页；浏览器先显示 reCAPTCHA、后为空白，未取得可读取 workbook；Wiley 返回 403；Europe PMC API 返回该文不属于其 OA supplementary API 的错误。不能声称已检索该 workbook 的字段 |

Fig2 原本只有 annex 指针。本轮新取得 20,606,799-byte 文件，其 MD5 为 `d23fd7777e29a2c74fb9221fd2d33b7f`，与官方指针一致；SHA256 为 `adfed0e6a534ab0a3b2b4a883306e1b4e948ec66a2f2bf98bf6e1f0cbed953d6`。原 snapshot 指针未改动。

两份 workbook 确实保存了作者 RF maps 和 `MxFr/MxX/MxY/Max` 正、负极值记录，不能概括成“作者没有公开任何 RF 数据”。12 条有明确这些字段的示例极值保存在 `official_saved_example_extrema.json`；它们是 map extrema，不能直接当作已验证的二维 DoG `(x0,y0)`。例如 Fig2/Data 的 A2:D2 为 frame 67、x 129、y 128、Max 157919；A3:D3 为 frame 70、x 129、y 127、Max −90663.4。没有根据 ON/OFF、幅值或与 LN 的接近程度从中选择中心。

**No per-cell experiment-time localization coordinate found in accessible official artifacts.** 该句的范围是本次可读资料；统计补充文件以及作者未公开的完整数据仍是明确证据缺口。README 也说明约 200 records 的更完整数据需由作者提供；本轮未联系作者。

## 3. 这些证据属于哪一层？为什么没有重新计算？

| 证据层级 | 本轮结果 | 独立性边界 |
|---|---|---|
| Experiment-time localization | 有实验方法描述，无目标 cell 的保存坐标 | 不能从方法描述恢复逐 cell 测量 |
| Author reverse-correlation / DoG estimate | 有非目标 cell 的保存示例；目标中心 0/22 | 如可映射，属于 **same biological data, independent RF-estimation method/model family**，不是 independent biological dataset |
| Our same-data reproduction | 未运行 | 若未来满足本任务的严格门槛，仍只能使用上一行的独立性表述 |
| 作者 notebook xoff/yoff | 另存于 `excluded_notebook_model_centers.json`，不纳入合格中心 | 原论文 P58（block 125）说明 RF location、radii 与 temporal parameters 都经过 cell/model response correlation 优化；这些是 prediction-model 参数，不是保存的 RC/DoG 测量 |

原论文 P60（block 127）报告 biophysical model 与 RC RF estimates 大致相差 1–2 pixels。这是作者的总体一致性陈述，没有提供目标 cell 的配对 x/y、逐 cell 误差或可验证的参数传递记录；不能把 notebook 参数重标记为独立 RC centers。

原文与公开代码支持的方法信息如下：

| 步骤 | 已定义 | 仍缺少的可执行约束 |
|---|---|---|
| 原始输入与时间 | 本地有当前 37 recording spike 文件和 1x10 movie；README 给出 timing、gamma/cone calibration。论文以 150 Hz bin spikes，累计 spike 前后各 64 frames | 独立 6x1 video payload 未见于 snapshot；acquisition 750/751 仍未解决 |
| Reverse correlation | 原论文 Methods blocks 30–33：STA 经 Fourier-domain 去相关；Eq.1；λ 对应 15 cpd、零 temporal frequency 的 stimulus power；同一 regularizer 用于全部 cells | 完整视频频谱与有限 STA lag window 的具体构造/归一化，以及 MPEG 8×8 block 谱峰的确切截断方法未公开为可执行流程 |
| Frame、极性、平滑 | 论文说拟合 peak response frame；workbook 同时保留正负极值、RevMap 和 SmoothRevMap；部分示例说明 26.6 ms | 没有覆盖全部 ON/OFF、biphasic maps 的固定峰值选择及 smoothing 规则；示例参数不等于完整算法 |
| DoG | 同心、圆对称二维 DoG；最小二乘；中心为共享的 `(x0,y0)` | RF-estimation 的 optimizer、初始值/边界、fit region 未找到；前向模型 `get_RF` 不替代该拟合程序 |
| 坐标 | 刺激标称 256 pixels 对应 4.6 degree；map 有 MxX/MxY 字段 | 保存 map 的 index origin、transpose/flip、y 方向尚无可验证的完整 transform |
| Recording / cell | README 说 10 min 更适合 RC，6×1 min 更适合模型；paper P38 说重复测量的 RF **radii** 相差 <5% | 没有把 recording-level center 合成为 cell-level center 的作者规则；radii 重复性不是中心合并规则 |

因此本任务条件 2、4、5 尚未同时满足。`reproduction_gate.json` 记录 **NO_GO**；没有人为补充算法、没有选择 recording 平均规则，也没有运行重新计算。`PROTOCOL.md` 的前提未成立，故不创建一份声称已经冻结可复现算法的协议。

**Frame 750/751：`UNVERIFIED_NOT_RECOMPUTED`。** 仅在边界、lag coverage 和 peak selection 均保持一致时，纯时间平移才可能仅移动 temporal peak 而不改变空间中心；本任务没有证据证明作者实际流程满足这些条件，不能断言空间中心不敏感。作者估计器尚不能按固定规则运行，因此两个 plausible alignments 的空间位移数值都无法合法获得。用自行设计的方法各跑一次会超出用户允许的 reproduction 条件。没有改变 `_LIVE_START_FRAME=751`，没有解决或绕过 frame-zero blocker。

## 4. 与 frozen LN centers 的方向和幅度吻合程度？

**UNVERIFIED：合格配对数为 0。** 没有可报告的 x/y sign、quadrant、cosine、angle、radial/x/y Pearson 或 Spearman、mean/median vector error。MC ON、MC OFF、PC ON、PC OFF、MC overall、PC overall 的配对数均为 0；未计算分组检验。

`ln_vs_official_centers.csv` 保留全部 22 cells / 37 recordings 和原 frozen LN 数值，official 与 comparison 字段留空，并明确标记原因。空值不等于零误差或不一致。

本轮重新核对了 LN 的坐标定义：raw x 向图像右、raw y 向图像下；转换到项目 degree 坐标为 `x = x_LN × 3 × 4.6/256`、`y = −y_LN × 3 × 4.6/256`，pitch 为 0.05390625 degree/pooled pixel。22 个 checkpoint center 与冻结表相同。这里只验证 LN 到项目坐标，**未验证 author RC map 到 LN 的方向映射**，也未根据 LN agreement 选择符号。作者前向模型采用 `linspace(-128,128,256)`，不能当作 RC map index convention 的证明。

## 5. Prediction improvement 是否也随作者 RF offset 增大？

**UNVERIFIED：没有 official offset 可用于该关系。** LN-versus-official center error 与 improvement 的关系同样无法计算。

作为冻结实验结果的核对，直接从旧 paired table 重新计算 `I = NLL_zero − NLL_aligned`：22 cells 中 18 个改善，mean I = **0.01491661099**；I 与 LN radial offset 的 Pearson 为 **0.8622918658**，Spearman 为 **0.8464144551**，与冻结 summary 一致。这些全部属于已有 **development prediction evidence**，不是本轮新训练或独立中心验证。

`alignment_prediction_relation.csv` 保留全部 cells、冻结 NLL、I、LN offset；official offset 和中心误差留空。没有按结果选择 subset，未补充 significance tests。

## 6. LN center 应解释为什么？

选择 **B — FUNCTIONAL ALIGNMENT SUPPORTED, BIOLOGICAL CENTER UNRESOLVED**。

已有实验支持 LN center 作为当前 development prediction 的 **functional alignment nuisance parameter**。原论文确认实验存在 RF mis-centering，但本轮没有得到目标 cell 的独立中心来建立方向和幅度对应，因此 **biological RF-center proxy 的逐 cell 解释尚未验证**。既没有证据达到 A，也没有恢复到明显冲突的独立中心以支持 C。没有将其解释为 anatomical center，没有新增训练实验。

## 冻结与交付证据

- Branch：`rgc-readout-v2`；HEAD：`fea28de038821fadee279b93728688b34bcb3bac`。开始时的完整 dirty git status、22-cell/37-recording mapping、22 个 LN final checkpoint hashes、旧 alignment experiment hashes 保存在 `evidence_manifest.initial.json` / `evidence_manifest.json`。
- `verification.json` 记录最终 input hashes、HEAD、git status 与 CSV 完整性核验。新增内容均在本轮目录；原 data、production source、models、checkpoints 和旧审计文件不改动。
- `source_inventory.csv`：116 个官方 snapshot 文件及原论文、在线 repository、补充文件访问状态。`source_search_evidence.json`：字段命中与 notebook/docx/PDF 提取证据。`workbook_inventory.json`、`workbook_relationships.json`、`external_link_summary.json`：工作表、公式、标签及缓存检索。`paper_blocks.json` 可按本文 index / P 标签定位原文。
- 没有可用目标中心，因此不生成带数值的 `official_rf_centers.csv`；非目标作者示例单独保存，避免混入 22-cell 主表。未重新估计中心，因此没有 `PROTOCOL.md`。
- 获取过程见 `retrieval_log.json`。部分直接公开下载因本机证书链验证失败使用了仅限这些请求的 TLS fallback，日志明确记录 `tls_certificate_verified=false`；未修改持久网络设置。Fig2 另经官方 annex MD5 校验。所有 XLSX 都只读取、未保存回源文件；读取器关于不支持 extension 的提示未造成源文件改动。

缺失的关键证据是：目标 cell 的实验定位表或作者 RC/DoG 中心表及坐标约定，或完整、固定、可复现示例的作者 RF-estimation 流程；统计补充 workbook 的内容也仍未验证。本报告不把访问失败当成“不存在”的证据。
