# RetiPath primate multi-level retinal dataset map

核查日期：2026-09-17。这是数据可得性与研究可执行性审计，不是机制验证、实验结果或实施计划。

## 1. Layer × dataset 总表

“有记录”表示原论文确实记录该层级，不表示 raw 已公开、对应状态已验证或可以直接训练。`非光` 为化学/电压命令实验；`群体` 为 ERG 间接群体信号。角色是有条件的推荐用途，不能越过数据可得性和状态映射门槛。

| Dataset ID | Cone | HC | BC | AC | RGC | Availability | 推荐角色 |
|---|---|---|---|---|---|---|---|
| [RA26_HC](#ra26_hc) | — | 有记录 | — | — | — | AUTHOR_REQUEST_REQUIRED | TRAINING_CONSTRAINT |
| [RA26_BC](#ra26_bc) | — | — | 有记录 | — | — | AUTHOR_REQUEST_REQUIRED | FUNCTIONAL_ASSAY_ONLY |
| [RA26_RGC](#ra26_rgc) | — | — | — | — | 有记录 | AUTHOR_REQUEST_REQUIRED | TRAINING_CONSTRAINT |
| [SL21_10MIN](#sl21_10min) | — | — | — | — | 有记录 | RAW_VERIFIED_AVAILABLE | TRAINING_CONSTRAINT |
| [CH24_CONE](#ch24_cone) | 有记录 | — | — | — | — | PROCESSED_ONLY | AUXILIARY_TRANSFER |
| [CH24_INNER](#ch24_inner) | — | 有记录 | 模态未明 | — | 有记录 | PAPER_ONLY | FUNCTIONAL_ASSAY_ONLY |
| [BA19](#ba19) | 有记录 | — | — | — | 有记录 | UNVERIFIED | HELD_OUT_PHYSIOLOGY_VALIDATION |
| [SA24](#sa24) | 有记录 | — | — | — | — | AUTHOR_REQUEST_REQUIRED | HELD_OUT_PHYSIOLOGY_VALIDATION |
| [SH20](#sh20) | — | — | — | — | 有记录 | PROCESSED_ONLY | FUNCTIONAL_ASSAY_ONLY |
| [KA25](#ka25) | — | — | — | — | 有记录 | RAW_VERIFIED_AVAILABLE | AUXILIARY_TRANSFER |
| [SR26](#sr26) | — | — | — | — | 有记录 | PROCESSED_ONLY | AUXILIARY_TRANSFER |
| [KR23](#kr23) | — | — | — | — | 有记录 | UNVERIFIED | NOT_CURRENTLY_USABLE |
| [SH25](#sh25) | — | — | — | — | 有记录 | UNVERIFIED | NOT_CURRENTLY_USABLE |
| [KI22_AC](#ki22_ac) | — | — | — | 有记录 | — | AUTHOR_REQUEST_REQUIRED | FUNCTIONAL_ASSAY_ONLY |
| [HR21](#hr21) | — | — | — | — | 有记录 | PROCESSED_ONLY | AUXILIARY_TRANSFER |
| [AB22](#ab22) | 群体 | — | 群体 | — | — | PROCESSED_ONLY | FUNCTIONAL_ASSAY_ONLY |
| [KW21](#kw21) | — | — | 非光 | — | — | PAPER_ONLY | NOT_CURRENTLY_USABLE |
| [SN04](#sn04) | — | 非光 | — | — | — | PAPER_ONLY | NOT_CURRENTLY_USABLE |
| [LI21_HC](#li21_hc) | — | 有记录 | — | — | — | AUTHOR_REQUEST_REQUIRED | HELD_OUT_PHYSIOLOGY_VALIDATION |
| [LI21_RGC](#li21_rgc) | — | — | — | — | 有记录 | AUTHOR_REQUEST_REQUIRED | HELD_OUT_PHYSIOLOGY_VALIDATION |
| [GR18_HC](#gr18_hc) | — | 有记录 | — | — | — | PROCESSED_ONLY | FUNCTIONAL_ASSAY_ONLY |
| [GR18_RGC](#gr18_rgc) | — | — | — | — | 有记录 | PROCESSED_ONLY | FUNCTIONAL_ASSAY_ONLY |
| [GR18_AII](#gr18_aii) | — | — | — | 有记录 | — | PROCESSED_ONLY | FUNCTIONAL_ASSAY_ONLY |
| [DA00_BC](#da00_bc) | — | — | 有记录 | — | — | PAPER_ONLY | FUNCTIONAL_ASSAY_ONLY |
| [PD02_HC](#pd02_hc) | — | 有记录 | — | — | — | PAPER_ONLY | FUNCTIONAL_ASSAY_ONLY |

本表共 **25 个按观测/公开版本划分的条目**，并非 25 个独立实验。300 个无序 dataset pairs 全部写入兼容性 CSV。可得性计数：AUTHOR_REQUEST_REQUIRED=7；PAPER_ONLY=5；PROCESSED_ONLY=8；RAW_VERIFIED_AVAILABLE=2；UNVERIFIED=3。

## 2. 真正核实的 raw、作者数据与最现实组合

**公开数据现状：`PARTIALLY_FEASIBLE`。现在不宜把 multi-level supervision 升级为 RetiPath 主线。** 已核实的公开原始观测主要位于 RGC 输出层；本次没有核实到可以直接执行的“HC raw A + BC raw A + RGC raw A，并以独立 HC/BC raw B 验证”组合。这里是检索与合同核验的结果，不是断言相关数据全球不存在。

### 2.1 哪些 raw 确实可获得

1. **SL21_10MIN：Schottdorf–Lee 冻结 DOI 归档中的 15-cell、10-min 子集。** 实际读取 ZIP 中央目录，确认逐事件响应文件、正式 `1x10_256.mpg` 以及标定文件存在。整包 28 个 raw TXT 中有 15 个 ten-minute、13 个 six-repeat 文件；不是论文全体 47 cells。README 描述的 `6x1_256.mpg` 不在已核查 ZIP，因此这部分未升级为完整 raw stimulus–response pair。[冻结归档](https://doi.gin.g-node.org/10.12751/g-node.xage77/)
2. **KA25：Karamanlis 的 marmoset 原始 session 镜像。** 六个 session 的 ZIP 内目录及 Manual 实查通过，含 event timestamps/unit IDs、实际图像与 fixation、frame onset/offset 所在文件。限定为 spike-sorted、未平均的事件级 raw；没有声称公开原始 MEA 电压。[session manifest](https://huggingface.co/api/datasets/open-retina/open-retina/tree/main/gollisch_lab/karamanlis_2024/sessions)、[Manual](https://huggingface.co/datasets/open-retina/open-retina/resolve/main/gollisch_lab/karamanlis_2024/Manual.pdf)

`RAW_VERIFIED_AVAILABLE` 在本报告统一采用**原始观测级**定义：连续生理 trace 或逐事件 spikes，加实际刺激或可重建的刺激文件均经目录/下载接口核实。Spike sorting 本身不自动降级；重复平均、平滑、归一化、预过滤输入或未核实真实刺激配对必须另外标注。它不意味着响应数值、时间对齐、许可证所有边界和所有细胞均已验证。

**不能混入 raw 清单：** CH24_CONE 为重复平均/暗电流归一化；SH20 的刺激已被 STA 时间滤波；SR26 实查的是 processed mirror，原 GIN raw 未过文件合同；HR21 有逐事件文件，但实际呈现刺激及时间配对未闭合；BA19 的公开 ZIP 存在，内部刺激/响应结构未核实。源图、XLSX、代码、元数据与 raw traces 分开记录。[Chen deposit](https://datadryad.org/dataset/doi:10.5061/dryad.q2bvq83vg)、[Shah deposit](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxkvk)、[Sridhar mirror](https://open-retina.org/package_docs/datasets/sridhar_2025/)、[HumRet reader](https://github.com/katjaReinhard/HumRet/blob/656a36b8dd4c9c879561d71401413b8fafdf46d3/_scripts/EXAMPLE_ReadData.m)

### 2.2 哪些需要联系作者

| 来源 | 需要的资料 | 作者承诺状态 |
|---|---|---|
| Raval2026 HC/BC/parasol | raw trials、实际刺激/同步、calibration、species/animal/retina/cell IDs；HC subtype；BC subtype/holding voltage | 没有核实到 study-specific 数据声明或提供承诺；按本任务规则记 `AUTHOR_REQUEST_REQUIRED`，表示下一步需要申请 |
| Liu2021 HC/RGC currents | 独立 HC Vm 与配套 glider 刺激；RGC EPSC/IPSC；细胞 lineage | 最终版明确 reasonable request；HC 仅 n=2，验证能力有限 |
| Saha2024 full raw | 新采 peripheral control L/M voltage 和 trial metadata | 明确其他数据向作者申请；公开 Source Data 不是完整 raw |
| Kim2022 AC | SAC/PAC voltage/spikes、刺激与细胞清单 | 明确 reasonable request；公开代码是模拟/形态资料 |
| Chen2024 Figure10 | HC voltage、BC measurement 类型、parasol原始spikes及刺激 | raw 未进已查 Dryad；提供承诺未核实，主表保留 `PAPER_ONLY` |
| 经典 H1/BC 论文 | 原始 trace 与完整实验日志是否仍可取得 | 未找到开放承诺；不能把“值得申请”写成“作者已经承诺提供” |

本轮没有联系任何作者。[Raval原始JATS](https://www.biorxiv.org/content/early/2026/03/23/2026.03.19.713068.source.xml)、[Liu数据声明](https://www.nature.com/articles/s41593-021-00899-1)、[Saha数据声明](https://www.nature.com/articles/s41467-024-53061-3)、[Kim数据声明](https://www.nature.com/articles/s41467-022-30405-5)、[Chen全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11537484/)

### 2.3 最现实的 HC/BC/RGC 组合

**优先申请同源 Raval HC voltage + parasol spikes，BC excitatory input current 保留作功能验证。** 这是最紧凑的跨层来源，但目前没有公开 raw 文件合同，而且不构成已验证的同动物/同片/同时记录。独立 HC 验证优先核查 LI21_HC；必须取得 ID，不能用“另一篇论文”代替样本独立性。

BC 是这个三层方案中的关键限制：Raval 的 BC 全部为电压钳兴奋性**输入**电流；当前 RetiPath 的 BC components 并非该输入电流或已鉴定的 BC 膜电位/释放状态。不能把这一层计为已就绪的 direct BC supervision。HC 名称也未自动证明 H1 subtype。[Raval正文与Fig1/4](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13041967/fullTextXML)

## 3. 可得性分级、范围与文件证据

| 状态 | 本报告的使用标准 |
|---|---|
| RAW_VERIFIED_AVAILABLE | 实查目录/接口并确认 response 与实际或可重建 stimulus；限定具体版本和子集 |
| PROCESSED_ONLY | 已验证公开 tier 为平均/归一化/分箱/图源等派生产品，或完整 raw 配对未闭合；不声称别处不存在 raw |
| METADATA_ONLY | 只有描述/注册元数据，无可核实数据文件；本轮未把单独索引当主条目 |
| PAPER_ONLY | 只有论文及图可确认，未发现匹配 raw 存档或明确的提供承诺 |
| AUTHOR_REQUEST_REQUIRED | 原文明确需索取，或本任务特别要求的 Raval raw 未公开；两种证据来源在字段中分开 |
| ACCESS_RESTRICTED | 已确认数据确实受审批/登录/访问条件限制；不能仅因工具403/timeout便使用 |
| UNVERIFIED | 存档、版本、刺激配对或字段仍有核查缺口；是证据不足，不是不存在或零 |

核查只读取论文、代码文本、README、manifest、目录、小型metadata；大 ZIP 仅 Range 中央目录和白名单说明。未下载大型数据主体，未加载响应数组、movie pixels、checkpoint，未执行模型或新 physiology assay。

| 可复核对象 | 实际检查与边界 |
|---|---|
| Schottdorf ZIP | 492,820,686 bytes；EOCD `492820664–492820685`；目录 `492812568–492820663` 共8096B；99 members；README约11.7kB。没有读取TXT响应内容 |
| Karamanlis sessions | 六个 marmoset ZIP 的 EOCD22B和精确目录约600B/包；Manual188091B；三movie session包含 fixationmovie_data 等容器；数值payload未读 |
| Chen Dryad | version325663：MAT41490462B+README1357B；未打开MAT；实际stimulus vector字段仍UNVERIFIED |
| Shah Dryad | version48728：Figure2/3/4/5/6五文件；没有Figure7 naturalistic数据文件 |
| Sridhar镜像 | NM responses.zip91470833B，目录544B列4个response PKL；fixation与stimulus archives列出，未下载 |
| Baudin Dryad | version23398：ZIP188452270B；内目录请求失败，不能从标题/usage notes提升raw状态 |
| Abbas GIN | ZIP569011362B/67entries；8 b-wave+16rod/cone CSV；只读说明和两份header，不读数值响应 |

精确接口、文件名及访问状态保存在各条目 `file_listing_evidence`、`download_interface_evidence`、`source_urls`；公开文件存在与内容/完整性/时序验证是不同证据层级。

## 4. RetiPath 状态映射与最小 observation head

以下结论来自**本轮读取当前源码**。没有导入模型。`DIRECT_MAPPING` 表示观测量可直接对应输出层的候选合同，不表示已完成适配或已证明生物机制。对内部 effective state，区分“直接细胞记录”与“已验证状态对应”；未满足后者时主表不标 direct。

| 当前量 | 源码事实 | 本次 mapping 边界 |
|---|---|---|
| cone-related input | RGB→gamma→L+M→空间pool→Weber；无R*/s绝对量、无独立cone photocurrent/voltage state | cone电流/电压主要是 `INDIRECT_FUNCTIONAL_CONSTRAINT`；S-cone/SBC不能由pooledL+M直映 |
| h1_state | graph drive→causal delay→low-pass；feedback另经transpose graph与amplitude | 实测HC Vm→`a*h1+b`仅是待验证候选；HC subtype/site/状态含义必须匹配。HC Vm不等于反馈电流 |
| BC components | 时空feature的有效混合/branch状态 | BC输入EPSC、胞体Vm、terminal release是不同观测；名字为BC不足以支持direct |
| AC states | BC输入的delay/low-pass以及负gate后的贡献 | 不代表AII、SAC、PAC或真实AC Vm；RGC IPSC也不是direct AC recording |
| membrane/readout | 归一化conductance/voltage及pool；非mV/nS | 不可把归一化状态当实测电压/电导；需已知驱动力、基线和状态对应 |
| RGC probability | sigmoid(logit)，每dt-bin条件Bernoulli occupancy | 原始事件正确分箱后可direct；概率不是Hz，PSTH/平均rate不是Bernoulli target |

源码：[输入变换](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:164)、[H1](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/h1_pathway.py:71)、[BC](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/bipolar_subunits.py:162)、[AC](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/amacrine_pathways.py:62)、[RGC readout](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/rgc_state.py:59)、[现有observation语义](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/mechanism_observation.py:33)。

**最小允许形式：** 对已确认同一生理量的连续观测，`y_hat = a*h+b`；确有仪器响应且传递函数已知时才使用 `known_filter(a*h+b)`。每种current/voltage独立指定单位与符号。`a`是measurement scale，`b`是baseline；它们的取值不能每个刺激/时间点自由变化。仅知道“3kHz filter”仍不足以任意补一个生物时间滤波器。每条目已记录scale、offset、temporal filter和noise要求。

连续trace的Gaussian/correlated residual只是可检验候选；有重复才可估计测量噪声，平均trace不能恢复单trial方差。Spike使用已有适当概率readout；现有实现是Bernoulli occupancy，不能因新数据按count存储便无声更换模型/损失。不同采样率的刺激对齐也不是本轮实施内容。

仿射head不能消除状态错配：当前H1在线性输入坐标下的graph/delay/low-pass，不会因为增加scale/offset就拥有未建模的局部非线性。作者数据取得后可能暴露表示能力不匹配，不能预设“拿到raw就一定能用原样模型拟合”。同样，未锚定的measurement scale会保留latent amplitude的尺度自由度；多层记录可能约束部分动力学/形状，不等于自动识别全部参数。本轮不运行验证、更不修改architecture或添加decoder。

## 5. Cross-layer 证据强度

| 来源 | 同论文/实验室 | 同物理preparation | 同stimulus | 可执行意义 |
|---|---|---|---|---|
| Raval2026 HC+BC+RGC | 是 | 未核实；same preparation method不等于same retina | Fig4匹配条件；Fig8RGC有中心mask差异 | 最相关申请组合；raw/HC subtype/BC observable仍缺 |
| Chen2024 cone+HC+BC+RGC | 是 | 未核实 | Fig10示例cone/HC/parasol同step+flash | Figure10不在Dryad；BC modality未明 |
| Liu2021 HC+RGC currents/spikes | 是 | 未核实；HC去RPE而RGC保留 | 同glider家族，电流分析有polarity选择 | HC held-out候选仅n=2，H1 subtype未核实；BC/AC只经RGC current间接观察 |
| Grimes2018 H1+ON parasol，Fig5限定 | 是 | **明确同retinal mount/piece** | **同背景/contrast/stimulus** | 最强同片匹配证据，但公开是summary且rod regime，不是raw训练组合 |
| Kim2022 SAC/PAC+RGC | 是 | 配对IDs未核实 | 相关motion assays | 真正AC记录存在，但特定AC类型不等于现有aggregateAC |
| Abbas2022 | 是 | donor/punch层级可有关联 | ERG flash条件 | 属群体信号，不计为single-cell cone+BC联合记录 |

依据：[Raval](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13041967/fullTextXML)、[Chen](https://pmc.ncbi.nlm.nih.gov/articles/PMC11537484/)、[Liu](https://pmc.ncbi.nlm.nih.gov/articles/PMC8728393/)、[Grimes Fig5](https://elifesciences.org/articles/38281)、[Kim](https://www.nature.com/articles/s41467-022-30405-5)、[Abbas](https://pmc.ncbi.nlm.nih.gov/articles/PMC10000337/)。

## 6. Compatibility 与参数共享资格

兼容性是针对“以最小观测关系组合这些资料”的**定性审计判断**，不是拟合结果或生理等同性测试。主CSV覆盖所有无序pairs，八维各给等级和理由，另列preparation、recording modality、raw门槛和独立性；对称矩阵由这些pairs恢复。`UNKNOWN` 不折算为中等；`HIGH` 也不跳过raw/状态资格。

- **A — STRUCTURE_SHARED_PLAUSIBLE：** 可以讨论匹配生理环节的equation/topology/sign/operator。对化学/电压命令实验和当前视觉输入模型，资格仍UNVERIFIED；一般拓扑相似不说明当前模型已包含全部可观测量。
- **B — PARAMETER_PARTIAL_POOLING_PLAUSIBLE：** 只是匹配species、eccentricity、subtype、RPE/温度、adaptation、stimulus、modality后的条件性shared prior候选。跨层不能把不同含义的参数捆在一起；资料尚缺时标UNVERIFIED。本表仅KA25–SR26列为有具体依据的条件性B候选：同marmoset、typedRGC、ex-vivo、85Hz及相近低photopic背景，允许考虑按subtype区分的RGC时程/readout prior；其eccentricity、实际背景与cohort仍须确认。其余pairs不因overall为MEDIUM/HIGH而自动取得B资格。
- **C — NUMERIC_PARAMETER_SHARING_NOT_JUSTIFIED：** 本次所有跨数据集pair都未建立直接数值共享资格，包括同片Grimes；“都是primate”不是理由。

| 关键pair | overall | 原因/用途限制 |
|---|---|---|
| Raval HC–BC / HC–RGC / BC–RGC | MEDIUM | 同源匹配assay有优势；物理配对、subtype、BC输入电流语义与raw未闭合 |
| Raval HC–Liu HC | MEDIUM | 候选HC验证，但H1 subtype、RPE、背景、刺激及independent IDs待核；n=2 |
| Grimes HC–RGC | HIGH | 仅Fig5实验条件匹配；不等于raw、cone工作点或数值参数共享 |
| Schottdorf–Raval | LOW | in vivo vs ex vivo、约3.5–8.3° vs20–50°、movie vsassay、绝对背景未配齐 |
| Karamanlis–Sridhar | MEDIUM | 同marmoset RGC/85Hz有利；几何、均值/contrast、cohort版本与位置需分开 |
| Chen cone–Baudin peripheral L/M | MEDIUM | 限定L/M子集；current/voltage、normalization及真实输入/lineage仍需核对 |
| Baudin–Saha | MEDIUM | 新采L/M可比较；Saha复用Baudin的S-cone部分作为independent holdout则INCOMPATIBLE |
| HumRet–Abbas | LOW | 同人类但RGC事件与群体ERG、25°Cvs37°C、光适应和细胞身份不同 |
| 视觉实验–cultured HC/command BC | INCOMPATIBLE | 化学/电压命令不能直接代替光输入合同 |

OpenRetina不新增独立生物样本。原数据与镜像技术对应性可能高，但互作train/held-out属于数据复用。当前官方目录中Karamanlis含mouse+marmoset、Sridhar为marmoset；Höfling/Goldin为mouse，Maheswaranathan为salamander，不计入primate主表。Karamanlis原marmoset85Hz不能套用mouse75Hz或统一config；Sridhar2026最终论文与旧preprint/新旧DOI及四piece镜像需明确版本。[OpenRetina目录](https://open-retina.org/package_docs/datasets/)、[Sridhar正式论文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014157)

## 7. Training 与 validation 分工

表中角色是**候选用途**，`currently_ready` 与 `role_activation_conditions` 决定是否真正可执行。本轮没有把任何新资料指定成已consumed的训练/测试集。

| 资料 | 推荐用途 | 独立性与边界 |
|---|---|---|
| RA26_HC、RA26_RGC | 获数据且映射合格后的TRAINING_CONSTRAINT候选 | 先按动物/preparation/cell与trial层级明确分工；不能事后按拟合效果换样本 |
| RA26_BC | 预留FUNCTIONAL_ASSAY_ONLY | 可作为未用于训练的BC input-current assay；不是独立dataset B，也不是release监督 |
| LI21_HC/LI21_RGC | HELD_OUT_PHYSIOLOGY_VALIDATION候选 | 作者提供raw和独立ID后才成立；HC n=2且subtype未核实，不能支持强群体结论 |
| CH24_CONE | AUXILIARY_TRANSFER | 已平均/归一化；无matching cone state，不直接训练当前输入 |
| BA19、SA24新采peripheral L/M | 保留早期层HELD_OUT候选 | 当前资格未过；剔除复用S-cone，不能与训练源重叠 |
| KA25、SR26、HR21 | AUXILIARY_TRANSFER或未来预留RGC预测验证 | 不是HC/BC physiology B；mirror不是独立cohort；精确分工尚未实施 |
| GR18、经典H1/BC、Kim | FUNCTIONAL_ASSAY_ONLY | 公开摘要或不同细胞/工作条件，不能替代rawtrace |

Held-out资料不能同时用来挑选参数、观测filter或measurement scale。若held-out实验自身需要校准，必须由独立校准信息或预先划出的校准部分确定，并明确其不再属于纯held-out评估；不能用验证响应拟合一个head后仍称“未见过的生理验证”。本报告不指定loss weights、优化步骤或新训练协议。

## 8. 最多三个 candidate MVP

### MVP-A — HC + RGC，同源申请版（最现实）

- **Dataset：** RA26_HC voltage subset + RA26_RGC；RA26_BC留作功能资料；LI21_HC作为独立性待核的B。
- **监督层：** HC voltage候选观测 + parasol spike output；BC只约束输入处理功能。
- **Training / held-out：** Raval HC/RGC候选训练；Raval保留BC assay和未用RGC trials；Liu HC raw在确认独立后作为B，不参与head/model选择。
- **最大blocker：** Raw package未获得；H1 subtype/affine状态对应、实际刺激与absolute operating point未闭合；独立HC B仅n=2。
- **联系作者：** 需要。属于作者数据依赖候选，尚不能宣布FEASIBLE_NOW；取得数据仍须通过状态资格。

### MVP-B — HC + BC + RGC，三层条件版

- **Dataset：** Raval三层；Chen Figure10、Dacey typedBC可作为额外BC/HC validation资料候选。
- **监督层：** HC与RGC同A；BC只有在被测input current存在对应状态时才有direct资格。
- **Training / held-out：** 不能把Raval、Chen、Dacey全部用于训练；至少留一套独立HC/BC资料，独立样本ID和模态须核实。
- **最大blocker：** Raval BC是输入EPSC、亚型/holding缺失；Chen BC模态未明；当前BC effective components不等于上述观测。尚无可执行的direct BC层。
- **联系作者：** 需要；目前不优于A，不设计新state/architecture来补合同。

### MVP-C — 公开数据限定的可行性/功能论文版本

- **Dataset：** SL21_10MIN提供RGC资料；KA25留作跨物种RGC transfer；Chen processed cones、Grimes source summaries作为预留功能参考。
- **监督层：** 只有RGC可直接监督；HC/BC只有公开功能或summary，不能称raw multilevel training。
- **Training / held-out：** RGC训练与RGC transfer分开；processed cone/HC资料不用于参数选择，只能提供受限的外部功能比较。
- **最大blocker：** 不能直接检验“multi-level physiological supervision reduces pathway ambiguity”；最多支撑数据/方法可行性与功能一致性的有限论题。
- **联系作者：** 使用已列公开tier本身不必，但完整原假设仍需作者数据及映射确认。

**选择MVP-A，作为优先数据协商候选，不作为已授权实施。** HC+BC+RGC raw joint training未就绪；没有raw HC trace时，只能明确写functional evidence，不能从图像数字化出“训练集”。

## 9. 原 hypothesis 当前是否可执行

> RGC-only supervision leaves internal retinal mechanisms underconstrained; multi-level physiological supervision reduces pathway ambiguity while preserving RGC prediction.

**本次唯一总体等级：`PARTIALLY_FEASIBLE`。** RGC输出数据和多个层级的真实记录来源存在，研究问题有可检验的候选资料基础；但已核实的公开资料还不足以执行完整比较。MVP-A存在 **FEASIBLE_WITH_AUTHOR_DATA 的条件性路径**，而非已满足该状态：raw、测量对应、校准和独立验证仍是实质门槛。

这句话中的“reduces”和“preserving”都是未来需检验的结果，不是本次审计结论。多数据集拟合成功也不等于真实生物机制已识别；最小head保留的尺度自由度、不同准备/工作点，以及可能的表示错配都会限制可声称的内部ambiguity范围。当前可合理推进的是取得和核实资料；是否正式改变RetiPath主线由用户决定。


## 逐数据集契约摘要

完整逐字段契约以 CSV/JSON 为准：所有要求字段均有值，无法核实的字段明确为 `UNVERIFIED`，不以空白代表零。以下为便于阅读的摘要；每条最后给出本次核查的原始来源。

### RA26_HC

**Raval 2026 horizontal-cell voltage/current** — Raval, Oaks-Leaf, Chen & Rieke. Origin and functional impact of early nonlinearities in primate retina (preprint v1, 2026-03-23)；DOI `10.64898/2026.03.19.713068`。

- **可得性**：`AUTHOR_REQUEST_REQUIRED`。User-required status when raw is not public; neither an on-request promise nor author consent was found。License：Article CC BY 4.0; data/code reuse license UNVERIFIED。
- **生物/样本**：Macaca nemestrina / mulatta / fascicularis; per-cell species UNVERIFIED；ex vivo; primarily flat mount with RPE; some HC/BC slices; per-cell preparation unknown；20–50 degrees peripheral; individual position/sector UNVERIFIED；type：Horizontal cell; H1 identity UNVERIFIED；cells：Fig2 n=16; Fig4 n=17; Fig8 n=3; unique total unknown (do not sum)；animals：UNVERIFIED。
- **记录与刺激**：intracellular voltage (current clamp); subset excitatory synaptic current (voltage clamp)；Gaussian noise, mean/contrast steps, spots, contrast-reversing gratings, flashed images, eye-movement-driven natural images。Response sampling：10000 Hz acquisition; 3000 Hz analog filtering；stimulus timing：UNVERIFIED: 10 kHz is acquisition, not visual refresh; mean/contrast epochs 500–1000 ms。
- **刺激合同**：UNVERIFIED pixel geometry; LED spot diameter 500 um; grating bar width reported in um；units：um; retinal eccentricity in degrees；contrast：Noise 50%, grating 0.9; RMS/Michelson/Weber formula UNVERIFIED；adaptation：Condition-specific 1500/5000/15000 R*/cone/s; grating/noise example 15000；channels：405 nm uniform LED; LightCrafter4500/eMagin spatial display; session L/M/S drives unknown。
- **校准/元数据**：Paper gives R*/cone/s; no session-linked calibration manifest obtained。Stimulus files：NO verified study waveform, seed, movie trajectory or display transform。Trials：Multiple assays; trial IDs/alignment unknown；repeats：Overall unknown; not transferable from RGC six-repeat example；cell map：No public animal-retina-cell-epoch manifest verified。Preprocessing：3 kHz recording filter; figures contain averages/LN/F2/RF summaries; raw processing unknown。
- **跨层/独立性**：Same preparation METHOD; same physical retina/animal UNVERIFIED；Fig4 matched grating/spot conditions; Fig8 RGC has added center mask。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`INDIRECT_FUNCTIONAL_CONSTRAINT` → h1_state candidate; h1_feedback is a separate quantity。Current H1 is an effective low-pass graph state, not validated HC voltage; HC is not proven H1。候选 head：`For voltage subset only: V_hat = known_acquisition_filter(a * h1_state_at_registered_site + b)`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=Continuous voltage residuals; Gaussian/correlated covariance only if supported by repeats; current needs separate head。
- **角色**：`TRAINING_CONSTRAINT`；当前：NO; paper functional evidence only。激活条件：Author raw voltage + displayed stimulus + geometry + cell IDs + H1 identity and affine correspondence; otherwise functional assay only。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：bioRxiv JATS + Europe PMC XML list only supplemental PDF, no dataset; targeted repositories not found。接口：https://api.biorxiv.org/details/biorxiv/10.64898/2026.03.19.713068 (v1, published NA)。
[来源1](https://www.biorxiv.org/content/early/2026/03/23/2026.03.19.713068.source.xml) · [来源2](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13041967/fullTextXML) · [来源3](https://github.com/Rieke-Lab/calibration-resources)

### RA26_BC

**Raval 2026 bipolar excitatory input currents** — Raval, Oaks-Leaf, Chen & Rieke. Origin and functional impact of early nonlinearities in primate retina (preprint v1, 2026-03-23)；DOI `10.64898/2026.03.19.713068`。

- **可得性**：`AUTHOR_REQUEST_REQUIRED`。User-required status when raw is not public; neither an on-request promise nor author consent was found。License：Article CC BY 4.0; data/code reuse license UNVERIFIED。
- **生物/样本**：Macaca nemestrina / mulatta / fascicularis; per-cell species UNVERIFIED；ex vivo; primarily flat mount with RPE; some HC/BC slices; per-cell preparation unknown；20–50 degrees peripheral; individual position/sector UNVERIFIED；type：Cone bipolar cells; ON/OFF and subtype unresolved；cells：Fig2 n=13; Fig4 n=14; unique total unknown；animals：UNVERIFIED。
- **记录与刺激**：synaptic current: ALL BC records are voltage-clamped excitatory INPUT currents; holding potential unknown；Gaussian noise, contrast-reversing gratings, matched spots; natural-movie BC recordings not verified。Response sampling：10000 Hz acquisition; 3000 Hz analog filtering；stimulus timing：UNVERIFIED: 10 kHz is acquisition, not visual refresh; mean/contrast epochs 500–1000 ms。
- **刺激合同**：UNVERIFIED pixel geometry; LED spot diameter 500 um; grating bar width reported in um；units：um; retinal eccentricity in degrees；contrast：Noise 50%, grating 0.9; RMS/Michelson/Weber formula UNVERIFIED；adaptation：Condition-specific 1500/5000/15000 R*/cone/s; grating/noise example 15000；channels：405 nm uniform LED; LightCrafter4500/eMagin spatial display; session L/M/S drives unknown。
- **校准/元数据**：Paper gives R*/cone/s; no session-linked calibration manifest obtained。Stimulus files：NO verified study waveform, seed, movie trajectory or display transform。Trials：Assay-specific epochs; public epoch IDs absent；repeats：UNVERIFIED；cell map：No public animal-retina-cell-epoch manifest verified。Preprocessing：3 kHz recording filter; figures contain averages/LN/F2/RF summaries; raw processing unknown。
- **跨层/独立性**：Same preparation METHOD; same physical retina/animal UNVERIFIED；Fig4 matched grating/spot conditions; Fig8 RGC has added center mask。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`INDIRECT_FUNCTIONAL_CONSTRAINT` → BC input processing; not bc_direct_effective_drive as a validated release state。Cone-to-BC EPSC is not BC voltage or terminal release; NO_CLEAN_MAPPING to current BC release candidate。候选 head：`I_hat = known_filter(a*h_input_current+b) ONLY if an existing state is proven to represent measured input current; no eligible current state established`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`FUNCTIONAL_ASSAY_ONLY`；当前：NO; paper functional evidence only。激活条件：Author raw currents + stimulus + subtype + clamp voltage; reserve BC assay for held-out functional comparison, not direct BC release training。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：bioRxiv JATS + Europe PMC XML list only supplemental PDF, no dataset; targeted repositories not found。接口：https://api.biorxiv.org/details/biorxiv/10.64898/2026.03.19.713068 (v1, published NA)。
[来源1](https://www.biorxiv.org/content/early/2026/03/23/2026.03.19.713068.source.xml) · [来源2](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13041967/fullTextXML) · [来源3](https://github.com/Rieke-Lab/calibration-resources)

### RA26_RGC

**Raval 2026 parasol spike physiology** — Raval, Oaks-Leaf, Chen & Rieke. Origin and functional impact of early nonlinearities in primate retina (preprint v1, 2026-03-23)；DOI `10.64898/2026.03.19.713068`。

- **可得性**：`AUTHOR_REQUEST_REQUIRED`。User-required status when raw is not public; neither an on-request promise nor author consent was found。License：Article CC BY 4.0; data/code reuse license UNVERIFIED。
- **生物/样本**：Macaca nemestrina / mulatta / fascicularis; per-cell species UNVERIFIED；ex vivo; primarily flat mount with RPE; some HC/BC slices; per-cell preparation unknown；20–50 degrees peripheral; individual position/sector UNVERIFIED；type：Parasol; Fig8 ON parasol; other assay ON/OFF mixture unknown；cells：Fig4 n=16; Fig8 n=4; unique total unknown；animals：UNVERIFIED。
- **记录与刺激**：extracellular spike, glass electrode；Contrast-reversing gratings, matched spots, contextual RF probes, center-masked surround gratings; RGC natural movies not verified。Response sampling：10000 Hz acquisition; 3000 Hz analog filtering；stimulus timing：UNVERIFIED: 10 kHz is acquisition, not visual refresh; mean/contrast epochs 500–1000 ms。
- **刺激合同**：UNVERIFIED pixel geometry; LED spot diameter 500 um; grating bar width reported in um；units：um; retinal eccentricity in degrees；contrast：Noise 50%, grating 0.9; RMS/Michelson/Weber formula UNVERIFIED；adaptation：Condition-specific 1500/5000/15000 R*/cone/s; grating/noise example 15000；channels：405 nm uniform LED; LightCrafter4500/eMagin spatial display; session L/M/S drives unknown。
- **校准/元数据**：Paper gives R*/cone/s; no session-linked calibration manifest obtained。Stimulus files：NO verified study waveform, seed, movie trajectory or display transform。Trials：Per-stimulus trials; Fig8 raster repeats；repeats：Fig8 example: 6 trials/stimulus; overall unknown；cell map：No public animal-retina-cell-epoch manifest verified。Preprocessing：3 kHz recording filter; figures contain averages/LN/F2/RF summaries; raw processing unknown。
- **跨层/独立性**：Same preparation METHOD; same physical retina/animal UNVERIFIED；Fig4 matched grating/spot conditions; Fig8 RGC has added center mask。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`DIRECT_MAPPING` → RGC probability/logit。Direct observed spike events map to existing probabilistic output when binning and parasol type match; not membrane supervision。候选 head：`Existing conditional Bernoulli readout on event occupancy in the declared dt; keep trial timing/history semantics`。
- **Nuisance/noise**：scale=No affine scale on spikes; retain existing readout calibration semantics；offset=Existing readout bias only; no new adapter；filter=No smoothing of target spikes; only verified timestamps/binning；noise=Existing Bernoulli occupancy likelihood; counts/PSTH are not Bernoulli observations。
- **角色**：`TRAINING_CONSTRAINT`；当前：NO; paper functional evidence only。激活条件：Author event timestamps + actual stimulus + trial and cell IDs; hold out cells/preparations and RGC trials before any future fit。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：bioRxiv JATS + Europe PMC XML list only supplemental PDF, no dataset; targeted repositories not found。接口：https://api.biorxiv.org/details/biorxiv/10.64898/2026.03.19.713068 (v1, published NA)。
[来源1](https://www.biorxiv.org/content/early/2026/03/23/2026.03.19.713068.source.xml) · [来源2](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13041967/fullTextXML) · [来源3](https://github.com/Rieke-Lab/calibration-resources)

### SL21_10MIN

**Schottdorf–Lee public 10-minute macaque movie subset** — Schottdorf & Lee. A quantitative description of macaque ganglion cell responses to natural scenes: the interplay of time and space；DOI `10.1113/JP281200`。

- **可得性**：`RAW_VERIFIED_AVAILABLE`。Frozen DOI ZIP directory verifies 15 ten-minute response files plus 1x10_256.mpg; status applies to this subset only。License：CC BY 4.0 dataset and code。
- **生物/样本**：Macaca fascicularis；in vivo, anaesthetized; single-cell recordings；Paper 2–15 degrees; verified raw subset 3.49–8.31 degrees from README；type：PC/midget, MC/parasol, S-on in paper; cell labels in public README；cells：15 unique cells with ten-minute raw files in frozen ZIP; paper 47 (26 PC,16 MC,5 S-on); archive coverage is not all 47；animals：Paper: 4 male animals; per-file animal mapping unknown。
- **记录与刺激**：extracellular spike event timestamps plus timing pulses；10-minute natural scene movie; separate 6x1-minute responses not raw-verified as a complete stimulus-response pair here。Response sampling：Paper event precision 0.1 ms; raw numeric timestamp unit not read; amplifier sampling UNVERIFIED；stimulus timing：150 Hz CRT/frame timing; paper versus README acquisition-rate multiplier conflicts (6x vs 3x)。
- **刺激合同**：256x256 over 4.6x4.6 degrees; 4-mm artificial pupil；units：degrees；contrast：Original RGB/gamma and relative cone excitations; project derives (L+M-background)/background Weber proxy；adaptation：Equal-energy white/matched movie mean; absolute R*/cone/s for 10-minute file UNVERIFIED；channels：RGB display; relative L/M/S conversion in README。
- **校准/元数据**：Gamma, gun spectra and geometry available; spectra in relative units; absolute retinal photon catch unresolved。Stimulus files：YES: stimuli/1x10_256.mpg; Spaceflowershow.mpg/timeflowershow.mpg are previews, not replacements for missing 6x1_256.mpg。Trials：751 blank + 90000 live + 750 blank frames in 10-minute file; audio synchronization every 5 s; no response content inspected；repeats：10-minute sequence; separate first-minute x6 protocol exists but full matching repeat movie absent from frozen ZIP；cell map：Filename cell key + README table; README 43 records/25 keys differs from frozen archive 28 raw files/15 keys。Preprocessing：Spikes detected/sorted upstream; unaveraged event times preserved; processed run_model files separate。
- **跨层/独立性**：UNVERIFIED；UNVERIFIED。Public subset is not automatically independent of existing project experiments; no new target split consumed in this audit。
- **映射**：`DIRECT_MAPPING` → RGC probability/logit。Existing adapter supports these spike events, conditional on matching exact archive and cell manifest。候选 head：`Existing Bernoulli occupancy readout; preserve dt, spike-history and trial boundaries`。
- **Nuisance/noise**：scale=Existing readout only；offset=Existing readout bias only；filter=None on spike target；noise=Bernoulli occupancy; no claims from this audit about fitted noise。
- **角色**：`TRAINING_CONSTRAINT`；当前：YES for a traceable RGC-only subset; NO for multilevel raw training。激活条件：Freeze archive/cell subset before later target access; physiological absolute background remains unresolved。联系作者：YES for full original records, missing one-minute movie and absolute calibration; NO for listed ten-minute subset。
- **实际检查**：HTTP Range: 22-byte ZIP footer + 8096-byte central directory; 99 members; 28 raw TXT=15 ten-minute+13 six-repeat; no 6x1_256.mpg; 15 unique cell keys。接口：DOI static ZIP responds to bounded Range; README (~11.7 kB) and directory inspected, no response or movie body。
[来源1](https://doi.gin.g-node.org/10.12751/g-node.xage77/) · [来源2](https://doi.org/10.1113/JP281200) · [来源3](https://gin.g-node.org/Manuel/Macaque-ganglion-cells)

### CH24_CONE

**Chen 2024 averaged primate cone photocurrents** — Chen et al. Predictably manipulating photoreceptor light responses to reveal their role in downstream visual responses (Version of Record)；DOI `10.7554/eLife.93795.3`。

- **可得性**：`PROCESSED_ONLY`。README explicitly: 4–5 repeats averaged and divided by saturating-flash dark current。License：Dryad CC0; code MIT; Figure10 raw license UNVERIFIED。
- **生物/样本**：Macaca fascicularis / nemestrina / mulatta；ex vivo；Peripheral >20 degrees; per-cell location unknown；type：Peripheral L/M cones；cells：Fig2 fitting cohort 6 primate cones; total distinct archive cells not inspected；animals：UNVERIFIED。
- **记录与刺激**：whole-cell photocurrent; neither cone voltage nor release；Variable-mean Gaussian noise; mean changes every 500 ms; fitting and manipulation protocols。Response sampling：Archive timestep 0.1 ms；stimulus timing：Archive timestep 0.1 ms; LED waveform; noise bandwidth up to 60 Hz。
- **刺激合同**：Uniform 600-um disk; no pixel movie；units：um；contrast：Variable-mean Gaussian noise; actual per-trial contrast vector UNVERIFIED；adaptation：Variable means around 22000 R*/cone/s in model-fitting protocol; no general transfer of this value to RGC datasets；channels：405 nm LED; L/M activation difference <10%。
- **校准/元数据**：Methods: power, spectra, cone sensitivity, 0.37 um² collecting area; session calibration files unverified。Stimulus files：README describes stimulus in R*/s; MAT field list/pairing not inspected; actual paired stimulus vector UNVERIFIED。Trials：2–3 min recordings; archived average traces；repeats：4–5 averaged repeats; single trials not verified；cell map：Per-cell/animal/session manifest UNVERIFIED。Preprocessing：Trial averaging and dark-current normalization。
- **跨层/独立性**：Same physical preparation/animal UNVERIFIED；Figure10 step/flash examples shared, but that dataset is outside this Dryad export。Chen primate consensus parameters reuse Angueyra2022; that predecessor is not independent parameter validation。
- **映射**：`INDIRECT_FUNCTIONAL_CONSTRAINT` → cone-related input only。Current input is pooled L+M Weber contrast, not a phototransduction-current state。候选 head：`a*h_photocurrent+b only if such a corresponding state already exists; none established in current RetiPath`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=Residual model for averaged trace only; repeat-level variance and effective n unavailable。
- **角色**：`AUXILIARY_TRANSFER`；当前：NO for the full multilevel hypothesis。激活条件：Validate stimulus fields and state meaning; cannot supervise photocurrent by relabeling Weber input。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：Dryad API: Figure2-SourceData1.mat 41490462 bytes; README.md 1357 bytes; PrimateConeResponses plus mouse/rod variables。接口：Public version325663 file API inspected; only README/metadata read。
[来源1](https://pmc.ncbi.nlm.nih.gov/articles/PMC11537484/) · [来源2](https://datadryad.org/dataset/doi:10.5061/dryad.q2bvq83vg) · [来源3](https://datadryad.org/api/v2/versions/325663/files) · [来源4](https://github.com/chrischen2/photoreceptorLinearization)

### CH24_INNER

**Chen 2024 Figure10 downstream cross-layer physiology** — Chen et al. Predictably manipulating photoreceptor light responses to reveal their role in downstream visual responses (Version of Record)；DOI `10.7554/eLife.93795.3`。

- **可得性**：`PAPER_ONLY`。Figure10 HC/BC/RGC is outside Figures1–5 Dryad deposit; no explicit promise of raw provision verified。License：Dryad CC0; code MIT; Figure10 raw license UNVERIFIED。
- **生物/样本**：Macaca fascicularis / nemestrina / mulatta；ex vivo；Peripheral >20 degrees; per-cell location unknown；type：HC subtype unknown; 3 cone BC subtype unknown; ON parasol；cells：Step/flash:10 cones,5 parasol; sinusoid:3 HC,3 BC; not a unique summed cohort；animals：UNVERIFIED。
- **记录与刺激**：HC membrane voltage; RGC spike-derived firing rate; BC measurement modality UNVERIFIED (trajectory-area summary only)；Original and modified step+flash; sinusoidal and modified waveforms。Response sampling：UNVERIFIED；stimulus timing：60 Hz OLED for step+flash; full per-assay sampling unknown。
- **刺激合同**：UNVERIFIED；units：UNVERIFIED per trial；contrast：UNVERIFIED；adaptation：Photopic; actual Figure10 trial backgrounds not verified；channels：Calibrated OLED; channel waveforms unknown。
- **校准/元数据**：Methods available; Figure10 per-session calibration/actual waveforms not obtained。Stimulus files：NO verified Figure10 waveform files。Trials：Step/flash and sinusoidal experiments distinct；repeats：UNVERIFIED；cell map：Per-cell/animal/session manifest UNVERIFIED。Preprocessing：Published averaged traces, spike rates, normalized trajectory areas。
- **跨层/独立性**：Same physical preparation/animal UNVERIFIED；YES identical original/modified step+flash in illustrated cone/HC/RGC; same animal/prep unknown。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`INDIRECT_FUNCTIONAL_CONSTRAINT` → H1 candidate and RGC output; BC state unresolved。HC voltage may be candidate affine observable; BC modality missing; average firing rate is not raw spikes。候选 head：`HC: a*h_HC+b only after correspondence; RGC: existing readout on future raw events; BC: no eligible head specified`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`FUNCTIONAL_ASSAY_ONLY`；当前：NO for the full multilevel hypothesis。激活条件：Author raw Figure10 package, modality and lineage; do not digitize plotted curves as raw。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：Dryad lists no Figure10 file; original Figure10 image labels HC mV and parasol sp/s。接口：https://cdn.elifesciences.org/articles/93795/elife-93795-fig10-v1.jpg (figure only)。
[来源1](https://pmc.ncbi.nlm.nih.gov/articles/PMC11537484/) · [来源2](https://datadryad.org/dataset/doi:10.5061/dryad.q2bvq83vg) · [来源3](https://datadryad.org/api/v2/versions/325663/files) · [来源4](https://github.com/chrischen2/photoreceptorLinearization)

### BA19

**Baudin 2019 cone and small-bistratified source-data archive** — Baudin et al. S-cone photoreceptors in the primate retina are functionally distinct from L and M cones；DOI `10.7554/eLife.39166`。

- **可得性**：`UNVERIFIED`。Public ZIP listed; internal stimulus/response fields and trial granularity not verified after bounded access failures。License：Dryad CC0。
- **生物/样本**：Macaca fascicularis / nemestrina / mulatta；ex vivo, foveal and peripheral pieces；Foveal/peripheral groups; exact cutoff not verified；type：L/M/S cones and small bistratified RGC；cells：At5000 R*/s: peripheral L49/M26/S36; foveal L25/M24/S29; not archive unique total；animals：15 peripheral pieces +7 foveal pieces; animal count unknown。
- **记录与刺激**：Cone voltage clamp(-60mV) and current clamp(0pA); RGC extracellular spikes；10-ms flashes, noise, sinusoids,3-s steps。Response sampling：10000 Hz;3000 Hz low-pass；stimulus timing：10000 Hz LED waveforms; noise bandwidth0–60Hz。
- **刺激合同**：Uniform ~500-um disk；units：um；contrast：Noise SD/mean=50%；adaptation：>1h dark adaptation; protocol-specific backgrounds; cone noise2500 and SBC noise1000/10000 R*/s；channels：406/515/640 nm LEDs。
- **校准/元数据**：Methods: measured power/spectra and0.37 um² collecting area; per-trial calibration not checked。Stimulus files：UNVERIFIED inside ZIP。Trials：Protocol-specific source/examples；repeats：Flash plots5–10 averages; archived single-trial status unknown；cell map：UNVERIFIED。Preprocessing：Averaged/normalized flashes and derived linear filters exist; single-trial content not inspected。
- **跨层/独立性**：UNVERIFIED；Protocol-family overlap; same trial/preparation unverified。Saha2024 Fig6A-C/SuppFig6 reuse BA19; those rows cannot be independent validation。
- **映射**：`INDIRECT_FUNCTIONAL_CONSTRAINT` → Peripheral L/M cone-related input candidate; S-cone/SBC outside pooled L+M contract。Peripheral L/M subset may constrain early dynamics; NO_CLEAN_MAPPING for S-specific pathway under current input。候选 head：`Separate a*h_current+b or a*h_voltage+b only for the matching existing observable; not interchangeable`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`HELD_OUT_PHYSIOLOGY_VALIDATION`；当前：NO for the full multilevel hypothesis。激活条件：Verify inner archive stimulus/trial fields and reserve disjoint peripheral L/M/control cells; no held-out data currently accepted。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：API version23398: Data for Upload.zip188452270 bytes; usage notes name flash_responses.mat,sbc_examples.mat,cellular_noise.mat,cone_linear_filters.mat; inner directory not verified。接口：https://datadryad.org/api/v2/versions/23398/files ; Range403/API download401; this is a tool access limit, not proof of globally restricted data。
[来源1](https://elifesciences.org/articles/39166) · [来源2](https://datadryad.org/api/v2/versions/23398/files) · [来源3](https://datadryad.org/dataset/doi:10.5061/dryad.gv5k2j3)

### SA24

**Saha 2024 regional cone adaptation (full raw request)** — Saha et al. Regional tuning of photoreceptor adaptation in the primate retina；DOI `10.1038/s41467-024-53061-3`。

- **可得性**：`AUTHOR_REQUEST_REQUIRED`。Paper explicitly requests lead-contact access for other data; public source XLSX is a processed tier, not verified raw pairs。License：Article CC BY-NC-ND4.0; raw reuse license UNVERIFIED。
- **生物/样本**：Macaca fascicularis / nemestrina / mulatta；ex vivo; control and drug conditions distinct；Foveal <0.5mm; peripheral >6mm；type：L/M and selected S cones；cells：Fig1 up to30 cells per group/condition; unique total unknown；animals：UNVERIFIED。
- **记录与刺激**：Photovoltage(current clamp0pA); separate voltage-step HCN currents；10-ms flashes, steps/flash-on-step; command-step experiments separate。Response sampling：10000 Hz;3000 Hz low-pass；stimulus timing：10000 Hz LED waveform。
- **刺激合同**：Uniform ~500-um disk；units：um; eccentricity mm；contrast：Example flash300%; actual trial contrast list unverified；adaptation：>1h dark adaptation; backgrounds0–50000 R*/s; <=4min recordings；channels：410/505/650nm LEDs。
- **校准/元数据**：Methods: power/spectrum,0.37um² collecting area; no verified session calibration table。Stimulus files：Full raw stimulus-response pairs not verified。Trials：UNVERIFIED；repeats：UNVERIFIED；cell map：UNVERIFIED。Preprocessing：Averaging/gain normalization; adaptation analysis includes50ms Savitzky–Golay and fitted summaries。
- **跨层/独立性**：UNVERIFIED；UNVERIFIED。EXCLUDE reused BA19 S-cone Fig6A-C and SupplementaryFig6 from independence claims。
- **映射**：`INDIRECT_FUNCTIONAL_CONSTRAINT` → Cone-related input; current model lacks photovoltage state。Photovoltage cannot be identified with photocurrent or pooled contrast through an arbitrary temporal filter。候选 head：`a*h_voltage+b only with a matching existing voltage state; not currently established`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`HELD_OUT_PHYSIOLOGY_VALIDATION`；当前：NO for the full multilevel hypothesis。激活条件：Request new untreated peripheral L/M trials and IDs; foveal/S/drug subsets remain separate transfer assays。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：Source Data XLSX linked; raw data statement explicitly on request; workbook contents not read。接口：https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-024-53061-3/MediaObjects/41467_2024_53061_MOESM4_ESM.xlsx。
[来源1](https://www.nature.com/articles/s41467-024-53061-3)

### SH20

**Shah 2020 processed nonlinear subunit dataset** — Shah et al. Inference of nonlinear receptive field subunits with spike-triggered clustering；DOI `10.7554/eLife.45743`。

- **可得性**：`PROCESSED_ONLY`。Deposited stimulus is temporally prefiltered with STA-derived filter; response is Time x cells matrix。License：DataCC0-1.0; codeGPL-3.0。
- **生物/样本**：Macaca sp.；ex vivo, isolated or RPE-attached；6–15mm temporal-equivalent peripheral；type：ON/OFF parasol; main subunit analysis OFF parasol；cells：Fig2 main cohort91 OFF parasol; unique deposit total unknown；animals：7 macaques in paper; public figure-to-animal mapping unknown。
- **记录与刺激**：Extracellular spikes after sorting/binned matrix; no BC recordings；Binary white noise, closed-loop RF-null; paper natural-scene Fig7 not present in verified deposit。Response sampling：Raw acquisition20kHz; public matrix bin metadata unverified；stimulus timing：CRT120Hz; WN new frames60Hz(Fig2),30Hz(Fig5/6)。
- **刺激合同**：White-noise pixels41.6 or20.8um; per-panel dimensions in schema；units：um；contrast：Michelson (max-min)/(max+min):48% or96%；adaptation：WN/null L/M800–2200,S400–900 R*/cone/s；channels：RGB CRT calibrated cone catches; primarily achromatic patterns。
- **校准/元数据**：Paper reports catches and instruments; per-session files not verified。Stimulus files：YES processed filtered matrices; original unfiltered frames UNVERIFIED。Trials：Fig2:30min WN,24min estimation,3min held-out; RF-null10s x30；repeats：RF-null30 repeats; naturalistic40 presentations described only, no Fig7 file；cell map：Figure/panel/response columns; cross-panel cell IDs unknown。Preprocessing：Spike sorting/binning; stimulus temporal prefilter from STA。
- **跨层/独立性**：UNVERIFIED；UNVERIFIED。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`INDIRECT_FUNCTIONAL_CONSTRAINT` → RGC functional nonlinearity; not BC state。Fitted subunits are not measured BC; prefiltered stimuli do not cleanly drive full dynamic RetiPath。候选 head：`Existing spike readout only after original stimulus/bin contract recovered; no head on inferred subunits`。
- **Nuisance/noise**：scale=Existing RGC readout only；offset=Existing bias only；filter=Published input already filtered; do not add a fitted inverse/adapter；noise=Counts versus occupancy must be determined from actual schema; no Poisson/Bernoulli assumption on averages。
- **角色**：`FUNCTIONAL_ASSAY_ONLY`；当前：NO for the full multilevel hypothesis。激活条件：Use published spatial/null assay interpretation; full dynamic training requires original inputs and bins。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：Five files:Figure_2.pkl256561638B;Figure_3.pkl4966556348B;Figure_4.pkl587967132B;Figure_5.mat21687B;Figure_6.pkl954610774B; NO Figure7。接口：https://datadryad.org/api/v2/versions/48728/files (five public files and download links inspected)。
[来源1](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7062463/fullTextXML) · [来源2](https://datadryad.org/api/v2/versions/48728/files) · [来源3](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxkvk) · [来源4](https://github.com/Chichilnisky-Lab/shah-elife-2020/blob/5efeaf8b4e8c35acb1c801fc32c77e841698d0e0/README.md)

### KA25

**Karamanlis 2025 marmoset original event-data mirror** — Karamanlis et al. Nonlinear receptive fields evoke redundant retinal coding of natural scenes (online2024; issue2025)；DOI `10.1038/s41586-024-08212-3`。

- **可得性**：`RAW_VERIFIED_AVAILABLE`。Original-session ZIP directories and Manual confirm event timestamps/unit IDs, displayed images+fixations and frame onset/offset signals; not raw MEA voltage。License：CC BY-SA4.0。
- **生物/样本**：Callithrix jacchus (mouse sessions excluded)；ex vivo；Peripheral7–10mm from fovea；type：Functional ON/OFF parasol,ON/OFF midget; unclassified separate；cells：Natural-video Fig1 typed166; model Fig4 typed165; flashed-images240 typed+193 unclassified; archive unique total not equated to these；animals：3 adult males in paper;6 archive sessions are not6 animals。
- **记录与刺激**：Extracellular spike events after Kilosort/manual sorting；Eye-movement-driven natural images,white noise,flashed images,flashed/flickering gratings。Response sampling：MEA acquisition metadata expdata.fs; event timestamps stored as sample indices；stimulus timing：Marmoset85Hz; WN21.25–85Hz depending Nblinks; mouse75Hz must not be substituted。
- **刺激合同**：800x600 projector;7.5um/pixel or some marmoset2.5um; use session projector metadata；units：um；contrast：WN binary100%; movie average contrast45%(train)/38%(test); image mean differs; formula/session definitions need verification；adaptation：Low photopic;~3000 M-cone R*/s and6000 rod R*/s; test mean10% below background；channels：Grayscale; no independent L/M/S channels。
- **校准/元数据**：Paper estimates catches; Manual gives projector metadata; complete per-session radiometric chain not verified。Stimulus files：YES images,fixation coordinates and fonsets/foffsets specified in actual named MAT containers。Trials：30–35cycles:40 sampled training images from325 and22 fixed test images;1s/image；repeats：Fixed test images repeat across cycles; exact per-session count metadata later；cell map：expdata.units Kilosort ID/electrode/quality;typelabels and cellclus_id; animal linkage needs checking。Preprocessing：Spike detection/sorting; unaveraged spiketimes retained alongside spikesbin/frozenbin/runningbin。
- **跨层/独立性**：UNVERIFIED；UNVERIFIED。Original GIN and OpenRetina mirror same data family; not independent validation。
- **映射**：`DIRECT_MAPPING` → RGC probability/logit。RGC events have appropriate output observable; no measured HC/BC/AC state。候选 head：`Existing Bernoulli occupancy readout with verified clock conversion; expected count is not raw occupancy`。
- **Nuisance/noise**：scale=Existing readout only；offset=Existing bias only；filter=No smoothing of event targets; preserve fonsets AND foffsets；noise=Bernoulli only after event-to-bin conversion; counts require declared appropriate likelihood。
- **角色**：`AUXILIARY_TRANSFER`；当前：YES event-level RGC input/output candidate; no multilevel data。激活条件：Select marmoset sessions; freeze unit IDs,85Hz timing,actual projector geometry and heldout split。联系作者：Not needed for listed files; needed if animal/calibration mapping absent。
- **实际检查**：6 marmoset session ZIP central directories verified;3 movie sessions contain expdata.mat,fixationmovie_data.mat,frozencheckerflicker_data.mat,gratingflicker_data.mat;3 image sessions analogous。接口：HF API + HTTP206 ZIP EOCD22B and directory~600B/session; Manual.pdf188091B read; no response body。
[来源1](https://www.nature.com/articles/s41586-024-08212-3) · [来源2](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11711096/fullTextXML) · [来源3](https://huggingface.co/api/datasets/open-retina/open-retina/tree/main/gollisch_lab/karamanlis_2024/sessions) · [来源4](https://huggingface.co/datasets/open-retina/open-retina/resolve/main/gollisch_lab/karamanlis_2024/Manual.pdf)

### SR26

**Sridhar 2026 marmoset OpenRetina processed mirror** — Sridhar et al. Modeling spatial contrast sensitivity in responses of primate retinal ganglion cells to natural movies (2026-04-07)；DOI `10.1371/journal.pcbi.1014157`。

- **可得性**：`PROCESSED_ONLY`。HF processed response PKL listing verified; original47GiB GIN raw directory not verified (timeout)。License：DataCC BY4.0; Tears of Steel movieCC BY3.0。
- **生物/样本**：Callithrix jacchus；ex vivo；Retinal pieces; precise eccentricity UNVERIFIED；type：OFF midget/parasol,ON parasol,Large OFF(combined clusters,not one confirmed type)；cells：Paper2retinas/180reliable,170main; mirror4pieces370+181+495+110=1156unfiltered; no identity bridge verified；animals：Unknown;2retinas does not establish2animals。
- **记录与刺激**：Extracellular spikes; mirror processed response bins；Tears of Steel movie+simulated fixations/saccades,white noise,reversing gratings。Response sampling：Original25kHz,300Hz–5kHz bandpass; model bins85Hz;grating analysis5ms；stimulus timing：85Hz projector;24Hz source movie resampled。
- **刺激合同**：800x600,7.5um/pixel;WN200x150 tiles30um;mirror movie200x150 after downsampling；units：um；contrast：WN100%Michelson (analysis ±1 Weber); movie RMS45%；adaptation：WN4.6/5.5mW/m²;M-cone2400–2900,S380–450,rod7200–8600R*/s;movie mean76%ofWN；channels：Grayscale display。
- **校准/元数据**：Paper physical irradiance and catches; session-specific chain/versions not reconciled。Stimulus files：YES processed movie/fixation archives and WN files listed; original display reconstruction not validated。Trials：Movie10x300s training with10interleaved test repeats;WN running150/300s+frozen30/60s；repeats：Movie10test repeats;WN session-dependent；cell map：Four response PKL piece IDs; mapping to paper2retinas/170cells UNVERIFIED。Preprocessing：Kilosort+Phy2;response bins;stimulus cropping/downsampling;no raw event array checked。
- **跨层/独立性**：UNVERIFIED；UNVERIFIED。Original+old/new DOI+OpenRetina+derivative model papers overlap; cannot be independent by publication name。
- **映射**：`DIRECT_MAPPING` → RGC output。Output bins can map after bin semantics verified; not membrane/AC recording。候选 head：`Existing spike likelihood after establishing whether PKL stores counts,occupancy or rates; no arbitrary adapter`。
- **Nuisance/noise**：scale=Existing readout only；offset=Existing bias only；filter=No fitted filter; account for fixed published downsampling/binning；noise=UNVERIFIED target semantics; do not treat all PKL values as Bernoulli。
- **角色**：`AUXILIARY_TRANSFER`；当前：NO for the full multilevel hypothesis。激活条件：Reconcile original DOI version,paper cohort,mirror4pieces,raw event/bin semantics and per-session adaptation。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：NM responses.zip91470833B,fixations.zip9112539B,stimuli_padded.zip9760517215B;544B ZIP directory lists cell_responses_01..04_fixation_movie.pkl;WN lists response/stimulus ZIPs。接口：https://huggingface.co/api/datasets/open-retina/open-retina/tree/main/gollisch_lab/sridhar_2025/marmoset/natural_movie ; https://huggingface.co/api/datasets/open-retina/open-retina/tree/main/gollisch_lab/sridhar_2025/marmoset/whitenoise。
[来源1](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014157) · [来源2](https://open-retina.org/package_docs/datasets/sridhar_2025/) · [来源3](https://doi.gin.g-node.org/10.12751/g-node.t43ph1/) · [来源4](https://huggingface.co/api/datasets/open-retina/open-retina/tree/main/gollisch_lab/sridhar_2025/marmoset/natural_movie)

### KR23

**Krueppel 2023 marmoset saccadic image shifts** — Krueppel et al. Diversity of Ganglion Cell Responses to Saccade-like Image Shifts in the Primate Retina；DOI `10.1523/JNEUROSCI.1561-22.2023`。

- **可得性**：`UNVERIFIED`。Public1.1GiB archive registered; raw response+stimulus directory not verified after GIN timeouts。License：CC BY-SA4.0。
- **生物/样本**：Callithrix jacchus；ex vivo；UNVERIFIED；type：ON/OFF midget,ON/OFF parasol,Large OFF；cells：80–600cells/recording;unique total unknown；animals：4animals/4retinas;3male1female。
- **记录与刺激**：extracellular spike；Saccade-like square-wave grating shifts,white noise。Response sampling：UNVERIFIED；stimulus timing：Fixation533ms,shift67ms;refresh rate unverified。
- **刺激合同**：90um grating stripe width;full grid unknown；units：um；contrast：UNVERIFIED；adaptation：UNVERIFIED；channels：UNVERIFIED。
- **校准/元数据**：UNVERIFIED。Stimulus files：UNVERIFIED actual listing。Trials：UNVERIFIED；repeats：UNVERIFIED；cell map：UNVERIFIED。Preprocessing：UNVERIFIED。
- **跨层/独立性**：UNVERIFIED；UNVERIFIED。Overlap with other Gollisch animals/sessions UNVERIFIED。
- **映射**：`DIRECT_MAPPING` → RGC output; saccadic suppression functional assay。Prospective event-output correspondence; no direct AC/HC recordings。候选 head：`Existing event likelihood after complete input/timing contract`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`NOT_CURRENTLY_USABLE`；当前：NO for the full multilevel hypothesis。激活条件：Verify manifest and per-cell type/calibration; then candidate held-out RGC function, not HC/BC validation。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：DOI landing only; no verified inner directory。接口：GIN repository/API timeout; not proof of restricted data。
[来源1](https://pmc.ncbi.nlm.nih.gov/articles/PMC10359029/) · [来源2](https://doi.gin.g-node.org/10.12751/g-node.thlt1j/)

### SH25

**Shahidi 2025 marmoset full-field suppression data candidate** — Shahidi et al. Filter-based models of suppression in retinal ganglion cells: Comparison and generalization across species and stimuli；DOI `10.1371/journal.pcbi.1013031`。

- **可得性**：`UNVERIFIED`。GRO.data API certificate error; Figshare initial-value table is not raw data。License：UNVERIFIED。
- **生物/样本**：Callithrix jacchus subset only;mouse/axolotl excluded；ex vivo；UNVERIFIED；type：ON/OFF,transient/sustained groups;not verified midget/parasol；cells：10marmoset recordings;unique cells unknown；animals：UNVERIFIED。
- **记录与刺激**：extracellular spike；Full-field Gaussian noise,steps,contrast/frequency chirps。Response sampling：UNVERIFIED；stimulus timing：60Hz OLED。
- **刺激合同**：800x600,7.5um/pixel;uniform full-field stimulation；units：um；contrast：Noise SD/mean30%；adaptation：Mesopic-low photopic0.75–2.8mW/m²；channels：Grayscale OLED。
- **校准/元数据**：Reported irradiance; per-session spectrum/metadata not verified。Stimulus files：UNVERIFIED。Trials：Marmoset WN pseudo-trials33.3s with6.7s heldout；repeats：NO frozen WN repeats in marmoset;chirp15repeats；cell map：UNVERIFIED。Preprocessing：UNVERIFIED。
- **跨层/独立性**：UNVERIFIED；UNVERIFIED。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`DIRECT_MAPPING` → RGC output only。Prospective output correspondence; suppressive model filters are not AC/HC measurements。候选 head：`Existing spike readout after timing/bin verification`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`NOT_CURRENTLY_USABLE`；当前：NO for the full multilevel hypothesis。激活条件：Verify manifest; full-field stimulus cannot identify spatial H1 geometry alone。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：No verified GRO.data file list; Figshare28924358 is5.5kB initial-parameter table。接口：Repository API TLS certificate verification failed; no data body accessed。
[来源1](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013031) · [来源2](https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/AP5BRQ) · [来源3](https://plos.figshare.com/articles/dataset/Initial_values_for_5_training_runs_/28924358)

### KI22_AC

**Kim 2022 starburst/polyaxonal AC physiology** — Kim et al. Origins of direction selectivity in the primate retina；DOI `10.1038/s41467-022-30405-5`。

- **可得性**：`AUTHOR_REQUEST_REQUIRED`。Explicit reasonable-request statement;source XLSX and model code are not verified raw waves。License：Article/source dataCC BY;requested raw license UNVERIFIED。
- **生物/样本**：Macaca nemestrina/fascicularis/mulatta；ex vivo flatmount,RPE/choroid retained,~36degreesC；AC physiology location UNVERIFIED;8mm morphological RGC sample is not AC location；type：ON starburst AC;ON-OFF A1/polyaxonal AC；cells：SAC radial Vm n6/RF n27;PAC Vm n5/spikes n12;not unique total；animals：Whole paper76animals/106retinas;not AC-specific count。
- **记录与刺激**：SAC current-clamp Vm (subset voltage-clamp inward current);PAC intracellular Vm+spikes；Radial/drifting gratings,spots,motion。Response sampling：10000Hz;Bessel2or5kHz;per-cell setting unknown；stimulus timing：UNVERIFIED;23.4Hz calcium acquisition is NOT visual refresh。
- **刺激合同**：SAC grating0.24cycles/deg0.5Hz;spots50–720um500ms;PAC grating200um period4Hz；units：um and degrees by assay；contrast：Reported100% grating contrast;precise definition unverified；adaptation：High photopic;Fig7~1e5R*/cone/s;per-trial backgrounds unknown；channels：MeasuredRGB peaks636/550/465nm。
- **校准/元数据**：PR705spectra and paper catch estimates;session calibration absent。Stimulus files：No verified raw displayed waveform/timing files。Trials：Cell/assay-specific;no public trial manifest；repeats：Full independent repeat counts unknown；cell map：No raw animal-retina-cell-epoch map obtained。Preprocessing：Published source summaries;raw transformation metadata unverified。
- **跨层/独立性**：Physical AC-RGC pairing/IDs UNVERIFIED；Related motion assays;paired raw stimulus not established。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`NO_CLEAN_MAPPING` → No identified SAC/PAC state。Current generic AC filters have no SAC/PAC-specific membrane/spike or directional dendritic observable。候选 head：`None justified;affine head cannot supply missing cell-specific computation`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`FUNCTIONAL_ASSAY_ONLY`；当前：NO for the full multilevel hypothesis。激活条件：Context or future independently authorized observable audit;not current genericAC supervision。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：Paper source XLSX;Zenodo6495531 simulation/morphology record;no rawAC manifest。接口：Primary published PDF and data statement inspected;no raw download。
[来源1](https://www.nature.com/articles/s41467-022-30405-5) · [来源2](https://bpb-us-e1.wpmucdn.com/sites.uw.edu/dist/3/8419/files/2022/05/Kim_et_al-2022-Nature_Communications.pdf) · [来源3](https://doi.org/10.5281/zenodo.6495531)

### HR21

**HumRet human RGC event/protocol export** — Reinhard & Muench. Visual properties of human retinal ganglion cells；DOI `10.1371/journal.pone.0246952`。

- **可得性**：`PROCESSED_ONLY`。Sorted events and processed exports listed;actual delivered stimulus+timing reconstruction not verified. Sorting alone is NOT the downgrade criterion。License：Repository data/code license UNVERIFIED;do not transfer articleCC BY。
- **生物/样本**：Homo sapiens；ex vivo enucleation tissue;25degreesC；Mid-peripheral;exact positions unknown；type：Presumed RGC;16functional clusters,not confirmed midget/parasol；cells：342light-responsive units;15pieces/10retinas；animals：10retinas;donor count not inferred。
- **记录与刺激**：extracellular spike；Steps,chirps,drifting gratings,moving bars。Response sampling：25000Hz；stimulus timing：UNVERIFIED;1–8Hz grating frequency is not frame rate。
- **刺激合同**：Grating periods100–4000um；units：um；contrast：Linearized grayscale0/128/255;formula unverified；adaptation：8e4rodR*/s;two pieces8e5;cone state uncertain；channels：Grayscale。
- **校准/元数据**：Rod catch estimates;session cone calibration unverified。Stimulus files：Protocol.phys+generation scripts listed;delivered waveform/clock pairing unverified。Trials：Repeated stimuli during2–6h recording；repeats：Protocol-dependent;unverified；cell map：Date/unit IDs and analyzed file ranges。Preprocessing：Offline sorting;event MAT+processed summaries。
- **跨层/独立性**：UNVERIFIED；UNVERIFIED。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`DIRECT_MAPPING` → RGC probability/logit。Conditional event-output correspondence;identity and timing remain gates。候选 head：`Existing probabilistic spike readout`。
- **Nuisance/noise**：scale=Existing readout only；offset=Existing bias only；filter=No target smoothing；noise=Existing occupancy likelihood after bin verification。
- **角色**：`AUXILIARY_TRANSFER`；当前：NO for the full multilevel hypothesis。激活条件：Resolve actual stimulus/timing,cell type and license;no automatic numerical pooling。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：GitHub tree656a36b8dd4c9c879561d71401413b8fafdf46d3: data/20120210/units/K20120210_CH10_sort1-0_unit_0001.mat;.phys;h_info.mat;h_responding.mat;stimulus scripts。接口：Pinned reader accesses Stimulus Protocol and per-stimulus spikes;no response arrays read;OSF401。
[来源1](https://pmc.ncbi.nlm.nih.gov/articles/PMC7886124/) · [来源2](https://github.com/katjaReinhard/HumRet/blob/656a36b8dd4c9c879561d71401413b8fafdf46d3/_scripts/EXAMPLE_ReadData.m)

### AB22

**Abbas 2022 human population ERG export** — Abbas et al. Revival of light signalling in the postmortem mouse and human retina；DOI `10.1038/s41586-022-04709-x`。

- **可得性**：`PROCESSED_ONLY`。README labels processed data and raw uploads in progress;no standalone stimulus files。License：CC BY4.0 verified in datacite.yml。
- **生物/样本**：Homo sapiens subset；ex vivo donor;37degreesC；Macula/periphery punches；type：Population photoreceptor/ON-bipolar ERG components;not single identified cells；cells：N/A population；animals：Paper17research donor pairs plus organ-donor cohorts;archive not equated to total。
- **记录与刺激**：transretinal field potential/ERG；Calibrated flashes。Response sampling：UNVERIFIED；stimulus timing：UNVERIFIED。
- **刺激合同**：6mm punch;2mm aperture for research donor setup；units：mm;photons/um²；contrast：Flash intensity,not Weber movie；adaptation：Dark-adapted and pharmacologically separated conditions；channels：UNVERIFIED。
- **校准/元数据**：Intensity units/methods;per-trial files not verified。Stimulus files：NO standalone waveform files in archive listing。Trials：Donor/punch/condition exports；repeats：UNVERIFIED；cell map：Donor,age,sex,punch area headers;no cell IDs。Preprocessing：Component isolation and subtraction;processed figure exports。
- **跨层/独立性**：Donor-level only；Some flash conditions;not paired single-cell layers。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`NO_CLEAN_MAPPING` → Population functional physiology。ERG b-wave is not a BC membrane/release trace。候选 head：`None justified for current cell-level states;field aggregation would require additional independently supported model`。
- **Nuisance/noise**：scale=Insufficient alone；offset=Insufficient alone；filter=Acquisition filter unknown；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`FUNCTIONAL_ASSAY_ONLY`；当前：NO for the full multilevel hypothesis。激活条件：Population context only;no direct BC supervision。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：ZIP569011362B/67entries;8DonorBWaves+16DonorRodsAndCones CSV;README/datacite+twoheaders only。接口：Static DOI ZIP HTTP Range;no numerical responses;Donor07_LPeriphery_1.csv and Donor09_LMacula_1.csv headers。
[来源1](https://pmc.ncbi.nlm.nih.gov/articles/PMC10000337/) · [来源2](https://doi.gin.g-node.org/10.12751/g-node.sayvud/)

### KW21

**Kawai 2021 human bipolar channel physiology** — Kawai et al. A subset of cone bipolar cells expresses the Na+ channel SCN2A in the human retina；DOI `10.1016/j.exer.2020.108299`。

- **可得性**：`PAPER_ONLY`。Paper located;no public paired physiology deposit verified。License：UNVERIFIED。
- **生物/样本**：Homo sapiens；Isolated bipolar cells；UNVERIFIED；type：ON/OFF-layer cone BC;channel subset；cells：Inward current2/11ON-layer,0/4OFF-layer;not total visual cohort；animals：UNVERIFIED。
- **记录与刺激**：Voltage-clamp ionic currents;RT-PCR separate；Depolarizing commands+TTX,not light。Response sampling：UNVERIFIED；stimulus timing：UNVERIFIED。
- **刺激合同**：N/A；units：N/A；contrast：N/A；adaptation：N/A command experiment；channels：N/A。
- **校准/元数据**：N/A optical。Stimulus files：No command files verified。Trials：UNVERIFIED；repeats：UNVERIFIED；cell map：UNVERIFIED。Preprocessing：UNVERIFIED。
- **跨层/独立性**：UNVERIFIED；UNVERIFIED。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`NO_CLEAN_MAPPING` → No eligible visual BC observable。Na-channel current is not generic BC state/release。候选 head：`None justified`。
- **Nuisance/noise**：scale=N/A；offset=N/A；filter=N/A；noise=N/A。
- **角色**：`NOT_CURRENTLY_USABLE`；当前：NO for the full multilevel hypothesis。激活条件：Does not satisfy visual-input contract。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：No matched deposit found。接口：UNVERIFIED。
[来源1](https://pubmed.ncbi.nlm.nih.gov/33068627/)

### SN04

**Shen 2004 cultured human horizontal receptor currents** — Shen et al. Glutamate receptor subtypes in human retinal horizontal cells；DOI `10.1017/S0952523804041094`。

- **可得性**：`PAPER_ONLY`。Paper only;no public matched raw deposit verified。License：UNVERIFIED。
- **生物/样本**：Homo sapiens；Cultured HC；UNVERIFIED；type：HC;H1/H2 unverified；cells：UNVERIFIED；animals：UNVERIFIED。
- **记录与刺激**：Whole-cell receptor currents；AMPA/kainate agonists/antagonists,not light。Response sampling：UNVERIFIED；stimulus timing：UNVERIFIED。
- **刺激合同**：N/A；units：N/A；contrast：N/A；adaptation：N/A；channels：N/A。
- **校准/元数据**：N/A optical。Stimulus files：No chemical/command waveform files verified。Trials：UNVERIFIED；repeats：UNVERIFIED；cell map：UNVERIFIED。Preprocessing：UNVERIFIED。
- **跨层/独立性**：UNVERIFIED；UNVERIFIED。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`NO_CLEAN_MAPPING` → No clean H1 observable。Cultured receptor current is not intact-retina HC Vm。候选 head：`None justified`。
- **Nuisance/noise**：scale=N/A；offset=N/A；filter=N/A；noise=N/A。
- **角色**：`NOT_CURRENTLY_USABLE`；当前：NO for the full multilevel hypothesis。激活条件：Excluded from direct visual supervision。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：No matched deposit found。接口：UNVERIFIED。
[来源1](https://pubmed.ncbi.nlm.nih.gov/15137585/)

### LI21_HC

**Liu 2021 horizontal-cell motion voltage** — Liu, Hong, Rieke & Manookin. Predictive encoding of motion begins in the primate retina；DOI `10.1038/s41593-021-00899-1`。

- **可得性**：`AUTHOR_REQUEST_REQUIRED`。Final article explicitly: data/analysis code on reasonable request。License：Raw license UNVERIFIED。
- **生物/样本**：Macaca fascicularis / mulatta / nemestrina；ex vivo wholemount,RPE removed；2–8mm;paper also states10–30degrees；type：Horizontal cell; H1/H2 subtype UNVERIFIED；cells：Figure6 HC n=2；animals：UNVERIFIED。
- **记录与刺激**：intracellular voltage,current clamp；Noise and two-/three-point motion gliders。Response sampling：10000Hz;3000Hz Bessel；stimulus timing：60Hz。
- **刺激合同**：Trial pixel geometry unknown；units：um；contrast：Michelson (source spelling Michaelson);0.25/0.5/1 protocol levels；adaptation：Medium/high photopic;Methods~1.5e4–5e5 L/M R*/s;per-cell levels unknown；channels：L/M-calibrated projector;channel files absent。
- **校准/元数据**：Paper catches;no session files verified。Stimulus files：No public waveform/seeds/frame logs verified。Trials：Protocol-specific epochs；repeats：UNVERIFIED；cell map：Animal/session/cell IDs not obtained。Preprocessing：Published binned information and averaged trace analyses。
- **跨层/独立性**：Physical pairing unknown;HC removes RPE,RGC retains RPE；Glider family;RGC polarity selected。Shared lab with Raval;disjoint animals/cells unverified;HC n=2 limits validation。
- **映射**：`INDIRECT_FUNCTIONAL_CONSTRAINT` → h1_state candidate。HC subtype is unverified; effective h1_state has not been validated as measured Vm。候选 head：`known_acquisition_filter(a*h1_state_at_registered_site+b),only after subtype/spatial/state correspondence`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`HELD_OUT_PHYSIOLOGY_VALIDATION`；当前：NO for the full multilevel hypothesis。激活条件：Obtain raw voltage/stimulus,subtype evidence and independent IDs;do not tune model/head on heldout response。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：No public listing;final on-request statement。接口：UNVERIFIED。
[来源1](https://pmc.ncbi.nlm.nih.gov/articles/PMC8728393/) · [来源2](https://pubmed.ncbi.nlm.nih.gov/34341586/) · [来源3](https://www.nature.com/articles/s41593-021-00899-1)

### LI21_RGC

**Liu 2021 RGC synaptic currents** — Liu, Hong, Rieke & Manookin. Predictive encoding of motion begins in the primate retina；DOI `10.1038/s41593-021-00899-1`。

- **可得性**：`AUTHOR_REQUEST_REQUIRED`。Final article explicitly: data/analysis code on reasonable request。License：Raw license UNVERIFIED。
- **生物/样本**：Macaca fascicularis / mulatta / nemestrina；ex vivo wholemount,RPE retained；2–8mm;paper also states10–30degrees；type：ON/OFF parasol and smooth monostratified；cells：EPSC ON15/OFF6;IPSC ON11/OFF4;overlap possible,do not sum；animals：UNVERIFIED。
- **记录与刺激**：RGC EPSC at~−70mV;IPSC at~0mV；Noise and two-/three-point motion gliders。Response sampling：10000Hz;3000Hz Bessel；stimulus timing：60Hz。
- **刺激合同**：Trial pixel geometry unknown；units：um；contrast：Michelson (source spelling Michaelson);0.25/0.5/1 protocol levels；adaptation：Medium/high photopic;Methods~1.5e4–5e5 L/M R*/s;per-cell levels unknown；channels：L/M-calibrated projector;channel files absent。
- **校准/元数据**：Paper catches;no session files verified。Stimulus files：No public waveform/seeds/frame logs verified。Trials：Protocol-specific epochs；repeats：UNVERIFIED；cell map：Animal/session/cell IDs not obtained。Preprocessing：Published binned information and averaged trace analyses。
- **跨层/独立性**：Physical pairing unknown;HC removes RPE,RGC retains RPE；Glider family;RGC polarity selected。Shared lab with Raval;disjoint animals/cells unverified;HC n=2 limits validation。
- **映射**：`INDIRECT_FUNCTIONAL_CONSTRAINT` → Pooled E/I current;indirect BC/AC function。RGC EPSC/IPSC are not directly recorded BC release/AC voltage。候选 head：`known_filter(a*signed_pooled_synaptic_current+b) only if driving-force/units match existing observable`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`HELD_OUT_PHYSIOLOGY_VALIDATION`；当前：NO for the full multilevel hypothesis。激活条件：Author currents/clamp/stimuli/independent IDs;held-out synaptic assay only。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：No public listing;final on-request statement。接口：UNVERIFIED。
[来源1](https://pmc.ncbi.nlm.nih.gov/articles/PMC8728393/) · [来源2](https://pubmed.ncbi.nlm.nih.gov/34341586/) · [来源3](https://www.nature.com/articles/s41593-021-00899-1)

### GR18_HC

**Grimes 2018 H1 same-mount modulation** — Grimes, Baudin, Azevedo & Rieke. Range, routing and kinetics of rod signaling in primate retina；DOI `10.7554/eLife.38281`。

- **可得性**：`PROCESSED_ONLY`。Figure-source XLSX listed;HC/AII trial-level raw and exact stimulus files not verified。License：Source data under articleCC BY;full raw experimental library license unverified。
- **生物/样本**：Macaca fascicularis / mulatta / nemestrina;mouse controls excluded；ex vivo wholemount,RPE removed；>=15degrees peripheral；type：H1；cells：Fig5 H1 n=4；animals：UNVERIFIED。
- **记录与刺激**：Whole-cell voltage；10ms flashes,sinusoidal modulations,light steps。Response sampling：UNVERIFIED；stimulus timing：UNVERIFIED。
- **刺激合同**：500–560um uniform disk；units：um；contrast：Fig5 short/long wavelength contrasts first equated in ON parasol at2Hz,then fixed forHC；adaptation：Rod/scotopic-mesopic;>=1h dark;new background>=1min;Fig5~10–20R*/rod/s,selected conditions~200R*/L-cone/s；channels：405/520/640nm LEDs。
- **校准/元数据**：Methods: spectra/collecting area to R*/photoreceptor/s;per-trial files not verified。Stimulus files：No exact per-trial waveforms verified。Trials：Figure-specific protocols；repeats：Example traces average5–20trials;raw trial IDs unknown；cell map：Figure groups;raw cell linkage unavailable。Preprocessing：Modulation summaries and averaged examples。
- **跨层/独立性**：YES Fig5 explicitly same physical retinal piece；YES Fig5 same background/contrast/stimulus。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`INDIRECT_FUNCTIONAL_CONSTRAINT` → H1 functional response。Rod-dominated stimulus not current cone-only input;public summaries not voltage traces。候选 head：`Affine voltage head only for future raw matched voltage state;none applied to summary statistics`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`FUNCTIONAL_ASSAY_ONLY`；当前：NO for the full multilevel hypothesis。激活条件：Published matched-condition function only;cannot train rawHC with XLSX modulation summaries。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：fig3-data1-v2.xlsx,fig4-data1-v2.xlsx,fig5-data1-v2.xlsx;Fig2MAT examples are rod/RGC,not rawHC。接口：https://elifesciences.org/articles/38281/figures (source links actually inspected)。
[来源1](https://elifesciences.org/articles/38281) · [来源2](https://elifesciences.org/articles/38281/figures)

### GR18_RGC

**Grimes 2018 ON-parasol same-mount modulation** — Grimes, Baudin, Azevedo & Rieke. Range, routing and kinetics of rod signaling in primate retina；DOI `10.7554/eLife.38281`。

- **可得性**：`PROCESSED_ONLY`。Figure-source XLSX listed;HC/AII trial-level raw and exact stimulus files not verified。License：Source data under articleCC BY;full raw experimental library license unverified。
- **生物/样本**：Macaca fascicularis / mulatta / nemestrina;mouse controls excluded；ex vivo wholemount,RPE removed；>=15degrees peripheral；type：Fig5 ON parasol;other figures include midget/OFF；cells：Fig5 ON parasol n=2;not total studyN；animals：UNVERIFIED。
- **记录与刺激**：Synaptic current;other study conditions extracellular spikes；10ms flashes,sinusoidal modulations,light steps。Response sampling：UNVERIFIED；stimulus timing：UNVERIFIED。
- **刺激合同**：500–560um uniform disk；units：um；contrast：Fig5 short/long wavelength contrasts first equated in ON parasol at2Hz,then fixed forHC；adaptation：Rod/scotopic-mesopic;>=1h dark;new background>=1min;Fig5~10–20R*/rod/s,selected conditions~200R*/L-cone/s；channels：405/520/640nm LEDs。
- **校准/元数据**：Methods: spectra/collecting area to R*/photoreceptor/s;per-trial files not verified。Stimulus files：No exact per-trial waveforms verified。Trials：Figure-specific protocols；repeats：Example traces average5–20trials;raw trial IDs unknown；cell map：Figure groups;raw cell linkage unavailable。Preprocessing：Modulation summaries and averaged examples。
- **跨层/独立性**：YES Fig5 same physical piece；YES Fig5 same background/contrast/stimulus。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`INDIRECT_FUNCTIONAL_CONSTRAINT` → RGC synaptic functional response。Processed modulation and rod routing;not direct cone-only spike supervision。候选 head：`Current-specific affine head only if correct existing signed current observable;spikes separate existing likelihood`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`FUNCTIONAL_ASSAY_ONLY`；当前：NO for the full multilevel hypothesis。激活条件：Matched rod-regime assay context only。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：fig3-data1-v2.xlsx,fig4-data1-v2.xlsx,fig5-data1-v2.xlsx;Fig2MAT examples are rod/RGC,not rawHC。接口：https://elifesciences.org/articles/38281/figures (source links actually inspected)。
[来源1](https://elifesciences.org/articles/38281) · [来源2](https://elifesciences.org/articles/38281/figures)

### GR18_AII

**Grimes 2018 AII rod-pathway source summaries** — Grimes, Baudin, Azevedo & Rieke. Range, routing and kinetics of rod signaling in primate retina；DOI `10.7554/eLife.38281`。

- **可得性**：`PROCESSED_ONLY`。Figure-source XLSX listed;HC/AII trial-level raw and exact stimulus files not verified。License：Source data under articleCC BY;full raw experimental library license unverified。
- **生物/样本**：Macaca fascicularis / mulatta / nemestrina;mouse controls excluded；ex vivo wholemount,RPE removed；>=15degrees peripheral；type：AII；cells：Assay-specific;uniqueN unverified；animals：UNVERIFIED。
- **记录与刺激**：AII whole-cell voltage；10ms flashes,sinusoidal modulations,light steps。Response sampling：UNVERIFIED；stimulus timing：UNVERIFIED。
- **刺激合同**：500–560um uniform disk；units：um；contrast：Fig5 short/long wavelength contrasts first equated in ON parasol at2Hz,then fixed forHC；adaptation：Rod/scotopic-mesopic;>=1h dark;new background>=1min;Fig5~10–20R*/rod/s,selected conditions~200R*/L-cone/s；channels：405/520/640nm LEDs。
- **校准/元数据**：Methods: spectra/collecting area to R*/photoreceptor/s;per-trial files not verified。Stimulus files：No exact per-trial waveforms verified。Trials：Figure-specific protocols；repeats：Example traces average5–20trials;raw trial IDs unknown；cell map：Figure groups;raw cell linkage unavailable。Preprocessing：Modulation summaries and averaged examples。
- **跨层/独立性**：Fig5 HC/RGC pairing does NOT establish AII pairing；Protocol-family overlap only。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`NO_CLEAN_MAPPING` → No identified AII state。Aggregate local/transient AC latent is not the rod AII circuit。候选 head：`None justified for current model`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`FUNCTIONAL_ASSAY_ONLY`；当前：NO for the full multilevel hypothesis。激活条件：Published physiology context;not genericAC voltage training。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：fig3-data1-v2.xlsx,fig4-data1-v2.xlsx,fig5-data1-v2.xlsx;Fig2MAT examples are rod/RGC,not rawHC。接口：https://elifesciences.org/articles/38281/figures (source links actually inspected)。
[来源1](https://elifesciences.org/articles/38281) · [来源2](https://elifesciences.org/articles/38281/figures)

### DA00_BC

**Dacey 2000 identified cone bipolar light responses** — Dacey et al. Center surround receptive field structure of cone bipolar cells in primate retina；DOI `10.1016/S0042-6989(00)00039-0`。

- **可得性**：`PAPER_ONLY`。Direct light-response paper located;no public raw repository or on-request promise verified。License：UNVERIFIED。
- **生物/样本**：Macaca fascicularis/nemestrina and baboon;per-cell mapping unknown；ex vivo；UNVERIFIED；type：Morphologically identified coneBC;exact subtype export unknown；cells：UNVERIFIED；animals：UNVERIFIED。
- **记录与刺激**：Intracellular light-response recordings;raw modality/units to verify from original files；Center/surround light stimuli;spots/annuli。Response sampling：UNVERIFIED；stimulus timing：UNVERIFIED。
- **刺激合同**：UNVERIFIED；units：UNVERIFIED；contrast：UNVERIFIED；adaptation：UNVERIFIED；channels：UNVERIFIED。
- **校准/元数据**：UNVERIFIED。Stimulus files：No verified files。Trials：UNVERIFIED；repeats：UNVERIFIED；cell map：UNVERIFIED。Preprocessing：UNVERIFIED。
- **跨层/独立性**：UNVERIFIED；UNVERIFIED。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`INDIRECT_FUNCTIONAL_CONSTRAINT` → BC effective components versus actual typedBC。Direct biology exists, but current aggregate components are not that cell-type membrane state。候选 head：`a*h_BC_voltage+b only after subtype/state correspondence;currently no eligible mapping established`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`FUNCTIONAL_ASSAY_ONLY`；当前：NO for the full multilevel hypothesis。激活条件：Raw plus typed-cell/geometry/timing contract needed before considering heldout BC physiology。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：Paper only;targeted public-archive search negative,not proof of absence。接口：UNVERIFIED。
[来源1](https://pubmed.ncbi.nlm.nih.gov/10837827/)

### PD02_HC

**Packer–Dacey 2002 H1 spatial receptive fields** — Packer & Dacey. Receptive field structure of H1 horizontal cells in macaque monkey retina；DOI `10.1167/2.4.1`。

- **可得性**：`PAPER_ONLY`。Author paper identified;no public paired raw stimulus/voltage archive verified。License：UNVERIFIED。
- **生物/样本**：Macaque；ex vivo；UNVERIFIED；type：H1；cells：UNVERIFIED；animals：UNVERIFIED。
- **记录与刺激**：Intracellular voltage/spatial RF measurements；Spatial receptive-field spot/annulus protocols;exact trial files absent。Response sampling：UNVERIFIED；stimulus timing：UNVERIFIED。
- **刺激合同**：UNVERIFIED；units：UNVERIFIED；contrast：UNVERIFIED；adaptation：UNVERIFIED；channels：UNVERIFIED。
- **校准/元数据**：UNVERIFIED。Stimulus files：No verified files。Trials：UNVERIFIED；repeats：UNVERIFIED；cell map：UNVERIFIED。Preprocessing：UNVERIFIED。
- **跨层/独立性**：UNVERIFIED；UNVERIFIED。Separate publication does not prove disjoint animals/cells; confirm lineage。
- **映射**：`INDIRECT_FUNCTIONAL_CONSTRAINT` → H1 spatial functional constraints。RF summary is not h1_state trace;current fixed graph geometry and actual recorded cells require registration。候选 head：`Future raw Vm: a*h1_state+b conditional;published RF summary remains functional`。
- **Nuisance/noise**：scale=Potentially needed; cannot identify latent absolute scale by itself；offset=Potentially needed for continuous baseline；filter=Only fixed documented acquisition filter; no free biological filter；noise=UNVERIFIED until trial-level repeats; do not infer noise from figure averages。
- **角色**：`FUNCTIONAL_ASSAY_ONLY`；当前：NO for the full multilevel hypothesis。激活条件：No digitized figure replacement for raw;author archival availability unknown。联系作者：YES for missing raw/calibration/identity fields。
- **实际检查**：No raw file manifest found。接口：UNVERIFIED。
[来源1](https://doi.org/10.1167/2.4.1) · [来源2](https://daceyretinalab.org/publications/landing/)


## 完整兼容性矩阵（摘要）

H=HIGH，M=MEDIUM，L=LOW，I=INCOMPATIBLE，?=UNKNOWN。对角线 `—` 是同一条目，不代表独立验证。每个非对角线单元的八维判断、理由、A/B/C共享资格及独立性限制均在兼容性 CSV。

| |RA26_HC|RA26_BC|RA26_RGC|SL21_10MIN|CH24_CONE|CH24_INNER|BA19|SA24|SH20|KA25|SR26|KR23|SH25|KI22_AC|HR21|AB22|KW21|SN04|LI21_HC|LI21_RGC|GR18_HC|GR18_RGC|GR18_AII|DA00_BC|PD02_HC|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|RA26_HC|—|M|M|L|?|?|?|?|?|?|?|?|?|L|?|L|I|I|M|?|?|?|L|?|?|
|RA26_BC|M|—|M|L|?|?|?|?|?|?|?|?|?|L|?|L|I|I|?|?|?|?|L|?|?|
|RA26_RGC|M|M|—|L|?|?|?|?|?|?|?|?|?|L|?|L|I|I|?|?|?|?|L|?|?|
|SL21_10MIN|L|L|L|—|L|L|L|L|L|L|L|?|L|L|?|L|I|I|L|L|L|L|L|?|?|
|CH24_CONE|?|?|?|L|—|M|M|M|?|?|?|?|?|L|?|L|I|I|?|?|?|?|L|?|?|
|CH24_INNER|?|?|?|L|M|—|?|?|?|?|?|?|?|L|?|L|I|I|M|M|?|?|L|?|?|
|BA19|?|?|?|L|M|?|—|M|M|L|L|?|?|L|?|L|I|I|?|?|?|?|L|?|?|
|SA24|?|?|?|L|M|?|M|—|M|L|L|?|?|L|?|L|I|I|?|?|?|?|L|?|?|
|SH20|?|?|?|L|?|?|M|M|—|L|L|?|L|L|?|L|I|I|M|M|?|?|L|?|?|
|KA25|?|?|?|L|?|?|L|L|L|—|M|?|L|L|?|L|I|I|L|L|?|?|L|?|?|
|SR26|?|?|?|L|?|?|L|L|L|M|—|?|L|L|?|L|I|I|L|L|?|?|L|?|?|
|KR23|?|?|?|?|?|?|?|?|?|?|?|—|?|L|?|L|I|I|?|?|?|?|L|?|?|
|SH25|?|?|?|L|?|?|?|?|L|L|L|?|—|L|?|L|I|I|L|L|?|?|L|?|?|
|KI22_AC|L|L|L|L|L|L|L|L|L|L|L|L|L|—|L|L|I|I|L|L|L|L|L|L|L|
|HR21|?|?|?|?|?|?|?|?|?|?|?|?|?|L|—|L|I|I|?|?|?|?|L|?|?|
|AB22|L|L|L|L|L|L|L|L|L|L|L|L|L|L|L|—|I|I|L|L|L|L|L|L|L|
|KW21|I|I|I|I|I|I|I|I|I|I|I|I|I|I|I|I|—|I|I|I|I|I|I|I|I|
|SN04|I|I|I|I|I|I|I|I|I|I|I|I|I|I|I|I|I|—|I|I|I|I|I|I|I|
|LI21_HC|M|?|?|L|?|M|?|?|M|L|L|?|L|L|?|L|I|I|—|M|?|?|L|?|?|
|LI21_RGC|?|?|?|L|?|M|?|?|M|L|L|?|L|L|?|L|I|I|M|—|?|?|L|?|?|
|GR18_HC|?|?|?|L|?|?|?|?|?|?|?|?|?|L|?|L|I|I|?|?|—|H|L|?|?|
|GR18_RGC|?|?|?|L|?|?|?|?|?|?|?|?|?|L|?|L|I|I|?|?|H|—|L|?|?|
|GR18_AII|L|L|L|L|L|L|L|L|L|L|L|L|L|L|L|L|I|I|L|L|L|L|—|L|L|
|DA00_BC|?|?|?|?|?|?|?|?|?|?|?|?|?|L|?|L|I|I|?|?|?|?|L|—|?|
|PD02_HC|?|?|?|?|?|?|?|?|?|?|?|?|?|L|?|L|I|I|?|?|?|?|L|?|—|

## 10. 搜索覆盖、排除与未决来源

首轮按七个独立来源范围核查：Raval跨层、Chen/cone、Schottdorf、Shah/subunit、marmoset/OpenRetina、人类与缺层、独立HC/BC/AC。随后沿线扩展到Chen Figure10、Baudin/Saha复用关系、Liu、Grimes、Kim、Shahidi/Krüppel及人类ERG。普通网页受阻时优先用官方XML/API、作者托管全文或小型Range目录；没有把抓取失败当作“资料不存在”。

| 来源/平台 | 实际发现与处置 |
|---|---|
| GIN/G-Node | 核查Schottdorf、Karamanlis、Sridhar、Krüppel、Abbas；区分冻结DOI与live/mirror；两个主机访问失败条目保留UNVERIFIED |
| Dryad | Chen、Shah实际file API；Baudin公开ZIP存在但内合同未过；Wang2023 human部分是形态图像 |
| CRCNS | 已检索retina与macaque；检索所得retina ret-1为mouse，其他macaque多为cortex；不列成primate retinal数据。[平台记录](https://crcns.org/news/thirteen-new-data-sets-added) |
| Figshare | Shahidi条目28924358仅初始化参数表；Wu2024有有限demo数据/代码，不能当完整raw语料。[参数表](https://plos.figshare.com/articles/dataset/Initial_values_for_5_training_runs_/28924358)、[Wu文件API](https://api.figshare.com/v2/articles/23929941) |
| Zenodo | Kim6495531为Neuron-C代码/形态；human ischemic16014189是OCT/YOLO相关代码、设计和训练资料，未见配对电生理。[Kim](https://doi.org/10.5281/zenodo.6495531)、[human listing](https://zenodo.org/records/16014189) |
| OSF/GitHub/作者仓库 | HumRet公开GitHub可读，OSF401；Rieke study-specific Raval包未找到；只读校准库不能建立本实验session对应。搜索无结果不是对所有未索引内容的排除 |
| OpenRetina | 是原始数据/处理镜像的入口；内部层名称出现在模型解释论文中，不等于记录了HC/BC/AC |
| GRO.data | Shahidi真实来源AP5BRQ；证书错误导致manifest未验证，保留UNVERIFIED，不以Figshare替代 |

补充来源筛选记录（未当作独立且合格的主数据集）：

- **Shah2022 individual variability**，完整大语料按请求提供，公开subset的实际input/response schema本轮未验证。论文的66macaques/75retinas/112recordings/21626RGC和1human donor不代表公开子集规模。[作者全文](https://med.stanford.edu/content/dam/sm/chichilnisky/documents/publications/Shah2022.pdf)
- **Wu2024**，Figshare只有129866368B的submission_code.zip；有限spike-sorted例子与prefitted models，完整>5TB电压不公开。这里只把它作为辅助RGC来源筛选，不视作HC/BC资料。[数据声明](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11390888/fullTextXML)、[文件API](https://api.figshare.com/v2/articles/23929941)
- **Freeman2015 subunits** 的single-cone resolution指刺激定位/从RGC反推的subunit，不是直接cone/BC记录；Gogliettino2024的primate-cnn-model仓库为代码，raw可得性UNVERIFIED。[Freeman](https://elifesciences.org/articles/05241)、[Gogliettino code](https://github.com/Chichilnisky-Lab/primate-cnn-model)
- **Angueyra2022** 有cone+HC，未核实raw存档；Chen2024沿用其primate consensus参数，不能作该参数的独立验证。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8883858/)
- **Dunn2007**（10.1038/nature06150）是额外cone/HC/BC/parasol记录线索；本轮仅核实论文/书目信息，raw availability仍UNVERIFIED，不纳入任何可执行MVP。[PubMed](https://pubmed.ncbi.nlm.nih.gov/17851533/)
- **Greschner2014 PAC+RGC** 的同时MEA是有价值的真实跨细胞类记录，但raw+stimulus存档未核实；Kim2022已作为更明确的AC数据申请条目。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3942577/)
- **Percival2022 AII** 为glutamate-puff电流和另外的RGC光反应，data on request；不能当light-driven AII raw监督。[作者全文](https://escholarship.org/content/qt25v77650/qt25v77650_noSplash_c74ac96f4d496e2c8cc0a0322c5053c4.pdf)
- **Soto2020 human RGC** 明确未公开deposit、需申请；**Wang2023 human ON-DSGC** 的人类部分仅形态，macaque部分有calcium资料；**Cowan2020**公开atlas不能替代其成人RGC生理配对文件。[Soto](https://pmc.ncbi.nlm.nih.gov/articles/PMC7442669/)、[Wang Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.47d7wm3kx)、[Cowan](https://pmc.ncbi.nlm.nih.gov/articles/PMC7505495/)
- **Grabner2023** 的cone–BC突触记录为ground squirrel，**Kuo/Rieke2025–26** BC synapse资料为mouse；未因引用macaque而纳入primate表。[Grabner](https://www.nature.com/articles/s41467-023-38943-2)、[Kuo/Rieke](https://elifesciences.org/articles/98817)

尚未闭合但已明确归类的门槛：BA19内部stimulus/response字段、SR26原始版本和四piece镜像的cell桥接、KR23与SH25原始目录、Raval和其他申请式数据的真实可提供范围。它们不是被默认为已可用的材料。本次停止在dataset map，未启动数据申请、补实验或实施。

## 11. 交付与验证边界

- [Dataset CSV](D:/PythonProject/retina_rf_SNN/data/research/primate_multilevel_retina_dataset_map.csv)：每条目完整contract、observation nuisance/noise、角色、文件证据与URL。
- [Compatibility CSV](D:/PythonProject/retina_rf_SNN/data/research/primate_multilevel_retina_compatibility.csv)：每个无序pair的八维状态/逐维理由、总评、A/B/C共享资格及独立性门槛；对称恢复完整matrix。
- [Machine-readable JSON](D:/PythonProject/retina_rf_SNN/data/research/primate_multilevel_retina_dataset_map.json)：同一份contract、定义和总判断。

文件检查涵盖：必填字段、枚举、唯一ID、所有pairs无遗漏/重复、CSV↔JSON逐字段一致、未以空字符串/零代替未知。所有RAW_VERIFIED条目有明确文件清单与刺激证据。未对response数值、刺激重放、时序同步或科学hypothesis作验证。

本轮读取的8个输入/模型/observation源码文件以SHA256做前后核对；这只覆盖实际读取文件，不冒充全仓库或全部checkpoint审计。保留已有未提交文件。本任务新增的仓库交付限定为上述四个文件；无训练、拟合、参数更新、architecture/loss/adapter实现或新生理assay。
