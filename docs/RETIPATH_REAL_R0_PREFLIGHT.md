# Population RetiPath real-data Phase R0：Schottdorf–Lee 前向接入

日期：2026-09-20。状态：**VERIFIED_ENGINEERING_PREFLIGHT_WITH_DECLARED_SCIENTIFIC_BLOCKERS**。

已建立第一条真实数据执行路径：现有 macaque L+M movie pipeline → 显式物理像素边界/时间合同 → 原 Q → 未修改的 Population v0.1 全 forward → 单条 recorded RGC observation port。**22 cells、37 recordings、137 个既有生物 trial 的全部 2192 个训练序列已完成前向，所有输入、中间变量和输出有限。** 参数不变；没有 loss、backward、optimizer、训练或 checkpoint 读写。

工程接入通过不等于真实训练已获批准或生理标定完成。9 个 MC cells 与当前 parasol-related family 相容；13 个 PC cells 只完成同极性端口的结构性前向，**没有实现 midget family，也没有把 PC 改标为 MC**。既有 750/751 帧零点争议和真实初态未知继续保留，未通过本轮结果选取新 alignment。

依据：[AGENTS](../AGENTS.md)、[Population v0.1](RETIPATH_POPULATION_ARCHITECTURE_V0_1.md)、[真实接入合同 v0](RETIPATH_REAL_DATA_INTEGRATION_V0.md)、[NEXT_TASK](NEXT_TASK.md)。此前文档阶段的“不运行”边界由本轮明确授权的真实适配和前向检查取代；训练、改架构、改 prior、重新解释数据仍未授权。

## 1. 新增内容与可复核工件

| 文件 | 内容 |
|---|---|
| [real_data.py](../experiments/retipath_population_v0_1/real_data.py) | 复用既有训练数据读取/分段；物理像素边界适配；recording→独立实例/极性端口；单通道概率选择 |
| [run_real_r0_preflight.py](../experiments/retipath_population_v0_1/run_real_r0_preflight.py) | 固定数据来源与访问窗口、逐 recording 全 forward、因果/reset/覆盖/参数不变断言和机器可读报告；不是 trainer |
| 本文 | R0 合同、逐 cell 结果、边界及停止记录 |
| [manifest.json](../output/real_data/retipath_population_r0_preflight/manifest.json) | 输入资产、来源结果清单、相关代码/规则/合同 SHA256；资产与原22-cell来源哈希匹配 |
| [dataset_contract.json](../output/real_data/retipath_population_r0_preflight/dataset_contract.json) | 首次打开数据 payload 前锁定的物理输入、时间/历史、实例映射、检查范围及访问合同 |
| [per_cell_preflight.csv](../output/real_data/retipath_population_r0_preflight/per_cell_preflight.csv) | 22行逐 cell 类型、recordings、shape、FOV、coverage、warmup、端口与阻塞项 |
| [verification.json](../output/real_data/retipath_population_r0_preflight/verification.json) | 37条 recording 的实际 shape、source IDs、输入/target/probability哈希、原始核验数值、来源与参数不变证据 |

入口为 `D:/anaconda/python.exe -B -m experiments.retipath_population_v0_1.run_real_r0_preflight`，从项目根目录执行。当前工件已存在，入口会拒绝覆盖；本轮不再次执行。未修改旧 loader、Canonical/Population circuit、Q、数据、checkpoint、历史结果、设计稿或 NEXT_TASK。

## 2. 数据来源及访问范围

复用 [既有22-cell/37-recording来源清单](../output/real_data/schottdorf_lee_2021_22cell_canonical_v1_revision4_fresh_20260829/results.json) 的 cohort、type、recording IDs、source hashes、adapter config 与 train/validation 边界。该清单也被 [shared-BC canonical real-data run](../output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/run.py) 用来锁定输入，未运行旧训练入口。

实际资产仍为 `data/real/schottdorf_lee_2021_repository/` 的37条 MC/PC记录及 `data/real/schottdorf_lee_2021_macaque/1x10_256.mpg`。Movie SHA256：`328e229e160eab35028f063468cc2e167045fc8c9fcdbe4a8e7823142a752a67`。Movie、README、两个 cell-list 文档、annex pointer 和37条 spike文件全部与原来源清单匹配；annex pointer不是实际movie。未下载、替换或建立新数据资产。

只解码校准前段及既有训练 live frames `[0,2400)`，即 `[0,16)`秒；不解码作为 validation 的 live `[2400,3000)`。没有构造 validation/test split tensor，没有对其 forward、评分、选参数或选时间偏移。原结果JSON中的数据清单/分段元数据被读取，未用于预测质量选择。

**访问边界须准确表述：**复用的 `parse_recording_spike_trials` 会读取完整原始文本，并解析其中窗口外 timestamps，之后仅把训练 `[0,16)`事件分箱。本轮不能声称“文件层面完全没有碰到窗口外时间戳”；可以确认没有构造、评价或利用 validation/test target 张量调整任何设置。完整文件哈希也读取文件字节。这是已声明的数据适配行为，不是新的 test 评价。

来源函数为 [schottdorf_lee_2021.py](../data/schottdorf_lee_2021.py) 的 `_load_calibrated_lm_drive`，以及 [multirecording](../data/schottdorf_lee_multirecording.py) 的 `_bin_trial` / `_make_trial_split`；spike parser与catalog沿用现有实现。没有调用会顺带构造validation tensor的整套cell loader。原 `lSS01184` 的6×1min override、重复payload处理、六个重复列与第七 maintained-activity列的排除均保持。

## 3. Real-data stimulus adapter 合同

### 3.1 空间与 Q

| 项目 | 实际使用 |
|---|---|
| Native source grid / FOV | 256×256；4.6×4.6 deg |
| Native degree/pixel | 4.6/256 = **0.01796875 deg** |
| 既有 crop/pool | 中央51×51 native pixels；每3×3 block按L+M值取平均，输出17×17；没有resize |
| 适配像素宽度 | 3×4.6/256 = **0.05390625 deg** |
| Crop physical FOV | 51×4.6/256 = **0.91640625 deg** 每轴 |
| 输入边界 | 每轴 **[−0.458203125,+0.458203125] deg**，显式 `[289,2,2]` pixel bounds和289个pixel areas |
| 空间方向 | 继承旧loader：row-major flatten；x随列增加，y随行减小；使用已有single-cell origin，不构造跨cell几何 |
| Q | 原面积交叠算子，除以完整0.10×0.10 deg aperture面积；25个Q节点；不裁剪后重归一化 |
| 模型支持 | 固定aperture包围范围[−0.35,+0.35]² deg；H1反馈、direct及broad AC等完整结构祖先支持都包含在本输入场内 |
| Coverage结论 | **FULL_TEMPLATE_ANCESTOR_SUPPORT**；只证明当前固定模板的输入域完整，不证明该尺寸涵盖真实cell的全部生理RF |

17×17是输入采样网格，绝不当成25个Q/H1节点或50个BC节点。没有把原crop重标为synthetic的2×2 deg域，也没有从pool后的像素恢复丢失细节。原单细胞原点和取整crop约定原样沿用，未进行新的RF中心配准；生理空间定位仍依赖旧记录合同。

Shared pixel edges按 `(edge_index*3 − 51/2)*4.6/256` 构建，保持已有physical FOV/crop/pool；中心位置与旧float32 center数组在其存储精度内核对。这样相邻像素共用同一数值边界，满足冻结Q的精确Cartesian tessellation检查。

### 3.2 L/M、background 与 Weber

原预处理先将BGR转换RGB，沿用原 `/256` gamma校正及权重：

| 通道 | offset + scale × (code/256)^gamma |
|---|---|
| R | 0.01451 + 0.9855 × (...)^2.3122 |
| G | 0.005123 + 0.9949 × (...)^2.2752 |
| B | 0.02612 + 0.9739 × (...)^2.2818 |

`L=2.74R+3.4G+1.34B`；`M=1.21×(1.06R+3.58G+2.07B)`。L+M经原3×3 mean pooling，再用原loader decoded `[0,751)` 的逐pixel均值作background；输入是 `(live_LM-background_LM)/background_LM`。这些运算继续使用原float32预处理；适配后仅转为Population默认float64作前向。

单位是无量纲relative L+M Weber drive。没有猜测absolute R*/cone/s，没有重新估计background、拆出新L/M机制或根据RGC响应修改calibration。旧loader把 `[0,751)` 当校准前段的行为被完整保留；帧边界争议不在R0重新解释。

### 3.3 时间、分箱、warmup 与 reset

| 项目 | 冻结合同行为 |
|---|---|
| Movie clock | 150 Hz；dt=1000/150 ms；native frame不做temporal重采样 |
| Alignment | live frame f对应zero-based decoded f+751，复用既有合同；不选择750替代 |
| Spike time | 校正后的live-relative整数ticks×0.1ms；不重复减Video Start(s) |
| Spike binning | `floor(time_ms*150/1000)`；保存counts，`count>0`形成Bernoulli occupancy；不把多spike计数直接当概率target |
| 训练段 | 16段×150 bins；每段1秒，来源frame offset与trial ID逐条写入verification |
| Population time tensor | 段内midpoints `(bin+0.5)*1000/150`；原global frame offset单独保存，不因reset丢失来源身份 |
| Warmup | 每段前30 bins =200ms：真实刺激/过去spikes参与递推，target valid mask为false；其余120 bins保持既有mask |
| External pre-roll | **0 bins**；未prepend校准blank，未跨段或跨repeat拼接新prehistory |
| Reset | 每个1秒sequence独立baseline reset；history/适应/上游state从既有默认值起，V=2/9；不隐式carry |
| Repeat handling | 原parser的六个live-relative repeats，共用10-min movie第一分钟；不凭一个Video Starts标量拼连续acquisition时间线 |

Reset和warmup的**实现合同已验证**；200ms warmup不保证IIR已忘记真实前史，baseline也不是测量到的真实膜初态。该限制逐cell登记，不延长warmup或改变split来修正。751相对于真实acquisition t=0的外部确认仍为 [UNRESOLVED_EXTERNAL_EVIDENCE_REQUIRED](../.omo/evidence/schottdorf_lee_frame_zero_resolution.md)；本轮只验证按旧合同执行时的tensor/time一致性。

## 4. RGC instance、类型和内部family

| Recorded type | Cell / recording数 | Population输出 | 当前适用范围 |
|---|---|---|---|
| MC ON | 5 / 8 | `probability[...,0]`、对应ON logit | 当前parasol-related family的spike候选；仍受全局alignment/reset限制 |
| MC OFF | 4 / 8 | `probability[...,1]`、对应OFF logit | 同上，OFF |
| PC ON | 9 / 15 | ON index0，仅工程polarity selector | **PC_MIDGET_FAMILY_NOT_IMPLEMENTED**；不得当作已实现midget或chromatic opponent prediction |
| PC OFF | 4 / 6 | OFF index1，仅工程polarity selector | 同上；未加PC专属机制、prior或decoder |

PC数据没有从报告中删除，也没有因不能生理匹配而阻塞其他可执行检查。PC的“forward finite”表示相同物理输入与有符号history可流经现有图，**不构成PC真实训练端口已准入的结论**。没有把旧Canonical的midget实例能力自动赋予Population v0.1。

每条recording独立建立 `SL21::cell_id::recording_id` 实例，共37个模型；所有模型保持相同constructor默认数值，但参数对象互不共享。即便catalog确认两条recording来自同一个cell，本R0也只在汇总表合并其行数，不拼成同一块真实retina、不跨记录carry。只共享architecture/type-level mapping规则。Backbone接收物理刺激和history，不读取dataset/session ID选择机制。

所有实例都保留25 H1、25 ON BC+25 OFF BC、local/broad×ON/OFF AC各9、两RGC输出×两effective modes及全部既有分支。被记录极性的direct BC分支和既有AC routing保持；另一RGC输出结构存在但没有该记录的真实target。H1 `h_H/feedback`、BC `s_B/delta_r_B`、AC `a_A/delta_o_A`、`d_E/d_I/gE/gI/V`全部只是模型内部变量，**本阶段没有任何内部层级真实监督**。

Recorded target为 `[B,T,1]`，送入history时只写对应极性列；另一列零值是未观测计算占位，不是“该细胞没有spike”的负标签。只有selected probability `[B,T,1]`可与recorded target配对。改变未观测列的history后selected output不变，已逐recording验证；不会把不同recorded cells填入同一模型的ON/OFF两列。

## 5. 前向验收的实际数值

全部2192序列、328800个sequence-time bins完成完整前向；其中65760个warmup bins，263040个既有valid target bins。137个trial包括17条10-min recording的单trial及20条6×1min的六次重复，**不是137个独立retina**。所有实际前向仅使用每trial最初16秒。

| 检查 | 结果 / 原始数值 |
|---|---|
| 源资产与既有实验相同 | Movie、37个spike文件和catalog来源哈希全部匹配；前后相关源码/资产哈希不变 |
| Shapes / source identity | 全37条通过；每段150×289 → Q150×25 → probability150×2；selected probability与target均150×1；source IDs/trial order/聚合train counts与原manifest一致 |
| 全forward finite | inputs、states、outputs、initial states均无NaN/Inf；gE/gI均正；概率范围 **0.002472455230979286–0.8236176503187591**，仅作数值检查，不是预测质量 |
| Physical Q | 每行积分权重和相对1的最大误差 **5.551115123125783×10⁻¹⁶**；独立NumPy面积积分与Q输出最大差 **3.552713678800501×10⁻¹⁵** |
| Physical ancestor coverage | 各port的完整结构祖先并集覆盖全部25个Q aperture；都在0.91640625 deg输入场内；没有删路径或填未知外围 |
| Future stimulus causality | 修改bin76以后输入，所有动态端口bin0–75的最大差 **0** |
| No current/future y leakage | 翻转recorded events从bin75开始，logit bin0–75最大差 **0**；之后确实受strictly-past history影响，最大差 **0.9999880327084436** |
| 未观测RGC history隔离 | 改另一列history，recorded probability最大差 **0** |
| Reset replay | 经其他forward后再次执行同sequence，最大logit差 **0**；各initial state匹配baseline；bin0 history=0；carry请求被拒绝 |
| Truncated-prefix agreement | 完整序列与只输入其前76 bins的共同输出最大差 **8.881784197001252×10⁻¹⁶** |
| Batch isolation | 同一sequence单独执行与置于batch的最大差 **2.220446049250313×10⁻¹⁵** |
| 参数与自由度 | 37个实例state_dict前后哈希相同；默认初值也相同但对象不共享；所有参数requires_grad=false且grad=None；未调用hierarchy loss或更新参数 |

因果/reset检查固定使用每条recording前至多4个训练sequence，cut=75；不是根据响应选择窗口。全forward/finite/shape检查覆盖所有16段及全部保留trials。正确性容差事前固定：因果/reset误差1e−10，Q检查2e−6；不是新的科学成功阈值。未来后缀翻转只是在内存中验证同批真实输入的因果性，不生成或保存新的synthetic dataset，不进行预测对比实验。

## 6. 每个cell的preflight

下表维度是该cell各**独立recording**的数量汇总，不表示把其trial接成一个连续sequence或多cell retina。所有行共同：FOV=0.91640625×0.91640625 deg，coverage=`FULL_TEMPLATE_ANCESTOR_SUPPORT`；每sequence150 bins、warmup30、external pre-roll0、Q末轴25；target与selected probability形状相同。唯一training时间内容为2400 bins，重复trial增加样本数而不增加unique stimulus长度。

阻塞码：**T**=绝对帧零点外部确认未闭合；**R**=真实初态未知，沿用baseline reset/warmup不能当精确生理前史；**P**=当前无midget family。T/R不阻止本轮按旧合同进行工程前向，但不能据此宣布生理合同READY。逐行完整字段也在CSV中提供。

| Cell | Type | Recordings | Input shape | Target / selected probability shape | RGC port | Finite / causality-reset | 保留项 |
|---|---|---|---|---|---|---|---|
| 67#4 | PC OFF | lSS01071 | [16,150,289] | [16,150,1] | OFF / 1 | VERIFIED | T/R/P |
| 67#6 | MC OFF | lSS01078;lSS01079 | [112,150,289] | [112,150,1] | OFF / 1 | VERIFIED | T/R |
| 67#7 | MC ON | lSS01086;lSS01087 | [112,150,289] | [112,150,1] | ON / 0 | VERIFIED | T/R |
| 67#14 | PC ON | lSS01110;lSS01112 | [112,150,289] | [112,150,1] | ON / 0 | VERIFIED | T/R/P |
| 67#21 | PC ON | lSS01130;lSS01131 | [112,150,289] | [112,150,1] | ON / 0 | VERIFIED | T/R/P |
| 67#26 | PC ON | lSS01141;lSS01142 | [112,150,289] | [112,150,1] | ON / 0 | VERIFIED | T/R/P |
| 67#33 | MC OFF | lSS01159;lSS01160 | [112,150,289] | [112,150,1] | OFF / 1 | VERIFIED | T/R |
| 67#34 | PC ON | lSS01167;lSS01168 | [112,150,289] | [112,150,1] | ON / 0 | VERIFIED | T/R/P |
| 68#3 | MC OFF | lSS01181;lSS01183 | [112,150,289] | [112,150,1] | OFF / 1 | VERIFIED | T/R |
| 68#4 | PC ON | lSS01184 | [96,150,289] | [96,150,1] | ON / 0 | VERIFIED | T/R/P |
| 68#7 | PC ON | lSS01194;lSS01196 | [112,150,289] | [112,150,1] | ON / 0 | VERIFIED | T/R/P |
| 68#10 | MC ON | lSS01221 | [16,150,289] | [16,150,1] | ON / 0 | VERIFIED | T/R |
| 68#11 | PC OFF | lSS01225;lSS01227 | [112,150,289] | [112,150,1] | OFF / 1 | VERIFIED | T/R/P |
| 69#3 | PC OFF | lSS01251;lSS01252 | [112,150,289] | [112,150,1] | OFF / 1 | VERIFIED | T/R/P |
| 69#4 | MC ON | lSS01254 | [96,150,289] | [96,150,1] | ON / 0 | VERIFIED | T/R |
| 69#6 | MC OFF | lSS01256;lSS01257 | [112,150,289] | [112,150,1] | OFF / 1 | VERIFIED | T/R |
| 69#7 | MC ON | lSS01258;lSS01259 | [112,150,289] | [112,150,1] | ON / 0 | VERIFIED | T/R |
| 69#21 | PC OFF | lSS01270 | [96,150,289] | [96,150,1] | OFF / 1 | VERIFIED | T/R/P |
| 70#1 | PC ON | lSS01278 | [96,150,289] | [96,150,1] | ON / 0 | VERIFIED | T/R/P |
| 70#7 | PC ON | lSS01284;lSS01285 | [112,150,289] | [112,150,1] | ON / 0 | VERIFIED | T/R/P |
| 70#15 | PC ON | lSS01287 | [96,150,289] | [96,150,1] | ON / 0 | VERIFIED | T/R/P |
| 70#34 | MC ON | lSS01299;lSS01300 | [112,150,289] | [112,150,1] | ON / 0 | VERIFIED | T/R |

## 7. 首次适配错误与修复留痕

第一次执行在`Stimulus.validate → validate_pixel_bounds`处拒绝输入，**尚未进行第一次模型forward**。初稿以旧float32像素中心独立加减float64半宽生成边界，产生最大 **1.788139347702611×10⁻⁸ deg** 的相邻边界误差；冻结validator要求边界精确相同，因此正确地拒绝。

只修复新adapter的边界构造：从同一native FOV/crop/pool得到共享edges，并核对旧centers。未放宽validator、修改Q、改变输入值、改degree/pixel、换frame或调整模型参数。随后重跑相同事前固定的R0窗口，得到本报告结果。不是训练失败后挑选新结果。

失败时的合同、manifest、两份代码快照及 [failure.json](../output/real_data/retipath_population_r0_preflight_attempt01/failure.json) 保存在独立 `retipath_population_r0_preflight_attempt01/`，没有覆盖或删除。第一次已读取训练movie prefix与lSS01071原文件，未构造validation/test tensor；这些访问已在失败记录中说明。

## 8. 完成与停止边界

R0已经验证真实输入能经物理Q和完整Population图到达recorded spike端口。没有评估拟合优度、泛化、内部状态恢复、parameter identifiability或生理干预；finite和causality通过不替代这些结论。

没有实现H1/BC/AC真实loss、E/I current loss、high-capacity adapter、trainer、optimizer或新prior；没有新机制、synthetic实验、下载、Git操作或旧checkpoint修改。未因PC阻塞增加midget family，也未重新审计/选择750或751。下一阶段真实训练未开始；本轮完成后停止。
