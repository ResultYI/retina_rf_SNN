# 当前单任务：RetiPath G3 synthetic robustness benchmark

状态：**G3 五个 world × 五条件 × 三 seeds 已完成，必要核验 VERIFIED；完成后停止，不自动启动后续任务。**

2026-09-18 交付：[RETIPATH_G3_ROBUSTNESS_RESULTS.md](D:/PythonProject/retina_rf_SNN/docs/RETIPATH_G3_ROBUSTNESS_RESULTS.md)。完整协议、CSV/JSON、75 个 final checkpoints 及核验工件位于 `D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g3_robustness_20260918/`。

## 本轮完成范围

用户明确授权 G3 synthetic robustness；并确认五个新 world、teacher raw-space 标准差 0.5 的事前生成规则和 G2 原三个配对 student 初值。G1/G2 的架构、五条件、scale split、loss、180-step 日程、primary metric 与 RF evaluator 保持不变。

- 在目标生成前冻结所有 teacher 参数、独立刺激/noise/event 随机流、manifest 和日程。
- 每 world 完整运行 A–E × 3 seeds，总计 75 个 fit、13,500 updates、36,000 microbatches；无失败重试或结果后重跑。
- 全部 checkpoint 锁定后执行各 world 一次最终评价，保留集已消费；数值 replay 仅为同批结果核验。
- 两个 primary 比较 C−B、D−C 按 world 单独报告并统计方向；同时保存 D/E 的 prediction、latent、direct-BC recovery，以及 RF secondary。完整结果见报告。
- 源码/数据/日程/optimizer/checkpoint 核对、逐 world 重放、独立 NumPy 指标复算及汇总检查全部通过。

## 停止边界

本轮未调参、未改架构/loss/budget/scale、未增加 condition、未修改 G2 或旧实验；未运行 Git、切换分支或训练真实数据。没有把 teacher/world 当作生物 population 做显著性推断，也没有作研究决策或追加后续任务。

下方原样保留 G2/G1/G0 任务，仅作历史。

<details>
<summary>G2 原任务原文及 G1/G0 历史</summary>

# 当前单任务：RetiPath G2 synthetic multilevel/multiscale benchmark

状态：**G2 五条件 × 三 seeds 已全部完成，必要核验 VERIFIED；完成后停止，不自动启动后续阶段。**

2026-09-18 交付：[RETIPATH_G2_MULTISCALE_RESULTS.md](D:/PythonProject/retina_rf_SNN/docs/RETIPATH_G2_MULTISCALE_RESULTS.md)。完整工件位于 `D:/PythonProject/retina_rf_SNN/output/experiments/retipath_g2_multiscale_20260918/`。

## 本轮完成范围

用户明确授权 G2 数据生成、五条件固定预算训练、checkpoint 与评价；本轮授权覆盖 G0/G1 当时的阶段限制，其他数据保护、冻结模型、审批及发布规则保持有效。

- 冻结 G1 架构与 loss，使用同一 realizable teacher、配对 student seeds/初值及事前冻结的 physical-scale 合同。
- A/B/C/D/E 各完成 3 个 fit，每 fit 180 updates、480 microbatches；共 15 份最终 checkpoint、2,700 updates、7,200 microbatches。
- 完成 held-out scales 的 direct-BC logit intervention primary，以及 RGC/excess CE、H1/BC recovery、跨 seed ambiguity、未见 layer×scale 组合和 common-probe RF secondary。
- 完成 source/data/schedule/optimizer 核对、checkpoint replay、CSV/JSON 复算与独立 NumPy 指标核验。一次训练前 Windows 路径读取修复已完整记录，没有重新生成数据或改变合同。
- 最终 checkpoint 全部锁定后登记 test consumption；test 已消费，后续 replay 只作本轮核验。没有根据结果调整 loss、architecture、condition、steps 或 scale split。

## 停止边界

本轮没有研究决策或下一阶段待办。未改正式模型、旧实验或 G1；未运行 Git、切换分支、训练真实数据、重跑 S0/S0.5 或追加 S0.6。结果与限制见交付报告；任何后续工作仍须用户明确指令。

下方保留 G1/G0 原任务，仅作历史，不重新激活旧待办或当时的阶段限制。

<details>
<summary>G1 原任务原文及 G0 历史</summary>

# 当前单任务：RetiPath G1 isolated prototype

状态：**G1 原型已实现，必要验收已通过；完成后停止，等待用户后续指令。**

2026-09-18 交付：[RETIPATH_G1_PROTOTYPE_IMPLEMENTATION.md](RETIPATH_G1_PROTOTYPE_IMPLEMENTATION.md)。设计依据：[RETIPATH_ARCHITECTURE_V0.md](RETIPATH_ARCHITECTURE_V0.md)。研究设计保持不变。

## 本轮授权与完成范围

用户明确授权在 `experiments/retipath_multiscale_v0/` 实现 G1 隔离原型并运行必要验收，覆盖 G0 文档保留的当时“仅文档/禁测试”阶段限制；其他数据、冻结模型、发布及审批边界不变。

- 已实现 physical-coordinate 输入/数值网格解耦、state/output/observation 分离、local BC 与 H1/direct/AC/EI/RGC 分支。
- 已实现保持既有语义的 `BLOCK_DIRECT_BC_DRIVE`，以及 RGC-only/joint/progressive 参数组接口。
- 5 组单元验收覆盖跨网格一致性、真实尺度变化、粗 pooling 信息丢失、state/output、干预、因果性、梯度路由和 ID 无关机制选择，最终全部通过。
- 当前接口按独立序列 reset；流式 carry/resume、正式数据生成与训练运行没有实现。

## 停止边界

本轮未训练、未生成正式 synthetic dataset、未改研究设计或正式模型、未增加机制或扩展实验，未读取/保存 checkpoint。没有自动创建 G2 任务；G2 的运行合同、生成器、数据生成、A/B/C 训练与评价仍须另行明确授权。

下方保留 G0 原任务，仅为历史记录，不将其中旧待办或阶段限制重新作为当前任务。

<details>
<summary>G0 原任务原文（历史）</summary>

# 当前单任务：RetiPath 架构与训练设计 v0

状态：**设计稿已完成，待用户批准实施**。任务类型：设计与文档。完成条件：产生可直接指导后续隔离原型的具体规格。

2026-09-18 交付：[RETIPATH_ARCHITECTURE_V0.md](RETIPATH_ARCHITECTURE_V0.md)。本轮仅完成文档及必要文本核对；下述任务范围原样保留。未实现或运行候选模型，未测试、训练或生成数据，不自动开始下一阶段。

## 目标

将 `docs/RESEARCH_PLAN.md` 落成一套最小、可实现的设计：物理坐标刺激→具有局部单元、分支与反馈的动态回路→分层观测；比较 RGC-only、joint-from-scratch 与逐层训练/回放；主要测试跨尺度和未见层级×尺度组合。

本轮完成“方案是什么、如何计算、如何训练和如何验收”。不得以等待真实数据、缺少新文献或继续旧 S0.5 分解替代设计交付。

## 授权与禁止

允许阅读现有相关源码、已有实验报告和配置；新增/编辑本轮指定的 Markdown 文档；用文本检查核对路径、表格与定义。

不运行任何模型、checkpoint、训练、数值试验、pytest 或数据生成；不读取 natural-movie/spike payload；不安装依赖、执行 Git 命令、切分支、推送、修改正式模型/loader/trainer 或旧实验文件；不联系作者或启动数据申请。

已知资料足够完成设计。只为会改变当前设计的关键方法缺项查原始来源；不要新增文献综述、dataset map 或审计报告。无需每个 synthetic 数值都来自生理文献：选择小型工程默认值，明确标记为设计假设并固定其理由。

## 最小阅读范围

先读 AGENTS.md 与 RESEARCH_PLAN.md；再读实际的 canonical forward、H1/BC/AC/EI state、observation/intervention 接口、输入坐标构造及 S0/S0.5 的结论段。模块具体路径从本地查找；不要假定远端旧路径等于最新实现。

当前 repo 中的 `DESIGN.md` 可能是作图规范，应先确认用途，不覆盖。旧报告的图表和原始结果无需重算。

## 唯一主要交付

生成 `docs/RETIPATH_ARCHITECTURE_V0.md`。一份文档内完成下面六项；不要拆成连续的 S0.x audit。

### 1. 一张计算图和一张逐层定义表

保留 H1 feedback、BC direct 与 BC→AC-associated 分支、E/I integration、RGC history/readout。

表中逐层列：input / state / output / observation、更新方程或足够明确的伪代码、单位/有效尺度、初始化、局部连接、共享与专有参数、当前已有实现/新设计候选。

H1 state 与 feedback 分开；BC input、state、output 分开但只引入必要状态。不能仅改变量名就称 voltage/release。优先保留现有后端；仅将完成局部多尺度计算所需的最小改动列入候选。

### 2. 明确的张量与物理输入接口

至少规定：
- `StimulusBatch`：values `[B,T,P,C]`、pixel bounds、空间单位、time axis/dt、mask、背景与校准状态。
- `CircuitGeometry`：各层节点位置、局部连接及跨网格积分规则。
- `StateBundle`：各层 state/output 分轴保存，区分像素P、节点N、分支/K轴与通道C。
- `ObservationBatch`：observed layer/variable、位置/selector、target、时间与有效mask、单位/噪声定义。
- `InterventionSpec`：作用端口、替换规则和下游重算语义，复用现有合同。

定义同一物理刺激换网格时的处理、真实尺寸改变时的处理和域外输入；不要写一个无约束 resize adapter。第一版只选择一个已知 synthetic 坐标单位和有限网格设计。

### 3. 参数与可更新范围

列出固定结构/geometry、共享机制、潜在细胞/实验 nuisance。第一版 synthetic 使用已知观测映射，identity heads，暂不做跨物种层级模型。

给出 HC、BC、RGC loss 到参数模块的可更新矩阵，由计算依赖确定。明确 state loss 不直接触及哪些 coupling。不要新增 overall gain 的精确冗余，不以任意 dataset embedding 生成独立 RF。

### 4. 完整的 A/B/C 训练日程

A=RGC-only；B=全部观测 joint-from-scratch；C=逐层预训练→新增模块拟合→回放与联合校正。

写出三者 optimizer/macro-step 伪代码、阶段切换、哪些参数冻结/解冻、每阶段各 dataset 的 batch 次数、loss/likelihood/归约/权重和总观测曝光量。B/C 使用相同数据和配对初值。

给出一个具体的小预算默认计划，明确只是待批准的运行规格。更新次数、数据曝光和计算量分别列出；不得留成“自动选合适预算”。交替抽样与梯度累积选一种并解释目标，不把它们当科学优劣结论。

### 5. 一轮小型 synthetic 验证规格

提出一个小 teacher 回路，包含 HC/局部BC/RGC 观测，保留 AC-associated 中介但不加 AC 标签。初版可 realizable，明确 teacher/student 共享已知条件和禁止复制的学习参数。

写出有限的尺度×观测层矩阵：共同锚点、各层训练尺度、未见尺度及未见组合；数据量、噪声、sequence 初态与时间谱均显式给出。禁止使用当前RGC结果选择刺激或让相同物理刺激跨划分泄漏。

指定一个主要通路目标及选择理由，分别测局部与RGC干预误差；同时报告未见尺度响应、excess CE和跨seed分散。不要继承总CE的1%作为新模型误差保持依据，不汇总成全通路总分。

先记录何种结果支持/限制该设想；若无法为某量合理指定二元阈值，预定连续指标与解释规则，不造PASS标准。

### 6. 实现落点与停止点

只列后续原型需复用/新增的最少文件，以及不超过五项真正必要的单元验收：坐标/积分、state更新、观测梯度路由、causal history、训练公平性或数据划分。现在不创建代码、不运行检查。

在给出一个推荐设计后，最多列三项需要用户决定的实质问题。一般工程常数用明确的 synthetic 默认值，不要把常规选择全部回抛给用户。任何未核实的生理事实保持 UNVERIFIED，不影响无生理宣称的 synthetic 设计。

## 输出要求与完成

给出具体方程、维度、参数路由、三条件日程、尺度划分和有限验收；不以概念综述结束。采用短段落、紧凑表格和少量伪代码，不生成图片或PPT。

交付后将本文件状态更新为“设计稿已完成，待用户批准实施”，保留原任务范围。最终只报告设计文件、主要选择、最多三项决策、未运行事项。完成后停止；不要自动创建 trainer、跑新S0或进入下一阶段。

</details>

</details>

</details>
