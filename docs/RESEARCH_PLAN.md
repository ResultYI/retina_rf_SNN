# RetiPath 研究目标与阶段计划 v0

文档修订：2026-09-18。已有实验依据截至用户报告的 2026-09-17。
状态：研究方向已修订；G0 设计稿已完成，待用户批准实施。见 [RETIPATH_ARCHITECTURE_V0.md](RETIPATH_ARCHITECTURE_V0.md)；本轮仅授权文档落地与设计稿，不授权实现或训练。

## 1. 核心目标

建立一个动态视网膜回路模型：在显式物理坐标和实验条件下，以不同层级、不同刺激尺度的部分观测约束回路计算；预测未见尺度、历史情境、观测组合和通路干预；保留无法由现有观测确定的机制歧义。

研究对象始终是同一个有分支与反馈的回路。不同数据集在不同位置观测这个系统。后段观测依赖前段计算，并可与已有前段观测共同校正相关模块。数学上可以采用多目标优化、部分观测系统辨识和逐步训练，无需为此创造新的学习范式名称。

“完美实现”转化为限定条件下的可验收目标：准确预测、受观测约束的通路解释、能够被证伪的实验预测。当前不承诺全部参数唯一恢复，也不要求每个 effective state 等同真实细胞生理量。

主物种目标保持 macaque；marmoset 作为后续辅助或迁移对象。Synthetic 先验证方法，不给它赋予未经验证的物种身份。

## 2. 三个分开的研究问题

| 问题 | 主要证据 | 不足以支持的结果 |
|---|---|---|
| 多层观测是否增加机制信息？ | 同架构下，相应保留状态/通路输出和未训练干预预测改善 | 训练 latent loss 降低、seed 更一致 |
| 多尺度覆盖是否支持组合泛化？ | 固定预算下，未见尺度或未见层级×尺度组合的响应预测改善 | 已见尺度拟合更好、模型产生不同 RF 图 |
| 逐层训练是否有额外价值？ | 同架构、同观测曝光量下，优于或更稳定地达到 joint-from-scratch 的目标 | 与拥有更少数据的 RGC-only 比较 |

每个问题均允许部分支持或无增量收益。不要预设逐层训练必然优于联合训练，也不要预设更多层的监督必然改善全部干预。

后续实验设计目标：在仍符合已有数据的候选机制间，选择有信息量的刺激及观测层级。该目标排在方法可行性之后，本轮不实现。

## 3. 现有证据与保留边界

以下是用户提供的已完成报告摘要；本轮不重跑或重审这些结果。对应文件路径用于本地定位，Codex 应核对实际存在性，不因缺少远端副本否定用户的本地结果。

| 工作 | 已支持的范围 | 当前处置 |
|---|---|---|
| Canonical RetiPath / conductance upgrade | 具有预测证据支持的 E/I integration；已有严格 intervention 与 trace 接口 | 冻结生产模型和 checkpoint，作为历史参考 |
| H1 frozen center-surround assay | 在注册刺激与窗口内 FUNCTIONALLY_CONSISTENT | 结题；不自动追加窗口、contrast 或 cells |
| F2 localization | 当前 outer-retina/H1-like 计算线性，F2 首先出现在 BC 非对称变换 | 保留代码/模型事实；与外部 HC 观测仅构成条件性候选差异 |
| Full early relocation / 2×2 diagnostic | 该具体 PReLU 迁移有 prediction 代价，存在早晚位置交互 | 不采用候选；不推断全部早期非线性无用 |
| Chen / Raval early frontend | 标定或完整定义不满足直接接入条件 | 暂停；不猜 absolute light level，不再连续替换前端 |
| Croner–Kaplan DoG | 当前 measurement 未得到稳定可比较半径 | 封存该定量路径；DoG 拟合不作为新模型准入条件 |
| Synthetic S0 | BC 观测显著改善 BC 状态与 direct-BC intervention；整体优势未成立 | 保留为初步证据，不修改旧 teacher、loss 或训练结果 |
| Synthetic S0.5 | state/feedback 监督没有形成清楚的 H1 intervention 增益 | 保留 MIXED_OR_OPTIMIZATION_LIMITED；不再自动启动 S0.6 |
| Multi-level dataset map | 真实资料有潜力，跨层原始配对尚有缺项 | 作为将来的接入约束；当前不等待数据申请再设计 |

S0 关键事实：A 的 BC nRMSE 约 0.3001，C 约 0.0104；direct-BC intervention error 约 0.1366→0.0737。额外 RGC 数据在多数其他干预和跨 seed 一致性上更好。不能写成多层监督普遍超过更多 RGC 数据。

S0 与 S0.5 都是单 teacher、同模型家族、有限预算结果。它们不证明结构不可辨识，也不证明优化故障。只要没有新的独立理由，不追加误差分解或修改指标来获得预期胜负。

历史参考报告：
- `docs/RETIPATH_MULTIOBS_SYNTHETIC_S0.md`
- `docs/RETIPATH_MULTIOBS_S05_H1_OBSERVABILITY.md`
- `docs/RETIPATH_PRIMATE_MULTILEVEL_DATASET_MAP.md`
- `docs/RETIPATH_F2_LOCALIZATION_PILOT.md`
- `docs/RETIPATH_CONE_PHOTOTRANSDUCTION_FEASIBILITY.md`
- `docs/RETIPATH_RAVAL_WEBER_FRONTEND_FEASIBILITY.md`

## 4. 下一版架构原则

### 4.1 有分支与反馈的动态回路

保留这一计算组织方向：

```text
物理坐标刺激 → 有效局部输入 → H1-like state → feedback output
                         └──── 经反馈调制的输入 ────┐
                                                  ↓
                                   局部 BC input / state / output
                                      ├→ direct excitation ─┐
                                      └→ AC-associated ─I─┤
                                                           ↓
                                                E/I conductance
                                                           ↓
                                         RGC state / adaptation / spikes
```

此图仅说明计算组织；实际解剖连接需要独立证据。H1 的延迟、返回映射和下游输入合成须在设计稿中给出明确更新次序；不默认引入瞬时循环求解。AC-associated 支路保留，第一版不加入命名 subtype、presynaptic inhibition 或新生化状态。

### 4.2 Input / state / output / observation

每个模块分别定义输入 `u_l`、动态状态 `s_l`、通路输出 `o_l`、实验观测 `y_d`。Observation selector 只能接入对应变量。

HC voltage、BC 输入电流、BC 输出、RGC membrane 与 spikes 具有不同语义。Synthetic 第一版采用已知 identity observation；真实版本才讨论受限 affine、已知测量滤波和 noise model。不能通过 MLP/CNN decoder 吸收状态失配。

局部状态准确不保证下游 coupling 准确。计算每个 loss 的祖先参数集合，明确 state loss 是否可触及 pathway gain；不能把“层级越深”当成唯一更新规则。

### 4.3 局部单元与共享

优先设计少量具有物理位置的局部 BC 有效单元，在向 RGC/AC 汇总前保留局部 state/output。方程和类型参数尽量共享，连接局部、符号受约束。具体粒度和非线性位置由设计稿选择并说明，不立即展开真实 cone mosaic、全部 BC/AC subtype 或树突区室。

这一改动的科学目的：使局部与全局空间整合、非线性与汇总的先后顺序成为可检验计算。不能根据是否增大 F2 决定结构。

### 4.4 参数分层

区分机制形式、细胞/位置参数和实验观测 nuisance。第一版 synthetic 可固定 geometry 与观测映射，只学习小型共享回路。

真实版本允许有明确含义的部分共享，不假设全部灵长类数值相同。未知 calibration 不用 RGC NLL 假装恢复。消除不影响函数的精确 scale gauge；剩余不可辨识性如实保留。新参数可学习与否由科学定义决定，不把“0 trainable”作为唯一合格条件。

## 5. 多尺度表示与动态 RF

显示像素与回路节点分离。输入需含像素空间范围、坐标单位、时间轴和背景定义。在像素常值约定下，节点输入可以由同一个物理权重场在各像素面积上的积分得到：

`u_i(t) = sum_p X_p(t) * integral_{pixel_p} W_theta(r - r_i) dr`。

不同数据集改变离散表示，不自动改变物理机制。空间权重不强制为 DoG 或圆形。规则应说明坐标方向、边界、抗混叠/面积积分与局部非线性发生的离散层。

三种问题必须分开：
- 同一物理刺激，换充分分辨的网格：检验数值一致性。
- 改变真实尺寸/速度/contrast：检验不同响应的预测。
- 相同探针之前施加不同历史：检验状态依赖和 dynamic RF。

不能从粗 pooling 恢复已经丢失的细节，不能把未知视野外输入当成背景，不能改旧 checkpoint 的 degree/pixel 后声称同一冻结模型跨尺度。

Dynamic RF 作为状态依赖敏感度的派生读数。第一版不增加预测 RF 半径的网络，不要求 RF 随刺激尺寸伸缩，不以“RF 更动态”作为优化目标。不可微点使用预定义的有限扰动或合适工作点，不把代码分支导数当成唯一数学导数。

150 Hz 是候选数值时钟，刺激频谱独立定义。旧 50/20/40 ms 建议未成为已批准的新合同；不能自动替换 S0 历史协议。

## 6. 训练设计方向

目标函数应明确到：每个数据集的 observable、损失/likelihood、有效 mask、归约方式、权重或采样概率，以及可更新参数。

`J(theta) = sum_d w_d * mean_observed(loss_d) + regularization`。

连续观测可使用已知 noise 的 likelihood 或明示尺度的 normalized MSE。不同单位、时间相关、通道数和抽样次数均影响实际权重。随机抽 dataset 与多 batch 累积都可实现目标；选择必须记录等价目标或差异，不能宣称交替训练自动解决平衡问题。

三条件构成下一版最小训练对照：
- A：相同候选架构，RGC-only。
- B：相同候选架构和全部观测，joint-from-scratch。
- C：相同候选架构和全部观测，逐层预训练→新增模块拟合→回放与联合校正。

B/C 使用相同数据、初值配对、总观测曝光量和明确预算；记录实际 forward/backward 与时间，不能只按 optimizer.step 数宣布完全公平。A/B 比较观测增量，B/C 比较训练顺序。若要声称多层观测优于额外 RGC 数据，另需针对该论断的对照；第一轮不自动增加第四组。

阶段训练允许上游后续修正。保留旧层约束通过旧观测回放/明确目标实现，不默认永久冻结或任意添加新 loss。阶段时长和停止规则在结果前冻结；不允许边看测试边改 schedule。

## 7. 下一轮 synthetic 的验证设计

先用小型可实现 teacher 检验接口、训练与跨尺度预测。Teacher 是模拟真值；student 不复制教师学习参数。共享 geometry、固定尺度和 prior 属于已知条件，须明确记录，不能宣称发现了这些结构。

Stimulus 应覆盖已知物理尺度并保留少量跨层共同条件。避免 HC=大斑、BC=细栅、RGC=电影完全绑定。把部分尺度和层级×尺度组合在生成前留出；先做插值，极端外推后置。相同物理波形及其重复/重采样版本不能泄漏到不同划分。

第一轮需指定一个主要通路问题；可优先考虑已有明确端口的 direct-BC output/block，选择依据是研究位置与观测的关系，不能根据新结果选择 winner。H1 与其他干预作为有限辅助范围或延期，不构造任意全通路平均总分。

评价层次：
1. 保留响应：synthetic sampled NLL 与 excess CE。`excess CE = CE(p_teacher,p_student) - H(p_teacher)`；不只用总 CE 的百分比。
2. 对应状态/局部输出：已声明观测量在未见尺度与组合上的误差。
3. 未训练干预：局部变化与端到端 RGC 变化分别报告，保留绝对与相对误差。
4. 跨 seed 分散程度：同时报告真值误差；少量 seeds 不代表完整 posterior。
5. Dynamic RF：解释共同探针下的变化，不替代保留响应作为主要验收。

训练数据量、噪声与观测精度是方法的一部分。无噪声连续标签与随机 spike 不具有自动匹配的信息量；必须在结论中限定。

允许的负结果：多层仅改善被观察变量；progressive 与 joint 相当；额外尺度未改善泛化；模型失配后收益消失。保留这些结论，采用更简单或更受证据支持的方案。

## 8. 阶段计划与执行授权

| 阶段 | 单一产物/问题 | 运行权限与停止点 |
|---|---|---|
| G0 当前 | 可实施的架构、训练与 synthetic 对照设计稿 | 本轮授权：文档与必要源码阅读。完成后提交审阅；不训练、不实现新模块 |
| G1 后续 | 隔离原型和少量接口/数值单元检查 | 需用户批准 G0；保留正式模型，先证明输入/状态/观测计算可运行 |
| G2 后续 | 小规模 A/B/C synthetic 对照 | 需单独冻结具体预算、划分和主指标；不自动复用旧 3000-step 或 population |
| G3 后续 | 一项明示失配与最小真实观测接入 | 根据 G2 决策；真实数据获取在这里影响验证，不阻塞 G0 |
| G4 长期 | 候选机制与刺激/记录位置选择 | 需要可靠候选与独立测试；本轮不建设平台 |

暂停旧 S0.6 分解、DoG 修复、cone 前端替换、全量 dataset 审计与 Mach/SBC 新实验。历史脚本和结果保留。禁止用新研究叙事覆盖已消费开发区间或重命名为 untouched test。

## 9. 文档关系

- `AGENTS.md`：长期执行与证据规则。
- 本文件：当前研究方向、已有证据和阶段计划。
- `docs/NEXT_TASK.md`：唯一当前任务与写入范围。
- `docs/RETIPATH_ARCHITECTURE_V0.md`：已完成的 G0 设计交付；文档完成不表示候选模型已经实现或验证。
- 根目录 `DESIGN.md` 在已查看远端版本中是 figure design system，不用来替换本架构稿。

上述规则整理自用户的研究讨论、S0/S0.5 报告及已完成数据/生理审计。本文件提出的新架构与训练计划均为待验证设计，不声称已获生理实验证实。

### 本地落地记录（2026-09-18）

实际项目为 `D:\PythonProject\retina_rf_SNN`。指定更新目录 `D:\PythonProject\retipath\_research\_update\_v0` 不存在，本轮直接读取用户附件 `C:\Users\win11-pc\Downloads\retipath_research_update_v0.zip` 中的 `project_files/`；三份候选文件的字节数与包内 SHA256 清单相符，没有创建替代更新目录。

本地原根规则已完整保存于 [history/agents_before_multiscale_v0.md](history/agents_before_multiscale_v0.md)，随后合并本次候选规则；未发现实质权限冲突。历史报告摘要仍为用户提供/旧报告所载结果，本轮未独立复核。仅完成 G0 文档；G1/G2 的实现、测试、数据生成与训练未获本轮授权、未执行。
