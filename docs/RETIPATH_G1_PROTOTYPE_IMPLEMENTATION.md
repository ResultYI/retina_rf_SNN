# RetiPath G1 isolated prototype：实现与验收

日期：2026-09-18。状态：**G1 已实现；本轮指定的必要验收 VERIFIED；完成后停止。**

依据：[架构设计 v0](RETIPATH_ARCHITECTURE_V0.md)。本轮用户明确授权隔离原型与必要验收，覆盖 G0 文档中“仅文档、不得运行测试”的当时阶段限制；并未授权 G2。没有改动研究设计、正式模型、loader、trainer、数据、checkpoint 或旧实验结果，没有执行 Git。

## 1. 实际交付与隔离范围

| 文件 | 本轮工作 |
|---|---|
| [contracts.py](../experiments/retipath_multiscale_v0/contracts.py) | 新建：物理像素网格、StimulusBatch、CircuitGeometry、StateBundle、ObservationBatch、干预合同、identity 选择与观测 loss 接线 |
| [circuit.py](../experiments/retipath_multiscale_v0/circuit.py) | 新建：固定几何、面积积分、局部 BC、完整 H1/direct/AC/EI/RGC 回路、12 参数及三条件参数组接口 |
| [test_contract.py](../experiments/retipath_multiscale_v0/test_contract.py) | 新建：5 组必要单元验收，覆盖本轮八项要求；仅内存手写输入、前向和梯度检查 |
| 本报告 | 新建：实现边界、验收方法、结果与限制 |
| [NEXT_TASK.md](NEXT_TASK.md) | 更新：G1 完成状态、当前停止点；G0 原任务保留为历史 |

没有创建 `protocol.json`、`synthetic_study.py`、训练入口、数据文件或 checkpoint。没有增加依赖或修改全局配置。`AGENTS.md`、`RESEARCH_PLAN.md` 和 `RETIPATH_ARCHITECTURE_V0.md` 的 SHA256 均与本轮开始时一致；保留 G0 日期与内容，不回写成当时已经实现。

## 2. 实现对应关系

- **坐标与表示解耦**：刺激 values 携带 `[P,xy,lower_upper]` 物理像素边界/面积，回路节点位置与 G/A/D/I 不依赖 P。Q 使用像素与固定 0.10 deg 方窗的精确矩形交叠面积；不使用 resize、像素中心近似或截断后重归一化。`regular_pixel_bounds` 只建立同一视场的数值网格，不改变物理尺寸。
- **输入边界**：验证面积、时间、单位、校准、形状、dtype/device；重复像素、网格间隙/重叠、缺失输入、节点方窗覆盖不足均拒绝。G1 支持完整 Cartesian tessellation（允许非均匀间距及像素重排），不承诺任意多边形或动态缺像素。
- **state/output/observation 分离**：`StateBundle.inputs/states/outputs` 分开保存张量；h 与反馈 f、BC 动态 s_B 与非线性 o_B 分离。观测 selector 在回路外读取指定端口；HC→h、BC→o_B、RGC→p/ell 的 Bernoulli likelihood，不引入可学习 decoder。观测 provenance 只用于核对对应关系。
- **局部计算与分支**：25 个 BC/H1 节点、9 个 AC-associated 节点、1 个 RGC、K=2。局部 fast/slow 状态产生 sustained/transient 有效分支，在空间汇总前施加共享 PReLU。H1 延迟低通后经 aH·Gᵀ 返回，direct 与 broad→AC 使用同一 o_B；AC signed drive 保留符号。
- **后端**：沿用已读源码的低通、严格过去 history 与指数 V 更新函数；后端常数、12 参数边界及 student 初始化中心照设计稿实现。未实例化旧 integrator 的多余 gain 参数，未迁移旧 checkpoint。所有回路量保持 synthetic/effective 语义。
- **干预**：`BLOCK_DIRECT_BC_DRIVE` 置零传输端口 d_E/uE，保留 tonic gE=1、BC 状态/输出、AC/gI 和相同过去 history；从原 V0 重算 V、m、adaptation、ell、p。没有把它改成 BC silencing、gE=0 或药理阻断。

每次 forward 都从设计指定的独立序列初态 reset，没有跨调用隐藏状态；本轮未提供流式 chunk carry/resume 接口。StateBundle 返回完整序列的显式状态，后续若需要持续流式接口，须按既有设计的 carry 约定单独实施，不把当前独立序列接口说成已经支持流式运行。

## 3. 三条件参数组接口

`TrainingCondition`、`active_parameter_names(condition, macro_step)`、`model.parameter_groups(...)` 和 `model.configure_trainable(...)` 只确定参数分组及 requires_grad，不创建 optimizer、不更新参数、不消费训练 batch。

| 条件/阶段 | 步编号 | 活跃参数 |
|---|---|---|
| RGC-only | 1–180 | 全部 12 参数 |
| joint | 1–180 | 全部 12 参数 |
| progressive C1 | 1–30 | tauH，1 个 |
| progressive C2 | 31–60 | aH、tauf、taus、alpha，4 个 |
| progressive C3 | 61–180 | 全部 12 参数 |

参数按 H/F/B_s/B_o/A/E/I/R 语义分组。冻结/解冻清空 `.grad`，不修改任何参数值；切换条件/阶段不改变 forward 图和输出。可提供 seed 进行设计规定的 raw 坐标扰动初始化；同 seed 初值逐项一致。正式 batch 日程、optimizer 动量、曝光公平性与训练效果均未运行，不能用本接口验收冒充 G2 结果。

## 4. 必要验收结果

运行环境：Windows，`D:\anaconda\python.exe`，Python 3.12.7，PyTorch 2.6.0+cpu，CPU。验收主断言使用 float64；另做设计运行 dtype float32 的前向/全部 12 参数梯度有限性及配对初始化检查。输入为最多 24 bins 的手写小数组，未采样 teacher 标签、未运行设计中的 136 序列数据规格。

命令（工作目录为 `D:\PythonProject\retina_rf_SNN`）：

```text
D:\anaconda\python.exe -B -m unittest experiments.retipath_multiscale_v0.test_contract -v
```

最终结果：**Ran 5 tests in 0.685s — OK**。下列数值来自这次运行，不是训练结果或生理结论。

| 用户要求 | 验收证据 | 状态 |
|---|---|---|
| 同一 physical stimulus 跨 grid 一致 | 同域 32×32/64×64 的同一物理方块：max Δp=`2.77555756156e−17`；常量积分一致。Gaussian 像素面积平均相对解析节点积分的最大误差：32 网格 `0.0111399373377`，64 网格 `0.00244949554638`；两网格 max Δp=`2.84353548744e−5` | VERIFIED，精确可表示场一致；一般平滑场是离散收敛，非有限网格完全相等 |
| 真正 scale 改变能改变 response | 同一 64 网格、固定节点/参数/时间波形，Gaussian sigma 从 0.15 改为 0.60 deg：max Δp=`0.00470007347324` | VERIFIED，仅该未训练回路的手写用例 |
| coarse pooling 信息不能恢复 | 64 网格交替正负细条纹与零场经 2×2 area pooling 后完全相同；coarse 和上采样零场响应一致；原 fine 与重建输入的 max Δp=`0.000261810407579` | VERIFIED，构造了同 coarse 表示对应不同 fine 输入的反例 |
| state/output 分离 | h 首两步与 q_f 首步匹配解析递推；负 s_B 映射到 alpha·s_B，o_B 不是 s_B 的别名；零输入 V 保持 V0 | VERIFIED |
| intervention semantics | block 下 BC/AC/history 及 gI 与 normal 逐项相同；d_E/uE=0、gE=1；独立逐 bin 标量计算重现 blocked V/adaptation/logit；max Δp=`0.007984468813` | VERIFIED，条件计算干预 |
| causal/no-future-leakage | 未来 stimulus/events 改动不改变此前 trace；ell_t 对 stimulus_>t 梯度为 0，对 events_≥t 梯度为 0；截短前缀、重复 reset、batch 序列隔离通过 | VERIFIED |
| gradient routing | HC loss 仅连接 tauH；BC output loss 连接 tauH/aH/tauf/taus/alpha；RGC loss 连接全部 12 参数；BC state 本身不连接 alpha | VERIFIED，连接组在手写用例均有非零梯度 |
| backbone 不按 dataset/session ID 选机制 | 改变 dataset/session metadata、sequence/split IDs 后全部输入/state/output trace 逐项相同；forward 源码不读取 metadata；观测对应关系检查仍有效 | VERIFIED |

数值容差属于工程验收：精确物理方块/常量的 float64 比较使用 `atol=5e−13`；Gaussian 响应差预定界限为 `1e−4`，同时必须比对解析节点积分并确认细化误差降低；尺度/干预非零检查使用 `1e−8` 排除浮点零。这些不是生物学有效性、统计显著性或未来训练 PASS 阈值。

首轮执行曾有 1 条断言失败：振荡输入未产生负 uI，因此不能检验禁止 `abs` 的符号要求。保留该符号要求，加入持续负输入的手写用例；未修改模型、参数或放宽断言。修正验收输入后最终 5 组全部通过。每组验收结束均逐项比较模型参数/几何 buffer 与开始值，未发生数值更新。

### 可复核文件指纹

| 文件 | SHA256 |
|---|---|
| `contracts.py` | `eb7a8bbf04a352fbf0bbfd1520745be14f40abd2539e0023ef739e3cd3c39e85` |
| `circuit.py` | `fe31b115b87ce4d4ab4a69b50f3615e9ef5054959ec434a68a640ed427594f46` |
| `test_contract.py` | `b81644e9954973c7ca1d4f11e95cc510ea3f6eceb239892e9b78c6b99773312b` |
| 冻结设计稿 `RETIPATH_ARCHITECTURE_V0.md` | `53be0d730754129e1e829aa5c9a907865806c323e7cdbad0e6882c1ff3fdac9c` |

已核对复用的 `models/mechanistic_retina/state.py`、`retipath_spatial_ei.py` 及参照的 `evaluation/mechanistic_retina/mechanism_observation.py` 在本轮前后 SHA256 一致。没有为完成原型改动这些正式模块，也没有执行旧 checkpoint 回放。

## 5. 结论边界与停止点

八项指定行为在上述必要用例中获得 VERIFIED；没有发现需要改变研究设计的实质问题。该结论仅覆盖 G1 接口/计算/梯度正确性，不证明收敛、参数可辨识、多层观测收益、progressive 优势、保留测试表现或真实生理机制，这些均未验证。

本轮为 **0 training、0 optimizer steps、0 parameter updates、0 正式 synthetic datasets、0 checkpoint 读取/保存**。未训练 A/B/C，未消费 G2 test，未重跑 S0/S0.5，未追加 S0.6，未扩展机制或实验。完成 G1 后停止；G2 仍需单独明确授权。
