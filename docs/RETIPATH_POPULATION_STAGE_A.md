# Population RetiPath v0.1 — Stage A correctness prototype

2026-09-18。状态：**Stage A 已实现；指定正确性验收 VERIFIED；29 tests passed。**

本轮用户明确授权 isolated forward 与 correctness tests，因而可执行本轮必要的前向、autograd 和 pytest；先前设计任务的“只写文档”停止边界不构成本轮实现禁令。训练、synthetic benchmark、旧模型/旧结果修改及 A.5 仍不在授权内。

依据：[AGENTS.md](../AGENTS.md)、[Population architecture v0.1](RETIPATH_POPULATION_ARCHITECTURE_V0_1.md)。本报告记录实现状态，不回写设计稿的历史状态。

## 1. 实际新增文件

| 文件 | 用途 |
|---|---|
| [experiments/retipath_population_v0_1/circuit.py](../experiments/retipath_population_v0_1/circuit.py) | 物理输入合同、层级参数、最小 Population forward、显式中间量与三种 intervention |
| [experiments/retipath_population_v0_1/test_stage_a.py](../experiments/retipath_population_v0_1/test_stage_a.py) | 指定 correctness tests；仅内存中的固定手工数值输入 |
| [experiments/retipath_population_v0_1/__init__.py](../experiments/retipath_population_v0_1/__init__.py) | 空包入口 |
| 本文件 | 实现、验收证据、默认值和限制 |

未修改正式源码、旧 isolated 模型、设计稿、AGENTS、NEXT_TASK、旧实验结果、数据或 checkpoint。没有执行 Git、切换分支、安装依赖或修改全局配置。

## 2. 已实现的前向与显式端口

```text
physical X / bounds / area / time / validity
  → fixed Q → q
  → H1 population h_H → a_H + F_H → H1_feedback
  → q - H1_feedback
  → ON/OFF BC u_B → v_B → w_B → s_B
  → selected BC output → r_B / r_B0 / delta_r_B
      ├→ gamma_BR + pi_BR → c_E → d_E
      └→ gamma_BA + pi_BA → AC a_A → o_A / o_A0 / delta_o_A
                                   → gamma_AR + pi_AR → c_I → d_I
  → positive gE/gI → V → adaptation + strictly-past event history
  → logit → probability
```

`PopulationRetipath` 默认 `LegacyPReLU`、CPU float64；构造时可选 `BaselineSoftplus`。`Stimulus.values` 为 `[B,T,P]` 的分箱保持 contrast；`pixel_bounds[P,2,2]`、`pixel_area[P]`、`time_ms[T]` 与 `input_valid[B,T,P]` 显式传入。时钟固定为 150 Hz，时间标记为连续 bin midpoint；观测事件是单独给定的 binary `[B,T,2]`。

返回 `Trace.inputs / states / outputs / initial_state`；`Trace.observation(port)` 是明确端口的 identity 读取，不是生理测量 adapter。下列变量均可直接读取：

| 分类 | 字段 | 形状 |
|---|---|---|
| input | q、u_H、q_modulated | B×T×25 |
| input | u_B、u_A | B×T×50；B×T×36 |
| state | h_H | B×T×25 |
| pathway output | H1_feedback | B×T×25 |
| state | v_B、w_B、s_B | B×T×50 |
| output | r_B、delta_r_B | B×T×50；r_B0 为 50，广播使用 |
| state / output | a_A、o_A、delta_o_A | B×T×36；o_A0 为 36 |
| local pathway contribution | c_E、c_I | B×T×2×2×50；B×T×2×2×36 |
| aggregate drive / conductance | d_E、d_I、gE、gI | B×T×2×2 |
| state | V；adaptation、history | B×T×2×2；B×T×2 |
| output | logit、probability | B×T×2 |

初值显式返回：神经有效状态、adaptation/history 为零，V 为 2/9。仅支持 `reset="baseline"` 的独立序列，明确拒绝 carry；不在实例中保留上次 batch 的状态。K=2 是 effective modes，不代表真实树突区室。

## 3. 物理图与固定工程默认

采用设计稿小图：Q/H1 各 25；ON/OFF BC 各 25；local ON、local OFF、broad ON、broad OFF AC 各 9；ON/OFF RGC 各 1。BC ON/OFF 复用相同坐标但使用相反 contrast polarity；各 RGC 的 direct route 只连接相同极性 BC。

Q 与 H1 几何复用既有纯数值函数：0.1 degree 方形 aperture，完整面积分母 0.01 deg²；节点坐标在 −0.30 至 0.30 degree 的 5×5 网格。H1 G 及返回 F_H=Gᵀ 保持其既有定义。没有输入缩放、学习 Q、外层非线性或独立 DoG surround。

AC 节点使用 −0.30、0、0.30 degree 的 3×3 网格。local BA 的 Gaussian sigma/radius 固定为 0.15/0.23 degree，broad 为 0.30/0.65 degree；BA 仅接受其所属 ON/OFF polarity。AR 在四个 family 内分别按完整支持归一化，保留 same/crossover 标记。BR/AR 的两个 mode 使用既有固定空间权重。以上是事前选定的 Stage A 工程 fixture，不是拟合的生理参数；验收过程中未调整这些值。

`pathway_support()` 返回完整结构祖先的 aperture mask。支持由固定图决定，不随 coupling 变小或 intervention 变化而缩小。当前是**完整图前向**：所有输入必须已知，任何 aperture 覆盖不全都拒绝；未知外围不填零、不裁剪后归一化。逐目标保留部分有效观测的加载/损失接口未实现，不能把本实现用于不完整真实视野后声称已完成部分覆盖处理。

## 4. 参数类别与 partial-pooling 合同

`physical_parameters()`、`parameter_groups()`、`parameter_registry()` 分别提供物理参数值、三类互不重复的可训练参数组、以及类别/归属/family/共享方式/边界/先验/允许数据端口信息。`gamma_BA` 独立登记为固定 coupling buffer，全部为 1，不进入可训练组。

默认可训练标量共 **359**：state **258**、output **56**、coupling **45**。这是参数注册数量，不是经过数据证明的可辨识自由度。

| 参数 | 物理变换边界 / 初值 | 个体与共享规则 |
|---|---|---|
| tau_H | (10,100) ms / 50 | 1 个 family center + 24 个零和 contrast |
| delay_H | (0,20) ms / 5 | 同上；raw deviation SD=0.3 |
| a_H | (0,0.8) / 0.3 | coupling；同上但 SD=0.1，强于 dynamics 收缩 |
| tau_f_B | (8,40) ms / 22 | 2 个 ON/OFF family centers + 48 contrasts |
| tau_gap_B | (20,140) ms / 78 | tau_s_B=tau_f_B+tau_gap_B，保证次序 |
| delay_B、kappa_B | (0,20) ms / 2；(0,1) / 0.5 | BC state；相同 family 层级，SD=0.6 |
| alpha_B 或 b_B | (0.05,1) / 0.5；或 (−4,4) / 0 | BC output，互斥注册，SD=0.6 |
| tau_A | (20,200) ms / [50,50,140,140] | 四 family 各一参数，无 unit contrast |
| delay_A、b_A | (0,20) ms / 5；(−4,4) / 0 | 四 family 共享 dynamics/output 工作点 |
| gamma_BR | (0.2,8) / 1.5 | 2 RGC×2 mode，固定 raw center prior SD=0.3 |
| gamma_AR | (0,8) / 0.375 | 2 RGC×2 mode×4 family，固定 raw center prior SD=0.3 |
| gamma_BA | 1 | fixed coupling，零可训练参数 |
| RGC bias | (−4,−0.5) / −2.4 | 两 RGC 参数，固定 raw center prior SD=0.3 |

边界为 sigmoid 变换的理论开区间；不将这些工程边界说成生理范围。H1 tau 的 deviation SD 同为 0.3；BC dynamics/output 均为 0.6。RGC type/mode/route 在本小图中每组仅有一个接收实例，故不伪造可估计的组内 variance，使用明确的固定 center prior，而不是每条 edge 独立 gain。

零和个体偏差由 family 内正交 contrast basis 构造，仅注册 n−1 个坐标；不能用重复的自由 unit 参数绕开 center/deviation 分解。`hierarchy_penalty()` 包含固定 SD 的二次项，`regularized_objective(data_loss)` 固定返回 data_loss 加 hierarchy term，没有关闭该项的开关。AC 无个体 deviation 参数；gamma_BA 无梯度参数；未启用的 BC 激活参数不存在。

验收证明参数可有个体差异、family 内偏差和为零、固定正则具有收缩梯度、参数组无遗漏/重复。它不证明未来任意外部 trainer 都会正确调用目标函数；本轮没有实现 optimizer/trainer，也没有声称已发生参数估计或收缩训练。

## 5. BC output 与 coupling 作用点

`LegacyPReLU` 返回 signed effective r_B，基线为零；`BaselineSoftplus` 返回 `softplus(b_B+s_B)`，基线 `softplus(b_B)`，两者都以 `delta_r_B=r_B-r_B0` 同时输入 direct 和 AC。`Trace.bc_quantity_kind` 分别标为 `signed_effective_output`、`nonnegative_release_like_proxy`。softplus 未被判定更好，也未被视为已验证生理 release。

H1 a_H 只在 h_H 到 feedback 的返回边；gamma_BR 只作用于 direct contribution；固定 gamma_BA 只作用于 BC→AC input；gamma_AR 只作用于 AC→inhibitory contribution。它们不混入 state 或 BC output 参数。signed delta 可减少兴奋或抑制，代码不取绝对值。

BC 使用设计稿的 cascade：先延迟输入、再 fast lowpass，再对 v_B 做 slow lowpass，s_B=v_B−kappa_B w_B。两低通按声明的离散序列组合，不声称逐 bin 顺序更新等于连续耦合 ODE 的联立精确解。AC 采用初稿的 family-shared 一阶 lowpass 与 baseline softplus 输出。未比较它们与 G1 双时间分量或旧 AC 的等价性。

RGC 使用既有 normalized conductance：softplus(b0+d_E/d_I)，b0=log(expm1(1))；gL=1、E_L=0、E_E=1、E_I=−1/3、C=60、V0=2/9。adaptation tau=120 ms，history tau=40 ms；logit=8(mean(V)−2/9)−0.5 adaptation−history+bias。这些量保持 effective 单位。

## 6. 三种 intervention 的实际语义

全部 intervention 对完整独立序列生效，给定同一刺激、事件历史及初值；从作用点重新前向，不覆盖已算好的 logit。

| intervention | 置零的量 | 保留的量 | 重新计算 |
|---|---|---|---|
| BLOCK_H1_FEEDBACK | H1_feedback | u_H、h_H、给定的过去 events | BC 输入/state/output、direct、AC、E/I、V、adaptation、logit/probability |
| BLOCK_DIRECT_BC_DRIVE | c_E、d_E | H1、BC、BA、AC、d_I/gI、history | gE 回 tonic 1，再计算 V、adaptation、logit/probability |
| BLOCK_AC_POSTSYNAPTIC_DRIVE | c_I、d_I | H1、BC、AC state/output、direct/gE、history | gI 回 tonic 1，再计算 V、adaptation、logit/probability |

AC postsynaptic block 不等于 AC silencing；direct block 不令 BC output 或 gE 变成零；H1 feedback block 不清除 h_H。三者都是计算作用点干预，没有药理等价声明。当前只支持单个 intervention 枚举，不添加组合干预实验。

## 7. 验收证据

运行环境：Windows / CPU，Python **3.12.7**、PyTorch **2.6.0+cpu**、pytest **7.4.4**。shape/finite/baseline 覆盖 float32 与 float64；其余机制/梯度核对为 float64。测试临时将 CPU 线程设为 1，结束时恢复原值。

最终命令在项目根目录执行；禁用 bytecode 与 pytest cache/plugin 自动加载仅限该测试进程，不修改配置文件：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& 'D:\anaconda\python.exe' -B -m pytest experiments/retipath_population_v0_1/test_stage_a.py -p no:cacheprovider -q
```

最终结果：**exit code 0；29 passed in 2.45s**。首轮 23 项通过后，补充了 6 项 intervention gradient checks；没有因为失败修改容差、物理参数或验收断言。BC quantity metadata 的最后补充后重新运行同一完整 focused suite。

| 验收点 / 测试函数 | cases | 实际检查 | 状态 |
|---|---:|---|---|
| shapes_finite_and_baselines | 4 | 两接口×两 dtype；全部显式端口形状、finite、正 conductance、零输入状态/基线 | VERIFIED |
| causal_prefix_events_input_gradients_and_reset | 2 | 改未来输入/事件保持过去前缀；对未来输入和当前/未来 event 的导数为零；重置与短前缀一致 | VERIFIED |
| state_output_coupling_gradient_routing | 2 | 11 个观测端口逐项核对非零祖先及无梯度的非祖先；三个参数组互斥且完整 | VERIFIED |
| output_and_coupling_changes_do_not_rewrite_upstream_states | 2 | 改 BC output 不改 s_B；改 a_H 不改 h_H；改 gamma_BR 不改 BC/AC states | VERIFIED |
| physical_grid_consistency_and_incomplete_coverage | 2 | 同一物理分块场在 32/64 grid 的全部 state/output 一致；所有 intervention 均拒绝缺失物理覆盖 | VERIFIED |
| on_off_local_broad_and_signed_drives | 2 | 固定远端局部输入不驱动 local AC 中心却驱动 broad；极性 mask、family normalization、signed excitation/disinhibition contribution | VERIFIED |
| interventions_preserve_upstream_and_recompute_membrane | 6 | 两接口×三阻断；上游保留、tonic conductance、下游变化；独立逐时间步递推核对 V/adaptation/logit/probability | VERIFIED |
| partial_pooling_individuality_and_mandatory_objective_term | 2 | 不同 unit 可不同、零和偏差、AC 无 unit kinetics、固定 BA、强弱 SD、正则值/收缩梯度、支持不随 gamma 缩小 | VERIFIED |
| intervention_gradient_disconnection | 6 | 阻断后 logit 对被切断路径参数无梯度，对其余有效祖先仍有梯度 | VERIFIED |
| input_contract_rejections_and_no_dataset_selector | 1 | 拒绝缺失输入/错误面积/错误时钟/非有限值/非 binary events/carry/未知配置；无 dataset/session 机制选择入口 | VERIFIED |

物理 grid 等价测试使用边界对齐的分块场，绝对/相对容差均为 2e−12；因果前缀与独立膜递推使用 1e−12 或 2e−12。它不是任意连续场有限网格逐位等价的声明。输入是固定、短小、内存内的代数 fixture，没有 teacher 抽样、数据划分、benchmark dataset 或外部 spike payload。

## 8. 文件保留与指纹

本轮新增实现/测试的 SHA256：

| 文件 | SHA256 |
|---|---|
| circuit.py | `2C269625854B139CAD43AACE55897DE4F9C30454B0C215B3ABDAC4F127BE307D` |
| test_stage_a.py | `0F4C5199A4C5C1839217D19B199FD6DBB8EC06791243DC8A561EF0FD1B0C88ED` |

以下六个直接相关旧文件的任务前后 SHA256 一致：

| 文件 | SHA256 |
|---|---|
| AGENTS.md | `2DF3C3551A6A04AEB664744FCA68CC8C6AD83AC01531718DFE9C81FA0F2A1FD2` |
| docs/RETIPATH_POPULATION_ARCHITECTURE_V0_1.md | `85D748E105E49CED3E23E48EACD7B8060BCB55F80EF67D636C9D0A34CA09B828` |
| experiments/retipath_multiscale_v0/circuit.py | `FE31B115B87CE4D4AB4A69B50F3615E9EF5054959EC434A68A640ED427594F46` |
| experiments/retipath_multiscale_v0/contracts.py | `EB7A8BBF04A352FBF0BBFD1520745BE14F40ABD2539E0023EF739E3CD3C39E85` |
| models/mechanistic_retina/state.py | `FB20576FA7C277DEB01261D0AD7F743211A7AAB2643986D8BFFD2BA47B3E486B` |
| models/mechanistic_retina/retipath_spatial_ei.py | `A0A1C4301BF04EDDD40D5561BB63D5242873F0687970F3116505EDB99EF5DE9C` |

这项核对只支持列出文件未变，不冒称全仓库或全部旧实验工件已重新审计。

## 9. 未执行与停止边界

- **未训练、无 optimizer step、无 checkpoint、无 synthetic benchmark、无调参。**测试中的手工参数扰动仅用于代数不变量和导数检查，没有拟合目标或评价条件排序。
- 未启用 BC coupling、AC→BC、outer nonlinearity 或其他机制；未新增 data adapter、trainer、synthetic generator、真实记录 current calibration 或 per-dataset gradient diagnostic runner。
- 全部证据限于上述 CPU correctness fixtures；GPU、长序列数值行为、流式 carry、部分覆盖观测装载、真实数据单位映射与科学可辨识性仍未验收。
- **未进入 A.5**：没有 Population/G1 全图 migration comparison，没有 F2/RF 实验，也没有 Stage B/C 数据或训练。复用既有数值函数不构成迁移等价证据。

**Stage A 交付完成；无阻塞本轮交付的用户决策。完成后停止。**
