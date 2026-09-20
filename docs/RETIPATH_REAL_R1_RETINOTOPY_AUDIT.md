# Population RetiPath real-data R1.1：retinotopic-registration audit

审计日期：2026-09-20。范围：R1 的 9 个 MC biological cells / 16 recordings；仅几何、坐标与配准合同。**R1 未使用 cell-specific LN retinotopic center：九个 cell 的 Population target 均为 crop 原点 (0,0)，而九个冻结 LN center 均非零。** 本文不判断配准对预测 NLL 的影响。

只读取已有源码、协议、报告、checkpoint 中的坐标/连接矩阵和来源元数据；进行坐标转换、矩形交面积和固定连接图祖先集合计算。没有加载 stimulus/spike payload 或 validation prediction/target 工件，没有实例化/运行模型 forward，没有优化、训练、位移搜索、RF 或新实验。唯一新增文件是本文；旧源码、checkpoint、结果均保持不变。

## 1. 证据范围与身份核对

按要求读取 [AGENTS.md](D:/PythonProject/retina_rf_SNN/AGENTS.md)、[Population v0.1 设计](D:/PythonProject/retina_rf_SNN/docs/RETIPATH_POPULATION_ARCHITECTURE_V0_1.md)、[R0 preflight](D:/PythonProject/retina_rf_SNN/docs/RETIPATH_REAL_R0_PREFLIGHT.md)、[R1 protocol](D:/PythonProject/retina_rf_SNN/docs/RETIPATH_REAL_R1_PROTOCOL.md)、[R1 results](D:/PythonProject/retina_rf_SNN/docs/RETIPATH_REAL_R1_RESULTS.md)，并核对研究导航。本次数值来自保存的 tensor，不从报告中的图或 NLL 反推中心。

| 证据 | 本次核对 |
|---|---|
| LN `cells/<slug>/ln-trained.pt` | 九份 SHA256 均等于 2026-09-05 alignment [manifest](D:/PythonProject/retina_rf_SNN/output/audits/macaque_fixed_alignment_experiment_20260905/manifest.json) 中事前记录的来源 hash；逐份提取 `model.center_xy` |
| LN `cells/<slug>/results.json` | 九份均声明 full-train fresh refit、original validation 不用于选择；checkpoint 的 best step / refit steps 与对应记录一致 |
| LN 推导相关源码 | 本次读取的 LN model、cell runner、source loader、LN trainer、inner-dev split/early-stopping 源码及原物理像素代码，共 6 文件，hash 均与 LN [run-manifest](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/run-manifest.json) 冻结值相同 |
| Population R1 `cells/<slug>/final.pt` | 九份均匹配原 [CHECKPOINT_LOCK.json](D:/PythonProject/retina_rf_SNN/output/real_data/retipath_population_r1/CHECKPOINT_LOCK.json)；九份坐标和固定路由 buffers 逐元素相同；两种 RGC port 坐标均为 (0,0) |
| Population R1 源码 | [source_lock.json](D:/PythonProject/retina_rf_SNN/output/real_data/retipath_population_r1/source_lock.json) 的 22 项 hash 全部匹配；因此本次检查的实际源码与运行冻结来源一致 |
| 后来的 fixed-LN-center Canonical | 九份 `cell_positions_degs` 与对应 LN center 按原 float32 转换的结果逐元素相等 |
| R1 选用的旧 Canonical baseline | 九份 checkpoint 的 `cell_positions_degs` 均为 (0,0)，与后来的 fixed-center lineage 不同 |

这里的来源核对不重新验证历史训练结果，也不读取历史或当前 NLL 来决定 center。LN 的“train-only”是**估计/选择不使用 outer validation**，不是声称旧 loader 从未加载 validation：旧 loader 会装载原 split，但 center 估计入口只接收 train。

## 2. LN 坐标定义与每个 cell 的中心

[CenterSurroundLN](D:/PythonProject/retina_rf_SNN/baselines/center_surround_ln.py:33) 使用连续参数 `center_xy=(x_LN,y_LN)`。其 grid 是 `arange(17)-8`，x 沿 column 向右，y 沿 row 向下；中心是两个 Gaussian component 共用的拟合参数，不是离散峰值或解剖位置。

因此**保存的原始 `center_xy` 本身就是相对 17×17 crop 中心的 pixel offset**。为避免把它误当左上角起算的坐标，同时列出：

- 原始/offset：`(x_LN,y_LN)`，单位 pooled pixel，y 向下。
- 17×17 内连续零基 pixel coordinate：`(column,row)=(8+x_LN,8+y_LN)`；crop 中心为 (8,8)。
- physical degree：`(x_deg,y_deg)=(x_LN*p,-y_LN*p)`，y 向上；`p=3×4.6/256=0.05390625°/pooled pixel`。沿用 [原 _cone_positions](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:220) 的 crop-relative 原点；**不添加半像素、不重新解释屏幕/眼底绝对原点**。这里不是原 256×256 movie 的绝对像素索引。

| cell | 类型 | 原始 center_xy = crop-center offset (pixel；x右/y下) | 17×17 内 (column,row)，零基 | 原始工件 | train-only center / 最终 refit |
|---|---|---|---|---|---|
| 67#6 | MC OFF | (-2.295853138, 0.907617807) | (5.704146862, 8.907617807) | [67_6/ln-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/67_6/ln-trained.pt) | 是；858 refit steps |
| 67#7 | MC ON | (1.614822388, -0.488627136) | (9.614822388, 7.511372864) | [67_7/ln-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/67_7/ln-trained.pt) | 是；917 refit steps |
| 67#33 | MC OFF | (-0.036879927, -1.822488546) | (7.963120073, 6.177511454) | [67_33/ln-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/67_33/ln-trained.pt) | 是；900 refit steps |
| 68#3 | MC OFF | (-2.796986341, -0.607611060) | (5.203013659, 7.392388940) | [68_3/ln-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/68_3/ln-trained.pt) | 是；920 refit steps |
| 68#10 | MC ON | (-4.570400715, -6.205354691) | (3.429599285, 1.794645309) | [68_10/ln-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/68_10/ln-trained.pt) | 是；996 refit steps |
| 69#4 | MC ON | (1.130275369, -0.207242042) | (9.130275369, 7.792757958) | [69_4/ln-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/69_4/ln-trained.pt) | 是；880 refit steps |
| 69#6 | MC OFF | (0.703732610, 1.300837278) | (8.703732610, 9.300837278) | [69_6/ln-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/69_6/ln-trained.pt) | 是；1000 refit steps |
| 69#7 | MC ON | (1.283694863, 0.704375207) | (9.283694863, 8.704375207) | [69_7/ln-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/69_7/ln-trained.pt) | 是；979 refit steps |
| 70#34 | MC ON | (-0.933394730, -1.403185606) | (7.066605270, 6.596814394) | [70_34/ln-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/70_34/ln-trained.pt) | 是；976 refit steps |

### Train-only 来源

[run_ln_cell](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/schottdorf_center_surround_ln.py:30) 调用 `select_and_refit_ln(data.train, history)`；[select_and_refit_ln](D:/PythonProject/retina_rf_SNN/training/mechanistic_retina/center_surround_ln.py:175) 在原 train 内建立 inner-fit / inner-dev，选择正则系数和步数，再从 fresh initialization 在 full original train 上 refit。保存的是 refit 的 `model.state_dict()`。外部 validation 的评价在该选择和 refit 完成后，未反馈给 center 拟合或选择。

九个 cell 的元数据均为 `full_train_fresh_refit=true`、`original_validation_used_for_selection=false`，且 refit 步数一致核对通过。“train-only”包含原训练集内的 inner-dev 选择，不应缩写成只用了 inner-fit。

### Degree 坐标与 Population target 的逐 cell 差异

为与已有配准工件保持完全相同的中心，本文采用原 fixed-center Canonical 的数值操作：
`c_i = float64(float32(center_xy) * float32([p,-p]))`。
即先复用旧 float32 degree 转换，再无损提升到 Population 的 float64。与直接用 float64 做理想公式转换的最大差为 **4.4703483693e-9°**；它不改变任何 coverage 判定。表中显示 9 位小数，附录保留完整存储精度。

“差”定义为 LN physical center − Population 指定 target center；最后一列只是欧氏几何距离。

| cell | LN center c_i (deg；x右/y上) | Population target (deg) | 几何差 (deg) | 距离 (deg) |
|---|---|---|---|---|
| 67#6 | (-0.123760834, -0.048926271) | (0, 0) | (-0.123760834, -0.048926271) | 0.133080893 |
| 67#7 | (0.087049022, 0.026340056) | (0, 0) | (0.087049022, 0.026340056) | 0.090946857 |
| 67#33 | (-0.001988059, 0.098243527) | (0, 0) | (-0.001988059, 0.098243527) | 0.098263640 |
| 68#3 | (-0.150775045, 0.032754034) | (0, 0) | (-0.150775045, 0.032754034) | 0.154291740 |
| 68#10 | (-0.246373162, 0.334507406) | (0, 0) | (-0.246373162, 0.334507406) | 0.415445471 |
| 69#4 | (0.060928907, 0.011171642) | (0, 0) | (0.060928907, 0.011171642) | 0.061944631 |
| 69#6 | (0.037935585, -0.070123263) | (0, 0) | (0.037935585, -0.070123263) | 0.079726912 |
| 69#7 | (0.069199175, -0.037970226) | (0, 0) | (0.069199175, -0.037970226) | 0.078932020 |
| 70#34 | (-0.050315809, 0.075640477) | (0, 0) | (-0.050315809, 0.075640477) | 0.090846917 |

九个 cell 都不匹配。Population 的**指定坐标中心为零**不等于断言其拟合后 effective RF 的质心必为零；本次没有计算 RF。

## 3. 旧 Canonical 对功能中心的使用：两个 lineage 必须分开

1. **R1 实际比较的 Canonical** 是 `schottdorf_canonical_v1_shared_bc_development_22cell_20260830`。其 [run.py](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/run.py:93) 将 loader 的 `data.cell_positions_degs` 传给 builder；[原 loader](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:141) 返回零坐标。九个保存 checkpoint 也均为零。
2. **后来注册到 LN center 的 Canonical** 是 `schottdorf_canonical_v1_fixed_ln_center_22cell_20260905`。其 [alignment / factory](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/preflight.py:58) 从冻结 LN checkpoint 取 `center_xy`，核对 train-only refit 元数据，执行上述 degree 转换并传入 builder。九个保存的 `cells/<slug>/model-trained.pt::cell_positions_degs` 与此次转换精确一致；[coordinate_mapping.json](D:/PythonProject/retina_rf_SNN/output/audits/macaque_fixed_alignment_experiment_20260905/coordinate_mapping.json) 保存了同一符号和 pitch 合同。
3. Canonical 的 [PathFeatureBank](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/bipolar_subunits.py:72) 将 `cell_positions` 用于 spatial basis 和 radius-defined support；[_spatial_basis](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/bipolar_subunits.py:259) 计算 cell-to-input distance；[support_partition](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/support_partition.py:34) 用相同中心定义 BC/AC/H1 路径 support。旧 H1 图自身基于 cone positions 构建，不能将其描述成已经采用 Population 的全部局部节点平移合同。

因此，已有 fixed-center 工件证明 LN center 已被另一条 Canonical lineage 使用，**不证明 Population R1 已继承该注册，也不改变 R1 固定的 baseline 身份或旧比较结果**。

## 4. Population R1 实际执行的坐标合同

实际源码入口：[Population constructor](D:/PythonProject/retina_rf_SNN/experiments/retipath_population_v0_1/circuit.py:143)、[fixed_geometry](D:/PythonProject/retina_rf_SNN/experiments/retipath_multiscale_v0/circuit.py:61)、[real adapter](D:/PythonProject/retina_rf_SNN/experiments/retipath_population_v0_1/real_data.py:86)、[R1 forward 包装](D:/PythonProject/retina_rf_SNN/experiments/retipath_population_v0_1/real_r1.py:191)。

| 对象/空间操作 | R1 实际值与含义 |
|---|---|
| 原始 stimulus 的坐标 | 固定 crop frame，289 个 pooled pixels；物理范围 x/y 各 [-0.458203125, +0.458203125]° |
| Q aperture centers `input_xy` | 25×2；x/y 各 {-0.30,-0.15,0,+0.15,+0.30}° 的 Cartesian grid；原点为 crop 中心 |
| H1 `h_xy` | 25×2，与 Q centers 相同；G 在这些固定节点之间连接，sigma=0.18°、radius=0.36°；feedback 使用 G 转置 |
| BC `bc_xy` | 50×2；ON/OFF 各一份相同 25-node grid，局部输入沿该位置取得 Q 及 H1 feedback |
| AC `ac_xy` | 36×2；local ON / local OFF / broad ON / broad OFF 各 9 nodes；每 family x/y 各 {-0.30,0,+0.30}° |
| BC→AC 空间路由 | 以各 AC receiver 为中心连接同极性 BC；local sigma/radius=0.15/0.23°，broad=0.30/0.65° |
| RGC `rgc_xy` | 2×2，ON/OFF 均 (0,0)；recorded port 仅按 MC ON/OFF 选择 |
| direct BC→RGC / AC→RGC | 以 (0,0) 为 receiver center；两空间 modes sigma=0.15/0.30°，radius=0.45°；不是一个可移动中心 |
| adapter 是否按 cell 平移 | **否**。`physical_stimulus` 固定实际 pixel bounds；`bundle.centers` 是 289 个 stimulus pixel centers，不是 LN/RGC center |
| Q 是否使用 cell-specific origin | **否**。`q = x @ area_integral_weights(pixel_bounds, self.input_xy).T`，其中 `input_xy` 对九个 cell 相同 |
| constructor / checkpoint | 每 cell 单独构建实例，但没有传入 center；九个 final checkpoint 的 `input_xy/h_xy/bc_xy/ac_xy/rgc_xy/G/F_H/pi_BR/pi_BA/pi_AR` 全部相同 |

因此，R0 的原点 coverage 通过只说明**零中心布局**被 crop 覆盖，并未验证 LN retinotopic registration。原点 Q aperture 的整体 bounding box 是 [-0.35,+0.35]°；四边各留 0.108203125°。节点之间的个体参数差异不改变这些坐标 buffers。

## 5. 把整个 circuit 平移到 c_i 后的 coverage

### 计算定义

每个 Q aperture 为边长 0.10° 的方形，面积 0.01 deg²，中心为 `input_xy[j]+c_i`。使用 adapter 原来的 17×17 pixel bounds；只做矩形交面积：

```text
W[j,p] = area(aperture[j] ∩ pixel[p]) / 0.01
coverage[j] = sum_p W[j,p]
```

此处的部分 overlap 数值**只用于记录缺失面积**，没有作为可执行 Q 输入，没有填零，也没有将 W 除以其行和。用 289 个 pixel bounds 逐项求交，与 crop 整体矩形求交的最大差为 **3.3306690738754696e-16**。

整个有限 circuit 的输入支撑是 25 个 aperture 的并集；其 bounding box 为 `c_i+[-0.35,+0.35]^2`。对当前正方形 crop，全部 Q 完整覆盖当且仅当：

```text
|c_i.x| <= 0.108203125 deg AND |c_i.y| <= 0.108203125 deg
```

这不是新增经验阈值，而是由既有 aperture/crop 边界直接得到的几何条件。最小边缘余量为 `0.108203125-max(|c_i.x|,|c_i.y|)`；负值表示越界。

| cell | Q 完整 / 部分 / 完全无覆盖（总25） | 最小 aperture coverage | 25 aperture 平均 coverage | 最小边缘余量 (deg) | 整体 circuit 输入域 |
|---|---|---|---|---|---|
| 67#6 | 20 / 5 / 0 | 0.844422907 | 0.968884581 | -0.015557709 | unknown spatial domain |
| 67#7 | 25 / 0 / 0 | 1.000000000 | 1.000000000 | 0.021154103 | 完整 |
| 67#33 | 25 / 0 / 0 | 1.000000000 | 1.000000000 | 0.009959598 | 完整 |
| 68#3 | 20 / 5 / 0 | 0.574280798 | 0.914856160 | -0.042571920 | unknown spatial domain |
| 68#10 | 12 / 4 / 9 | 0.000000000 | 0.517913151 | -0.226304281 | unknown spatial domain |
| 69#4 | 25 / 0 / 0 | 1.000000000 | 1.000000000 | 0.047274218 | 完整 |
| 69#6 | 25 / 0 / 0 | 1.000000000 | 1.000000000 | 0.038079862 | 完整 |
| 69#7 | 25 / 0 / 0 | 1.000000000 | 1.000000000 | 0.039003950 | 完整 |
| 70#34 | 25 / 0 / 0 | 1.000000000 | 1.000000000 | 0.032562648 | 完整 |

原始缺失位置（索引按现有 grid：y=-0.30 到 +0.30，每行 x=-0.30 到 +0.30，零基）：

| cell | 缺失支撑的定位与原始数值 |
|---|---|
| 67#6 | 左侧越界 0.015557709336°；Q indices 0,5,10,15,20 部分覆盖，各 coverage≈0.844422906637。所有节点中心本身仍在 crop 内，但 aperture 不完整 |
| 68#3 | 左侧越界 0.042571920156°；Q indices 0,5,10,15,20 部分覆盖，各 coverage≈0.574280798435。节点中心在 crop 内同样不能保证 aperture 完整 |
| 68#10 | 左侧越界 0.138170036674°，上侧越界 0.226304280758°；indices 16–19 各 coverage≈0.236957192421；indices 0,5,10,15,20,21,22,23,24 完全无覆盖；其余 12 个完整 |

### H1 / BC / AC / RGC 的路径级 coverage

依据 [pathway_support](D:/PythonProject/retina_rf_SNN/experiments/retipath_population_v0_1/circuit.py:244) 对 checkpoint 中固定连接矩阵独立做 Boolean 祖先集合运算。一个节点/模式仅在所有祖先 Q apertures 完整时算完整；不按训练后的幅度变小、某次 intervention 或当前 stimulus 值缩减 support。

```text
H support        = nonzero(G)
feedback support = nonzero(F_H) compose H support
BC support       = feedback support union own-Q   (repeat ON/OFF)
AC support       = nonzero(pi_BA) compose BC support
direct support   = nonzero(pi_BR) compose BC support
inhibitory       = nonzero(pi_AR) compose AC support
RGC support      = union(direct modes, inhibitory modes)
```

`compose` 是“存在一条固定连接”的布尔复合。local/broad 的 radius 是**有限节点间连接半径**，不是额外接收未建模连续圆盘输入的 aperture。没有自行扩充到无限 H1/BC/AC 平面。反之，也不能只检查 local AC 附近的 BC 坐标，忽略那些 BC 的 H1-feedback 祖先。

| cell | 完整 h_H | 完整 H1 feedback | 完整 BC state/output | 完整 AC：local ON, local OFF, broad ON, broad OFF | 完整 direct modes（2 ports×2 modes） | 完整 inhibitory modes（2×2） | 完整 RGC ports |
|---|---|---|---|---|---|---|---|
| 67#6 | 10/25 | 0/25 | 0/50 | 0/9, 0/9, 0/9, 0/9 | 0/4 | 0/4 | 0/2 |
| 67#7 | 25/25 | 25/25 | 50/50 | 9/9, 9/9, 9/9, 9/9 | 4/4 | 4/4 | 2/2 |
| 67#33 | 25/25 | 25/25 | 50/50 | 9/9, 9/9, 9/9, 9/9 | 4/4 | 4/4 | 2/2 |
| 68#3 | 10/25 | 0/25 | 0/50 | 0/9, 0/9, 0/9, 0/9 | 0/4 | 0/4 | 0/2 |
| 68#10 | 2/25 | 0/25 | 0/50 | 0/9, 0/9, 0/9, 0/9 | 0/4 | 0/4 | 0/2 |
| 69#4 | 25/25 | 25/25 | 50/50 | 9/9, 9/9, 9/9, 9/9 | 4/4 | 4/4 | 2/2 |
| 69#6 | 25/25 | 25/25 | 50/50 | 9/9, 9/9, 9/9, 9/9 | 4/4 | 4/4 | 2/2 |
| 69#7 | 25/25 | 25/25 | 50/50 | 9/9, 9/9, 9/9, 9/9 | 4/4 | 4/4 | 2/2 |
| 70#34 | 25/25 | 25/25 | 50/50 | 9/9, 9/9, 9/9, 9/9 | 4/4 | 4/4 | 2/2 |

67#6 与 68#3 的完整 h_H 节点是局部 x∈{+0.15,+0.30} 的 10 个；68#10 仅局部 (x,y)=(+0.15,-0.30)、(+0.30,-0.30) 两个。其余 h_H 依赖缺失 aperture。G→F_H 的反馈复合使这三个 cell 的全部 BC state/output 都具有至少一个未知输入祖先；进而 local/broad 两类 AC、direct/inhibitory modes 和两个 RGC ports 均不具完整路径支撑。

**完整覆盖的六个 cell：67#7、67#33、69#4、69#6、69#7、70#34。**  
**存在 unknown spatial domain 的三个 cell：67#6、68#3、68#10。**

这不表示未覆盖通路的输出必须为零，或其全部输入均未知；表示在冻结合同下不能声称其完整输入可由该 crop 提供。也不把旧 Canonical 的有限域归一化做法搬入 Population Q。

## 6. 唯一 R1.1 registration contract（设计，未实现）

1. **中心身份固定。** 每个 biological cell 使用本报告对应的冻结 LN final-refit `center_xy` 和 SHA256；采用第 2 节已存在的 float32 degree 转换，再提升 float64，得到常量 `c_i`。同 cell 的全部原 recordings 共用它。禁止从 validation、响应峰值、NLL 最优 shift 或新拟合获取替代中心；center 不是 trainable parameter。
2. **全局 stimulus frame 保持。** 保留真实 pixel values、bounds、pitch、crop/FOV、时间、L/M/Weber、history、warmup 和 split。使用 stimulus 原坐标，不 resize、重新 crop、平移图像或重标 degree/pixel。
3. **整体局部 circuit 平移一次。** 对 `input_xy/h_xy/bc_xy/ac_xy/rgc_xy` 一律设置 `xy_i=xy_local+c_i`，两极性和四个 AC families 使用同一平移；25 Q apertures 随 input nodes 同步移动。RGC target 正好为 `c_i`。节点数、相对位置、边集合和路由权重保持冻结，G/F_H/pi_BR/pi_BA/pi_AR 无须因刚性平移改变；不得只改 RGC 元数据而仍在原点算 Q。
4. **Q 的唯一计算。** 使用实际 pixel bounds 与平移后的 aperture 按上述 area overlap 重算 W；分母始终为原 aperture 面积 0.01 deg²。不得裁掉未知部分后归一化，不以 0/背景填补未知输入，不通过改变中心或缩小节点间距绕过 coverage。
5. **coverage 是实施前置条件。** 六个 cell 的几何条件已满足。三个 cell 在现有 0.91640625° crop 下不满足；现有 `area_integral_weights` 的完整覆盖检查也不会接受它们。**在不改 crop、cohort 和 relative geometry 的约束下，目前不能据此启动一个完整九-cell R1.1 run。** 记录为 coverage 阻塞，不自动丢弃 cell、改 aperture、换输入资产或另设 center；本文不实施补救方案。
6. **其余合同冻结。** LegacyPReLU、tau/delay/gain、hierarchy/prior、loss、joint fit、seed/lr/clip/batch/dtype、inner selection 与 refit schedule、9-cell/16-recording cohort、训练/评价 bins 全部沿用 R1；不把旧 Canonical 的 K 代入 Population，也不利用本次几何结果调参。不同 biological cells 仍是独立 instance。
7. **后续实施时必须保存的配准元数据。** cell/recordings、LN checkpoint hash、原始 center、单位/轴方向、degree 转换精度、固定 c_i、原/平移坐标、pixel bounds、aperture coverage 与每个 pathway validity；声明坐标不在 optimizer 中。这只是接口合同，没有写 trainer、生成新 checkpoint 或运行 R1.1。

## 7. 可复核的原始中心与 hash

原始 center artifact 的共同根目录为 [冻结 LN run](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830)；第 2 节已给出每份 checkpoint 链接。以下数字直接取保存 tensor（degree 值为已保存 fixed-center Canonical 的同值转换），未按结果筛选或改写：

```json
[
  {
    "cell": "67#6",
    "recordings": [
      "lSS01078",
      "lSS01079"
    ],
    "ln_center_xy_pixel": [
      -2.2958531379699707,
      0.9076178073883057
    ],
    "registered_center_deg": [
      -0.12376083433628082,
      -0.04892627149820328
    ]
  },
  {
    "cell": "67#7",
    "recordings": [
      "lSS01086",
      "lSS01087"
    ],
    "ln_center_xy_pixel": [
      1.6148223876953125,
      -0.4886271357536316
    ],
    "registered_center_deg": [
      0.08704902231693268,
      0.026340056210756302
    ]
  },
  {
    "cell": "67#33",
    "recordings": [
      "lSS01159",
      "lSS01160"
    ],
    "ln_center_xy_pixel": [
      -0.036879926919937134,
      -1.82248854637146
    ],
    "registered_center_deg": [
      -0.001988058676943183,
      0.09824352711439133
    ]
  },
  {
    "cell": "68#3",
    "recordings": [
      "lSS01181",
      "lSS01183"
    ],
    "ln_center_xy_pixel": [
      -2.7969863414764404,
      -0.6076110601425171
    ],
    "registered_center_deg": [
      -0.15077504515647888,
      0.03275403380393982
    ]
  },
  {
    "cell": "68#10",
    "recordings": [
      "lSS01221"
    ],
    "ln_center_xy_pixel": [
      -4.570400714874268,
      -6.205354690551758
    ],
    "registered_center_deg": [
      -0.24637316167354584,
      0.33450740575790405
    ]
  },
  {
    "cell": "69#4",
    "recordings": [
      "lSS01254"
    ],
    "ln_center_xy_pixel": [
      1.1302753686904907,
      -0.20724204182624817
    ],
    "registered_center_deg": [
      0.06092890724539757,
      0.011171641759574413
    ]
  },
  {
    "cell": "69#6",
    "recordings": [
      "lSS01256",
      "lSS01257"
    ],
    "ln_center_xy_pixel": [
      0.7037326097488403,
      1.3008372783660889
    ],
    "registered_center_deg": [
      0.03793558478355408,
      -0.07012326270341873
    ]
  },
  {
    "cell": "69#7",
    "recordings": [
      "lSS01258",
      "lSS01259"
    ],
    "ln_center_xy_pixel": [
      1.283694863319397,
      0.7043752074241638
    ],
    "registered_center_deg": [
      0.06919917464256287,
      -0.037970226258039474
    ]
  },
  {
    "cell": "70#34",
    "recordings": [
      "lSS01299",
      "lSS01300"
    ],
    "ln_center_xy_pixel": [
      -0.933394730091095,
      -1.4031856060028076
    ],
    "registered_center_deg": [
      -0.05031580850481987,
      0.0756404772400856
    ]
  }
]
```

| cell | LN ln-trained.pt SHA256 |
|---|---|
| 67#6 | `8c001a97989696651f314c1970b024eee8f188526b79b7cb85b8d6728722a014` |
| 67#7 | `9574ed9c694567a308fa70ac2848ec34ad81732f921677f72c80234cf64ba02a` |
| 67#33 | `2214095516353e591e483d020a0c7726a1c3fdba30199738479487fa1d2c06f4` |
| 68#3 | `547f727d8f54359e16ac42e49b721a17709d7393c10e484d3bb5af4dd5c21bdb` |
| 68#10 | `521e5ea0af1296f890f0fb367dc6a054bf6f823ace15841cb8f8b46cdee7b83c` |
| 69#4 | `a57ba32aa6c5e2bb948479947a979b08888872f3e37ea67f64c276b9d6836bb8` |
| 69#6 | `55c5d91f0647261d7ba6ede81ea482fe8b97aa14f0c941e1c99f94cf906af61a` |
| 69#7 | `5198fb2a42694de36b4df06547dfaec2ee2c5bff5d244c9d9ed6da62e26f1270` |
| 70#34 | `45f85e951ca4af71236b8526ffbe83c1bc1dbabe4bb47ae7f81f06e41c978447` |

Population checkpoint 的逐 cell hashes 保留于原 [CHECKPOINT_LOCK](D:/PythonProject/retina_rf_SNN/output/real_data/retipath_population_r1/CHECKPOINT_LOCK.json)，本次九份全部匹配。旧 fixed-center Canonical 的坐标工件共同位置为 [fixed-center run](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905) 下的 `cells/<slug>/model-trained.pt`。九份保存的 degree centers 全部与上列值相等。

## 8. 边界与停止

本次确认的是 LN functional center 与 Population R1 **指定 retinotopic origin** 的不匹配，并同时确认平移后六个完整、三个覆盖不足。未检验中心的生理准确性，未声称中心等于真实 soma 或解剖 RF center，未证明该不匹配造成 R1 NLL 差异，未预测注册后性能会改善。没有搜索替代原因、没有模型评价或训练。报告写入后停止。

CONFIRMED_REGISTRATION_MISMATCH

