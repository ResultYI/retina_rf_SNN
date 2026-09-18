# H1 external-geometry frozen assay interface

本接口为后续外部定义的 functional assay 做准备。本轮没有读取正式 checkpoints、运行 population assay 或收集生理验证证据。单元测试中的数值仅用于检验坐标与结果结构，不是推荐生理参数。

## 配置与必要外部定义

复制 `configs/h1_external_geometry_assay.template.json` 到新的实验配置，再依据外部文献/实验记录填写；模板中的任何 `REQUIRED` 都会使入口在打开 checkpoint 之前报错。未知字段也会报错，不能以 `bc_support`、`h1_support` 或响应选择规则替代外部几何。

| 字段 | 正式运行前必须确定的内容 |
|---|---|
| `provenance` | geometry、registration、contrast、timing、history、response_metric 的来源及方法；记录文献图表/实验条件和单位转换 |
| `geometry.center_deg` | 刺激中心在当前输入坐标系中的 `[x_deg,y_deg]`；或显式 `{cell_id: [x_deg,y_deg]}` 登记表，不能缺 cell 后退回模型中心 |
| `geometry.spot_radii_deg` | 至少两个严格递增的正半径；也可改用 `spot_diameters_deg`，两者只能提供一个 |
| `geometry.annulus` | 不使用时明确为 `null`；使用时提供 `inner_radius_deg` 和 `outer_radius_deg`，二者均须来自外部定义 |
| `contrast_weber` | 正的 L+M Weber 振幅；外部 luminance/contrast 定义怎样转换到该输入，需写入 provenance，不从训练数据或旧 assay 取值 |
| `polarity` | 显式填写 `[1]`、`[-1]` 或 `[1,-1]`；这是刺激 contrast 符号，不是自动按 cell 的 ON/OFF 翻转 |
| `temporal_ms` | onset 前背景时长、stimulus duration、offset 后 recovery 时长，均以 ms 给出 |
| `history` | 明确选择 `FIX_HISTORY_ZERO` 或 `FIXED_OCCUPANCY`；前者 `occupancy=null`，后者必须提供每 bin 一个 0/1 的完整列表 |
| `measurement.window_ms` | baseline-subtracted logit 的平均响应窗口 `[start,end)`，相对整条 sequence 起点；不是自动沿用旧 late 窗口 |
| `measurement.reference_spot_index` / `large_spot_index` | 配置中 spot 列表的零基索引，须满足 reference < large；不能根据模型响应选 preferred spot |

`sample_rate_hz=150` 是当前冻结模型的工程合同，不是新增的生理默认值。ms 必须可在此采样率上精确表示，否则直接停止，不静默舍入。背景/reference 固定为 0 Weber；真实背景亮度、光谱、适应状态与模型 Weber proxy 的对应仍须在外部定义中说明，接口不能凭空恢复它们。

annulus 的完整结构是 `{"inner_radius_deg": "REQUIRED", "outer_radius_deg": "REQUIRED"}`。填实值后需满足 `reference_radius <= inner_radius < outer_radius`。接口同时构建 annulus-only 和 reference-spot + annulus；两者使用同一 contrast 和时间协议。

## Degree 到 17×17 输入的映射

只从 canonical checkpoint 读取原输入格点坐标 `cone_positions_degs`，保留其展平顺序，不读取模型 support 来构建刺激。所有刺激中心与尺寸来自配置。

- diameter 显式除以 2 转为 radius。
- 对每个实际输入坐标计算 `d = sqrt((x-cx)^2 + (y-cy)^2)`。
- spot 使用 `d <= r`；annulus 使用 `inner < d <= outer`；组合刺激取两 mask 的并集，不在重叠区域叠加 contrast。
- 采用格点中心采样，无插值、抗锯齿、最近点吸附或模型 support 裁剪。保存请求尺寸、实际 mask 索引、格点数、mask hash、格点间距和坐标。
- 输入视野边界按最外格点外半个格距定义。外部圆形范围超出视野则停止，不自动裁剪或扩展输入；小刺激未覆盖任何格点也停止。不同外部半径映射为同一离散 mask 时保留原尺寸并记录 `same_mask_as`，不调整半径。

刺激形状为 `[cases,T,289]`，history 为 `[cases,T,1]`。第一条永远是全零刺激参考；其余为每个指定 contrast 符号的全部 spots 与可选 annulus 条件。每条独立 sequence 由原模型状态初始化。所有空间条件、背景参考和两种模型操作使用同一个显式 history 序列；不会读取任何 natural-movie targets，也不会反馈生成 spikes。

## 冻结推理及输出

新入口是 `evaluation.mechanistic_retina.h1_external_geometry_assay.run_frozen_checkpoint(config_path, checkpoint_path, expected_sha256, output_dir)`。它只处理调用者指定的一个 checkpoint；调用者需从 canonical migration manifest 提供确定的 path/hash。接口没有 cohort 扫描、checkpoint selection、seed selection、训练或自动 population loop。

复用旧 assay 的 canonical checkpoint loader 和已有 `observe_mechanism` / `InterventionSpec`。历史 assay 中构建脉冲的函数固定为 150 bins，不能直接承载外部 duration；因此只在新模块中实现配置化脉冲与几何，保留历史脚本及其所有结果不变。

未来显式调用时，先验证完整配置、checkpoint 身份、时间/格点转换，再在全新输出目录写 `lock.json`，之后才运行 NORMAL 与 BLOCK_H1_FEEDBACK。模型保持 eval、requires_grad=False，采用原 forward 和 no-gradient 推理，并检查状态及 checkpoint 未变。H1 block 的 graph/state 保留、feedback 归零和下游重算完全使用已有 API，不增加其他 pathway 操作。

FIX_HISTORY_ZERO 通过给两次 API 调用和背景参考传入同一全零 occupancy 实现，等价于既有 history 条件；它不是第二种 pathway block。FIXED_OCCUPANCY 同样固定两条件的输入 history，只保留严格过去的原过滤规则。二者都不是 unconditional/free-running prediction。

每个未来单-checkpoint 输出目录包含：

- `lock.json`：完整外部配置及 hash、checkpoint path/hash、cell/seed、原 cell polarity、实际 grid/masks、时间 bins、模型条件及统计定义。先于 response 写入。
- `raw_results.npz`：原始 logits/probability `[2,cases,T,1]`、stimulus、history、case IDs、cell/seed，保留逐 seed 原始结果。
- `results.json`：逐 cell/seed/condition/stimulus 的响应与 normalized size curve，以及两种条件的 paired interactions。不同 seed 不会在该入口自动聚合。

统计定义固定、无隐藏阈值：

1. `R = mean_window(logit_stimulus(t) - logit_zero_stimulus(t))`。背景是同 checkpoint、同模型条件、同 history、同一时刻的参考，不是任意常数 bias。
2. Normalized size-tuning：`R(r) / D`，`D=max_configured_spots |R_NORMAL(r)|`。同 cell/seed/contrast 的两条件共享 D；只归一化输出幅度，不据响应选择或修改任何尺寸。D=0 返回 `null` 与明确状态，不加 epsilon。
3. `S = R_reference - R_surround`；size pair 使用显式 reference/large 索引，可选 annulus pair 使用 reference 与 reference+annulus。
4. `SSI = S / |R_reference|`，NORMAL/BLOCK 分别用自身 reference。零分母返回 `null`；保留负响应的符号，不 clip 到 [0,1]。负 reference 下这是 signed 描述性指标，不宜直接称为发放抑制百分比。
5. `Delta_S_H1 = S_NORMAL - S_BLOCK = Delta_surround - Delta_reference`；另保存 `Delta_SSI_H1`，不把两种不同分母下的指标与 raw interaction 混用。

这些输出公式必须与选定文献的响应变量及指标定义核对；本接口的 logit 不是实测 H1 membrane voltage 或 firing rate。仅准备 external geometry 接口不等于完成 independent physiology validation。本轮不提供验收 verdict，也不根据旧 assay 的 FUNCTIONALLY_CONSISTENT 结果设置任何新尺寸或时序。

## 本轮最小检查

运行 `D:\anaconda\python.exe -B -m pytest tests/test_h1_external_geometry_assay.py -q -p no:cacheprovider`。

本轮结果：**4 passed**。测试只用代码内明确标为 synthetic 的 17×17 toy grid、配置和人工 logits，检查 degree/diameter 映射、annulus、输入索引、时间脉冲、history 数据结构、缺项停止、零分母及 paired 指标。没有加载正式 checkpoint 或运行真实模型 response。正式外部实验仍须另行授权并先填实全部 REQUIRED 字段。
