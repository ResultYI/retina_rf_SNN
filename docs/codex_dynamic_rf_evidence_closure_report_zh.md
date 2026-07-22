# 动态感受野证据闭环静态实施报告

## 结论

本次已按 `retina_rf_snn` checkpoint revision 3 的新契约完成静态代码闭环。修改范围仅包含源代码、配置、测试定义与文档；未运行 pytest、训练、验证、评估、CUDA smoke test、benchmark、数据生成或报告生成，也未改写现有 HDF5、run 与 checkpoint 产物。

当前实现把“训练后存在上下文依赖”改成了更严格的“训练后效应相对精确初始化参考是否增加”的配对证据。动态 RF 和 RGC 功能分型都使用同一批输入、同一选择方案和同一判据比较 initialized 与 trained 模型，避免把架构先验或初始化异质性误报为学习结果。

## 已完成的契约

- 时间探针覆盖每个 RGC unit 的中心 cone，分别施加正负 impulse、step 与按 `dt_ms` 换算半周期的 4 Hz 方波 flicker；所有形状特征来自同一 preferred polarity，并输出有效响应掩码及 hard spike 质量计数。
- `evaluation/__init__.py` 保持空白，runner 直接导入具体子模块，避免评估包导入时的级联副作用。
- 能源预算在 bootstrap 内更新 EMA reference；bootstrap 后 reference 与 target 固定，只线性推进 current budget。训练损失和 dual 使用 current budget，最终 gate 使用 hard validation energy / target budget。
- feasible validation checkpoint 仅允许在 budget ramp 完成后产生；target 不存在时报告 `not_identifiable`。
- checkpoint schema 升级到 revision 3，revision 2 明确不兼容且不提供转换器。
- runner 在 optimizer 构造与 resume 恢复前创建或验证 `initial_reference.pt`，其中包含 schema、resolved config 与 CPU 克隆的初始模型状态。
- 动态 RF 的 unit selection 只由 trained 模型生成一次，并由 initialized/trained 两个模型共享；每个 source 的 low/high/reset/delayed state 只构造一次并复用，有限差分 forward 位于 `torch.no_grad()`，连续 Jacobian 路径保留 autograd。
- 动态 RF record 按有限差分局部性、相对误差、kernel norm 与 reset error 过滤；recovery 以最大 delay 的距离相对 `D(0)` 归一化。先在 source 内取中位数，再进行 paired source bootstrap。
- 动态 RF 状态限定为 `learned_dynamic_rf_supported`、`learned_gain_only`、`architecture_induced_context_dependence`、`not_supported`、`not_identifiable` 或 `not_run`。
- RGC 聚类仅使用有效空间半径、impulse time-to-peak、impulse width、step sustained index 与 normalized flicker response。无效 unit assignment 为 `-1`，内部动力学参数、rate、activity 与原始 amplitude 不进入主聚类空间。
- RGC 报告同时计算 initialized/trained silhouette 与 cluster separation；只有训练后 separation 达到相对或绝对增益门槛才报告 `learned_functional_pairing_candidate`。
- reconstruction scale 改由固定 seed 的 augmented clean targets 拟合。
- 顶层 `model.debug_checks` 传递至 H1、Bipolar、Amacrine 与 RGC；默认实验关闭逐步有限值检查，optimizer 端非有限 loss 防护保留。
- 参数审计覆盖 H1、Bipolar、Amacrine、RGC 与 Decoder，并输出初始化/最终范围、绝对变化及边界接近比例。

## 主要文件

- `training/trainer.py`：固定 target budget、validation eligibility 与新日志字段。
- `training/checkpointing.py`：revision 3 兼容性边界。
- `scripts/run_experiment.py`：初始化参考、双模型配对评估、最终 gate 与参数审计编排。
- `evaluation/temporal_probes.py`：逐 unit、双极性的完整时间探针。
- `evaluation/dynamic_rf.py`：共享 selection、状态缓存、Jacobian 与有限差分。
- `evaluation/dynamic_rf_summary.py`：source-first 聚合与 paired bootstrap 判定。
- `evaluation/rgc_types.py`：纯功能特征聚类与 initialized/trained separation gate。
- `evaluation/reconstruction.py`：确定性 augmented target scale。
- `evaluation/reporting.py`：新状态、配对证据与 `not_identifiable` 能源语义。
- `evaluation/parameter_audit.py`：初始化到训练后参数变化审计。
- `configs/experiment.yaml`：动态 RF、RGC 判据与 `debug_checks` 的唯一活动配置。

## 静态验证边界

本次只新增和更新测试定义，没有执行它们。新增契约测试覆盖 revision 2 拒绝、固定 target、时间探针字段、纯功能 RGC feature list、augmented scale seed 确定性、共享选择对象以及 source-paired 动态 RF 汇总。运行时数值结论仍需后续由用户授权的正式测试或实验产生；本报告不声称训练结果、显存、速度或生理证据已经通过运行验证。
