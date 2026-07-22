# Retina RF SNN 代码主线静态重构执行报告

## 1. 执行边界

- 未运行测试。
- 未运行训练。
- 未运行评估。
- 未执行 Git 命令。
- 未启动任何 Python 程序。
- 未修改或删除 `data/**/*.h5`、`data/isetbio_bsds300_4deg/`、`runs/`、`checkpoints/`、外部 HumRet 数据或 ISETBio/MATLAB 安装。
- 完成检查仅包括文件阅读、源码编辑、目录清理和 `rg` 纯文本搜索。

## 2. 删除文件

共删除 98 个文件，其中 84 个源码、配置、测试或文档文件，另有 5 个预览 PNG 和 9 个 Python 缓存文件。

### 模型与训练

- `models/v9_retina.py`：双匿名 bank 模型已迁移为 canonical 单 pool。
- `models/cells/rgc_connectivity.py`：只服务旧双 population 稀疏连接。
- `training/v9.py`：数据、augmentation、长反向传播和 checkpoint 逻辑已拆入 canonical 模块。
- `training/stage1.py`
- `training/stage1_types.py`
- `training/stage1_runtime.py`
- `training/stage1_reporting.py`
- `training/stage1_preflight.py`
- `training/stage1_checkpoint.py`
- `training/hybrid.py`
- `training/epoch_metrics.py`

上述旧训练文件依赖阶段切换、decoder warmup、显式 population 或旧 checkpoint contract，均由 `training/trainer.py`、`training/config.py`、`training/data.py` 和 `training/checkpointing.py` 取代。

### 评估

- `evaluation/v9.py`
- `evaluation/temporal_probes.py`
- `evaluation/rf_probe.py`
- `evaluation/rf_identifiability.py`
- `evaluation/rf_agreement.py`
- `evaluation/reconstruction_baselines.py`
- `evaluation/population_ablation.py`
- `evaluation/functional.py`
- `evaluation/feasibility.py`
- `evaluation/dynamics.py`
- `evaluation/checkpoint_tensors.py`
- `evaluation/checkpoint_runner.py`
- `evaluation/checkpoint_probes.py`
- `evaluation/checkpoint_payloads.py`
- `evaluation/checkpoint_metrics.py`
- `evaluation/checkpoint_loaders.py`
- `evaluation/checkpoint_contracts.py`
- `evaluation/checkpoint_context.py`

这些文件只服务旧 checkpoint runner、旧 population contract 或重复的 RF/reconstruction 流程。通用职责已收敛到 `evaluation/reconstruction.py`、`evaluation/dynamic_rf.py`、`evaluation/rgc_types.py` 和 `evaluation/reporting.py`。

### 入口与配置

- `scripts/train_stage1.py`
- `scripts/evaluate_checkpoint.py`
- `scripts/run_v9_emergent_rgc_dynamic_rf.py`
- `scripts/audit_full_pipeline_shapes.py`
- `scripts/isetbio_stage1.py`
- `scripts/generate_isetbio_h5.py`：该文件仅转发到已删除的旧入口，未保留兼容壳。
- `configs/v9_emergent_rgc_dynamic_rf.yaml`

### 测试

- `tests/test_v9_emergent_rgc.py`
- `tests/test_train_stage1_cli.py`
- `tests/test_stage1_schedule.py`
- `tests/test_stage1_runtime.py`
- `tests/test_stage1_reporting.py`
- `tests/test_stage1_entrypoints.py`
- `tests/test_rgc_cell.py`
- `tests/test_rf_identifiability.py`
- `tests/test_retina_snn.py`
- `tests/test_reconstruction_baselines.py`
- `tests/test_physiology_profiles.py`
- `tests/test_p0_p1_protocol.py`
- `tests/test_natural_motion_sequences.py`
- `tests/test_local_decoder.py`
- `tests/test_isetbio_generation_wrapper.py`
- `tests/test_isetbio_data_contract.py`
- `tests/test_hybrid_training.py`
- `tests/test_humret_evaluation.py`
- `tests/test_humret_checkpoint_payload.py`
- `tests/test_horizontal_cell.py`
- `tests/test_evaluation_protocol.py`
- `tests/test_epoch_metrics.py`
- `tests/test_dataset_interfaces.py`
- `tests/test_context_history_gate.py`
- `tests/test_cone_response_io.py`
- `tests/test_checkpoint_probes.py`
- `tests/test_checkpoint_evaluation.py`
- `tests/test_checkpoint_context.py`
- `tests/test_bipolar_nonlinearity.py`
- `tests/test_bipolar_cell.py`
- `tests/test_amacrine_cell.py`
- `tests/checkpoint_evaluation_fixtures.py`

旧测试绑定已删除接口，已由三个 canonical contract 测试文件取代。本次未运行新测试。

### 数据集与文档

- `datasets/rgc_response_dataset.py`：无活动调用者，且不属于 ISETBio cone-response 训练合同。
- `docs/current_pipeline_status_and_codex_handoff_zh.md`
- `docs/current_pipeline_status_and_codex_handoff_zh.md.orig`
- `docs/shared_energy_decoder_calibration_v8.md`
- `docs/retina_snn_full_pipeline_report.md`

参数证据、HDF5 contract 和历史执行报告保留。

### 架构预览目录

已删除 `retina_architecture_previews/` 下全部文件：

- `__init__.py`、`test_previews.py`、`primitives.py`、`design.py`、`build_previews.py`、`DESIGN.md`
- `page1.py`、`page2.py`、`page3.py`、`page4.py`、`page5.py`
- `output/page1_isetbio_photoreceptor_input.png`
- `output/page2_outer_retina.png`
- `output/page3_amacrine_functional_role.png`
- `output/page4_rgc_populations.png`
- `output/page5_overall_overview.png`
- `__pycache__/design.cpython-313.pyc`
- `__pycache__/page1.cpython-313.pyc`
- `__pycache__/page2.cpython-313.pyc`
- `__pycache__/page3.cpython-313.pyc`
- `__pycache__/page4.cpython-313.pyc`
- `__pycache__/page5.cpython-313.pyc`
- `__pycache__/primitives.cpython-313.pyc`
- `__pycache__/test_previews.cpython-313-pytest-9.1.1.pyc`
- `__pycache__/__init__.cpython-313.pyc`

删除后的预览文件不可由本次工作区操作恢复；需要时应从用户自己的历史副本恢复。

## 3. 新建文件

共新建 16 个文件：

- `configs/experiment.yaml`：唯一活动实验配置。
- `training/config.py`：严格嵌套 dataclass 配置读取、未知键拒绝和跨字段校验。
- `training/data.py`：train-only normalization、source-disjoint 检查、配置驱动 augmentation 和固定验证 clips。
- `training/checkpointing.py`：schema revision 1 的唯一 checkpoint 读写。
- `training/trainer.py`：joint training、长时间信用分配、activation checkpoint 和 energy schedule。
- `evaluation/reconstruction.py`：train baseline scale 与 representation skill。
- `evaluation/rgc_types.py`：per-unit 特征和无外部依赖的二类 k-means。
- `evaluation/dynamic_rf.py`：matched-context 连续 readout RF、finite difference、reset 和 recovery。
- `evaluation/reporting.py`：门控汇总与 JSON/Markdown 输出。
- `scripts/run_experiment.py`：唯一实验入口。
- `tests/test_retina_model.py`：模型形状、per-unit 参数和 tied decoder contract。
- `tests/test_training_contract.py`：时间、严格配置、energy bootstrap 和 checkpoint schema contract。
- `tests/test_dynamic_rf.py`：同源 context 与 identical probe contract。
- `docs/architecture.md`：当前 canonical 架构。
- `docs/experiment_contract.md`：数据、时间、schedule、评估和兼容边界。
- `docs/code_refactor_execution_report_zh.md`：本报告。

## 4. 修改文件

共修改 17 个既有文件。

| 文件 | 主要类/函数 | 旧职责 | 新职责与关键接口变化 |
|---|---|---|---|
| `README.md` | 文档入口 | 并行训练/评估命令 | 唯一 config、runner、单 pool、tied decoder、连续 RF 和不兼容边界 |
| `configs/physiology_profiles.py` | `PhysiologyProfile`, `human_macaque` | 同时携带旧 RGC/decoder population 配置 | 只提供 H1、bipolar、amacrine 前端生理配置 |
| `models/cells/rgc_types.py` | `RGCConfig`, `RGCState`, `RGCOutput` | 双 population tensor/dataclass | 单 pool state 与四种连续/离散输出 |
| `models/cells/rgc_runtime.py` | bounded/state helpers | 双 population LIF runtime | canonical bounded 参数和单 state flatten/detach helper |
| `models/cells/rgc.py` | `HeterogeneousRGCPool` | 显式双 population 稀疏 pool | dense masked softmax、per-unit 参数、unit-level subunit adaptation |
| `models/retina_snn.py` | `RetinaModel`, `build_retina_model` | 双 population sequence stack | 唯一模型入口、sequence-level spatial-weight 复用、canonical state flatten |
| `models/decoder/local_decoder.py` | `TiedLocalDecoder` | 两套独立 local projection | encoder-transpose 权重、per-unit gain、per-cone bias |
| `loss/retina.py` | `RetinaObjective`, `RetinaLosses` | reconstruction、population energy/homeostasis | 统一 reconstruction、inequality energy、wiring、variance、repulsion、homeostasis |
| `training/__init__.py` | canonical exports | 导出 hybrid/stage trainer | 导出 config、data 和 `RetinaTrainer` |
| `loss/__init__.py` | canonical exports | 导出旧 loss config | 导出唯一 objective/loss contract |
| `models/cells/__init__.py` | canonical exports | 导出 population types | 导出 heterogeneous pool 和单 pool types |
| `models/decoder/__init__.py` | canonical exports | 导出旧 projection decoder | 只导出 tied decoder |
| `evaluation/__init__.py` | canonical exports | 导出 population ablation/RF probe | 导出 reconstruction、dynamic RF 和 RGC typing |
| `evaluation/parameter_audit.py` | `audit_parameters` | 审计双 population、旧 decoder | 审计 per-unit RGC、固定 rate tau 和 tied decoder |
| `datasets/retina_training_batch.py` | `RetinaTrainingBatch` | 依赖旧 hybrid targets | 自包含的 input/current batch contract |
| `datasets/isetbio_h5_dataset.py` | `ISETBioH5Dataset` | 通过旧 training type | 使用自包含 canonical batch type |
| `datasets/__init__.py` | dataset exports | 导出未使用 RGC response dataset | 保留 ISETBio 与原始自然图像生成所需 exports |

## 5. 架构变化

- 双 bank 改为单一无标签异质 RGC pool。
- bank-level 参数改为 per-unit 参数，形状均为 `[Nunit]`。
- learnable rate tau 改为固定共享 `readout_rate_tau_ms` buffer。
- 全局 decoder magnitude 改为 `raw_unit_gain[polarity,unit]` 与 `cone_bias[cone]`。
- decoder 空间映射直接复用 encoder dense masked-softmax 权重。
- cone-level 重复 subunit state 改为 unit-level `[batch,polarity,kinetics,unit]`。
- hard-rate RF 改为 generator probability/generator potential 连续 RF。
- invalid zero-state duplicate control 改为 `D_normal`、`D_reset` 和 suppression。
- 阶段训练改为从第一步 joint training；block boundary 只做 activation checkpoint，不 detach state。

## 6. 删除的兼容性

- 不再接受任何旧 checkpoint schema；只接受 `retina_rf_snn` revision 1。
- 删除旧训练 CLI、旧 checkpoint evaluation CLI 和版本化 runner。
- 删除所有指向旧模型、trainer、evaluation runner 与 population tensor 的 import。
- 不提供 forwarding wrapper、deprecated alias 或转换器。
- 旧 checkpoint 的 state 轴、RGC 参数、decoder、objective、optimizer group 和 dual state 均与当前 contract 不同，因此用户需要重新训练。

## 7. 静态风险

由于未运行代码，以下风险尚未通过运行时证据排除：

- import error：第三方环境、可选 HDF5/HumRet 路径或包初始化顺序可能暴露遗漏。
- shape mismatch：`[batch,time,polarity,unit]` 与前端 `[batch,polarity,kinetics,cone]` 的轴转换可能有错误。
- config mismatch：严格类型读取目前依赖 PyYAML 的值类型，旧命令行覆盖方式不再存在。
- checkpoint serialization issue：scheduler、CUDA RNG 或 generator state 的实际运行时恢复尚未验证。
- activation checkpoint closure issue：八个 state tensors 与四个输出 tensors 的顺序尚未通过反向传播执行验证。
- evaluation tensor-axis issue：continuous Jacobian、recovery、RGC feature aggregation 和 tied reconstruction 的轴尚未用真实数据验证。

## 8. 后续人工执行建议

先运行测试：

```powershell
python -m pytest -q
```

测试通过后再启动训练：

```powershell
python scripts/run_experiment.py --config configs/experiment.yaml --device cuda
```

本次未执行上述命令。
