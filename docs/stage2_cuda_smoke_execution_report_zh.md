# Retina RF SNN 阶段二 CUDA smoke 执行报告

## 1. 状态

`STAGE_2_PASS`

阶段一状态为 `STAGE_1_PASS`，且 `stage_2_smoke_training = GO`。最初的阻断来自选错 Python 解释器：默认环境是 CPU-only PyTorch；项目已有的 `snn_env` 则具备可用 CUDA PyTorch。切换到该解释器后，10-step CUDA smoke 与 checkpoint resume 检查均通过。

## 2. 环境

| 项目 | 值 |
|---|---|
| Python | 3.11.15 |
| executable | `D:\anaconda\envs\snn_env\python.exe` |
| PyTorch | 2.10.0+cu126 |
| CUDA runtime | 12.6 |
| CUDA available | true |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| GPU total memory | 8,585,216,000 bytes |
| GPU free memory before smoke | 7,398,752,256 bytes |
| pytest | selected environment 中未安装；阶段一测试已通过 |

`snn_env` 缺少 PyYAML，因此本次仅在启动包装器中追加已有 base 环境的 site-packages，并确认加载的是纯 Python PyYAML；PyTorch 仍来自 `snn_env`。未安装、升级或卸载任何依赖。

正式数据预检通过：train 12 个 HDF5，validation 4 个 HDF5，`sequence_steps = 320`，cone geometry 为 `(29, 2)`，train/validation source disjoint，`dt_ms = 5.000000000000004`。

## 3. 使用的 smoke 配置

已生成独立配置副本，仅修改：

- `max_optimizer_steps: 6000 -> 10`
- `validation_interval_steps: 100 -> 2`

正式配置未修改。

## 4. 训练结果

CUDA smoke 正常退出，完成 10 个 optimizer steps 和 5 次 validation，约用时 410 秒。

| 指标 | 首次 | 末次/范围 |
|---|---:|---:|
| `loss_total` | 0.41057027131319046 | 0.3279859311878681 |
| `reconstruction` | 0.14799240417778492 | 0.11797929741442204 |
| `gradient_norm` | — | 0.22317355871200562–0.3898410201072693 |
| `temporal_gradient_norm` | — | 0.016859134659171104–0.03303796052932739 |

全部记录为有限值；hard/surrogate energy 最大绝对差为 0。`mean_rate`、ON/OFF active fraction 均保持正值。PyTorch 记录的峰值显存为 20,329,984 bytes，约占 GPU 总显存 0.2368%，未发生 OOM。

## 5. Checkpoint

- 已生成 `checkpoint_last.pt`。
- 已生成 `checkpoint_best_reconstruction.pt`。
- 未生成 `checkpoint_best_feasible.pt`，符合 10-step smoke 尚未通过最终 feasibility gate 的预期。
- resume 正常退出；checkpoint schema 为 `retina_rf_snn`，revision 为 3，恢复到 optimizer step 10。
- resume 前后 `training.jsonl` 均为 5 行，没有重复训练或追加记录。

## 6. Final gate

- representation skill：0.07394690605046006，`representation_passed = false`。
- energy：`not_identifiable`，`target_energy_ratio = null`。
- dynamic RF：`not_run`。
- RGC typing：`not_run`。

这些结果符合仅执行 10-step CUDA smoke 的范围；本阶段验证的是训练链路、日志、显存和恢复能力，不将其解释为正式科学结果。

## 7. 修复记录

仅修改 `scripts/run_experiment.py`，在 `training.jsonl` validation 记录中补齐：

- `gradient_norm`
- `temporal_gradient_norm`
- `peak_memory_bytes`

根因是初次预检使用了默认 CPU-only 解释器，并非机器缺少 CUDA。没有修改训练算法、模型结构或正式配置。

## 8. 风险

- 10-step smoke 不能证明 energy bootstrap/ramp/dual 在更长训练中的收敛性。
- 尚未生成 best-feasible checkpoint。
- 尚未运行正式 dynamic RF 与 RGC typing。
- 尚未运行 6000-step pilot。
- 本次启动依赖现有 base 环境中的纯 Python PyYAML；后续固定运行环境时应显式记录这一依赖来源。

## 9. 下一阶段结论

`stage_3_budget_smoke = GO`

阶段二所需的 CUDA 端到端训练、指标记录、checkpoint 保存与 resume 链路均已验证。应停止在阶段二，不在本轮继续 200-step、6000-step、正式 dynamic RF 或 RGC typing。

测试产物目录：`test_artifacts/stage2_20260722_161857/`
