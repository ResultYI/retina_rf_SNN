# Retina RF SNN 阶段三 energy-budget smoke 执行报告

## 1. 阶段状态

`STAGE_3_PASS`

本轮完成 160 个 optimizer steps、8 次 validation、checkpoint resume 和一次只读 loss-component gradient audit。未执行正式训练、dynamic RF 或 RGC typing。

## 2. 环境与模块来源

| 项目 | 值 |
|---|---|
| Python | `D:\anaconda\envs\snn_env\python.exe`（3.11.15） |
| PyTorch | `2.10.0+cu126`，来自 `snn_env` |
| NumPy | `1.26.4`，来自 `snn_env` |
| h5py | `3.16.0`，来自 `snn_env` |
| PyYAML | `6.0.1`，来自 base site-packages 的纯 Python 实现 |
| CUDA | 12.6，available=true |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| GPU 总显存 | 8,585,216,000 bytes |

启动器只用 `sys.path.append(...)` 追加 base site-packages，因此没有覆盖 `snn_env` 中的 torch、NumPy 或 h5py。未安装、升级或卸载依赖。

## 3. 阶段三配置差异

阶段三使用独立配置副本，仅修改：

- `reconstruction_bootstrap_steps: 1000 -> 40`
- `budget_ramp_end_step: 2500 -> 100`
- `validation_interval_steps: 100 -> 20`
- `minimum_representation_skill: 0.25 -> 2.0`

正式 `configs/experiment.yaml` 未修改。`max_optimizer_steps` 保持 6000，实际运行由 `--stop-after-steps 160` 截止。

## 4. Scheduler horizon 与 stop step

- resolved config horizon：6000 steps。
- 实际停止：optimizer step 160。
- step 160 model LR：`0.00019964928592495045`。
- 正式 6000-step cosine schedule 的理论值：`0.00019964928592495045`。

LR 未在 step 160 衰减到 0，停止参数没有改变 scheduler horizon。

## 5. Runtime 与显存

- 主 run runtime：3534.151 秒，约 58 分 54 秒。
- 峰值 PyTorch allocated memory：20,330,496 bytes，约占总显存 0.2368%。
- exit code：0。
- 未出现 NaN、Inf、CUDA OOM 或 graph reuse error。

## 6. Reconstruction、gradient 与活动

| 指标 | 首次/最小值 | 末次/最大值 |
|---|---:|---:|
| validation MSE | 0.611298680305481 | 0.6183733940124512 |
| representation skill | 0.07363338684205678 | 0.06291231253430107 |
| gradient norm | 0.15966063737869263 | 0.7764072418212891 |
| temporal gradient norm | 0.016651593148708344 | 0.3388592600822449 |

最后 validation MSE 未超过首次的 2 倍。`mean_rate` 最小值为 0.02632712945342064；post-ramp ON/OFF activity 最小值分别为 0.02773100797639927 和 0.016921470776651404，没有全部静默。

## 7. Reference、target 与 current budget

- bootstrap 的 step 20、40：target/current 均为 `None`。
- 固定 reference energy：0.2465886000676684。
- 固定 target budget：0.22192974006090155。
- `target/reference = 0.9`。
- step 60 current budget：0.23836898006541277。
- step 80 current budget：0.23014936006315717。
- step 100 起 current budget 恒等于 target budget。
- 线性 ramp 最大相对误差：0.0。

reference 与 target 在 bootstrap 后均保持不变。

## 8. Violation、penalty 与 dual

- violation 全部非负，最大值 0.5422694981098175。
- penalty 全部非负，最大值 0.3642512522637844。
- dual 始终位于 `[0, 10]`，最终值为 0.019653823798492894。
- 出现 violation 后 dual 明确变化，没有“violation 存在但 dual 不更新”的失败。

训练日志中的 energy 与 violation 分别对 4 个 gradient-accumulation clips 求均值。因此 step 80、100、140 可出现“平均 energy 低于 budget，但平均 per-clip ReLU violation 为正”；这是聚合顺序造成的可解释现象，不是 loss 公式不一致。

## 9. Hard/surrogate energy

8 个 validation 日志点中：

```text
max(abs(hard_energy - surrogate_budget_energy)) = 0.0
```

满足 `<= 1e-7` 的要求。

## 10. Checkpoint 事件

- `initial_reference.pt`：存在。
- `checkpoint_last.pt`：存在，schema revision 3，optimizer step 160。
- `checkpoint_best_reconstruction.pt`：存在；最佳事件发生在 step 20。
- `checkpoint_best_feasible.pt`：不存在。

step 100/120/140/160 的 validation target-energy ratios 分别为 1.7922970052732803、1.758308278231618、1.730085541217931、1.699890209232053，全部大于 1.05。因此不生成 best-feasible checkpoint 的行为正确；ramp 前也没有 best-feasible 事件。

## 11. Resume

- resume exit code：0。
- runtime：9.031 秒。
- checkpoint 恢复 step：160。
- `training.jsonl` 行数：8 -> 8，没有新增 optimizer step 或 validation 行。
- energy state、validation state、sampling/augmentation/CUDA/torch RNG 均可加载。

## 12. Loss-component gradient audit

固定 source `100075.jpg`、augmentation seed 20019，在 checkpoint step 160 上执行；未调用 `optimizer.step()`。

| 分量 | 全参数 gradient norm | 相对 reconstruction |
|---|---:|---:|
| normalized reconstruction | 0.22393342852592468 | 1.0 |
| weighted wiring | 0.0008783607045188546 | 0.003922418864842089 |
| weighted diversity | 0.00035704931360669434 | 0.0015944440093514616 |
| energy penalty | 0.0 | 0.0 |

全部 norm 有限，reconstruction norm 大于 0。wiring 与 diversity ratio 均低于 0.01，属于辅助压力偏弱警告，但不触发 `> 0.25` 的 NO-GO 条件。该固定 audit clip 的 violation 为 0，因此 energy gradient 为 0 与公式一致。

另外已分别记录 RGC phenotype、temporal 和 decoder 参数组的梯度范数，见 `gradient_audit.json`。

## 13. 代码修改

- `scripts/run_experiment.py`：增加 `--stop-after-steps`，并补充 reference/budget/checkpoint-event 日志字段。
- `tests/test_training_contract.py`：增加停止参数保持正式 scheduler horizon 的定向测试。

定向测试完成 red -> green：修改前 1 failed；修改后 1 passed、6 deselected。没有进行模型架构、T-BPTT、正式 LR 或 loss 公式修改。

## 14. 风险

- 160-step smoke 没有达到 energy target，不能作为科学收敛结论。
- wiring/diversity gradient ratio 均低于建议区间 0.01–0.10，进入 pilot 后应重点观察其长期作用；本轮不做权重 sweep。
- 最终科学 evaluation 选用 step-20 best-reconstruction checkpoint，因此 energy status 为 `not_identifiable`；真实 post-ramp ratio 已单独从训练日志记录。
- 日志中的 energy/violation 是对 accumulation clips 分别聚合，不能把“均值 energy”直接代入 ReLU 重算“均值 violation”。

## 15. 下一阶段结论

`stage_4_pilot = GO`

阶段三的 budget bootstrap、target 固定、线性 ramp、dual 更新、hard/surrogate 一致性、checkpoint 资格、resume 和 gradient audit 均通过机械门禁。辅助项偏弱作为 pilot 风险保留，不在本轮自动进入阶段四。

产物目录：`test_artifacts/stage3_20260722_165524/`
