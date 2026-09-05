Current-Checkpoint Applicability Audit — 2026-08-31

1. **Overall current-result applicability：PASS**

本结论仅覆盖指定 22 个 final checkpoints 的四类已知 architecture 问题。22/22 identity、实际掩码、effective support、N=1 parameterization 和 legacy applicability 均通过。未修改 production source、checkpoint 或既有 artifacts；未训练、未重生成 checkpoint，未运行 illusion 或 baseline。

2. **22-cell checkpoint identity table**

所有行均直接读取实际 `model-trained.pt`，使用原样保存的 `model_config`（不覆盖 architecture mode）构建 production model，并 `strict=True` 加载。
共同 identity：schema=`schottdorf_canonical_v1_shared_bc_development`，revision=4，stage=`trained`；causal config/state=`h1-shared-bc-direct-broad-ac`；spatial config/state=`bc-central-disk_ac-overlapping-full-disk`。
causal identity bytes SHA256=`3643d2317f6ef05afacf9b635acb027b9467f1d770935e2d514be7fd8a4d2a07`。表中 checkpoint SHA256 缩写为前12位，完整 SHA256、config 与 tensor identity 见 JSON/CSV。

空间几何来源：checkpoint 实际坐标与 `data/schottdorf_lee_2021.py::_cone_positions` 按已存 adapter config 生成的17×17格点逐位相同，289个输入点，cell center=(0,0)°。加载后的 `spatial_basis/path_spatial_basis/BC support/AC support` 与默认 geometry 构建逐位相同，无自定义 geometry。
R_BC/R_AC 不是 checkpoint config 中独立保存的字段；它们由冻结 production `support_partition.py` 按实际 cell type 固定：midget=0.06/0.13°，parasol=0.10/0.15°。当前源码与训练时 manifest 一致。

| Cell | N | Architecture mode | Schema / causal | Geometry | R_BC / R_AC (°) | Graph nodes / edges | SHA256 (12) | Final checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 67#4 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.06 / 0.13 | 1 / 1 | f9ab8103f00d | [67_4/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_4/model-trained.pt) |
| 67#6 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.10 / 0.15 | 1 / 1 | 02f6b00debb4 | [67_6/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_6/model-trained.pt) |
| 67#7 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.10 / 0.15 | 1 / 1 | d9e78dbed900 | [67_7/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_7/model-trained.pt) |
| 67#14 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.06 / 0.13 | 1 / 1 | 16255822b90a | [67_14/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_14/model-trained.pt) |
| 67#21 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.06 / 0.13 | 1 / 1 | b0fe33138757 | [67_21/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_21/model-trained.pt) |
| 67#26 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.06 / 0.13 | 1 / 1 | fb3e3c5227ac | [67_26/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_26/model-trained.pt) |
| 67#33 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.10 / 0.15 | 1 / 1 | a519568baef8 | [67_33/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_33/model-trained.pt) |
| 67#34 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.06 / 0.13 | 1 / 1 | 4ec9c3f8bbe1 | [67_34/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_34/model-trained.pt) |
| 68#3 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.10 / 0.15 | 1 / 1 | d7cd8f67ebd8 | [68_3/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_3/model-trained.pt) |
| 68#4 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.06 / 0.13 | 1 / 1 | 0fc06f738cec | [68_4/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_4/model-trained.pt) |
| 68#7 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.06 / 0.13 | 1 / 1 | 479abfdf9dc3 | [68_7/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_7/model-trained.pt) |
| 68#10 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.10 / 0.15 | 1 / 1 | c394ffbd7c30 | [68_10/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_10/model-trained.pt) |
| 68#11 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.06 / 0.13 | 1 / 1 | ae5ea08ce414 | [68_11/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_11/model-trained.pt) |
| 69#3 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.06 / 0.13 | 1 / 1 | 50b334692cbf | [69_3/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_3/model-trained.pt) |
| 69#4 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.10 / 0.15 | 1 / 1 | b40878f31dca | [69_4/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_4/model-trained.pt) |
| 69#6 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.10 / 0.15 | 1 / 1 | 336e3d8e2de8 | [69_6/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_6/model-trained.pt) |
| 69#7 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.10 / 0.15 | 1 / 1 | e2c234c61343 | [69_7/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_7/model-trained.pt) |
| 69#21 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.06 / 0.13 | 1 / 1 | 3a8e2d91dc71 | [69_21/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_21/model-trained.pt) |
| 70#1 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.06 / 0.13 | 1 / 1 | e4d15d07d2ac | [70_1/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_1/model-trained.pt) |
| 70#7 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.06 / 0.13 | 1 / 1 | f58a9ed77ca1 | [70_7/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_7/model-trained.pt) |
| 70#15 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.06 / 0.13 | 1 / 1 | 2f6178198ae6 | [70_15/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_15/model-trained.pt) |
| 70#34 | 1 | mechanism_identifiable | PASS | 默认 17×17 | 0.10 / 0.15 | 1 / 1 | ae01dfe02d83 | [70_34/model-trained.pt](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_34/model-trained.pt) |

22/22 均 N=1，22/22 均当前 shared-BC causal architecture；没有实际进入 legacy mode 的 cell。所有 checkpoint SHA256 亦与既存 comparison 中引用的 final checkpoint hash 一致。

3. **22-cell actual-mask vs expected-full-disk table**

Expected 使用 checkpoint 实际坐标，在 float64 下独立计算 `hypot(dx,dy) ≤ R`。Actual 来自本次真实 production forward 返回的 `basis_kernels()` 的非零空间支持，同时与加载后的 `path_spatial_basis` 和存储 support 逐元素核对。这里的 full disk 指覆盖当前离散 cone grid 上所有满足 d≤R 的点。

| Cell | Cones | BC expected / actual | AC expected / actual | Mismatch BC / AC | Max radius BC / AC (°) | Holes BC / AC | BC ⊂ AC | AC full disk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 67#4 | 289 | 5 / 5 | 21 / 21 | 0 / 0 | 0.053906 / 0.120538 | 0 / 0 | 是 | 是 |
| 67#6 | 289 | 9 / 9 | 21 / 21 | 0 / 0 | 0.076235 / 0.120538 | 0 / 0 | 是 | 是 |
| 67#7 | 289 | 9 / 9 | 21 / 21 | 0 / 0 | 0.076235 / 0.120538 | 0 / 0 | 是 | 是 |
| 67#14 | 289 | 5 / 5 | 21 / 21 | 0 / 0 | 0.053906 / 0.120538 | 0 / 0 | 是 | 是 |
| 67#21 | 289 | 5 / 5 | 21 / 21 | 0 / 0 | 0.053906 / 0.120538 | 0 / 0 | 是 | 是 |
| 67#26 | 289 | 5 / 5 | 21 / 21 | 0 / 0 | 0.053906 / 0.120538 | 0 / 0 | 是 | 是 |
| 67#33 | 289 | 9 / 9 | 21 / 21 | 0 / 0 | 0.076235 / 0.120538 | 0 / 0 | 是 | 是 |
| 67#34 | 289 | 5 / 5 | 21 / 21 | 0 / 0 | 0.053906 / 0.120538 | 0 / 0 | 是 | 是 |
| 68#3 | 289 | 9 / 9 | 21 / 21 | 0 / 0 | 0.076235 / 0.120538 | 0 / 0 | 是 | 是 |
| 68#4 | 289 | 5 / 5 | 21 / 21 | 0 / 0 | 0.053906 / 0.120538 | 0 / 0 | 是 | 是 |
| 68#7 | 289 | 5 / 5 | 21 / 21 | 0 / 0 | 0.053906 / 0.120538 | 0 / 0 | 是 | 是 |
| 68#10 | 289 | 9 / 9 | 21 / 21 | 0 / 0 | 0.076235 / 0.120538 | 0 / 0 | 是 | 是 |
| 68#11 | 289 | 5 / 5 | 21 / 21 | 0 / 0 | 0.053906 / 0.120538 | 0 / 0 | 是 | 是 |
| 69#3 | 289 | 5 / 5 | 21 / 21 | 0 / 0 | 0.053906 / 0.120538 | 0 / 0 | 是 | 是 |
| 69#4 | 289 | 9 / 9 | 21 / 21 | 0 / 0 | 0.076235 / 0.120538 | 0 / 0 | 是 | 是 |
| 69#6 | 289 | 9 / 9 | 21 / 21 | 0 / 0 | 0.076235 / 0.120538 | 0 / 0 | 是 | 是 |
| 69#7 | 289 | 9 / 9 | 21 / 21 | 0 / 0 | 0.076235 / 0.120538 | 0 / 0 | 是 | 是 |
| 69#21 | 289 | 5 / 5 | 21 / 21 | 0 / 0 | 0.053906 / 0.120538 | 0 / 0 | 是 | 是 |
| 70#1 | 289 | 5 / 5 | 21 / 21 | 0 / 0 | 0.053906 / 0.120538 | 0 / 0 | 是 | 是 |
| 70#7 | 289 | 5 / 5 | 21 / 21 | 0 / 0 | 0.053906 / 0.120538 | 0 / 0 | 是 | 是 |
| 70#15 | 289 | 5 / 5 | 21 / 21 | 0 / 0 | 0.053906 / 0.120538 | 0 / 0 | 是 | 是 |
| 70#34 | 289 | 9 / 9 | 21 / 21 | 0 / 0 | 0.076235 / 0.120538 | 0 / 0 | 是 | 是 |

4. **22-cell effective-support derivative table**

CPU PyTorch 2.6.0+cpu，production float32，2 threads。预先固定 outside absolute tolerance=1e−12，未依据结果调整。
确定性审计输入 [1,32,289]，history全零；在实际 forward 中捕获 X_H1。对 t=15、31 的 sustained/transient 两分量分别求导，共176次VJP；对输入时点、输出分量和选定输出时点取绝对值最大值，不先合并有符号导数。当前 BC 对 X_H1 的路径为线性时不变、lag=16、operator禁用；t=15覆盖完整16 lag，t=31复核平移后的支持。表中 disk内最大值为相同聚合；每一次VJP均通过disk内非零检查，所有 expected disk内 cone 均至少一次具有非零依赖。

| Cell | Outside BC max abs(∂BC_direct/∂X_H1) | Outside AC max abs(∂BC_broad/∂X_H1) | Inside BC max | Inside AC max | PASS |
| --- | --- | --- | --- | --- | --- |
| 67#4 | 0 | 0 | 1.24976292e-1 | 6.74386770e-2 | PASS |
| 67#6 | 0 | 0 | 6.89356849e-2 | 3.66988778e-2 | PASS |
| 67#7 | 0 | 0 | 7.82529861e-2 | 4.16790843e-2 | PASS |
| 67#14 | 0 | 0 | 1.29061639e-1 | 6.60802200e-2 | PASS |
| 67#21 | 0 | 0 | 1.28517762e-1 | 7.10701644e-2 | PASS |
| 67#26 | 0 | 0 | 1.37104392e-1 | 7.49269798e-2 | PASS |
| 67#33 | 0 | 0 | 8.00577626e-2 | 4.47689667e-2 | PASS |
| 67#34 | 0 | 0 | 1.52821094e-1 | 8.36535841e-2 | PASS |
| 68#3 | 0 | 0 | 7.00480938e-2 | 3.69243547e-2 | PASS |
| 68#4 | 0 | 0 | 8.13584253e-2 | 4.46872413e-2 | PASS |
| 68#7 | 0 | 0 | 9.49005634e-2 | 4.71749566e-2 | PASS |
| 68#10 | 0 | 0 | 4.75571416e-2 | 2.49100607e-2 | PASS |
| 68#11 | 0 | 0 | 1.34011209e-1 | 6.93704039e-2 | PASS |
| 69#3 | 0 | 0 | 1.54835850e-1 | 8.83842334e-2 | PASS |
| 69#4 | 0 | 0 | 6.25690967e-2 | 3.29109766e-2 | PASS |
| 69#6 | 0 | 0 | 7.15552345e-2 | 3.83473113e-2 | PASS |
| 69#7 | 0 | 0 | 7.49972239e-2 | 3.96582931e-2 | PASS |
| 69#21 | 0 | 0 | 1.21397294e-1 | 6.67197704e-2 | PASS |
| 70#1 | 0 | 0 | 9.85817313e-2 | 5.52286953e-2 | PASS |
| 70#7 | 0 | 0 | 1.32522359e-1 | 6.74919859e-2 | PASS |
| 70#15 | 0 | 0 | 7.96426460e-2 | 3.83835360e-2 | PASS |
| 70#34 | 0 | 0 | 6.47649691e-2 | 3.41142975e-2 | PASS |

5. **N=1 shared-subunit parameterization：PASS**

22/22 的 `shared_subunits.raw_connections` 为 named buffer，不是 named parameter，requires_grad=False；原值=[0.5413248538970947]，edge_index=[[0],[0]]，connection_matrix=[[1.0]]。
逐模型实际构建 `build_phase1_optimizer`，raw_connections 的 param-group membership 均为0（14个训练参数张量；未执行任何 optimizer step）。
在各自 checkpoint 加载出的内存模型上分别加−0.2、+0.2、+2，再还原；66次 production logit 比较全部逐位相同、max absolute difference=0，每次恢复后完整 state_dict 与 checkpoint 逐 tensor 一致。
final checkpoint 未保存 optimizer state；这里直接验证的是当前 production 构建的 optimizer membership，不能声称已恢复训练当时的 optimizer snapshot。

6. **Legacy-mode applicability：PASS**

逐cell实际 forward 中，同一个 BipolarSubunits 实例的同一 raw_weights 参数对象被调用两次；共享 feature_bank.raw_tau=[2,3]、raw_delay=[2]。AC 的实际输入对象就是输出中的 BC_broad；AmacrinePathways 的参数仅 raw_tau、raw_delay。无旧 independent stimulus→AC encoder 参数或实际调用；operator.depthwise 未进入 forward。
每cell均观察到非零 AC→BC_broad 导数；AC stimulus-gradient 与经 BC_broad 链式回传的结果逐位一致，残差为0。此链式检查与输入对象、参数清单和生产源码追踪共同使用。

| Cell | Mixer buffer / requires_grad | Optimizer occurrences | max abs(Δlogit) (−0.2,+0.2,+2) | Shared BC identity | AC input = BC_broad | Independent AC encoder | Legacy verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 67#4 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 67#6 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 67#7 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 67#14 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 67#21 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 67#26 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 67#33 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 67#34 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 68#3 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 68#4 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 68#7 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 68#10 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 68#11 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 69#3 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 69#4 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 69#6 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 69#7 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 69#21 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 70#1 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 70#7 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 70#15 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |
| 70#34 | 是 / False | 0 | 0 / 0 / 0 | PASS | PASS | 不存在 | PASS |

7. **四项 architecture FAIL 的 current-result impact classification**

| Architecture issue | Current-result impact | 直接证据 |
| --- | --- | --- |
| A. cross-cell support widening | NOT_REACHED_BY_CURRENT_22 | 22/22 graph N=1、E=1、M=[[1]]；所有 outside-support VJP 为精确零。 |
| B. irregular/full-disk geometry enforcement | NOT_REACHED_BY_CURRENT_22 | 22/22 checkpoint 空间基与默认构建逐位一致；真实 forward 核支持与独立 d≤R 掩码逐元素一致；无内部孔洞。 |
| C. self-only trainable structural-null connection | NOT_REACHED_BY_CURRENT_22 | 22/22 raw_connections 为 buffer、requires_grad=False，当前 optimizer 不包含；66/66 内存扰动后 logit 逐位不变。 |
| D. legacy-mode reachability | NOT_REACHED_BY_CURRENT_22 | 22/22 原始 mode 为 mechanism_identifiable；strict load 和共享 BC 参数/AC 输入的真实 forward 追踪通过。 |

8. **受影响 cells / tensors / results**

本次四项检查未发现受影响 cell 或 tensor/contract；未据此要求任何现有结果重新训练或重算。

9. **全局 architecture audit 的状态保持不变**

Architecture Conformance Audit 的全局软件合同仍然 FAIL，但这四类已知问题在当前 22-cell shared-BC checkpoints 的实际 production path 中未被触发。

10. **Audit artifacts 与 source/result integrity**

审计前后：262个 production/data/evaluation/training 源文件、22个 final checkpoints、该 lineage 全部213个既有 artifacts 的 SHA256 字典完全相同；既有 tracked git status 不变。另核对263个训练 manifest 引用的 Python 文件（含 run.py）与当前文件 SHA256 全部一致。完整模型 state 在所有审计 forward 后逐位未变，未残留 parameter.grad。

审计目录：`D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/`。

本目录仅新增审计文件，用户已明确授权落盘。证据文件：

- [22-cell checkpoint identity 与完整 SHA256](D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/checkpoint-identity.csv)
- [22-cell 实际/预期掩码](D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/actual-mask-vs-full-disk.csv)
- [22-cell 导数最大值](D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/effective-support-derivatives.csv)
- [实际 forward 追踪、掩码索引、逐 cone 导数和 VJP identity](D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/runtime-audit.json)
- [独立 mixer/optimizer/扰动检查记录](D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/mixer-audit.json)
- [审计前 SHA256](D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/integrity-before.json)
- [审计后 SHA256](D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/integrity-after.json)
- [前后完整性比较](D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/integrity-comparison.json)
- [训练 manifest 与当前源文件对照](D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/historical-source-integrity.json)
- [实际执行的 production mask/derivative/legacy 审计脚本](D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/runtime_audit.py)
- [实际执行的 mixer 审计正文，附运行注释](D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/mixer_audit.py)
- [只读完整性快照脚本](D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/integrity_snapshot.py)
- [独立方法复核与证据边界](D:/PythonProject/retina_rf_SNN/output/audits/current_checkpoint_applicability_20260831/method-review.txt)
