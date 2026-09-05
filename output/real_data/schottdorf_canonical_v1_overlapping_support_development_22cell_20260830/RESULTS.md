# Canonical V1 overlapping-support: 22-cell development fitting

旧值：exclusive-annulus R4-dev；新值：overlapping-support。差值均为新 − 旧。总体与类型汇总采用 equal-cell mean。

## Validation Bernoulli NLL

| 类型 | Cells | 旧 | 新 | 差值 |
| --- | --- | --- | --- | --- |
| ALL | 22 | 0.439670384 | 0.440025040 | +0.000354656 |
| MC_ON | 5 | 0.428413886 | 0.431602967 | +0.003189081 |
| MC_OFF | 4 | 0.443922713 | 0.443060599 | -0.000862114 |
| PC_ON | 9 | 0.428319640 | 0.426482081 | -0.001837558 |
| PC_OFF | 4 | 0.475027852 | 0.477988727 | +0.002960876 |

## Per-cell validation / stopping

| Cell | 类型 | 旧 NLL | 新 NLL | 差值 | 旧 best/stop | 新 best/stop |
| --- | --- | --- | --- | --- | --- | --- |
| 67#4 | PC_OFF | 0.465950638 | 0.472746164 | +0.006795526 | 532/732 | 137/337 |
| 67#6 | MC_OFF | 0.445089072 | 0.443044573 | -0.002044499 | 327/527 | 370/570 |
| 67#7 | MC_ON | 0.456725717 | 0.458423227 | +0.001697510 | 369/569 | 276/476 |
| 67#14 | PC_ON | 0.375346601 | 0.369550139 | -0.005796462 | 471/671 | 817/1000 |
| 67#21 | PC_ON | 0.309472620 | 0.308554322 | -0.000918299 | 423/623 | 565/765 |
| 67#26 | PC_ON | 0.515579343 | 0.515001595 | -0.000577748 | 263/463 | 263/463 |
| 67#33 | MC_OFF | 0.371960282 | 0.369999975 | -0.001960307 | 907/1000 | 581/781 |
| 67#34 | PC_ON | 0.387827367 | 0.387233406 | -0.000593960 | 384/584 | 382/582 |
| 68#3 | MC_OFF | 0.521496654 | 0.521043360 | -0.000453293 | 140/340 | 331/531 |
| 68#4 | PC_ON | 0.433831573 | 0.424961418 | -0.008870155 | 918/1000 | 906/1000 |
| 68#7 | PC_ON | 0.385125488 | 0.381518781 | -0.003606707 | 170/370 | 169/369 |
| 68#10 | MC_ON | 0.420747250 | 0.427795291 | +0.007048041 | 425/625 | 325/525 |
| 68#11 | PC_OFF | 0.374798000 | 0.374329567 | -0.000468433 | 332/532 | 332/532 |
| 69#3 | PC_OFF | 0.555871546 | 0.557392120 | +0.001520574 | 535/735 | 535/735 |
| 69#4 | MC_ON | 0.431954324 | 0.434953779 | +0.002999455 | 645/845 | 193/393 |
| 69#6 | MC_OFF | 0.437144846 | 0.438154489 | +0.001009643 | 563/763 | 520/720 |
| 69#7 | MC_ON | 0.459406048 | 0.464252800 | +0.004846752 | 442/642 | 278/478 |
| 69#21 | PC_OFF | 0.503491223 | 0.507487059 | +0.003995836 | 603/803 | 251/451 |
| 70#1 | PC_ON | 0.547985017 | 0.546360016 | -0.001625001 | 410/610 | 402/602 |
| 70#7 | PC_ON | 0.471348464 | 0.472477108 | +0.001128644 | 263/463 | 248/448 |
| 70#15 | PC_ON | 0.428360283 | 0.432681948 | +0.004321665 | 970/1000 | 106/306 |
| 70#34 | MC_ON | 0.373236090 | 0.372589737 | -0.000646353 | 222/422 | 225/425 |

## RF norms

RF：全部 validation sequences 的 endpoint logit Jacobian 平均；最后 16 input bins。

| 类型 | RF | 旧 norm | 新 norm | 差值 |
| --- | --- | --- | --- | --- |
| ALL | global | 0.246662712 | 0.225322993 | -0.021339719 |
| ALL | H1 | 0.005645866 | 0.006099973 | +0.000454107 |
| ALL | BC | 0.220308639 | 0.282721675 | +0.062413036 |
| ALL | AC | 0.098581364 | 0.130943973 | +0.032362609 |
| MC_ON | global | 0.233193621 | 0.205080178 | -0.028113443 |
| MC_ON | H1 | 0.008487450 | 0.007723881 | -0.000763569 |
| MC_ON | BC | 0.192685383 | 0.248310799 | +0.055625416 |
| MC_ON | AC | 0.128379549 | 0.136948309 | +0.008568760 |
| MC_OFF | global | 0.249920658 | 0.227048511 | -0.022872148 |
| MC_OFF | H1 | 0.009336173 | 0.010709488 | +0.001373315 |
| MC_OFF | BC | 0.220090844 | 0.255613573 | +0.035522729 |
| MC_OFF | AC | 0.109417185 | 0.108097137 | -0.001320048 |
| PC_ON | global | 0.231421829 | 0.206122257 | -0.025299572 |
| PC_ON | H1 | 0.003930503 | 0.004514118 | +0.000583615 |
| PC_ON | BC | 0.206513699 | 0.276241146 | +0.069727447 |
| PC_ON | AC | 0.093919553 | 0.142478364 | +0.048558812 |
| PC_OFF | global | 0.294533116 | 0.292102650 | -0.002430466 |
| PC_OFF | H1 | 0.002263144 | 0.003028746 | +0.000765602 |
| PC_OFF | BC | 0.286094117 | 0.367424562 | +0.081330445 |
| PC_OFF | AC | 0.060986888 | 0.120333010 | +0.059346122 |

## Structural perturbation magnitude

与旧分析相同：全部 validation sequence bins（包含 warmup），off − normal 的绝对值均值。

| 类型 | Clamp | 旧 mean \|Δlogit\| | 新 mean \|Δlogit\| | 差值 | 旧 mean \|Δprobability\| | 新 mean \|Δprobability\| | 差值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | H1-off | 0.068427057 | 0.073935281 | +0.005508224 | 0.008129850 | 0.008660536 | +0.000530686 |
| ALL | BC-off | 1.055895311 | 1.431789486 | +0.375894175 | 0.135562410 | 0.170222791 | +0.034660381 |
| ALL | AC-off | 0.773259224 | 1.173514880 | +0.400255656 | 0.100845156 | 0.146141684 | +0.045296528 |
| MC_ON | H1-off | 0.101536590 | 0.094775415 | -0.006761176 | 0.011002056 | 0.010119690 | -0.000882365 |
| MC_ON | BC-off | 0.989612594 | 1.391613888 | +0.402001294 | 0.119142387 | 0.148356625 | +0.029214237 |
| MC_ON | AC-off | 0.974481934 | 1.379224751 | +0.404742818 | 0.130794850 | 0.180095110 | +0.049300260 |
| MC_OFF | H1-off | 0.105490215 | 0.120159488 | +0.014669273 | 0.014861452 | 0.017169881 | +0.002308428 |
| MC_OFF | BC-off | 1.169858295 | 1.339971488 | +0.170113193 | 0.166009490 | 0.188828993 | +0.022819503 |
| MC_OFF | AC-off | 0.819905070 | 1.007921003 | +0.188015934 | 0.098860266 | 0.115795576 | +0.016935309 |
| PC_ON | H1-off | 0.051229645 | 0.057575853 | +0.006346208 | 0.005566372 | 0.005850078 | +0.000283706 |
| PC_ON | BC-off | 0.919880043 | 1.308764308 | +0.388884264 | 0.108712415 | 0.130045848 | +0.021333433 |
| PC_ON | AC-off | 0.737686830 | 1.168340563 | +0.430653732 | 0.103513497 | 0.160992421 | +0.057478924 |
| PC_OFF | H1-off | 0.028671162 | 0.038469623 | +0.009798461 | 0.003575815 | 0.004650778 | +0.001074963 |
| PC_OFF | BC-off | 1.330820078 | 1.850633635 | +0.519813558 | 0.186052847 | 0.269347418 | +0.083294572 |
| PC_OFF | AC-off | 0.555122880 | 1.093613633 | +0.538490754 | 0.059389163 | 0.100631852 | +0.041242689 |

## H1 effective amplitude

| 类型 | 旧 | 新 | 差值 |
| --- | --- | --- | --- |
| ALL | 0.131096261 | 0.129318729 | -0.001777532 |
| MC_ON | 0.182124996 | 0.168100610 | -0.014024386 |
| MC_OFF | 0.164555926 | 0.181811720 | +0.017255794 |
| PC_ON | 0.127478749 | 0.117177491 | -0.010301258 |
| PC_OFF | 0.041990080 | 0.055666171 | +0.013676091 |

## Learned BC/AC spatial-mode weights

以下为 22-cell mean：对 effective normalized weights 求 temporal-mode 轴之和。每组 BC 四个值共同和为 1；AC 每个 pathway 的两个值和为 1。mode 0/1：PC σ=0.05/0.14°，MC σ=0.09/0.20°。

| Path | Pathway | Mode | 旧 | 新 | 差值 |
| --- | --- | --- | --- | --- | --- |
| BC | sustained | 0 | 0.062815732 | 0.083987920 | +0.021172187 |
| BC | sustained | 1 | 0.074808701 | 0.096473109 | +0.021664409 |
| BC | transient | 0 | 0.250062048 | 0.297183725 | +0.047121677 |
| BC | transient | 1 | 0.612313500 | 0.522355252 | -0.089958248 |
| AC | local | 0 | 0.698245563 | 0.786810642 | +0.088565079 |
| AC | local | 1 | 0.301754424 | 0.213189372 | -0.088565052 |
| AC | transient | 0 | 0.723737901 | 0.892893138 | +0.169155237 |
| AC | transient | 1 | 0.276262103 | 0.107106847 | -0.169155256 |

## Artifacts

- [逐 cell 全指标及差值](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_overlapping_support_development_22cell_20260830/per-cell-comparison.csv)
- [逐 cell BC/AC spatial-mode weights 及差值](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_overlapping_support_development_22cell_20260830/spatial-mode-weights.csv)
- [完整汇总及验证记录](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_overlapping_support_development_22cell_20260830/comparison.json)
- [训练结果与各 cell 配置](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_overlapping_support_development_22cell_20260830/results.json)
- [运行 provenance](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_overlapping_support_development_22cell_20260830/run-manifest.json)
- [global/temporal/pathway RF tensors](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_overlapping_support_development_22cell_20260830/rf-tensors.pt)
- [perturbation tensors](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_overlapping_support_development_22cell_20260830/perturbation-tensors.pt)
- [learned effective parameters](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_overlapping_support_development_22cell_20260830/effective-parameters.pt)

各 cell 的 `cells/<cell_id>/` 保存 `model-raw.pt`、`model-inner-best.pt`、`model-trained.pt`、`inner-trajectory.csv`、`refit-trajectory.csv`、`validation-predictions.pt`。

22/22：相同 target/mask/segments、精确 prediction replay、inner-dev best/stopping 核对、structural clamp exact-zero、inference 参数不变检查通过。训练与分析源文件哈希检查通过。训练未读取旧 checkpoint；分析仅将旧 checkpoint tensors 作为参数对比数据，未载入模型。
