# Population RetiPath real-data R1 results

完成时间：2026-09-20T07:46:32.307892+00:00。状态：**COMPLETE / VERIFIED**。

严格执行 [冻结 R1 protocol](RETIPATH_REAL_R1_PROTOCOL.md)：MC ON 5 / MC OFF 4 biological cells，16 recordings；每 cell 独立参数与 optimizer，同 cell recordings 共用参数，每个 sequence baseline reset。LegacyPReLU，359 raw scalars 全部列入 joint optimizer；仅 RGC conditional Bernoulli NLL + 原 hierarchy penalty。

9 个唯一 final checkpoints 全部写入并锁定后，独立 evaluator 才开始读取 outer-validation 响应。未进行 validation-driven 选模或重跑；没有改 loss/prior/architecture、缩减 cohort 或加入其他实验。

## 每 cell validation NLL

单位：nats / scored bin。负的 Population−baseline 差表示 Population NLL 较低。所有模型使用相同 recording/trial/frame order、target occupancy 与 mask；旧模型 logits 以 float64 依同一 stable Bernoulli 公式归约，没有重训 baseline。

| Cell | Type | Scored bins | Population | Constant | LN | CNN | Canonical |
|---|---|---:|---:|---:|---:|---:|---:|
| 67#6 | MC OFF | 3360 | 0.503477408 | 0.522224374 | 0.430726051 | 0.401670415 | 0.442264447 |
| 67#7 | MC ON | 3360 | 0.481681304 | 0.529005751 | 0.428921598 | 0.386178350 | 0.456886777 |
| 67#33 | MC OFF | 3360 | 0.428861373 | 0.496288713 | 0.373482376 | 0.374034091 | 0.369917267 |
| 68#3 | MC OFF | 3360 | 0.657377777 | 0.605145592 | 0.486940591 | 0.469915102 | 0.519419882 |
| 68#10 | MC ON | 480 | 0.386375936 | 0.449517405 | 0.315079445 | 0.312622425 | 0.421071056 |
| 69#4 | MC ON | 2880 | 0.468536103 | 0.513997780 | 0.427978826 | 0.371905149 | 0.433251069 |
| 69#6 | MC OFF | 3360 | 0.490539394 | 0.511627079 | 0.430207526 | 0.412060017 | 0.435077964 |
| 69#7 | MC ON | 3360 | 0.515941479 | 0.521962997 | 0.458073703 | 0.451202614 | 0.463302954 |
| 70#34 | MC ON | 3360 | 0.387842511 | 0.465801975 | 0.349417721 | 0.341680931 | 0.368021958 |

## 等 cell 权重汇总

| Group | Cells | Population | Constant | LN | CNN | Canonical |
|---|---:|---:|---:|---:|---:|---:|
| ALL_9 | 9 | 0.480070365 | 0.512841296 | 0.411203093 | 0.391252122 | 0.434357042 |
| MC ON | 5 | 0.448075467 | 0.496057182 | 0.395894259 | 0.372717894 | 0.428506763 |
| MC OFF | 4 | 0.520063988 | 0.533821440 | 0.430339136 | 0.414419906 | 0.441669890 |

## Population − baseline

| Cell / Group | −Constant | −LN | −CNN | −Canonical |
|---|---:|---:|---:|---:|
| 67#6 | -0.018746966 | +0.072751357 | +0.101806993 | +0.061212962 |
| 67#7 | -0.047324447 | +0.052759706 | +0.095502954 | +0.024794527 |
| 67#33 | -0.067427340 | +0.055378997 | +0.054827282 | +0.058944106 |
| 68#3 | +0.052232185 | +0.170437186 | +0.187462675 | +0.137957895 |
| 68#10 | -0.063141469 | +0.071296491 | +0.073753511 | -0.034695120 |
| 69#4 | -0.045461677 | +0.040557277 | +0.096630954 | +0.035285034 |
| 69#6 | -0.021087685 | +0.060331868 | +0.078479376 | +0.055461430 |
| 69#7 | -0.006021518 | +0.057867777 | +0.064738865 | +0.052638525 |
| 70#34 | -0.077959464 | +0.038424790 | +0.046161580 | +0.019820552 |
| ALL_9 | -0.032770931 | +0.068867272 | +0.088818243 | +0.045713323 |
| MC ON | -0.047981715 | +0.052181208 | +0.075357573 | +0.019568704 |
| MC OFF | -0.013757451 | +0.089724852 | +0.105644082 | +0.078394098 |

## 训练审计

Inner-dev 含 step 0；每步按旧 DevelopmentStop 选 K，max 1000 / patience 200 / min_delta 1e−7。丢弃 inner 权重后，从相同 constructor 初值、重置的原 sampling seed，在 full train 恰好更新 K 次。Adam lr=0.03，batch=4，CPU float64 / 2 threads，无 gradient clip。

| Cell | Seed | K | Inner stop | Full-train updates | Trainable | Optimizer-listed | Actually-updated scalars |
|---|---:|---:|---:|---:|---:|---:|---:|
| 67#6 | 20260829 | 998 | 1000 | 998 | 359 | 359 | 348 |
| 67#7 | 20260830 | 963 | 1000 | 963 | 359 | 359 | 348 |
| 67#33 | 20260834 | 905 | 1000 | 905 | 359 | 359 | 348 |
| 68#3 | 20260836 | 850 | 1000 | 850 | 359 | 359 | 348 |
| 68#10 | 20260839 | 648 | 848 | 648 | 359 | 359 | 348 |
| 69#4 | 20260842 | 623 | 823 | 623 | 359 | 359 | 348 |
| 69#6 | 20260843 | 566 | 766 | 566 | 359 | 359 | 348 |
| 69#7 | 20260844 | 259 | 459 | 259 | 359 | 359 | 348 |
| 70#34 | 20260849 | 389 | 589 | 389 | 359 | 359 | 348 |

总 optimizer updates：13686；nonfinite / failed cells：0 / 0。Actually-updated 按与 constructor 初值精确不相等的 raw scalar 计数，不等同于生理参数已辨识。每步 data NLL / penalty / objective、inner-dev NLL、抽样 indices、initial/final state hashes 和 per-tensor optimizer step 均保存在 raw artifacts。

## 工件与核验范围

根目录：`output/real_data/retipath_population_r1/`。

- `protocol.md` / `protocol.json` / `source_lock.json` / `source_snapshot/` / `data_lock.json`。
- `preflight.json` 与各 cell 的 `training_data_contract.json`。
- `cells/<cell>/final.pt`、`inner-trajectory.csv`、`refit-trajectory.csv`、`training-summary.json`。
- `CHECKPOINT_LOCK.json`、`EVALUATION_STARTED.json`，记录先锁定后评价的时间与哈希。
- 各 cell 的 `validation-predictions.pt` / `validation-metrics.json`：五模型 logits、真实 targets、counts、masks 与身份。
- `per-cell-metrics.csv/json`、`aggregate-metrics.csv/json`、`comparison.csv/json`、`training-audit.csv`。
- `verification.json`、`training-manifest.json`、`manifest.json`。

合同检查覆盖冻结哈希、9-cell/16-recording 映射、train-only 返回对象、精确 masks/bin counts、physical Q support、recorded port、strict-past history、prefix causality、reset/batch 隔离、完整无重复 optimizer 参数。原 spike parser 读取整份原始文本，但训练适配器只返回 [0,2400) bins；完整 parser 对象不返回 trainer、不缓存为外层 targets。训练前没有物化 validation targets 或读取旧 validation predictions。

## 结果边界

本报告仅是已消费旧 benchmark validation 上的 matched-cohort conditional prediction 比较；独立新 real test = **NONE**。保留 R0 的 750/751 绝对帧零点未决、未知绝对光强、有限 FOV 与 baseline-reset 限制。不同 baseline 的历史容量、正则和计算预算不同。本轮没有新增显著性检验、成功阈值、机制分析、RF、人工刺激或 pathway ablation，不作下一轮研究决策。
