# 随机阈值训练诊断报告

日期：2026-07-24

## 结论

训练期 logistic threshold dither 未通过科学门禁，不应进入 150、200、1000 或 6000 optimizer steps，也不应纳入正式训练主线。

实验支持继续使用确定性 hard-spike 模型，并将下一阶段问题重新定义为：

> 当前单次、确定性 LIF 事件码的容量与 current cone-wise reconstruction 任务是否匹配。

本轮不支持把模型改为 stochastic RGC population code。

## 实验合同

- 总训练步数：50
- steps 0–39：训练期 logistic threshold dither
- steps 40–49：恢复确定性 threshold
- paired noisy views 共享同一 threshold dither
- sampled hard event 驱动 membrane reset、adaptation 和 filtered rate
- decoder 冻结
- core learning rate：`2e-4`
- 总体 retinal architecture、T-BPTT、数据与其他目标保持不变
- 主要门禁：确定性 step-50 checkpoint
- 次要诊断：16 次 stochastic repeated-trial inference

## 确定性结果

Source-CV MSE 的 step-0 到 step-50 相对变化如下；正值表示恶化。

| Readout | 相对变化 |
|---|---:|
| Generator potential | +0.40% |
| Spike probability | +0.34% |
| Probability soft rate | +1.47% |
| Hard rate, 10 ms | +4.03% |
| Hard rate, 20 ms | +4.31% |
| Hard rate, 50 ms | +4.31% |
| Hard rate, 100 ms | +4.33% |

其他结果：

- 50 ms deterministic hard-rate validation MSE 恶化 0.41%。
- 4/4 validation sources 的 50 ms deterministic hard-rate MSE 均恶化。
- fixed calibrated decoder MSE 改善 0.55%，但只在 2/4 sources 改善。
- probability saturation fraction 保持为 0。
- zero-spike unit fraction 保持为 0。
- hard-spike fraction 从约 0.0849 增至约 0.0869，没有出现能解释性能退化的大幅放电变化。
- probability variance 保留约 99.85%。

训练日志显示，stochastic 阶段早期仍短暂保留 generator 改善，但该改善逐步缩小；到 step 40 generator source-CV 已开始恶化，恢复 deterministic threshold 后没有获得迁移收益。

## 随机重复试验结果

16 次 stochastic inference 的变化：

| 指标 | 相对变化 |
|---|---:|
| Single-trial source-CV MSE | -0.22% |
| Trial-mean source-CV MSE | -0.31% |
| Trial-mean validation MSE | -0.12% |

Trial-mean validation 在 3/4 sources 上微弱改善，但总体幅度远低于预注册的 2% hard-rate source-CV 门槛，也不足以抵消确定性路径的 4.31% 退化。event-probability calibration MAE 基本不变。

因此，这些变化应解释为微弱、可能处于运行方差范围内的 stochastic benefit，不能支持把模型重新定义为依赖内在随机性的 population code。

## 科学判断

本轮假设为：

> 期望匹配的 stochastic threshold 可以减少 surrogate-gradient probability 与 deterministic reset trajectory 之间的训练偏差，并把连续表征改善迁移到确定性 hard-event code。

结果否定了该假设：

1. 确定性 hard-rate source-CV 明显恶化；
2. 所有验证 source 的确定性 hard-rate 都恶化；
3. 连续 generator/probability 表示到 step 50 也没有保持改善；
4. stochastic repeated-trial 的收益过小；
5. 恢复确定性 threshold 后没有形成可用迁移。

## 代码处置

失败的训练算法已从 canonical code 撤回，包括：

- stochastic-threshold CLI 与配置；
- threshold dither RNG 与 checkpoint state；
- stochastic hard event/reset 训练路径；
- stochastic readout 正式评估入口；
- 对应实验性测试。

保留：

- 原有 deterministic readout ladder；
- 已验证的行为等价模块拆分；
- 本报告和原始诊断产物。

Canonical checkpoint schema 恢复为 revision 5。

## 后续建议

暂不继续叠加 auxiliary loss、修改 T-BPTT、提高学习率或增加 retinal cell mechanism。

若继续本研究，应先重新审视：

1. 每个 cone center 两个 RGC units 的事件码容量；
2. 96-step observation window 对单次 spike code 的可辨识性；
3. current cone-wise MSE 是否过度要求精确、连续、逐 cone 可逆的表征；
4. 是否应将主任务改为更符合 spike-efficient coding 的 rate–distortion 或下游信息目标。

在这些研究选择明确前，长训练仍为 NO-GO。

## 产物

- `test_artifacts/stochastic_threshold_50step_20260724/readout_ladder.json`
- `test_artifacts/stochastic_threshold_50step_20260724/stochastic_readout.json`
- `test_artifacts/stochastic_threshold_50step_20260724/representation_comparison.json`
- `test_artifacts/stochastic_threshold_50step_20260724/training.jsonl`
- `test_artifacts/stochastic_threshold_50step_20260724/evaluation.json`
