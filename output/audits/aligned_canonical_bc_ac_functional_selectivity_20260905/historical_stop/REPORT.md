# Phase M1 — stopped at exact-replay reference gate

前置状态：**STOP_MISSING_EXACT_REPLAY_REFERENCE**。本轮未开展selectivity analysis，不应将此状态解释为科学NO-GO。

## 1. AC predictive consequence是否随 central-vs-broad context mismatch 增大？

**UNVERIFIED**。已核对22个细胞的原development normal/raw BC-off/AC-off tensors，以及44个training-only frozen biases。缺少development `[16,20)`的既存recalibrated off logits及对应NLL参考，无法满足用户第3节的完整exact replay要求。

## 2. direct-BC 是否也表现相同关系？

**UNVERIFIED**；因exact-replay参考门槛未满足，未计算本项。

## 3. AC raw E 是多少？

**UNVERIFIED**；因exact-replay参考门槛未满足，未计算本项。

## 4. direct-BC raw E 是多少？

**UNVERIFIED**；因exact-replay参考门槛未满足，未计算本项。

## 5. normalized AC-vs-BC differentiation D 是多少？

**UNVERIFIED**；因exact-replay参考门槛未满足，未计算本项。

## 6. 控制 central |drive| 后 AC effect还剩多少？

**UNVERIFIED**；因exact-replay参考门槛未满足，未计算本项。

## 7. 控制 generic spatial heterogeneity 后还剩多少？

**UNVERIFIED**；因exact-replay参考门槛未满足，未计算本项。

## 8. 哪些 cell classes 最明显？

**UNVERIFIED**；因exact-replay参考门槛未满足，未计算本项。

## 9. consumed descriptive reuse方向是否一致？

**UNVERIFIED**；因exact-replay参考门槛未满足，未计算本项。

## 10. 最终 verdict：GO / MIXED / NO-GO？

**UNVERIFIED；不作GO/MIXED/NO-GO科学判定。** 上轮bias目录仅保存training `[0,16)` logits；其recalibration分析读取的是已consumed `[20,60)`，没有保存development recalibrated tensors/NLL。不能把本轮首次生成的结果用作自身的既存exact参考。

训练、bias重估、模型/生产源修改、selectivity统计和artificial stimuli均为0。

本轮未取得足以支持进入 Phase M2 的证据；这是前置验证未完成，不是对AC context selectivity的否定结果。
