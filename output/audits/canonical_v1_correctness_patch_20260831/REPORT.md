# Canonical V1 implementation-contract correctness patch

日期：2026-08-31。四项已确认的全局 implementation-contract FAIL 已修复并通过对应回归检查。当前 22 个 N=1 checkpoints 的冻结输出逐位不变。没有训练、checkpoint conversion 或架构 revision 更名。

## 修改文件

| 文件 | 修改范围 |
| --- | --- |
| [bipolar_subunits.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/bipolar_subunits.py:163) | target-masked mixed kernels；custom geometry 使用原半径生成的 expected supports；保留 N=1 运算顺序 |
| [shared_subunits.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/shared_subunits.py:90) | 仅 multi-neighbor rows 保留 trainable raw connections；self-only 固定 identity |
| [model.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/model.py:70) | 构造器 Canonical config guard；production forward 使用已修复的 feature 路径 |
| [pathway_rf.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/pathway_rf.py:13) | RF basis helper 与 production 使用同一 feature 路径 |
| [pathway_spatial_geometry.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/pathway_spatial_geometry.py:25) | custom BC/AC masks 必须等于完整 radius-defined disks |
| [support_partition.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/support_partition.py:36) | 明确拒绝非有限空间坐标；原有半径与 disk 计算不变 |
| [spatial_contract.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/spatial_contract.py:55) | 在 state_dict copy 前校验实际支持与空间 basis 的支持模式，拒绝旧 geometry 绕过 |
| [canonical_contract.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/canonical_contract.py:12)（新增） | 当前 mode、causal contract、spatial contract 的共同入口校验 |
| [karamanlis_v1_rf_validation.py](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/karamanlis_v1_rf_validation.py:60) | 正式 V1 checkpoint validator 校验原有 schema/stage、当前 revision、完整 config、state markers |
| [test_canonical_mixing_correctness.py](D:/PythonProject/retina_rf_SNN/tests/test_canonical_mixing_correctness.py)（新增） | support、self-only、multi-neighbor、N=1 与 loaded-edge-order 回归 |
| [test_canonical_geometry_entry_correctness.py](D:/PythonProject/retina_rf_SNN/tests/test_canonical_geometry_entry_correctness.py)（新增） | custom geometry、legacy entry、checkpoint identity 与加载前拒绝回归 |

源码精确差分：[correctness_patch.diff](D:/PythonProject/retina_rf_SNN/output/audits/canonical_v1_correctness_patch_20260831/correctness_patch.diff)。既有 loss、训练代码、radius、τ/delay/gain bounds、BC/AC physiology、illusion 和 baseline 未修改。原有测试文件未修改。

## 四项具体修复及全局判定

| 原全局 FAIL | 修复方式 | 对应回归判定 |
| --- | --- | --- |
| multi-cell target effective support 超出规定圆盘 | 在仍保留 cone 轴的 basis kernels 上进行 cross-cell mixing，随后乘 target 的 `[BC,BC,AC,AC]` masks，再与刺激做 contraction；不重新归一化 mixed kernel | **PASS**：原六细胞反例中 direct/broad disk 外导数最大值均为精确 0；新测试逐 cell、逐 BC 分量验证 disk 内存在非零 dependency |
| custom geometry 可含 interior holes/annulus/contour mask | 用原有半径得到 expected full disks，与提供的 masks 精确比较；保留非空 BC、AC 严格扩展条件；加载前也检查实际 masks 与 path basis 支持，不能只靠正确 marker 绕过 | **PASS**：不合规构造与旧错误 geometry state 均在 forward 前拒绝，包括 strict=False |
| self-only row 存在无效 trainable connection | 仅 row degree>1 的 edges 分配 trainable raw 参数；self-only row 为固定 diagonal 1；N=1 原 buffer key/shape/value 保留，matrix 固定为 [[1.0]] | **PASS**：self-only 不进入 optimizer；multi-neighbor 参数仍有非零梯度且扰动改变 mixing；checkpoint edge 顺序变化不会使 derived row metadata 失效 |
| 正式 V1 entry 仍接受 legacy | builder 与直接类构造均要求 mechanism_identifiable/shared-BC/full-disk config；正式 checkpoint validator 检查当前 serialized identity；原有 causal/spatial load hooks 不受 strict=False 绕过 | **PASS**：原 legacy config/checkpoint 接受反例已转为明确拒绝 |

**对应这四项的全局 Architecture Conformance 判定：FAIL → PASS。** 原历史审计文件保留原样；本报告记录修后状态。

加载检查只判断所需的 geometry 支持及共享视图条件，不重新计算并覆写保存的归一化权重；因此合法 float32→float64 模型的 strict roundtrip 不会因重新归一化的舍入差异而被误拒绝。

## Regression tests

修复前，新增 mixing 回归重现 **7 FAIL**；新增 geometry/entry 回归重现 **13 FAIL / 8 PASS**。随后针对带正确 marker 的旧错误 geometry state 补充反例，修前 **6 FAIL / 1 PASS**；针对 cached edge metadata 的中间实现反例也完成 RED→GREEN。

最终限定范围的回归运行：**101 PASS，1 deselected，0 FAIL**。未运行 training；deselected 项为完整 RF application run，未跳过四项修复反例。修前原有相关检查 **46 PASS**，修后仍全部通过。独立 QA 重跑指定的 60 项检查，也为 **60 PASS**。

覆盖：原始 N=6/E=10、最小 N=2 outside-support、N=4 self-only 扰动、mixed-degree optimizer 参数、custom interior-hole/annulus/contour/nonbinary mask、legacy config 与 serialized identity、strict=True/False 加载前拒绝；以及原有 stimulus/history causality、clamp exact-zero、shared BC parameter identity、no AC bypass、RF/autograd/finite-difference。

未修改原始 audit probe，直接在修后代码上重新运行 [runtime_probe.py](D:/PythonProject/retina_rf_SNN/output/architecture_conformance_20260831/runtime_probe.py)，结果保存为 [original_runtime_after.json](D:/PythonProject/retina_rf_SNN/output/audits/canonical_v1_correctness_patch_20260831/original_runtime_after.json)：

- direct outside BC disk max derivative = **0.0**。
- broad outside AC disk max derivative = **0.0**。
- 实际 AC input 与 BC_broad 为同一对象；切断 BC_broad 后，不存在到刺激的 AC 路径。
- AC 梯度完全经 BC_broad 传递，独立梯度复算逐位相同。

证据：[regression_final.xml](D:/PythonProject/retina_rf_SNN/output/audits/canonical_v1_correctness_patch_20260831/regression_final.xml)、[mixing_red.txt](D:/PythonProject/retina_rf_SNN/output/audits/canonical_v1_correctness_patch_20260831/mixing_red.txt)、[mixing_green.txt](D:/PythonProject/retina_rf_SNN/output/audits/canonical_v1_correctness_patch_20260831/mixing_green.txt)、[entry/REPORT.md](D:/PythonProject/retina_rf_SNN/output/audits/canonical_v1_correctness_patch_20260831/entry/REPORT.md)、[review_qa.md](D:/PythonProject/retina_rf_SNN/output/audits/canonical_v1_correctness_patch_20260831/review_qa.md)。

## 22-cell patch-before / patch-after tensor comparison

使用用户指定的 `schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/*/model-trained.pt`，原始 config、不覆盖 mode、strict-load。生产代码修改前捕获全部参考输出；修后使用同一已保存 temporal 和 illusion 输入、相同 normal forward。temporal 选择各 cell 原类型/极性对应的输入及原协议零 history；illusion 直接使用保存的 history。

每个 cell、每个输入 bank 保存 26 个 production 输出字段，加上 hook 捕获的实际 AC input；合计 **44 份参考 NPZ、1,188 个张量、300,920,400 个 scalar elements**。

| 必须比较的张量 | 22-cell 结果 | 最大绝对误差 |
| --- | --- | ---: |
| normal logits | bitwise identical | 0.0 |
| H1 state / contribution | bitwise identical | 0.0 |
| BC_direct | bitwise identical | 0.0 |
| BC_broad | bitwise identical | 0.0 |
| AC 实际 input / local+transient state / current | bitwise identical | 0.0 |
| final current | bitwise identical | 0.0 |
| 其余全部 exposed output fields | bitwise identical | 0.0 |

比较使用 tensor 原始字节，包括 signed-zero representation；没有自行接受任何非零数值误差。**1,188/1,188 全部逐位一致。** Clamp 分别由上述 production regression tests 验证，本张量 before/after 表的模式范围为 normal。

**22/22 checkpoint strict-load PASS，无 conversion。** 每个 checkpoint 的 52 个 state_dict tensors：keys、shapes、dtypes、原始数值及 parameter/buffer 角色完全不变。全部 parameter/buffer 名称及每个 parameter 的 requires_grad 均不变。22 个 checkpoint 文件和两份保存输入文件的 SHA256 均不变。

参考捕获的 per-parameter requires_grad 补充记录来自修前源码快照的独立 strict-load，未进行额外 forward；导入路径及源码哈希已校验。原始参考 NPZ 未重新生成或覆写。

证据：[lineage/SUMMARY.json](D:/PythonProject/retina_rf_SNN/output/audits/canonical_v1_correctness_patch_20260831/lineage/SUMMARY.json)、[lineage/README.md](D:/PythonProject/retina_rf_SNN/output/audits/canonical_v1_correctness_patch_20260831/lineage/README.md)、`lineage/reference/`、`lineage/candidate/`。**当前 22 个 N=1 checkpoints 及本次指定输入上的全部比较输出完全不变。**

## 受影响 multi-cell 配置的参数量变化

设 S 为 self-only rows 的数量，删除对应的 S 个无功能 trainable scalar；保留每个 multi-neighbor row 的原 mixing 参数。以下为不带额外 cell gains 的固定审计 fixture，统计 requires_grad=True 的参数；构造 optimizer 仅检查成员，未执行 step。

| 配置 | trainable scalars 修前→修后 | trainable tensors 修前→修后 |
| --- | ---: | ---: |
| 原 N=6 / E=10，2 个 self-only rows | 88→86 | 13→13 |
| N=4 / E=4，全 self-only | 80→76 | 13→12 |
| N=3 / E=5，1 个 self-only row | 38→37 | 13→13 |

三个 fixture 的初始 connection matrix 与修前逐位一致。当前 22 个 N=1 lineage 的参数量、state_dict 语义与 optimizer 可训练成员不变；不需要 checkpoint conversion。证据：[mixing_parameter_counts.json](D:/PythonProject/retina_rf_SNN/output/audits/canonical_v1_correctness_patch_20260831/mixing_parameter_counts.json)。

## Artifacts

根目录：[output/audits/canonical_v1_correctness_patch_20260831/](D:/PythonProject/retina_rf_SNN/output/audits/canonical_v1_correctness_patch_20260831/)。

- `source_before/`、`source_before_manifest.json`：修前源码/测试快照；共享工作区已有修改全部保留。
- `correctness_patch.diff`、`patched_source_manifest.json`：本次精确修改。
- `regression_final.xml`、`existing_before.xml`、`mixing_*`、`entry/`：修前反例与修后回归。
- `original_runtime_after.json`：原六细胞 audit probe 的修后测量。
- `lineage/`：全部 22 checkpoint 的 before/after 张量及完整性证据。
- `review_goal.md`、`review_code.md`、`review_qa.md`、`review_security.md`、`review_context.md`：独立复核记录。
- `FINAL_MANIFEST.json`：最终审计文件哈希清单。
