# RF audit evidence archive

Created with explicit user authorization from this same audit session after all probes completed. No probe was rerun. Original run was in-memory with bytecode disabled. The original memo below describes the pre-authorization state; the files in this directory are now the authorized archive. Production code and checkpoints were not modified.

H：**PASS（已测试的正式 RF / counterfactual 路径）**。另发现三条旧分析入口实际无法导入；未修复。

| 核查 | 观察结果 |
|---|---|
| `effective_rf` vs production final-logit Jacobian，normal / H1-off / direct-BC-off / AC-off | 四组均逐元素相等，最大绝对差 **0** |
| 中心差分 | 39 组坐标与步长，156 个标量；最大绝对差 **2.0192425509435452e-10** |
| 四条 `effective_pathway_rf` 求和 vs 同一 final-logit Jacobian | 最大绝对差 **8.326672684688674e-17** |
| `pathway_basis_features` 乘共享 BC weights vs 四条真实 currents | 四条件最大绝对差不超过 **6.938893903907228e-18** |
| `base_rf` vs production 零输入 total-current Jacobian | 四条件最大绝对差不超过 **1.1102230246251565e-16** |
| `collect_responses` 的 normal / AC-off logits vs 独立 production forward | 逐元素相等，最大绝对差 **0**；upstream unchanged，AC 两项 exact-zero |
| 检查前后 | 全部 model state 与 26 个 source 文件 SHA256 不变 |

fixture：CPU，Python 3.12.7，PyTorch 2.6.0+cpu，float64，seed 73021；默认 `MECHANISM_IDENTIFIABLE` 配置，初始化 float32 参数/geometry 后由 production builder 转 float64。输入 `[1,16,25]`，history `[1,16,4]`，四 cell 为 midget ON/OFF、parasol ON/OFF。

- cone 坐标：`[-0.16,-0.08,0,0.08,0.16]²`。
- cell 坐标：`[(0,0),(0.02,0),(0,0.02),(0.02,0.02)]`。
- stimulus：`0.27*sin(0.37*t+0.19*c)+0.12*cos(0.23*t-0.41*c)+0.02*t/15`。
- history：`(arange(64).reshape(1,16,4)%11==0).float64`。
- FD 步长：`1e-4,1e-5,1e-6`；预设 `atol=2e-8, rtol=2e-5`，全部通过。
- 四条件 Jacobian 最大绝对值分别为 `0.2414741942750953 / 0.2415354971159595 / 0.025430163846511596 / 0.2569796792880012`，因此并非全零比较。

来源与边界：

| 文件 | 本次 source trace |
|---|---|
| [rf_effective.py:16](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/rf_effective.py:16) | 真 `forward_sequence(..., clamps)` → 最后时刻 logits → stimulus autograd |
| [pathway_decomposition.py:30](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/pathway_decomposition.py:30) | 同次真 forward，通过 `∂logit/∂total_current` 与真实各 current 进行一阶链式分解 |
| [pathway_rf.py:59](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/pathway_rf.py:59) | 真 forward 四条 current 的零输入 Jacobian |
| [karamanlis_v1_ac_runtime.py:57](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/karamanlis_v1_ac_runtime.py:57) | normal / AC-off 均调用真 forward；已运行验证 |
| [pathway_rf.py:13](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/pathway_rf.py:13) | 独立线性 basis helper，不含 BC mixture weights 或 RGC；上述 contraction 验证与真实 current 一致，不能当作错误 final-logit surrogate |
| [rf_base.py:9](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/rf_base.py:9) | current RF，不是 final-logit RF；两者不能直接比较后判 mismatch |

`effective_rf` 只返回末尾 `lag_steps`。本次 `T=lag_steps=16`，比较的是完整 stimulus window，没有把不同时间切片相混。

三条旧入口经独立 `importlib.import_module` 各一次确认：

| 入口 | 实际错误 |
|---|---|
| `evaluation.mechanistic_retina.mechanism_runtime` | `ImportError: cannot import name 'CandidateTeacherUsage' from 'evaluation.mechanistic_retina.rf_base'` |
| `evaluation.mechanistic_retina.direct_model_eval` | `ImportError: cannot import name 'Candidate0Reference' from 'evaluation.mechanistic_retina.rf_base'` |
| `training.mechanistic_retina.stages` | 同上 `Candidate0Reference` 缺失 |

这些旧入口当前不可运行，不能列为本次成功执行的分析路径，也没有证据将它们称为生产可达的另一套 stimulus encoder。

关键 SHA256：

```text
model.py
29271cc1634ed8eadc577e95b0b6c01a8c5b5ad240f303c05b496b80a5586ed1
pathway_rf.py
ffb02588e5664a58c02ebaad8f7ee9e9860849e378e0363c697aa6901eba8f7f
rf_effective.py
dc87ed315f9145b8f9da9886bb93c7b0cba54ca7f1cc885308af2cc45c81973a
pathway_decomposition.py
de2170f6865f9349c5b4615f51e7d9314f1dda3bcfd7a3f46caf080bbc8288e3
karamanlis_v1_ac_runtime.py
e7de9cf11602e839dbc9ac45d56b68204d9079f1c5522298044cf328f7140eb0
fixture stimulus
6d078151897ba5ea569e48adf715a5dad4f0c0984938a6e1cbeb148bd9d89737
fixture history
0c7067fe59abef40171f7f4978aa5a15a097c94bcac8a8d20af2435ed704871c
```

**Artifacts：未创建任何文件**，遵守主 agent 在授权待定时的明确要求；完整 runtime 数值、source/state hashes 保留在本次工具 stdout。使用 `-B` 和 `PYTHONDONTWRITEBYTECODE=1`；未训练、未接触 checkpoint、未修改现有代码。

<oai-mem-citation>
<citation_entries>
MEMORY.md:28-28|note=[read-only execution preference only; no prior architectural conclusion used]
</citation_entries>
<rollout_ids>
</rollout_ids>
</oai-mem-citation>
