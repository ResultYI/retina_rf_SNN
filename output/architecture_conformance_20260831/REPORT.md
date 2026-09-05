# Canonical V1 — Independent Architecture Conformance Audit
日期：2026-08-31。唯一 normative specification：用户附件中的 A–K 数学合同。

## 1. Overall
**FAIL**

已确认四类合同问题：mask 后跨 cell 混合扩大实际支持；自定义 geometry 允许非完整圆盘；self-only 多 cell 图存在结构无效 trainable connection；正式 V1 validator/构建段仍可接受并执行 legacy 模式。未发现 stimulus→AC bypass 或第二套 BC encoder。

本次仅审计；没有修改 production source、现有 checkpoint 或既有结果，没有训练。所有扰动仅作用于新建 synthetic model 的内存实例。用户随后明确授权在本目录新增审计探针和证据。未以 RESULTS.md、verification.json、CURRENT_STATE.md 或历史审计的结论为证据；先前记忆检索只用于定位目录。

## 2. 实际实现的 causal graph
```text
X [B,T,C]
 ├─────────────────────────────────────────────┐
 └→ fixed cone graph G → H1 delay → H1 LP        │
                       → same graph transpose Gᵀ
                       → × effective a_H1 → surround
X_H1 = X − surround ←───────────────────────────┘
 → lagged stimulus × polarity
 → shared temporal basis × spatial basis masked per source cell
 → F[B,T,N,4,2,3]
 → M_ij: row-normalized positive cross-cell mixture AFTER masking
 → mixed F
    ├─ paths 0:2 → single shared BC weights → BC_direct
    │                                  → direct mask → optional BC gain ─┐
    └─ paths 2:4 → SAME BC weights → BC_broad = AC input                  │
                   → AC delay → AC LP → negative group gate → AC gain ──┤
                                         total_current = sum 4 paths ←─┘
 → LP(abs(total_current)) → divisive normalization → membrane LP
 → subtract adaptation LP and observed-count history → response bias
 → final logit → sigmoid
```

| Tensor | Producer / inputs | Learnable parameters directly used | Downstream consumers |
|---|---|---|---|
| X | Caller [B,T,C] | none | H1 graph; residual X−surround |
| H1 graph drive/state | GX → fractional delay → LP | h1.raw_delay, h1.raw_tau | Gᵀ and amplitude |
| H1 contribution | a_H1 Gᵀ state | gates.raw_h1_amplitude | X_H1 subtraction |
| X_H1 | X−H1 contribution | inherited H1 dependency | feature_bank only |
| F | lagged X_H1 × spatial masks × temporal basis | feature_bank.raw_tau/raw_delay | shared_subunits |
| mixed F | Σ_j M_ij F_j | shared_subunits.raw_connections | both BC calls |
| BC_direct | mixed F paths 0:2 × normalized weights | bipolar.raw_weights | direct clamp/gain |
| BC_broad | mixed F paths 2:4 × same normalized weights | same bipolar.raw_weights | AC input |
| AC input | exact Python tensor object BC_broad | none additional | AC delay/LP |
| AC state/output | LP_tauAC(Delay_dAC(BC_broad)) | amacrine.raw_tau/raw_delay | negative pathway mixture |
| direct BC current | direct mask × BC_direct × gain | optional BC gain | total current |
| AC current | −group mixture × AC state × gain | gates.ac_local/ac_transient; optional AC gain | total current |
| final current | sum of two BC and two AC currents | no extra parameters | RGC dynamics |
| final logit | normalized current → membrane/adaptation; observed_counts → history | rgc.response_bias, gates.history; upstream parameters | sigmoid |

Source anchors: [models/mechanistic_retina/model.py:138](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/model.py:138), [models/mechanistic_retina/h1_pathway.py:78](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/h1_pathway.py:78), [models/mechanistic_retina/bipolar_subunits.py:162](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/bipolar_subunits.py:162), [models/mechanistic_retina/shared_subunits.py:97](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/shared_subunits.py:97), [models/mechanistic_retina/amacrine_pathways.py:62](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/amacrine_pathways.py:62), [models/mechanistic_retina/rgc_state.py:62](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/rgc_state.py:62).

## 3. Specification vs implementation
| Requirement | Verdict | Evidence / boundary |
|---|---|---|
| Shared BC encoder weights and spatial-mode mixture | PASS | same bipolar module/parameter object; both autograd graphs contain same leaf |
| Shared temporal basis, BC τ and delay | PASS | single feature_bank parameters, temporal_basis.repeat(2,1,1); same leaves in both graphs |
| Only allowed view difference is support | PASS for encoder parameterization | same weights, temporal basis, mixing; actual support violation separately below |
| All AC stimulus information through BC_broad | PASS | AC input is BC_broad object; graph cut removes every observed X path |
| No independent AC stimulus encoder | PASS | complete parameter enumeration and source trace |
| Single H1 upstream, fixed graph and subtraction sign | PASS | one H1 call; G and Gᵀ are serial operations in that branch |
| AC downstream τ/delay/mixture/gain only | PASS | no stimulus basis/weights in AmacrinePathways |
| Default stored BC/AC masks are nested full disks | PASS | d≤R; AC includes BC and nonempty extension |
| Effective encoder per-cell support remains prescribed disk | **FAIL** | mask→cross-cell mixing; nonzero dBC/dX_H1 outside target disk |
| Custom geometry enforces full disks | **FAIL** | accepts mask with an interior hole; production RF contour route exists |
| H1-off exact-zero amplitude, recompute downstream | PASS | retained tensors show zero contribution; both BC and AC change |
| AC-off changes only final AC contribution | PASS | upstream and pre-clamp AC state exact equality; AC current exact-zero |
| direct-BC-off preserves broad BC and AC | PASS | retained tensors exact equality; only direct current zero |
| No renormalization under contribution clamps | PASS | source applies masks after mixture normalization; tensor invariance confirmed |
| Same parameter object/storage/state_dict/optimizer identity | PASS | all enumerated leaves matched; each optimizer object once |
| No orphan or optimizer-missing trainable parameter | PASS for tested core/gain variants | all tensors connected; optimizer sets match; see structural null exception |
| No trainable parameters with no functional influence | **FAIL for self-only multi-cell graph** | raw_connections normalization gives identity independent of parameter |
| RF helper matches same final-logit Jacobian | PASS | four conditions exact equality; central finite differences pass |
| No reachable legacy configuration through V1 entry | **FAIL** | validator accepts legacy; matching state strict-loads; real forward executes |
| Biological correctness / recovery / real-data prediction | UNVERIFIED by this audit | not assessed by synthetic architecture conformance |

## 4. Parameter ownership table
Main fixture: N=6, G=4, E=10; **13 trainable tensors / 88 scalar parameters**. Aggregate gain variant: 15 / 100; alternative pathway mixture variant: 17 / 112. Gain modes are mutually exclusive; the latter is listed because the constructor exposes it, not asserted to be the aggregate-gain V1 candidate.

Every row below uses its exact state_dict key. All have requires_grad=True, the corresponding state_dict tensor shares storage, and optimizer identity count is exactly one. Gradient is for a deterministic scalar of production logits, with nonzero observed history; it does not establish physiological identifiability or training updates.

| Parameter / state_dict key | Shape | Owner | Optimizer identity count | Autograd connected | max abs gradient |
|---|---|---|---:|---|---:|
| `h1.raw_tau` | scalar | H1 τ | 1 | 是 | 8.912594e-2 |
| `h1.raw_delay` | scalar | H1 delay | 1 | 是 | 1.653816e-2 |
| `feature_bank.raw_tau` | 2×3 | shared BC encoder τ | 1 | 是 | 1.738576e+0 |
| `feature_bank.raw_delay` | 2 | shared BC encoder delay | 1 | 是 | 3.549528e+0 |
| `shared_subunits.raw_connections` | 10 | shared BC encoder / cell mixing | 1 | 是 | 1.602596e+0 |
| `bipolar.raw_weights` | 4×2×2×3 | shared BC spatial/temporal mixture | 1 | 是 | 6.270581e-1 |
| `amacrine.raw_tau` | 2 | AC downstream τ | 1 | 是 | 1.646498e+0 |
| `amacrine.raw_delay` | 2 | AC downstream delay | 1 | 是 | 1.509358e+0 |
| `gates.raw_h1_amplitude` | scalar | H1 amplitude | 1 | 是 | 2.771935e-1 |
| `gates.ac_local` | 4 | AC group mixture | 1 | 是 | 4.527693e+0 |
| `gates.ac_transient` | 4 | AC group mixture | 1 | 是 | 4.527693e+0 |
| `gates.history` | scalar | RGC/history | 1 | 是 | 2.253943e+0 |
| `rgc.response_bias` | 6 | RGC/readout | 1 | 是 | 1.759190e+2 |
| `cell_gains.log_bc` | 6 | direct BC readout/gain (aggregate) | 1 | 是 | 5.787442e+1 |
| `cell_gains.log_ac` | 6 | AC downstream gain (aggregate) | 1 | 是 | 2.082543e+1 |
| `cell_gains.log_bc_sustained` | 6 | direct BC readout/gain (pathway_mixture) | 1 | 是 | 2.696009e+1 |
| `cell_gains.log_bc_transient` | 6 | direct BC readout/gain (pathway_mixture) | 1 | 是 | 3.091871e+1 |
| `cell_gains.log_ac_local` | 6 | AC downstream gain (pathway_mixture) | 1 | 是 | 7.820558e+0 |
| `cell_gains.log_ac_transient` | 6 | AC downstream gain (pathway_mixture) | 1 | 是 | 1.325287e+1 |

The main two BC calls observed identical (module id, parameter id, data_ptr, storage_ptr):
`[3037339451520,3037348861568,2683558560128,2683558560128]`.
Both calls matched this tuple. Exact per-parameter addresses are recorded in runtime_results.json; a separate process's shared-leaf membership observations are in identity_edges_results.json. Addresses are process-local, not cross-run identifiers.

No independent learnable AC stimulus encoder parameters. The two operator.depthwise parameters are requires_grad=False and independent operators are explicitly rejected in mechanism mode. Fixed spatial basis/masks, graphs, polarity, RGC decays/gains/slope/threshold are buffers, not omitted trainable parameters. No optimizer step was performed.

**Structural null exception:** in a row with only its self edge, M_ii=softplus(r)/softplus(r)=1. The six-cell fixture contains two such coordinates. A separate four-cell self-only fixture keeps the entire trainable raw_connections tensor in the optimizer, but perturbing all entries leaves logits bitwise identical (max difference 0). Its autograd numerical residual ≤5.615e−16 is roundoff, not functional dependence. This is distinct from a parameter never read by forward.

Optimizer source: [training/mechanistic_retina/optimizer.py:17](D:/PythonProject/retina_rf_SNN/training/mechanistic_retina/optimizer.py:17).

## 5. Runtime dependency matrix
CPU PyTorch 2.6.0+cpu, float64, seed=831, X=[1,40,169], history=[1,40,6]. Cones form [-0.24,0.24]^2 with step .04; four cell groups, two same-group neighbor pairs, nonzero deterministic history. Formula and geometry are in runtime_probe.py and runtime_results.json.

Each entry: equality to normal / maximum absolute difference. EQ means torch.equal, not tolerance comparison. ZERO means every element is exactly zero. H1_state is pre-amplitude; it remains unchanged under H1-off, while H1_contribution becomes zero.

| Tensor | H1-off | direct-BC-off | AC-off |
|---|---|---|---|
| X_H1 | NE / 3.677636e-3 | EQ / 0 | EQ / 0 |
| H1_graph | EQ / 0 | EQ / 0 | EQ / 0 |
| H1_state | EQ / 0 | EQ / 0 | EQ / 0 |
| H1_contribution | NE / 3.677636e-3 / ZERO | EQ / 0 | EQ / 0 |
| AC_input | NE / 4.031158e-3 | EQ / 0 | EQ / 0 |
| AC_state | NE / 3.207790e-3 | EQ / 0 | EQ / 0 |
| BC_direct | NE / 3.913544e-3 | EQ / 0 | EQ / 0 |
| BC_broad | NE / 4.031158e-3 | EQ / 0 | EQ / 0 |
| direct_current | NE / 3.913544e-3 | NE / 4.074689e-1 / ZERO | EQ / 0 |
| AC_current | NE / 1.603895e-3 | EQ / 0 | NE / 2.055996e-1 / ZERO |
| total_current | NE / 4.628892e-3 | NE / 7.557910e-1 | NE / 3.474213e-1 |
| logits | NE / 4.473866e-3 | NE / 7.331178e-1 | NE / 3.364984e-1 |

Hooks capture the actual H1 result and AC argument/state; production output exposes both pre-clamp BC tensors. Final current and logits are compared as well, so the checks do not only inspect reporting values. Full observations for all four conditions are retained in captured_tensors.npz.

## 6. Adversarial dependency test results
Each raw-parameter perturbation adds 0.2 to its first scalar, then restores it; no parameter fitting.

| Perturbed parameter | ΔX_H1 | ΔBC_direct | ΔBC_broad | ΔAC state | Verdict |
|---|---:|---:|---:|---:|---|
| `bipolar.raw_weights` | 0 | 6.965170e-3 | 7.811619e-3 | 6.330512e-3 | PASS |
| `feature_bank.raw_tau` | 0 | 2.491247e-3 | 2.286427e-3 | 1.761226e-3 | PASS |
| `feature_bank.raw_delay` | 0 | 6.983952e-3 | 7.757000e-3 | 5.548451e-3 | PASS |
| `amacrine.raw_tau` | 0 | 0 | 0 | 7.311217e-3 | PASS |
| `amacrine.raw_delay` | 0 | 0 | 0 | 3.606149e-3 | PASS |
| `gates.raw_h1_amplitude` | 7.650575e-4 | 8.141334e-4 | 8.386007e-4 | 6.673156e-4 | PASS |

- Equal-support test changes the actual path_spatial_basis slices, not only reporting buffers; BC_direct and BC_broad become bitwise equal, max difference 0, without changing encoder parameters.
- For q=sum(AC_state²), max|∂q/∂BC_broad|=0.62417654716446; max|∂q/∂X|=0.3388012772387518. The former proves nonzero downstream dependence. The latter is expected, not bypass evidence.
- Full ∂q/∂X equals the VJP routed through BC_broad exactly (residual 0). Traversing actual autograd nodes reaches X before the cut and cannot reach X after excluding BC_broad.grad_fn. Together with source trace, this supports no stimulus→AC bypass in this path.
- Four shared parameter leaves (BC τ, delay, cell mixing, weights) occur by object identity in both BC graphs.
- Spatial negative test on X_H1, avoiding H1's intended upstream mixing: max outside-disk derivatives are 0.0027841088803540167 (direct) and 0.00038928530029573024 (broad). This is a conformance failure.
- Minimal two-cell example: positions/cones (0,0),(.07,0), both midget ON; R_BC=.06, R_AC=.13, shared radius=.08. Stored BC mask=I and AC mask=ones. Mixing off-diagonal≈0.0909091; direct cell0 derivative at the outside cone is 0.02389010889779217. Both effective supports contain both cones, although direct/broad output values need not be equal.
- Custom geometry with cones at 0,.04,.12 is accepted with BC=[1,0,0], AC=[1,0,1]; forward is finite although both masks omit an interior cone.
- Self-only graph negative test: full trainable connection perturbation gives exact logit invariance; see §4.

## 7. RF/helper vs production-forward consistency
Independent RF fixture: seed73021, float64, [B,T,C]=[1,16,25], N=4, T=lag_steps. This compares the same final logit and the same complete input window; current RF and logit RF are not conflated.

| Check | Maximum absolute difference / result |
|---|---|
| effective_rf vs production autograd, normal | 0; bitwise equal |
| same, H1-off | 0; bitwise equal |
| same, direct-BC-off | 0; bitwise equal |
| same, AC-off | 0; bitwise equal |
| Central FD (39 coordinate/step groups; 156 scalar comparisons) | 2.0192425509435452e−10 |
| Sum of four effective_pathway_rf terms vs final-logit Jacobian | 8.326672684688674e−17 |
| Linear basis helper contracted with shared BC weights vs real four currents | ≤6.938893903907228e−18 |
| base_rf vs zero-input total-current Jacobian | ≤1.1102230246251565e−16 |
| Formal collect_responses normal/AC-off vs independent production forward | 0; bitwise equal |

FD steps=1e−4,1e−5,1e−6, pre-set atol=2e−8/rtol=2e−5. Four-condition Jacobians are nonzero. effective_rf returns only the last lag_steps when T is larger; this is a window definition, not full lifetime state memory.

The linear basis helper distributes linear AC dynamics over basis terms before contracting weights. This differs algebraically in order, but the independent numeric contraction agrees; it is not an additional AC encoder or a final-logit surrogate. Actual final-logit RF and formal counterfactual paths call forward_sequence.

Sources: [evaluation/mechanistic_retina/rf_effective.py:16](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/rf_effective.py:16), [evaluation/mechanistic_retina/pathway_decomposition.py:30](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/pathway_decomposition.py:30), [models/mechanistic_retina/pathway_rf.py:59](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/pathway_rf.py:59), [evaluation/mechanistic_retina/karamanlis_v1_ac_runtime.py:57](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/karamanlis_v1_ac_runtime.py:57).

## 8. Bypass / duplicate / dead parameter / optimizer / legacy findings
| Finding | Classification | Evidence |
|---|---|---|
| Stimulus→AC bypass | Not found in audited core | source + tensor identity + graph cut |
| Duplicate BC encoder / second H1 | Not found | single objects and shared leaves |
| Per-target spatial support widened by cell mixing | **FAIL, production reachable** | §6 derivatives; shared_subunits.py:97–107 |
| Non-disk custom geometry accepted | **FAIL, production reachable** | §6; pathway_spatial_geometry.py:52–65 |
| RF-derived support uses contour, not enforced full disk | **FAIL at contract enforcement; exact real instance UNVERIFIED** | [data/karamanlis_rf_artifact.py:121](D:/PythonProject/retina_rf_SNN/data/karamanlis_rf_artifact.py:121); AC=sampled contour, BC=contour∩radius |
| Trainable self-only connection | **FAIL, structurally ineffective** | normalization identity and perturbation |
| Missing/duplicate optimizer parameter identity | Not found in tested core/optional gain variants | all requires_grad objects exactly once |
| V1 entry accepts legacy | **FAIL, production reachable configuration** | validator ignores mode; strict load and real forward succeeded |
| Old ambiguous BC-off in canonical enum | Not accepted | enum conversion uses only explicit no-direct-BC-* |
| Independent pathway operator in canonical forward | Rejected | mechanism mode guard; parameters frozen |
| Other response_snn AC implementation | Outside this canonical call chain | separate production family, not canonical bypass |
| Annulus words in stimulus builders | Not default AC support | not used as canonical support mask |

**Legacy reproduction boundary:** in-memory synthetic payload with V1 schema, stage best_trained, aggregate cell gains and architecture_mode=legacy passes validator. Corresponding builder and strict state load accept all 50 keys, no missing/unexpected keys; forward logits are finite and equal to the source model. Actual direct/broad spatial kernels and presynaptic outputs are equal although support buffers differ. No real checkpoint was read; this establishes a reachable configuration, not that current saved data/checkpoints actually use legacy.

Sources: [evaluation/mechanistic_retina/karamanlis_v1_rf_validation.py:56](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/karamanlis_v1_rf_validation.py:56) and :90; [evaluation/mechanistic_retina/karamanlis_v1_ac_perturbation.py:102](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/karamanlis_v1_ac_perturbation.py:102); [models/mechanistic_retina/bipolar_subunits.py:113](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/bipolar_subunits.py:113).

Three stale analysis/training entries fail direct import: mechanism_runtime requires missing CandidateTeacherUsage; direct_model_eval and training.mechanistic_retina.stages require missing Candidate0Reference from rf_base. They were not repaired. They are currently broken entrypoints, not an observed executing surrogate graph. Optimizer construction itself was directly exercised successfully; the whole historical training workflow was not run.

## 9. Audit artifacts
Directory: `D:/PythonProject/retina_rf_SNN/output/architecture_conformance_20260831`.

- REPORT.md — this complete audit.
- runtime_probe.py / runtime_results.json — core interventions, parameter identities, perturbations and autograd evidence.
- identity_edges_probe.py / identity_edges_results.json — shared graph leaves, optional gains and two-cell support counterexample.
- spatial_dead_parameters_probe.py / spatial_dead_parameters_results.json — self-only ineffective parameters and irregular geometry.
- capture_observations.py / captured_tensors.npz / capture_run_results.json — independently repeated deterministic capture; 50 input/intermediate arrays, no model state/checkpoint.
- source_manifest.json — SHA256 of 136 production source files; unchanged throughout capture.
- rf/ — independent RF source trace, probes, numeric results and state/source invariance.
- legacy/ — independent legacy-entry reproduction and parameter/source notes.

Runtime identity addresses differ between processes by design. The original JSON observations and subsequent tensor capture are separate documented runs of the same deterministic fixture. RF audit additionally checked its complete model state and 26 source files before/after and found no change.

All verdicts above concern implementation conformance to the supplied contract. The separately requested physiology comparison does not change these verdicts.

