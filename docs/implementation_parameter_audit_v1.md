# Implementation Parameter Audit V1

Scope: implementation-relevant checks against `docs/parameter_evidence_human_macaque_v1.md`. No RF loss, no new mechanism, no full training.

## 1. Applied Fixes

| Area | Fix | Files |
|---|---|---|
| `dt_ms` source | `dt_ms_from_time_axis_seconds` now uses `median(diff(time_axis_seconds)) * 1000` | `configs/physiology_profiles.py` |
| Dataset `dt_ms` | `ISETBioDataset` and `ISETBioH5Dataset` expose data-derived `dt_ms` | `data/dataset.py`, `datasets/isetbio_h5_dataset.py` |
| HDF5 contract aliases | Loader prefers `/cone_response_achromatic`, `/cone_xy_deg`, `/cone_type`; keeps old names as compatibility fallback | `data/cone_response.py` |
| HDF5 metadata | Loader reads `metadata/config` when present; falls back to `config_json` | `data/cone_response.py` |
| RGC ordered kinetics | Legacy `raw_kinetic_mix` is rejected; current latents enforce only midget sustained>transient and parasol transient>sustained | `models/cells/rgc.py`, `training/stage1_runtime.py` |
| Training objective | Removed future-ΔC horizons; dataset now masks deterministic cone columns across the input history and reconstructs the clean current fine/coarse contrast only on projected mask weights | `data/dataset.py`, `loss/retina.py` |
| Decoder capacity | Fixed local support is retained, but support-internal spatial values are learnable and exactly row-normalized | `models/decoder/local_decoder.py` |
| Tests | Added/updated tests for real-H5 `dt_ms`, current H5 names, median interval, and ordered kinetics | `tests/test_dataset_interfaces.py`, `tests/test_cone_response_io.py`, `tests/test_physiology_profiles.py`, `tests/test_rgc_cell.py` |

## 2. Dataset And HDF5 Contract

Real HDF5 checked: `data/isetbio_h5_input_png_test/input_seed7.h5`.

| Contract item | Implementation status |
|---|---|
| `/cone_response_achromatic` | read, preferred |
| `/time_axis_seconds` | read and validated |
| `/cone_xy_deg` | read, preferred |
| `/cone_type` | read, preferred |
| `metadata/config` | supported |
| `config_json` | supported fallback; current real H5 uses this |

Shape audit output: `results/stage0_audit/dataset_shape_audit.md`.

Current sample check:

| Tensor | Shape |
|---|---:|
| `input_cone` / `x_cone` | `[3, 4401]` |
| `target_fine` | `[Nfine]` |
| `target_coarse` | `[Ncoarse]` |
| `loss_mask_fine` | `[Nfine]` |
| `loss_mask_coarse` | `[Ncoarse]` |
| `dt_ms` | `5.0` |

Indexing is causal: sample index 0 uses input frames `0..2`, anchor/clean target `2`, and no future frame. Masked cone columns are zeroed over the whole input window; zero is the train-normalized mean.

## 3. Config Separation

| Parameter group | Current location | Status |
|---|---|---|
| Physiology/circuit params | `configs/physiology_profiles.py`, cell config dataclasses | mostly separated |
| Objective/task params | `data/dataset.py` mask fraction/seed and target pools; `loss/retina.py` loss weights | separated from cell configs |
| Training/engineering params | `training/hybrid.py` | separated |
| Analysis params | no central analysis config | not implemented |

`PhysiologyProfile` still contains decoder geometry because decoder support shares the same visual-degree coordinate system. Future-horizon task dimensions have been removed from the physiology profile.

## 4. Midget/Parasol Kinetics

Current mode: ordered non-exclusive kinetic mix.

| Population | Bipolar channel mix | Local amacrine channel mix | Spatial mask |
|---|---|---|---|
| midget-like | sustained share > transient share; starts `[0.75,0.25]` | same mix | `midget_pool` |
| parasol-like | transient share > sustained share; starts `[0.25,0.75]` | same mix | `parasol_pool` |
| residual | mean over sustained/transient | mean over sustained/transient | `residual_pool` |

The 0.75 midpoint is mathematical/engineering, not a precise physiological value. HumRet has no midget/parasol truth labels, so learned differences must be reported as `-like` model outputs.

## 5. H1 Audit

| Item | Current implementation |
|---|---|
| State shape | `[B, NH]`; smoke fixture was `[2, 2]` |
| Spatial type | H1 node lattice, not global scalar |
| Input | current `cone_drive [B,Ncone]` |
| Output modulation | `modulated_drive = cone_drive - gain * h1_to_cone(next_state)` |
| Kernel normalization | `cone_to_h1` and `h1_to_cone` rows checked stochastic |
| Gain | bounded non-negative, `gain_max * sigmoid(raw_gain)` |
| Tau | bounded positive |

No high-risk scalar-H1 simplification found.

## 6. Local Recurrent Amacrine Audit

| Item | Current implementation |
|---|---|
| Input source | current bipolar output `[B,2,2,Ncone]` |
| Output target | previous local amacrine state inhibits bipolar in the next step; current state also inhibits RGC in the current step |
| State shape | `[B,2,2,Ncone]`; smoke fixture was `[2,2,2,4]` |
| Radius/sigma | profile: `0.16/0.10 deg` |
| Delay/tau | causal recurrent filtering; sustained `100 ms`, transient `40 ms` initial |
| Gain bounds | non-negative bounded via sigmoid |
| Future frame access | none |
| Temporal kernel | causal first-order recurrence |

Risk: the local amacrine state is computed from current bipolar output before the current RGC update. This is causal within a frame but does not assert a biological transmission delay onto RGC.

## 7. Residual Constraint Audit

| Item | Current status |
|---|---|
| Residual unit count | smoke fixture: `1`; production mosaic factory not implemented |
| Activity scale | `residual_drive_scale=0.25`, fixed |
| Decoder residual weights | primary residual readout disabled; optional path bounded by `residual_weight_max * tanh(raw_weight)` |
| Decoder weight norm | `0.0` while disabled |
| Residual sparsity/activity loss | connected to total loss |
| Residual decoder penalty | connected to total loss |
| Residual ablation metric | not implemented |

High-risk: residual can still absorb task signal if penalties are too weak. This needs ablation/reporting, not a mechanism change here.

## 8. Decoder Leakage Audit

Detailed output: `results/stage0_audit/decoder_leakage_audit.md`.

Current decoder reads only `RGCOutput` rates. It does not receive H1, bipolar, local amacrine state, cone input, clean target, or loss mask. Support indices are fixed; only support-internal row-normalized spatial values and polarity coefficients learn.

## 9. Bounded Learnable Parameters

Detailed output: `results/stage0_audit/parameter_values.csv`.

Compliant bounded learnable parameters:

- H1 `raw_tau`, `raw_gain`
- bipolar `raw_tau_sustained`, `raw_tau_transient`, `raw_g_ab_sustained`, `raw_g_ab_transient`
- Local amacrine `raw_tau_sustained`, `raw_tau_transient`, `raw_g_ba_sustained`, `raw_g_ba_transient`
- RGC `raw_g_ag_midget`, `raw_g_ag_parasol`, `raw_g_ag_residual`
- residual decoder raw weights through tanh

Mismatch with evidence doc:

- RGC `membrane_tau_ms`, `adaptation_tau_ms`, `rate_tau_ms` are fixed, not bounded learnable.
- `residual_drive_scale` is fixed, not bounded learnable.

Tau lower-bound check:

- bipolar transient lower bound is `5 ms`, equal to the current real-H5 `dt_ms=5 ms`. This is not an immediate failure, but it is a stability edge.

## 10. Mask Sparsity

Detailed output: `results/stage0_audit/mask_sparsity.csv`.

All inspected sparse masks are row-stochastic. Dense support appears in small smoke fixtures for residual and some decoder masks; this is fixture-scale behavior, not proof of production sparsity.

## 11. High-Risk Mismatches

| Risk | Why it matters | Action |
|---|---|---|
| RGC taus fixed | evidence doc recommends bounded learning for temporal dynamics | immediate design decision before full experiments |
| `residual_drive_scale` fixed | residual may be too weak or absorb shortcut behavior depending on data | ablation or bounded scalar later |
| ordered kinetic midpoint | midget/parasol temporal separation is partly imposed as relative order | report as primate-supported relative order plus engineering midpoint, not exact physiology |
| no residual ablation metric | cannot prove residual is not absorbing main task | add evaluation metric before claims |
| `metadata/config` absent in current H5 | current export writes `config_json` instead | export contract should converge later |

## 12. Immediate Fix List

Done:

- `dt_ms` median conversion.
- Dataset-level `dt_ms` exposure.
- Current H5 field aliases.
- Explicit RGC routing mode marker.
- Real-H5 `dt_ms` test.

Still needed before full experiments:

- Decide whether RGC taus remain fixed V1 engineering constants or become bounded learnable.
- Add residual ablation metric/report.
- Freeze `mask_fraction` as an engineering protocol parameter before formal comparisons.
- Quantify whether learned decoder spatial weights remain local and non-degenerate after training.
