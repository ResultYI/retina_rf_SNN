# Implementation Parameter Audit V1

Scope: implementation-relevant checks against `docs/parameter_evidence_human_macaque_v1.md`. No RF loss, no new mechanism, no full training.

## 1. Applied Fixes

| Area | Fix | Files |
|---|---|---|
| `dt_ms` source | `dt_ms_from_time_axis_seconds` now uses `median(diff(time_axis_seconds)) * 1000` | `configs/physiology_profiles.py` |
| Dataset `dt_ms` | `ISETBioDataset` and `ISETBioH5Dataset` expose data-derived `dt_ms` | `data/dataset.py`, `datasets/isetbio_h5_dataset.py` |
| HDF5 contract aliases | Loader prefers `/cone_response_achromatic`, `/cone_xy_deg`, `/cone_type`; keeps old names as compatibility fallback | `data/cone_response.py` |
| HDF5 metadata | Loader reads `metadata/config` when present; falls back to `config_json` | `data/cone_response.py` |
| RGC routing marker | Current hard-exclusive routing is explicit as `routing_mode="hard_v1_simplification"`; `biased_mixed` is reserved but not active | `models/cells/rgc_types.py` |
| Tests | Added/updated tests for real-H5 `dt_ms`, current H5 names, median interval, and routing marker | `tests/test_dataset_interfaces.py`, `tests/test_cone_response_io.py`, `tests/test_physiology_profiles.py`, `tests/test_rgc_cell.py` |

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
| `target_fine` | `[2, 2]` |
| `target_coarse` | `[2, 2]` |
| `valid_mask` | not implemented |
| `dt_ms` | `5.0` |

Indexing is causal for model input: sample index 0 uses input frames `0..2`, anchor `2`, target frames `3,4`. Target deltas are computed after the input window.

## 3. Config Separation

| Parameter group | Current location | Status |
|---|---|---|
| Physiology/circuit params | `configs/physiology_profiles.py`, cell config dataclasses | mostly separated |
| Objective/task params | `data/dataset.py` horizons and target pools; `loss/retina.py` loss weights | separated from cell configs |
| Training/engineering params | `training/hybrid.py` | separated |
| Analysis params | no central analysis config | not implemented |

Implementation mismatch: `PhysiologyProfile` still contains `decoder: LocalDecoderConfig`, and `LocalDecoderConfig.horizon_count` is task-shaped. This does not leak RF, but it is a config taxonomy mismatch. No behavior change was made.

## 4. Midget/Parasol Routing

Current mode: `hard_v1_simplification`.

| Population | Bipolar channel read | A2 channel read | Spatial mask |
|---|---|---|---|
| midget-like | sustained only | sustained only | `midget_pool` |
| parasol-like | transient only | transient only | `parasol_pool` |
| residual | mean over sustained/transient | mean over sustained/transient | `residual_pool` |

Reserved interface: `routing_mode="biased_mixed"` is accepted by config validation but does not change behavior yet.

Risk: hard-exclusive routing may make learned RF differences partly architectural rather than fully emergent.

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

## 6. A2 Audit

| Item | Current implementation |
|---|---|
| Input source | current bipolar output `[B,2,2,Ncone]` |
| Output target | previous A2 inhibits bipolar in next step; current A2 also inhibits RGC in current step |
| State shape | `[B,2,2,Ncone]`; smoke fixture was `[2,2,2,4]` |
| Radius/sigma | profile: `0.16/0.10 deg` |
| Delay/tau | causal recurrent leak; sustained `100 ms`, transient `40 ms` initial |
| Gain bounds | non-negative bounded via sigmoid |
| Future frame access | none |
| Temporal kernel | causal first-order recurrence |

Risk: A2 is computed from current bipolar output before current RGC update. This is causal within a frame but is not a strict one-frame delayed inhibition onto RGC.

## 7. Residual Constraint Audit

| Item | Current status |
|---|---|
| Residual unit count | smoke fixture: `1`; production mosaic factory not implemented |
| Activity scale | `residual_drive_scale=0.25`, fixed |
| Decoder residual weights | bounded by `residual_weight_max * tanh(raw_weight)` |
| Decoder weight norm | initial penalty `0.0` |
| Residual sparsity/activity loss | connected to total loss |
| Residual decoder penalty | connected to total loss |
| Residual ablation metric | not implemented |

High-risk: residual can still absorb task signal if penalties are too weak. This needs ablation/reporting, not a mechanism change here.

## 8. Decoder Leakage Audit

Detailed output: `results/stage0_audit/decoder_leakage_audit.md`.

Current decoder reads only `RGCOutput` rates. It does not receive H1, bipolar, A2, future cone frames, or target tensors. Existing static test covers the public decoder signature.

## 9. Bounded Learnable Parameters

Detailed output: `results/stage0_audit/parameter_values.csv`.

Compliant bounded learnable parameters:

- H1 `raw_tau`, `raw_gain`
- bipolar `raw_tau_sustained`, `raw_tau_transient`, `raw_g_ab_sustained`, `raw_g_ab_transient`
- A2 `raw_tau_sustained`, `raw_tau_transient`, `raw_g_ba_sustained`, `raw_g_ba_transient`
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
| hard-exclusive routing | midget/parasol separation is partly imposed | report as V1 simplification; compare with mixed routing later |
| no residual ablation metric | cannot prove residual is not absorbing main task | add evaluation metric before claims |
| no `valid_mask` | no masked loss path for invalid target frames | only matters when variable-validity data is introduced |
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
- Decide whether `valid_mask` is required for the actual training data.
- Separate decoder/task horizon config from `PhysiologyProfile` if a central experiment config is introduced.
