# Parameter Audit: Retina SNN V1

## 1. Summary

This audit covers the current repository state of Retina SNN V1. It is a parameter inventory only; it does not change model mechanisms, defaults, or training behavior.

- Data-derived parameters: ISETBio `cone_response`, `cone_positions_degs`, `cone_types`, `time_axis_seconds`, `eye_trace_degs`, response units, normalization mean/scale when provided or explicitly fit, and optional target pooling matrices.
- Fixed structural hyperparameters: ON/OFF polarity count, sustained/transient channel count, sparse local Gaussian pooling, nearest one-to-one midget pooling, row-stochastic pool checks, target horizons, BPTT truncation, and loss weights.
- Bounded learnable parameters: H1 gain/tau, bipolar tau/g_AB, A2 tau/g_BA, RGC g_AG, and residual decoder projection weights. They use raw `nn.Parameter` values with sigmoid or tanh bounded parameterization.
- Learnable but unbounded parameters: non-residual local decoder projection weights, initialized to zero.
- Training hyperparameters: `HybridTrainingConfig.t_bptt`, `grad_clip_norm`, test-only SGD `lr=0.05`, and stage behavior implemented through `torch.no_grad()` during decoder warmup.
- Parameters currently missing literature evidence: most biological time constants, spatial radii/sigmas, RGC LIF threshold/surrogate/adaptation constants, H1/A2/RGC gain bounds, residual population scale, decoder local masks, and smoke-gate thresholds. Current concrete model values mostly live in tests, so they are marked `test-only`.
- Not implemented yet: `configs/`, `evaluation/`, a central default config factory, CLI argparse defaults, A3/STP/gap coupling/RF loss, and a real smoke gate threshold config.

## 2. Parameter Tables

### ISETBio Data / Dataset Config

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `h5_path` | required by caller | path | no | must load H5 contract | dataclass field | ISETBio cone export source | data-derived | `data/dataset.py:19` `ISETBioDatasetConfig`; `data/cone_response.py:24` `load_cone_response` |
| `format_version` | `retina-snn-cone-response-v1` | string | no | exact match | H5 dataset decode | rejects incompatible exports | engineering-prior | `data/cone_response.py:27-29` |
| `cone_response` | H5 dataset `cone_response` | response units from file | no | finite, non-negative, shape `[T,Ncone]` | `_logical_array` plus `validate_response` | model input before log normalization | data-derived | `data/cone_response.py:36`, `data/cone_response.py:68-74` |
| `positions_degs` | H5 dataset `cone_positions_degs` | degrees | no | shape `[Ncone,2]` | loaded into dataset and model geometry | spatial coordinates for H1, bipolar, A2, RGC, decoder | data-derived | `data/cone_response.py:37-39`; `data/dataset.py:162-174` |
| `cone_types` | H5 dataset `cone_types` | categorical id | no | shape `[Ncone]` | loaded and order-checked during stats fit | preserves cone ordering across exports | data-derived | `data/cone_response.py:40`; `data/dataset.py:65-66` |
| `time_axis_seconds` | H5 dataset `time_axis_seconds` | seconds | no | finite, strictly increasing, stable interval CV <= `1e-3` | `_validate_time_axis` | data clock; not auto-converted to model `dt_ms` yet | data-derived | `data/cone_response.py:41`, `data/cone_response.py:77-86` |
| `eye_trace_degs` | H5 dataset `eye_trace_degs` | degrees | no | shape `[T,2]` | loaded into dataset property | eye trajectory metadata | data-derived | `data/cone_response.py:42-44`; `data/dataset.py:164-186` |
| `input_steps` | production default `16`; tests use `2` or `3` | frames | no | `>= 1` | dataclass default and validation | input temporal window length | engineering-prior | `data/dataset.py:22`, `data/dataset.py:241-243`; `tests/test_isetbio_data_contract.py:28`, `tests/test_isetbio_data_contract.py:71` |
| `allow_fit_stats` | production default `False`; smoke/test can set `True` | boolean | no | boolean | explicit gate before fitting stats from same H5 | prevents accidental non-train stats fitting | engineering-prior | `data/dataset.py:26`, `data/dataset.py:128-134`; `tests/test_isetbio_data_contract.py:69-75` |

### Normalization / Clipping

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `eps` | production default `1e-6`; tests also use `1e-5` for round trip | response offset | no | `> 0` | added before `log`; also minimum stats scale | numerical stability for log cone response | engineering-prior | `data/dataset.py:24`, `data/dataset.py:39-42`, `data/dataset.py:72`; `tests/test_isetbio_data_contract.py:126` |
| `mean` | caller-provided or explicitly fit | log-response units | no | finite shape `[Ncone]` | `_validate_stats`; stored as `normalization_mean` | per-cone centering | data-derived | `data/dataset.py:45-72`, `data/dataset.py:136`, `data/dataset.py:192-198` |
| `scale` | caller-provided or explicitly fit, lower-bounded by `eps` when fit | log-response units | no | finite, positive shape `[Ncone]` | `_validate_stats`; `np.maximum(scale, eps)` | per-cone standardization | data-derived | `data/dataset.py:70-72`, `data/dataset.py:256-269` |
| `clip` | production default `5.0`; tests use `0.5` and `100.0` | normalized log contrast | no | `> 0`; applied symmetrically as `[-clip, clip]` | `np.clip`; `clip_fraction` computed before clipping | bounds normalized input and target base | engineering-prior | `data/dataset.py:25`, `data/dataset.py:144-149`, `data/dataset.py:248-249` |
| `clip_fraction` | computed | fraction | no | `[0,1]` by construction | mean of `abs(unclipped_contrast) > clip` | clipping diagnostic | data-derived | `data/dataset.py:144`, `data/dataset.py:200-202` |

### Target Horizons and Target Pooling

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `horizons` | production default `(1, 2, 4)`; tests use `(1, 2)` and `(1,)` | frames | no | non-empty, every horizon `>= 1` | dataclass default and validation | future offsets for target delta | engineering-prior | `data/dataset.py:23`, `data/dataset.py:244-245`; `tests/test_isetbio_data_contract.py:29`, `tests/test_isetbio_data_contract.py:72` |
| `target_delta` | computed as contrast at `anchor+horizon` minus base | normalized log contrast | no | sequence length must support max horizon | `np.stack([...])` | fine-grained future-change target | data-derived | `data/dataset.py:215-226` |
| `target_fine_pool` | production default `None`; tests use identity sparse pool | sparse row-stochastic weights | no | sparse COO, finite, non-negative, shape `[Ntarget,Ncone]`, rows sum to 1 | `_validate_target_pool`; `torch.sparse.mm` | optional fine target pooling | data-derived | `data/dataset.py:27`, `data/dataset.py:150-154`, `data/dataset.py:272-289`; `tests/test_isetbio_data_contract.py:92` |
| `target_coarse_pool` | production default `None`; tests use two-row average pool | sparse row-stochastic weights | no | same as `target_fine_pool`; must be provided together | `_validate_target_pool`; `torch.sparse.mm` | optional coarse target pooling | data-derived | `data/dataset.py:28`, `data/dataset.py:155-158`, `data/dataset.py:229-237`; `tests/test_isetbio_data_contract.py:93-104` |

### Deterministic Midget / Parasol / Residual Mosaic

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `RGCMosaic.bipolar_positions_degs` | required by caller; tests use `[[0,0],[0.1,0],[0.2,0],[0.3,0]]` | degrees | no | shape `[N,2]`, finite | dataclass plus `positions_tensor` | source positions for RGC pooling | test-only | `models/cells/rgc_types.py:16-21`; `models/cells/rgc_runtime.py:66-72`; `tests/test_rgc_cell.py:18-27` |
| `midget_positions_degs` | caller-provided; tests equal bipolar positions | degrees | no | must align one-to-one with bipolar positions | `torch.allclose(..., atol=1e-6)` then nearest one-to-one pool | midget population layout | test-only | `models/cells/rgc.py:54-60`, `models/cells/rgc.py:73-76`; `tests/test_rgc_cell.py:22-25` |
| `parasol_positions_degs` | caller-provided; tests use `[[0.05,0],[0.25,0]]` | degrees | no | `residual_count <= parasol_count < midget_count` | local Gaussian pool | parasol population layout | test-only | `models/cells/rgc.py:61-68`, `models/cells/rgc.py:77-82`; `tests/test_rgc_cell.py:25` |
| `residual_positions_degs` | caller-provided; tests use `[[0.15,0]]` | degrees | no | `residual_count <= parasol_count` | local Gaussian pool | residual population layout | test-only | `models/cells/rgc.py:50-53`, `models/cells/rgc.py:83-88`; `tests/test_rgc_cell.py:26` |
| deterministic mosaic generator | not implemented yet | n/a | no | n/a | caller supplies `RGCMosaic` | no production midget/parasol/residual mosaic factory | pending-evidence | `models/cells/rgc_types.py:16-21`; no `configs/` or mosaic factory found |

### H1 Horizontal Network

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `radius_degs` | no production default; tests use `0.16` | degrees | no | `> 0` | local Gaussian cone to H1 support | H1 node support radius | test-only | `models/cells/horizontal.py:43-55`, `models/cells/horizontal.py:106-111`; `tests/test_horizontal_cell.py:14` |
| `sigma_degs` | no production default; tests use `0.1` | degrees | no | `> 0` | Gaussian `exp(-0.5 * (distance/sigma)^2)` | H1 input weighting scale | test-only | `models/cells/horizontal.py:45`, `data/geometry.py:13-32`; `tests/test_horizontal_cell.py:15` |
| `feedback_radius_degs` | no production default; tests use `0.21` | degrees | no | `> 0` | local Gaussian H1 to cone feedback | surround feedback support | test-only | `models/cells/horizontal.py:46`, `models/cells/horizontal.py:112-117`; `tests/test_horizontal_cell.py:16` |
| `feedback_sigma_degs` | no production default; tests use `0.12` | degrees | no | `> 0` | local Gaussian H1 to cone feedback | surround feedback weighting scale | test-only | `models/cells/horizontal.py:47`, `models/cells/horizontal.py:112-117`; `tests/test_horizontal_cell.py:17` |
| `h1_spacing_degs` | no production default; tests use `0.2` | degrees | no | `> 0`; supported H1 nodes must be fewer than cones | centered grid then support filter | H1 node spacing | test-only | `models/cells/horizontal.py:48`, `models/cells/horizontal.py:97-105`, `models/cells/horizontal.py:258-293`; `tests/test_horizontal_cell.py:18` |
| `dt_ms` | no production default; tests use `5.0` | ms | no | `> 0` | stored as `_dt_ms` | temporal leak clock | test-only | `models/cells/horizontal.py:49`, `models/cells/horizontal.py:141`; `tests/test_horizontal_cell.py:19` |
| `initial_tau_ms` | no production default; tests use `50.0` | ms | yes | `tau_min_ms < initial < tau_max_ms`; tests bounds `10.0` to `200.0` | `raw_tau = logit((initial-min)/(max-min))`; effective `min + range * sigmoid(raw_tau)` | H1 recurrent low-pass tau | test-only | `models/cells/horizontal.py:50-52`, `models/cells/horizontal.py:134-151`; `tests/test_horizontal_cell.py:20-22` |
| `initial_gain` | no production default; tests use `0.01` | unitless gain | yes | `0 < initial_gain < gain_max`; tests max `0.2` | `raw_gain = logit(initial/gain_max)`; effective `gain_max * sigmoid(raw_gain)` | subtracts H1 surround from cone drive | test-only | `models/cells/horizontal.py:53-54`, `models/cells/horizontal.py:131-146`, `models/cells/horizontal.py:201-203`; `tests/test_horizontal_cell.py:23-24` |
| `debug_checks` | default `True` | boolean | no | boolean | finite checks on input/state | runtime guard | engineering-prior | `models/cells/horizontal.py:55`, `models/cells/horizontal.py:183-195` |

### ON/OFF Sustained/Transient Bipolar Layer

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| polarity channels | `ON=0`, `OFF=1` | index | no | fixed two channels | `BipolarPolarity` enum | splits positive/negative drive | engineering-prior | `models/cells/bipolar_types.py:15-17`; `models/cells/bipolar.py:131-134` |
| kinetics channels | `SUSTAINED=0`, `TRANSIENT=1` | index | no | fixed two channels | `BipolarKinetics` enum | separates low-pass and high-pass outputs | engineering-prior | `models/cells/bipolar_types.py:20-22`; `models/cells/bipolar.py:154-155` |
| `dt_ms` | no production default; tests use `5.0` | ms | no | `> 0` | `_dt_ms` | leak clock | test-only | `models/cells/bipolar_types.py:51`, `models/cells/bipolar.py:55`; `tests/test_bipolar_cell.py:17` |
| `initial_tau_sustained_ms` | no production default; tests use `80.0` | ms | yes | `60.0` to `200.0` in tests | raw sigmoid bounded tau | sustained bipolar leak | test-only | `models/cells/bipolar_types.py:52-54`, `models/cells/bipolar.py:35-39`, `models/cells/bipolar.py:70-80`; `tests/test_bipolar_cell.py:18-20` |
| `initial_tau_transient_ms` | no production default; tests use `20.0` | ms | yes | `5.0` to `40.0` in tests; max must be below sustained min | raw sigmoid bounded tau | transient bipolar leak | test-only | `models/cells/bipolar_types.py:55-57`, `models/cells/bipolar_types.py:93-96`, `models/cells/bipolar.py:40-44`; `tests/test_bipolar_cell.py:21-23` |
| `initial_g_ab_sustained` | no production default; tests use `0.01` | unitless inhibition gain | yes | `0.0` to `0.1` in tests | `g_ab_max * sigmoid(raw_g_ab_sustained)` | A2 to sustained bipolar inhibition | test-only | `models/cells/bipolar_types.py:58-59`, `models/cells/bipolar.py:45-49`, `models/cells/bipolar.py:168-173`; `tests/test_bipolar_cell.py:24-25` |
| `initial_g_ab_transient` | no production default; tests use `0.01` | unitless inhibition gain | yes | `0.0` to `0.3` in tests; transient max must exceed sustained max | `g_ab_max * sigmoid(raw_g_ab_transient)` | A2 to transient bipolar inhibition | test-only | `models/cells/bipolar_types.py:60-61`, `models/cells/bipolar.py:50-54`; `tests/test_bipolar_cell.py:26-27` |

### Transient High-Pass / Baseline Subtraction

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| transient baseline state | initialized to zeros | normalized drive | no | same shape as `[batch,2,Ncone]` | `BipolarState.transient_baseline` | stores polarity-specific baseline | engineering-prior | `models/cells/bipolar_types.py:43-47`, `models/cells/bipolar.py:100-116` |
| transient drive | `relu(private_drive - state.transient_baseline)` | normalized drive | no | non-negative via ReLU | high-pass subtraction before transient channel | transient response to increases above baseline | engineering-prior | `models/cells/bipolar.py:154-155` |
| baseline tau | uses sustained bipolar tau, no separate config | ms | yes, indirectly through sustained tau | sustained tau bounds | `baseline_leak = leak_values[SUSTAINED]` | baseline adaptation for transient subtraction | test-only | `models/cells/bipolar.py:174-178` |
| separate transient baseline tau | not implemented yet | n/a | no | n/a | no config field | no independent baseline timescale | pending-evidence | `models/cells/bipolar_types.py:50-61`; `models/cells/bipolar.py:174` |

### A2 Amacrine Layer

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `radius_degs` | no production default; tests use `0.15` or core fixture `0.16` | degrees | no | `> 0` | local Gaussian positions to positions | A2 spatial pooling radius | test-only | `models/cells/amacrine.py:40`, `models/cells/amacrine.py:100-102`; `tests/test_amacrine_cell.py:17`; `tests/test_retina_snn.py:60` |
| `sigma_degs` | no production default; tests use `0.1` | degrees | no | `> 0` | Gaussian weighting | A2 spatial pooling scale | test-only | `models/cells/amacrine.py:41`, `data/geometry.py:27`; `tests/test_amacrine_cell.py:18` |
| `dt_ms` | no production default; tests use `5.0` | ms | no | `> 0` | `_dt_ms` | leak clock | test-only | `models/cells/amacrine.py:42`, `models/cells/amacrine.py:137`; `tests/test_amacrine_cell.py:19` |
| `initial_tau_sustained_ms` | no production default; tests use `100.0` | ms | yes | `40.0` to `250.0` in tests | raw sigmoid bounded tau | A2 sustained channel state tau | test-only | `models/cells/amacrine.py:43-45`, `models/cells/amacrine.py:117-121`, `models/cells/amacrine.py:153-163`; `tests/test_amacrine_cell.py:20-22` |
| `initial_tau_transient_ms` | no production default; tests use `40.0` | ms | yes | `15.0` to `100.0` in tests | raw sigmoid bounded tau | A2 transient channel state tau | test-only | `models/cells/amacrine.py:46-48`, `models/cells/amacrine.py:122-126`; `tests/test_amacrine_cell.py:23-25` |
| `initial_g_ba_sustained` | no production default; tests use `0.03` | unitless drive gain | yes | `0.0` to `0.3` in tests | `g_ba_max * sigmoid(raw_g_ba_sustained)` | scales pooled sustained bipolar output into A2 state | test-only | `models/cells/amacrine.py:49-50`, `models/cells/amacrine.py:127-131`, `models/cells/amacrine.py:166-174`, `models/cells/amacrine.py:230-233`; `tests/test_amacrine_cell.py:26-27` |
| `initial_g_ba_transient` | no production default; tests use `0.05` | unitless drive gain | yes | `0.0` to `0.5` in tests | `g_ba_max * sigmoid(raw_g_ba_transient)` | scales pooled transient bipolar output into A2 state | test-only | `models/cells/amacrine.py:51-52`, `models/cells/amacrine.py:132-136`; `tests/test_amacrine_cell.py:28-29` |
| self weight diagnostics | computed from local pool diagonal | row-stochastic weight | no | pool rows sum to 1 | `self_weight_mean`, `self_weight_max` buffers | monitors self-vs-neighbor pooling | data-derived | `models/cells/amacrine.py:107-116` |

### RGC Midget / Parasol / Residual Population Layer

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `parasol_radius_degs` | no production default; tests use `0.16` | degrees | no | `> 0` | local Gaussian bipolar to parasol pool | parasol spatial integration radius | test-only | `models/cells/rgc_types.py:26`, `models/cells/rgc.py:77-82`; `tests/test_rgc_cell.py:32` |
| `parasol_sigma_degs` | no production default; tests use `0.1` | degrees | no | `> 0` | local Gaussian sigma | parasol spatial weighting | test-only | `models/cells/rgc_types.py:27`, `models/cells/rgc.py:77-82`; `tests/test_rgc_cell.py:33` |
| `residual_radius_degs` | no production default; tests use `0.25` | degrees | no | `> 0` | local Gaussian bipolar to residual pool | residual spatial integration radius | test-only | `models/cells/rgc_types.py:28`, `models/cells/rgc.py:83-88`; `tests/test_rgc_cell.py:34` |
| `residual_sigma_degs` | no production default; tests use `0.12` | degrees | no | `> 0` | local Gaussian sigma | residual spatial weighting | test-only | `models/cells/rgc_types.py:29`, `models/cells/rgc.py:83-88`; `tests/test_rgc_cell.py:35` |
| `initial_g_ag_midget` | no production default; tests use `0.01` | unitless inhibition gain | yes | `0.0` to `0.1` in tests | `g_ag_max * sigmoid(raw_g_ag_midget)` | A2 sustained inhibition on midget current | test-only | `models/cells/rgc_types.py:37-38`, `models/cells/rgc.py:108-111`, `models/cells/rgc.py:194-204`; `tests/test_rgc_cell.py:43-44` |
| `initial_g_ag_parasol` | no production default; tests use `0.03` | unitless inhibition gain | yes | `0.0` to `0.3` in tests; parasol max must exceed midget max | `g_ag_max * sigmoid(raw_g_ag_parasol)` | A2 transient inhibition on parasol current | test-only | `models/cells/rgc_types.py:39-40`, `models/cells/rgc.py:112-115`; `tests/test_rgc_cell.py:45-46` |
| `initial_g_ag_residual` | no production default; tests use `0.01` | unitless inhibition gain | yes | `0.0` to `0.1` in tests | `g_ag_max * sigmoid(raw_g_ag_residual)` | A2 inhibition on residual current | test-only | `models/cells/rgc_types.py:41-42`, `models/cells/rgc.py:116-119`; `tests/test_rgc_cell.py:47-48` |
| `residual_drive_scale` | no production default; tests use `0.25` | unitless scale | no | `(0,1]` | multiplied onto residual current | down-scales residual population drive | test-only | `models/cells/rgc_types.py:43`, `models/cells/rgc_types.py:79-80`, `models/cells/rgc.py:199-204`; `tests/test_rgc_cell.py:49` |
| midget sustained mapping | midget uses sustained bipolar/A2 channels | channel index | no | fixed by enum | `bipolar_output[:, :, SUSTAINED]` | maps midget to sustained pathway | engineering-prior | `models/cells/rgc.py:189-196`; `tests/test_rgc_cell.py:110-123` |
| parasol transient mapping | parasol uses transient bipolar/A2 channels | channel index | no | fixed by enum | `bipolar_output[:, :, TRANSIENT]` | maps parasol to transient pathway | engineering-prior | `models/cells/rgc.py:190-198`; `tests/test_rgc_cell.py:110-123` |

### RGC Adaptive LIF

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `dt_ms` | no production default; tests use `5.0` | ms | no | `> 0` | used in three leak buffers | clock for RGC dynamics | test-only | `models/cells/rgc_types.py:30`, `models/cells/rgc_runtime.py:22-33`; `tests/test_rgc_cell.py:36` |
| `membrane_tau_ms` | no production default; tests use `20.0` | ms | no | `> 0` | `membrane_leak = exp(-dt/tau)` | membrane state decay | test-only | `models/cells/rgc_types.py:31`, `models/cells/rgc_runtime.py:22-25`; `tests/test_rgc_cell.py:37` |
| `adaptation_tau_ms` | no production default; tests use `80.0` | ms | no | `> 0` | `adaptation_leak = exp(-dt/tau)` | spike-triggered adaptation decay | test-only | `models/cells/rgc_types.py:32`, `models/cells/rgc_runtime.py:26-29`; `tests/test_rgc_cell.py:38` |
| `rate_tau_ms` | no production default; tests use `50.0` | ms | no | `> 0` | `rate_leak = exp(-dt/tau)` | low-pass rate history | test-only | `models/cells/rgc_types.py:33`, `models/cells/rgc_runtime.py:30-33`; `tests/test_rgc_cell.py:39` |
| `threshold` | no production default; tests use `0.2` | current/state units | no | `> 0` through shared positive check | hard threshold plus surrogate | spike threshold | test-only | `models/cells/rgc_types.py:34`, `models/cells/rgc_runtime.py:34`, `models/cells/rgc_runtime.py:50-54`; `tests/test_rgc_cell.py:40` |
| `surrogate_slope` | no production default; tests use `5.0` | inverse current/state units | no | `> 0` through shared positive check | `sigmoid(slope * (pre_reset - threshold))` | surrogate gradient steepness | test-only | `models/cells/rgc_types.py:35`, `models/cells/rgc_runtime.py:35`, `models/cells/rgc_runtime.py:51-54`; `tests/test_rgc_cell.py:41` |
| `adaptation_strength` | no production default; tests use `0.1` | state increment scale | no | `> 0` through shared positive check | `(1-adaptation_leak) * adaptation_strength * spikes` | spike-triggered adaptation increment | test-only | `models/cells/rgc_types.py:36`, `models/cells/rgc_runtime.py:36`, `models/cells/rgc_runtime.py:56-61`; `tests/test_rgc_cell.py:42` |

### Local Decoder

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `horizon_count` | no production default; tests use `3` | horizons | no | `>= 1` | config field and raw weight first dimension | prediction horizon channels | test-only | `models/decoder/local_decoder.py:25-31`, `models/decoder/local_decoder.py:206-208`; `tests/test_local_decoder.py:36-43` |
| `fine_radius_degs` | no production default; tests use `0.11`; core integration test uses `0.16` | degrees | no | `> 0` | local Gaussian mask from RGC positions to fine targets | local support for fine prediction | test-only | `models/decoder/local_decoder.py:27`, `models/decoder/local_decoder.py:71-92`; `tests/test_local_decoder.py:38`; `tests/test_retina_snn.py:211` |
| `fine_sigma_degs` | no production default; tests use `0.08` | degrees | no | `> 0` | local Gaussian mask | fine prediction spatial weighting | test-only | `models/decoder/local_decoder.py:28`, `models/decoder/local_decoder.py:198-204`; `tests/test_local_decoder.py:39` |
| `coarse_radius_degs` | no production default; tests use `0.21` | degrees | no | `> 0` | local Gaussian mask | local support for coarse prediction | test-only | `models/decoder/local_decoder.py:29`, `models/decoder/local_decoder.py:93-114`; `tests/test_local_decoder.py:40` |
| `coarse_sigma_degs` | no production default; tests use `0.12` | degrees | no | `> 0` | local Gaussian mask | coarse prediction spatial weighting | test-only | `models/decoder/local_decoder.py:30`, `models/decoder/local_decoder.py:198-204`; `tests/test_local_decoder.py:41` |
| `residual_weight_max` | no production default; tests use `0.1` | decoder weight | no | `> 0` | used only for residual projections | bounds residual decoder contribution | test-only | `models/decoder/local_decoder.py:31`, `models/decoder/local_decoder.py:85-92`, `models/decoder/local_decoder.py:107-114`; `tests/test_local_decoder.py:42` |
| `residual_initial_weight_fraction` | production profile and default `0.05` | fraction of decoder weight bound | no | `(0,1)` | initializes residual ON/OFF weights with opposite signs | prevents a zero-rate/zero-readout gradient deadlock | engineering-prior (E), not physiological | `models/decoder/local_decoder.py`; `configs/physiology_profiles.py`; `tests/test_local_decoder.py` |
| non-residual decoder `raw_weight` | initialized to `0` | decoder weight | yes | unbounded | `effective_weight = raw_weight` | midget/parasol local decoder weights | engineering-prior | `models/decoder/local_decoder.py:206-216` |
| residual decoder `raw_weight` | initialized to a small ON/OFF-antisymmetric value | decoder weight | yes | effective `[-residual_weight_max, residual_weight_max]` via tanh | `effective_weight = residual_weight_max * tanh(raw_weight)` | residual local decoder weights | engineering-prior (E) initialization; learned value is latent (D) | `models/decoder/local_decoder.py`; `tests/test_local_decoder.py` |

### Loss / Regularization

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `fine_weight` | production default `1.0`; hybrid test uses `1.0` | loss multiplier | no | `>= 0` | weighted MSE | fine target prediction loss | engineering-prior | `loss/retina.py:23-24`, `loss/retina.py:95`, `loss/retina.py:112`; `tests/test_hybrid_training.py:32` |
| `coarse_weight` | production default `1.0`; hybrid test uses `0.5` | loss multiplier | no | `>= 0` | weighted MSE | coarse target prediction loss | engineering-prior | `loss/retina.py:25`, `loss/retina.py:96`, `loss/retina.py:113`; `tests/test_hybrid_training.py:33` |
| `rate_weight` | production default `1e-3`; hybrid test uses `0.01` | loss multiplier | no | `>= 0` | mean squared RGC rates over populations | rate regularization | engineering-prior | `loss/retina.py:26`, `loss/retina.py:97-101`, `loss/retina.py:114`; `tests/test_hybrid_training.py:34` |
| `homeostasis_weight` | production default `1e-3`; hybrid test uses `0.02` | loss multiplier | no | `>= 0` | squared mean deviation from target rate for midget/parasol | firing-rate homeostasis | engineering-prior | `loss/retina.py:27`, `loss/retina.py:102-105`, `loss/retina.py:115`; `tests/test_hybrid_training.py:35` |
| `decorrelation_weight` | production default `1e-4`; hybrid test uses `0.001` | loss multiplier | no | `>= 0` | squared trace correlation between midget and parasol | midget/parasol decorrelation | engineering-prior | `loss/retina.py:28`, `loss/retina.py:106-109`, `loss/retina.py:132-146`; `tests/test_hybrid_training.py:36` |
| `residual_activity_weight` | production default `1e-3`; hybrid test uses `0.03` | loss multiplier | no | `>= 0` | absolute mean residual rate | discourage residual activity | engineering-prior | `loss/retina.py:29`, `loss/retina.py:110`, `loss/retina.py:117`; `tests/test_hybrid_training.py:37` |
| `residual_decoder_weight` | production default `1e-3`; hybrid test uses `0.04` | loss multiplier | no | `>= 0` | residual decoder weight penalty | discourage residual decoder reliance | engineering-prior | `loss/retina.py:30`, `loss/retina.py:118`; `models/decoder/local_decoder.py:178-182`; `tests/test_hybrid_training.py:38` |
| `target_rate` | production default `0.05`; hybrid test uses `0.1` | spikes/frame or rate unit | no | `[0,1]` | target in homeostasis term | desired midget/parasol mean rate | engineering-prior | `loss/retina.py:31`, `loss/retina.py:45-46`, `loss/retina.py:102-105`; `tests/test_hybrid_training.py:39` |
| RF loss / A3 / STP / gap coupling loss | not implemented yet | n/a | no | n/a | no field in `RetinaLossConfig`; tests assert no `rf_target` | confirms V1 excludes RF loss | pending-evidence | `loss/retina.py:23-31`; `tests/test_hybrid_training.py:95` |

### Hybrid Trainer

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `t_bptt` | production default `8`; hybrid test uses `2` | frames | no | `>= 1` | splits sequence into no-grad prefix plus train window | truncated BPTT length | engineering-prior | `training/hybrid.py:47-55`, `training/hybrid.py:97-102`; `tests/test_hybrid_training.py:56` |
| `grad_clip_norm` | production default `1.0`; can be `None` | gradient norm | no | `> 0` when set | `clip_grad_norm_((*core.parameters(), *decoder.parameters()), norm)` | gradient clipping | engineering-prior | `training/hybrid.py:49`, `training/hybrid.py:54-55`, `training/hybrid.py:134-138` |
| optimizer | no production factory; tests use `torch.optim.SGD((*core.parameters(), *decoder.parameters()), lr=0.05)` | optimizer config | no | PyTorch optimizer rules | one group with all core and decoder params | training update surface | test-only | `training/hybrid.py:66-79`; `tests/test_hybrid_training.py:47-50` |
| `TrainingStage.DECODER_WARMUP` | string `decoder_warmup` | enum | no | fixed enum | core forward inside `torch.no_grad()` | updates decoder without core gradients | engineering-prior | `training/hybrid.py:29-31`, `training/hybrid.py:105-112`; `tests/test_hybrid_training.py:98-123` |
| `TrainingStage.CORE_FINETUNE` | string `core_finetune` | enum | no | fixed enum | core forward with gradients; decoder still used normally | trains core; decoder may also update unless optimizer/group policy prevents it | engineering-prior | `training/hybrid.py:113-118`, `training/hybrid.py:122-139`; `tests/test_hybrid_training.py:126-143` |

### Smoke Gate Thresholds

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| real smoke gate config | not implemented yet | n/a | no | n/a | no `evaluation/`, no `configs/`, no gate module found | no production pass/fail thresholds | pending-evidence | repository scan; `evaluation/` and `configs/` absent |
| dataset smoke permission | `allow_fit_stats=True` in test only | boolean | no | explicit opt-in | smoke-like test uses same H5 stats only when allowed | data-contract smoke path | test-only | `data/dataset.py:128-134`; `tests/test_isetbio_data_contract.py:69-75` |
| transient decay expectation | `final_transient < 0.05 * final_sustained` | ratio | no | test assertion only | unit test threshold | checks high-pass adaptation | test-only | `tests/test_bipolar_cell.py:117-150` |
| A2 decay expectation | `state < 0.01 * peak` after 200 zero steps | ratio | no | test assertion only | unit test threshold | checks A2 recovery | test-only | `tests/test_amacrine_cell.py:94-109` |
| RGC rate decay expectation | `output.rates.midget < 0.01 * post_pulse_rate` after 200 zero steps | ratio | no | test assertion only | unit test threshold | checks rate/adaptation recovery | test-only | `tests/test_rgc_cell.py:144-172` |

## 3. Time-Constant Parameters

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| dataset `time_axis_seconds` | from H5; no hard default | seconds | no | stable interval CV <= `1e-3` | `_validate_time_axis` | source clock for ISETBio export | data-derived | `data/cone_response.py:41`, `data/cone_response.py:77-86` |
| dataset `dt_ms` | not computed into model configs yet | ms | no | n/a | no bridge from `time_axis_seconds` to config `dt_ms` | missing conversion point | pending-evidence | `data/dataset.py:180-182`; model configs accept manual `dt_ms` |
| prediction horizons | production default `(1,2,4)` | frames | no | positive future offsets | dataset target slicing | target time offsets | engineering-prior | `data/dataset.py:23`, `data/dataset.py:218-220` |
| H1 tau | test fixture `50.0`, bounds `10.0` to `200.0` | ms | yes | bounded sigmoid | `tau_min + range * sigmoid(raw_tau)` | H1 low-pass recurrent tau | test-only | `models/cells/horizontal.py:149-155`; `tests/test_horizontal_cell.py:20-22` |
| bipolar sustained tau | test fixture `80.0`, bounds `60.0` to `200.0` | ms | yes | bounded sigmoid | `tau_ms[SUSTAINED]`, leak `exp(-dt/tau)` | sustained bipolar low-pass | test-only | `models/cells/bipolar.py:70-80`; `tests/test_bipolar_cell.py:18-20` |
| bipolar transient tau | test fixture `20.0`, bounds `5.0` to `40.0` | ms | yes | bounded sigmoid; transient max below sustained min | `tau_ms[TRANSIENT]`, leak `exp(-dt/tau)` | transient bipolar low-pass after high-pass drive | test-only | `models/cells/bipolar.py:70-80`; `tests/test_bipolar_cell.py:21-23` |
| transient baseline tau | no separate value; uses sustained tau | ms | yes, indirectly | sustained tau bounds | `baseline_leak = leak_values[SUSTAINED]` | baseline subtraction timescale | test-only | `models/cells/bipolar.py:174-178` |
| A2 sustained tau | test fixture `100.0`, bounds `40.0` to `250.0` | ms | yes | bounded sigmoid | A2 leak `exp(-dt/tau)` | sustained A2 state decay | test-only | `models/cells/amacrine.py:153-163`; `tests/test_amacrine_cell.py:20-22` |
| A2 transient tau | test fixture `40.0`, bounds `15.0` to `100.0` | ms | yes | bounded sigmoid | A2 leak `exp(-dt/tau)` | transient A2 state decay | test-only | `models/cells/amacrine.py:153-163`; `tests/test_amacrine_cell.py:23-25` |
| RGC membrane tau | test fixture `20.0` | ms | no | `> 0` | buffer `exp(-dt/membrane_tau)` | membrane decay | test-only | `models/cells/rgc_runtime.py:22-25`; `tests/test_rgc_cell.py:37` |
| RGC adaptation tau | test fixture `80.0` | ms | no | `> 0` | buffer `exp(-dt/adaptation_tau)` | adaptation decay | test-only | `models/cells/rgc_runtime.py:26-29`; `tests/test_rgc_cell.py:38` |
| RGC rate tau | test fixture `50.0` | ms | no | `> 0` | buffer `exp(-dt/rate_tau)` | rate history low-pass | test-only | `models/cells/rgc_runtime.py:30-33`; `tests/test_rgc_cell.py:39` |
| `T_bptt` / `t_bptt` | production default `8`; test `2` | frames | no | `>= 1` | train prefix detached/no-grad | truncated BPTT horizon | engineering-prior | `training/hybrid.py:48`, `training/hybrid.py:97-102`; `tests/test_hybrid_training.py:56` |

Should receive human/macaque/marmoset evidence later: H1 tau, bipolar sustained/transient tau, transient baseline tau policy, A2 sustained/transient tau, RGC membrane/adaptation/rate tau, prediction horizon frame offsets, and `T_bptt` if presented as biologically motivated rather than engineering-only.

## 4. Spatial-Scale Parameters

All explicit spatial config names use `_degs` and all pools consume `positions_degs`. No code path uses micrometers. If a scale is only in tests and has no citation or data-derived factory, its evidence status remains `test-only` or `pending-evidence`.

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| H1 spacing | test fixture `0.2` | degrees | no | `> 0`; supported nodes fewer than cones | centered grid from cone extent | H1 node density | test-only | `models/cells/horizontal.py:48`, `models/cells/horizontal.py:258-305`; `tests/test_horizontal_cell.py:18` |
| H1 radius | test fixture `0.16` | degrees | no | `> 0` | local Gaussian support and node filter | cone to H1 support | test-only | `models/cells/horizontal.py:104-111`; `tests/test_horizontal_cell.py:14` |
| H1 sigma | test fixture `0.1` | degrees | no | `> 0` | Gaussian kernel | cone to H1 weighting | test-only | `models/cells/horizontal.py:106-111`; `tests/test_horizontal_cell.py:15` |
| H1 feedback radius | test fixture `0.21` | degrees | no | `> 0` | H1 to cone local Gaussian | surround feedback support | test-only | `models/cells/horizontal.py:112-117`; `tests/test_horizontal_cell.py:16` |
| H1 feedback sigma | test fixture `0.12` | degrees | no | `> 0` | H1 to cone Gaussian | surround feedback weighting | test-only | `models/cells/horizontal.py:112-117`; `tests/test_horizontal_cell.py:17` |
| A2 radius | tests use `0.15`; core fixture uses `0.16` | degrees | no | `> 0` | local Gaussian positions to positions | A2 pooling support | test-only | `models/cells/amacrine.py:100-102`; `tests/test_amacrine_cell.py:17`; `tests/test_retina_snn.py:60` |
| A2 sigma | test fixture `0.1` | degrees | no | `> 0` | Gaussian kernel | A2 pooling scale | test-only | `models/cells/amacrine.py:100-102`; `tests/test_amacrine_cell.py:18` |
| parasol radius | test fixture `0.16` | degrees | no | `> 0` | local Gaussian bipolar to parasol | parasol RF support | test-only | `models/cells/rgc.py:77-82`; `tests/test_rgc_cell.py:32` |
| parasol sigma | test fixture `0.1` | degrees | no | `> 0` | Gaussian kernel | parasol RF scale | test-only | `models/cells/rgc.py:77-82`; `tests/test_rgc_cell.py:33` |
| residual radius | test fixture `0.25` | degrees | no | `> 0` | local Gaussian bipolar to residual | residual RF support | test-only | `models/cells/rgc.py:83-88`; `tests/test_rgc_cell.py:34` |
| residual sigma | test fixture `0.12` | degrees | no | `> 0` | Gaussian kernel | residual RF scale | test-only | `models/cells/rgc.py:83-88`; `tests/test_rgc_cell.py:35` |
| fine target pooling | production `None`; tests use identity sparse matrix | `positions_degs`-aligned weights | no | sparse COO, row-stochastic | dataset target sparse matmul | fine target aggregation | data-derived | `data/dataset.py:150-154`, `data/dataset.py:229-233`; `tests/test_isetbio_data_contract.py:92` |
| coarse target pooling | production `None`; tests use pair averages | `positions_degs`-aligned weights | no | sparse COO, row-stochastic | dataset target sparse matmul | coarse target aggregation | data-derived | `data/dataset.py:155-158`, `data/dataset.py:234-237`; `tests/test_isetbio_data_contract.py:93-104` |
| decoder fine radius | tests use `0.11` or integration `0.16` | degrees | no | `> 0` | local Gaussian mask, empty rows allowed | local fine decoder support | test-only | `models/decoder/local_decoder.py:71-92`, `models/decoder/local_decoder.py:198-204`; `tests/test_local_decoder.py:38`; `tests/test_retina_snn.py:211` |
| decoder coarse radius | test fixture `0.21` | degrees | no | `> 0` | local Gaussian mask, empty rows allowed | local coarse decoder support | test-only | `models/decoder/local_decoder.py:93-114`; `tests/test_local_decoder.py:40` |
| mosaic stride / spacing | not implemented as production config | degrees | no | n/a | caller supplies mosaic positions | deterministic population layout missing | pending-evidence | `models/cells/rgc_types.py:16-21`; `tests/test_rgc_cell.py:18-27` |

## 5. Learnable Parameter Audit

| Parameter | Module | Bounded | Raw parameter name | Effective parameter formula | Optimizer group | Stage 1 freeze | Stage 2 train | Source file |
|---|---|---|---|---|---|---|---|---|
| H1 gain | H1 | yes | `raw_gain` | `gain_max * sigmoid(raw_gain)` | no named core group; included in test SGD all-params group | core forward under `torch.no_grad()`, no core update | yes, core forward tracks grad | `models/cells/horizontal.py:131-146`; `training/hybrid.py:105-118`; `tests/test_hybrid_training.py:47-56` |
| H1 tau | H1 | yes | `raw_tau` | `tau_min + (tau_max - tau_min) * sigmoid(raw_tau)` | same all-params optimizer | frozen by no-grad core pass | yes | `models/cells/horizontal.py:134-155` |
| bipolar sustained tau | Bipolar | yes | `raw_tau_sustained` | `_bounded(raw, (tau_sustained_min, tau_sustained_max))` | same all-params optimizer | frozen by no-grad core pass | yes | `models/cells/bipolar.py:35-44`, `models/cells/bipolar.py:70-80` |
| bipolar transient tau | Bipolar | yes | `raw_tau_transient` | `_bounded(raw, (tau_transient_min, tau_transient_max))` | same all-params optimizer | frozen by no-grad core pass | yes | `models/cells/bipolar.py:40-44`, `models/cells/bipolar.py:70-80` |
| bipolar sustained g_AB | Bipolar | yes | `raw_g_ab_sustained` | `g_ab_sustained_max * sigmoid(raw)` | same all-params optimizer | frozen by no-grad core pass | yes | `models/cells/bipolar.py:45-49`, `models/cells/bipolar.py:83-91` |
| bipolar transient g_AB | Bipolar | yes | `raw_g_ab_transient` | `g_ab_transient_max * sigmoid(raw)` | same all-params optimizer | frozen by no-grad core pass | yes | `models/cells/bipolar.py:50-54`, `models/cells/bipolar.py:83-91` |
| A2 sustained tau | A2 | yes | `raw_tau_sustained` | `_bounded(raw, (tau_sustained_min, tau_sustained_max))` | same all-params optimizer | frozen by no-grad core pass | yes | `models/cells/amacrine.py:117-126`, `models/cells/amacrine.py:153-163` |
| A2 transient tau | A2 | yes | `raw_tau_transient` | `_bounded(raw, (tau_transient_min, tau_transient_max))` | same all-params optimizer | frozen by no-grad core pass | yes | `models/cells/amacrine.py:122-126`, `models/cells/amacrine.py:153-163` |
| A2 sustained g_BA | A2 | yes | `raw_g_ba_sustained` | `g_ba_sustained_max * sigmoid(raw)` | same all-params optimizer | frozen by no-grad core pass | yes | `models/cells/amacrine.py:127-131`, `models/cells/amacrine.py:166-174` |
| A2 transient g_BA | A2 | yes | `raw_g_ba_transient` | `g_ba_transient_max * sigmoid(raw)` | same all-params optimizer | frozen by no-grad core pass | yes | `models/cells/amacrine.py:132-136`, `models/cells/amacrine.py:166-174` |
| RGC midget g_AG | RGC | yes | `raw_g_ag_midget` | `g_ag_midget_max * sigmoid(raw)` | same all-params optimizer | frozen by no-grad core pass | yes | `models/cells/rgc.py:108-134` |
| RGC parasol g_AG | RGC | yes | `raw_g_ag_parasol` | `g_ag_parasol_max * sigmoid(raw)` | same all-params optimizer | frozen by no-grad core pass | yes | `models/cells/rgc.py:112-134` |
| RGC residual g_AG | RGC | yes | `raw_g_ag_residual` | `g_ag_residual_max * sigmoid(raw)` | same all-params optimizer | frozen by no-grad core pass | yes | `models/cells/rgc.py:116-134` |
| decoder midget weights | Decoder | no | `_LocalProjection.raw_weight` in `fine_midget`, `coarse_midget` | `raw_weight` | no named decoder group; included in test SGD all-params group | yes, decoder receives gradients from frozen RGC output | yes, decoder also remains in optimizer | `models/decoder/local_decoder.py:71-77`, `models/decoder/local_decoder.py:206-216`; `training/hybrid.py:122-139` |
| decoder parasol weights | Decoder | no | `_LocalProjection.raw_weight` in `fine_parasol`, `coarse_parasol` | `raw_weight` | same all-params optimizer | yes | yes | `models/decoder/local_decoder.py:78-84`, `models/decoder/local_decoder.py:93-106` |
| decoder residual weights | Decoder | yes | `_LocalProjection.raw_weight` in `fine_residual`, `coarse_residual` | `residual_weight_max * tanh(raw_weight)` | same all-params optimizer | yes | yes | `models/decoder/local_decoder.py:85-92`, `models/decoder/local_decoder.py:107-114`, `models/decoder/local_decoder.py:212-216` |

Optimizer note: there is no implemented optimizer factory, no explicit `core` group, and no explicit `decoder` group. Tests build one SGD group over `(*core.parameters(), *decoder.parameters())` with `lr=0.05`.

## 6. Hard-coded Constants

| Constant | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| H5 format version | `retina-snn-cone-response-v1` | string | no | exact match | decoded text compare | data contract version | engineering-prior | `data/cone_response.py:27-29` |
| time-axis stability threshold | `1e-3` | relative variation | no | reject above threshold | `std(intervals)/(mean+1e-12)` | rejects unstable frame interval | engineering-prior | `data/cone_response.py:83-85` |
| time-axis denominator eps | `1e-12` | seconds | no | denominator floor additive | `mean + 1e-12` | avoid zero division | engineering-prior | `data/cone_response.py:83` |
| default log eps | `1e-6` | response units | no | `> 0` | defaults in log/stats functions | log and stats stability | engineering-prior | `data/dataset.py:24`, `data/dataset.py:39`, `data/dataset.py:45`, `data/dataset.py:75`, `data/dataset.py:92` |
| default clip | `5.0` | normalized log contrast | no | `> 0` | dataset clip | clips model inputs/targets | engineering-prior | `data/dataset.py:25`, `data/dataset.py:144-149` |
| stats position tolerance | `atol=1e-6` | degrees | no | allclose tolerance | export consistency check | rejects mismatched training exports | engineering-prior | `data/dataset.py:61-63` |
| pool row-sum tolerance | `1e-5` | row sum | no | allclose tolerance | target/H1/A2/RGC row-stochastic checks | sparse pool validation | engineering-prior | `data/dataset.py:287`, `models/cells/horizontal.py:310`, `models/cells/amacrine.py:104`, `models/cells/rgc_runtime.py:87` |
| Gaussian exponent coefficient | `-0.5` | unitless | no | fixed Gaussian form | `exp(-0.5 * square(distance/sigma))` | spatial weighting | engineering-prior | `data/geometry.py:27` |
| Gaussian normalization floor | `1e-12` | row sum | no | clamp minimum | `row_sum.clamp_min(1e-12)` | sparse row normalization stability | engineering-prior | `data/geometry.py:32` |
| ON/OFF count | `2` | channels | no | fixed shapes | enum and tensor shapes | polarity split | engineering-prior | `models/cells/bipolar_types.py:15-17`; `models/cells/bipolar.py:101-115` |
| sustained/transient count | `2` | channels | no | fixed shapes | enum and tensor shapes | kinetics split | engineering-prior | `models/cells/bipolar_types.py:20-22`; `models/cells/bipolar.py:101-115` |
| RGC threshold | tests use `0.2` | current/state units | no | positive | hard threshold | spike emission | test-only | `tests/test_rgc_cell.py:40`; `models/cells/rgc_runtime.py:50` |
| RGC surrogate slope | tests use `5.0` | inverse current/state units | no | positive | sigmoid surrogate | gradient through spikes | test-only | `tests/test_rgc_cell.py:41`; `models/cells/rgc_runtime.py:51-54` |
| RGC adaptation strength | tests use `0.1` | state scale | no | positive | adaptation update | spike-triggered adaptation | test-only | `tests/test_rgc_cell.py:42`; `models/cells/rgc_runtime.py:56-61` |
| residual drive scale | tests use `0.25` | unitless | no | `(0,1]` | residual current multiplier | suppress residual pathway | test-only | `models/cells/rgc_types.py:43`, `tests/test_rgc_cell.py:49` |
| loss defaults | `1.0`, `1e-3`, `1e-4`, `0.05` | loss/rate units | no | weights `>=0`, target rate `[0,1]` | `RetinaLossConfig` | objective terms | engineering-prior | `loss/retina.py:23-31` |
| gradient clip default | `1.0` | norm | no | `> 0` when set | `HybridTrainingConfig` | optimizer stability | engineering-prior | `training/hybrid.py:48-55`, `training/hybrid.py:134-138` |
| BPTT default | `8` | frames | no | `>= 1` | `HybridTrainingConfig` | train window length | engineering-prior | `training/hybrid.py:48`, `training/hybrid.py:97-102` |
| test optimizer lr | `0.05` | learning rate | no | optimizer-defined | SGD fixture only | hybrid training smoke update | test-only | `tests/test_hybrid_training.py:47-50` |
| decoder raw weight init | `0` | decoder weight | yes | residual effective bounded only | `torch.zeros(horizon_count, local_mask._nnz())` | neutral decoder initialization | engineering-prior | `models/decoder/local_decoder.py:206-208` |
| test decoder fill | `0.05` | decoder weight | yes | residual tanh bound if residual | test setup only | forces nonzero decoder path | test-only | `tests/test_retina_snn.py:213-215`; `tests/test_hybrid_training.py:129-131` |
| transient test threshold | `0.05` | ratio | no | assertion only | unit test | high-pass decay check | test-only | `tests/test_bipolar_cell.py:148-150` |
| A2/RGC decay thresholds | `0.01` | ratio | no | assertion only | unit tests | recovery checks | test-only | `tests/test_amacrine_cell.py:109`; `tests/test_rgc_cell.py:171` |

## 7. Evidence Gap List

P0:

- `dt_ms`: current code passes `5.0 ms` manually in tests, while ISETBio provides `time_axis_seconds`.
  - Why evidence is needed: all temporal tau and horizon interpretation depend on the true frame interval.
  - Recommended evidence source type: ISETBio.
- Bipolar sustained/transient taus: current test values are sustained `80 ms` and transient `20 ms`.
  - Why evidence is needed: these define the sustained/transient biological claim and high-pass behavior.
  - Recommended evidence source type: human / macaque / marmoset.
- RGC Adaptive LIF taus and threshold: current test values are membrane `20 ms`, adaptation `80 ms`, rate `50 ms`, threshold `0.2`.
  - Why evidence is needed: directly controls spike timing, rate smoothing, and parasol timing credibility.
  - Recommended evidence source type: macaque / marmoset / internal smoke statistics.
- Spatial RF radii and sigmas for H1, A2, parasol, residual, and decoder local masks.
  - Why evidence is needed: these determine population specificity and whether the model is biologically plausible in degrees.
  - Recommended evidence source type: human / macaque / marmoset / ISETBio.
- Missing real smoke gate thresholds.
  - Why evidence is needed: the repo has unit-test assertions but no production smoke gate for first-version training acceptance.
  - Recommended evidence source type: internal smoke statistics.

P1:

- H1 gain and tau: test value gain `0.01`, max `0.2`, tau `50 ms`.
  - Why evidence is needed: excessive surround subtraction could remove local contrast; weak gain may make H1 ineffective.
  - Recommended evidence source type: human cone/H1-inspired literature or internal smoke statistics.
- A2 g_BA and tau bounds: sustained `0.03` max `0.3`, transient `0.05` max `0.5`, taus `100/40 ms`.
  - Why evidence is needed: A2 strength and self-pooling can dominate bipolar/RGC inhibition.
  - Recommended evidence source type: mammalian retina physiology, macaque/marmoset where available.
- RGC g_AG bounds: midget/residual max `0.1`, parasol max `0.3`.
  - Why evidence is needed: determines A2 suppression and ON/OFF current sign balance.
  - Recommended evidence source type: macaque / marmoset / internal smoke statistics.
- Residual pathway scale and penalties: `residual_drive_scale=0.25`, residual activity/decoder defaults `1e-3`.
  - Why evidence is needed: too weak may hide residual utility; too strong may let residual absorb population-specific signals.
  - Recommended evidence source type: internal smoke statistics.
- Loss weights and target rate: defaults are engineering values, target rate `0.05`.
  - Why evidence is needed: affects training interpretation and spike-rate plausibility.
  - Recommended evidence source type: internal smoke statistics / macaque spike-rate ranges.

P2:

- Dataset clipping `5.0` and eps `1e-6`.
  - Why evidence is needed: mostly numerical, but clip fraction should be justified by training-set distribution.
  - Recommended evidence source type: ISETBio / internal smoke statistics.
- BPTT `8` and grad clip `1.0`.
  - Why evidence is needed: training stability parameters, not biological claims.
  - Recommended evidence source type: internal smoke statistics.
- Local decoder residual tanh bound `0.1`.
  - Why evidence is needed: engineering control of residual leakage.
  - Recommended evidence source type: internal smoke statistics.

## 8. Suggested Literature-Evidence Mapping

- human cone photoresponse -> input temporal smoothing or cone delay, if added later; current repo has no explicit cone temporal smoothing beyond data frame clock. citation needed.
- ISETBio `time_axis_seconds` -> `dt_ms` and horizon conversion. citation needed for the dataset/export contract if reported externally.
- human/macaque bipolar temporal response -> sustained vs transient bipolar tau bounds and baseline subtraction policy. citation needed.
- mammalian A2 amacrine temporal/spatial integration -> A2 tau, radius/sigma, g_BA bounds. citation needed.
- human/macaque midget/parasol RGC temporal RF -> midget sustained vs parasol transient constraint, RGC membrane/adaptation/rate tau, threshold/rate target. citation needed.
- macaque parasol spike timing/response correlation -> RGC `rate_tau_ms`, parasol transient scale, decorrelation loss interpretation. citation needed.
- marmoset RGC white-noise STA -> RF lag window and midget/parasol spatial/temporal comparison. citation needed.
- human/macaque RF size in degrees/eccentricity -> H1, A2, parasol, residual, decoder local radii/sigmas and mosaic spacing. citation needed.
- internal smoke statistics -> clipping range, loss weights, residual penalties, grad clip, BPTT, and acceptance thresholds. citation needed.

## 9. Red Flags

- `dt_ms` is not automatically derived from `time_axis_seconds`; current model fixtures manually use `5.0 ms`.
- There is no production default config factory, no `configs/` directory, and no CLI defaults; concrete V1 model values are test fixtures.
- Transient baseline subtraction uses the sustained bipolar tau as the baseline tau; there is no separate baseline timescale.
- H1 gain/tau, bipolar/A2/RGC gains, and biological tau bounds are bounded and trainable where implemented, but their current numeric ranges lack literature evidence.
- H1 node density is only constrained to be fewer than cones; no evidence-backed node ratio target exists.
- A2 self weight depends on local Gaussian radius/sigma and can become high when the support is narrow.
- RGC `rate_tau_ms=50 ms` may smooth parasol timing too aggressively unless supported by spike timing evidence.
- `residual_drive_scale=0.25` plus residual penalties are engineering priors; residual pathway credibility needs smoke statistics.
- Decoder local masks allow empty rows; this prevents crashes but can silently create target positions with no local support.
- The hybrid trainer has no explicit core/decoder optimizer groups; Stage 1 freezes core through `torch.no_grad()`, and Stage 2 appears to train both core and decoder when the optimizer contains both.
- Smoke gate thresholds are not implemented as a real config; only unit-test assertions exist.

## 10. Next Actions

- [ ] 补 human cone temporal evidence
- [ ] 补 human/macaque RGC temporal evidence
- [ ] 补 marmoset dataset RF/temporal filter evidence
- [ ] 确认 `dt_ms` 从 `time_axis_seconds` 自动计算
- [ ] 确认所有 `radius_degs` 单位一致
- [ ] 确认 Stage 1/2 optimizer group 中参数归属正确
- [ ] 补 bipolar sustained/transient tau 与 transient baseline tau 证据
- [ ] 补 H1/A2/parasol/residual/decoder spatial scale 证据
- [ ] 将真实 smoke gate 阈值从 unit-test assertion 独立成可审计配置
