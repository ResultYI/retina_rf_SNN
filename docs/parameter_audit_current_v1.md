# Parameter Audit: Retina SNN V1

## 1. Summary

This audit covers the current repository state of Retina SNN V1. It is a parameter inventory only; it does not change model mechanisms, defaults, or training behavior.

- Data-derived parameters: ISETBio `cone_response`, `cone_positions_degs`, `cone_types`, `time_axis_seconds`, `eye_trace_degs`, response units, normalization mean/scale when provided or explicitly fit, and optional target pooling matrices.
- Fixed structural hyperparameters: ON/OFF polarity count, sustained/transient channel count, sparse local support, nearest one-to-one foveal midget topology, row-stochastic pool checks, BPTT truncation, and loss weights.
- Bounded learnable parameters: H1 gain/tau, bipolar tau/g_AB, local amacrine tau/g_BA, RGC g_AG, RGC ordered kinetic preferences, decoder ON/OFF coefficients, three-value radial mixtures, and population-level temporal decays. They use raw `nn.Parameter` values with sigmoid, softmax, ordered, or tanh parameterization.
- Decoder locality is fixed. There are no per-edge learnable decoder logits; each RGC population shares one mixture over three row-normalized radial bases across every target position.
- Training hyperparameters: `HybridTrainingConfig.t_bptt`, `grad_clip_norm`, Stage-1 core/decoder AdamW groups, and stage behavior implemented through `torch.no_grad()` during decoder warmup.
- Parameters currently missing literature evidence: most biological time constants, spatial radii/sigmas, RGC LIF threshold/surrogate/adaptation constants, H1/A2/RGC gain bounds, residual population scale, decoder local masks, and smoke-gate thresholds. Current concrete model values mostly come from `configs/physiology_profiles.py` and Stage-1 factories, but literature support is still incomplete.
- Not implemented yet: A3/STP/gap coupling/RF loss and a real smoke gate threshold config.

## 2. Parameter Tables

### ISETBio Data / Dataset Config

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `h5_path` | required by caller | path | no | must load H5 contract | dataclass field | ISETBio cone export source | data-derived | `data/dataset.py:19` `ISETBioDatasetConfig`; `data/cone_response.py:24` `load_cone_response` |
| `format_version` | `retina-snn-cone-response-v1` | string | no | exact match | H5 dataset decode | rejects incompatible exports | engineering-prior | `data/cone_response.py:27-29` |
| `cone_response` | H5 dataset `cone_response` | response units from file | no | finite, non-negative, shape `[T,Ncone]` | `_logical_array` plus `validate_response` | model input before log normalization | data-derived | `data/cone_response.py:36`, `data/cone_response.py:68-74` |
| `positions_degs` | H5 dataset `cone_positions_degs` | degrees | no | shape `[Ncone,2]` | loaded into dataset and model geometry | spatial coordinates for H1, bipolar, A2, RGC, decoder | data-derived | `data/cone_response.py:37-39`; `data/dataset.py:162-174` |
| `cone_types` | H5 dataset `cone_types` | categorical id | no | shape `[Ncone]` | loaded and order-checked during stats fit | preserves cone ordering across exports | data-derived | `data/cone_response.py:40`; `data/dataset.py:65-66` |
| `time_axis_seconds` | H5 dataset `time_axis_seconds` | seconds | no | finite, strictly increasing, stable interval CV <= `1e-3` | `_validate_time_axis`; Stage-1 CLI derives `dt_ms` from it | data clock for profile construction | data-derived | `data/cone_response.py:41`, `data/cone_response.py:77-86`; `configs/physiology_profiles.py:35-48`; `scripts/train_stage1.py:147-164` |
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

### Masked Current Target and Target Pooling

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `mask_fraction` | production default `0.25` | fraction | no | strictly `(0,1)` | deterministic mask from `(mask_seed, anchor)` | engineering corruption rate; not physiology | engineering-prior | `data/dataset.py`; `scripts/train_stage1.py` |
| `mask_seed` | production default `7` | integer seed | no | non-negative | deterministic per-anchor mask | repeatable train/evaluation corruption | engineering-prior | `data/dataset.py` |
| `target_current` | clean normalized log contrast at anchor | normalized log contrast | no | finite after clipping | direct anchor slice before masking | current reconstruction target | data-derived | `data/dataset.py` |
| `loss_mask_current` | anchor cone mask | weight | no | 0/1, positive total | selected cone entries are zeroed only in the anchor input frame | restricts loss to current information recoverable from visible history and neighbors | engineering-prior | `data/dataset.py` |
| `target_fine_pool` | production default `None`; tests use identity sparse pool | sparse row-stochastic weights | no | sparse COO, finite, non-negative, shape `[Ntarget,Ncone]`, rows sum to 1 | `_validate_target_pool`; `torch.sparse.mm` | optional fine target pooling | data-derived | `data/dataset.py:27`, `data/dataset.py:150-154`, `data/dataset.py:272-289`; `tests/test_isetbio_data_contract.py:92` |
| `target_coarse_pool` | production default `None`; tests use two-row average pool | sparse row-stochastic weights | no | same as `target_fine_pool`; must be provided together | `_validate_target_pool`; `torch.sparse.mm` | optional coarse target pooling | data-derived | `data/dataset.py:28`, `data/dataset.py:155-158`, `data/dataset.py:229-237`; `tests/test_isetbio_data_contract.py:93-104` |

### Deterministic Midget / Parasol / Residual Mosaic

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `RGCMosaic.bipolar_positions_degs` | production Stage-1 uses cone positions | degrees | no | shape `[N,2]`, finite | Stage-1 factory plus dataclass validation | source positions for RGC pooling | data-derived / engineering-prior | `training/stage1.py:129-134`; `models/cells/rgc_types.py:16-21` |
| `midget_positions_degs` | foveal mode uses cone positions; convergent mode spatially subsamples | degrees | no | foveal one-to-one or lower-density local mosaic | Stage-1 `MidgetSamplingMode` | midget population layout | data-derived / engineering-prior | `training/stage1.py:118-127`; `models/cells/rgc.py:54-76` |
| `parasol_positions_degs` | production Stage-1 spatial subsample of cone positions | degrees | no | `residual_count <= parasol_count < midget_count` | local Gaussian pool | parasol population layout | engineering-prior | `training/stage1.py:110-117`; `models/cells/rgc.py:61-82` |
| `residual_positions_degs` | production Stage-1 subsample of parasol positions | degrees | no | `residual_count <= parasol_count` | local Gaussian pool | residual population layout | engineering-prior | `training/stage1.py:114-117`; `models/cells/rgc.py:50-88` |
| deterministic mosaic generator | implemented in Stage-1 factory | n/a | no | stride validation in `Stage1BuildConfig` | derives midget/parasol/residual positions | production midget/parasol/residual mosaic path | engineering-prior | `training/stage1.py:33-67`, `training/stage1.py:99-134` |

### H1 Horizontal Network

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `radius_degs` | profile `1.75 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian cone to H1 support | H1 node support radius | engineering-prior | `configs/physiology_profiles.py:71-83`; `models/cells/horizontal.py:43-55` |
| `sigma_degs` | profile `0.90 * cone_spacing_deg` | degrees | no | `> 0` | Gaussian `exp(-0.5 * (distance/sigma)^2)` | H1 input weighting scale | engineering-prior | `configs/physiology_profiles.py:71-83`; `data/geometry.py:13-32` |
| `feedback_radius_degs` | profile `1.75 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian H1 to cone feedback | surround feedback support | engineering-prior | `configs/physiology_profiles.py:71-83`; `models/cells/horizontal.py:112-117` |
| `feedback_sigma_degs` | profile `0.90 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian H1 to cone feedback | surround feedback weighting scale | engineering-prior | `configs/physiology_profiles.py:71-83`; `models/cells/horizontal.py:112-117` |
| `h1_spacing_degs` | profile `1.45 * cone_spacing_deg` | degrees | no | `> 0`; supported H1 nodes must be fewer than cones | centered grid then support filter | H1 node spacing | engineering-prior | `configs/physiology_profiles.py:71-83`; `models/cells/horizontal.py:97-105` |
| `dt_ms` | derived from H5 `time_axis_seconds` in Stage-1 entrypoint | ms | no | `> 0` | stored as `_dt_ms` | temporal leak clock | data-derived | `configs/physiology_profiles.py:35-48`; `scripts/train_stage1.py:147-164` |
| `initial_tau_ms` | profile `50.0`, bounds `10.0` to `200.0` | ms | yes | `tau_min_ms < initial < tau_max_ms` | `raw_tau = logit((initial-min)/(max-min))`; effective `min + range * sigmoid(raw_tau)` | H1 recurrent low-pass tau | engineering-prior | `configs/physiology_profiles.py:71-83`; `models/cells/horizontal.py:134-151` |
| `initial_gain` | profile `0.01`, max `0.2` | unitless gain | yes | `0 < initial_gain < gain_max` | `raw_gain = logit(initial/gain_max)`; effective `gain_max * sigmoid(raw_gain)` | subtracts H1 surround from cone drive | engineering-prior | `configs/physiology_profiles.py:71-83`; `models/cells/horizontal.py:131-146` |
| `debug_checks` | default `True` | boolean | no | boolean | finite checks on input/state | runtime guard | engineering-prior | `models/cells/horizontal.py:55`, `models/cells/horizontal.py:183-195` |

### ON/OFF Sustained/Transient Bipolar Layer

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| polarity channels | `ON=0`, `OFF=1` | index | no | fixed two channels | `BipolarPolarity` enum | splits positive/negative drive | engineering-prior | `models/cells/bipolar_types.py:15-17`; `models/cells/bipolar.py:131-134` |
| kinetics channels | `SUSTAINED=0`, `TRANSIENT=1` | index | no | fixed two channels | `BipolarKinetics` enum | separates low-pass and high-pass outputs | engineering-prior | `models/cells/bipolar_types.py:20-22`; `models/cells/bipolar.py:154-155` |
| `dt_ms` | derived from H5 `time_axis_seconds` in Stage-1 entrypoint | ms | no | `> 0` | `_dt_ms` | leak clock | data-derived | `configs/physiology_profiles.py:35-48`; `scripts/train_stage1.py:147-164` |
| `initial_tau_sustained_ms` | profile `80.0`, bounds `20.0` to `200.0` | ms | yes | bounded positive | raw sigmoid bounded tau | sustained bipolar leak | engineering-prior | `configs/physiology_profiles.py:84-96`; `models/cells/bipolar.py:35-80` |
| `initial_tau_transient_ms` | profile `20.0`, bounds `5.0` to `120.0` | ms | yes | bounded positive; ordered relative to sustained | raw sigmoid bounded tau | transient bipolar leak | engineering-prior | `configs/physiology_profiles.py:84-96`; `models/cells/bipolar.py:40-80` |
| `initial_g_ab_sustained` | profile `0.01`, max `0.10` | unitless inhibition gain | yes | `0.0` to max | `g_ab_max * sigmoid(raw_g_ab_sustained)` | A2 to sustained bipolar inhibition | engineering-prior | `configs/physiology_profiles.py:84-96`; `models/cells/bipolar.py:45-173` |
| `initial_g_ab_transient` | profile `0.01`, max `0.30` | unitless inhibition gain | yes | `0.0` to max | `g_ab_max * sigmoid(raw_g_ab_transient)` | A2 to transient bipolar inhibition | engineering-prior | `configs/physiology_profiles.py:84-96`; `models/cells/bipolar.py:50-54` |
| polarity gain ON/OFF | initial `1.0/1.0` | unitless gain | yes | `0.25` to `4.0` | bounded sigmoid latent, fixed ON/OFF signs | permits response-scale calibration without polarity reversal | latent model parameter (D) | `models/cells/bipolar_types.py`; `models/cells/bipolar.py` |
| polarity threshold ON/OFF | initial `0.0/0.0` | normalized contrast drive | yes | `-1.0` to `1.0` | bounded sigmoid latent | permits bounded operating-point asymmetry | latent model parameter (D) | `models/cells/bipolar_types.py`; `models/cells/bipolar.py` |
| rectifier softness | initial `0.05` | normalized drive | yes | `0.01` to `0.50` | `s*softplus(x/s)` | replaces hard ReLU while preserving non-negative output | latent model parameter (D) | `models/cells/bipolar_types.py`; `models/cells/bipolar.py` |

### Transient High-Pass / Baseline Subtraction

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| transient baseline state | initialized to zeros | normalized drive | no | same shape as `[batch,2,Ncone]` | `BipolarState.transient_baseline` | stores polarity-specific baseline | engineering-prior | `models/cells/bipolar_types.py:43-47`, `models/cells/bipolar.py:100-116` |
| transient drive | `s*softplus((private_drive-state.transient_baseline)/s)` | normalized drive | indirectly through learned `s` | non-negative smooth transfer | high-pass subtraction before transient channel | transient response to increases above baseline with nonzero weak-side gradient | latent model parameter (D) | `models/cells/bipolar.py` |
| baseline tau | uses sustained bipolar tau, no separate config | ms | yes, indirectly through sustained tau | sustained tau bounds | `baseline_leak = leak_values[SUSTAINED]` | baseline adaptation for transient subtraction | test-only | `models/cells/bipolar.py:174-178` |
| separate transient baseline tau | not implemented yet | n/a | no | n/a | no config field | no independent baseline timescale | pending-evidence | `models/cells/bipolar_types.py:50-61`; `models/cells/bipolar.py:174` |

### A2 Amacrine Layer

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `radius_degs` | profile `3.60 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian positions to positions | A2 spatial pooling radius | engineering-prior | `configs/physiology_profiles.py:97-111`; `models/cells/amacrine.py:40`, `models/cells/amacrine.py:100-102` |
| `sigma_degs` | profile `1.80 * cone_spacing_deg` | degrees | no | `> 0` | Gaussian weighting | A2 spatial pooling scale | engineering-prior | `configs/physiology_profiles.py:97-111`; `models/cells/amacrine.py:41`, `data/geometry.py:27` |
| `dt_ms` | derived from H5 `time_axis_seconds` in Stage-1 entrypoint | ms | no | `> 0` | `_dt_ms` | leak clock | data-derived | `configs/physiology_profiles.py:35-48`; `scripts/train_stage1.py:147-164`; `models/cells/amacrine.py:42`, `models/cells/amacrine.py:137` |
| `initial_tau_sustained_ms` | profile `100.0`, bounds `20.0` to `250.0` | ms | yes | bounded positive | raw sigmoid bounded tau | A2 sustained channel state tau | engineering-prior | `configs/physiology_profiles.py:97-111`; `models/cells/amacrine.py:43-45`, `models/cells/amacrine.py:117-121`, `models/cells/amacrine.py:153-163` |
| `initial_tau_transient_ms` | profile `40.0`, bounds `15.0` to `180.0` | ms | yes | bounded positive | raw sigmoid bounded tau | A2 transient channel state tau | engineering-prior | `configs/physiology_profiles.py:97-111`; `models/cells/amacrine.py:46-48`, `models/cells/amacrine.py:122-126` |
| `initial_g_ba_sustained` | profile `0.03`, max `0.30` | unitless drive gain | yes | `0.0` to max | `g_ba_max * sigmoid(raw_g_ba_sustained)` | scales pooled sustained bipolar output into A2 state | engineering-prior | `configs/physiology_profiles.py:97-111`; `models/cells/amacrine.py:49-50`, `models/cells/amacrine.py:127-131`, `models/cells/amacrine.py:166-174`, `models/cells/amacrine.py:230-233` |
| `initial_g_ba_transient` | profile `0.05`, max `0.50` | unitless drive gain | yes | `0.0` to max | `g_ba_max * sigmoid(raw_g_ba_transient)` | scales pooled transient bipolar output into A2 state | engineering-prior | `configs/physiology_profiles.py:97-111`; `models/cells/amacrine.py:51-52`, `models/cells/amacrine.py:132-136` |
| self weight diagnostics | computed from local pool diagonal | row-stochastic weight | no | pool rows sum to 1 | `self_weight_mean`, `self_weight_max` buffers | monitors self-vs-neighbor pooling | data-derived | `models/cells/amacrine.py:107-116` |

### RGC Midget / Parasol / Residual Population Layer

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `parasol_radius_degs` | profile `3.60 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian bipolar to parasol pool | parasol spatial integration radius | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:26`, `models/cells/rgc.py:77-82` |
| `parasol_sigma_degs` | profile `1.80 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian sigma | parasol spatial weighting | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:27`, `models/cells/rgc.py:77-82` |
| `residual_radius_degs` | profile `5.80 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian bipolar to residual pool | residual spatial integration radius | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:28`, `models/cells/rgc.py:83-88` |
| `residual_sigma_degs` | profile `2.90 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian sigma | residual spatial weighting | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:29`, `models/cells/rgc.py:83-88` |
| `raw_pool_values.{midget,parasol,residual}` | initialized as log local-pool weights | edge logits | yes | effective weights are positive, fixed-support and row-normalized | sparse row-softmax | lets the task loss shape BC-to-RGC weights without RF targets | latent model parameter (D) | `models/cells/rgc.py`; `models/cells/rgc_runtime.py`; `tests/test_rgc_cell.py` |
| `initial_g_ag_midget` | profile `0.01`, max `0.10` | unitless inhibition gain | yes | `0.0` to max | `g_ag_max * sigmoid(raw_g_ag_midget)` | A2 sustained inhibition on midget current | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:37-38`, `models/cells/rgc.py:108-111`, `models/cells/rgc.py:194-204` |
| `initial_g_ag_parasol` | profile `0.03`, max `0.30` | unitless inhibition gain | yes | `0.0` to max; parasol max must exceed midget max | `g_ag_max * sigmoid(raw_g_ag_parasol)` | A2 transient inhibition on parasol current | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:39-40`, `models/cells/rgc.py:112-115` |
| `initial_g_ag_residual` | profile `0.01`, max `0.10` | unitless inhibition gain | yes | `0.0` to max | `g_ag_max * sigmoid(raw_g_ag_residual)` | A2 inhibition on residual current | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:41-42`, `models/cells/rgc.py:116-119` |
| `residual_drive_scale` | profile `0.25` | unitless scale | no | `(0,1]` | multiplied onto residual current | down-scales residual population drive | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:43`, `models/cells/rgc_types.py:79-80`, `models/cells/rgc.py:199-204` |
| `raw_midget_sustained_preference` | initialized `0`, effective mix `[0.75,0.25]` | latent logit | yes | sustained share stays `>0.5` | ordered sigmoid midpoint | midget sustained>transient relative order | engineering-prior (D/E) | `models/cells/rgc.py:116-145`; `tests/test_rgc_cell.py:145-160` |
| `raw_parasol_transient_preference` | initialized `0`, effective mix `[0.25,0.75]` | latent logit | yes | transient share stays `>0.5` | ordered sigmoid midpoint | parasol transient>sustained relative order | engineering-prior (D/E) | `models/cells/rgc.py:116-145`; `tests/test_rgc_cell.py:145-160` |

### RGC Adaptive LIF

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `dt_ms` | derived from H5 `time_axis_seconds` in Stage-1 entrypoint | ms | no | `> 0` | used in three leak buffers | clock for RGC dynamics | data-derived | `configs/physiology_profiles.py:35-48`; `scripts/train_stage1.py:147-164`; `models/cells/rgc_types.py:30`, `models/cells/rgc_runtime.py:22-33` |
| `membrane_tau_ms` | profile `20.0`, bounds `5.0` to `80.0` | ms | no | `> 0` | `membrane_leak = exp(-dt/tau)` | membrane state decay | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:31`, `models/cells/rgc_runtime.py:22-25` |
| `adaptation_tau_ms` | profile `80.0`, bounds `20.0` to `250.0` | ms | no | `> 0` | `adaptation_leak = exp(-dt/tau)` | spike-triggered adaptation decay | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:32`, `models/cells/rgc_runtime.py:26-29` |
| `rate_tau_ms` | profile `50.0` | ms | no | `> 0` | `rate_leak = exp(-dt/tau)` | low-pass rate history | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:33`, `models/cells/rgc_runtime.py:30-33` |
| `threshold` | profile `0.20` | current/state units | no | `> 0` through shared positive check | hard threshold plus surrogate | spike threshold | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:34`, `models/cells/rgc_runtime.py:34`, `models/cells/rgc_runtime.py:50-54` |
| `surrogate_slope` | profile `5.0` | inverse current/state units | no | `> 0` through shared positive check | `sigmoid(slope * (pre_reset - threshold))` | surrogate gradient steepness | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:35`, `models/cells/rgc_runtime.py:35`, `models/cells/rgc_runtime.py:51-54` |
| `adaptation_strength` | profile `0.10` | state increment scale | no | `> 0` through shared positive check | `(1-adaptation_leak) * adaptation_strength * spikes` | spike-triggered adaptation increment | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:36`, `models/cells/rgc_runtime.py:36`, `models/cells/rgc_runtime.py:56-61` |

### Local Decoder

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `current_radius_degs` | profile `3.60 * cone_spacing_deg` | degrees | no | `> 0` | fixed local support from RGC positions to reconstruction targets | locality bound for current-frame reconstruction | engineering-prior | `configs/physiology_profiles.py`; `models/decoder/local_decoder.py` |
| `current_sigma_degs` | profile `1.80 * cone_spacing_deg` | degrees | no | `> 0` | center scale for three fixed radial bases at `0.5×`, `1×`, and `2×` | capacity-controlled spatial readout basis; not a physiological RF width | engineering-prior | `configs/physiology_profiles.py`; `models/decoder/local_decoder.py` |
| decoder `raw_weight` | initialized to `0` | decoder weight | yes | effective `[-current_weight_max,current_weight_max]` through `tanh` | one ON/OFF pair per population | bounded population readout | engineering-prior (E) | `models/decoder/local_decoder.py` |
| decoder `raw_basis_mix` | initialized mostly on the center radial basis | basis logits | yes | three-value softmax per population | convex mixture of fixed, row-normalized radial bases shared over positions | low-capacity dynamic spatial projection | engineering-prior (E) | `models/decoder/local_decoder.py`; `tests/test_local_decoder.py` |

### Loss / Regularization

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| current reconstruction | coefficient fixed at `1.0` | normalized contrast MSE | no | n/a | masked MSE on the anchor target | preserve useful local stimulus information | task-defined | `loss/retina.py`; `data/dataset.py` |
| `energy_weight` | production default `0.10`; hybrid test uses `0.01` | loss multiplier | no | `>= 0` | linear mean RGC spike cost | engineering Lagrange multiplier, not a metabolic constant | engineering-prior | `loss/retina.py`; `tests/test_hybrid_training.py` |
| `homeostasis_weight` | production default `1e-3`; hybrid test uses `0.02` | loss multiplier | no | `>= 0` | squared population-rate deviation outside the band | weak anti-silence/anti-saturation guard | engineering-prior | `loss/retina.py`; `tests/test_hybrid_training.py` |
| homeostasis rate band | production defaults `0.01` to `0.20` | filtered spike rate | no | `0 <= min < max <= 1` | lower/upper band for each RGC population | prevents energy minimization by population collapse | engineering-prior | `loss/retina.py` |
| RF loss / residual / STP / gap coupling loss | not implemented | n/a | no | n/a | absent from `RetinaLossConfig` | keeps RF and extra mechanisms out of the training target | design constraint | `loss/retina.py` |

### Hybrid Trainer

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| `t_bptt` | production default `8`; hybrid test uses `2` | frames | no | `>= 1` | splits sequence into no-grad prefix plus train window | truncated BPTT length | engineering-prior | `training/hybrid.py:47-55`, `training/hybrid.py:97-102`; `tests/test_hybrid_training.py:56` |
| `grad_clip_norm` | production default `1.0`; can be `None` | gradient norm | no | `> 0` when set | `clip_grad_norm_((*core.parameters(), *decoder.parameters()), norm)` | gradient clipping | engineering-prior | `training/hybrid.py:49`, `training/hybrid.py:54-55`, `training/hybrid.py:134-138` |
| optimizer | production `build_stage1_optimizer` with core/decoder AdamW groups; some unit tests still use local SGD fixtures | optimizer config | no | PyTorch optimizer rules | separate core and decoder parameter groups | training update surface | engineering-prior | `training/stage1.py:164`; `scripts/train_stage1.py:26,63`; `tests/test_hybrid_training.py:47-50` |
| `TrainingStage.DECODER_WARMUP` | string `decoder_warmup` | enum | no | fixed enum | core forward inside `torch.no_grad()` | updates decoder without core gradients | engineering-prior | `training/hybrid.py:29-31`, `training/hybrid.py:105-112`; `tests/test_hybrid_training.py:98-123` |
| `TrainingStage.CORE_FINETUNE` | string `core_finetune` | enum | no | fixed enum | core forward with gradients; decoder still used normally | trains core; decoder may also update unless optimizer/group policy prevents it | engineering-prior | `training/hybrid.py:113-118`, `training/hybrid.py:122-139`; `tests/test_hybrid_training.py:126-143` |

### Smoke Gate Thresholds

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| real smoke gate threshold config | not implemented yet | n/a | no | n/a | `evaluation/` and `configs/` exist, but no versioned acceptance-threshold config is defined | no auditable production pass/fail thresholds | pending-evidence | `evaluation/`; `configs/`; `evaluation/feasibility.py:69-80` |
| dataset smoke permission | `allow_fit_stats=True` in test only | boolean | no | explicit opt-in | smoke-like test uses same H5 stats only when allowed | data-contract smoke path | test-only | `data/dataset.py:128-134`; `tests/test_isetbio_data_contract.py:69-75` |
| transient decay expectation | `final_transient < 0.05 * final_sustained` | ratio | no | test assertion only | unit test threshold | checks high-pass adaptation | test-only | `tests/test_bipolar_cell.py:117-150` |
| A2 decay expectation | `state < 0.01 * peak` after 200 zero steps | ratio | no | test assertion only | unit test threshold | checks A2 recovery | test-only | `tests/test_amacrine_cell.py:94-109` |
| RGC rate decay expectation | `output.rates.midget < 0.01 * post_pulse_rate` after 200 zero steps | ratio | no | test assertion only | unit test threshold | checks rate/adaptation recovery | test-only | `tests/test_rgc_cell.py:144-172` |

## 3. Time-Constant Parameters

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| dataset `time_axis_seconds` | from H5; no hard default | seconds | no | stable interval CV <= `1e-3` | `_validate_time_axis` | source clock for ISETBio export | data-derived | `data/cone_response.py:41`, `data/cone_response.py:77-86` |
| dataset `dt_ms` | derived from H5 `time_axis_seconds` in Stage-1 entrypoint | ms | no | positive stable interval | `dt_ms_from_time_axis_seconds` then `Stage1BuildConfig.dt_ms` | data clock bridge into model profiles | data-derived | `configs/physiology_profiles.py:35-48`; `training/stage1.py:34-45`; `scripts/train_stage1.py:147-164` |
| mask fraction | production default `0.25` | fraction | no | strictly `(0,1)` | per-anchor deterministic cone-column masking | non-predictive reconstruction corruption level | engineering-prior | `data/dataset.py`; `scripts/train_stage1.py` |
| H1 tau | profile `50.0`, bounds `10.0` to `200.0` | ms | yes | bounded sigmoid | `tau_min + range * sigmoid(raw_tau)` | H1 low-pass recurrent tau | engineering-prior | `configs/physiology_profiles.py:71-83`; `models/cells/horizontal.py:149-155` |
| bipolar sustained tau | profile `80.0`, bounds `20.0` to `200.0` | ms | yes | bounded sigmoid | `tau_ms[SUSTAINED]`, leak `exp(-dt/tau)` | sustained bipolar low-pass | engineering-prior | `configs/physiology_profiles.py:84-96`; `models/cells/bipolar.py:70-80` |
| bipolar transient tau | profile `20.0`, bounds `5.0` to `120.0` | ms | yes | bounds overlap sustained range; fitted model enforces `tau_transient < tau_sustained` | `tau_ms[TRANSIENT]`, leak `exp(-dt/tau)` | transient bipolar low-pass after high-pass drive | engineering-prior | `configs/physiology_profiles.py`; `models/cells/bipolar.py` |
| transient baseline tau | no separate value; uses sustained tau | ms | yes, indirectly | sustained tau bounds | `baseline_leak = leak_values[SUSTAINED]` | baseline subtraction timescale | test-only | `models/cells/bipolar.py:174-178` |
| A2 sustained tau | profile `100.0`, bounds `20.0` to `250.0` | ms | yes | bounded sigmoid | A2 leak `exp(-dt/tau)` | sustained A2 state decay | engineering-prior | `configs/physiology_profiles.py:97-111`; `models/cells/amacrine.py:153-163` |
| A2 transient tau | profile `40.0`, bounds `15.0` to `180.0` | ms | yes | bounded sigmoid | A2 leak `exp(-dt/tau)` | transient A2 state decay | engineering-prior | `configs/physiology_profiles.py:97-111`; `models/cells/amacrine.py:153-163` |
| RGC membrane tau | profile `20.0`, bounds `5.0` to `80.0` | ms | no | `> 0` | buffer `exp(-dt/membrane_tau)` | membrane decay | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_runtime.py:22-25` |
| RGC adaptation tau | profile `80.0`, bounds `20.0` to `250.0` | ms | no | `> 0` | buffer `exp(-dt/adaptation_tau)` | adaptation decay | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_runtime.py:26-29` |
| RGC rate tau | profile `50.0` | ms | no | `> 0` | buffer `exp(-dt/rate_tau)` | rate history low-pass | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_runtime.py:30-33` |
| `T_bptt` / `t_bptt` | production default `8`; test `2` | frames | no | `>= 1` | train prefix detached/no-grad | truncated BPTT horizon | engineering-prior | `training/hybrid.py:48`, `training/hybrid.py:97-102`; `tests/test_hybrid_training.py:56` |

Should receive human/macaque/marmoset evidence later: H1 tau, bipolar sustained/transient tau, transient baseline tau policy, local amacrine sustained/transient tau, RGC membrane/adaptation/rate tau, and `T_bptt` if presented as biologically motivated rather than engineering-only. `mask_fraction` remains an engineering protocol parameter.

## 4. Spatial-Scale Parameters

All explicit spatial config names use `_degs` and all pools consume `positions_degs`. No code path uses micrometers. Current production spatial constants are profile-derived from `cone_spacing_deg`; values that exist only in tests remain explicitly marked `test-only`.

| Parameter | Current value / default | Unit | Learnable | Bounds | Implementation | Role | Evidence status | Source file |
|---|---:|---|---|---|---|---|---|---|
| H1 spacing | profile `1.45 * cone_spacing_deg` | degrees | no | `> 0`; supported nodes fewer than cones | centered grid from cone extent | H1 node density | engineering-prior | `configs/physiology_profiles.py:71-83`; `models/cells/horizontal.py:48`, `models/cells/horizontal.py:258-305` |
| H1 radius | profile `1.75 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian support and node filter | cone to H1 support | engineering-prior | `configs/physiology_profiles.py:71-83`; `models/cells/horizontal.py:104-111` |
| H1 sigma | profile `0.90 * cone_spacing_deg` | degrees | no | `> 0` | Gaussian kernel | cone to H1 weighting | engineering-prior | `configs/physiology_profiles.py:71-83`; `models/cells/horizontal.py:106-111` |
| H1 feedback radius | profile `1.75 * cone_spacing_deg` | degrees | no | `> 0` | H1 to cone local Gaussian | surround feedback support | engineering-prior | `configs/physiology_profiles.py:71-83`; `models/cells/horizontal.py:112-117` |
| H1 feedback sigma | profile `0.90 * cone_spacing_deg` | degrees | no | `> 0` | H1 to cone Gaussian | surround feedback weighting | engineering-prior | `configs/physiology_profiles.py:71-83`; `models/cells/horizontal.py:112-117` |
| A2 radius | profile `3.60 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian positions to positions | A2 pooling support | engineering-prior | `configs/physiology_profiles.py:97-111`; `models/cells/amacrine.py:100-102` |
| A2 sigma | profile `1.80 * cone_spacing_deg` | degrees | no | `> 0` | Gaussian kernel | A2 pooling scale | engineering-prior | `configs/physiology_profiles.py:97-111`; `models/cells/amacrine.py:100-102` |
| parasol radius | profile `3.60 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian bipolar to parasol | parasol RF support | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc.py:77-82` |
| parasol sigma | profile `1.80 * cone_spacing_deg` | degrees | no | `> 0` | Gaussian kernel | parasol RF scale | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc.py:77-82` |
| residual radius | profile `5.80 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian bipolar to residual | residual RF support | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc.py:83-88` |
| residual sigma | profile `2.90 * cone_spacing_deg` | degrees | no | `> 0` | Gaussian kernel | residual RF scale | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc.py:83-88` |
| fine target pooling | production `None`; tests use identity sparse matrix | `positions_degs`-aligned weights | no | sparse COO, row-stochastic | dataset target sparse matmul | fine target aggregation | data-derived | `data/dataset.py:150-154`, `data/dataset.py:229-233`; `tests/test_isetbio_data_contract.py:92` |
| coarse target pooling | production `None`; tests use pair averages | `positions_degs`-aligned weights | no | sparse COO, row-stochastic | dataset target sparse matmul | coarse target aggregation | data-derived | `data/dataset.py:155-158`, `data/dataset.py:234-237`; `tests/test_isetbio_data_contract.py:93-104` |
| decoder fine radius | profile `1.50 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian mask, empty rows allowed | local fine decoder support | engineering-prior | `configs/physiology_profiles.py:138-146`; `models/decoder/local_decoder.py:71-92`, `models/decoder/local_decoder.py:198-204` |
| decoder coarse radius | profile `3.60 * cone_spacing_deg` | degrees | no | `> 0` | local Gaussian mask, empty rows allowed | local coarse decoder support | engineering-prior | `configs/physiology_profiles.py:138-146`; `models/decoder/local_decoder.py:93-114` |
| mosaic stride / spacing | production `Stage1BuildConfig` has midget/parasol/residual stride controls | degrees | no | positive integer strides | Stage-1 factory derives population positions | deterministic population layout path | engineering-prior | `training/stage1.py:33-67`, `training/stage1.py:110-127`; `models/cells/rgc_types.py:16-21` |

## 5. Learnable Parameter Audit

| Parameter | Module | Bounded | Raw parameter name | Effective parameter formula | Optimizer group | Stage 1 freeze | Stage 2 train | Source file |
|---|---|---|---|---|---|---|---|---|
| H1 gain | H1 | yes | `raw_gain` | `gain_max * sigmoid(raw_gain)` | core optimizer group | core forward under `torch.no_grad()`, no core update | yes, core forward tracks grad | `models/cells/horizontal.py:131-146`; `training/hybrid.py:105-118`; `training/stage1.py:164` |
| H1 tau | H1 | yes | `raw_tau` | `tau_min + (tau_max - tau_min) * sigmoid(raw_tau)` | core optimizer group | frozen by no-grad core pass | yes | `models/cells/horizontal.py:134-155`; `training/stage1.py:164` |
| bipolar sustained tau | Bipolar | yes | `raw_tau_sustained` | `_bounded(raw, (tau_sustained_min, tau_sustained_max))` | core optimizer group | frozen by no-grad core pass | yes | `models/cells/bipolar.py:35-44`, `models/cells/bipolar.py:70-80`; `training/stage1.py:164` |
| bipolar transient tau | Bipolar | yes | `raw_tau_transient` | `_bounded(raw, (tau_transient_min, tau_transient_max))` | core optimizer group | frozen by no-grad core pass | yes | `models/cells/bipolar.py:40-44`, `models/cells/bipolar.py:70-80`; `training/stage1.py:164` |
| bipolar sustained g_AB | Bipolar | yes | `raw_g_ab_sustained` | `g_ab_sustained_max * sigmoid(raw)` | core optimizer group | frozen by no-grad core pass | yes | `models/cells/bipolar.py:45-49`, `models/cells/bipolar.py:83-91`; `training/stage1.py:164` |
| bipolar transient g_AB | Bipolar | yes | `raw_g_ab_transient` | `g_ab_transient_max * sigmoid(raw)` | core optimizer group | frozen by no-grad core pass | yes | `models/cells/bipolar.py:50-54`, `models/cells/bipolar.py:83-91`; `training/stage1.py:164` |
| A2 sustained tau | A2 | yes | `raw_tau_sustained` | `_bounded(raw, (tau_sustained_min, tau_sustained_max))` | core optimizer group | frozen by no-grad core pass | yes | `models/cells/amacrine.py:117-126`, `models/cells/amacrine.py:153-163`; `training/stage1.py:164` |
| A2 transient tau | A2 | yes | `raw_tau_transient` | `_bounded(raw, (tau_transient_min, tau_transient_max))` | core optimizer group | frozen by no-grad core pass | yes | `models/cells/amacrine.py:122-126`, `models/cells/amacrine.py:153-163`; `training/stage1.py:164` |
| A2 sustained g_BA | A2 | yes | `raw_g_ba_sustained` | `g_ba_sustained_max * sigmoid(raw)` | core optimizer group | frozen by no-grad core pass | yes | `models/cells/amacrine.py:127-131`, `models/cells/amacrine.py:166-174`; `training/stage1.py:164` |
| A2 transient g_BA | A2 | yes | `raw_g_ba_transient` | `g_ba_transient_max * sigmoid(raw)` | core optimizer group | frozen by no-grad core pass | yes | `models/cells/amacrine.py:132-136`, `models/cells/amacrine.py:166-174`; `training/stage1.py:164` |
| RGC midget g_AG | RGC | yes | `raw_g_ag_midget` | `g_ag_midget_max * sigmoid(raw)` | core optimizer group | frozen by no-grad core pass | yes | `models/cells/rgc.py:108-134`; `training/stage1.py:164` |
| RGC parasol g_AG | RGC | yes | `raw_g_ag_parasol` | `g_ag_parasol_max * sigmoid(raw)` | core optimizer group | frozen by no-grad core pass | yes | `models/cells/rgc.py:112-134`; `training/stage1.py:164` |
| RGC residual g_AG | RGC | yes | `raw_g_ag_residual` | `g_ag_residual_max * sigmoid(raw)` | core optimizer group | frozen by no-grad core pass | yes | `models/cells/rgc.py:116-134`; `training/stage1.py:164` |
| decoder midget ON/OFF weights | Decoder | no | `_LocalProjection.raw_weight` in `current_midget` | `current_weight_max * tanh(raw_weight)` | decoder optimizer group | yes, decoder receives gradients from frozen RGC output | yes, decoder remains trainable | `models/decoder/local_decoder.py`; `training/stage1.py` |
| decoder parasol ON/OFF weights | Decoder | no | `_LocalProjection.raw_weight` in `current_parasol` | `current_weight_max * tanh(raw_weight)` | decoder optimizer group | yes | yes | `models/decoder/local_decoder.py`; `training/stage1.py` |
| decoder radial mixtures | Decoder | no | three-value `raw_basis_mix` in each population projection | softmax convex mixture over fixed row-normalized bases | decoder optimizer group | yes | yes; shared across target positions | `models/decoder/local_decoder.py`; `tests/test_local_decoder.py` |

Optimizer note: Stage-1 uses separate core and decoder optimizer groups. Their learning rates are engineering hyperparameters, not biological evidence.

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
| RGC threshold | profile `0.20` | current/state units | no | positive | hard threshold | spike emission | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_runtime.py:50` |
| RGC surrogate slope | profile `5.0` | inverse current/state units | no | positive | sigmoid surrogate | gradient through spikes | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_runtime.py:51-54` |
| RGC adaptation strength | profile `0.10` | state scale | no | positive | adaptation update | spike-triggered adaptation | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_runtime.py:56-61` |
| residual drive scale | profile `0.25` | unitless | no | `(0,1]` | residual current multiplier | suppress residual pathway | engineering-prior | `configs/physiology_profiles.py:112-137`; `models/cells/rgc_types.py:43` |
| loss defaults | current reconstruction `1.0`, linear spike energy `0.10`, homeostasis `1e-3`, rate band `0.01` to `0.20` | loss/spike-rate units | no | non-negative weights; rate band `[0,1]` | `RetinaLossConfig` | task plus operational energy constraint | engineering-prior | `loss/retina.py` |
| gradient clip default | `1.0` | norm | no | `> 0` when set | `HybridTrainingConfig` | optimizer stability | engineering-prior | `training/hybrid.py:48-55`, `training/hybrid.py:134-138` |
| BPTT default | `8` | frames | no | `>= 1` | `HybridTrainingConfig` | train window length | engineering-prior | `training/hybrid.py:48`, `training/hybrid.py:97-102` |
| test optimizer lr | `0.05` | learning rate | no | optimizer-defined | SGD fixture only | hybrid training smoke update | test-only | `tests/test_hybrid_training.py:47-50` |
| decoder ON/OFF raw weight init | `[0,0]` | decoder weight | yes | effective value bounded by `current_weight_max * tanh` | `torch.zeros(2)` | neutral polarity initialization | engineering-prior | `models/decoder/local_decoder.py` |
| decoder radial-basis mix init | `[0.05,0.90,0.05]` before logit conversion | mixture fraction | yes | three-value softmax | one mixture per population, shared across positions | starts near the profile center scale without freezing it | engineering-prior | `models/decoder/local_decoder.py` |
| test decoder fill | `0.05` | decoder weight | yes | residual tanh bound if residual | test setup only | forces nonzero decoder path | test-only | `tests/test_retina_snn.py:213-215`; `tests/test_hybrid_training.py:129-131` |
| transient test threshold | `0.05` | ratio | no | assertion only | unit test | high-pass decay check | test-only | `tests/test_bipolar_cell.py:148-150` |
| A2/RGC decay thresholds | `0.01` | ratio | no | assertion only | unit tests | initialization-forgetting/recovery checks | engineering threshold | `tests/test_amacrine_cell.py:109`; `tests/test_rgc_cell.py:171` |

## 7. Evidence Gap List

P0:

- `dt_ms`: current Stage-1 CLI derives `dt_ms` from ISETBio `time_axis_seconds`; standalone tests may still pass fixture values.
  - Why evidence is needed: all temporal tau and horizon interpretation depend on the true frame interval.
  - Recommended evidence source type: ISETBio.
- Bipolar sustained/transient taus: current profile values are sustained `80 ms` and transient `20 ms`.
  - Why evidence is needed: these define the sustained/transient biological claim and high-pass behavior.
  - Recommended evidence source type: human / macaque / marmoset.
- RGC Adaptive LIF taus and threshold: current profile values are membrane `20 ms`, adaptation `80 ms`, rate `50 ms`, threshold `0.20`.
  - Why evidence is needed: directly controls spike timing, rate smoothing, and parasol timing credibility.
  - Recommended evidence source type: macaque / marmoset / internal smoke statistics.
- Spatial RF radii and sigmas for H1, A2, parasol, residual, and decoder local masks.
  - Why evidence is needed: these determine population specificity and whether the model is biologically plausible in degrees.
  - Recommended evidence source type: human / macaque / marmoset / ISETBio.
- Missing real smoke gate thresholds.
  - Why evidence is needed: the repo has unit-test assertions but no production smoke gate for first-version training acceptance.
  - Recommended evidence source type: internal smoke statistics.

P1:

- H1 gain and tau: profile value gain `0.01`, max `0.2`, tau `50 ms`.
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
- human/macaque midget/parasol RGC temporal RF -> relative order only: midget sustained>transient and parasol transient>sustained; do not convert to exact tau or 0.75 physiology. citation needed.
- macaque parasol spike timing/response correlation -> RGC `rate_tau_ms`, parasol transient scale, and readout timing interpretation. citation needed.
- marmoset RGC white-noise STA -> RF lag window and midget/parasol spatial/temporal comparison. citation needed.
- human/macaque RF size in degrees/eccentricity -> H1, A2, parasol, residual, decoder local radii/sigmas and mosaic spacing. citation needed.
- internal smoke statistics -> clipping range, loss weights, residual penalties, grad clip, BPTT, and acceptance thresholds. citation needed.

## 9. Red Flags

- Stage-1 CLI derives `dt_ms` from `time_axis_seconds`; remaining risk is preserving that provenance when components are constructed outside the CLI.
- `configs/`, `evaluation/`, Stage-1 factories, and CLI defaults exist; remaining risk is that most profile values still lack literature evidence.
- Transient baseline subtraction uses the sustained bipolar tau as the baseline tau; there is no separate baseline timescale.
- H1 gain/tau, bipolar/A2/RGC gains, and biological tau bounds are bounded and trainable where implemented, but their current numeric ranges lack literature evidence.
- H1 node density is only constrained to be fewer than cones; no evidence-backed node ratio target exists.
- A2 self weight depends on local Gaussian radius/sigma and can become high when the support is narrow.
- RGC `rate_tau_ms=50 ms` may smooth parasol timing too aggressively unless supported by spike timing evidence.
- `residual_drive_scale=0.25` plus residual penalties are engineering priors; residual pathway credibility needs smoke statistics.
- Decoder local masks allow empty rows; this prevents crashes but can silently create target positions with no local support.
- Core/decoder optimizer groups exist, but their learning rates remain engineering hyperparameters.
- Evaluation modules exist, but smoke gate thresholds are not implemented as a versioned acceptance-threshold config.

## 10. Next Actions

- [ ] 补 human cone temporal evidence
- [ ] 补 human/macaque RGC temporal evidence
- [ ] 补 marmoset dataset RF/temporal filter evidence
- [ ] Preserve `dt_ms` provenance from `time_axis_seconds` in non-CLI construction paths
- [ ] 确认所有 `radius_degs` 单位一致
- [ ] 确认 Stage 1/2 optimizer group 中参数归属正确
- [ ] 补 bipolar sustained/transient tau 与 transient baseline tau 证据
- [ ] 补 H1/A2/parasol/residual/decoder spatial scale 证据
- [ ] 将真实 smoke gate 阈值从 unit-test assertion 独立成可审计配置
