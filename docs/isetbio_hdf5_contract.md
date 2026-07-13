# Stage -1 ISETBio HDF5 Contract

Stage -1 produces real ISETBio cone-response movies from natural images or
image sequences. These files are the only acceptable input for real-data smoke
training. Synthetic HDF5 files may be used for unit tests, but they do not pass
the real-data gate.

## Required Datasets

| Dataset | Logical shape | dtype | Unit | Use |
|---|---:|---|---|---|
| `/cone_response_lms` | `[T, N_cone, 3]` or `[T, H, W, 3]` | `float32` | isomerizations per integration time | Raw L/M/S-preserved response for audit and alternative preprocessing |
| `/cone_response_achromatic` | `[T, N_cone]` or `[T, H, W]` | `float32` | isomerizations per integration time | Primary current V1 Dataset input |
| `/time_axis_seconds` | `[T]` | `float64` | seconds | Frame clock; `dt_ms = median(diff(time_axis_seconds)) * 1000` |
| `/cone_xy_deg` | `[N_cone, 2]` | `float32` | visual degrees | Local masks, mosaic diagnostics |
| `/cone_type` | `[N_cone]` | `uint8` | cone type id | LMS routing and diagnostics |
| `/eye_movement_xy_deg` | `[T, 2]` | `float32` | visual degrees | Reproducibility and motion diagnostics |
| `/config_json` | UTF-8 bytes | `uint8` | JSON text | Generation config snapshot |
| `/source_image_path` | UTF-8 bytes | `uint8` | path text | Reproducibility |
| `/source_image_id` | UTF-8 bytes | `uint8` | id text | Manifest join key |

For compatibility with the current Python `ISETBioDataset`, Stage -1 also
writes these aliases:

| Alias | Meaning |
|---|---|
| `/cone_response` | same logical values as `/cone_response_achromatic` |
| `/cone_positions_degs` | same as `/cone_xy_deg` |
| `/cone_types` | same as `/cone_type` |
| `/eye_trace_degs` | same as `/eye_movement_xy_deg` |
| `/format_version` | `retina-snn-cone-response-v1` |
| `/response_shape_time_cone` | logical `[T, N_cone]` shape for `/cone_response` |

MATLAB HDF5 storage may appear transposed when inspected from Python. The
contract above defines logical axes. Python readback must accept MATLAB storage
order only after confirming the time axis and cone count agree.

## Required Root Attributes

| Attribute | Unit / type | Meaning |
|---|---|---|
| `dt_ms` | ms | Median frame interval recorded from the generated time axis |
| `field_of_view_deg` | degrees | Effective exported cone-response field of view |
| `source_mosaic_field_of_view_deg` | degrees | ISETBio cMosaic field used before export cropping |
| `export_crop_fov_deg` | degrees | Centered crop applied to the computed ISETBio cone response |
| `eccentricity_deg` | `[x, y]` degrees | Mosaic eccentricity |
| `mosaic_type` | string | Expected `cMosaic` |
| `mosaic_seed` | integer | Mosaic-generation seed or recorded deterministic seed |
| `stimulus_seed` | integer | Stimulus / eye-movement seed |
| `is_achromatic_stimulus` | boolean-like integer | Whether RGB input was projected to achromatic before ISETBio |
| `achromatic_projection_method` | string | Current value: `type_routed_lms_sum` |
| `ISETBio_git_commit` | string | Source checkout commit, or `unknown` |
| `ISETCam_git_commit` | string | Source checkout commit, or `unknown` |
| `MATLAB_version` | string | MATLAB version used for generation |
| `generation_date` | string | MATLAB `datestr(now, 30)` timestamp |

When `export_crop_fov_deg` is smaller than
`source_mosaic_field_of_view_deg`, MATLAB first computes the full ISETBio
cMosaic response, then keeps cones satisfying `abs(x) <= fov / 2` and
`abs(y) <= fov / 2`. The selected response columns, positions, cone types,
and LMS channels are written together. Python does not create or crop a cone
mosaic for this path.

## Achromatic Projection

When `achromatic_stimulus_enabled=true`, the input RGB image is converted to a
luminance image with:

```text
Y = 0.2126 R + 0.7152 G + 0.0722 B
```

The same `Y` image is copied into all RGB display channels before ISETBio scene
construction. After cMosaic computation, each cone's scalar response is routed
into one LMS channel according to `cone_type`; `/cone_response_achromatic` is
the sum across that routed LMS axis. For an irregular cone mosaic, this gives
logical `[T, N_cone]` achromatic cone response while retaining `/cone_response_lms`
for audit.

## Target Derivation

Training must derive model tensors only from past and present frames:

```text
x_cone = C[t - input_steps + 1 : t]
y_h = C[t + h] - C[t]
```

`C` is the log-normalized achromatic cone response loaded from
`/cone_response_achromatic` through the compatibility alias `/cone_response`.
Fine and coarse targets are derived later in Python by row-stochastic local
pooling matrices.

The Dataset must never read future target frames into `x_cone`. `cone_xy_deg`,
`cone_type`, and `eye_movement_xy_deg` are metadata for masks, diagnostics, and
reproducibility; they are not model inputs.

For frame-content motion, a Stage -1 source directory contains one ordered
sequence of frames. The wrapper may treat every direct child directory of a
split root as one source sequence via
`treat_child_directories_as_sequences=true`. In this v0 path,
`eye_movement_enabled=false` keeps the temporal signal attributable to the
sequence rather than mixed with a second drift process. Every split must use
the same `mosaic_seed` when downstream per-cone normalization and local masks
assume a shared cone ordering.

## Stage -1 Gate

The wrapper must reject a generated file if:

- HDF5 cannot be opened by `h5py`.
- Any required dataset or attribute is missing.
- `time_axis_seconds` is not strictly increasing.
- `dt_ms` does not match `median(diff(time_axis_seconds)) * 1000`.
- Cone responses contain NaN or Inf.
- `cone_response_achromatic` is all zero.
- Response time dimension does not align with `time_axis_seconds`.
- Config metadata is absent.
- A repeated run with the same config and seed changes numeric response arrays.

Failure must be classified as one of:

```text
MATLAB environment
ISETBio path
scene construction
cMosaic
eye movement
HDF5 export
Python readback
Python Dataset readback
```
