# Retina RF SNN

This repository is the clean experiment workspace for a physiologically constrained predictive retina SNN.

The first implemented layer is the ISETBio data path:

```text
video or frame sequence
-> ISETBio optical image / cMosaic response export
-> native cone-response movie [T, Ncone]
-> log cone contrast normalization
-> short-horizon future cone contrast change target
```

## Layout

```text
data/
  cone_response.py  HDF5 loading and response/time-axis validation
  dataset.py        Train-only normalization, clipping diagnostics, and sample slicing
  geometry.py       Local sparse masks for private-line and pooling connections
external/isetbio_pipeline/
  run_export.py     Python launcher for MATLAB/ISETBio export
  cone_response_io.py
  matlab/
models/cells/       H1 surround and cone-aligned bipolar dynamics
models/             Planned retina_snn.py and decoder.py
loss/               Planned training criteria
utils/              Planned baselines and diagnostics
tests/              Smoke tests for the data contract
```

Large data, exported HDF5 files, checkpoints, and results stay local and are ignored by Git.

## Current Data Contract

`ISETBioDataset` expects an HDF5 export with cone response in logical `[T, Ncone]` order. A sample returns:

- `x_cone`: `[T_in, Ncone]`
- `target_delta`: `[Nhorizon, Ncone]`
- `target_fine`: `[Nhorizon, Nfine]` when `target_fine_pool` is configured
- `target_coarse`: `[Nhorizon, Ncoarse]` when `target_coarse_pool` is configured
- `time_index`: scalar anchor frame index

Training targets are future changes in clipped, normalized log cone response. Fit
normalization statistics on training exports with `fit_log_cone_stats`, then pass
the same `mean` and `scale` to every split. Dataset-side fitting is disabled by
default; `allow_fit_stats=True` is reserved for explicit training or smoke tests.
`dataset.clip_fraction` reports the fraction clipped before target construction.

Fine and coarse pooling matrices must be sparse, non-negative, row-normalized
`[Ntarget, Ncone]` tensors. Source-to-target pooling therefore uses
`torch.sparse.mm(pool, source.T).T`. RF losses remain outside this data layer.

## Current Cell Contract

`H1HorizontalNetwork` provides a normalized local surround state and bounded
subtractive modulation. `BipolarLayer` is a cone-aligned private-line bank with
ON/OFF and sustained/transient channels. Its transient drive subtracts a slow,
causal baseline; spatial pooling for parasol-like populations remains downstream
in the A2/RGC stages. `A2AmacrineLayer` supplies the local recurrent inhibitory
state. `RGCPopulationLayer` exposes midget-like, parasol-like, and constrained
residual spikes/rates while keeping membrane and adaptation state internal.

## Minimal Checks

```powershell
python -m pytest tests/test_isetbio_data_contract.py
```

`tests/test_cone_response_io.py` additionally checks real HDF5 read/write behavior and requires a working `h5py`/`numpy` binary environment.
