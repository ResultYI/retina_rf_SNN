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
models/cells/       H1, bipolar, local amacrine, and RGC circuit modules
models/retina_snn.py  Causal H1 -> bipolar -> local amacrine -> RGC core
models/decoder/     Fixed local RGC pooling with horizon/ON-OFF readout gains
loss/               Prediction, rate, band-homeostasis, and residual criteria
training/           Deterministic mosaic construction and truncated-BPTT training
evaluation/         Baselines, dynamics, STA/Jacobian/GLM RF, and feasibility readouts
tests/              Data, cell, training, and evaluation smoke tests
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
in the amacrine/RGC stages. `LocalAmacrineLayer` supplies a physiologically
motivated local recurrent inhibitory
state. `RGCPopulationLayer` exposes midget-like, parasol-like, and constrained
residual spikes/rates while keeping membrane and adaptation state internal.
All internal `tau` values are filtering parameters, not asserted biological
transmission delays. Response latency, time-to-peak, crossover, recovery, and
transience are measured at the RGC output. No explicit delay is enabled in the
frozen architecture; an integer delay is considered only if bounded filtering
cannot resolve a reproducible output-latency mismatch.

The Stage-1 decoder is deliberately not a second learned RF bank. Each
population is pooled through a fixed normalized local Gaussian mask, followed
only by a learned ON/OFF coefficient for each prediction horizon. Residual
coefficients remain bounded. This makes post-training RF readout attributable
to the retina core rather than a high-capacity decoder.

`build_stage1_components` uses a cone-aligned midget private line only in the
explicit foveal mode. The convergent mode uses a lower-density midget mosaic
with local normalized pooling. Parasol positions are selected from spatial
cells rather than cone-array order, and the residual mosaic is derived from
that lower-density population. Every train and validation export must share
cone positions, cone ordering, and `dt_ms`.

## Stage-1 Entry Point

After the Stage -1 ISETBio HDF5 gate has passed, run decoder warm-up first:

```powershell
python scripts/train_stage1.py `
  --train-h5 data/train_a.h5 data/train_b.h5 `
  --val-h5 data/val_a.h5 `
  --output-dir runs/stage1_warmup `
  --stage decoder_warmup --device cuda --formal-evidence
```

Then fine-tune the core with the same data contract:

```powershell
python scripts/train_stage1.py `
  --train-h5 data/train_a.h5 data/train_b.h5 `
  --val-h5 data/val_a.h5 `
  --output-dir runs/stage1_finetune `
  --stage core_finetune --t-bptt 8 --device cuda `
  --resume runs/stage1_warmup/checkpoint.pt
```

Each run writes per-batch train rows plus full-dataset `train_eval` and
validation epoch summaries, normalization statistics, per-export clipping
summaries, train-fit zero/global/local-AR baselines, bounded-parameter audit,
and `checkpoint.pt`.
`best_checkpoint.pt` is updated only when the validation aggregate loss
improves, or the training aggregate loss when no validation export is given.
A decoder warm-up checkpoint can initialize core fine-tuning; otherwise resume
requires a matching training stage, for example
`--resume runs/stage1_finetune/checkpoint.pt`.
RF losses are not part of this entry point; use the post-training probes only
after the prediction and population-usage gates pass.

## Smoke Sequence Input

The image-motion builder below is smoke-test infrastructure. It creates
parametric sequences from still images and labels them
`parametric_image_sequence`; it is not valid evidence for natural-video
prediction. Build source-disjoint short smoke sequences with:

```powershell
python scripts/build_natural_motion_sequences.py `
  --input-dir data/raw_natural_images `
  --output-dir data/natural_sequences `
  --train-count 8 --val-count 2 --test-count 2 `
  --frame-count 32 --image-size 256 --seed 7
```

Each direct child of `data/natural_sequences/train` is one ordered input
sequence. Use a Stage -1 YAML with the following settings for that split:

```yaml
input_path: ../../data/natural_sequences/train
treat_input_directory_as_sequence: false
treat_child_directories_as_sequences: true
eye_movement_enabled: false
mosaic_seed: 17
```

Run the same process separately for validation and test, retaining the same
`mosaic_seed` so every exported HDF5 uses an identical cone mosaic. This
wrapper invokes MATLAB/ISETBio once per child sequence directory; Python does
not synthesize cone responses. Temporal motion comes from frame content in
this v0 path, so eye movement is disabled to keep the source of motion
identifiable.

For the formal evidence run, use frames extracted from genuine source videos,
keep each source movie in one split only, and set:

```yaml
stimulus_source_kind: natural_video
source_movie_id: stable_original_movie_id
```

Use the same `source_movie_id` for every clip derived from one original movie;
the natural-video exporter rejects a missing explicit movie ID. Training with
`--formal-evidence` rejects non-video exports, missing source IDs, and overlap
between training and validation source movies.

`train_stage1.py` writes `data_summary.json` before enforcing its default
`--max-clip-fraction 0.01` gate. A higher value is only suitable for a
deliberate diagnostic run after inspecting that summary; it does not establish
a valid training dataset.

## Minimal Checks

```powershell
python -m pytest tests/test_isetbio_data_contract.py
```

`tests/test_cone_response_io.py` additionally checks real HDF5 read/write behavior and requires a working `h5py`/`numpy` binary environment.
