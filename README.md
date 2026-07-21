# Retina RF SNN

This repository is the experiment workspace for a physiologically constrained retina SNN trained by current cone-contrast reconstruction under an explicit spike-energy bottleneck.

The first implemented layer is the ISETBio data path:

```text
natural still image + ISETBio fixational eye movement
-> ISETBio optical image / cMosaic response export
-> native cone-response movie [T, Ncone]
-> log cone contrast normalization
-> deterministic current cone-contrast reconstruction target
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
models/decoder/     Fixed local support with normalized learnable spatial weights
loss/               Baseline-normalized reconstruction, continuous normalized spike cost, and weak homeostasis
training/           Deterministic mosaic construction and truncated-BPTT training
evaluation/         Baselines, dynamics, STA/Jacobian/GLM RF, and feasibility readouts
tests/              Data, cell, training, and evaluation smoke tests
```

Large data, exported HDF5 files, checkpoints, and results stay local and are ignored by Git.

## Current Data Contract

`ISETBioDataset` expects an HDF5 export with cone response in logical `[T, Ncone]` order. A sample returns:

- `x_cone`: `[T_in, Ncone]`
- `target_current`: `[Ncone]`
- `time_index`: scalar anchor frame index

The input contains the complete causal cone-response history, and the target
is the clipped, normalized log cone contrast at the anchor. The current export
sets ISETBio cone noise to `none`, so this is not a paired noisy-input/clean-
target denoising task. This is a
rate-distortion/efficient-coding task, not predictive coding: no future frame is
read, and no final frame is artificially hidden. Fit normalization statistics
on training exports with `fit_log_cone_stats`, then pass
the same `mean` and `scale` to every split. Dataset-side fitting is disabled by
default; `allow_fit_stats=True` is reserved for explicit training or smoke tests.
`dataset.clip_fraction` reports the fraction clipped before target construction.

RF losses remain outside this data layer.

## Current Cell Contract

`H1HorizontalNetwork` provides a normalized local surround state and bounded
subtractive modulation. `BipolarLayer` is a cone-aligned private-line bank with
ON/OFF and sustained/transient channels. Its transient drive subtracts a slow,
causal baseline. ON/OFF signs are fixed, while bounded gain, threshold, and
softness parameters define a smooth softplus transfer instead of a hard ReLU;
spatial pooling for parasol-like populations remains downstream
in the amacrine/RGC stages. `LocalAmacrineLayer` supplies a physiologically
motivated local recurrent inhibitory
state. `RGCPopulationLayer` exposes midget-like and parasol-like spikes/rates
while keeping membrane and adaptation state internal.
Its bipolar-to-RGC support is fixed by the local physiological mask, while
positive within-support weights are learned under exact row normalization.
Before spatial pooling, each population applies a bounded learnable local gain
control driven by a causal bipolar-energy state. Midget-like and parasol-like
populations use independent adaptive LIF parameter sets with the same broad
bounds; any learned difference is an output of the task, not a fixed precise
human time constant.
All internal `tau` values are filtering parameters, not asserted biological
transmission delays. Response latency, time-to-peak, crossover, recovery, and
transience are measured at the RGC output. No explicit delay is enabled in the
frozen architecture; an integer delay is considered only if bounded filtering
cannot resolve a reproducible output-latency mismatch.

The Stage-1 decoder is deliberately not a second unconstrained RF bank. Each
population keeps a fixed local support, learns non-negative row-normalized
weights only inside that support, and applies bounded signed ON/OFF coefficients.
It reads only the current RGC rates; it has neither temporal averaging nor a
stimulus-conditioned spatial kernel.
The limited readout keeps post-training RF attribution centered on the retina
core rather than a high-capacity decoder.

`build_stage1_components` uses a cone-aligned midget private line only in the
explicit foveal mode. The convergent mode uses a lower-density midget mosaic
with local normalized pooling. Parasol positions are selected from spatial
cells rather than cone-array order. Every train and validation export must share
cone positions, cone ordering, and `dt_ms`.

## Human-Centered RGC Evaluation Contract

No single public dataset supplies human foveal cell-type labels, dynamic RF maps,
and temporal responses. External evaluation is therefore split without treating
the sources as interchangeable:

- [Bucci et al. (2025)](https://www.nature.com/articles/s41593-025-02011-3)
  human foveal/foveolar flash recordings are the primary human temporal anchor;
- [Godat et al. (2022)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0278261)
  macaque foveolar spatial tuning is the primary central spatial anchor;
- [Reinhard and Münch (2021)](https://doi.org/10.1371/journal.pone.0246952)
  HumRet is a secondary human mid-peripheral functional-population reference.

`evaluation.humret` freezes the human-comparison interface:

- 24 drifting-grating conditions: spatial periods 100, 200, 500, 1000, 2000,
  and 4000 micrometers crossed with 1, 2, 4, and 8 Hz, with F1 response compared
  as a population tuning distribution;
- the published 8-second frequency chirp, 2 Hz contrast chirp with an 8-second
  contrast ramp, and grey-black-grey-white-grey 2-second flash phases;
- lazy loading of `h_normPeaks.mat` and `h_chirp.mat`, without copying HumRet
  into this repository;
- explicit conversion of the internal smoothed spike probability per time bin
  to spikes/s only at the evaluation boundary.

The `build_humret_*` tensors are centered achromatic contrast templates, not
ISETBio cone responses. A formal HumRet run must render the matching photometric
stimuli through the same human optics/cone front end and reuse the training
normalization contract; direct tensor injection is allowed only as a circuit
diagnostic and cannot count as the human comparison.

HumRet provides human mid-peripheral functional response distributions, not morphologically
identified midget/parasol labels. Model populations therefore remain
`midget-like` and `parasol-like`; agreement is assessed at population and
response-property level. STA, Jacobian, and local GLM remain internal RF
consistency tests because HumRet does not provide a matching white-noise RF map
for every recorded unit. No internal model `tau` is fitted directly to a HumRet
latency or described as a transmission delay.

## Stage-1 Entry Point

After the Stage -1 ISETBio HDF5 gate has passed, run the unified two-phase
trainer. The warm-up fits the bounded local decoder with a frozen core; the
second phase updates the retinal core plus only the decoder's normalized static
within-support spatial weights. Its bounded signed ON/OFF coefficients remain
frozen and its learning rate is capped at the core rate, so the readout cannot
become a temporal or stimulus-conditioned RF mechanism:

```powershell
python scripts/train_stage1.py `
  --train-h5 data/train_a.h5 data/train_b.h5 `
  --val-h5 data/val_a.h5 `
  --output-dir runs/stage1 `
  --decoder-warmup-epochs 2 --core-finetune-epochs 5 `
  --input-steps 231 --t-bptt 8 --device cuda --formal-evidence
```

With the human/macaque V1 profile at 5 ms sampling, `231` is the filtering-
context requirement derived from the configured upper tau bound and the 1%
initialization-residual criterion. The first 223 frames advance state under
`no_grad`; only the final 8 frames participate in BPTT. This number is an
experimental context bound, not a physiological transmission delay.

Each run writes per-batch train rows plus full-dataset `train_eval` and
validation epoch summaries, normalization statistics, per-export clipping
summaries, train-fit zero-contrast/global-mean/local-linear baselines, bounded-parameter audit,
and `checkpoint.pt`. The spike penalty is a shared per-example cost normalized
by target-cone count and population density; its budget is an engineering
resource scale, not an asserted human firing-rate measurement.
`best_checkpoint.pt` is updated only when the validation aggregate loss
improves, or the training aggregate loss when no validation export is given.
Resume uses the unified run checkpoint and its saved phase progress. Checkpoints
from older decoder schemas are intentionally incompatible and require a fresh run.
RF losses are not part of this entry point; use the post-training probes only
after the reconstruction and population-usage gates pass. Checkpoints from the
removed future-horizon decoder are intentionally incompatible and require a fresh run.

## Checkpoint Evaluation Entry Point

One read-only command now collects held-out current-reconstruction skill, population usage
and single-population ablations, generic temporal diagnostics, STA/Jacobian/local
GLM RF agreement, and bounded-parameter audit into one evidence bundle:

```powershell
python scripts/evaluate_checkpoint.py `
  --checkpoint runs/stage1_finetune/best_checkpoint.pt `
  --normalization-stats runs/stage1_finetune/normalization_stats.npz `
  --train-h5 data/train_a.h5 data/train_b.h5 `
  --eval-h5 data/test_a.h5 `
  --output-dir runs/stage1_finetune/test_evaluation `
  --input-steps 231 --device cuda --formal-evidence
```

The runner fits global-mean and local-linear baselines only on `--train-h5` and
evaluates the checkpoint only on `--eval-h5`; it never refits normalization or
updates model parameters. It writes `evaluation_summary.json` and
`rf_probes.npz`. The NPZ also stores same-unit low/high natural-contrast
Jacobians with an identical final probe frame and only the preceding context
changed. Their difference is a context-dependence diagnostic, not sufficient
evidence of a dynamic RF without repeatability, uncertainty, and adaptation-
recovery checks. The generic temporal probes are explicitly labeled direct
normalized-contrast circuit diagnostics, not ISETBio/HumRet evidence.
For the current aggregate-count local GLM, the runner now reports
`rf_probe_status="not_identifiable"` rather than RF-map agreement whenever the
number of held-out sequences does not exceed its free coefficients. This is a
validity guard, not a failed RF result.

Formal HumRet comparison is enabled only when both `--humret-root` and
`--humret-model-response` are supplied. The latter must be an externally
generated ISETBio-front-end response artifact containing the required grating
and chirp arrays; without it the JSON records `humret.status="not_run"`. The runner does not inject the contrast
templates directly or fabricate midget/parasol ground-truth labels.

## Natural-image Microdrift Input

Natural still images become temporal retinal input through ISETBio fixational
eye movements; Python does not manufacture motion or cone responses. Formal
evidence accepts `natural_image_microdrift` exports when train/validation image
IDs are disjoint and each export contains a non-static eye trace. The builder
below remains smoke-test infrastructure: it creates parametric image sequences,
not formal natural-input evidence.

```powershell
python scripts/build_natural_motion_sequences.py `
  --input-dir data/raw_natural_images `
  --output-dir data/natural_sequences `
  --train-count 8 --val-count 2 --test-count 2 `
  --frame-count 32 --image-size 256 --seed 7
```

For the experiment path, point the Stage -1 YAML directly at source-disjoint
natural-image directories and enable ISETBio eye movements:

```yaml
input_path: ../../data/bsds300_raw/BSDS300/images/train
stimulus_source_kind: natural_image_microdrift
treat_input_directory_as_sequence: false
treat_child_directories_as_sequences: false
eye_movement_enabled: true
mosaic_seed: 17
```

Run export separately for validation/test, retaining the same `mosaic_seed` so
all HDF5 files share a cone mosaic. Training with `--formal-evidence` rejects
unsupported source kinds, static/missing eye traces, missing source IDs and
train/validation source overlap. It also requires enough causal context for
initialization forgetting before the target frame. Natural-video exports remain
accepted for compatibility but are not required by this experiment.

`train_stage1.py` writes `data_summary.json` before enforcing its default
`--max-clip-fraction 0.01` engineering gate. A higher value is only suitable
for a deliberate diagnostic run after inspecting that summary; it does not
establish a valid training dataset.

## Minimal Checks

```powershell
python -m pytest tests/test_isetbio_data_contract.py
```

`tests/test_cone_response_io.py` additionally checks real HDF5 read/write behavior and requires a working `h5py`/`numpy` binary environment.
