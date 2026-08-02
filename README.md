# Retina RF SNN

This repository fits recorded retinal ganglion cell responses with a causal,
physiologically constrained spiking retinal network.

```text
cone response
-> H1 surround
-> ON/OFF sustained/transient bipolar pathways
-> recurrent amacrine state
-> known recorded RGC cells with soft type priors
-> spike likelihood
-> static and context-dependent effective RF
```

Each output unit corresponds to one recorded RGC. Cell polarity, type, position,
and eccentricity are observed metadata. Type information supplies overlapping
soft priors; it is not treated as an emergent discovery. Cell-specific residuals
remain learnable.

The canonical objective is Bernoulli response likelihood. During conditional
training, the response for time `t` is predicted before the observed event at
`t` updates reset and adaptation state for `t+1`. Cell-wise conditional
spike-logit RF is the primary RF endpoint. It is reported under zero history,
matched observed history, and deterministic standard-train-rate history;
endogenous observed-history response statistics are separate prediction
metrics. Type-prior predictive, RF-stability, and data-efficiency value are
secondary validation-only endpoints. Type/polarity RF signs and direction
agreement are exploratory only.
Deterministic free-running RF is reported only as an auxiliary diagnostic.
Poisson free-running is not enabled in the canonical pipeline.

## Data contract

Canonical inputs use `retina-rgc-response-v1` HDF5 files containing aligned cone
responses, repeated RGC spike/count targets, masks, source/context identifiers,
and cell metadata. See [docs/experiment_contract.md](docs/experiment_contract.md).

The repository does not yet contain aligned real RGC recordings. The synthetic
point-process benchmark validates the method and software path only.

## Run

```powershell
python scripts/run_experiment.py `
  --config configs/experiment.yaml `
  --device cuda `
  --output runs/rgc_response
```

The best checkpoint is selected by held-out response NLL and uses:

```json
{"schema": "retina_rgc_response_snn", "schema_revision": 4}
```

`model.parameter_sharing_mode` selects the RGC parameter grouping:
`type_aware` uses observed type bases plus cell residuals, `type_blind` uses one
pooled type base plus cell residuals, `cell_only` uses one bounded base per cell
without residuals, and `shuffled_type` applies a seeded count-preserving type
label shuffle for type-control runs.

The final report contains response prediction, a static point-process GLM
baseline, conditional spike-logit static RFs, and matched-context dynamic RFs.
Unstable dynamic RFs are treated as model-internal explanations, not biological
truth.

## Synthetic method validation

```powershell
python scripts/generate_synthetic_response_benchmark.py `
  --train-glob "data/isetbio_bsds300_4deg/train/*.h5" `
  --validation-glob "data/isetbio_bsds300_4deg/val/*.h5" `
  --output-dir runs/synthetic_response_smoke `
  --teacher adaptive `
  --cells-per-type-polarity 4 `
  --trials 2 `
  --test-count 3
```

The default replicated teacher has 16 cells: four position-matched cells for
each ON/OFF by midget/parasol group. Use `--test-count 1` only for an
engineering smoke check. A formal adaptive method benchmark needs at least
three independent held-out context pairs, and a two-step experiment run only
verifies the CLI/reporting contract; it is not scientific support.
`configs/synthetic_smoke.yaml` disables RF finite-difference checks only for
that bounded engineering smoke. Canonical experiments keep the default checks.

Compare `type_aware`, `type_blind`, `cell_only`, and `shuffled_type` runs with
`scripts/compare_type_prior_variants.py`. The comparator accepts validation
runs only and rejects mismatched datasets, cell/cone identity, source-pair
counts, history contracts, or training budgets.

The former cone-reconstruction, anonymous-population, bootstrap, and readout
diagnostic pipelines have been removed. Git history remains the source for
those superseded experiments.
