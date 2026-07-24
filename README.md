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

The canonical objective is Bernoulli or Poisson response likelihood. During
conditional training, the response for time `t` is predicted before the observed
event at `t` updates reset and adaptation state for `t+1`. Evaluation reports
both conditional likelihood and deterministic free-running predictions.

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
{"schema": "retina_rgc_response_snn", "schema_revision": 1}
```

The final report contains response prediction, a static point-process GLM
baseline, spike-logit static RFs, and matched-context dynamic RFs.

## Synthetic method validation

```powershell
python scripts/generate_synthetic_response_benchmark.py `
  --train-glob "data/isetbio_bsds300_4deg/train/*.h5" `
  --validation-glob "data/isetbio_bsds300_4deg/val/*.h5" `
  --output-dir runs/synthetic_response_smoke `
  --teacher adaptive `
  --trials 2 `
  --test-count 1
```

The former cone-reconstruction, anonymous-population, bootstrap, and readout
diagnostic pipelines have been removed. Git history remains the source for
those superseded experiments.
