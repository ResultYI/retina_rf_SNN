# Experiment contract

## Configuration

`configs/experiment.yaml` is the only active experiment configuration. `training/config.py` rejects unknown or missing keys, validates cross-section timing constraints, and produces the resolved checkpoint configuration from nested dataclasses.

## Data

Training statistics are fitted only on training exports. Validation reuses those per-cone log-response statistics. Formal train and validation exports must be source-disjoint and must share cone geometry, eccentricity, sampling interval, and configured sequence length. Augmentation preserves a deterministic clean target and adds configuration-driven gain context and synthetic noise to the input. Reconstruction normalization is fitted from seeded augmented clean targets, so it matches the supervised target distribution without using noisy inputs.

## Credit assignment

The active timing contract is:

```text
0..63    no-gradient burn-in
64       one state detach
64..319  one differentiable state chain
224..319 reconstruction supervision
```

The 256 differentiable steps are recomputed in 32-step activation-checkpoint blocks. Block boundaries do not detach state. A full-sequence gradient-audit interface exists for manual comparison and is not part of routine training.

## Optimization schedule

All model and decoder parameters are jointly optimized from the first step.

- Steps 0–1000 establish reconstruction and the EMA reference energy while constraint weights ramp from zero.
- Steps 1000–2500 linearly move the current inequality budget from the frozen reference to the fixed target at 90% of that reference.
- Steps 2500–6000 use the fixed target budget and full constrained objective.

The dual update and loss use the current budget. The energy penalty activates only above that current budget, and activity below budget receives no additional reward.

Validation runs on a fixed clip set at the configured interval and scores only steps 224–319. Source sampling and augmentation use separate checkpointed PyTorch generators. A validation result is eligible for the feasible checkpoint only after step 2500 and only when hard validation energy divided by the fixed target budget passes the configured ratio. If no target exists, energy evidence is `not_identifiable`, never infinity.

## Evaluation

Reconstruction reports MSE relative to a train-only mean baseline. Before optimizer construction or resume restoration, the runner creates or validates `initial_reference.pt`. Only after reconstruction and energy gates pass, dynamic RF uses same-source low/high contexts, an identical final probe, one trained-model selection plan shared with the initialized model, continuous-readout Jacobians, local finite differences, cached recovery states, and reset reproducibility checks. Unit evidence is reduced within each source and then compared with a paired source bootstrap.

Positive and negative impulse, step, and 4 Hz square-wave probes use the unit's center cone over the canonical 320-step sequence with onset at step 224. RGC clustering uses only effective spatial radius, impulse time-to-peak, impulse width, step sustained index, and normalized flicker response. Excluded units retain assignment `-1`. Trained separation must exceed initialization-level separation before a learned functional pairing candidate is reported.

## Compatibility

Only checkpoint schema `retina_rf_snn` revision 3 is accepted. Revision 2 has no converter because its moving budget semantics cannot establish the frozen target-energy contract required for final selection and reporting.
