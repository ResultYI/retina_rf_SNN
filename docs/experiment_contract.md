# Experiment contract

## Configuration

`configs/experiment.yaml` is the only active experiment configuration. `training/config.py` rejects unknown or missing keys, validates cross-section timing constraints, and produces the resolved checkpoint configuration from nested dataclasses.

## Data

Training statistics are fitted only on training exports. Validation reuses those per-cone log-response statistics. Formal train and validation exports must be source-disjoint and must share cone geometry, eccentricity, sampling interval, and configured sequence length. Augmentation preserves a deterministic clean target and adds configuration-driven gain context and synthetic noise to the input.

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

- Steps 0–1000 establish reconstruction and the reference energy while constraint weights ramp from zero.
- Steps 1000–2500 ramp the inequality budget from the reference energy to 90% of that reference.
- Steps 2500–6000 use the full constrained objective.

The energy penalty activates only above budget. Activity below budget receives no additional reward.

Validation runs on a fixed clip set at the configured interval and scores only steps 224–319. Source sampling and augmentation use separate checkpointed PyTorch generators. Model selection records the best reconstruction checkpoint and the best reconstruction checkpoint satisfying the energy gate; final evaluation prefers the latter.

## Evaluation

Reconstruction reports MSE relative to a train-only mean baseline. Only after reconstruction and energy gates pass, dynamic RF uses same-source low/high contexts, an identical final probe, continuous-readout Jacobians, target-event-aware finite differences, recovery delays, and an identical-reset kernel error check. Independent impulse, step, and flicker probes supply temporal features for RGC clustering. Candidate clusters require minimum size, silhouette, spatial-radius difference, and sustained-response difference gates.

## Compatibility

Only checkpoint schema `retina_rf_snn` revision 2 is accepted. Earlier checkpoints cannot be loaded because they do not contain the revised bounded parameters, surrogate energy output, dual RNG states, and validation-selection state.
