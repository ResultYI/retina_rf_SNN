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

## Evaluation

Reconstruction reports MSE relative to a train-only mean baseline. Dynamic RF uses same-source low/high contexts, an identical final probe, continuous-readout Jacobians, hard-event-aware finite differences, recovery delays, and a reset-state suppression control. RGC clusters are reported after training and are not training targets.

## Compatibility

Only checkpoint schema `retina_rf_snn` revision 1 is accepted. Earlier checkpoints cannot be loaded because their state axes, population assumptions, decoder parameters, objective state, and optimizer groups do not match the canonical architecture.

