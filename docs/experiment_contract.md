# Experiment contract

## HDF5

```text
/format_version                 "retina-rgc-response-v1"
/cone_response                  [stimulus,time,cone]
/spike_counts                   [stimulus,trial,time,cell]
/valid_mask                     [stimulus,trial,time,cell]
/time_axis_seconds              [time]
/cone/position_degs             [cone,2]
/cell/id                        [cell]
/cell/type_id                   [cell]
/cell/polarity                  [cell]
/cell/position_degs             [cell,2]
/cell/eccentricity_deg          [cell]
/stimulus/source_id             [stimulus]
/stimulus/context_id            [stimulus]
attribute response_target_kind  "bernoulli"
```

Train, validation, and test source IDs must be disjoint. Geometry, cell order,
target kind, and sampling interval must agree. Cone normalization is fitted on
training stimuli only and reused unchanged.

## Time and likelihood

```text
0..63    no-gradient burn-in using observed response history
64       one state detach
64..319  one differentiable chain
64..319  response likelihood at every bin
```

The 256 differentiable bins use 32-bin activation-checkpoint blocks without
additional detach boundaries. Bernoulli targets must be binary. Poisson
free-running is not enabled in the canonical pipeline.

## Selection and evidence

`checkpoint_best_nll.pt` is selected only by held-out response NLL. The
canonical checkpoint schema is `retina_rgc_response_snn` revision 4 and rejects
legacy cone-reconstruction checkpoints.

Response prediction is reported on held-out stimuli/trials and compared with a
static point-process GLM. Static RF uses spike-logit Jacobians plus finite
differences. Cell-wise conditional spike-logit RF is primary and is reported
separately under `zero`, `matched_observed`, and `standard_train_rate` history
contracts. `zero` uses no observed spike history, `matched_observed` reuses one
held-out observed history for each matched low/high probe pair, and
`standard_train_rate` uses a deterministic training-rate-matched binary
schedule. Endogenous observed-history and free-running response statistics are
prediction diagnostics and do not populate conditional RF fields.

Dynamic RF uses same-source low/high contexts with an identical final probe.
Unstable dynamic RFs are model-internal explanations, not biological truth.
Type-prior predictive, RF-stability, and data-efficiency value are secondary
validation-only endpoints. Type/polarity signs, signed gains, and direction
agreement are exploratory outputs, not primary pass/fail gates.

`model.parameter_sharing_mode` must be one of `type_aware`, `type_blind`,
`cell_only`, or `shuffled_type`. The canonical default is `type_aware`.
`type_aware` shares type bases with cell residuals, `type_blind` pools the type
base with cell residuals, `cell_only` uses one bounded base per cell, and
`shuffled_type` uses a deterministic count-preserving type-label shuffle.

Synthetic static/adaptive teachers are method validation. A real-retina claim
requires an aligned recording that passes this contract and a held-out test
evaluation.

The canonical synthetic teacher has 16 cells, with
`cells_per_type_polarity=4` for each ON/OFF by midget/parasol group and matched
center-replicate positions across groups. Formal adaptive validation requires
at least three independent held-out context pairs. A one-pair synthetic example
is only an engineering smoke check, and a two-step CLI run cannot be reported as
scientific support.
The synthetic smoke config may disable RF finite-difference checks to keep that
engineering check bounded; canonical experiment configs leave the default
finite-difference checks enabled.

The type-prior comparator consumes completed validation runs only. It compares
`type_aware`, `type_blind`, `cell_only`, and `shuffled_type` after checking
dataset fingerprint, cell/cone identity, split, history contracts, source-pair
count, and training budget.
