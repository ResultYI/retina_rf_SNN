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
attribute response_target_kind  "bernoulli" or "poisson"
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
additional detach boundaries. Bernoulli targets must be binary. Poisson targets
must be non-negative integer counts.

## Selection and evidence

`checkpoint_best_nll.pt` is selected only by held-out response NLL. The
canonical checkpoint schema is `retina_rgc_response_snn` revision 1 and rejects
legacy cone-reconstruction checkpoints.

Response prediction is reported on held-out stimuli/trials and compared with a
static point-process GLM. Static RF uses spike-logit Jacobians plus finite
differences. Dynamic RF uses same-source low/high contexts with an identical
final probe.

Synthetic static/adaptive teachers are method validation. A real-retina claim
requires an aligned recording that passes this contract and a held-out test
evaluation.
