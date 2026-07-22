# Canonical architecture

## Signal path

The model accepts normalized ISETBio cone-response sequences shaped `[batch,time,cone]`. H1 surround feedback, ON/OFF sustained/transient bipolar dynamics, and local recurrent amacrine dynamics preserve causal state. A single heterogeneous RGC pool converts those channels into hard events, continuous event probabilities, filtered rates, and generator potentials.

For `Ncone` centers and `units_per_center = 2`, the RGC pool contains `2 × Ncone` anonymous units. No cell-type identity is present in model, loss, trainer, or configuration code.

## RGC spatial encoder

Each unit is assigned a cone center and learns one bounded spatial sigma. A fixed support mask and squared-distance matrix have shape `[unit,cone]`. The sequence call constructs one dense masked-softmax weight tensor and passes the same tensor through every RGC step, the tied decoder, and wiring loss.

There are no per-edge residual parameters. Heterogeneity comes from per-unit spatial sigma, sustained mixture, membrane tau, adaptation tau, adaptation gain, amacrine gain, threshold, subunit tau, and subunit gain. The readout-rate tau is a fixed shared buffer.

## State and outputs

RGC states are:

```text
membrane       [batch,polarity,unit]
adaptation     [batch,polarity,unit]
rate           [batch,polarity,unit]
subunit_energy [batch,polarity,kinetics,unit]
```

Sequence outputs are `[batch,time,polarity,unit]` for hard spikes, spike probability, filtered rate, and generator potential. Detached hard events drive reset and adaptation events. The surrogate spike carries reconstruction gradients into temporal and spatial parameters.

## Tied decoder and objective

The decoder applies positive per-unit gains with fixed ON/OFF signs, projects through the transpose of the encoder weights, and adds a per-cone bias. It does not learn another spatial or temporal kernel.

The sole objective is defined in `loss/retina.py`. It supervises clean current cone contrast over the final 96 steps and combines reconstruction with energy, wiring, variance-floor, phenotype-repulsion, and homeostasis terms.

## Post-training interpretation

RGC typing is evaluation-only. Per-unit spatial, temporal, adaptation, inhibition, rate, and activity features are standardized and clustered with a dependency-free two-cluster k-means. Candidate physiological names are emitted only if preregistered between-cluster relationships hold.

