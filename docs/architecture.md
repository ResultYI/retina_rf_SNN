# Canonical architecture

The shared front end retains the H1, bipolar, and amacrine state equations.
`TypedRGCPopulation` then creates exactly one output unit per recorded cell.

RGC states and outputs use `[batch,cell]` and `[batch,time,cell]`:

```text
membrane
adaptation
filtered rate
subunit energy
spike logit
spike probability
deterministic hard event
```

Each cell selects its known ON or OFF upstream channel. Spatial weights are a
local Gaussian pooling over the recorded cell center. Parameters are a
type-level base plus a bounded cell residual. Type ranges may overlap and the
data can move individual cells away from the type mean.

Training predicts the current spike logit from the previous state, computes the
point-process likelihood, then uses the observed current event to update the
next membrane reset and adaptation state. Free-running evaluation uses the
model's deterministic event instead.

RFs are never supervised. Static and dynamic effective RFs are derivatives of a
cell's spike logit with respect to lagged cone input. Dynamic RF compares
identical final probes reached through different preceding contexts.
