# Canonical architecture

## Canonical V1 causal contract

`X_H1 = H1(cone)` is unchanged. For view `v` in `{direct, broad}`:

```text
Phi[v,p,s,r] = shared_locality_mix(sign * spatial_basis[s, support_v]
                                 * causal_BC_temporal_basis[p,r] * X_H1)
B[v,p] = sum_s,r W_BC[group,p,s,r] * Phi[v,p,s,r]
A[p] = lowpass_tau_AC[p](fractional_delay_ms_AC[p](B[broad,p]))
I_RGC = g_BC * sum_p direct_mask[p] * B[direct,p]
        - g_AC * sum_p AC_mask[p] * softmax(AC_group_logits)[p] * A[p]
```

`W_BC` is normalized to sum one over pathway, spatial mode and temporal mode
within each group. Both views use the same weights, spatial scales, BC temporal
basis, tau and delay. Only the spatial support differs: the unchanged smaller
BC disk is contained in the unchanged broader full AC disk. Spatial basis
weights are normalized separately on each support. No independent AC
stimulus/spatiotemporal encoder remains. The existing two AC delay parameters
now act downstream of BC; their values and bounds are unchanged. AC state tau,
group mixture and cell gain remain. No new trainable parameter is introduced.

H1-off zeroes only H1 amplitude before recomputing all downstream signals.
AC-off zeroes only final AC currents; direct-BC-off zeroes only direct BC
currents. Neither changes the broad BC presynaptic signal. RF helpers use the
actual forward/autograd graph; the configured RF lag window is distinct from
state tau, explicit pathway delay, and the strictly-past RGC history shift.

Config and state dict carry `h1-shared-bc-direct-broad-ac`. Missing or different
causal identities are rejected before state loading, including `strict=False`.
The public name remains Canonical V1.

## Historical response-model description (not the current V1 contract)

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
