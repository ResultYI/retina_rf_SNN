# Phase R1 official reliability preflight

Status: **UNVERIFIED — incomplete official numerical definition**.

This document freezes the recovered definition and its gaps. It does not authorize an inferred replacement algorithm. No smoothing, reliability coefficient, model score or normalization was computed.

## Primary sources

1. Schottdorf & Lee (2021), DOI [10.1113/JP281200](https://doi.org/10.1113/JP281200), [author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC8998785/): Methods, Correlation (`P17`), model response processing (`P24`), Results, Reliability and correlation, Table 2. Local original HTML and BioC XML are preserved in `output/audits/schottdorf_official_rf_center_recovery_20260905/`; this audit read the original HTML, not only a prior report. Their original retrieval log records a local TLS-verification limitation; indexed current primary-source text agrees on the recovered method.
2. [Official data DOI](https://doi.gin.g-node.org/10.12751/g-node.xage77/), linking the [author repository](https://gin.g-node.org/Manuel/Macaque-ganglion-cells). Local checkout: `data/real/schottdorf_lee_2021_repository`, HEAD `cffefb08c760f04c9a951da46061b361d2288e9b`, origin matches the official URL, worktree clean at inspection. This is the inspected revision, not a claim that all historical or unpublished author code was searched.
3. Official `README.md`: lines 36–40 for raw timestamps/repeats; 94–96 for movie frames; 116–120 for firing-rate processing. `retinatools/library.py:5–12` defines a model low-pass primitive. `run_model/Piecharts.ipynb`, zero-based cells 2, 3, 5, 8, 11, uses supplied, rounded reliability constants rather than estimating them from six raw spike trains.
4. van Hateren et al. (2002), DOI [10.1523/JNEUROSCI.22-22-09945.2002](https://doi.org/10.1523/JNEUROSCI.22-22-09945.2002), [institutional manuscript](https://pure.rug.nl/ws/portalfiles/portal/3046327/2002JNeuroscivHateren.pdf), p. 9948. This confirms the cascade structure, not the missing 2021 trial aggregation implementation.

## Recovered versus unresolved details

| Requirement | Evidence and status |
|---|---|
| Acquisition resolution | 0.1 ms, confirmed by paper and README; distinct from analysis bin width |
| Raw trials | Six presentations of the first live movie minute; seventh column is subsequent maintained activity, not repeat 7 |
| Analysis bin size | 1 ms, confirmed; exact endpoint assignment of spikes falling on bin boundaries is not specified |
| Smoothing family/order | Cascade of eight first-order low-pass filters, confirmed |
| Time constant | MC 2 ms; PC/S 4 ms, confirmed |
| Discrete implementation | Not recovered for the raw-response pipeline; a model `SimpleLowpass` helper alone does not establish that it generated the published reliability responses |
| Initialization/edges | Raw-response initialization, use of pre-live spikes, padding and endpoint treatment not recovered |
| Temporal range | Paper describes correlations over 60 seconds; exact sampled endpoints and any filter-edge handling not recovered |
| Correlation | Correlation coefficients between smoothed repeat responses, confirmed at method level; exact executable reduction not recovered |
| Pair construction | Single-trial comparisons are stated; selected pair set, all-pair versus successive-pair aggregation, and averaging transform are not specified in inspected sources |
| Trial-mean/self inclusion | A six-repeat mean is used for author model fitting. No executable raw reliability routine establishes a trial-versus-mean estimator or its self-exclusion policy; do not substitute one |
| Normalization | Pie charts subtract squared model correlation from an externally supplied reliability quantity. This does not supply a finite-repeat-mean noise-ceiling estimator for R1 |

The model helper's recurrence is `a=dt/(RC+dt)`, `v[0]=a*x[0]`, `v[k]=v[k-1]+a*(x[k]-v[k-1])`. Its existence is a code fact; reusing it eight times for raw spikes would be an unverified implementation choice. No such substitution was made.

## Reproduction gate

`REPRODUCED` requires a sufficiently specified author method and compatible numerical reference. `PARTIAL` may describe an implemented method with explained numerical differences, such as a different available cohort. It cannot certify an algorithm whose necessary choices have not been recovered.

The missing raw-response filtering and trial aggregation choices leave the method **UNVERIFIED** under Phase R1 sections 3–4. Matching rounded paper values by trying alternatives would not establish correctness and was not attempted. Stop before the response-processing, normalization and model-score stages.

The inspected official Python/notebook code units and matching locations are in `official_code_search.json`; source hashes, the 66 frozen checkpoint identities and raw inventory are in `inventory_lock.json`.

## Population comparison boundary

Paper Table 2 covers a larger class-stratified sample; the public raw subset available for the current aligned cohort is not that full sample. Its reported class means span 0.58–0.81. Those are external published values, not the reliability of the current 20 available cells. No project-versus-paper numerical agreement is claimed.
