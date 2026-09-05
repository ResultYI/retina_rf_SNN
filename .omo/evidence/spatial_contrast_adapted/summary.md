# SC-adapted 22-cell comparison

STATUS: **COMPLETED_CORRECTED** — only the two authorized 69#6 fits were rerun with official `a0=4`. The other 42 fit JSON/prediction artifacts are SHA256-identical to their pre-correction versions.

## Authorized 69#6 initialization correction

The official runner initializes `a=running_resp.max()` over the entire training response array, before temporal cropping. The initial execution used the maximum inside the fitting loss mask. These maxima agree for 21/22 cells; **69#6** has full-training maximum **4**, fitting-mask maximum **3**. Its two superseded fits initialized `a0=3`; the two replacement fits initialize `a0=4`.

Only 69#6 SC-adapted and matched w=0 were rerun, once each, using the already-corrected implementation. Data, split, masks, filters, fitting-only Z-score, other initial values, objective, optimizer, bounds and defaults were unchanged. No tests, tuning, other fitting or source-model changes were performed in this correction. Existing LN and Canonical results are unchanged. The current benchmark contains 44 retained fits; execution history contains 44 original fits plus 2 authorized replacements.

| 69#6 model | Train NLL before | Train NLL after | Validation NLL before | Validation NLL after |
|---|---:|---:|---:|---:|
| SC-adapted | 0.503126247425 | 0.503558776866 | 0.484774227875 | 0.485252522932 |
| Matched w=0 | 0.506499090529 | 0.506176576953 | 0.484304416509 | 0.483982962375 |

Both replacements returned optimizer success=true, status=0; SC-adapted used 22 iterations, matched w=0 used 30. These are solver termination statuses, not a small-gradient assertion.

Correction manifest, source hashes, before/after values and unchanged-fit hashes: `correction_69_6_a0_4/correction.json`. Superseded results and reports are preserved under `correction_69_6_a0_4/before/`. Original execution provenance remains in `preflight.json` and `executed_source/`; it is historical, not the corrected two-fit execution manifest.

## Implemented definition and provenance

For each final training-only selected/refitted center-surround LN checkpoint, recover center `G_c`, positive center amplitude `A_c`, and signed unit-L2 temporal filter `k_c[0:60]`. Ignore the surround component, LN bias, and spike-history coefficient. Checkpoint identity, state reconstruction, and full-LN prediction replay were verified for all 22 cells.

Let `s=A_c G_c` be the actual native float32 spatial component, `S=sum_x s(x)`, and `v=S k_c`. Supply `s,v` to the official SC function. Then:

- `h(t,x)=sum_l v[l] X(t-l,x)`.
- `Imean(t)=sum_x s(x) h(t,x)/S=sum_l,x A_c G_c(x) k_c[l] X(t-l,x)`.
- `LSC(t)=sqrt(sum_x s(x)(h(t,x)-Imean(t))^2/S)`.
- Fitting-mask-only mean and population standard deviation (ddof=0) produce `Z_I,Z_C`.
- Official output: `lambda=a log(1+exp(b+w1 Z_I+w2 Z_C))`, expected count **per bin**, with no extra dt multiplication.
- Bernoulli probability: `p=1-exp(-lambda)`.
- Matched center-only control fixes `w2=0` and fits only `a,b,w1`.
- No spike history; no smoothing; no surround mixture; no new filter learning or hyperparameter selection.
- Per sequence, use the same 59-bin causal zero padding, 150-bin reset and existing score bins 30..149. No validation samples enter fit statistics or fitting.

The inherited LN artifacts retain their training-only inner-development lambda/step selection and full-training refit. Each retained SC result uses a single full-training output fit with L-BFGS-B, official bounds `a>=0`, others unbounded, official defaults `options=None`, and analytic gradients. The fitting objective is Bernoulli NLL sum divided by fitting event count (official response-sum normalization); reported NLL divides by the number of valid bins. No new regularizer, candidate search, or validation selection was used; the two authorized initialization corrections were not selected by validation performance.

Official source: [SpatiotemporalSCModel, pinned commit 76b7334](https://github.com/gollischlab/SpatiotemporalSCModel/tree/76b733421cc16131c2229e66a7714d8892de39d7).
The executed CPU convolution, mean/contrast calculation, softplus and derivative function bodies exactly match the saved official sources. The wrapper narrows imports to the author's NumPy fallback; `MAX_FLOAT_SIZE=500` comes from the same commit's `project_variables.py`. MIT copyright/license is retained. Source snapshots/manifest: `../spatial_contrast_baseline/`.

Final LN filter source:
`output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/<cell>/ln-trained.pt`.

Final Canonical comparison source:
`output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/`.

This is **SC-adapted**, not faithful original SC. Per-cell counts: source LN **128** raw parameters; inherited center **64** raw coordinates; new fitted output **4** (SC) or **3** (control), inherited-plus-new accounting **68/67**. These are not identifiable DoF or a matched-capacity claim. Temporal normalization removes one raw scale coordinate; fitting-only Z-score also cancels inherited positive amplitude scaling. Four Z-score statistics are fixed fitting-data statistics, not optimized parameters. Canonical source records total **129**, requires-grad/optimizer-listed **33** per cell. No claim that SC has only four total parameters.

## Updated validation NLL

Equal weight per cell; same 22 cells, 37 recordings, input. This is the frozen macaque **16 training / 4 validation one-second segments per recording/trial**, native **150 Hz**, score bins **30..149**. Training/validation mask totals are **263040 / 65760**.

| Group | Cells | SC-adapted | Matched w=0 | Center-surround LN | Canonical V1 |
|---|---:|---:|---:|---:|---:|
| overall | 22 | 0.458313940340 | 0.461344872700 | 0.425997944041 | 0.438956145536 |
| MC_ON | 5 | 0.422290021842 | 0.426130383206 | 0.395894259214 | 0.428506755829 |
| MC_OFF | 4 | 0.503657426318 | 0.504626638031 | 0.430339150131 | 0.441669881344 |
| PC_ON | 9 | 0.428669068882 | 0.432810389738 | 0.418741527531 | 0.427831285530 |
| PC_OFF | 4 | 0.524701313263 | 0.526283805899 | 0.475613281131 | 0.474335081875 |

| Cell | Group | SC-adapted | Matched w=0 | Center-surround LN | Canonical V1 | Official initialization |
|---|---|---:|---:|---:|---:|---|
| 67#4 | PC_OFF | 0.528139314136 | 0.524592566043 | 0.479695677757 | 0.464896202087 | PASS |
| 67#6 | MC_OFF | 0.488901451752 | 0.491220155617 | 0.430726110935 | 0.442264437675 | PASS |
| 67#7 | MC_ON | 0.446571966286 | 0.450475922268 | 0.428921610117 | 0.456886798143 | PASS |
| 67#14 | PC_ON | 0.376259271923 | 0.392327020046 | 0.366128653288 | 0.372924745083 | PASS |
| 67#21 | PC_ON | 0.304101728669 | 0.309561828813 | 0.298146069050 | 0.309125483036 | PASS |
| 67#26 | PC_ON | 0.525191626746 | 0.526657090357 | 0.515616238117 | 0.512515842915 | PASS |
| 67#33 | MC_OFF | 0.466717875721 | 0.468196940630 | 0.373482376337 | 0.369917273521 | PASS |
| 67#34 | PC_ON | 0.396439078905 | 0.399480588840 | 0.377446562052 | 0.386381626129 | PASS |
| 68#3 | MC_OFF | 0.573757854869 | 0.575106493503 | 0.486940622330 | 0.519419848919 | PASS |
| 68#4 | PC_ON | 0.437613226712 | 0.439352975049 | 0.423185050488 | 0.427100300789 | PASS |
| 68#7 | PC_ON | 0.372152367112 | 0.372870113641 | 0.356706738472 | 0.385217726231 | PASS |
| 68#10 | MC_ON | 0.361510521586 | 0.367045594069 | 0.315079480410 | 0.421071052551 | PASS |
| 68#11 | PC_OFF | 0.438922587840 | 0.450215368338 | 0.387451291084 | 0.372821658850 | PASS |
| 69#3 | PC_OFF | 0.618071202765 | 0.617604721029 | 0.563168525696 | 0.553157746792 | PASS |
| 69#4 | MC_ON | 0.453406195194 | 0.458238811597 | 0.427978813648 | 0.433251053095 | PASS |
| 69#6 | MC_OFF | 0.485252522932 | 0.483982962375 | 0.430207490921 | 0.435077965260 | PASS: corrected a0=4 |
| 69#7 | MC_ON | 0.465085161530 | 0.469013194192 | 0.458073705435 | 0.463302940130 | PASS |
| 69#21 | PC_OFF | 0.513672148312 | 0.512722568186 | 0.472137629986 | 0.506464719772 | PASS |
| 70#1 | PC_ON | 0.529321234078 | 0.531980987343 | 0.525036752224 | 0.545601189137 | PASS |
| 70#7 | PC_ON | 0.482691464473 | 0.487284956265 | 0.476247549057 | 0.472401469946 | PASS |
| 70#15 | PC_ON | 0.434251621322 | 0.435777947287 | 0.430160135031 | 0.439213186502 | PASS |
| 70#34 | MC_ON | 0.384876264615 | 0.385878393905 | 0.349417686462 | 0.368021935225 | PASS |

Per-cell inherited lambda/refit steps, output parameter vectors, initialization, termination reason, iterations and gradients: `per_cell.csv`, `cells/<cell>/<model>.json`. No SC hyperparameters were selected beyond the frozen definition.

## 69#21 read-only numeric status

| Model | success / status | Final objective | Train NLL | Validation NLL | Endpoint and outputs |
|---|---|---:|---:|---:|---|
| SC-adapted | true / 0 | 2.283508999053 | 0.516961065063 | 0.513672148312 | finite |
| Matched w=0 | true / 0 | 2.296250855163 | 0.519845679711 | 0.512722568186 | finite |

Both saved optimizer messages are `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`. Final objective, gradient, parameters, training/validation expected counts and probabilities were evaluated read-only and are finite. Validation outputs exactly match the saved arrays; the maximum train-NLL replay error is 1.1102230246251565e-16. Both original fits are retained byte-for-byte. Evidence: `correction_69_6_a0_4/69_21_read_only.json`.

## Verification and numerical warnings (original-run checks retained)

- 22/22 exact frozen LN state reconstruction and saved production target/mask/order comparisons passed.
- Full-LN validation-logit replay maximum absolute error: **0**.
- Center-only activation reconstruction maximum absolute error: **7.105427357601002e-15** (double arithmetic), **2.5678937678463853e-6** (native float32 reference).
- 44/44 saved probability links, per-cell NLLs and all population/group means independently recomputed with maximum error **0**.
- Prior-turn initializer regression red before correction, green after correction; prior-turn relevant suite **8 passed**. No test was added or run during this two-fit correction.
- 44/44 L-BFGS-B calls returned success with finite endpoint parameters, gradient and predictions. **39** terminated on relative objective reduction, **5** on projected-gradient criterion. This is not evidence that all reached small gradients.
- During **69#21 matched w=0** line search, official softplus emitted `overflow encountered in exp` and its derivative emitted `invalid value encountered in divide`. Endpoint remained finite. No numerical rewrite, parameter-bound adjustment or restart was made.
- Largest endpoint absolute gradient: **0.3143941690494607** (67#34 matched w=0); 69#21 matched w=0: **0.2064298850685266**. All gradients are saved without threshold-based pass claims.
- Canonical V1, LN source artifacts, raw data, split, loss and fitting source files were unchanged during this correction. Original-run source fixes remain documented in the historical provenance; exact executed versions are preserved.

## Files

- `results.json`: corrected numerical results and resolved initialization-deviation metadata.
- `per_cell.csv`, `group_summary.csv`: per-cell and equal-cell aggregate NLLs.
- `preflight.json`: original-run contract, filter/data checks, git and source hashes; retained unchanged as historical evidence.
- `initialization_count_check.json`: full training maximum vs masked maximum for each cell.
- `output_verification.json`: independent saved-output arithmetic and executed-source identity.
- `filters/*.npz`: recovered center Gaussian/amplitude/temporal arrays and official input filters.
- `cells/*/*-validation.npz`: predicted counts/probabilities and unchanged target/mask/order.
- `executed_source/*.py.txt`: exact pre-correction source versions; not executable new runs.
- `runtime.md`: workspace-only SciPy compatibility preparation.
- `rerun_69_6.py`: executed targeted correction; refuses to rerun into the existing correction directory.
- `correction_69_6_a0_4/correction.json`: two-fit execution, current source/hash identity, before/after results, unchanged other 42 fits.
- `correction_69_6_a0_4/69_21_read_only.json`: saved optimizer status and read-only endpoint computation.
- `correction_69_6_a0_4/replacement_verification.json`: corrected saved-output arithmetic, aggregate/CSV equality, initialization consistency and preservation of the other 42 fits; no training or tests.
- `correction_69_6_a0_4/before/`: recoverable pre-correction copies of the two fits, predictions and aggregate reports.
