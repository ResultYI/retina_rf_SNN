# Parameter Evidence: Human and Macaque Retina V1

更新日期：2026-07-14。本文以 `docs/parameter_audit_current_v1.md` 为参数清单，并将 Reinhard & Münch (2021) HumRet 人类 RGC 数据确定为主要输出级生理评价锚点；不以该数据反推不可识别的内部参数，不新增机制，不运行正式训练。

## 1. Executive Summary

Because the model uses a human cone mosaic and is evaluated against HumRet, direct human retinal evidence is primary. Macaque evidence is secondary support for structure, sign, and relative timing where human evidence is unavailable. Marmoset evidence is reported separately and is never pooled with human recordings as equivalent ground truth.

结论很直接：

- **应使用现有结论固定或数据派生的参数**：`dt_ms`、cone/RGC positions、normalization stats、ON/OFF 符号结构、midget/parasol/residual 的相对空间层级、local mask 的单位和拓扑约束、target pooling 的 row-stochastic 约束。
- **应使用现有结论给初值和 bounds，但让训练学习的参数**：H1 tau/gain，bipolar sustained/transient tau，local recurrent amacrine tau，RGC adaptation/membrane-like dynamics，所有跨层抑制增益 `g_AB/g_BA/g_AG`，residual drive scale。RGC rate smoother 在冻结架构中保持 fixed，并作为输出校准量审计。
- **应主要通过学习得到的参数**：midget/parasol 的 bounded ON/OFF readout、跨位置共享的三值径向基混合与 population-level causal temporal decay。decoder 不再保留逐边 spatial logits，主架构也不含 residual readout。
- **不建议从文献硬拷贝的参数**：所有 gain、threshold、surrogate slope、loss weights、grad clip、BPTT、clip range、smoke gate thresholds。它们要么是归一化模型内部量，要么依赖训练目标与数据分布。
- **HumRet 直接约束的是输出而不是内部状态**：优先比较 flash、frequency/contrast chirp、24 条件 drifting-grating、spikes/s 与群体响应分布；不能把一个人类 latency 数值直接赋给 H1/BC/AC/RGC 的某个 `tau`。

## 2. Evidence Grades

- **A**: direct human retina / human cone / human RGC measurement, including HumRet observable response distributions.
- **B**: macaque retina measurement; secondary non-human primate support when direct human evidence is absent.
- **C**: marmoset retina measurement; useful external primate benchmark, not mixed into the main human/macaque table.
- **D**: modeling prior, indirect evidence, or engineering initialization.

## 3. Main Decision Table

| Parameter group | Recommendation | Use existing conclusion for | Learn? | Evidence grade | Rationale |
|---|---|---|---|---|---|
| `dt_ms` | **fixed, data-derived** | derive from ISETBio `time_axis_seconds` | no | A/D | The clock comes from the export. Literature should not override the actual sampling interval. |
| `mask_fraction` | **fixed task hyperparameter** | masked current-contrast reconstruction corruption rate | no, freeze before formal comparison | E | This is an engineering objective choice, not a biological parameter. |
| normalization mean/scale | **fixed, data-derived** | train-set statistics | no | D | Must be computed from training data to avoid leakage. |
| clipping range | **fixed/tuned by data stats** | clip fraction target | no, tune by smoke stats | D | No human/macaque physiology maps to normalized log-contrast clipping. |
| cone positions / cone types | **fixed, data-derived** | human cone mosaic/topography | no | A | Human cone topology is relevant because the model uses human ISETBio cone mosaic. |
| RGC/mosaic spatial layout | **fixed or generated from human priors** | eccentricity-dependent density and midget/parasol relative scale | not in V1 | A/B | Human morphology/topography supports midget/parasol spatial organization; do not let training move cell positions in V1. |
| local Gaussian masks | **fixed masks; evidence-bounded radii** | local support, row-stochastic constraints, midget smaller than parasol | mask no; weights yes | A/B/D | Literature supports relative spatial scale, but exact Gaussian radius in degrees is model abstraction. |
| H1 radius/sigma/spacing | **fixed engineering-biological prior** | surround is broader/slower than center | no in V1 | B/D | Direct human H1 numeric scale is weak; use as constrained prior, not free learned geometry. |
| H1 tau | **bounded learnable** | initial range and “slower surround” constraint | yes | B/D | Horizontal feedback timing is indirect and circuit/context dependent. |
| H1 gain | **bounded learnable** | sign and range only | yes | D | Gain is normalized-model specific and should not be hard-copied from physiology. |
| bipolar sustained/transient tau | **bounded learnable** | sustained > transient within each fitted model; bounds may overlap | yes | B/D | Primate pathway literature supports relative timing better than exact tau or artificially disjoint latent-parameter intervals. |
| transient baseline tau | **do not add for current V1; if added later, bounded learnable** | baseline slower than transient drive | yes if implemented | D | Current code ties baseline to sustained tau; no direct evidence justifies a separate fixed value. |
| `g_AB` bipolar inhibition | **bounded learnable** | inhibitory sign and plausible max | yes | D | Inhibitory strength depends on model normalization and training objective. |
| Local amacrine radius/sigma | **fixed local mask prior** | local amacrine pooling and self/neighbor diagnostic | no in V1 | B/D | Spatial support can be constrained; exact radius should be validated, not learned first. |
| Local amacrine sustained/transient tau | **bounded learnable** | relative filtering timescale and RGC output dynamics | yes | B/D | Direct primate cell-type numeric transfer to this model is weak. |
| `g_BA` local amacrine drive gain | **bounded learnable** | positive drive/inhibition pathway only | yes | D | Normalized gain is not a literature-measurable conductance. |
| RGC membrane tau | **bounded learnable or fixed engineering prior** | rough LIF timescale range | preferably yes | D | This is not a direct biological membrane constant after normalized upstream drive. |
| RGC rate tau | **fixed in frozen V1; reconsider only after a specific output-bandwidth failure** | HumRet firing-rate waveform is the calibration target | no in V1 | A/B/D | Human data constrain observable response bandwidth; the internal smoother remains an engineering/latent quantity and is not a measured human membrane constant. |
| RGC adaptation tau | **bounded learnable** | HumRet flash/chirp recovery and transiency are calibration targets | yes | A/B/D | Human output dynamics constrain the combined circuit, not a uniquely identifiable internal adaptation constant. |
| RGC threshold | **fixed calibration parameter initially** | positive threshold only | no in V1 | D | Threshold, gains, and input scale are not identifiable if all are learned at once. |
| surrogate slope | **fixed engineering parameter** | training stability | no | D | This is an optimization surrogate, not physiology. |
| `g_AG` RGC inhibition | **bounded learnable** | inhibitory sign; parasol can tolerate larger transient inhibition | yes | D | Exact strength is not directly transferable from literature. |
| decoder local radius | **fixed support mask** | local readout constraint | no in V1 | D/E | Prevents leakage; exact radius is an engineering support bound informed by geometry. |
| decoder weights | **bounded/shared learnable** | fixed support; two population-shared ON/OFF weights, one three-basis radial mixture and one temporal decay per population | midget/parasol yes | D/E | Readout parameters are model inference/engineering, not physiology. |
| residual branch | **absent from the frozen main architecture** | no current parameter | no | E | Reintroduction requires a specific, reproducible failure that midget/parasol cannot explain; it is not the default response to poor optimization. |
| loss weights | **validation/smoke tuned** | objective priorities | not gradient-learned | D | Not biological parameters. |
| smoke gate thresholds | **fixed after internal smoke stats** | empirical pass/fail distribution | no | D | Must come from project data, not physiology. |

## 4. Parameters To Fix From Existing Evidence

### 4.1 Data-derived fixed values

- `dt_ms`: compute from `time_axis_seconds`; do not learn.
- `positions_degs`, `cone_types`, `eye_trace_degs`, response units: load from ISETBio export; do not learn.
- normalization `mean/scale`: fit on training data only; do not learn inside the model.
- target pools: use explicit row-stochastic sparse matrices; do not learn unless creating a separate learned target objective later.

### 4.2 Structure fixed by retina conclusions

- Keep ON/OFF split fixed.
- Keep sustained/transient split fixed.
- Keep sustained and transient channels available to both RGC populations; impose only the primate-supported relative order midget sustained>transient and parasol transient>sustained. The 0.75/0.25 start is a mathematical midpoint, not a measured physiological value.
- Keep midget spatial scale smaller than parasol; residual should not become the main readout.
- Keep spatial units in degrees from `positions_degs`; do not mix micrometers and degrees.

Human evidence is strongest for cone/RGC topology and midget/parasol morphology, not for every temporal constant. Dacey and Petersen, Dacey, Curcio, Allen, and Watson-type human topography sources are appropriate for fixed geometry priors.

## 5. Parameters To Initialize From Evidence But Learn

These should not be hard constants. Literature should give ordering, plausible bounds, and initial values:

| Code parameter | Current audit value | Recommended treatment | Constraint to preserve |
|---|---:|---|---|
| `H1.raw_tau` | test init `50 ms`, bounds `10-200 ms` | bounded learnable | H1/surround slower than feedforward cone input |
| `H1.raw_gain` | test init `0.01`, max `0.2` | bounded learnable | subtractive surround sign, avoid gain large enough to erase local contrast |
| `Bipolar.raw_tau_sustained` | profile init `80 ms`, bounds `20-200 ms` | bounded learnable | sustained tau > transient tau within the fitted model |
| `Bipolar.raw_tau_transient` | profile init `20 ms`, bounds `5-120 ms` | bounded learnable | transient tau < sustained tau; overlapping bounds are allowed |
| `Bipolar.raw_g_ab_*` | test init `0.01` | bounded learnable | non-negative local amacrine-to-bipolar inhibition |
| `LocalAmacrine.raw_tau_sustained` | profile init `100 ms`, bounds `20-250 ms` | bounded learnable | sustained filtering slower than transient filtering |
| `LocalAmacrine.raw_tau_transient` | profile init `40 ms`, bounds `15-180 ms` | bounded learnable | faster filtering component; not a transmission delay |
| `LocalAmacrine.raw_g_ba_*` | test init `0.03/0.05` | bounded learnable | positive bipolar drive into the local amacrine state |
| `RGC membrane_tau_ms` | profile init `20 ms`, bounds `5-80 ms` | bounded learnable in current code | stable spike dynamics; interpret as latent filtering |
| `RGC rate_tau_ms` | profile fixed `50 ms` | keep fixed for the frozen architecture; calibrate output bandwidth | should not erase human transient timing; this smoother is not membrane physiology |
| `RGC adaptation_tau_ms` | profile init `80 ms`, bounds `20-250 ms` | bounded learnable in current code | adaptation slower than membrane response |
| `RGC.raw_g_ag_*` | test init `0.01/0.03/0.01` | bounded learnable | non-negative local amacrine-to-RGC inhibition |
| `residual_drive_scale` | test fixed `0.25` | bounded learnable or validation-tuned | residual remains auxiliary |

## 6. Parameters To Learn Directly

- `LocalDecoder.current_midget.raw_weight`
- `LocalDecoder.current_parasol.raw_weight`
- each enabled projection's three-value `raw_basis_mix`, softmax-normalized across
  fixed radial bases shared by every target position
- the two population-level causal temporal-decay parameters
Decoder weights are task readout parameters. Existing literature should constrain locality and population interpretation, not assign numeric decoder weights.

## 7. Parameters Not To Learn In V1

- `dt_ms`: must be data-derived.
- `positions_degs`: cell geometry should remain human/ISETBio-derived.
- `cone_types`: categorical metadata.
- ON/OFF and sustained/transient channel indices.
- row-stochastic radial decoder bases and fixed local support indices; only the
  population-shared mixture over three bases may learn.
- `surrogate_slope`: optimizer surrogate.
- loss weights, `grad_clip_norm`, `t_bptt`, smoke thresholds: tune by validation/smoke statistics, not by model gradient.

## 8. Evidence Notes By Source Type

### Human evidence: primary for geometry and observable RGC function

- Human photoreceptor topography supports cone mosaic density/eccentricity priors and reinforces using `positions_degs` as a fixed data source.
- Human ganglion cell topography and midget/parasol morphology support the relative spatial ordering: midget smaller/denser, parasol larger/sparser.
- HumRet directly supports output-level comparison of flash polarity/transiency, frequency and contrast chirps, drifting-grating F1 tuning, preferred spatial/temporal conditions, and firing rates in spikes/s.
- HumRet units were not morphologically identified. Its functional templates/clusters are secondary analyses, not hard midget/parasol labels; primary comparison is therefore population-level and response-property-level.
- The HumRet recordings are mid/non-foveal ex-vivo samples. They do not justify applying a foveal one-cone private line to all eccentricities.

### Macaque evidence: secondary for missing circuit-level constraints

- Macaque cone transduction and RGC recordings remain useful for connection sign, relative timing, and phenomena HumRet does not resolve at the interneuron or identified-cell level.
- Macaque parasol/M-cell dynamics may support a faster transient ordering, but direct human output comparisons take priority and exact tau values remain latent model parameters.
- Macaque horizontal/bipolar/amacrine evidence is useful for sign and relative timing, but not enough to hard-code gains.

## 9. Direct V1 Recommendation

For the current codebase, the cleanest V1 policy is:

1. **Fix data and geometry**: `dt_ms`, `positions_degs`, target pools, local masks.
2. **Use literature for ordering and bounds**: transient filtering faster than sustained within a model; midget-like spatial support smaller than parasol-like; inhibitory signs non-negative. Bounds need not be disjoint when evidence only fixes order.
3. **Learn bounded time constants and gains**: H1, bipolar, local recurrent amacrine, RGC dynamics and inhibition gains; calibrate the combined output to HumRet rather than fitting internal tau directly.
4. **Learn decoder weights**: keep locality/support fixed, learn row-normalized spatial values and low-capacity polarity readout weights.
5. **Use smoke statistics for engineering thresholds**: clip, loss weights, residual penalties, BPTT, grad clipping, smoke gates; a 1% clip gate is an engineering threshold.

This gives the paper a defensible story: human anatomy constrains geometry, human functional data judge the output, macaque data fill specific structure/sign gaps, and training decides ambiguous normalized latent parameters.

## 10. Caution Notes

- Do not use cortical/V1 or behavioral reaction time as retinal delay.
- Do not directly copy marmoset spatial values into human degree coordinates.
- Do not mix micrometers and degrees without conversion.
- If evidence only supports relative timing, do not invent exact tau or treat the 0.75 kinetic midpoint as physiology.
- If a parameter is only indirectly supported, keep it as bounded learnable.
- Human evidence has priority over macaque; macaque is used when human data are sparse.
- Gains in normalized SNN layers are not directly equal to synaptic conductances or current amplitudes.
- Do not convert HumRet functional clusters into morphologically verified midget/parasol labels.
- Convert internal smoothed spike probability per bin to spikes/s at the evaluation boundary; do not compare raw per-bin values with human Hz.
- Formal HumRet stimuli must pass through the same ISETBio human optics/cone-response and train-only normalization path; direct contrast-template injection is diagnostic only.

## 11. P0 Evidence Gaps

- `dt_ms`
  - Current status: `dt_ms_from_time_axis_seconds` derives it from each ISETBio export and split validation requires agreement.
  - Remaining need: preserve the source time axis and reject irregular sampling; literature must not override it.
- bipolar tau bounds
  - Current status: profile init `80/20 ms`, overlapping bounds `20-200/5-120 ms`, with `tau_transient < tau_sustained` enforced per model.
  - Remaining need: calibrate the combined output to HumRet and audit boundary accumulation; do not make ranges disjoint without direct evidence.
- RGC temporal parameters
  - Current status: profile membrane/adaptation initial values `20/80 ms` are bounded learnable; rate smoother is fixed at `50 ms` in the frozen architecture.
  - Human observable target: HumRet flash/chirp/grating response waveforms and spikes/s distributions.
  - Remaining gap: infer defensible bounds for non-identifiable internal tau without equating them to response latency; use macaque evidence only where it resolves an otherwise missing ordering.
- spatial radius/sigma in degrees
  - Current status: profile derives radii/sigmas as fixed multiples of cone spacing.
  - Needed: human midget/parasol RF scale by eccentricity and conversion to the model's `positions_degs`.
  - Search keywords: `human midget parasol receptive field size`, `primate retina receptive field structure`.
- smoke gate thresholds
  - Current status: not implemented.
  - Needed: internal smoke distributions, not physiology.

## References

- [Reinhard and Münch 2021] Visual properties of human retinal ganglion cells. PLOS ONE. DOI: https://doi.org/10.1371/journal.pone.0246952. Data/code: https://github.com/katjaReinhard/HumRet and https://osf.io/zf9rd/
- [Dacey and Petersen 1992] Dendritic field size and morphology of midget and parasol ganglion cells of the human retina. PNAS. DOI: https://doi.org/10.1073/pnas.89.20.9666
- [Perry and Cowey 1985] Parasol and midget ganglion cells of the human retina. Journal of Comparative Neurology. DOI: https://doi.org/10.1002/cne.902330107
- [Dacey 1993] The mosaic of midget ganglion cells in the human retina. Journal of Neuroscience. DOI: https://doi.org/10.1523/jneurosci.13-12-05334.1993
- [Curcio et al. 1990] Human photoreceptor topography. Journal of Comparative Neurology. DOI: https://doi.org/10.1002/cne.902920402
- [Curcio and Allen 1990] Topography of ganglion cells in human retina. Journal of Comparative Neurology. DOI: https://doi.org/10.1002/cne.903000103
- [Watson 2014] A formula for human retinal ganglion cell receptive field density as a function of visual field location. Journal of Vision. DOI: https://doi.org/10.1167/14.7.15
- [Croner and Kaplan 1996] Receptive field structure in the primate retina. Vision Research. DOI: https://doi.org/10.1016/0042-6989(95)00167-0
- [Schnapf et al. 1990] Visual transduction in cones of the monkey Macaca fascicularis. Journal of Physiology. DOI: https://doi.org/10.1113/jphysiol.1990.sp018193
- [Benardete and Kaplan 1999] The dynamics of primate M retinal ganglion cells. Visual Neuroscience. DOI: https://doi.org/10.1017/s0952523899162151
- [Uzzell and Chichilnisky 2004] Precision of Spike Trains in Primate Retinal Ganglion Cells. Journal of Neurophysiology. DOI: https://doi.org/10.1152/jn.01171.2003
- [Chichilnisky and Kalmar 2002] Functional Asymmetries in ON and OFF Ganglion Cells of Primate Retina. Journal of Neuroscience. DOI: https://doi.org/10.1523/jneurosci.22-07-02737.2002
- [Verweij et al. 1996] Horizontal cells feed back to cones by shifting the cone calcium-current activation range. Vision Research. DOI: https://doi.org/10.1016/s0042-6989(96)00142-3
- [Boycott and Wassle 1991] Cone bipolar cells and cone synapses in the primate retina. Visual Neuroscience. DOI: https://doi.org/10.1017/s0952523800010932
- [Hopkins and Boycott 1997] The cone synapses of cone bipolar cells of primate retina. Journal of Neurocytology. DOI: https://doi.org/10.1023/a:1018504718282
- [Wandell et al. 2015] ISETBIO: Computational tools for modeling early human vision. Imaging and Applied Optics. DOI: https://doi.org/10.1364/isa.2015.it4a.4
