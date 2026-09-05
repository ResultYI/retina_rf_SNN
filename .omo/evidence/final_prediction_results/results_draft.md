# Results draft

## Main text

We compared frozen prediction fits for 22 macaque RGCs from the Schottdorf–Lee dataset using the same validation cells, stimulus representation, temporal split, valid-bin mask, and Bernoulli target. NLL was averaged over valid bins within each cell and then equally across cells. Mean validation NLL was 0.509817 nats/bin for the constant-rate baseline, 0.425998 for the center-surround LN model, 0.422598 for the compact causal CNN, and 0.438956 for Canonical V1 (Figure 1A; Table 1). No model was retrained or reselected for this comparison.

The mean paired CNN−LN difference was −0.003400 nats/bin (cell-paired bootstrap 95% CI: −0.012847 to 0.005479), with a median difference of −0.005354 and lower CNN NLL in 13/22 cells. The corresponding CNN−Canonical V1 difference was −0.016358 (95% CI: −0.031573 to −0.002146; median −0.016017), with lower CNN NLL in 16/22 cells. Canonical V1−LN was 0.012958 (95% CI: 0.004098 to 0.024396; median 0.007866), with lower Canonical V1 NLL in 6/22 cells (Figure 1B). Thus, the CNN−LN mean-difference interval included zero, whereas the other two intervals did not. These are unadjusted, cell-level bootstrap intervals for the frozen fits, not estimates of training-seed uncertainty or evidence of model equivalence.

Cell-type summaries were descriptive only: CNN had the lowest group-mean NLL among the four main models for MC ON and MC OFF cells; LN had the lowest for PC ON cells, and Canonical V1 for PC OFF cells. The groups contained 5, 4, 9, and 4 cells, respectively, and no group-level significance tests were performed. CNN and Canonical V1 contained 2,990 and 33 trainable, optimizer-listed scalar parameters per independently fitted cell, respectively. These counts include their trainable output parameters, exclude frozen parameters and buffers, and are not counts of independent functional degrees of freedom. The comparison therefore concerns prediction at different model capacities, not matched-capacity performance.

## Supplementary text

The frozen SC-adapted model had a mean validation NLL of 0.458314 nats/bin. Its predictions are reported separately in Supplementary Figure S1 and the supplementary tables. SC-adapted inherited the frozen LN center filter and fitted four output parameters without spike history; its 64 inherited center coordinates plus four fitted coordinates should not be described as a four-parameter model in total or as a faithful fit of the original SC model.

## Statistical methods note

For each requested comparison, the per-cell difference was defined as first-model NLL minus second-model NLL. Negative values favor the first model. We resampled the 22 complete cell pairs with replacement 100,000 times, drawing 22 cells per replicate and using the same draws for all three comparisons. The fixed random generator was NumPy PCG64 with seed 20260831. The 2.5th and 97.5th percentiles of the resampled mean and median differences, using linear quantile interpolation, define the reported intervals. The table reports both mean and median intervals; Figure 1B shows mean intervals only. Resampling was not stratified by cell type. Win counts use strict lower NLL with exact equality counted as a tie; all three comparisons had zero ties. No p-values, multiplicity-adjusted intervals, or group-level inferential tests were computed. Cells, rather than animals, recordings, bins, or repeated fits, were the requested resampling unit; the intervals do not account for any dependence between cells or for training-seed variability.

## Figure captions

**Figure 1. Frozen prediction comparison.** (A) Validation NLL for Constant, center-surround LN, compact causal CNN, and Canonical V1. Thin lines connect measurements from the same cell; symbol color and shape identify MC/PC × ON/OFF class. Black diamonds indicate equal-cell means. (B) Mean paired NLL differences and unadjusted 95% percentile intervals from 100,000 cell-paired bootstrap replicates. The dashed vertical line denotes zero difference; negative values favor the first named model. All values use the frozen original validation mask and Bernoulli target.

**Per-cell paired-difference plot.** Each row represents one cell, ordered by MC ON, MC OFF, PC ON, and PC OFF and then by its original manifest order. Columns show CNN−LN, CNN−Canonical V1, and Canonical V1−LN. Segments run from zero to the observed paired difference and are not uncertainty intervals. All columns use the same horizontal scale. Wins above each panel list first-model wins, second-model wins, and exact ties.

**Supplementary Figure S1. SC-adapted predictions.** Per-cell SC-adapted validation NLL against LN and Canonical V1, with identical axis limits and identity lines. Points above the identity line have higher SC-adapted NLL. Colors and markers encode the same cell classes as Figure 1. These panels are descriptive and do not add inferential tests.
