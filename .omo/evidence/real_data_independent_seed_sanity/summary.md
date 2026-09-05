# Independent-seed mechanism sanity

STATUS: COMPLETED

4 primary fits preserved; exactly 8 new development-selection/full-train-refit runs. No retries, tuning, architecture, loss, split, bounds or intervention changes.
Selection was saved before training. Closest within-class primary validation NLL to median; lexical cell-ID tie break. New seeds: primary +100000 and +200000; minibatch seed is seed +1000003 as in the frozen protocol.
Adam LR=0.03, batch=4, max=1000, patience=200, min_delta=1e-7; no regularizer. Each run freshly refits for its own inner-dev best step; original validation never selects weights or stopping.
Canonical signatures are existing original-stimulus paired logit mean-on (300–400 ms). Mach dark/bright are the saved x=-4/+4 ramp-minus-matched-uniform pairs. SBC is bright-minus-dark surround; Hermann intersection-minus-corridor; White on-bright-minus-on-dark bar. No polarity inversion. Same saved input/history tensors and production reset contract.
AC-off sign reversal means normal and AC-off paired means have opposite nonzero signs; the existing 1e-9 zero tolerance is retained. No NLL-closeness threshold was invented.

## Validation and pathway-off effects

| Cell | Group | Fit | Seed | NLL | Delta vs primary | H1-off | direct-BC-off | AC-off | best / stop |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 69#4 | MC ON | primary | 20260842 | 0.433251053 | 0.000000000 | 0.013389996 | 1.482159678 | 1.698397112 | 133 / 333 |
| 69#4 | MC ON | fresh_1 | 20360842 | 0.438102543 | 0.004851490 | 0.012712632 | 1.478702090 | 1.648807966 | 124 / 324 |
| 69#4 | MC ON | fresh_2 | 20460842 | 0.435479611 | 0.002228558 | 0.015195349 | 1.465160159 | 1.720428606 | 154 / 354 |
| 67#6 | MC OFF | primary | 20260829 | 0.442264438 | 0.000000000 | 0.086765163 | 1.225330707 | 0.961093295 | 369 / 569 |
| 67#6 | MC OFF | fresh_1 | 20360829 | 0.445984870 | 0.003720433 | 0.051648307 | 1.347565924 | 1.187268594 | 208 / 408 |
| 67#6 | MC OFF | fresh_2 | 20460829 | 0.444663167 | 0.002398729 | 0.090562983 | 1.090913527 | 0.918219055 | 551 / 751 |
| 68#4 | PC ON | primary | 20260837 | 0.427100301 | 0.000000000 | 0.011559233 | 1.035173537 | 1.578568676 | 193 / 393 |
| 68#4 | PC ON | fresh_1 | 20360837 | 0.427809536 | 0.000709236 | 0.010005891 | 1.034474838 | 1.481513422 | 171 / 371 |
| 68#4 | PC ON | fresh_2 | 20460837 | 0.428338200 | 0.001237899 | 0.007941675 | 1.081563923 | 1.509284806 | 181 / 381 |
| 67#4 | PC OFF | primary | 20260828 | 0.464896202 | 0.000000000 | 0.006554498 | 1.662654424 | 0.808444749 | 140 / 340 |
| 67#4 | PC OFF | fresh_1 | 20360828 | 0.463528782 | -0.001367420 | 0.006668410 | 1.699213475 | 0.823509884 | 145 / 345 |
| 67#4 | PC OFF | fresh_2 | 20460828 | 0.461889267 | -0.003006935 | 0.005483431 | 1.564486913 | 0.640648917 | 198 / 398 |

## Paired logit: normal / AC-off

| Cell | Fit | Mach dark | Mach bright | SBC | Hermann | White | AC-off reverses normal |
|---|---|---:|---:|---:|---:|---:|---|
| 69#4 | primary | -0.003327386 / +0.021974571 | +0.003327393 / -0.021974564 | -0.206090018 / -0.004334958 | -0.102707475 / -0.001475449 | +0.103178613 / +0.002161535 | Mach dark, Mach bright |
| 69#4 | fresh_1 | -0.001634717 / +0.021542788 | +0.001634733 / -0.021542791 | -0.188617900 / -0.003891245 | -0.093948051 / -0.001324526 | +0.094418481 / +0.001940282 | Mach dark, Mach bright |
| 69#4 | fresh_2 | -0.003351967 / +0.021657061 | +0.003351990 / -0.021657057 | -0.203088298 / -0.004448112 | -0.101190448 / -0.001514105 | +0.101675607 / +0.002218072 | Mach dark, Mach bright |
| 67#6 | primary | -0.001093264 / -0.015360149 | +0.001093268 / +0.015360149 | +0.144634649 / +0.029978195 | +0.069621064 / +0.010212088 | -0.072756991 / -0.014950752 | none |
| 67#6 | fresh_1 | -0.000225838 / -0.017629739 | +0.000225838 / +0.017629735 | +0.158668354 / +0.018864457 | +0.077757433 / +0.006423911 | -0.079660207 / -0.009406821 | none |
| 67#6 | fresh_2 | -0.000002042 / -0.013424372 | +0.000002042 / +0.013424372 | +0.138067678 / +0.030414153 | +0.066360377 / +0.010365153 | -0.069483854 / -0.015168905 | none |
| 68#4 | primary | -0.020625822 / +0.006829238 | +0.020625822 / -0.006829238 | -0.149149746 / -0.002119064 | -0.074919015 / -0.000738573 | +0.074696429 / +0.001078987 | Mach dark, Mach bright |
| 68#4 | fresh_1 | -0.019415641 / +0.006885354 | +0.019415641 / -0.006885354 | -0.149737507 / -0.002139727 | -0.075192824 / -0.000745670 | +0.074992374 / +0.001089366 | Mach dark, Mach bright |
| 68#4 | fresh_2 | -0.019635526 / +0.007265059 | +0.019635534 / -0.007265059 | -0.156076342 / -0.001981703 | -0.078314692 / -0.000690460 | +0.078155614 / +0.001008844 | Mach dark, Mach bright |
| 67#4 | primary | -0.003268786 / -0.012213850 | +0.003268802 / +0.012213842 | +0.051991936 / +0.002000149 | +0.025725484 / +0.000695666 | -0.026029555 / -0.001018500 | none |
| 67#4 | fresh_1 | -0.003282722 / -0.012729307 | +0.003282730 / +0.012729311 | +0.056518730 / +0.001950518 | +0.027984150 / +0.000678213 | -0.028293133 / -0.000993077 | none |
| 67#4 | fresh_2 | -0.004980795 / -0.012073140 | +0.004980795 / +0.012073135 | +0.039554123 / +0.001374213 | +0.019565154 / +0.000478077 | -0.019797644 / -0.000699917 | none |

## Pairwise disagreements

Opposite-sign / different-reversal rows: 0 of 60 fit-pair × signature rows.

## Verification and provenance

All four primary validation logits/NLL and original illusion normal/AC-off logits replay exactly. All four primary same-seed raw states and raw training NLL replay exactly. Training/data/loss/selection source hashes match the final training manifest. Existing core-source byte drift is listed in provenance.json; N=1 uses fixed identity mixing and unchanged standard geometry. Historical source-byte identity is not claimed.
All checkpoint hashes, saved evaluation NLL, pathway effects and paired means independently recomputed. All outputs finite; clamps exact-zero; inference state unchanged; paired controls exact-zero. All input/source hashes unchanged during training.

Artifacts: selection.json; provenance.json; preflight.json; run-manifest.json; per_fit.csv; illusion_paired_logits.csv; pairwise_comparison.csv; mechanism_disagreement.json; verification.json; fits/<cell>/<fit>/ (new checkpoints, trajectories and saved logits).
