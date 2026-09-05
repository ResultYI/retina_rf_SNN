# Original Mach replay: plateau excursions

Unchanged original metric: per-cell maximum above/minimum below the two remote plateau mean-on responses, in fixed boundary regions x=-6..-2 and +2..+6 pixels. Counts use 1e-9 as in the original report. Values remain response units.

| Group | Model/mode | Channel | Profile | Above cells | Below cells | Mean max above | Mean min below |
|---|---|---|---|---:|---:|---:|---:|
| all | R4_normal | logit | ramp | 22/22 | 22/22 | +0.007144058 | -0.007144085 |
| all | R4_normal | logit | matched_uniform | 0/22 | 0/22 | +0.000000000 | +0.000000000 |
| all | R4_normal | probability | ramp | 22/22 | 22/22 | +0.001224760 | -0.001093031 |
| all | R4_normal | probability | matched_uniform | 0/22 | 0/22 | +0.000000000 | +0.000000000 |
| all | R4_H1_off | logit | ramp | 21/22 | 21/22 | +0.006068373 | -0.006068441 |
| all | R4_H1_off | logit | matched_uniform | 0/22 | 0/22 | +0.000000000 | +0.000000000 |
| all | R4_H1_off | probability | ramp | 21/22 | 21/22 | +0.001053025 | -0.000927760 |
| all | R4_H1_off | probability | matched_uniform | 0/22 | 0/22 | +0.000000000 | +0.000000000 |
| all | R4_AC_off | logit | ramp | 22/22 | 22/22 | +0.001706188 | -0.001706210 |
| all | R4_AC_off | logit | matched_uniform | 0/22 | 0/22 | +0.000000000 | +0.000000000 |
| all | R4_AC_off | probability | ramp | 22/22 | 22/22 | +0.000317758 | -0.000223244 |
| all | R4_AC_off | probability | matched_uniform | 0/22 | 0/22 | +0.000000000 | +0.000000000 |
| all | LN | logit | ramp | 9/22 | 11/22 | +0.001893677 | -0.002023816 |
| all | LN | logit | matched_uniform | 0/22 | 0/22 | +0.000000000 | +0.000000000 |
| all | LN | probability | ramp | 9/22 | 12/22 | +0.000242708 | -0.000253872 |
| all | LN | probability | matched_uniform | 0/22 | 0/22 | +0.000000000 | +0.000000000 |
| MC_ON | R4_normal | logit | ramp | 5/5 | 5/5 | +0.010442853 | -0.010442853 |
| MC_ON | R4_normal | logit | matched_uniform | 0/5 | 0/5 | +0.000000000 | +0.000000000 |
| MC_ON | R4_normal | probability | ramp | 5/5 | 5/5 | +0.001682013 | -0.001590982 |
| MC_ON | R4_normal | probability | matched_uniform | 0/5 | 0/5 | +0.000000000 | +0.000000000 |
| MC_ON | R4_H1_off | logit | ramp | 5/5 | 5/5 | +0.009526896 | -0.009526944 |
| MC_ON | R4_H1_off | logit | matched_uniform | 0/5 | 0/5 | +0.000000000 | +0.000000000 |
| MC_ON | R4_H1_off | probability | ramp | 5/5 | 5/5 | +0.001550251 | -0.001429787 |
| MC_ON | R4_H1_off | probability | matched_uniform | 0/5 | 0/5 | +0.000000000 | +0.000000000 |
| MC_ON | R4_AC_off | logit | ramp | 5/5 | 5/5 | +0.002612281 | -0.002612209 |
| MC_ON | R4_AC_off | logit | matched_uniform | 0/5 | 0/5 | +0.000000000 | +0.000000000 |
| MC_ON | R4_AC_off | probability | ramp | 5/5 | 5/5 | +0.000480765 | -0.000340199 |
| MC_ON | R4_AC_off | probability | matched_uniform | 0/5 | 0/5 | +0.000000000 | +0.000000000 |
| MC_ON | LN | logit | ramp | 4/5 | 4/5 | +0.003004479 | -0.002689314 |
| MC_ON | LN | logit | matched_uniform | 0/5 | 0/5 | +0.000000000 | +0.000000000 |
| MC_ON | LN | probability | ramp | 4/5 | 5/5 | +0.000296095 | -0.000302525 |
| MC_ON | LN | probability | matched_uniform | 0/5 | 0/5 | +0.000000000 | +0.000000000 |
| MC_OFF | R4_normal | logit | ramp | 4/4 | 4/4 | +0.009310946 | -0.009311020 |
| MC_OFF | R4_normal | logit | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| MC_OFF | R4_normal | probability | ramp | 4/4 | 4/4 | +0.001715831 | -0.001479309 |
| MC_OFF | R4_normal | probability | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| MC_OFF | R4_H1_off | logit | ramp | 4/4 | 4/4 | +0.007769808 | -0.007769912 |
| MC_OFF | R4_H1_off | logit | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| MC_OFF | R4_H1_off | probability | ramp | 4/4 | 4/4 | +0.001467511 | -0.001246370 |
| MC_OFF | R4_H1_off | probability | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| MC_OFF | R4_AC_off | logit | ramp | 4/4 | 4/4 | +0.002676472 | -0.002676606 |
| MC_OFF | R4_AC_off | logit | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| MC_OFF | R4_AC_off | probability | ramp | 4/4 | 4/4 | +0.000514332 | -0.000346377 |
| MC_OFF | R4_AC_off | probability | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| MC_OFF | LN | logit | ramp | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| MC_OFF | LN | logit | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| MC_OFF | LN | probability | ramp | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| MC_OFF | LN | probability | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| PC_ON | R4_normal | logit | ramp | 9/9 | 9/9 | +0.006024493 | -0.006024559 |
| PC_ON | R4_normal | logit | matched_uniform | 0/9 | 0/9 | +0.000000000 | +0.000000000 |
| PC_ON | R4_normal | probability | ramp | 9/9 | 9/9 | +0.000970172 | -0.000859670 |
| PC_ON | R4_normal | probability | matched_uniform | 0/9 | 0/9 | +0.000000000 | +0.000000000 |
| PC_ON | R4_H1_off | logit | ramp | 8/9 | 8/9 | +0.005021479 | -0.005021559 |
| PC_ON | R4_H1_off | logit | matched_uniform | 0/9 | 0/9 | +0.000000000 | +0.000000000 |
| PC_ON | R4_H1_off | probability | ramp | 8/9 | 8/9 | +0.000833615 | -0.000737597 |
| PC_ON | R4_H1_off | probability | matched_uniform | 0/9 | 0/9 | +0.000000000 | +0.000000000 |
| PC_ON | R4_AC_off | logit | ramp | 9/9 | 9/9 | +0.001104302 | -0.001104315 |
| PC_ON | R4_AC_off | logit | matched_uniform | 0/9 | 0/9 | +0.000000000 | +0.000000000 |
| PC_ON | R4_AC_off | probability | ramp | 9/9 | 9/9 | +0.000178885 | -0.000123069 |
| PC_ON | R4_AC_off | probability | matched_uniform | 0/9 | 0/9 | +0.000000000 | +0.000000000 |
| PC_ON | LN | logit | ramp | 5/9 | 7/9 | +0.002959834 | -0.003453043 |
| PC_ON | LN | logit | matched_uniform | 0/9 | 0/9 | +0.000000000 | +0.000000000 |
| PC_ON | LN | probability | ramp | 5/9 | 7/9 | +0.000428789 | -0.000452506 |
| PC_ON | LN | probability | matched_uniform | 0/9 | 0/9 | +0.000000000 | +0.000000000 |
| PC_OFF | R4_normal | logit | ramp | 4/4 | 4/4 | +0.003372699 | -0.003372625 |
| PC_OFF | R4_normal | logit | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| PC_OFF | R4_normal | probability | ramp | 4/4 | 4/4 | +0.000734944 | -0.000609376 |
| PC_OFF | R4_normal | probability | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| PC_OFF | R4_H1_off | logit | ramp | 4/4 | 4/4 | +0.002399296 | -0.002399325 |
| PC_OFF | R4_H1_off | logit | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| PC_OFF | R4_H1_off | probability | ramp | 4/4 | 4/4 | +0.000510681 | -0.000409484 |
| PC_OFF | R4_H1_off | probability | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| PC_OFF | R4_AC_off | logit | ramp | 4/4 | 4/4 | +0.000957534 | -0.000957578 |
| PC_OFF | R4_AC_off | logit | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| PC_OFF | R4_AC_off | probability | ramp | 4/4 | 4/4 | +0.000229891 | -0.000179309 |
| PC_OFF | R4_AC_off | probability | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| PC_OFF | LN | logit | ramp | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| PC_OFF | LN | logit | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| PC_OFF | LN | probability | ramp | 0/4 | 0/4 | +0.000000000 | +0.000000000 |
| PC_OFF | LN | probability | matched_uniform | 0/4 | 0/4 | +0.000000000 | +0.000000000 |

Time courses: figures/mach-time-*.png. Full scan-position curves and metrics remain in responses.pt and per-cell-responses.csv.
