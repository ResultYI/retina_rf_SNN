# Canonical V1 shared-BC：Schottdorf–Lee 22-cell 结果

差值 = 当前 shared-BC − 上一版 overlapping-support。Population 为 22-cell arithmetic mean。

## Prediction

| Group | Cells | 上一版 NLL | 当前 NLL | 差值 |
| --- | --- | --- | --- | --- |
| ALL | 22 | 0.440025040 | 0.438956146 | -0.001068894 |
| MC_ON | 5 | 0.431602967 | 0.428506756 | -0.003096211 |
| MC_OFF | 4 | 0.443060599 | 0.441669881 | -0.001390718 |
| PC_ON | 9 | 0.426482081 | 0.427831286 | +0.001349204 |
| PC_OFF | 4 | 0.477988727 | 0.474335082 | -0.003653646 |

| Cell | Group | 上一版 NLL | 当前 NLL | 差值 | 当前 best / stop | 上一版 best / stop |
| --- | --- | --- | --- | --- | --- | --- |
| 67#4 | PC_OFF | 0.472746164 | 0.464896202 | -0.007849962 | 140 / 340 | 137 / 337 |
| 67#6 | MC_OFF | 0.443044573 | 0.442264438 | -0.000780135 | 369 / 569 | 370 / 570 |
| 67#7 | MC_ON | 0.458423227 | 0.456886798 | -0.001536429 | 276 / 476 | 276 / 476 |
| 67#14 | PC_ON | 0.369550139 | 0.372924745 | +0.003374606 | 250 / 450 | 817 / 1000 |
| 67#21 | PC_ON | 0.308554322 | 0.309125483 | +0.000571162 | 432 / 632 | 565 / 765 |
| 67#26 | PC_ON | 0.515001595 | 0.512515843 | -0.002485752 | 264 / 464 | 263 / 463 |
| 67#33 | MC_OFF | 0.369999975 | 0.369917274 | -0.000082701 | 581 / 781 | 581 / 781 |
| 67#34 | PC_ON | 0.387233406 | 0.386381626 | -0.000851780 | 384 / 584 | 382 / 582 |
| 68#3 | MC_OFF | 0.521043360 | 0.519419849 | -0.001623511 | 140 / 340 | 331 / 531 |
| 68#4 | PC_ON | 0.424961418 | 0.427100301 | +0.002138883 | 193 / 393 | 906 / 1000 |
| 68#7 | PC_ON | 0.381518781 | 0.385217726 | +0.003698945 | 79 / 279 | 169 / 369 |
| 68#10 | MC_ON | 0.427795291 | 0.421071053 | -0.006724238 | 211 / 411 | 325 / 525 |
| 68#11 | PC_OFF | 0.374329567 | 0.372821659 | -0.001507908 | 151 / 351 | 332 / 532 |
| 69#3 | PC_OFF | 0.557392120 | 0.553157747 | -0.004234374 | 573 / 773 | 535 / 735 |
| 69#4 | MC_ON | 0.434953779 | 0.433251053 | -0.001702726 | 133 / 333 | 193 / 393 |
| 69#6 | MC_OFF | 0.438154489 | 0.435077965 | -0.003076524 | 288 / 488 | 520 / 720 |
| 69#7 | MC_ON | 0.464252800 | 0.463302940 | -0.000949860 | 207 / 407 | 278 / 478 |
| 69#21 | PC_OFF | 0.507487059 | 0.506464720 | -0.001022339 | 252 / 452 | 251 / 451 |
| 70#1 | PC_ON | 0.546360016 | 0.545601189 | -0.000758827 | 413 / 613 | 402 / 602 |
| 70#7 | PC_ON | 0.472477108 | 0.472401470 | -0.000075638 | 222 / 422 | 248 / 448 |
| 70#15 | PC_ON | 0.432681948 | 0.439213187 | +0.006531239 | 57 / 257 | 106 / 306 |
| 70#34 | MC_ON | 0.372589737 | 0.368021935 | -0.004567802 | 171 / 371 | 225 / 425 |

## Pathway RF

RF = validation sequence endpoint logit Jacobian 的 sequence mean；norm 覆盖最后 16 bins × cone dimensions。dt = 6.666666667 ms，RF lag window = 106.666666667 ms。

既有 decomposition：direct-BC = RF(H1-off, AC-off)；AC = RF(H1-off) − direct-BC；H1 = global − RF(H1-off)。

| Pathway | 上一版 mean norm | 当前 mean norm | 差值 |
| --- | --- | --- | --- |
| H1 | 0.006099973 | 0.002217820 | -0.003882153 |
| direct_BC | 0.282721675 | 0.235542232 | -0.047179443 |
| AC | 0.130943973 | 0.107269160 | -0.023674814 |

| Cell | H1 norm | ΔH1 norm | direct-BC norm | Δdirect-BC norm | AC norm | ΔAC norm |
| --- | --- | --- | --- | --- | --- | --- |
| 67#4 | 0.000331320 | -0.000027007 | 0.289430325 | -0.056749507 | 0.071997543 | -0.056386113 |
| 67#6 | 0.007377615 | -0.002171771 | 0.206992216 | -0.033279067 | 0.096404921 | -0.020349810 |
| 67#7 | 0.003420267 | -0.006274040 | 0.213252662 | -0.037090935 | 0.131251938 | -0.007173963 |
| 67#14 | 0.001013562 | -0.010695483 | 0.284313177 | -0.035678205 | 0.116295523 | +0.008222567 |
| 67#21 | 0.000663875 | -0.003336551 | 0.214850761 | -0.034502086 | 0.087298661 | -0.025580956 |
| 67#26 | 0.000662613 | -0.001372918 | 0.224466282 | -0.063714981 | 0.093886907 | -0.046416306 |
| 67#33 | 0.004896434 | -0.010991215 | 0.303849518 | -0.035190131 | 0.100825261 | -0.001709942 |
| 67#34 | 0.000607636 | -0.003474139 | 0.234865544 | -0.067286932 | 0.104317709 | -0.035454803 |
| 68#3 | 0.002027799 | -0.006683837 | 0.207997546 | -0.020499119 | 0.104118172 | -0.006107416 |
| 68#4 | 0.000515743 | -0.009941806 | 0.154707355 | +0.010569539 | 0.169195890 | -0.012466865 |
| 68#7 | 0.000420950 | -0.000108925 | 0.268177976 | -0.059533673 | 0.106524346 | -0.047259042 |
| 68#10 | 0.000317129 | -0.003632829 | 0.190207354 | -0.036470974 | 0.110681080 | -0.010672876 |
| 68#11 | 0.000630822 | +0.000324091 | 0.408024813 | -0.055159046 | 0.074182974 | -0.013901853 |
| 69#3 | 0.011808332 | +0.000957875 | 0.308365339 | -0.069669776 | 0.087390380 | -0.040681664 |
| 69#4 | 0.000685523 | -0.006172515 | 0.203400719 | -0.051965585 | 0.137889903 | -0.018562698 |
| 69#6 | 0.006006010 | -0.002683273 | 0.183760662 | -0.030886032 | 0.101872845 | -0.001000183 |
| 69#7 | 0.003430961 | -0.005318711 | 0.195213570 | -0.052969846 | 0.128498275 | -0.010804074 |
| 69#21 | 0.000281386 | -0.000318085 | 0.208315093 | -0.073984348 | 0.075736987 | -0.061054527 |
| 70#1 | 0.000132666 | -0.000521338 | 0.184007504 | -0.035930539 | 0.115135595 | -0.021477624 |
| 70#7 | 0.001650076 | -0.004820738 | 0.269159749 | -0.074084082 | 0.119460166 | -0.037940435 |
| 70#15 | 0.000321152 | -0.000366887 | 0.236545315 | -0.054915692 | 0.106620474 | -0.045196545 |
| 70#34 | 0.001590173 | -0.007777261 | 0.192025613 | -0.068956736 | 0.120335962 | -0.008870775 |

## Structural perturbation

Δresponse = off − normal；每 cell 先对与上一版相同的全部 validation sequence bins（含 warmup）取 mean absolute，再对 cells 取平均。上一版 BC-off 对应本次 direct-BC-off。

| Clamp | 旧 mean abs Δlogit | 当前 mean abs Δlogit | 差值 | 旧 mean abs Δp | 当前 mean abs Δp | 差值 |
| --- | --- | --- | --- | --- | --- | --- |
| H1-off | 0.073935281 | 0.027476066 | -0.046459216 | 0.008660536 | 0.003436545 | -0.005223990 |
| direct_BC-off | 1.431789486 | 1.270634900 | -0.161154587 | 0.170222791 | 0.150953316 | -0.019269475 |
| AC-off | 1.173514880 | 1.045035151 | -0.128479729 | 0.146141684 | 0.140343818 | -0.005797866 |

### Per-cell mean absolute Δlogit

| Cell | H1-off | 与上一版差值 | direct-BC-off | 与上一版差值 | AC-off | 与上一版差值 |
| --- | --- | --- | --- | --- | --- | --- |
| 67#4 | 0.005929761 | -0.000587127 | 1.508245269 | -0.535654789 | 0.722186822 | -0.531254546 |
| 67#6 | 0.079660086 | -0.023295691 | 1.122469760 | -0.207636763 | 0.865936445 | -0.207732669 |
| 67#7 | 0.040789104 | -0.074568355 | 1.233566394 | -0.081743906 | 1.290193731 | -0.049216179 |
| 67#14 | 0.012103570 | -0.134020071 | 1.375185595 | +0.164085938 | 1.106602829 | +0.223284426 |
| 67#21 | 0.009553223 | -0.044331673 | 1.074514145 | -0.110672287 | 0.831140892 | -0.098618211 |
| 67#26 | 0.008630064 | -0.016241630 | 1.076411474 | -0.335146254 | 0.811608892 | -0.332377200 |
| 67#33 | 0.056916820 | -0.128155413 | 1.607933731 | -0.093838818 | 0.943046675 | -0.012578342 |
| 67#34 | 0.007345849 | -0.040899444 | 1.102946561 | -0.269294124 | 0.915643715 | -0.252304439 |
| 68#3 | 0.024866024 | -0.075848555 | 1.281544417 | +0.023722041 | 1.083168367 | +0.034593855 |
| 68#4 | 0.010446871 | -0.132955967 | 0.932930923 | +0.250628631 | 1.444894393 | +0.317134562 |
| 68#7 | 0.006548323 | -0.000739673 | 1.492178548 | -0.254414412 | 1.173401734 | -0.292849829 |
| 68#10 | 0.004454295 | -0.043266472 | 1.374859720 | -0.070040136 | 1.302987354 | -0.034973262 |
| 68#11 | 0.011878125 | +0.006528062 | 2.029750987 | -0.252895021 | 0.758114334 | -0.065715438 |
| 69#3 | 0.152303299 | +0.019806095 | 1.234095202 | -0.266837893 | 0.678357750 | -0.388218202 |
| 69#4 | 0.011909594 | -0.091890465 | 1.356355786 | -0.109629077 | 1.522748807 | -0.081630774 |
| 69#6 | 0.065372246 | -0.026523117 | 1.045321183 | -0.024863319 | 0.940070536 | -0.013744834 |
| 69#7 | 0.043107860 | -0.050583787 | 1.138829313 | -0.185870296 | 1.242950602 | -0.065836890 |
| 69#21 | 0.004642859 | -0.004871476 | 1.102976872 | -0.472078507 | 0.750188560 | -0.480418880 |
| 70#1 | 0.002302805 | -0.008691560 | 1.002461820 | -0.097410505 | 1.041862341 | -0.064101443 |
| 70#7 | 0.020013759 | -0.053705813 | 1.276004972 | -0.168285026 | 1.088987345 | -0.148297416 |
| 70#15 | 0.004962188 | -0.004690194 | 1.375222972 | -0.250513717 | 1.221529841 | -0.231263530 |
| 70#34 | 0.020736721 | -0.092570422 | 1.210162144 | -0.197012668 | 1.255151354 | -0.050434804 |

### Per-cell mean absolute Δprobability

| Cell | H1-off | 与上一版差值 | direct-BC-off | 与上一版差值 | AC-off | 与上一版差值 |
| --- | --- | --- | --- | --- | --- | --- |
| 67#4 | 0.000661106 | -0.000034626 | 0.242474682 | -0.079297101 | 0.067042358 | -0.031969752 |
| 67#6 | 0.011691766 | -0.003708528 | 0.165205621 | -0.032083785 | 0.107549578 | -0.016453738 |
| 67#7 | 0.004456234 | -0.007481750 | 0.140316908 | -0.005305729 | 0.171132913 | +0.001991063 |
| 67#14 | 0.001244420 | -0.012477108 | 0.142125043 | +0.000153713 | 0.144997597 | +0.036423323 |
| 67#21 | 0.000870927 | -0.003976199 | 0.084847065 | -0.003039461 | 0.103319544 | -0.009816529 |
| 67#26 | 0.001391179 | -0.002725909 | 0.152527963 | -0.025369046 | 0.134053351 | -0.041697435 |
| 67#33 | 0.006821662 | -0.015464281 | 0.175939347 | -0.017701825 | 0.099713583 | +0.005707151 |
| 67#34 | 0.000701993 | -0.003838413 | 0.113856432 | -0.014188341 | 0.117301856 | -0.027429635 |
| 68#3 | 0.004114948 | -0.012327985 | 0.202022315 | +0.001407065 | 0.144157939 | +0.006629801 |
| 68#4 | 0.001051138 | -0.010826632 | 0.074753142 | +0.007888643 | 0.202410085 | +0.058846803 |
| 68#7 | 0.000870912 | +0.000017575 | 0.150937548 | -0.002796955 | 0.168127599 | -0.025075424 |
| 68#10 | 0.000621089 | -0.005632774 | 0.135218060 | -0.008357930 | 0.198545425 | +0.001447081 |
| 68#11 | 0.001053203 | +0.000648850 | 0.207516462 | -0.023515876 | 0.055881829 | -0.000138412 |
| 69#3 | 0.017010025 | +0.000821963 | 0.203412906 | -0.051513802 | 0.093841937 | -0.037153816 |
| 69#4 | 0.001288145 | -0.008918802 | 0.142987527 | -0.009406406 | 0.207475935 | +0.008597855 |
| 69#6 | 0.009847783 | -0.004702569 | 0.151818619 | -0.011951526 | 0.111555999 | +0.003911583 |
| 69#7 | 0.005533619 | -0.006542376 | 0.140540642 | -0.019509068 | 0.182575477 | -0.000646434 |
| 69#21 | 0.000647704 | -0.000667259 | 0.200191059 | -0.069467784 | 0.080644916 | -0.035854389 |
| 70#1 | 0.000402770 | -0.001543538 | 0.112190225 | -0.014634958 | 0.191704159 | -0.003092672 |
| 70#7 | 0.002631243 | -0.006830414 | 0.156950375 | -0.009512638 | 0.162337557 | -0.007403837 |
| 70#15 | 0.000620350 | -0.000665132 | 0.102384631 | -0.018341169 | 0.178471202 | -0.026963432 |
| 70#34 | 0.002071784 | -0.008051880 | 0.122756378 | -0.017384476 | 0.164723157 | +0.012587792 |

## Learned pathway quantities

| Cell | Group | H1 amplitude | direct-BC gain | AC gain | AC local mixture | AC transient mixture |
| --- | --- | --- | --- | --- | --- | --- |
| 67#4 | PC_OFF | 0.008410 | 0.751732 | 0.792327 | 0.500222 | 0.499778 |
| 67#6 | MC_OFF | 0.150295 | 0.801533 | 0.861873 | 0.093228 | 0.906772 |
| 67#7 | MC_ON | 0.074373 | 0.779602 | 1.230705 | 0.193256 | 0.806744 |
| 67#14 | PC_ON | 0.020809 | 0.768775 | 1.184752 | 0.340671 | 0.659329 |
| 67#21 | PC_ON | 0.027682 | 0.589580 | 0.927009 | 0.426928 | 0.573072 |
| 67#26 | PC_ON | 0.028854 | 0.634069 | 0.788831 | 0.205206 | 0.794794 |
| 67#33 | MC_OFF | 0.058704 | 1.100884 | 0.995463 | 0.260872 | 0.739128 |
| 67#34 | PC_ON | 0.015335 | 0.666737 | 1.015980 | 0.320452 | 0.679548 |
| 68#3 | MC_OFF | 0.049362 | 0.737340 | 1.048223 | 0.259441 | 0.740559 |
| 68#4 | PC_ON | 0.024940 | 0.564246 | 2.139089 | 0.028755 | 0.971245 |
| 68#7 | PC_ON | 0.015117 | 0.701974 | 1.122170 | 0.395512 | 0.604488 |
| 68#10 | MC_ON | 0.017240 | 0.684851 | 1.149969 | 0.263480 | 0.736520 |
| 68#11 | PC_OFF | 0.011240 | 1.016348 | 1.019006 | 0.592532 | 0.407468 |
| 69#3 | PC_OFF | 0.178413 | 0.758735 | 1.109316 | 0.587800 | 0.412200 |
| 69#4 | MC_ON | 0.025250 | 0.727356 | 1.352408 | 0.265992 | 0.734008 |
| 69#6 | MC_OFF | 0.138744 | 0.705319 | 0.942418 | 0.126663 | 0.873337 |
| 69#7 | MC_ON | 0.090101 | 0.729783 | 1.201694 | 0.165097 | 0.834903 |
| 69#21 | PC_OFF | 0.014005 | 0.558587 | 0.767147 | 0.430530 | 0.569470 |
| 70#1 | PC_ON | 0.028719 | 0.505780 | 0.899138 | 0.104516 | 0.895484 |
| 70#7 | PC_ON | 0.050404 | 0.712997 | 0.983214 | 0.238957 | 0.761043 |
| 70#15 | PC_ON | 0.018569 | 0.672249 | 1.216757 | 0.345904 | 0.654096 |
| 70#34 | MC_ON | 0.043944 | 0.706970 | 1.179319 | 0.234899 | 0.765101 |

### Shared BC spatial-mode weights

每个数值为 effective normalized BC weights 对 temporal basis 轴求和；下表四项合计为 1。Modes 的 σ：PC = [0.05, 0.14] deg；MC = [0.09, 0.20] deg。

| Cell | Sustained mode 0 | Sustained mode 1 | Transient mode 0 | Transient mode 1 |
| --- | --- | --- | --- | --- |
| 67#4 | 0.087072 | 0.080039 | 0.601116 | 0.231773 |
| 67#6 | 0.063424 | 0.065052 | 0.500871 | 0.370652 |
| 67#7 | 0.061430 | 0.060813 | 0.507270 | 0.370487 |
| 67#14 | 0.113789 | 0.102764 | 0.487421 | 0.296026 |
| 67#21 | 0.126459 | 0.107369 | 0.591512 | 0.174659 |
| 67#26 | 0.105719 | 0.101760 | 0.589588 | 0.202933 |
| 67#33 | 0.010825 | 0.010963 | 0.800140 | 0.178072 |
| 67#34 | 0.091644 | 0.089142 | 0.612711 | 0.206502 |
| 68#3 | 0.079676 | 0.079162 | 0.445376 | 0.395785 |
| 68#4 | 0.266561 | 0.264394 | 0.350612 | 0.118432 |
| 68#7 | 0.152864 | 0.149429 | 0.398597 | 0.299110 |
| 68#10 | 0.136858 | 0.138171 | 0.363510 | 0.361461 |
| 68#11 | 0.038124 | 0.034513 | 0.593573 | 0.333791 |
| 69#3 | 0.031354 | 0.022023 | 0.793488 | 0.153135 |
| 69#4 | 0.078581 | 0.078554 | 0.438685 | 0.404180 |
| 69#6 | 0.080794 | 0.080921 | 0.507561 | 0.330724 |
| 69#7 | 0.079266 | 0.078325 | 0.458763 | 0.383647 |
| 69#21 | 0.110867 | 0.104103 | 0.596651 | 0.188379 |
| 70#1 | 0.177738 | 0.177618 | 0.515077 | 0.129568 |
| 70#7 | 0.083007 | 0.080716 | 0.510380 | 0.325898 |
| 70#15 | 0.192274 | 0.185882 | 0.324148 | 0.297697 |
| 70#34 | 0.079932 | 0.080104 | 0.442292 | 0.397671 |

### τ (ms)

| Cell | H1 | BC-S0 | BC-S1 | BC-S2 | BC-T0 | BC-T1 | BC-T2 | AC local state | AC transient state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 67#4 | 28.483435 | 27.736614 | 48.902561 | 93.098137 | 9.778109 | 11.228095 | 17.103630 | 25.180153 | 17.315458 |
| 67#6 | 13.610455 | 27.004856 | 63.458561 | 164.027740 | 6.131807 | 6.828493 | 9.643791 | 25.710909 | 15.521210 |
| 67#7 | 16.936798 | 23.869255 | 65.701904 | 115.072540 | 6.990528 | 7.599343 | 8.435610 | 25.957062 | 15.933421 |
| 67#14 | 14.508286 | 26.162041 | 35.232632 | 102.342072 | 7.685502 | 8.418303 | 10.293527 | 30.737087 | 16.627443 |
| 67#21 | 27.759937 | 22.213608 | 79.664619 | 124.442131 | 8.096545 | 10.246637 | 10.484266 | 23.646191 | 18.011694 |
| 67#26 | 25.366680 | 24.063835 | 73.563599 | 129.165878 | 7.237331 | 7.950094 | 9.102520 | 24.048716 | 16.301861 |
| 67#33 | 13.107317 | 24.814249 | 28.967192 | 29.343369 | 6.257749 | 6.267142 | 11.256004 | 25.431858 | 15.969792 |
| 67#34 | 16.162506 | 25.693966 | 58.897045 | 128.783386 | 6.920104 | 7.396780 | 8.344155 | 41.343887 | 16.192980 |
| 68#3 | 18.612823 | 25.175060 | 30.329048 | 116.730713 | 7.581129 | 8.766867 | 11.454160 | 25.977142 | 16.201843 |
| 68#4 | 58.487434 | 64.913895 | 98.899673 | 143.579681 | 6.138451 | 8.068508 | 32.746574 | 28.545271 | 15.510221 |
| 68#7 | 29.798466 | 25.800982 | 50.436138 | 101.628845 | 11.295412 | 13.795228 | 17.188423 | 27.778770 | 17.565817 |
| 68#10 | 29.969931 | 22.487417 | 69.409134 | 111.883820 | 11.508640 | 12.845411 | 12.461733 | 26.872686 | 16.763563 |
| 68#11 | 43.113480 | 25.020147 | 31.513702 | 36.146515 | 10.945656 | 11.001307 | 12.455508 | 27.664345 | 21.544666 |
| 69#3 | 10.504631 | 24.165154 | 29.324652 | 29.215044 | 10.178128 | 10.254177 | 12.172455 | 27.356750 | 21.492666 |
| 69#4 | 35.575146 | 23.667210 | 55.743797 | 102.402809 | 8.846705 | 10.136488 | 10.827351 | 26.335575 | 17.056824 |
| 69#6 | 13.023571 | 25.468281 | 41.998238 | 148.710480 | 6.670897 | 7.484336 | 8.685798 | 24.746557 | 15.656293 |
| 69#7 | 18.535198 | 24.539448 | 81.116394 | 119.683174 | 6.987890 | 7.643443 | 8.554175 | 26.030956 | 15.785656 |
| 69#21 | 34.133461 | 24.294735 | 31.046991 | 80.653236 | 7.347667 | 13.013548 | 16.372875 | 23.611340 | 16.953918 |
| 70#1 | 53.622967 | 23.093594 | 75.815872 | 119.310120 | 9.880663 | 13.298150 | 13.637646 | 23.789804 | 15.997605 |
| 70#7 | 18.559219 | 23.507410 | 58.099812 | 116.504761 | 9.224392 | 9.550943 | 9.993811 | 24.908110 | 16.141542 |
| 70#15 | 31.843086 | 26.688925 | 72.549797 | 110.963028 | 8.481846 | 14.058543 | 20.510479 | 26.940781 | 16.913530 |
| 70#34 | 19.490009 | 23.861029 | 71.345184 | 115.761162 | 8.023805 | 8.653714 | 9.400820 | 26.061401 | 16.197939 |

### Explicit pathway delay (ms)

| Cell | H1 | BC sustained | BC transient | AC local downstream | AC transient downstream |
| --- | --- | --- | --- | --- | --- |
| 67#4 | 3.968574 | 15.362779 | 12.159400 | 8.811447 | 5.782973 |
| 67#6 | 19.700211 | 16.794132 | 15.217829 | 16.278934 | 14.151438 |
| 67#7 | 17.312649 | 14.693800 | 12.428380 | 16.800428 | 14.174872 |
| 67#14 | 13.439188 | 15.385842 | 13.785748 | 17.802433 | 14.802691 |
| 67#21 | 18.105232 | 21.266520 | 17.570206 | 11.428585 | 9.098211 |
| 67#26 | 14.892472 | 18.480188 | 15.333973 | 9.866995 | 7.441782 |
| 67#33 | 19.434517 | 16.123833 | 15.084780 | 21.335541 | 19.632458 |
| 67#34 | 18.290277 | 19.840733 | 17.378332 | 19.512152 | 16.816820 |
| 68#3 | 18.372620 | 14.591481 | 13.294294 | 16.306862 | 14.202930 |
| 68#4 | 6.442266 | 16.520693 | 1.299251 | 5.721334 | 1.854110 |
| 68#7 | 4.640273 | 16.446728 | 12.529642 | 11.780376 | 7.887702 |
| 68#10 | 6.991621 | 20.685173 | 17.128202 | 7.675368 | 4.580566 |
| 68#11 | 13.302060 | 12.540104 | 10.692314 | 17.716372 | 14.822658 |
| 69#3 | 0.216525 | 18.532690 | 17.004402 | 18.184406 | 15.911083 |
| 69#4 | 15.559727 | 11.613822 | 8.257568 | 16.569839 | 13.644557 |
| 69#6 | 18.787497 | 18.411140 | 17.057325 | 16.310232 | 14.316389 |
| 69#7 | 15.286242 | 15.675103 | 12.568373 | 16.037951 | 13.297011 |
| 69#21 | 3.178016 | 16.428864 | 10.877281 | 5.571681 | 3.148203 |
| 70#1 | 1.990146 | 20.922096 | 14.635897 | 4.471296 | 1.766375 |
| 70#7 | 18.471106 | 20.593382 | 18.105347 | 11.173774 | 8.497901 |
| 70#15 | 3.058449 | 16.049599 | 10.541677 | 8.720669 | 4.447929 |
| 70#34 | 17.373425 | 17.797585 | 15.019074 | 16.060877 | 13.488235 |

RF lag window 与 τ / explicit pathway delay 分列；RGC history shift = 1 bin = 6.666666667 ms。

## Verification

| Check | Passed cells |
| --- | --- |
| identity_current | 22/22 |
| shared_encoder_parameter_identity | 22/22 |
| views_reconstruct_from_same_weights | 22/22 |
| views_differ_only_by_support | 22/22 |
| one_BC_temporal_encoder | 22/22 |
| AC_has_only_downstream_parameters | 22/22 |
| AC_depends_on_BC_broad | 22/22 |
| AC_without_BC_has_no_stimulus_gradient | 22/22 |
| direct_BC_off_preserves_BC_broad_and_AC | 22/22 |
| AC_off_preserves_H1_and_BC_views | 22/22 |
| H1_off_propagates_to_BC_and_AC | 22/22 |
| H1_exact_zero | 22/22 |
| direct_BC_exact_zero | 22/22 |
| AC_exact_zero | 22/22 |
| all_outputs_finite | 22/22 |
| state_dict_unchanged | 22/22 |
| all_passed | 22/22 |
| prediction_replay_exact | 22/22 |
| targets_masks_segments_equal_previous | 22/22 |
| best_stopping_step_replayed | 22/22 |
| inference_state_unchanged_including_RF | 22/22 |
| inference_parameter_gradients_absent | 22/22 |
| inference_training_mode_unchanged | 22/22 |
| RF_and_parameters_finite | 22/22 |

旧 checkpoint 读取：false。模型/训练/data/baseline 源码 SHA256 unchanged：true。Fresh initialization、fresh optimizer、train/dev boundaries、refit steps 的检查在各 cell 训练运行中通过。

## Artifacts

- [Prediction / RF / perturbation per-cell & group comparison](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/comparison.json)
- [Per-cell CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/per-cell-comparison.csv)
- [Group CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/group-comparison.csv)
- [Learned pathway quantities](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/learned-pathway-quantities.csv)
- [Effective parameter tensors](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/effective-parameters.pt)
- [RF tensors](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/rf-tensors.pt)
- [Perturbation tensors](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/perturbation-tensors.pt)
- [Training results](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/results.json)
- [Frozen-source manifest](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/run-manifest.json)
- [Training log](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/training.log)


| Cell | Checkpoints / prediction / causal tensors | Inner-dev trajectory | Refit trajectory |
| --- | --- | --- | --- |
| 67#4 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_4) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_4/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_4/refit-trajectory.csv) |
| 67#6 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_6) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_6/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_6/refit-trajectory.csv) |
| 67#7 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_7) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_7/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_7/refit-trajectory.csv) |
| 67#14 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_14) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_14/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_14/refit-trajectory.csv) |
| 67#21 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_21) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_21/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_21/refit-trajectory.csv) |
| 67#26 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_26) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_26/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_26/refit-trajectory.csv) |
| 67#33 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_33) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_33/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_33/refit-trajectory.csv) |
| 67#34 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_34) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_34/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/67_34/refit-trajectory.csv) |
| 68#3 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_3) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_3/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_3/refit-trajectory.csv) |
| 68#4 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_4) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_4/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_4/refit-trajectory.csv) |
| 68#7 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_7) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_7/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_7/refit-trajectory.csv) |
| 68#10 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_10) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_10/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_10/refit-trajectory.csv) |
| 68#11 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_11) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_11/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/68_11/refit-trajectory.csv) |
| 69#3 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_3) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_3/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_3/refit-trajectory.csv) |
| 69#4 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_4) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_4/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_4/refit-trajectory.csv) |
| 69#6 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_6) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_6/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_6/refit-trajectory.csv) |
| 69#7 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_7) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_7/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_7/refit-trajectory.csv) |
| 69#21 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_21) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_21/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/69_21/refit-trajectory.csv) |
| 70#1 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_1) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_1/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_1/refit-trajectory.csv) |
| 70#7 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_7) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_7/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_7/refit-trajectory.csv) |
| 70#15 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_15) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_15/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_15/refit-trajectory.csv) |
| 70#34 | [cell artifacts](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_34) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_34/inner-trajectory.csv) | [CSV](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells/70_34/refit-trajectory.csv) |

