# Canonical V1 shared-BC frozen application replay

22 cells: MC ON 5; MC OFF 4; PC ON 9; PC OFF 4. All group/population values are equal-cell means.

Original saved stimulus tensors were loaded. Temporal: 7 probes + 2 existing center references + blank, 450 bins at 150 Hz. Illusion: 72 saved sequences, 150 bins at 150 Hz. No training, optimizer, new stimulus family or RMS normalization.

Temporal mean absolute clamp effect uses the unchanged 300–2300 ms window. Signed peak/integral suppression–facilitation is the unchanged condition-minus-same-clamp-center-only difference. Illusion signed mean-on uses 300–400 ms. Peak, integral, latency and onset/offset definitions are unchanged.

All version deltas below are shared-BC minus overlapping-support. direct-BC-off has no previous counterpart. Tau, explicit pathway delay, RF lag window and strictly-past history shift are unchanged and are not estimated here.

## all

### Temporal clamp effects

logit: mean absolute off−normal

| Probe | H1-off | direct-BC-off | AC-off | H1 effect Δ vs old | AC effect Δ vs old |
|---|---|---|---|---|---|
| slow_step | 0.00167848096 | 0.411421039 | 0.190414364 | -0.00195901283 | -0.0553847165 |
| slow_1Hz | 0.001809913 | 0.411484965 | 0.189859197 | -0.00234065367 | -0.0559027236 |
| slow_2Hz | 0.00206804521 | 0.4116122 | 0.188730803 | -0.00305679028 | -0.0569611131 |
| rapid_10Hz | 0.0023093329 | 0.412103278 | 0.183508144 | -0.0042119078 | -0.0596447296 |
| rapid_20Hz | 0.00123881176 | 0.412185592 | 0.181635204 | -0.00197976566 | -0.0597544522 |
| transient_50ms | 0.00107855876 | 0.41221342 | 0.180629206 | -0.00138213937 | -0.059654964 |
| large_field_50ms | 0.000637587385 | 0.0204986175 | 0.0177682535 | -0.00115320302 | -0.00227151413 |

probability: mean absolute off−normal

| Probe | H1-off | direct-BC-off | AC-off | H1 effect Δ vs old | AC effect Δ vs old |
|---|---|---|---|---|---|
| slow_step | 0.000324133841 | 0.0656441479 | 0.0337774565 | -0.000294042141 | -0.010928775 |
| slow_1Hz | 0.000349757406 | 0.0656458764 | 0.0337659613 | -0.00035749219 | -0.0110316639 |
| slow_2Hz | 0.000400098078 | 0.0656487816 | 0.0337430754 | -0.000476779881 | -0.0112409118 |
| rapid_10Hz | 0.000450859047 | 0.0657529604 | 0.0337885459 | -0.000692227443 | -0.0118950227 |
| rapid_20Hz | 0.000240327114 | 0.0658186803 | 0.0336770158 | -0.000334795664 | -0.0118790746 |
| transient_50ms | 0.000211380749 | 0.0658711242 | 0.0334929799 | -0.000232770528 | -0.011842313 |
| large_field_50ms | 0.000110455372 | 0.00320658314 | 0.00276855112 | -0.000170862058 | -0.000374391278 |

### Illusion signed response signatures

logit: signed mean-on pair difference

| Signature | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|
| Mach dark | -0.00371583231 | -0.00413708009 | -0.00638466993 | 0.00267793842 | -0.000345297584 | -0.000521461669 | 0.000394574594 |
| Mach bright | 0.00371583501 | 0.0041370807 | 0.00638467162 | -0.00267793507 | 0.000345299192 | 0.000521457166 | -0.000394572848 |
| SBC | -0.0389033195 | -0.041900723 | -0.0419592022 | 0.00305592033 | -0.0021933324 | -0.00365544263 | 0.00922306065 |
| Hermann original | -0.0199128712 | -0.0209471296 | -0.0209211272 | 0.00105291016 | -0.00122396504 | -0.00182774231 | 0.00318779182 |
| Hermann diagnostic | -0.0199151617 | -0.0209471296 | -0.0209210725 | 0.00105057691 | -0.00122621833 | -0.00182774231 | 0.0031801206 |
| White original | 0.0194458473 | 0.02095797 | 0.020987958 | -0.00154212036 | 0.000905633311 | 0.00182722661 | -0.00466388632 |
| White diagnostic | 0.0194380595 | 0.02095797 | 0.0209881358 | -0.00155008995 | 0.000894496641 | 0.00182722661 | -0.00468988327 |

probability: signed mean-on pair difference

| Signature | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|
| Mach dark | -0.000608610499 | -0.000705995684 | -0.00120779456 | -8.72221982e-05 | -9.52876222e-05 | -9.89019788e-05 | 0.000157084307 |
| Mach bright | 0.000498400407 | 0.000571247353 | 0.000517005061 | -0.000708154067 | 5.93350507e-05 | 9.68403509e-05 | -2.18992884e-05 |
| SBC | -0.00503107205 | -0.00564286695 | -0.00571418127 | 0.000683450893 | -0.000728884627 | -0.000907658864 | 0.00118270203 |
| Hermann original | -0.00330734411 | -0.00348886937 | -0.00180294393 | 0.000111664131 | -0.000156204015 | -0.000269345034 | 0.000708708153 |
| Hermann diagnostic | -0.00330780686 | -0.00348886937 | -0.00180279623 | 0.000111363718 | -0.000156489709 | -0.000269345034 | 0.000707167607 |
| White original | 0.00251076041 | 0.00282175153 | 0.00285642149 | -0.000345711838 | 0.000340211057 | 0.000454259306 | -0.000598186013 |
| White diagnostic | 0.00250903389 | 0.00282175153 | 0.0028565648 | -0.000347585639 | 0.000338791467 | 0.000454259306 | -0.000601522272 |

### Diagnostic minus original

| Family | Channel | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|---|
| White | logit | -7.78743687e-06 | 0 | 1.78452694e-07 | -7.96968326e-06 | -1.11355924e-05 | 0 | -2.59970178e-05 |
| White | probability | -1.72650277e-06 | 0 | 1.43367233e-07 | -1.87379845e-06 | -1.41965172e-06 | 0 | -3.3362571e-06 |
| Hermann | logit | -2.29062449e-06 | 0 | 5.49085464e-08 | -2.33325093e-06 | -2.25341685e-06 | 0 | -7.67129842e-06 |
| Hermann | probability | -4.62771372e-07 | 0 | 1.47679532e-07 | -3.00416438e-07 | -2.85786214e-07 | 0 | -1.54057687e-06 |

[Temporal figure](comparison-figures/temporal-all.png) · [Illusion figure](comparison-figures/illusion-all.png) · [Diagnostic figure](comparison-figures/diagnostic-all.png)

## MC_ON

### Temporal clamp effects

logit: mean absolute off−normal

| Probe | H1-off | direct-BC-off | AC-off | H1 effect Δ vs old | AC effect Δ vs old |
|---|---|---|---|---|---|
| slow_step | 0.000424836203 | 0.408854103 | 0.237173942 | -0.0019086261 | -0.0428808779 |
| slow_1Hz | 0.000605444075 | 0.408941323 | 0.237133098 | -0.0023728424 | -0.0428694367 |
| slow_2Hz | 0.000965581386 | 0.409114701 | 0.237065336 | -0.00329438013 | -0.0428463548 |
| rapid_10Hz | 0.00175346259 | 0.4097839 | 0.233816615 | -0.00551182341 | -0.0430344731 |
| rapid_20Hz | 0.000767711096 | 0.40987128 | 0.231285015 | -0.00257815298 | -0.0432567447 |
| transient_50ms | 0.000336074823 | 0.409892696 | 0.230091932 | -0.00161793489 | -0.0430642158 |
| large_field_50ms | 0.000639354673 | 0.0204004746 | 0.0222271267 | -0.00184012351 | -0.000859645009 |

probability: mean absolute off−normal

| Probe | H1-off | direct-BC-off | AC-off | H1 effect Δ vs old | AC effect Δ vs old |
|---|---|---|---|---|---|
| slow_step | 7.44519726e-05 | 0.0641921498 | 0.0423824854 | -0.000311228042 | -0.00750293136 |
| slow_1Hz | 0.000106661867 | 0.0641959831 | 0.0424914241 | -0.000390541732 | -0.00753500611 |
| slow_2Hz | 0.000170853196 | 0.0642027378 | 0.0427171424 | -0.000547766373 | -0.00759981275 |
| rapid_10Hz | 0.000314564147 | 0.0643201292 | 0.0435584024 | -0.000940249424 | -0.00812611654 |
| rapid_20Hz | 0.000138200271 | 0.0644041076 | 0.0434807241 | -0.000447386732 | -0.00812458768 |
| transient_50ms | 5.92781798e-05 | 0.0644636832 | 0.0432644211 | -0.000281227224 | -0.00805972144 |
| large_field_50ms | 0.000104758312 | 0.00325659853 | 0.00355434148 | -0.000277781306 | -3.78603581e-05 |

### Illusion signed response signatures

logit: signed mean-on pair difference

| Signature | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|
| Mach dark | -0.00264964898 | -0.00209546721 | -0.0213008448 | 0.0186472081 | 0.00175493085 | 0.000158303662 | -0.000266650319 |
| Mach bright | 0.00264965848 | 0.0020954752 | 0.0213008497 | -0.0186472047 | -0.00175491977 | -0.000158303522 | 0.000266646966 |
| SBC | -0.179823506 | -0.175696868 | -0.170059052 | -0.00977978371 | 0.0449789643 | 0.0329830557 | 0.0329987582 |
| Hermann original | -0.0892868161 | -0.0877763525 | -0.0858655587 | -0.0033305621 | 0.0208897278 | 0.0164531469 | 0.0112232631 |
| Hermann diagnostic | -0.0892844826 | -0.0877763525 | -0.0858690128 | -0.00332476688 | 0.020883505 | 0.0164531469 | 0.0112038942 |
| White original | 0.0901475996 | 0.0878842041 | 0.0852745727 | 0.00487688053 | -0.0231836408 | -0.0165036842 | -0.0164523032 |
| White diagnostic | 0.090159823 | 0.0878842041 | 0.0852674037 | 0.00489626243 | -0.0232197866 | -0.0165036842 | -0.0165173026 |

probability: signed mean-on pair difference

| Signature | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|
| Mach dark | -0.000474183951 | -0.000385141573 | -0.00403066729 | 0.00242187618 | 0.000252714151 | 1.75066772e-05 | 0.000108920224 |
| Mach bright | 0.000399304836 | 0.000307460426 | 0.00281235976 | -0.00354873044 | -0.000278409701 | -3.34554992e-05 | 3.26359179e-05 |
| SBC | -0.0289012402 | -0.028227181 | -0.0272940166 | -0.00160970749 | 0.00617570728 | 0.00436744615 | 0.0050524373 |
| Hermann original | -0.0153791785 | -0.0151289981 | -0.0117611369 | -0.000662815 | 0.00335807167 | 0.002674881 | 0.00221930119 |
| Hermann diagnostic | -0.0153788291 | -0.0151289981 | -0.0117615484 | -0.00066166858 | 0.00335726589 | 0.002674881 | 0.00221555934 |
| White original | 0.014488315 | 0.0141181886 | 0.0136862261 | 0.000802714619 | -0.00319184083 | -0.00218329374 | -0.00251897056 |
| White diagnostic | 0.0144903194 | 0.0141181886 | 0.0136850398 | 0.000805905281 | -0.00319732968 | -0.00218329374 | -0.00252892132 |

### Diagnostic minus original

| Family | Channel | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|---|
| White | logit | 1.22229259e-05 | 0 | -7.16845198e-06 | 1.93818414e-05 | -3.61442566e-05 | 0 | -6.49992639e-05 |
| White | probability | 2.00450415e-06 | 0 | -1.18613242e-06 | 3.19063669e-06 | -5.48938915e-06 | 0 | -9.95079663e-06 |
| Hermann | logit | 2.33093899e-06 | 0 | -3.45389042e-06 | 5.79516091e-06 | -6.22749335e-06 | 0 | -1.93691252e-05 |
| Hermann | probability | 3.49084549e-07 | 0 | -4.11768761e-07 | 1.14639599e-06 | -8.06649535e-07 | 0 | -3.74197962e-06 |

[Temporal figure](comparison-figures/temporal-MC_ON.png) · [Illusion figure](comparison-figures/illusion-MC_ON.png) · [Diagnostic figure](comparison-figures/diagnostic-MC_ON.png)

## MC_OFF

### Temporal clamp effects

logit: mean absolute off−normal

| Probe | H1-off | direct-BC-off | AC-off | H1 effect Δ vs old | AC effect Δ vs old |
|---|---|---|---|---|---|
| slow_step | 0.00297783833 | 0.406921424 | 0.175817754 | -0.00313804101 | -0.0339258201 |
| slow_1Hz | 0.00317936094 | 0.406967819 | 0.175793733 | -0.00339358108 | -0.0339218043 |
| slow_2Hz | 0.00357182644 | 0.407059409 | 0.175755501 | -0.00390095561 | -0.0339135788 |
| rapid_10Hz | 0.00481992884 | 0.407526739 | 0.17364464 | -0.00540774321 | -0.0341124237 |
| rapid_20Hz | 0.00299433956 | 0.407660104 | 0.171698857 | -0.00332480937 | -0.0342500545 |
| transient_50ms | 0.00283940506 | 0.407708623 | 0.170901764 | -0.00298555227 | -0.0341581032 |
| large_field_50ms | 0.00129289126 | 0.0195309636 | 0.0156873628 | -0.00145822689 | -0.000874337507 |

probability: mean absolute off−normal

| Probe | H1-off | direct-BC-off | AC-off | H1 effect Δ vs old | AC effect Δ vs old |
|---|---|---|---|---|---|
| slow_step | 0.000523493538 | 0.0696378546 | 0.0339562357 | -0.000554905331 | -0.00716480007 |
| slow_1Hz | 0.000562081957 | 0.0696430868 | 0.0339999665 | -0.00060560905 | -0.00717581343 |
| slow_2Hz | 0.000637297075 | 0.0696528824 | 0.0340922619 | -0.000706428946 | -0.00719800452 |
| rapid_10Hz | 0.000883506764 | 0.0697711604 | 0.0343471775 | -0.00102372135 | -0.00742132403 |
| rapid_20Hz | 0.000554994192 | 0.0698511386 | 0.034229124 | -0.000632675648 | -0.00743289199 |
| transient_50ms | 0.000531220612 | 0.0698927566 | 0.034088118 | -0.00056814032 | -0.00739072729 |
| large_field_50ms | 0.000213936302 | 0.00328685914 | 0.00267925416 | -0.000242148319 | -0.000149700791 |

### Illusion signed response signatures

logit: signed mean-on pair difference

| Signature | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|
| Mach dark | -0.00353039992 | -0.00526031863 | 0.0147791596 | -0.0182814733 | -0.00182039717 | 7.53869099e-05 | -0.000519667054 |
| Mach bright | 0.00353039992 | 0.00526030961 | -0.0147791633 | 0.0182814766 | 0.00182038726 | -7.54028842e-05 | 0.000519662164 |
| SBC | 0.137252245 | 0.124750905 | 0.116965253 | 0.02030918 | -0.0311342664 | -0.017373722 | -0.0261335326 |
| Hermann original | 0.0666653132 | 0.0622589244 | 0.0597652141 | 0.00691719272 | -0.0135322707 | -0.00866246317 | -0.00887954806 |
| Hermann diagnostic | 0.0666580293 | 0.0622589244 | 0.0597700216 | 0.00690513855 | -0.0135251265 | -0.00866246317 | -0.00886446814 |
| White original | -0.0689106258 | -0.0623939801 | -0.058786395 | -0.0101298133 | 0.0159211671 | 0.00869122893 | 0.0130201874 |
| White diagnostic | -0.0689410288 | -0.0623939801 | -0.0587763321 | -0.0101702849 | 0.0159534942 | 0.00869122893 | 0.0130707084 |

probability: signed mean-on pair difference

| Signature | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|
| Mach dark | -0.00065230877 | -0.000960931655 | 0.0022172516 | -0.00362743536 | -0.000325622184 | 2.09950094e-05 | -6.75581978e-05 |
| Mach bright | 0.000422806803 | 0.000684708364 | -0.00283776893 | 0.00249636485 | 0.000270717725 | -3.31737829e-06 | 9.58419114e-05 |
| SBC | 0.0234496025 | 0.0213899652 | 0.0200746353 | 0.00337931042 | -0.00512897549 | -0.00288084266 | -0.00435537071 |
| Hermann original | 0.0102249924 | 0.00955096795 | 0.0113040786 | 0.00090834139 | -0.00188813731 | -0.00117346738 | -0.00108315395 |
| Hermann diagnostic | 0.0102237947 | 0.00955096795 | 0.0113050577 | 0.000906732792 | -0.00188697572 | -0.00117346738 | -0.00108122626 |
| White original | -0.0117745006 | -0.0106995464 | -0.0100900171 | -0.00168548999 | 0.00262504839 | 0.001441153 | 0.00216995217 |
| White diagnostic | -0.0117795279 | -0.0106995464 | -0.0100883238 | -0.00169222106 | 0.00263037696 | 0.001441153 | 0.00217837779 |

### Diagnostic minus original

| Family | Channel | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|---|
| White | logit | -3.04013488e-05 | 0 | 1.0066231e-05 | -4.04715547e-05 | 3.23295599e-05 | 0 | 5.05208973e-05 |
| White | probability | -5.02740346e-06 | 0 | 1.69326859e-06 | -6.73110264e-06 | 5.32890368e-06 | 0 | 8.42561303e-06 |
| Hermann | logit | -7.28170079e-06 | 0 | 4.80810797e-06 | -1.2054046e-05 | 7.14858393e-06 | 0 | 1.50799751e-05 |
| Hermann | probability | -1.19755668e-06 | 0 | 9.79254629e-07 | -1.60858038e-06 | 1.16154552e-06 | 0 | 1.92771353e-06 |

[Temporal figure](comparison-figures/temporal-MC_OFF.png) · [Illusion figure](comparison-figures/illusion-MC_OFF.png) · [Diagnostic figure](comparison-figures/diagnostic-MC_OFF.png)

## PC_ON

### Temporal clamp effects

logit: mean absolute off−normal

| Probe | H1-off | direct-BC-off | AC-off | H1 effect Δ vs old | AC effect Δ vs old |
|---|---|---|---|---|---|
| slow_step | 0.000749800539 | 0.385962602 | 0.194196708 | -0.00272869662 | -0.056950084 |
| slow_1Hz | 0.000809642355 | 0.386033787 | 0.192973117 | -0.00326273197 | -0.0581395129 |
| slow_2Hz | 0.000926266762 | 0.386176146 | 0.190475312 | -0.00422061607 | -0.0605624517 |
| rapid_10Hz | 0.000749815968 | 0.38667657 | 0.181914692 | -0.00491466713 | -0.0667630964 |
| rapid_20Hz | 0.000406925119 | 0.386725316 | 0.180123246 | -0.00201595198 | -0.0669920113 |
| transient_50ms | 0.000360894799 | 0.386739118 | 0.179015827 | -0.00131055867 | -0.0670297874 |
| large_field_50ms | 0.000210287261 | 0.0196285759 | 0.0183974399 | -0.00117615196 | -0.0017801933 |

probability: mean absolute off−normal

| Probe | H1-off | direct-BC-off | AC-off | H1 effect Δ vs old | AC effect Δ vs old |
|---|---|---|---|---|---|
| slow_step | 0.000114931415 | 0.0541589844 | 0.0308370206 | -0.00038419817 | -0.010050622 |
| slow_1Hz | 0.000124715916 | 0.0541547487 | 0.0307288237 | -0.000466201873 | -0.0102528428 |
| slow_2Hz | 0.000143786323 | 0.0541458935 | 0.0305065491 | -0.000612552765 | -0.0106628756 |
| rapid_10Hz | 0.000121011266 | 0.054201538 | 0.030140026 | -0.00072763273 | -0.0117208817 |
| rapid_20Hz | 6.5557007e-05 | 0.054247465 | 0.0300482054 | -0.000304607123 | -0.0117070656 |
| transient_50ms | 5.81269374e-05 | 0.0543033667 | 0.0298560708 | -0.000193158585 | -0.0117083555 |
| large_field_50ms | 2.95429952e-05 | 0.00272285277 | 0.00256212314 | -0.000158955101 | -0.000250075258 |

### Illusion signed response signatures

logit: signed mean-on pair difference

| Signature | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|
| Mach dark | -0.00483037812 | -0.00464336692 | -0.013855215 | 0.00902512607 | -0.000345432277 | -0.000600394393 | 0.000104811521 |
| Mach bright | 0.00483037723 | 0.0046433695 | 0.0138552154 | -0.00902512266 | 0.000345429619 | 0.000600391686 | -0.000104806813 |
| SBC | -0.0826428355 | -0.0812264081 | -0.0789467109 | -0.00369877112 | -0.00934276854 | -0.0115108983 | 0.0180585683 |
| Hermann original | -0.0410912453 | -0.0405608262 | -0.0397673044 | -0.00128741221 | -0.00483492783 | -0.00575241322 | 0.00627294915 |
| Hermann diagnostic | -0.0410894089 | -0.0405608262 | -0.0397688728 | -0.00128399442 | -0.00484009708 | -0.00575241322 | 0.00625616412 |
| White original | 0.0414087876 | 0.0406213248 | 0.0395264781 | 0.00188298931 | 0.00436733622 | 0.00575801709 | -0.00917627206 |
| White diagnostic | 0.0414162452 | 0.0406213248 | 0.0395222695 | 0.00189465767 | 0.00434487789 | 0.00575801709 | -0.00923316399 |

probability: signed mean-on pair difference

| Signature | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|
| Mach dark | -0.000730170752 | -0.000706718534 | -0.00228340116 | 0.00101996824 | -7.04240394e-05 | -7.20213629e-05 | 4.96043471e-05 |
| Mach bright | 0.000685831844 | 0.000656571386 | 0.00164593086 | -0.0014944296 | 2.07388245e-05 | 8.06742469e-05 | 2.77458134e-05 |
| SBC | -0.011624262 | -0.0114241463 | -0.0110883189 | -0.000536175844 | -0.00127257003 | -0.00154442182 | 0.00253968993 |
| Hermann original | -0.00608334856 | -0.00600103183 | -0.00477363273 | -0.000227333659 | -0.000600403511 | -0.000783807572 | 0.00110826123 |
| Hermann diagnostic | -0.0060830762 | -0.00600103183 | -0.00477381365 | -0.000226729996 | -0.000601228327 | -0.000783807572 | 0.00110535246 |
| White original | 0.00582416837 | 0.00571258107 | 0.00555128816 | 0.000272961071 | 0.000592681967 | 0.000772206578 | -0.00129050054 |
| White diagnostic | 0.00582523905 | 0.00571258107 | 0.00555066488 | 0.000274651639 | 0.000589590157 | 0.000772206578 | -0.00129850372 |

### Diagnostic minus original

| Family | Channel | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|---|
| White | logit | 7.45808635e-06 | 0 | -4.20852945e-06 | 1.16683818e-05 | -2.24572637e-05 | 0 | -5.68919716e-05 |
| White | probability | 1.07067605e-06 | 0 | -6.23310055e-07 | 1.69056434e-06 | -3.09188067e-06 | 0 | -8.00319286e-06 |
| Hermann | logit | 1.83670613e-06 | 0 | -1.56826442e-06 | 3.41777443e-06 | -5.16882663e-06 | 0 | -1.67851096e-05 |
| Hermann | probability | 2.72415305e-07 | 0 | -1.80911133e-07 | 6.03662613e-07 | -8.24641306e-07 | 0 | -2.90876184e-06 |

[Temporal figure](comparison-figures/temporal-PC_ON.png) · [Illusion figure](comparison-figures/illusion-PC_ON.png) · [Diagnostic figure](comparison-figures/diagnostic-PC_ON.png)

## PC_OFF

### Temporal clamp effects

logit: mean absolute off−normal

| Probe | H1-off | direct-BC-off | AC-off | H1 effect Δ vs old | AC effect Δ vs old |
|---|---|---|---|---|---|
| slow_step | 0.0040357105 | 0.476410806 | 0.138051227 | 0.000888820505 | -0.0889513344 |
| slow_1Hz | 0.00419666017 | 0.476446815 | 0.137825966 | 0.000827185795 | -0.0891424753 |
| slow_2Hz | 0.00451134527 | 0.47651799 | 0.137362793 | 0.000702970399 | -0.0895490833 |
| rapid_10Hz | 0.00400248796 | 0.476789132 | 0.134071328 | 0.000190030623 | -0.0899235308 |
| rapid_20Hz | 0.00194390475 | 0.476889588 | 0.132911194 | 0.000194681423 | -0.0895964764 |
| transient_50ms | 0.00186056129 | 0.476936303 | 0.132158343 | 0.000354961354 | -0.0892969072 |
| large_field_50ms | 0.000941499675 | 0.0235465434 | 0.0128598833 | 6.21065537e-05 | -0.00653899903 |

probability: mean absolute off−normal

| Probe | H1-off | direct-BC-off | AC-off | H1 effect Δ vs old | AC effect Δ vs old |
|---|---|---|---|---|---|
| slow_step | 0.000907581936 | 0.0893070567 | 0.0294583719 | 0.000191154493 | -0.0209508985 |
| slow_1Hz | 0.000947645633 | 0.0893160701 | 0.0294586872 | 0.000176533387 | -0.0210106843 |
| slow_2Hz | 0.00102615663 | 0.0893337335 | 0.0294584893 | 0.000147091288 | -0.0211357744 |
| rapid_10Hz | 0.00093073746 | 0.0895165 | 0.0292267636 | 2.89558338e-05 | -0.0214716713 |
| rapid_20Hz | 0.000446551329 | 0.0895896722 | 0.0290350956 | 3.58989364e-05 | -0.0214053863 |
| transient_50ms | 0.000426490171 | 0.0896362476 | 0.0288665858 | 7.40432633e-05 | -0.0213235426 |
| large_field_50ms | 0.000196148613 | 0.00415218121 | 0.00234007309 | 7.28261057e-06 | -0.00129945646 |

### Illusion signed response signatures

logit: signed mean-on pair difference

| Signature | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|
| Mach dark | -0.00272626581 | -0.00442671227 | 0.00790544553 | -0.0106054091 | -0.00149518048 | -0.00179041828 | 0.0027873143 |
| Mach bright | 0.00272627076 | 0.00442670888 | -0.00790543947 | 0.0106054124 | 0.00149519136 | 0.00179041541 | -0.00278730621 |
| SBC | 0.0595052596 | 0.0471756216 | 0.0424630498 | 0.0170453464 | -0.016131538 | -0.018060511 | -0.0050198602 |
| Hermann original | 0.0278777173 | 0.0235141623 | 0.0219769699 | 0.00593369323 | -0.00843310915 | -0.00901362346 | -0.00173081139 |
| Hermann diagnostic | 0.0278653549 | 0.0235141623 | 0.0219803094 | 0.00591798048 | -0.00843323721 | -0.00901362346 | -0.00172610553 |
| White original | -0.0299914856 | -0.0235904208 | -0.0213076274 | -0.00868467527 | 0.00821286067 | 0.00903258426 | 0.00254042893 |
| White diagnostic | -0.0300359745 | -0.0235904208 | -0.0212982821 | -0.00873851764 | 0.00821499527 | 0.00903258426 | 0.00255618092 |

probability: signed mean-on pair difference

| Signature | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|
| Mach dark | -0.000459434843 | -0.000850500935 | 0.00131586505 | -0.00217456049 | -0.000355898337 | -0.000424791173 | 0.000683761828 |
| Mach bright | 0.000276142742 | 0.000595540925 | -0.00153749736 | 0.00140716744 | 0.000356974826 | 0.000396241627 | -0.000319510975 |
| SBC | 0.0111606411 | 0.00856257195 | 0.00756360608 | 0.0035981995 | -0.00373624149 | -0.00409563968 | -0.0011696171 |
| Hermann original | 0.00449612242 | 0.00367381971 | 0.00422182452 | 0.00104583082 | -0.00181766646 | -0.00188796452 | -0.000286665459 |
| Hermann diagnostic | 0.00449372537 | 0.00367381971 | 0.00422257924 | 0.00104299587 | -0.00181753631 | -0.00188796452 | -0.000285844109 |
| White original | -0.00563108973 | -0.00428186334 | -0.00379784574 | -0.00183348081 | 0.00190237904 | 0.00204892555 | 0.000592364162 |
| White diagnostic | -0.00564047287 | -0.00428186334 | -0.00379586546 | -0.00184484774 | 0.00190306036 | 0.00204892555 | 0.000596034739 |

### Diagnostic minus original

| Family | Channel | normal | H1-off | direct-BC-off | AC-off | normal Δ vs old | H1-off Δ vs old | AC-off Δ vs old |
|---|---|---|---|---|---|---|---|---|
| White | logit | -4.44889056e-05 | 0 | 9.34501509e-06 | -5.38428641e-05 | 2.13384632e-06 | 0 | 1.57515208e-05 |
| White | probability | -9.38301305e-06 | 0 | 1.98036434e-06 | -1.13668544e-05 | 6.81479786e-07 | 0 | 3.67065262e-06 |
| Hermann | logit | -1.23629964e-05 | 0 | 3.339847e-06 | -1.57127777e-05 | -1.28149992e-07 | 0 | 4.70578659e-06 |
| Hermann | probability | -2.39697598e-06 | 0 | 7.54743799e-07 | -2.83494591e-06 | 1.3038516e-07 | 0 | 8.21302356e-07 |

[Temporal figure](comparison-figures/temporal-PC_OFF.png) · [Illusion figure](comparison-figures/illusion-PC_OFF.png) · [Diagnostic figure](comparison-figures/diagnostic-PC_OFF.png)

## Controls and verification

All five paired contextual controls, all 22 cells and all four conditions: exact-zero difference (maximum |mean-on|, peak and integral = 0). Mach matched-uniform controls and boundary extrema are retained separately in the full tables.

22/22 checkpoint/source hashes matched. Both applications: 22/22 exact-zero H1/direct-BC/AC clamps; direct-BC-off preserves BC_broad and AC; AC-off preserves H1 and both BC views; H1-off propagates downstream; inference leaves state unchanged; all outputs finite. Normal reentry is bitwise identical. Previous illusion metric replay matched for 22/22 cells.

## Complete numerical artifacts

- Temporal: [per-cell all metrics](temporal/per-cell.csv), [population/four classes](temporal/group-summary.csv), [per-cell version deltas](temporal/per-cell-vs-overlapping.csv), [group version deltas](temporal/group-vs-overlapping.csv), [each onset/offset event](temporal/per-event-onset-offset.csv), [event version deltas](temporal/per-event-vs-overlapping.csv), [responses](temporal/responses.pt), [stimuli](temporal/inputs.pt).
- Illusions: [per-cell signatures and clamp changes](illusion/per-cell-responses.csv), [population/four classes](illusion/group-responses.csv), [per-cell metric deltas](illusion/per-cell-metric-differences.csv), [group metric deltas](illusion/group-metric-differences.csv), [diagnostic differences](illusion/per-cell-comparisons.csv), [group diagnostic differences](illusion/group-comparisons.csv), [Mach boundary extrema](illusion/mach-boundary-extrema.csv), [Mach group extrema](illusion/mach-boundary-group.csv), [Mach version deltas](illusion/mach-boundary-differences.csv), [Mach group deltas](illusion/mach-boundary-group-differences.csv), [responses](illusion/responses.pt), [stimuli](illusion/inputs.pt).
- Figures: [all population/group/per-cell figures](comparison-figures). Solid lines: shared-BC; dashed: previous overlapping-support.
- [Verification](verification.json), [input/checkpoint manifest](input-manifest.json), [report provenance](report-manifest.json).
