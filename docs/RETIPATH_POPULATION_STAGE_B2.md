# Population RetiPath Stage B.2：downstream center-anchor targeted follow-up

状态：**30 个新 fits、60 个冻结 checkpoint 的 fresh-test 统一评价及必要核对已完成，VERIFIED；停止于 B.2。**

本轮是 **post-hoc targeted follow-up，不是 confirmatory**。只检验 CX 的 residual RGC prediction tradeoff 与指定 downstream fixed center anchors 的关系。

在同一 B.2 fresh test 上，原 CX−A 的 RGC excess CE gap 为 0.0002345754429，移除指定锚点后的 CXF−AF 为 -1.797194529e-05，差中差为 -0.0002525473882 nats/RGC-bin。CXF−AF direct-block delta-logit RMSE 差为 -0.01335361593，负差方向覆盖 5/5 teachers、15/15 paired seeds。

解释只限五个 synthetic teacher instances 与冻结训练合同；不是生物 population 推断。不把三项 prior 的联合移除拆解成某一字段的独立因果贡献，不宣称唯一原因。


**本轮结论：fixed downstream center anchors contributed to the previous residual tradeoff.**

gap在5/5 teachers、15/15 paired seeds中缩小。CXF−AF的excess CE在4/5 teachers、12/15 paired seeds中为负；W02三个seed仍为正，保留该teacher的residual tradeoff。这里没有等价margin，不能把剩余正差写成已消失。结论仅说明这些固定锚点有贡献，不是唯一原因，也不分解γBR、γAR、bias各自的贡献。

prediction改善没有带来所有机制指标的同步改善：相对原CX，CXF的direct-block RMSE由0.01391102397升至0.01473542918，但仍低于AF。其d_I与AC-block RMSE均值也高于CX；所有方向均在后表保留。

参数结果分别为：相对CX，CXF的γBR teacher RMSE从0.2967074585变为0.3122827513；γAR从0.1113450006变为0.3804242401；RGC bias从0.1796074414降至0.020265441。γBR/γAR平均恢复未改善，bias恢复改善；不合成coupling score。a_H prior保持不变，其恢复变化另列。

Anchor-free条件的seed分散增大：CX→CXF的γBR seed-pair RMSE为2.465736796e-5→0.1043891966，γAR为9.442496904e-6→0.08146344428；direct-block effect的seed-pair RMSE为8.475080769e-5→0.001565697888。原样报告该分散度，不追加正则或延长训练。


## 1. 唯一改动及冻结合同

只对 AF/CXF 的训练目标移除 gamma_BR、gamma_AR、RGC bias 的 fixed constructor-center quadratic penalty：

`P_free = sum(field.penalty() for all fields except gamma_BR, gamma_AR, bias)`。

被排除的三个字段均无 unit contrasts；其原惩罚为 `0.5*sum(((raw_center-initial_center)/0.3)^2)`。initial_center 是固定 constructor anchor，不是扰动后的 paired student initialization。原 physical anchors 为 γBR=1.5、γAR=0.375、bias=−2.4；边界分别为 (0.2,8)、(0,8)、(−4,−0.5)，保持不变。三个字段仍可训练，没有冻结或重初始化。

a_H contrast SD=0.1、H1/BC partial pooling、AC family sharing 以及其他所有 hierarchy 项完全保留。Population v0.1、LegacyPReLU、Q、routing、conductance、interventions 的源码与 forward 不变。全部359个 raw parameters仍进入同一Adam。原A/C/E/CW/CX checkpoints与旧结果均保持原字节。

AF：`(L_R+L_R1+L_R2)/3 + P_free`；CXF：`(L_R+L_R1+L_R2)/3 + 0.4 L_H + 0.4 L_BC + P_free`。H/BC是正项。RGC mean Bernoulli BCE；H/BC原sigma0.03 Gaussian mean loss；BC state/output一并平均。

复用Stage B同5 teachers、3 paired initial states、所有train pools和逐步schedules。H1为center/cardinal外侧5/25 nodes，BC为相同位置ON5+OFF5，只监督h_H、s_B、delta_r_B；其余latent为evaluator-only。未加入AC或E/I observations。

每fit固定400 updates、batch4、Adam lr0.01/betas(0.9,0.999)/eps1e-8/weight_decay0、global clip1、CPU float32、每process1线程。每步prior只加一次。无early stop、selection、延长、重跑或结果后调参，只用step400。


| 条件 | checkpoint来源 | 每步streams | RGC batches/fit | 全部batches/fit | 全部sequence exposures |
| --- | --- | --- | --- | --- | --- |
| A | Stage B冻结 | base_R+base_R_1+base_R_2 | 1200 | 1200 | 4800 |
| CX | B.1冻结 | 3 base_R + H + BC | 1200 | 2000 | 8000 |
| AF | 本轮15 fits | 与A完全相同 | 1200 | 1200 | 4800 |
| CXF | 本轮15 fits | 与CX完全相同 | 1200 | 2000 | 8000 |



四组RGC exposure、来源、总loss系数均匹配；CX/CXF额外消费H/BC，不能称总compute或information-matched。本轮新训练共12,000 updates、48,000 microbatches、192,000 sequence exposures。

## 2. Fresh-test合同与指标

训练前生成并锁定80条新序列：每teacher16条，held-out scales 0.225/0.45 deg各8条；32×32 grid、完整2×2deg、300bins@150Hz、60bin warmup。沿用刺激分布与评价定义，namespace=`PopulationStageB2Fresh20260920:Wxx:v1`；waveform/event numeric seeds、records全部保存，与Stage B/B.1已消费test及旧训练流无重复，不按响应筛选或重抽。没有读取旧test payload来评分。

trainer有tensor白名单并拒绝evaluator_only访问。全部30新final与原A/CX共30个final锁定后才登记TEST_CONSUMED，统一评价60个checkpoint。此处trainer指训练worker；协调程序对旧teacher checkpoint只做SHA256完整性读取，不反序列化或向worker提供truth。参数truth只在评价阶段解码；fresh-test payload在生成后保持封存至统一评价，新test不参与优化或选择。

误差先逐序列算，scale内等权，再两scales等权；teacher和seed等权。RGC excess CE单位nats/RGC-bin，以teacher概率计算。intervention delta-logit=blocked−normal，沿用相同teacher-normal events、baseline reset、strictly-past history；是计算通路干预。不新增RF、显著性检验、等价margin、成功阈值或生物population推断。负差表示误差更低。

## 3. Primary：RGC excess CE


| A | CX | AF | CXF |
| --- | --- | --- | --- |
| 0.001032068033 | 0.001266643476 | 8.83469579e-05 | 7.037501261e-05 |


| 比较 | 定位 | 均值差 | 负差teacher | 负差paired seeds |
| --- | --- | --- | --- | --- |
| CXF-AF | primary | -1.797194529e-05 | 4/5 | 12/15 |
| CXF-CX | primary | -0.001196268463 | 5/5 | 15/15 |
| AF-A | primary | -0.0009437210753 | 5/5 | 15/15 |
| CX-A | 冻结的解释参照 | 0.0002345754429 | 0/5 | 0/15 |



每teacher三seed均值：


| teacher | A | CX | AF | CXF | CXF-AF | CXF-CX | AF-A | CX-A | gap change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W01 | 0.0006564966774 | 0.0008709431358 | 4.903471875e-05 | 3.115779115e-05 | -1.78769276e-05 | -0.0008397853446 | -0.0006074619587 | 0.0002144464583 | -0.0002323233859 |
| W02 | 0.002092505926 | 0.002586435779 | 0.0001013954076 | 0.000109131228 | 7.735820425e-06 | -0.002477304551 | -0.001991110519 | 0.0004939298522 | -0.0004861940318 |
| W03 | 0.0009706281161 | 0.001251024713 | 8.564802402e-05 | 6.304215971e-05 | -2.260586431e-05 | -0.001187982553 | -0.0008849800921 | 0.0002803965969 | -0.0003030024612 |
| W04 | 0.0006207661635 | 0.0007658509417 | 0.0001470204253 | 0.0001233316386 | -2.368878676e-05 | -0.0006425193032 | -0.0004737457382 | 0.0001450847783 | -0.000168773565 |
| W05 | 0.0008199432828 | 0.0008589628113 | 5.863621383e-05 | 2.521224565e-05 | -3.342396819e-05 | -0.0008337505657 | -0.0007613070689 | 3.901952856e-05 | -7.244349675e-05 |



全部15 paired initializations：


| teacher/seed | A | CX | AF | CXF | CXF-AF | CXF-CX | AF-A | CX-A | gap change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W01/4101 | 0.0006561506033 | 0.0008709150252 | 5.021678091e-05 | 3.124467066e-05 | -1.897211025e-05 | -0.0008396703545 | -0.0006059338224 | 0.0002147644219 | -0.0002337365321 |
| W01/4102 | 0.0006603391242 | 0.0008712538652 | 4.85265207e-05 | 3.096610309e-05 | -1.756041761e-05 | -0.0008402877621 | -0.0006118126035 | 0.000210914741 | -0.0002284751586 |
| W01/4103 | 0.0006530003047 | 0.0008706605169 | 4.836085465e-05 | 3.12625997e-05 | -1.709825495e-05 | -0.0008393979172 | -0.0006046394501 | 0.0002176602122 | -0.0002347584671 |
| W02/4101 | 0.002085063763 | 0.002580640119 | 0.0001006023797 | 0.0001088294823 | 8.227102603e-06 | -0.002471810637 | -0.001984461384 | 0.0004955763556 | -0.000487349253 |
| W02/4102 | 0.002088360231 | 0.002585848767 | 9.934788338e-05 | 0.000109798635 | 1.045075162e-05 | -0.002476050132 | -0.001989012348 | 0.0004974885361 | -0.0004870377845 |
| W02/4103 | 0.002104093784 | 0.002592818449 | 0.0001042359597 | 0.0001087655667 | 4.529607052e-06 | -0.002484052882 | -0.001999857825 | 0.000488724665 | -0.0004841950579 |
| W03/4101 | 0.0009734299799 | 0.001257711104 | 8.464472712e-05 | 6.270925506e-05 | -2.193547206e-05 | -0.001195001849 | -0.0008887852528 | 0.0002842811242 | -0.0003062165963 |
| W03/4102 | 0.0009717261904 | 0.001250111916 | 8.575127083e-05 | 6.359473978e-05 | -2.215653106e-05 | -0.001186517177 | -0.0008859749195 | 0.0002783857259 | -0.000300542257 |
| W03/4103 | 0.000966728178 | 0.001245251119 | 8.654807411e-05 | 6.282248431e-05 | -2.37255898e-05 | -0.001182428634 | -0.0008801801039 | 0.0002785229406 | -0.0003022485304 |
| W04/4101 | 0.0006210988518 | 0.0007675466874 | 0.0001473880709 | 0.0001236207838 | -2.376728706e-05 | -0.0006439259035 | -0.0004737107809 | 0.0001464478355 | -0.0001702151226 |
| W04/4102 | 0.0006245587978 | 0.0007664544506 | 0.0001477249533 | 0.0001237090832 | -2.401587005e-05 | -0.0006427453674 | -0.0004768338445 | 0.0001418956528 | -0.0001659115229 |
| W04/4103 | 0.0006166408408 | 0.0007635516872 | 0.0001459482518 | 0.0001226650486 | -2.328320318e-05 | -0.0006408866386 | -0.000470692589 | 0.0001469108464 | -0.0001701940496 |
| W05/4101 | 0.000818136627 | 0.0008581499666 | 5.814198768e-05 | 2.520102884e-05 | -3.294095884e-05 | -0.0008329489378 | -0.0007599946393 | 4.001333964e-05 | -7.295429848e-05 |
| W05/4102 | 0.0008207794408 | 0.0008593405153 | 5.800896959e-05 | 2.527051306e-05 | -3.273845653e-05 | -0.0008340700022 | -0.0007627704712 | 3.856107449e-05 | -7.129953102e-05 |
| W05/4103 | 0.0008209137805 | 0.0008593979521 | 5.975768423e-05 | 2.516519504e-05 | -3.459248919e-05 | -0.000834232757 | -0.0007611560963 | 3.848417155e-05 | -7.307666075e-05 |



## 4. Secondary：接口分别报告

state、output、drive、intervention与prediction分别统计，不合成mechanism score。全部来自同一新fresh test。


| RMSE | A | CX | AF | CXF | CXF-AF | CXF-CX | AF-A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| observed_h_H_rmse | 0.02142386401 | 0.003791645608 | 0.01014300505 | 0.003791384401 | -0.006351620646 | -2.612063849e-07 | -0.01128085897 |
| unobserved_h_H_rmse | 0.02213621834 | 0.005231227443 | 0.009879077952 | 0.005234340153 | -0.004644737799 | 3.112710196e-06 | -0.01225714038 |
| observed_s_B_rmse | 0.02064848929 | 0.006833320523 | 0.01653946286 | 0.00683331519 | -0.009706147665 | -5.333339167e-09 | -0.004109026439 |
| observed_delta_r_B_rmse | 0.02049051275 | 0.006932906045 | 0.01548028071 | 0.006934919165 | -0.008545361544 | 2.013120258e-06 | -0.005010232037 |
| unobserved_s_B_rmse | 0.01887359002 | 0.008387446442 | 0.01541562098 | 0.008390627863 | -0.00702499312 | 3.181421101e-06 | -0.003457969038 |
| unobserved_delta_r_B_rmse | 0.02043357855 | 0.008584656415 | 0.01500301602 | 0.008580593007 | -0.006422423011 | -4.063408111e-06 | -0.005430562537 |
| d_E_rmse | 0.03156142877 | 0.01429138522 | 0.02627756776 | 0.0154716039 | -0.01080596386 | 0.001180218679 | -0.005283861012 |
| d_I_rmse | 0.009790430004 | 0.009991875894 | 0.02035984926 | 0.02553156073 | 0.005171711473 | 0.01553968484 | 0.01056941926 |
| direct_delta_logit_rmse | 0.03587813194 | 0.01391102397 | 0.02808904511 | 0.01473542918 | -0.01335361593 | 0.0008244052073 | -0.007789086829 |
| AC_delta_logit_rmse | 0.008146421999 | 0.00836283775 | 0.01742921056 | 0.02185406351 | 0.004424852946 | 0.01349122576 | 0.009282788564 |
| probability_rmse | 0.01218136722 | 0.01345271163 | 0.00359512319 | 0.003138098181 | -0.000457025008 | -0.01031461344 | -0.008586244035 |


| RMSE | 比较 | 负差teacher | 负差paired seeds |
| --- | --- | --- | --- |
| observed_h_H_rmse | CXF-AF | 5/5 | 15/15 |
| observed_h_H_rmse | CXF-CX | 3/5 | 9/15 |
| observed_h_H_rmse | AF-A | 5/5 | 15/15 |
| unobserved_h_H_rmse | CXF-AF | 5/5 | 15/15 |
| unobserved_h_H_rmse | CXF-CX | 2/5 | 7/15 |
| unobserved_h_H_rmse | AF-A | 5/5 | 15/15 |
| observed_s_B_rmse | CXF-AF | 5/5 | 15/15 |
| observed_s_B_rmse | CXF-CX | 3/5 | 8/15 |
| observed_s_B_rmse | AF-A | 5/5 | 14/15 |
| observed_delta_r_B_rmse | CXF-AF | 5/5 | 15/15 |
| observed_delta_r_B_rmse | CXF-CX | 2/5 | 5/15 |
| observed_delta_r_B_rmse | AF-A | 5/5 | 15/15 |
| unobserved_s_B_rmse | CXF-AF | 5/5 | 15/15 |
| unobserved_s_B_rmse | CXF-CX | 1/5 | 2/15 |
| unobserved_s_B_rmse | AF-A | 5/5 | 15/15 |
| unobserved_delta_r_B_rmse | CXF-AF | 5/5 | 15/15 |
| unobserved_delta_r_B_rmse | CXF-CX | 4/5 | 11/15 |
| unobserved_delta_r_B_rmse | AF-A | 5/5 | 15/15 |
| d_E_rmse | CXF-AF | 5/5 | 15/15 |
| d_E_rmse | CXF-CX | 1/5 | 3/15 |
| d_E_rmse | AF-A | 4/5 | 12/15 |
| d_I_rmse | CXF-AF | 1/5 | 3/15 |
| d_I_rmse | CXF-CX | 0/5 | 1/15 |
| d_I_rmse | AF-A | 1/5 | 3/15 |
| direct_delta_logit_rmse | CXF-AF | 5/5 | 15/15 |
| direct_delta_logit_rmse | CXF-CX | 1/5 | 3/15 |
| direct_delta_logit_rmse | AF-A | 4/5 | 12/15 |
| AC_delta_logit_rmse | CXF-AF | 1/5 | 3/15 |
| AC_delta_logit_rmse | CXF-CX | 0/5 | 1/15 |
| AC_delta_logit_rmse | AF-A | 1/5 | 3/15 |
| probability_rmse | CXF-AF | 4/5 | 12/15 |
| probability_rmse | CXF-CX | 5/5 | 15/15 |
| probability_rmse | AF-A | 5/5 | 15/15 |



Direct advantage与prediction gap的逐teacher事实：


| teacher | CXF−AF excess CE | CXF−AF direct-block RMSE | CXF−AF d_E RMSE | CXF−AF AC-block RMSE | CXF−AF d_I RMSE |
| --- | --- | --- | --- | --- | --- |
| W01 | -1.78769276e-05 | -0.01308643838 | -0.01152034504 | 0.002824208536 | 0.003195224965 |
| W02 | 7.735820425e-06 | -0.009817659872 | -0.008303720821 | 0.009039848181 | 0.01055154193 |
| W03 | -2.260586431e-05 | -0.01085097573 | -0.007934330496 | 0.003362253637 | 0.003786333645 |
| W04 | -2.368878676e-05 | -0.02124549566 | -0.01597357897 | 0.02078690837 | 0.02443067795 |
| W05 | -3.342396819e-05 | -0.01176751003 | -0.01029784398 | -0.01388895399 | -0.01610522112 |



全部secondary逐teacher/seed误差及差值均在per_fit.csv、paired_differences.csv、teacher_summary.csv，无筛选。

## 5. 参数恢复：各字段独立

由原始checkpoint的raw center/contrast在float64中按冻结有界sigmoid解码；不运行额外回路forward。teacher RMSE先在字段的physical unit coordinates内计算，再对seed/teacher等权。γBR为4坐标，γAR为16坐标，bias为2坐标，a_H为25坐标。这些是模型归一化参数，不能当生理单位或合成coupling score。


| field teacher RMSE | A | CX | AF | CXF | CXF-AF | CXF-CX | AF-A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gamma_BR | 0.2967117227 | 0.2967074585 | 0.3672084032 | 0.3122827513 | -0.05492565192 | 0.01557529274 | 0.07049668046 |
| gamma_AR | 0.1113461045 | 0.1113450006 | 0.3369845421 | 0.3804242401 | 0.043439698 | 0.2690792395 | 0.2256384375 |
| bias | 0.1796663392 | 0.1796074414 | 0.02439973051 | 0.020265441 | -0.004134289507 | -0.1593420004 | -0.1552666087 |
| a_H | 0.09937671174 | 0.03501655091 | 0.05490016094 | 0.03439879626 | -0.02050136468 | -0.0006177546467 | -0.0444765508 |


| field | 比较 | 负差teacher | 负差paired seeds |
| --- | --- | --- | --- |
| gamma_BR | CXF-AF | 2/5 | 8/15 |
| gamma_BR | CXF-CX | 1/5 | 4/15 |
| gamma_BR | AF-A | 1/5 | 3/15 |
| gamma_AR | CXF-AF | 2/5 | 5/15 |
| gamma_AR | CXF-CX | 0/5 | 0/15 |
| gamma_AR | AF-A | 0/5 | 0/15 |
| bias | CXF-AF | 3/5 | 11/15 |
| bias | CXF-CX | 5/5 | 15/15 |
| bias | AF-A | 5/5 | 15/15 |
| a_H | CXF-AF | 5/5 | 13/15 |
| a_H | CXF-CX | 3/5 | 8/15 |
| a_H | AF-A | 5/5 | 15/15 |



逐teacher参数恢复（每格三seed均值）：


| teacher/field | A | CX | AF | CXF |
| --- | --- | --- | --- | --- |
| W01/gamma_BR | 0.2092553358 | 0.2092459955 | 0.4168509484 | 0.2188072039 |
| W01/gamma_AR | 0.141469207 | 0.1414722958 | 0.2238389715 | 0.2118468203 |
| W01/bias | 0.1548053512 | 0.1547153755 | 0.01167772063 | 0.01200244737 |
| W01/a_H | 0.1533399048 | 0.03651987965 | 0.05086765433 | 0.03606457172 |
| W02/gamma_BR | 0.2283380366 | 0.2283314266 | 0.2805437982 | 0.2979789638 |
| W02/gamma_AR | 0.1145964026 | 0.1145956808 | 0.5306750254 | 0.6067881074 |
| W02/bias | 0.2585642458 | 0.2584519199 | 0.01377829981 | 0.008481737602 |
| W02/a_H | 0.0555323139 | 0.01524425079 | 0.03546194116 | 0.01536777683 |
| W03/gamma_BR | 0.292001745 | 0.291993686 | 0.5391390725 | 0.4083233436 |
| W03/gamma_AR | 0.1295726639 | 0.1295695626 | 0.186680807 | 0.2103590106 |
| W03/bias | 0.1824828621 | 0.1824321868 | 0.040795853 | 0.04119090461 |
| W03/a_H | 0.1594518121 | 0.07382654745 | 0.08912371161 | 0.07301647836 |
| W04/gamma_BR | 0.5268914624 | 0.526894948 | 0.257639571 | 0.2609472341 |
| W04/gamma_AR | 0.09675746825 | 0.09675249925 | 0.245458218 | 0.507680587 |
| W04/bias | 0.1332156667 | 0.1331923773 | 0.037944671 | 0.03620154148 |
| W04/a_H | 0.06295291662 | 0.03214123518 | 0.04790093645 | 0.02986564181 |
| W05/gamma_BR | 0.2270720338 | 0.2270712365 | 0.3418686257 | 0.3753570108 |
| W05/gamma_AR | 0.07433478093 | 0.07433496444 | 0.4982696884 | 0.365446675 |
| W05/bias | 0.1692635702 | 0.1692453476 | 0.01780210809 | 0.003450573949 |
| W05/a_H | 0.06560661133 | 0.01735084148 | 0.05114656117 | 0.0176795126 |



Initial→final movement和final距旧anchor分别报告。raw movement含该字段所有trainable center/contrast坐标；raw center distance只含center。physical distance以constructor reference为基准；**a_H的constructor reference没有fixed center penalty**，其contrast partial pooling从未移除。


| field/统计 | A | CX | AF | CXF |
| --- | --- | --- | --- | --- |
| gamma_BR/physical_movement_rms | 0.09466309509 | 0.0946628354 | 0.3309289558 | 0.3017375844 |
| gamma_BR/raw_parameter_movement_rms | 0.0879564012 | 0.08795611784 | 0.2988515821 | 0.2920215053 |
| gamma_BR/raw_center_distance_from_constructor | 1.984262602e-05 | 1.515867293e-05 | 0.2949399647 | 0.290640476 |
| gamma_BR/physical_distance_from_constructor | 2.149611656e-05 | 1.6421986e-05 | 0.3300306549 | 0.3026365773 |
| gamma_AR/physical_movement_rms | 0.03776285726 | 0.03776303231 | 0.2971167022 | 0.3527104967 |
| gamma_AR/raw_parameter_movement_rms | 0.1030947505 | 0.1030952584 | 0.5806571375 | 0.6881878613 |
| gamma_AR/raw_center_distance_from_constructor | 3.475570614e-05 | 1.435212212e-05 | 0.6004710232 | 0.7040258411 |
| gamma_AR/physical_distance_from_constructor | 1.242590772e-05 | 5.130111244e-06 | 0.3042919214 | 0.3586291426 |
| bias/physical_movement_rms | 0.05960998308 | 0.05960203946 | 0.180556527 | 0.182676435 |
| bias/raw_parameter_movement_rms | 0.06859587578 | 0.06858676838 | 0.2078299827 | 0.2101980425 |
| bias/raw_center_distance_from_constructor | 0.0004534358704 | 0.0005173041122 | 0.1982645559 | 0.1991910849 |
| bias/physical_distance_from_constructor | 0.000393850401 | 0.0004493278856 | 0.1723170328 | 0.1731825917 |
| a_H/physical_movement_rms | 0.08589442013 | 0.05754617463 | 0.05324451354 | 0.05583729886 |
| a_H/raw_parameter_movement_rms | 0.1094786742 | 0.06416369546 | 0.05582119265 | 0.0621232214 |
| a_H/raw_center_distance_from_constructor | 0.5820183953 | 0.3499559383 | 0.2485022459 | 0.3395886093 |
| a_H/physical_distance_from_constructor | 0.0921178345 | 0.06282254567 | 0.04766747883 | 0.06108381932 |



parameter_coordinates.csv保存每个teacher/condition/seed/coordinate的teacher、initial、final、constructor reference；parameter_raw_centers.csv保存raw center。counterfactual_old_priors.csv仅说明在新final点若重新施加旧prior会有多少惩罚，未加入训练或重评目标。

## 6. Cross-seed ambiguity

同teacher的3对seed间RMSE，低分散不等于真值恢复或唯一可辨识。保持任何增加，不追加正则。


| trajectory seed-pair RMSE | A | CX | AF | CXF |
| --- | --- | --- | --- | --- |
| observed_h_H_rmse | 0.0008652449986 | 4.416001052e-05 | 0.001809595444 | 4.102936639e-05 |
| unobserved_h_H_rmse | 0.0008362133648 | 4.224239645e-05 | 0.001749861407 | 3.925196216e-05 |
| observed_s_B_rmse | 0.001064571752 | 6.853654294e-05 | 0.00186928957 | 7.844227215e-05 |
| observed_delta_r_B_rmse | 0.0009559168969 | 5.494746191e-05 | 0.001817577892 | 6.323989183e-05 |
| unobserved_s_B_rmse | 0.001006874888 | 6.572269077e-05 | 0.001776239015 | 7.472105857e-05 |
| unobserved_delta_r_B_rmse | 0.0009006216414 | 5.26753468e-05 | 0.001718190477 | 6.022805736e-05 |
| d_E_rmse | 0.00146221996 | 7.748550249e-05 | 0.006013565059 | 0.005016442058 |
| d_I_rmse | 0.0003435746253 | 0.0002532537433 | 0.004143995184 | 0.00369374882 |
| direct_delta_logit_rmse | 0.001683642483 | 8.475080769e-05 | 0.003371065745 | 0.001565697888 |
| AC_delta_logit_rmse | 0.0002991235217 | 0.0002188234868 | 0.002899303392 | 0.002134603773 |
| probability_rmse | 0.0001224801949 | 3.599804234e-05 | 0.0001624869145 | 9.402764975e-05 |


| parameter seed-pair RMSE | A | CX | AF | CXF |
| --- | --- | --- | --- | --- |
| gamma_BR | 3.015511275e-05 | 2.465736796e-05 | 0.1179792322 | 0.1043891966 |
| gamma_AR | 2.247978579e-05 | 9.442496904e-06 | 0.07928997458 | 0.08146344428 |
| bias | 0.0003200608168 | 0.0003480752049 | 0.002161663168 | 0.0009966583054 |
| a_H | 0.01659653957 | 0.002378892115 | 0.02131786376 | 0.003095313973 |



两类ambiguity的逐teacher、逐seed-pair数据和配对比较分别保存在ambiguity_*与parameter_ambiguity_* CSV/JSON。

## 7. 训练诊断与完整性

诊断仅记录，不改变optimizer。base_R BCE来自实际更新前batch；窗口内取平均，横向逐update配对，窗口间batch变化。


| update窗口 | A | CX | AF | CXF |
| --- | --- | --- | --- | --- |
| 1-50 | 0.2906734913 | 0.2906991092 | 0.2898527984 | 0.2898598457 |
| 51-100 | 0.2909768503 | 0.2910672812 | 0.2898035394 | 0.2898207848 |
| 101-200 | 0.2905482744 | 0.2907173234 | 0.2895013405 | 0.2895293886 |
| 201-300 | 0.2903115605 | 0.2905464874 | 0.2893117725 | 0.2893474407 |
| 301-400 | 0.2906747922 | 0.290937014 | 0.2897085347 | 0.2897501056 |
| 351-400 | 0.2904521001 | 0.2907195038 | 0.289495642 | 0.2895395018 |


| 条件 | clip触发/6000 | 最后触发step | 最大preclip norm |
| --- | --- | --- | --- |
| A | 278/6000 | 23 | 7.035794258 |
| CX | 296/6000 | 24 | 7.577471256 |
| AF | 238/6000 | 24 | 4.80009985 |
| CXF | 240/6000 | 23 | 4.973976135 |



每update的raw/weighted gradient norm、state/output/coupling组norm、共同autograd-supported坐标cosine与global clip均完整保存。零均值cosine不能排除局部冲突，本轮不自动用gradient conflict解释结果。

必要验证VERIFIED：30fits均400步、359参数保持trainable/optimizer-listed；新旧配对首步dataset raw gradient norms最大差0；预检中除22个指定center坐标外prior梯度差0，a_H contract不变。所有新训练worker evaluator-only读取0。80条fresh序列先锁定，60个checkpoint锁定后才消费。独立NumPy复算已保存raw arrays的最大指标差=1.387778781e-17，ambiguity差=1.734723476e-18；参数坐标复算差=1.110223025e-16。数值容差只用于实现核对，不构成科学成功阈值。验证没有额外student forward或optimizer update。

## 8. 工件、源码入口与停止边界

工件目录：[retipath_population_stage_b2_20260920](D:/PythonProject/retina_rf_SNN/output/synthetic/retipath_population_stage_b2_20260920)。

- protocol.json / PROTOCOL.md、SOURCE_LOCK.json、PARENT_INPUT_LOCK.json、B1_INPUT_LOCK.json：合同和旧工件哈希。
- FRESH_TEST_LOCK.json、worlds/*/FRESH_TEST_MANIFEST.json、evaluator_only/fresh_test.pt：新测试随机流与truth。
- worlds/*/fits/{AF,CXF}_*/：30个final.pt及全部训练轨迹/梯度/完成记录。
- CHECKPOINT_LOCK.json、TEST_CONSUMED.json、worlds/*/evaluation/*_raw.pt：60个checkpoint统一评价证据。
- results.json、per_sequence/per_fit/paired_differences/teacher_summary/descriptive_summary.csv：主次指标完整数据。
- parameter_results.json、parameter_*.csv、prediction_gap_change.csv：参数恢复、movement、anchor距离、分散度及gap变化。
- ambiguity_*.csv、diagnostics.json、gradient_*_summary.csv、clip_summary.csv：分散度及诊断。
- PREFLIGHT.json、VERIFICATION.json、verified_exposures.csv、initial_gradient_pairing.csv、FILE_MANIFEST.json：合同验收。

新执行入口：[stage_b2.py](D:/PythonProject/retina_rf_SNN/experiments/retipath_population_v0_1/stage_b2.py)，参数/数值核对与报告入口：[stage_b2_delivery.py](D:/PythonProject/retina_rf_SNN/experiments/retipath_population_v0_1/stage_b2_delivery.py)。训练worker源码与B.1仅prior调用不同；原circuit.py、stage_b.py、stage_b1.py哈希保持。

本轮未执行Git，未覆盖旧checkpoint/结果，未更改a_H prior、H/BC权重、训练预算或架构。未新增E/I观测、RF、机制、Softplus、BC coupling、AC→BC或outer nonlinearity。完成后停止，不追加条件/正则/重训，不进入Stage C，不作下一阶段研究决策。
