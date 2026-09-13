# Canonical V1 fixed-LN-center spatial-alignment experiment

预注册日期：2026-09-05，Asia/Tokyo。此文件在任何 aligned fit / aligned validation 输出之前写入；以训练前 manifest 的 SHA256 锁定，之后不根据结果改写。

## 对象与唯一改变因素

Reference：`output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/` 的22个最终N=1 Canonical V1；causal=`h1-shared-bc-direct-broad-ac`，spatial=`bc-central-disk_ac-overlapping-full-disk`。中心来源仅为 `output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells/<cell>/ln-trained.pt` 的 final fresh full-train refit。上一审计为 `output/audits/macaque_spatial_alignment_20260905/`。

实验条件仅把固定 `cell_positions_degs=[[0,0]]` 替换成冻结LN final-refit center经源码及数值单元测试确定的degree坐标。无trainable center、clip、grid/sign/scale search、手工offset或subset selection。半径、Gaussian sigma、H1图和空间basis family不变；production构造器自然生成平移后的support/basis。网络层次/因果拓扑相同，采样grid上的support成员/数量允许随固定中心自然变化，必须记录，不能误称support tensor逐元素不变。每cell可训练scalar仍为33。

## H1与null

H1：固定零中心的空间失配可能限制预测。定义 Δ_i=NLL_aligned,i−NLL_zero,i，I_i=−Δ_i。预期22-cell等权mean Δ<0；预先冻结offset较大的cells改善更多；offset与I的Pearson/Spearman方向为正。若存在pathway compensation，RF/pathway量可能改变，但这是secondary，不作为prediction成功条件。

Null/failure：mean基本无改善、改善不随offset增大、大/小offset无区别或系统性恶化均降低该假设可信度。负结果正常接受，禁止为挽救结果改变实验条件。

## 执行顺序及STOP gates

1. 记录branch、HEAD、完整git status；冻结reference lineage、前次alignment audit、LN final refit、实际production源码和raw movie/spike hashes。
2. 不训练坐标测试：LN centers `(0,0),(1,0),(-1,0),(0,1),(0,-1)`；核对LN Gaussian peak/centroid、Canonical Gaussian及BC/AC support centroid在MC/PC两类中的physical方向。依据实际 `grid_xy`、`_cone_positions`、row/column、原点和pitch；不以旧CSV或validation选择映射。任何歧义则全局STOP。
3. 逐cell固定geometry preflight：BC/AC非空、BC严格包含于AC且有外延、完整距离disk规则无非法hole、finite、center在当前grid域内；记录domain边界截断及support数。中心不clip。无法合法构造的cell STOP并明确记录；其余cells可依原协议继续，但缺失cells时不声称完成22-cell总体判断。
4. 在任何新训练前，用当前production loader从原movie/spikes重建数据，strict-load并重新评价全部22个zero final checkpoints。target/mask/IDs/trial order、logits要求torch.equal，NLL要求与冻结reference精确一致。任何reference replay失败则全局STOP，不训练aligned。
5. 核对zero/aligned factory的trainable names、初始trainable tensors和optimizer membership完全相同，trainable scalar=33。与reference `model-raw.pt`核对初始trainable state。仅与center直接相关的geometry buffers可不同，H1 graph必须相同。
6. 所有全局gate通过后，对22个合法cell各运行一次alignment condition，使用原production `select_and_refit_r4`；每cell一次inner selection和一次fresh full-train refit，分别计数，不混淆条件数量与优化过程数量。
7. 冻结每cell final后才评价原validation；完成后统一汇总prediction、RF、pathway和structural interventions。全程不运行illusion。

## 冻结训练合同

原raw sources；16 train / 4 validation temporal segments及原trial展开；native150Hz，17×17 L+M Weber drive；Bernoulli spike-event targets；原warmup/score mask、strictly-past history；原training-only 80/20 inner-dev及60-bin guard。Adam，LR=0.03，batch=4，max inner steps=1000，patience=200，min_delta=1e−7；每cell沿用reference seed及minibatch RNG规则；fresh optimizer和fresh full-train refit；refit步数由aligned inner-dev按既有规则选出，不强制等于zero条件的selected step。原validation不参与任何选择。CPU torch与threads=2沿reference运行。不得更改production实现、超参数、τ/delay bounds或旧结果。

## Primary analysis（development evidence）

原validation已长期用于development，本实验不是untouched test。全22 cells等权；保存各cell零中心、aligned、LN NLL，Δ、I、原Canonical−LN gap和冻结offset。报告mean/median Δ、aligned/zero/tie counts、4组descriptive means，不做group significance tests。

Paired-cell bootstrap：固定NumPy RNG seed=20260905，100,000次，每次有放回抽22个cell，以paired Δ的mean作为统计量；报告2.5%/97.5% percentile CI。不是bin/trial bootstrap，也不是生理层面的因果置信区间。

Offset诊断：全部22 cells上I与冻结radial offset的Pearson、平均秩Spearman。预先按offset排序为较小/较大各11 cells（ties用cell_id排序）；报告两半的mean I。额外只报告预先已知高影响cell `68#10` 的leave-one-out mean Δ，主结果始终22 cells，不据此选择训练或最终subset。

原gap>0时定义R_i=I_i/(NLL_zero,i−NLL_LN,i)，不截断R>1或R<0；原gap<=0时R留空。报告符合定义cells的mean/median R和sum(I)/sum(original gap)，明确二者权重不同；亦报告总体mean gap前后变化。均为描述量，不叫causal variance explained。

## Secondary机制分析

复用当前shared-BC lineage实际analysis definitions：`learned_parameter_values`的H1 amplitude、BC/AC gains、BC τ/delay、AC τ/delay、AC local/transient mixture、history gate；另报告H1 τ/delay仅作同一已定义参数集的背景。

RF：复用当前 `clean_sampled_reporting.rf_bundle`及其有序分解，保持16-lag和同一validation输入/history/聚合方式，保存global、H1 ordered、direct-BC ordered、AC ordered tensors；不改lag定义。比较zero/aligned的norm、relative L2 difference、cosine及energy centroid，所有定义在结果产生前由分析代码固定。缺乏真实RF时不解释为physiology更准确。

Intervention：相同validation输入/history计算H1-off、direct-BC-off、AC-off的mean absolute Δlogit。当前lineage的既有aggregation包含全部validation sequence bins（包括warmup），因此保留该口径作直接比较；另明确标注score-mask-only结果以对应prediction bins，两者不得混列。保留同一有序RF分解与结构clamp语义。验证状态不变和finite，不运行SBC/Mach/White/Hermann。

## 结果判定与禁止自动延伸

为使“清楚总体改善”可审查：prediction支持要求mean Δ<0且paired bootstrap95%上限<0；offset方向支持要求大offset半组mean I>小offset半组，且Pearson/Spearman均>0；报告median与去掉68#10后的mean判断集中性。

- SUPPORTED：总体改善与offset方向均支持，且去掉68#10后mean仍改善。解释严格限定为training-derived proxy改善当前development evaluation。
- PARTIAL / MIXED：有改善迹象但上述条件不完整、改善集中于少数cells或类别之间方向不一致。RF/参数变动另作secondary不确定性，不为正结果调整判据。
- NOT SUPPORTED：总体无明确改善且没有预注册offset方向支持，或总体恶化。小正负效应同时结合CI呈现，不发明显著性阈值。

Primary与secondary若不一致，分别报告；生理中心是否正确仍未知。无论结果如何，均不自动训练trainable-center model，不改变offset/radius/sign/scale、不扩展illusion。后续方向只在最终五问中据证据提出，须另轮授权才能执行。

## 输出与范围

新fits：`output/real_data/schottdorf_canonical_v1_fixed_ln_center_22cell_20260905/`；新audit：本目录。保存protocol、preflight、coordinate mapping、paired/逐cell结果、correlation、mechanism comparison、报告和hash manifest；逐cell保存raw/final state、inner/refit trajectories、selected step、metadata/固定坐标及validation predictions。新增脚本只在这两个目录。训练或preflight失败不删除证据，不重置已有目录，不静默重训。

## 执行细节冻结（训练前补齐）

仅在output-local runner中替换传给`build_mechanistic_retina`的内存center；metadata同时保存原零坐标与aligned坐标。直接调用production `select_and_refit_r4`，不复制优化循环。manifest覆盖models/training/evaluation/data全部Python源码、所有reference与旧audit文件、LN final checkpoints/results、原movie/spikes、protocol及本轮执行脚本。每个训练attempt从空cells目录开始；失败保留证据并停止，不自动重试或续训。

当前stimulus domain定义为规则17×17 cone grid的x/y最小最大值闭区间；无hole定义为support逐元素精确等于`distance <= fixed_radius`，不引入额外连通性合同。边界造成自然disk截断单列记录，不能clip中心。reference logits以`torch.equal`检查，NLL与JSON读出的float用`==`检查，不调整容差。

RF是validation各sequence末端logit对最后16个input lags的Jacobian，随后按当前实现跨sequence取均值；不改成mask/all-bin RF。RF相对L2为`||aligned-zero||/||zero||`，cosine为展平向量夹角余弦，energy centroid按各cone跨lag平方和加权；零分母留空，不加人为epsilon。

最终第五问的预先决策：NOT SUPPORTED降低该方向优先级、停止本轮spatial work；PARTIAL / MIXED优先寻找独立真实RF-center证据以区分proxy与模型限制；SUPPORTED仍先区分development收益与生理正确性。仅凭剩余LN gap不能确认proxy有问题；只有独立中心证据矛盾或明确boundary限制等证据，才认为讨论另一轮trainable-center实验的两个必要条件同时成立。本轮均不执行后续实验。
