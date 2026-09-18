# RetiPath capacity diagnosis：type-shared population fit

正式训练前冻结。本轮只有一个新增训练条件：type-shared RetiPath。沿用正式 `models.mechanistic_retina.retipath.RetiPath` 的 H1→BC→AC→两个有效空间模式的 E/I conductance 架构。无新增空间 basis、连接、隐藏层、非线性、RF loss 或机制；不修改原模型文件、alpha bounds 或历史 artifact。

## 参数共享与模型身份

22 cells，按真实 checkpoint 中 ON/OFF × midget/parasol 分四组（9 ON midget、4 OFF midget、5 ON parasol、4 OFF parasol）。共享同一 Parameter 对象，不是训练后平均权重。每组29个可学习标量：H1 tau/delay 2、BC tau/delay 8、BC sustained/transient × K=2 × temporal basis=3 的混合权重12、AC tau/delay 4、完整BC分支非对称斜率1、conductance正值输入映射 aE/aI 2。

每cell保留8个可学习标量：H1 amplitude 1、AC local/transient gate logits 2、BC/AC pathway gain 2、response bias 1、history gate 1、conductance后output scale 1。geometry/center、空间滤波器、polarity和既有固定量保持该cell原配置，均不学习。每cell每mode的初始训练RMS是固定输入单位，不是额外可学习参数。共4×29+22×8=292个可学习标量；当前逐cell模型为22×37=814。不以参数比宣传效率。原无效operator的96个固定Parameter/cell仍未启用，不计可学习参数。

固定两空间basis；保留整条BC分支选斜率后作用于各mode的语义，AC驱动按真实抑制符号还原。EL=0、EE=1、EI=−1/3、gL=1、V0=2/9、Cm=3×5ms；每bin解析指数更新，dt=1000/150ms；aE/aI正值、固定softplus基线1、等权mode平均、原adaptation与严格过去history均保持。数学模型不变，训练封装将同类型共用的H1状态与AC/电导运算批处理，逐cell输出与梯度必须和正式forward在记录的浮点误差内一致。各cell的响应条件独立，不新增cell间生理耦合。

## 数据、初始化与选模

复用 OpenRetina 能力评估的缓存数据及 `make_inner_dev`：train [0,16)s，inner fit [0,12.4)，guard [12.4,12.8)，inner-validation评分 [12.8,16)。validation在guard之前的输入/history置零，沿用150-bin独立sequence、30-bin warmup/120-bin评分及原mask交集。17×17 L+M Weber、150Hz Bernoulli occupancy；source IDs、真实recording/trial身份和严格过去30ms history不变。不补零响应、不拼伪同时记录。

三个seeds固定2026091301/2026091302/2026091303。每update抽4个同源movie窗口，每cell为各窗口独立抽取真实trial，与OpenRetina相同的MovieBank采样程序及generator seed+1000003。优化每cell平均masked Bernoulli NLL后对22cells等权平均；无额外正则。此为联合训练，当前参照为既有逐cell训练，更新的观察数及联合选模机会不同，结果不能隔离成纯粹统计正则化的因果效应。

不从已拟合/已见inner-validation的checkpoint权重初始化。只读取正式registry的模型配置、固定位置、类型、极性，与Phase2 fresh初始化一致（bias −2、alpha/aE/aI/output scale=1；原其他初值）。inner阶段在初始模型和inner-fit输入计算每cell固定RMS，排除guard/inner-validation；full refit在初始模型和完整[0,16)重算固定RMS，之后不随训练更新。初始参数不从development统计取得。历史geometry/架构已经选择过，保留其历史先验边界。

本轮数据选择/refit/评分合同与OpenRetina评估相同；优化器和预算沿用正式RetiPath Phase2：Adam lr=.003、betas=(.9,.999)、eps=1e−8、weight_decay=0，无scheduler，clip global norm=5，最多3000联合updates。step0及每25步评估完整inner-fit/inner-validation。以22-cell等权inner-validation NLL精确最小值选择一个全population step，每seed一次选择，平局最早；400 updates无超过1e−7改善早停（min_delta仅用于patience）。没有配置搜索或追加seed/重启/延长预算。记录每cell与population曲线、真实成本和停止原因；预算耗尽不代表充分收敛。

三次selection全部冻结后，丢弃其参数和optimizer；每seed使用相同初值全新Adam，fresh refit完整[0,16)到该seed选定step。三次refit全部冻结后才评分后续区间。数值运行采用原RetiPath CPU FP32、单线程/worker、确定性算法，三个seed可并行执行。工程检查更新丢弃，不构成额外模型选择。

## 冻结评价与统计

只评价已消费[16,20)（development描述）及[20,60)（主要能力比较）。不读取[240,300)或其他时间block，不做RF/错视。当前RetiPath与OpenRetina作者实现的任务适配版本复用上一轮每cell每seed冻结scores/logits，核对输入、target、mask、source IDs及checkpoint hashes；不重训或重新选择这两个参照。

NLL单位nats/scored bin，float64评分。Primary Δ=type-shared RetiPath−current RetiPath；secondary Δ=type-shared RetiPath−OpenRetina-adapted，负数有利共享模型。每区间、每seed及seed aggregate报告absolute equal-cell mean NLL、paired mean/median、wins/losses/ties、paired-cell percentile95% CI；100000次bootstrap、seed20260908、cell字典序、tie容差1e−7。先逐cell平均三个seed的loss再bootstrap cells，不平均概率/权重，不把cell×seed、不同时间段当额外生物重复。四type各自报告相同配对结果及N，属描述性分组，不作多重校正的发现声明。CI跨零称未分辨，不称等效。

OpenRetina gap减少量=current−shared（原gap=current−OpenRetina，剩余gap=shared−OpenRetina）；原gap为正时同时给出均值gap减少比例，但不由比例单独判断预测胜负。新结果只定位共享训练方案是否有帮助，不能从未改善直接证明表示容量不足或H1→BC加层必要。最终回答四个指定问题，完成后停止。
