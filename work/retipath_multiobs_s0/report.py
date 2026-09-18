from __future__ import annotations

import csv
import statistics

from run import ROOT, OUT, CONDITIONS, INTERVENTIONS, read_json, sha, write_json


def cell(value):
    return f"{value['mean']:.6f} [{value['min']:.6f}, {value['max']:.6f}]"


def report(answers):
    summary = read_json(OUT / "summary.json")
    verification = read_json(OUT / "verification/final.json")
    protocol = read_json(OUT / "protocol.json")
    assert summary["status"] == "COMPLETE" and verification["status"] == "PASS"
    assert len(answers) == 8
    conditions = summary["conditions"]
    labels = dict(zip(CONDITIONS, ("A RGC_ONLY", "B RGC_H1", "C RGC_BC", "D RGC_H1_BC", "E EXTRA_RGC_CONTROL"), strict=True))
    lines = ["# RetiPath Synthetic MultiObs Stage S0", "",
        "核查日期：2026-09-17。单个 frozen canonical teacher 的 realizable synthetic proof-of-principle；不是真实视网膜机制验证。", "",
        "## 1. 八个问题的回答", ""]
    for index, answer in enumerate(answers, 1):
        lines.extend([f"{index}. {answer}", ""])
    lines += ["## 2. 运行身份与边界", "",
        "有效运行完成 **5 conditions × 3 student seeds × 3000 macro-updates = 45,000 optimizer updates**。仅使用每次训练的 final checkpoint，无 early stopping、checkpoint selection、延长或结果驱动条件扩展。正式模型与默认训练入口未修改。", "",
        f"Teacher：cell `67#7`，seed `2026091301`，`CanonicalGainRetiPath`；checkpoint SHA256 `{protocol['teacher']['sha256']}`。全部参数冻结。其学习参数仅定义已知 synthetic dynamical system，不称为真实生理 ground truth。", "",
        "Students：seeds `2026091701/02/03`。复用标准 fresh RetiPath 初始化，再通过现有 exact canonical coordinate migration 得到各自33个可学习标量；未从教师学习参数或 optimizer state warm-start。所有条件在同一seed共享完全相同初值与相同dataset minibatch schedule。几何、配置、固定RMS沿用教师，以保持同一个可实现的坐标系统；未读取真实刺激或spike targets。", "",
        f"协议 SHA256：`{summary['protocol_sha256']}`。有效运行协议锁定时间 `{protocol['frozen_utc']}`。运行设备 CPU、float32、每fit1线程、deterministic algorithms；统计归约使用float64。", "",
        "### 作废尝试的完整披露", "",
        "首版固定光栅仅有少量phase/sign组合。逐条stimulus哈希审计发现D_B held-out中7个、mechanism_B中10个唯一波形与train重复，因此在任何held-out指标查看之前终止了首批5个部分训练。没有final checkpoint，所有结果排除；未根据拟合效果修改实验。", "",
        "该尝试保存在 `output/experiments/retipath_multiobs_synthetic_s0_20260917_invalid_stimulus_overlap/`。末次日志更新数：A2500、B1750、C1250、D1250、E2500；日志每250步记录，精确中断步数为UNVERIFIED。**45,000只计有效benchmark，不能当成本轮全部计算量**。源码、旧协议、日志和`INVALIDATION.json`、`interruption_accounting.json`均保留。", "",
        "修正仅为固定空间光栅引入独立连续phase及每15bin独立contrast amplitude，保持原定刺激家族。全部有效数据和protocol重新锁定后才开始有效学生训练。576条序列及其计分段逐条哈希唯一，无train/held-out/mechanism完全重复。", "",
        "## 3. Synthetic 数据与观测合同", "",
        "全部sequence为150bins、150Hz、独立state reset；前30bins只作warmup，后120bins评分。Weber输入固定在[-0.8,+0.8]内，不进行结果后幅度调整。", "",
        "**150Hz是simulation/data clock（dt≈6.67ms），不是刺激以150Hz明暗振荡。** 当前被冻结的noise components均为AR(1)：H rho=.85（相关tau≈41.02ms），BC flicker rho=.30（tau≈5.54ms），R rho=.92（tau≈79.95ms）。对应线性AR分量的3dB频率约3.89/33.10/1.99Hz；tanh、steps和reversals会改变最终频谱，不能把这些数当作完整刺激的硬cutoff或生理参数。", "",
        "运行中收到的50/20/40ms建议不同于上述已冻结合同，尤其BC flicker原设计更宽带。用户随后明确选择完成当前冻结S0并报告差异，因此未替换刺激或预算。本报告不把现有结果描述为已经检验了该新参数版本，也不据此声称biological plausibility或real transfer有效性。", "",
        "| Dataset | Train / held-out | 预先固定generator |",
        "|---|---:|---|",
        "| D_H | 64 / 32 | Gaussian空间平滑noise，sigma2 grid pixels；ARrho=.85；30bin contrast steps；有界tanh |",
        "| D_B | 64 / 32 | 一半full-field ARrho=.3 flicker；一半8pixel周期固定空间光栅、0/45/90/135度、每15bin reversal；独立phase Uniform[0,2pi)、block amplitude Uniform[.2,.6] |",
        "| D_R | 64 / 32 | sigma1.25空间相关、rho=.92时间相关local+global Gaussian过程，再有界tanh |",
        "| D_R_EXTRA | 192 / 无专属held-out | 同D_R generator，独立seed；E仅用该192条，评价使用共同D_R held-out |",
        "| Mechanism bank | 0 / 96 | 全新独立seeds，H/B/R families各32条；只用于interventions和cross-seed ambiguity |", "",
        f"H1：固定functional center `{protocol['observation']['functional_center_deg']}` 附近最近graph node `{protocol['observation']['h1_node_index']}`，位置 `{protocol['observation']['h1_node_deg']}`；不依据student或响应挑选。", "",
        "BC：canonical `parts.direct` 和 `parts.broad` 在K=2空间模式轴求和，得到direct sustained/transient、broad sustained/transient四值，位于composition/gain之前。与正式forward的`bc_direct_presynaptic`/`bc_broad_presynaptic`逐项核对。不是BC膜电位或释放的生理声明。", "",
        "Observation head为identity，没有measurement noise、scale/offset学习或temporal decoder。H1及每个BC channel使用各自teacher TRAIN计分段的mean和population std；所有student使用相同固定常数。`L_H`为标准化MSE，`L_B`对time×4channels平均，`L_R`为项目现有Bernoulli NLL；active dataset losses直接相加，系数全为1。", "",
        "Spikes复用项目`clean_sampled_data._sample_spikes`，逐时刻从teacher probability采样，过去sampled spikes进入下一时刻history。Teacher true conditional probabilities与training targets分文件保存，训练函数不读取它们。评价时teacher和student都以同一已采样过去历史为条件。教师history gate实际为0：因果采样语义保留，但该teacher的history contribution为零；另用标准初值的非零history gate验证采样重放。", "",
        "## 4. 固定训练与评价", "",
        "Adam lr=.003、betas=(.9,.999)、eps=1e-8、weight decay0、batch4/dataset、global gradient clip5。采用当前RetiPath Phase2的optimizer/batch/上限；旧400-step/.03 synthetic合同对应较早的pre-spatial模型，不用于本次canonical gain optimizer。H→B→R依次累积梯度，再一次optimizer.step；只执行该condition存在的数据集。沿用原参数约束，包括alpha[.05,2]和history gate[0,1]。", "",
        "H1 recovery在D_H held-out，BC recovery在D_B held-out；nRMSE分母为各自teacher TRAIN std。BC_mean4为四个channel指标的算术平均。相关系数为scored bins展平后的Pearson correlation，详见CSV。", "",
        "四项正式interventions均在同一独立96-sequence mechanism bank上运行，NORMAL和intervened两边都FIX_HISTORY_ZERO。Delta=normal logit−intervention logit；E=RMS(student Delta−teacher Delta)/(RMS(teacher Delta)+1e-8)。Intervention targets从不进入loss。", "",
        "A_latent也使用这同一个96-sequence bank，H1/BC差异用teacher TRAIN std标准化；A_intervention以同intervention的teacher Delta RMS+eps标准化。每condition保留三对seed的原始值，并报告其平均。不能用raw parameter variance替代这些指标。", "",
        "下列表格为三个seed均值及[min,max]；不进行3-seed显著性或population推断。预先固定的“prediction保持”描述界限是expected CE相对比较条件均值上升不超过1%，不是统计non-inferiority结论。", "",
        "## 5. 预测与latent recovery", "",
        "| Condition | Sampled NLL | Expected CE | H1 nRMSE | BC mean4 nRMSE |",
        "|---|---:|---:|---:|---:|"]
    for name, value in conditions.items():
        lines.append(f"| {labels[name]} | {cell(value['sampled_NLL'])} | {cell(value['expected_CE'])} | {cell(value['H1_nRMSE'])} | {cell(value['BC_nRMSE'])} |")
    lines += ["", f"Teacher sampled NLL={summary['teacher']['sampled_NLL']:.9f}，teacher expected CE/entropy={summary['teacher']['expected_CE_entropy']:.9f} nats/bin。后者是已知true probability的参考；完整excess CE在`heldout_prediction.csv`。", "",
        "## 6. Primary：held-out intervention recovery", "",
        "| Condition | H1 feedback block | Direct BC drive block | AC postsynaptic drive block | Adaptation term removal |",
        "|---|---:|---:|---:|---:|"]
    for name, value in conditions.items():
        lines.append("| " + labels[name] + " | " + " | ".join(cell(value["interventions"][key]) for key in INTERVENTIONS) + " |")
    with (OUT / "intervention_recovery.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    lines += ["", "Teacher Delta RMS（logit units）：" + "；".join(f"{key}={float(next(r['teacher_Delta_RMS_logit'] for r in rows if r['intervention']==key)):.9g}" for key in INTERVENTIONS) + "。同时报告absolute RMS error，防止小分母的相对误差被误读。", "",
        "## 7. Cross-seed ambiguity", "",
        "| Condition | H1 latent | BC mean4 latent | H1 intervention | BC intervention | AC intervention | Adaptation intervention |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in conditions.items():
        lines.append("| " + labels[name] + " | " + " | ".join(f"{value['ambiguity'][key]:.6f}" for key in ("H1", "BC_mean4", *INTERVENTIONS)) + " |")
    lines += ["", "## 8. Secondary：canonical parameter recovery", "",
        "`parameter_recovery.csv`提供20个对应量×15fits=300行：H1 tau/delay/amplitude、六个BC basis taus及两类delay、alpha、canonical G_E/G_I、两项独立composition weights、两类AC tau/delay。保存teacher值、student值和绝对/相对误差。未比较旧raw gain/gauge坐标，也未把这些参数误差作为主结果或唯一可识别性的证明。", "",
        "## 9. 解释边界与验证", "",
        "- A/B/C/D的RGC训练集相同；E使用192条独立RGC序列。相同optimizer steps和batch不等于相同计算量：D每步三个dataset batches，E每步一个。E也不是information-matched control；无噪声连续latent观测与随机binary spikes的信息量不同。", 
        "- 初始化随机性沿用项目标准的小幅BC perturbation，另有各seed minibatch差异。只有一个teacher、三个seeds，不能据此断言消除了全部机制歧义或发现了所有等价解。",
        "- 固定有限训练预算下的恢复差异，不能单独区分有限样本、优化不足与结构不可辨识。Output-close/mechanism-far是本benchmark的观测现象，不是RGC-only在无限数据/充分优化下必然不可辨识的数学证明。",
        "- H1 state与反馈amplitude是不同量；H1 identity监督不直接观测amplitude。BC观测是四个K-summed branch values，不是两个K modes分别的全状态。未监督AC或RGC内部变量。",
        "- 任何正结果仅支持当前realizable synthetic teacher下的latent/intervention约束作用。真实HC/BC observation correspondence、measurement nuisance和model mismatch均未测试。",
        "- 11项预训练检查通过；15个final checkpoint与optimizer step数核实；teacher与正式源码hash未变；保存数组对prediction/latent/intervention/ambiguity CSV逐行公式重算通过。这是执行者独立公式重算，不冒称独立人员审计。",
        "- 未读取真实physiology数据、未做population training、未改architecture、未调loss weights、未运行S1。完成此S0后停止。", "",
        "## 10. 文件", "",
        "结果根目录：`output/experiments/retipath_multiobs_synthetic_s0_20260917/`。", "",
        "- `protocol.json`、`teacher_metadata.json`、`normalization.json`、`dataset_manifest.json`",
        "- `datasets/`：train/held-out inputs、synthetic observations、分离的teacher probabilities和独立mechanism stimuli",
        "- `training_curves.csv`、`heldout_prediction.csv`、`latent_recovery.csv`、`parameter_recovery.csv`",
        "- `intervention_recovery.csv`、`ambiguity.csv`、`summary.json`",
        "- `initial_states/`、`checkpoints/`、`evaluation_arrays/`、`verification/`、`evaluation_manifest.json`",
        "- 实现：`work/retipath_multiobs_s0/`；`run.py prepare`→`test_contract.py`→`launch.py`→`evaluate.py`→`verify_results.py`。现有输出拒绝自动覆盖或重新训练。", ""]
    path = ROOT / "docs/RETIPATH_MULTIOBS_SYNTHETIC_S0.md"
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    write_json(OUT / "delivery.json", {"report": str(path.relative_to(ROOT)), "sha256": sha(path),
        "valid_optimizer_updates": 45000, "invalid_predecessor_disclosed": True,
        "summary_sha256": sha(OUT / "summary.json"), "verification_sha256": sha(OUT / "verification/final.json")})
    print(path)
