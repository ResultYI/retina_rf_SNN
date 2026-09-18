from __future__ import annotations

from pathlib import Path

from run import ROOT, OUT, CONDITIONS, read_json, write_json, sha


def cell(value):
    return f"{value['mean']:.6f} [{value['min']:.6f}, {value['max']:.6f}]"


def report(answers):
    summary = read_json(OUT / "summary.json")
    p = read_json(OUT / "protocol.json")
    verification = read_json(OUT / "verification/final.json")
    normalization = read_json(OUT / "normalization.json")
    assert summary["status"] == "COMPLETE" and verification["status"] == "PASS"
    assert len(answers) == 8
    labels = dict(zip(CONDITIONS, ("A RGC_ONLY", "B RGC_STATE", "C RGC_FEEDBACK", "D RGC_STATE_FEEDBACK"), strict=True))
    lines = ["# Synthetic S0.5 — H1 Observation–Mechanism Alignment", "",
        "2026-09-17。一个已知synthetic RetiPath teacher的固定预算实验；不是真实HC voltage或真实视网膜机制验证。", "",
        f"**Verdict：{summary['verdict']}**。四条件×三个seed均完成3000次macro-updates，共36,000次有效optimizer updates；最终数值重算通过。", "",
        "## 1. 八个问题", ""]
    for index, answer in enumerate(answers, 1):
        lines += [f"{index}. {answer}", ""]
    lines += ["## 2. 冻结身份、数据与公平性", "",
        f"Teacher沿用S0：cell `67#7`、seed `2026091301`、checkpoint SHA256 `{p['teacher']['sha256']}`。Teacher全冻结，学习参数仅代表此synthetic dynamical system。", "",
        f"S0.5 protocol SHA256：`{summary['protocol_sha256']}`，冻结时间 `{p['frozen_utc']}`；来源S0 protocol SHA256：`{p['S0_protocol_sha256']}`。", "",
        "学生seeds为2026091701/02/03，逐字节复用S0保存的fresh canonical initial states、固定geometry/RMS与33个可学习标量。未从S0 final或teacher学习参数warm-start。A/B重新按本轮预算训练，三个A与三个B的final model state hashes分别与S0 RGC_ONLY/RGC_H1完全一致。", "",
        "复用D_H train64/held-out32、D_R train64/held-out32，以及同一个96条mechanism bank。没有重新生成任何stimulus或spike。Mechanism bank仍为原H/B/R各32条；其中B-family只是被冻结评价库的一部分，本轮未读取D_B观测数据、D_R_EXTRA或真实数据。", "",
        "这些held-out stimuli已用于S0评价。本轮按用户指定复用，属于既有synthetic bank上的限定比较，independent new confirmation test = NONE。协议与checkpoint核对中的文件哈希读取只用于身份验证；held-out response指标在12个final完成后统一计算。", "",
        "每条150bins，150Hz，独立state reset，前30bins warmup、后120bins计分。时间相关参数原封不动：H的AR rho=.85（相关tau约41.02ms），R的rho=.92（约79.95ms）；机制库也逐字节复用，包括原B片段。没有应用后来提出的50/20/40ms方案。", "",
        "Adam lr=.003、betas=(.9,.999)、eps1e-8、weight decay0、batch4、clip5、原project_mechanism_parameters与全部参数约束。每fit固定3000步，final-only，无early stopping或checkpoint selection。CPU float32、每fit单线程、deterministic algorithms；统计归约float64。", "",
        "所有active loss系数为1：A仅RGC Bernoulli NLL；B增加state MSE；C增加feedback MSE；D增加两者。B/C/D每步使用完全相同的S0 H batch序列；D的两个loss来自同一个H forward和同一个batch，求和后backward，再累积RGC梯度，只调用一次optimizer.step。RGC batch序列也与S0一致。", "",
        "B/C均监督一个scalar trace；D同时监督两个scalar traces，因此D与B/C不是观测信息量完全相同的比较。相同optimizer steps也不意味着A与其余条件计算量相同。", "",
        "## 3. State与feedback的正式定义", "",
        "正式计算：`graph_drive = G(x)`；`state = lowpass(delay(graph_drive))`；`feedback = A_H × G.transpose_apply(state)`；下游输入为`x − feedback`。实现直接复用`H1Pathway.forward`，没有重写反馈量或更改架构。", "",
        "State observable固定为node146。State由H1 tau/delay和固定输入图决定，没有对H1 amplitude的直接依赖；RGC loss仍可通过下游间接约束amplitude。", "",
        f"Feedback observable仅取pixel `{p['observation']['feedback_pixel']}`，位置 `{p['observation']['feedback_pixel_deg']}` deg。功能中心 `{p['observation']['functional_center_deg']}` deg。规则为输入geometry到冻结中心的平方距离argmin，平局取首索引；已在任何student response之前写入protocol。", "",
        "虽然两者索引均为146，feedback并不是简单的node146 state乘amplitude：它先聚合返回映射所覆盖的多个state。Feedback监督同时约束动态状态的返回映射结果与amplitude，不是amplitude-only监督，也不代表测得的feedback current。", "",
        f"State normalization完全复用S0 TRAIN常数：mean={normalization['state']['mean'][0]:.12g}，std={normalization['state']['std'][0]:.12g}。Feedback仅使用teacher D_H TRAIN的7680个scored bins冻结：mean={normalization['feedback']['mean'][0]:.12g}，std={normalization['feedback']['std'][0]:.12g}。Population std，identity observation head，无noise、可学习scale/offset/filter。", "",
        "正式forward逐项对齐检查通过；H1 block使feedback归零而保留state。预训练autograd检查也确认state loss路径对amplitude无直接梯度、feedback路径具有非零amplitude梯度。检查未执行optimizer updates。", "",
        "## 4. 评价与预先冻结判据", "",
        "以下表格为三个seed均值及[min,max]。只有三个seed，不作显著性或population推断。State/feedback分别在相同D_H held-out评价，nRMSE分母为对应teacher TRAIN std，correlation为计分段展平后的Pearson correlation。", "",
        "唯一primary intervention为BLOCK_H1_FEEDBACK。NORMAL与block均使用FIX_HISTORY_ZERO，Delta=normal logit−blocked logit；E_H1=RMS(student Delta−teacher Delta)/(RMS(teacher Delta)+1e-8)。未计算其他干预，也未使用任何intervention loss。", "",
        "A_state、A_feedback和A_intervention全部在同一个96条mechanism bank计分段上计算，分别用state TRAIN std、feedback TRAIN std和teacher Delta RMS+1e-8标准化；保留三对seed原始值并报告其均值。三个pair共享seed，不能当作三个独立重复实验。", "",
        "用户确认并在训练前冻结：明确改善=平均归一化误差至少下降10%，且3/3配对seed严格同向改善。D保持改善=相对A/B均保留上述改善，且feedback与H1 intervention平均误差各不超过C的105%。", "",
        "- STATE_SUPERVISION_SUFFICIENT：B相对A同时明确改善state、feedback和H1 intervention。",
        "- FEEDBACK_OBSERVATION_REQUIRED：B明确改善state recovery，或A_state下降至少10%；C相对A和B均明确改善feedback与H1 intervention；D保持上述改善。",
        "- MIXED_OR_OPTIMIZATION_LIMITED：不满足前两者，包括配对seed方向不稳定。按上述顺序裁决，无结果后改判据。", "",
        "RGC prediction在共同D_R held-out评价，使用相同teacher sampled past；teacher true probabilities只用于评价。沿用S0的1% expected CE均值描述界限，不是统计non-inferiority证明。Teacher history gate实际为0，正确因果采样语义保留。", "",
        "## 5. Held-out state、feedback与primary intervention", "",
        "| Condition | State nRMSE | State correlation | Feedback nRMSE | Feedback correlation | H1 intervention E |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, value in summary["conditions"].items():
        lines.append("| " + labels[name] + " | " + " | ".join(cell(value[key]) for key in
                     ("state_nRMSE", "state_correlation", "feedback_nRMSE", "feedback_correlation", "H1_intervention_error")) + " |")
    lines += ["", f"Teacher H1 Delta RMS={summary['teacher']['H1_Delta_RMS_logit']:.12g} logit units；CSV同时保存absolute RMS error。", "",
        "## 6. 三个H1参数（secondary）", "",
        "| Condition | Tau ms | Delay ms | Amplitude |",
        "|---|---:|---:|---:|"]
    names = ("H1_tau_ms", "H1_delay_ms", "H1_amplitude")
    lines.append("| Teacher | " + " | ".join(f"{summary['teacher']['parameters'][key]:.9f}" for key in names) + " |")
    for name, value in summary["conditions"].items():
        lines.append("| " + labels[name] + " | " + " | ".join(cell(value["parameters"][key]["value"]) for key in names) + " |")
    lines += ["", "CSV保存每个seed的teacher/student值、带符号误差、绝对误差和相对绝对误差。未扩展全参数audit。参数误差与intervention误差的共同变化只能描述对应关系；本轮未做amplitude单独替换或因果中介实验，不能把所有改善归因于amplitude。", "",
        "## 7. Cross-seed ambiguity与RGC prediction", "",
        "| Condition | A_state | A_feedback | A_intervention | Sampled NLL | Expected CE |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, value in summary["conditions"].items():
        lines.append("| " + labels[name] + " | " + " | ".join(f"{value['ambiguity'][key]:.6f}" for key in ("state", "feedback", "delta"))
                     + " | " + cell(value["sampled_NLL"]) + " | " + cell(value["expected_CE"]) + " |")
    lines += ["", f"Teacher sampled NLL={summary['teacher']['sampled_NLL']:.9f}，expected CE/entropy={summary['teacher']['expected_CE_entropy']:.9f} nats/bin。", "",
        "## 8. 判据逐项结果", "", "| Gate | Result |", "|---|---|"]
    for key, value in summary["verdict_gate_results"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "每项mean ratio、相对变化和三个seed的paired differences保存在summary.json；不使用correlation或ambiguity替代ground-truth recovery裁决。", "",
        "## 9. 验证与解释边界", "",
        "- 7项预训练检查通过；12个final checkpoints均为3000步，每个optimizer参数的step均核实。A/B六个模型与S0对应结果逐字节相同。",
        "- 从保存数组使用NumPy另写公式重算state/feedback、H1干预、预测、三参数和pairwise ambiguity，并核对loss总和、summary与固定verdict；全部通过。这是执行者的独立公式重算，不冒称独立人员审计。",
        "- Source与输入hash保持冻结，S0 artifacts只读。实现仅位于work/retipath_multiobs_s05；新结果位于本轮独立目录。",
        "- State loss不直接依赖amplitude是代码与数学性质；有限预算训练结果不能单独区分有限样本、优化限制与结构不可辨识，也不能证明唯一参数恢复。",
        "- HC voltage到此H1 state的真实观测对应关系未建立。即使synthetic feedback监督有效，也不等于真实HC voltage或可测feedback量能提供相同约束。",
        "- 未改architecture、未用D_B或D_R_EXTRA观测、未加BC/AC监督、未调loss weights、未增加seed或steps、未进入S1。完成后停止。", "",
        "## 10. 文件", "",
        "结果目录：`output/experiments/retipath_multiobs_s05_h1_observability_20260917/`。", "",
        "- protocol.json、normalization.json、feedback_train.pt、preparation_complete.json",
        "- training_curves.csv、heldout_state_feedback.csv、h1_parameter_recovery.csv",
        "- intervention_recovery.csv、ambiguity.csv、prediction.csv、summary.json",
        "- checkpoints/、curves/、evaluation_arrays/、verification/、evaluation_manifest.json",
        "- 运行实现：work/retipath_multiobs_s05/run.py、launch.py、evaluate.py、verify_results.py；报告生成器report.py。", ""]
    path = ROOT / "docs/RETIPATH_MULTIOBS_S05_H1_OBSERVABILITY.md"
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    write_json(OUT / "delivery.json", {"report": str(path.relative_to(ROOT)), "report_sha256": sha(path),
        "summary_sha256": sha(OUT / "summary.json"), "verification_sha256": sha(OUT / "verification/final.json"),
        "report_generator_sha256": sha(Path(__file__)), "valid_fits": 12, "optimizer_updates": 36000})
    print(path)
