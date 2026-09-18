from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from adapter import AUTHOR, AUTHOR_SHA, WORK, OpenRetinaAdapted, author_configuration
from data_adapter import DEST, FINAL, ROOT, digest, exclusive_json, read_json, training_banks
from training_engine import loss_for_sample, optimizer_and_scheduler, SEEDS, MAX_UPDATES, EVAL_EVERY


def main() -> None:
    if (DEST / "PROTOCOL.md").exists():
        raise FileExistsError("Protocol already exists; do not replace a frozen run")
    banks, data_metadata = training_banks()
    cells = tuple(banks["inner_fit"].cells)
    assert len(cells) == 22 and MAX_UPDATES == 15 * EVAL_EVERY
    assert EVAL_EVERY == (max(len(o.source_ids) for o in banks["inner_fit"].cells.values()) + 3) // 4
    registry = read_json(FINAL / "model_registry.json")
    assert registry["entry_point"] == "models.mechanistic_retina.retipath.RetiPath"
    assert registry["seeds"] == list(SEEDS)
    sys.path.insert(0, str(ROOT / "work"))
    from retipath_final_common import load_retipath
    from retipath_spatial_ei_pilot import model_args
    from models.mechanistic_retina.retipath import RetiPath
    identities = {}
    for cell in cells:
        identities[cell] = {}
        for seed in SEEDS:
            ref = registry["cells"][cell]["RetiPath"][str(seed)]
            path = ROOT / ref["path"]
            assert digest(path) == ref["sha256"]
            model, cp = load_retipath(path)
            identity_model = RetiPath(*model_args(cp), rms_e=torch.tensor(cp["rms"]["e"]),
                                     rms_i=torch.tensor(cp["rms"]["i"]))
            n = sum(p.numel() for p in identity_model.parameters() if p.requires_grad)
            assert n == 37 and cp["backend"] == "spatial_conductance" and cp["phase"] == "refit"
            assert cp["step"] == cp["best_step"] == ref["selected_updates"]
            identities[cell][str(seed)] = {**ref, "trainable_parameters": n,
                "fixed_parameter_entries": sum(p.numel() for p in identity_model.parameters() if not p.requires_grad),
                "backend": cp["backend"], "spatial_modes": 2,
                "history_tau_ms": cp["model_config"]["history_tau_ms"],
                "history_gain": cp["model_config"]["history_gain"]}
    del model
    cfg = author_configuration()
    from openretina.modules.core.base_core import SimpleCoreWrapper
    from openretina.modules.readout.multi_readout import MultiSampledGaussianReadout
    args = {k: v for k, v in cfg["core"].items() if not k.startswith("_") and k != "channels"}
    args["channels"] = (1, *cfg["hidden_channels"])
    native_core = SimpleCoreWrapper(**args).cuda().eval()
    native_readout = MultiSampledGaussianReadout(in_shape=(64, 1, 3, 3), n_neurons_dict={"native_example": 1},
        **{k: v for k, v in cfg["readout"].items() if not k.startswith("_") and k != "in_shape"}).eval()
    with torch.no_grad():
        native_features = native_core(torch.zeros(1, 1, 31, 47, 47, device="cuda")).cpu()
        native_rate = native_readout(native_features)
    assert native_features.shape == (1, 64, 1, 3, 3) and bool(torch.isfinite(native_rate).all())
    del native_core, native_readout, native_features
    bank = banks["inner_fit"]
    schedule = torch.Generator().manual_seed(2026091398)
    windows, rows = bank.sample(4, schedule)
    models, checks = [], []
    for repeat in range(2):
        model = OpenRetinaAdapted(cells, bank.means(), 2026091398).place()
        optimizer, scheduler = optimizer_and_scheduler(model)
        ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert ids == {id(p) for p in model.parameters() if p.requires_grad}
        model.train()
        before = model.cpu_state()
        begin = time.perf_counter()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, stats = loss_for_sample(model, bank, windows, rows)
            assert bool(torch.isfinite(loss))
            loss.backward()
            missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
            assert not missing, missing
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
            optimizer.step(); scheduler.step(); model.project_readout()
        elapsed = time.perf_counter() - begin
        state = model.cpu_state()
        changed = [name for name, p in model.named_parameters() if not torch.equal(before[name], state[name])]
        assert len(changed) == len(list(model.named_parameters()))
        checks.append({"repeat": repeat, "two_update_seconds": elapsed, "gradient_norm": gradient_norm,
                       "all_expected_parameters_optimizer_listed_and_updated": True, **stats})
        models.append(state)
    assert all(torch.equal(models[0][k], models[1][k]) for k in models[0])
    model.eval()
    x = bank.inputs[:1].clone()
    cell = cells[0]
    obs = bank.cells[cell]
    events = obs.targets[:1].clone()
    with torch.no_grad():
        z = model(x, events, cell)
        future = x.clone(); future[:, 91:] += 0.5
        z_future = model(future, events, cell)
        future_error = float((z[:, :91] - z_future[:, :91]).abs().max())
        torch.testing.assert_close(z[:, :91], z_future[:, :91], atol=1e-6, rtol=1e-6)
        changed_events = events.clone(); changed_events[:, 80:] = 1 - changed_events[:, 80:]
        changed_z = model(x, changed_events, cell)
        history_error = float((z[:, :81] - changed_z[:, :81]).abs().max())
        assert history_error == 0
        h = model.history_feature(events)
        expected = torch.zeros_like(h)
        for t in range(1, len(h[0])):
            expected[:, t] = model.history_decay * expected[:, t - 1] + (1 - model.history_decay) * events[:, t - 1]
        torch.testing.assert_close(h, expected, atol=1e-7, rtol=1e-6)
        features = model.core_features(x)
        direct_raw = model.readout[model.key(cell)](features.permute(0, 2, 1, 3, 4).reshape(-1, 64, 17, 17)).reshape(1, 150, 1)
        assert torch.equal(direct_raw, model.stimulus_drive(features, cell))
        assert torch.equal(z, direct_raw - model.history_coefficient(cell) * h)
        assert all(float(model.history_coefficient(c)) > 0 for c in cells)
        manual = -(events.double() * F.logsigmoid(z.double()) + (1 - events.double()) * F.logsigmoid(-z.double()))
        mask = obs.mask[:1]
        loss_error = abs(float(manual[mask].mean()) - float(F.binary_cross_entropy_with_logits(z.double()[mask], events.double()[mask])))
        assert loss_error < 1e-12
    DEST.mkdir(exist_ok=False)
    (DEST / "checkpoints").mkdir()
    evidence = DEST / "checkpoints" / "provenance"
    evidence.mkdir()
    torch.save(model.cpu_state(), evidence / "engineering_roundtrip.pt")
    restored = OpenRetinaAdapted(cells, bank.means(), 2026091398).place()
    restored.load_state_dict(torch.load(evidence / "engineering_roundtrip.pt", weights_only=True), strict=True)
    restored.eval()
    with torch.no_grad():
        assert torch.equal(restored(x, events, cell), z)
    frozen_files = [ROOT / "configs/retipath_final.json", ROOT / "models/mechanistic_retina/retipath.py",
        FINAL / "model_registry.json", FINAL / "completion.json", FINAL / "REPORT.md", FINAL / "prediction/per_cell_seed.csv"]
    frozen_files += list((ROOT / "models/mechanistic_retina").glob("*.py"))
    frozen_files += [FINAL / "prediction/cells" / (c.replace("#", "_") + suffix) for c in cells for suffix in (".npz", ".json")]
    frozen_files += list((AUTHOR / "openretina/modules").rglob("*.py"))
    frozen_files += [AUTHOR / "openretina/models/core_readout.py", AUTHOR / "CITATION.cff", AUTHOR / "pyproject.toml",
        AUTHOR / "configs/vystrcilova_2024_nm_cnn.yaml", AUTHOR / "configs/model/core_gaussian_readout.yaml",
        AUTHOR / "configs/optimizer/adamw.yaml", AUTHOR / "configs/lr_scheduler/one_cycle_lr.yaml",
        AUTHOR / "configs/training_callbacks/early_stopping.yaml"]
    frozen_files += [WORK / name for name in ("adapter.py", "data_adapter.py", "training_engine.py", "prepare.py")]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(frozen_files))}
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "author_sha": AUTHOR_SHA,
        "architecture_config": cfg, "retipath_identity": identities, "data": data_metadata,
        "frozen_sha256": hashes, "seeds": list(SEEDS), "maximum_updates": MAX_UPDATES,
        "evaluation_interval": EVAL_EVERY, "configuration_count": 1,
        "official_native_forward_shape": list(native_rate.shape), "engineering_training_checks": checks,
        "repeated_two_update_full_state_bitwise_equal": True, "future_frame_max_abs_error": future_error,
        "current_future_spike_max_abs_error": history_error, "Bernoulli_manual_max_abs_error": loss_error,
        "official_pre_nonlinearity_drive_verified": True, "single_readout_bias": True,
        "history_inhibitory_constraint": True, "checkpoint_roundtrip_bitwise_equal": True,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "shared_core_parameters": sum(p.numel() for p in model.core.parameters()),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0),
        "new_target_blocks": 0, "formal_training_started": False}
    exclusive_json(evidence / "pretraining_lock.json", record)
    installed = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (evidence / "environment.txt").write_text(installed.stdout, encoding="utf-8")
    (DEST / "PROTOCOL.md").write_text(protocol(record), encoding="utf-8")
    (DEST / "adapter_notes.md").write_text(notes(record), encoding="utf-8")
    exclusive_json(DEST / "correctness.json", {"status": "ENGINEERING_VERIFIED_BEFORE_TRAINING", **record})
    print(f"ENGINEERING PASSED: official SHA={AUTHOR_SHA}, parameters={record['model_parameters']}, 22 cells, 3 frozen seeds; maximum {MAX_UPDATES} selection updates/seed", flush=True)


def protocol(record: dict) -> str:
    link = f"https://github.com/open-retina/open-retina/blob/{AUTHOR_SHA}"
    return f"""# 当前 RetiPath 能力评估：冻结协议

冻结时间 {record['created_utc']}；工程检查通过后、正式训练前写入。仅新增一个比较模型，不修改或训练 RetiPath，不重跑 RF/错视，不访问新的时间块。

正式名称：**OpenRetina作者实现的任务适配版本**。来源为 [官方 marmoset natural-movie CNN 配置]({link}/configs/vystrcilova_2024_nm_cnn.yaml)，commit `{AUTHOR_SHA}`。选择依据是公开的 primate natural-movie Core+Readout 任务及完整配置可复现性；没有按本项目评分选架构。框架论文为 D'Agostino、Zenkel 等，openretina: Collaborative Retina Modelling Across Datasets and Species，doi:10.1101/2025.03.07.642012。配置对应 Vystrčilová 等 A systematic comparison of predictive models on the retina，doi:10.1101/2024.03.06.583740；自然电影数据来源为 Sridhar 等 Modeling spatial contrast sensitivity in responses of primate retinal ganglion cells to natural movies，doi:10.1101/2024.03.05.583449。配置/代码是本次具体实现和超参数依据；不宣称原论文成绩复现或 SOTA。软件 pyproject=1.1.1，CITATION.cff 仍为1.1.0；以固定 commit 为准。

## 架构与数据合同

保留作者 SimpleCoreWrapper、custom_separable 三层卷积、64/64/64 channels、temporal kernels 15/11/7、spatial kernels 21/15/11、BatchNorm3d、Bias3DLayer、ELU、Dropout3d=0.03097237571817182，以及 MultiSampledGaussianReadout/PointGaussianReadout、full covariance Gaussian sampling、作者初始化和五项正则设计。仅适配每层空间 zero padding、输入时间左 padding、输出 link、history、独立 recording 数据/训练接口。详细方程见 adapter_notes.md。无预训练权重。

RetiPath 由实际 final-model registry 解析：正式入口 models.mechanistic_retina.retipath.RetiPath，spatial_conductance，K=2，逐cell37个可学习参数；三个 Phase 2 full-train fresh-refit seeds 为 {list(SEEDS)}。全部66个 checkpoint 哈希、selected step、backend与参数数目在 checkpoints/provenance/pretraining_lock.json 核对。前端、geometry、history和模型参数全部冻结。

train=[0,16)s。原 make_inner_dev：inner fit [0,12.4)，guard [12.4,12.8)，inner validation评分 [12.8,16)。同源movie在所有cells/recordings使用同一划分。150Hz，原17×17 L+M Weber，Bernoulli occupancy，150-bin独立sequence，30-bin warmup及原mask交集，严格过去30ms指数history。训练阶段只解引用旧mmap归档的train部分；不使用development统计。各记录身份和source IDs原样保留。

模型共享 Core，22个cell各有独立作者Readout和抑制history系数。同cell的不同recordings/trials沿用同一head，来源映射显式保存。每update抽取4个同源movie窗口，每cell对每个窗口独立抽取一条真实recording/trial；这是22组独立条件似然之和，不是同时记录或population coupling。输入逐位一致才共享Core计算；targets/history/mask不互换，不补零。训练每cell等权采样，稀少重复的cell会有更高重用率。所有真实重复均参与完整train/inner-validation评分。

## 一套训练配置与预算

只用1套来源配置，无训练/正则搜索，无额外seeds。三个固定seeds与RetiPath匹配。作者 AdamW：max lr=.005，betas=(.9,.999)，eps=1e-8，weight_decay=.01，amsgrad=False；作者 OneCycleLR：pct_start=.3、cos anneal、momentum .85..95、div_factor25、final_div_factor10000；每update调度，gradient norm clip=1。FP32、禁TF32，确定性CUDA Core和确定性CPU作者Readout。

预算按作者max_epochs=15建立：最长cell的inner-fit有91个真实sequence；batch4/cell，每个名义平衡epoch ceil(91/4)=23 updates，最多15×23=345 updates/seed。随机有放回采样，名义epoch不声称逐样本遍历。每23步及step0对完整inner fit/validation评估。早停沿用10次验证无至少.001改善的patience，监控改为需最小化的equal-cell Bernoulli NLL；精确最低NLL选step（包括step0），并列取最早，min_delta仅用于patience。达到345步仅表示预算上限，不表示充分收敛。完整记录原作者15epochs与本项目数据量、batch、联合session更新的差别。

正则系数逐值保留：gamma_input=.4697500757926789；gamma_in_sparse=.2989540104057545；gamma_hidden=.44741472694032014；gamma_temporal=92.19523607880937；readout gamma=.23618371929818793、reg_avg=false。作者PoissonLoss3d默认sum，原objective为sum NLL+core reg+session readout reg。本任务用sum Bernoulli NLL+一次core reg+22个head各一次readout reg；不使用mean NLL再加未缩放正则。缺失/非评分位置在sum中排除。该联合session批次的有效观察数与作者原数据不同，明确记录NLL/正则实际尺度。AdamW的decoupled decay也保留。

全部3次selection完成并写入selection lock后，分别从同seed全新随机权重和optimizer fresh refit完整[0,16)，仅重算本阶段train-only bias初始化，训练到该seed选定updates。OneCycle仍使用345步定义，重放相同LR/momentum schedule前缀，不根据refit loss改长度，不带入inner权重或optimizer。全部3个refit checkpoints冻结并写evaluation lock后，才启动后续评分。工程两步重复性检查的模型不进入正式训练。

## 评价和停止

仅[16,20) development描述、[20,60) primary能力比较(22 cells)、[240,300)较远时间secondary(17 cells)。均为已消费区间的descriptive temporal evaluation，不消费[300,360)或其他时间。优先重用正式RetiPath已保存logits/scores并核对相同输入、target、mask和source ID哈希；必要的冻结重放获授权，但不重选checkpoint。模型选择冻结后不因结果追加训练。

ΔNLL=NLL_RetiPath−NLL_OpenRetina-adapted，负有利RetiPath。每区间、每seed及seed aggregate报告两者absolute equal-cell mean NLL、paired mean/median、wins/losses/ties、paired-cell percentile95% bootstrap CI。100000 resamples，seed20260908，cell字典序；数值tie沿用既有1e-7 nats/bin。先逐cell平均3个seed loss，再汇总和重采样cells，不ensemble概率/权重，不把cell×seed或时间段当额外生物重复。CI跨零只报告未分辨。

已有正式RF和Mach/SBC只作来源引用，不重跑。报告仅回答用户指定7项能力评估问题，区分数值稳定、验证平台期、预算限制、表示与优化未可识别性。完成三段评价后停止，不修改RetiPath或启动后续机制/优化。

来源、数据身份、工程检查及环境列表保存于checkpoints/provenance/。运行日志、逐步曲线、选择与refit checkpoints保存在checkpoints/；根目录仅保留要求的交付文件。协议冻结后只允许记录必要的工程纠错，不改变科学定义或预算。
"""


def notes(record: dict) -> str:
    link = f"https://github.com/open-retina/open-retina/blob/{AUTHOR_SHA}"
    return f"""# OpenRetina任务适配说明

模型名为“OpenRetina作者实现的任务适配版本”，固定作者版本 `{AUTHOR_SHA}`。核心与Readout直接导入该仓库源码，没有重新实现小CNN。具体依据：[marmoset配置]({link}/configs/vystrcilova_2024_nm_cnn.yaml)、[Core]({link}/openretina/modules/core/base_core.py)、[Gaussian Readout]({link}/openretina/modules/readout/multi_readout.py)、[原sum Poisson loss]({link}/openretina/modules/losses/poisson.py)。这不是原论文成绩复现，也没有使用其他物种/细胞的预训练输出。

- 输入[B,T,289]按原flatten次序还原[B,1,T,17,17]，数值/空间坐标/Weber标定不变，无resize、额外通道、frame-rate resampling或development normalization。三个原spatial kernels=21/15/11大于此输入所允许的valid卷积，因此分别设置每层空间padding10/7/5，原kernel宽度和参数保留；场外零代表Weber基线。未给RetiPath换输入。
- 时间只在sequence左侧补30个零，三层valid时间卷积总共缩短30bins，输出[B,64,T,17,17]与原150bins一一对齐；第t个输出依赖x[max(0,t−30):t+1]。150Hz下31个输入bins，最远lag200ms。第30bin已具有完整31-frame窗口，不删除任何scored bin。每sequence独立reset。官方BatchNorm训练时沿batch/time/space估计统计；评价使用冻结running stats，不读取评价序列未来帧来归一化。
- 原MultiSampledGaussianReadout先用PointGaussianReadout得到feature加权和+bias，再做softplus。此接口的`nonlinearity_function`设为Identity，明确在softplus之前取drive d。保留该head唯一的bias；没有另加adapter bias，也没有对正发放率套sigmoid。初始bias用当次允许训练区间occupancy的logit替代Poisson语义初始化。
- h[t]=(1−a)y[t−1]+a h[t−1]，a=exp(−(1000/150)/30)，h[0]=0。复用项目fixed_one_bin_history_state，单位ms，只输入occupancy历史，和RetiPath同一严格过去30ms指数特征。z=d−softplus(w_h)h；p=sigmoid(z)。w_h逐cell可学，初始系数.02，约束为抑制方向；历史观测不作为未来或当前target shortcut。RetiPath自身history参数完全不变。
- sum Bernoulli BCE只在原mask=true位置计算。计算L=sum_cell,sum_valid BCE(z,y)+R_core+sum_cell R_readout；作者所有gamma以及AdamW decay保留。评价另报float64 nats/scored-bin mean，不把正则加进NLL，不与Poisson/count NLL混比。
- 共享Core输入可复用的前提是逐位相同movie片段；每cell通过自己的head预测，loss逐recording/trial独立计算。source IDs、trial indices及mask保留；没有拼造22维同时记录。仅在本任务内共享Core，RetiPath仍为每cell独立训练，这一先验差别不能隐去。
- 官方Gaussian采样、full covariance、feature L1等原样保留。CUDA grid_sample backward不保证确定性，所以Core在GPU、原Readout在CPU运行，传递梯度而不detach。整个模型两次同seed两步更新的完整state逐位相同；作者文件未改动。readout位置的[-1,1]投影是官方已有行为；在每次optimizer后显式完成，确保冻结评价不再改变权重。

参数共{record['model_parameters']}，其中共享Core {record['shared_core_parameters']}；其余为逐cell作者Readout和history系数。参数比不用于宣传效率。环境隔离在work/retipath_author_baseline/venv，仅只读复用已有GPU PyTorch；新增依赖安装于该venv，未升级原环境。环境列表、作者版本和文件哈希见checkpoints/provenance/。
"""


if __name__ == "__main__":
    main()
