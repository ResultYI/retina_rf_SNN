from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import csv
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from data.retinal_recording import RealSequenceSplit
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.local_bc_nonlinearity import LocalBCNonlinearRetina, LocalBCPlacement
from models.mechanistic_retina.retipath_spatial_ei import (
    SpatialEIBackend, SpatialEIRetiPath, spatial_ei_parameters,
)
from training.mechanistic_retina.losses import expected_bernoulli_nll

POP = ROOT / 'output/experiments/local_bc_subunit_nonlinearity_population_20260907'
CONTEXT = ROOT / '.omo/evidence/context-rf-gain-20260907'
THRESHOLDS = ROOT / 'output/experiments/context_dependent_rf_gain_20260907/context_thresholds.csv'
CELLS = ('69#4', '67#6', '68#4', '67#4')
SEED = 20260913
BACKENDS = {'A': SpatialEIBackend.RETIPATH, 'B': SpatialEIBackend.CURRENT,
            'C': SpatialEIBackend.CONDUCTANCE}


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def save_json(path: Path, data: dict, *, exclusive: bool = False) -> None:
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def sources(cell_id: str) -> dict:
    manifest = load_json(POP / 'evidence_manifest.json')
    lock = load_json(POP / 'population_training_lock.json')
    cell = next(c for c in lock['cells'] if c['cell_id'] == cell_id)
    safe = cell_id.replace('#', '_')
    matches = [k for k in manifest['artifacts_sha256'] if k == f'checkpoints/{safe}/M2/best.pt']
    if len(matches) != 1:
        raise ValueError(f'unresolved final manifest checkpoint: {cell_id}')
    key = matches[0]
    paths = {'checkpoint': POP / key, 'metadata': ROOT / cell['checkpoint_path'],
             'input': POP / cell['input_path'],
             'replay': POP / str(Path(key).with_name('best_development_predictions.pt')),
             'context': CONTEXT / f'{safe}.npz'}
    expected = {'checkpoint': manifest['artifacts_sha256'][key],
                'metadata': cell['checkpoint_sha256'], 'input': cell['input_sha256'],
                'replay': manifest['artifacts_sha256'][str(Path(key).with_name('best_development_predictions.pt')).replace('\\', '/')]}
    for name, digest in expected.items():
        if sha(paths[name]) != digest:
            raise ValueError(f'source hash mismatch: {cell_id} {name}')
    return {'cell': cell, 'paths': {k: str(v) for k, v in paths.items()},
            'hashes': {k: sha(v) for k, v in paths.items()}}


def load_inputs(ref: dict) -> tuple[dict, dict, RealSequenceSplit, RealSequenceSplit]:
    cp = torch.load(ref['paths']['checkpoint'], weights_only=True)
    metadata = torch.load(ref['paths']['metadata'], weights_only=True)
    raw = torch.load(ref['paths']['input'], weights_only=True)
    if set(raw) != {'train', 'development'}:
        raise ValueError('DATA MISMATCH: cached splits contain unpermitted interval')
    splits = tuple(RealSequenceSplit(**raw[s]) for s in ('train', 'development'))
    for split, (lo, hi) in zip(splits, ((0, 2400), (2400, 3000)), strict=True):
        if split.cone_drive.shape[1:] != (150, 289):
            raise ValueError('DATA MISMATCH: input shape')
        for source_id in split.source_image_ids:
            match = re.search(r'live-frames-(\d+)-(\d+)-trial-', source_id)
            if match is None or not lo <= int(match[1]) <= int(match[2]) < hi:
                raise ValueError('DATA MISMATCH: source interval')
        if not torch.equal(split.spike_events, (split.spike_counts > 0).to(split.spike_events)):
            raise ValueError('DATA MISMATCH: occupancy')
        expected = torch.ones_like(split.valid_mask)
        expected[:, :30] = False
        if not torch.equal(expected, split.valid_mask):
            raise ValueError('DATA MISMATCH: scoring mask')
    if cp['cell_id'] != ref['cell']['cell_id'] or cp['variant'] != 'M2':
        raise ValueError('DATA MISMATCH: checkpoint identity')
    return cp, metadata, splits[0], splits[1]


def model_args(metadata: dict) -> tuple:
    return (MechanisticRetinaConfig(**metadata['model_config']), metadata['cone_positions_degs'],
            metadata['cell_positions_degs'], tuple(metadata['cell_types']), tuple(metadata['polarities']))


def make_model(metadata: dict, state: dict, variant: str, rms: dict) -> SpatialEIRetiPath:
    torch.manual_seed(SEED)
    model = SpatialEIRetiPath(*model_args(metadata), backend=BACKENDS[variant],
                             rms_e=torch.tensor(rms['e']), rms_i=torch.tensor(rms['i']))
    full = model.state_dict()
    full.update(state)
    model.load_state_dict(full, strict=True)
    return model


def evaluate(model: LocalBCNonlinearRetina, split: RealSequenceSplit) -> tuple[float, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(split.cone_drive[s:s+8], observed_counts=split.spike_events[s:s+8]).logits
                            for s in range(0, len(split.cone_drive), 8)])
        value = expected_bernoulli_nll(logits.double(), split.spike_events.double(), split.valid_mask)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('nonfinite evaluation logits')
    return float(value), logits


def context_values(split: RealSequenceSplit, support: torch.Tensor) -> np.ndarray:
    x = split.cone_drive.double().numpy()[..., support.numpy()]
    c = np.sqrt(np.mean((x - x.mean(-1, keepdims=True)) ** 2, axis=-1))
    cum = np.concatenate([np.zeros((len(x), 1)), np.cumsum(c, axis=1)], axis=1)
    out = np.full(c.shape, np.nan)
    out[:, 45:] = (cum[:, 45:-1] - cum[:, :-46]) / 45
    return out


def schedule(n: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    return torch.stack([torch.randint(n, (4,), generator=generator) for _ in range(1000)])


def compare(left: torch.Tensor, right: torch.Tensor, *, rtol: float = 1e-5, atol: float = 1e-6) -> dict:
    torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
    return {'max_abs_error': float((left - right).abs().max()),
            'reference_abs_max': float(right.abs().max()), 'bitwise': torch.equal(left, right),
            'rtol': rtol, 'atol': atol}


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    (out / 'checkpoints').mkdir()
    torch.set_num_threads(1)
    thresholds = {r['cell_id']: r for r in csv.DictReader(THRESHOLDS.open())}
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
                'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'updates': 1000, 'optimizer': {'name': 'Adam', 'lr': 0.003, 'weight_decay': 0,
                'batch': 4, 'betas': [0.9, 0.999], 'eps': 1e-8, 'scheduler': None, 'clip_norm': 5},
                'eval_steps': list(range(0, 1001, 25)), 'train_seconds': [0, 16], 'development_seconds': [16, 20],
                'sampling_hz': 150, 'sequence_bins': 150, 'warmup_bins': 30, 'cells': {},
                'rf_checkpoints': ['final', 'best'], 'rf_cap_per_context': 20,
                'spatial_radius_grid_deg': np.linspace(0, 1.5, 301).tolist(),
                'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
                 [POP / 'evidence_manifest.json', POP / 'population_training_lock.json', THRESHOLDS]}}
    checks = {}
    for cell_id in CELLS:
        ref = sources(cell_id)
        cp, metadata, train, dev = load_inputs(ref)
        args = model_args(metadata)
        old = LocalBCNonlinearRetina(*args, placement=LocalBCPlacement.POST_POOL)
        old.load_state_dict(cp['model'], strict=True)
        default = SpatialEIRetiPath(*args)
        default.load_state_dict(cp['model'], strict=True)
        old_nll, old_logits = evaluate(old, dev)
        _, default_logits = evaluate(default, dev)
        replay = torch.load(ref['paths']['replay'], weights_only=True)
        for name in ('target', 'valid_mask'):
            target = dev.spike_events if name == 'target' else dev.valid_mask
            if not torch.equal(replay[name], target):
                raise ValueError('DATA MISMATCH: saved development target/mask')
        if replay['source_image_ids'] != dev.source_image_ids or replay['trial_indices'] != dev.trial_indices:
            raise ValueError('DATA MISMATCH: saved development alignment')
        results = {'default_replay': compare(default_logits, old_logits, rtol=0, atol=0),
                   'saved_checkpoint_replay': compare(old_logits, replay['logits'], rtol=0, atol=0),
                   'start_dev_nll': old_nll, 'source_best_step': cp['step']}
        square_e, square_i, count = torch.zeros(1, 2, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64), 0
        with torch.no_grad():
            for start in range(0, len(train.cone_drive), 8):
                parts = default.spatial_components(train.cone_drive[start:start+8])
                square_e += parts.u_e.double().square().sum((0, 1))
                square_i += parts.u_i.double().square().sum((0, 1))
                count += parts.u_e.shape[0] * parts.u_e.shape[1]
            rms = {'e': (square_e / count).sqrt().float().tolist(),
                   'i': (square_i / count).sqrt().float().tolist(), 'bins_including_warmup': count}
            x, h = train.cone_drive[:4], train.spike_events[:4]
            parts = default.spatial_components(x)
            original = old(x, observed_counts=h)
            results['BC_direct_sum'] = compare(parts.direct.sum(-2), original.bc_direct_presynaptic)
            results['BC_broad_sum'] = compare(parts.broad.sum(-2), original.bc_broad_presynaptic)
            results['AC_state_sum'] = compare(parts.ac_states.sum(-2), torch.stack((original.amacrine_local_state, original.amacrine_transient_state), -1))
            results['pathway_current_sum'] = compare(parts.currents.sum(-2), torch.stack((original.bc_sustained_current, original.bc_transient_current, original.amacrine_local_current, original.amacrine_transient_current), -1))
            results['components_shape'] = list(parts.direct.shape)
        bc_models = [make_model(metadata, cp['model'], v, rms) for v in ('B', 'C')]
        assert all(torch.equal(v, bc_models[1].state_dict()[k]) for k, v in bc_models[0].state_dict().items())
        for variant, model in zip(('B', 'C'), bc_models, strict=True):
            params = spatial_ei_parameters(model)
            assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
            model.zero_grad(set_to_none=True)
            output = model(x, observed_counts=h)
            expected_bernoulli_nll(output.logits, h, train.valid_mask[:4]).backward()
            gradients = {name: {'numel': p.numel(), 'optimizer_listed': id(p) in {id(q) for q in params},
                         'gradient_present': p.grad is not None,
                         'gradient_finite': p.grad is not None and bool(torch.isfinite(p.grad).all()),
                         'gradient_abs_max': 0.0 if p.grad is None or not p.numel() else float(p.grad.abs().max())}
                         for name, p in model.named_parameters() if p.requires_grad}
            assert all(v['gradient_present'] and v['gradient_finite'] for v in gradients.values())
            assert all(gradients[k]['gradient_abs_max'] > 0 for k in gradients if k.startswith('spatial_ei.'))
            x_grad = x[:1].clone().requires_grad_()
            y_grad = h[:1].clone().requires_grad_()
            logits = model(x_grad, observed_counts=y_grad).logits
            dx, dy = torch.autograd.grad(logits[0, 80, 0], (x_grad, y_grad))
            assert torch.count_nonzero(dx[:, 81:]) == 0 and torch.count_nonzero(dy[:, 80:]) == 0
            assert bool(torch.isfinite(dx).all()) and float(dx[:, :65].abs().max()) > 0
            altered = x[:1].clone(); altered[:, 81:] += 3
            future = model(altered, observed_counts=h[:1]).logits
            prefix = model(x[:1, :81], observed_counts=h[:1, :81]).logits
            results[variant] = {'trainable_parameters': sum(p.numel() for p in params), 'gradients': gradients,
                              'future_stimulus': compare(future[:, :81], logits.detach()[:, :81], rtol=0, atol=0),
                              'full_prefix': compare(prefix, logits.detach()[:, :81]),
                              'future_stimulus_jacobian_max': float(dx[:, 81:].abs().max()),
                              'current_future_history_jacobian_max': float(dy[:, 80:].abs().max()),
                              'past_16_lag_gradient_max': float(dx[:, :65].abs().max())}
        assert results['B']['trainable_parameters'] == results['C']['trainable_parameters']
        support = default.feature_bank.bc_support[0] > 0
        row = thresholds[cell_id]
        assert row['input_sha256'] == ref['hashes']['input'] and row['support_sha256'] == tensor_sha(support)
        cdev = context_values(dev, support)
        ctr = context_values(train, support)
        q = np.quantile(ctr[np.isfinite(ctr) & train.valid_mask[..., 0].numpy().astype(bool)], [.2, .8])
        np.testing.assert_allclose(q, [float(row['LOW_threshold']), float(row['HIGH_threshold'])], rtol=0, atol=1e-14)
        selections = {}
        with np.load(ref['paths']['context'], allow_pickle=False) as archive:
            for label in ('LOW', 'HIGH'):
                indices = archive[f'indices_{label}']
                indices = indices[np.linspace(0, len(indices)-1, min(20, len(indices))).astype(np.int64)]
                assert len(indices) > 0
                selected_values = cdev[indices[:, 0], indices[:, 1]]
                assert bool(np.all(selected_values <= q[0] if label == 'LOW' else selected_values >= q[1]))
                assert bool(dev.valid_mask[indices[:, 0], indices[:, 1]].all())
                selections[label] = indices.tolist()
        ref.update({'rms': rms, 'selected_observations': selections, 'context_thresholds': q.tolist(),
                    'model_config': metadata['model_config'], 'schedule_sha256': tensor_sha(schedule(len(train.cone_drive)))})
        protocol['cells'][cell_id] = ref
        checks[cell_id] = results
        print('prepared', cell_id, 'B/C parameters', results['B']['trainable_parameters'], flush=True)
    source_paths = {Path(m.__file__).resolve() for m in tuple(sys.modules.values())
                    if getattr(m, '__file__', None) and '/models/mechanistic_retina/' in str(Path(m.__file__)).replace('\\', '/')}
    source_paths.update([Path(__file__), ROOT / 'tests/test_retipath_spatial_ei.py'])
    protocol['source_hashes'].update({str(p.relative_to(ROOT)): sha(p) for p in source_paths if p.suffix == '.py'})
    unit = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                           'tests/test_retipath_spatial_ei.py', 'tests/test_local_bc_nonlinearity.py'],
                          cwd=ROOT, text=True, capture_output=True)
    assert unit.returncode == 0, unit.stdout + unit.stderr
    text = '''# RetiPath 空间兴奋/抑制整合 development pilot

训练前冻结。仅四个指定细胞、12 fits、每 fit 1000 updates。A 为冻结最终 M2 best 的原 RetiPath continuation；B/C 为两个已有空间 basis 的共同表示，仅膜整合分别为静息点线性电流/电导方程。主要比较 C−B；B/C−A 同时包含表示和后端变化。独立检验 = NONE。

文献依据：Latimer, Rieke & Pillow (2019), https://elifesciences.org/articles/47012 ，及作者 https://github.com/pillowlab/CBEM 。电导型膜方程和正值输入映射有文献依据。本轮 EL=0, EE=1, EI=−1/3, gL=1 是用户固定的有效归一化尺度；对应用户给定作者代码 −60/0/−80 mV 的相对位置。作者代码数值本次远程读取未独立核实（UNVERIFIED），不影响按明确指定数值实现，不声称恢复真实电压/电导。K=2、共同整流斜率、RMS、线性化 B、等权输出和训练预算均为本轮设计，并非照搬 CBEM。

BC 只汇聚时间 basis，完整分支 S 决定共同斜率 1/alpha。direct/broad 为 [B,T,N,2,2]。Broad 各 K 经原 AC delay/filter/gate，保留 cell gain；uE=sum direct current，uI=−sum AC current，禁止 abs。固定 RMS 使用初始 checkpoint 的所有训练输入 bin（含原 warmup、按原重复次数），float64 累加后保存 float32，development 不参与。负驱动表示基线减少。

gE/I=softplus(log(expm1(1))+exp(log_aE/I)*uE/I/sE/I)。aE/aI 跨 K 共享，output_scale 均初值1、正值可学习。G0=3,V0=2/9,Cm=3*tau_ref；tau_ref=起点固定 membrane_tau_ms；dt/tau/Cm 使用 ms。C: G=1+gE+gI,Vinf=(gE−gI/3)/G。B: G=3,Vinf=V0+[(gE−1)(1−V0)+(gI−1)(−1/3−V0)]/3。逐 bin V←V+(1−exp(−dt*G/Cm))*(Vinf−V)，每 sequence 从 V0 初始化。膜输出=output_scale*(mean_K(V)−V0)。保留原 adaptation、threshold/logit_slope、bias、strictly-past observed history；不经过旧 divisive normalization/membrane。所有原可学习量继续学习、原固定量仍固定；alpha bounds [0.05,2]。

train [0,16), development [16,20)；仅访问来源缓存两 split，无其他 spike 数据。150Hz Bernoulli occupancy，独立150-bin sequence，30 warmup/120 scored，同 front-end/alignment/observed history/mask。Adam lr=.003,wd=0,batch=4,clip=5，无scheduler，单一seed，匹配schedule。step0 和每25步评估 train/dev。主报告固定 step1000，另报最低 dev 的预定评估点（并列保留最早）。新 Adam 状态从零开始，控制 continuation 额外训练。保存 start/final/best；不追加预算或按结果 restart。

RF 复用既有 train q20/q80、45个严格过去帧空间对比度 context、既有 development 索引；从已保存至多100个观测按 floor(linspace(0,n−1,min(20,n))) 子采样，不搜索新 context。A/B/C 使用完全相同观测。final 为主要 RF，best 为补充。逐 observation 使用从 sequence reset 到目标 bin 的完整因果 prefix logit Jacobian，固定 observed spike history；没有16-lag截断、状态 detach 或跨sequence历史。G=||J||2；P(x)=sum_lag J²/sum_lag,x J²。每观测先单位化，context 内等权平均 P，再计算空间重心和围绕自身重心的二阶矩半径（degree）。同时报告逐观测半径 median、gain median 和归一化 profile 的 TV=0.5*sum|P_HIGH−P_LOW|。移除重心后用相对自身重心的径向累计分布 F(r) 比较，固定 r=0..1.5 degree、301点，报告最大差及CDF差积分（degree）。径向指标平移不变；有限网格离散性单独保留，非阈值圈面积。P 原数组、每观测指标和索引嵌入 spatial CSV，便于重算。

报告逐cell差值以及mean/median/wins，不bootstrap。训练曲线、通道驱动/电导范围与变异、映射饱和比例、两模式输出相关、边界命中和末段训练走势仅作诊断，不用于调参或新增损失。不把预算耗尽称充分收敛，不把context关联称因果适应，RF变化不自动等于正确。

## 冻结实际配置与必要来源 hash

```json
'''
    (out / 'PROTOCOL.md').write_text(text + json.dumps(protocol, ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
    save_json(out / 'correctness.json', {'status': 'VERIFIED', 'protocol_sha256': sha(out / 'PROTOCOL.md'),
              'protocol': protocol, 'checks': checks, 'unit_tests': unit.stdout.strip(),
              'training_complete': False}, exclusive=True)
    print('PREPARED', out, flush=True)


def checkpoint_payload(model: SpatialEIRetiPath, optimizer: torch.optim.Optimizer, meta: dict,
                       step: int, train_nll: float, dev_nll: float, ref: dict, protocol_hash: str) -> dict:
    return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
            'model_config': meta['model_config'], 'cone_positions_degs': meta['cone_positions_degs'],
            'cell_positions_degs': meta['cell_positions_degs'], 'cell_types': meta['cell_types'],
            'polarities': meta['polarities'], 'backend': model.backend.value, 'seed': SEED,
            'train_nll': train_nll, 'dev_nll': dev_nll, 'source': ref,
            'protocol_sha256': protocol_hash, 'torch_rng_state': torch.get_rng_state()}


def train_cell(out_string: str, cell_id: str, events) -> None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        out = Path(out_string)
        correctness = load_json(out / 'correctness.json')
        ref = correctness['protocol']['cells'][cell_id]
        cp, metadata, train, dev = load_inputs(ref)
        batches = schedule(len(train.cone_drive))
        assert tensor_sha(batches) == ref['schedule_sha256']
        for variant in ('A', 'B', 'C'):
            destination = out / 'checkpoints' / cell_id.replace('#', '_') / variant
            destination.mkdir(parents=True, exist_ok=False)
            model = make_model(metadata, cp['model'], variant, ref['rms'])
            params = spatial_ei_parameters(model)
            optimizer = torch.optim.Adam(params, lr=.003, weight_decay=0)
            fixed = {k: v.clone() for k, v in model.named_buffers()}
            initial = {k: v.detach().clone() for k, v in model.named_parameters()}
            best, best_step = float('inf'), 0
            gradient_seen = {k: False for k, p in model.named_parameters() if p.requires_grad}
            curve = []
            started = time.perf_counter()
            for step in range(1001):
                batch_loss, grad_norm = '', ''
                if step:
                    model.train()
                    idx = batches[step-1]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(train.cone_drive[idx], observed_counts=train.spike_events[idx]).logits
                    loss = expected_bernoulli_nll(logits, train.spike_events[idx], train.valid_mask[idx])
                    loss.backward()
                    for name, p in model.named_parameters():
                        if p.requires_grad:
                            assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
                            gradient_seen[name] |= bool(torch.count_nonzero(p.grad))
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 5, error_if_nonfinite=True))
                    optimizer.step()
                    model.project_mechanism_parameters()
                    assert all(bool(torch.isfinite(p).all()) for p in params)
                    batch_loss = float(loss.detach())
                if step % 25:
                    continue
                train_nll, _ = evaluate(model, train)
                dev_nll, _ = evaluate(model, dev)
                assert all(torch.equal(v, dict(model.named_buffers())[k]) for k, v in fixed.items())
                improved = dev_nll < best
                if improved:
                    best, best_step = dev_nll, step
                row = {'cell_id': cell_id, 'condition': variant, 'step': step,
                       'train_nll': train_nll, 'dev_nll': dev_nll, 'sampled_batch_nll': batch_loss,
                       'gradient_norm_before_clip': grad_norm, 'best_dev_nll': best, 'best_step': best_step,
                       'alpha': float(model.alpha.detach()), 'elapsed_seconds': time.perf_counter()-started}
                if model.spatial_ei is not None:
                    row.update({'aE': float(model.spatial_ei.log_a_e.detach().exp()),
                                'aI': float(model.spatial_ei.log_a_i.detach().exp()),
                                'output_scale': float(model.spatial_ei.log_output_scale.detach().exp())})
                else:
                    row.update({'aE': '', 'aI': '', 'output_scale': ''})
                curve.append(row)
                events.put(('row', row))
                if step in (0, 1000) or improved:
                    payload = checkpoint_payload(model, optimizer, metadata, step, train_nll, dev_nll,
                                                 ref, correctness['protocol_sha256'])
                    payload.update({'cell_id': cell_id, 'condition': variant,
                                    'schedule': batches, 'training_curve': curve.copy(),
                                    'gradient_nonzero_seen': gradient_seen.copy(),
                                    'parameters_actually_changed': {k: not torch.equal(v, dict(model.named_parameters())[k]) for k,v in initial.items()},
                                    'fixed_buffers_unchanged': True})
                    if step == 0:
                        torch.save(payload, destination / 'start.pt')
                    if improved:
                        torch.save(payload, destination / 'best.pt')
                    if step == 1000:
                        torch.save(payload, destination / 'final.pt')
                if step % 250 == 0:
                    print(cell_id, variant, step, 'dev', round(dev_nll, 7), flush=True)
        events.put(('done', cell_id))
    except Exception:
        events.put(('error', cell_id + '\n' + traceback.format_exc()))
        raise


def train_all(out: Path) -> None:
    data = load_json(out / 'correctness.json')
    assert data['status'] == 'VERIFIED' and sha(out / 'PROTOCOL.md') == data['protocol_sha256']
    for path, digest in data['protocol']['source_hashes'].items():
        assert sha(ROOT / path) == digest, path
    for ref in data['protocol']['cells'].values():
        for key, digest in ref['hashes'].items():
            assert sha(Path(ref['paths'][key])) == digest
    fields = ['cell_id', 'condition', 'step', 'train_nll', 'dev_nll', 'sampled_batch_nll',
              'gradient_norm_before_clip', 'best_dev_nll', 'best_step', 'alpha', 'elapsed_seconds',
              'aE', 'aI', 'output_scale']
    ctx = mp.get_context('spawn')
    events = ctx.Queue()
    workers = [ctx.Process(target=train_cell, args=(str(out), cell_id, events)) for cell_id in CELLS]
    with (out / 'training.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); stream.flush()
        for worker in workers:
            worker.start()
        done, errors = set(), []
        while len(done) < len(CELLS):
            try:
                tag, value = events.get(timeout=2)
            except queue.Empty:
                if all(not worker.is_alive() for worker in workers):
                    break
                continue
            if tag == 'row':
                writer.writerow(value); stream.flush()
            elif tag == 'done':
                done.add(value)
            else:
                errors.append(value)
                break
        for worker in workers:
            worker.join()
        if errors or len(done) != 4 or any(worker.exitcode != 0 for worker in workers):
            raise RuntimeError(str(errors) + ' completed cells: ' + str(done))
    rows = list(csv.DictReader((out / 'training.csv').open()))
    assert len(rows) == 12 * 41
    data['training_complete'] = True
    data['completed_fits'] = 12
    data['optimizer_updates'] = 12000
    save_json(out / 'correctness.json', data)
    print('TRAINING COMPLETE: 12 fits, 12000 optimizer updates', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'train'))
    parser.add_argument('--out', type=Path, default=ROOT / 'output/experiments/retipath_spatial_ei_pilot')
    args = parser.parse_args()
    if args.mode == 'prepare':
        out = args.out
        if out.exists():
            out = out / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        prepare(out)
    else:
        train_all(args.out)


if __name__ == '__main__':
    main()
