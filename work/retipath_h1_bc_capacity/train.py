from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from common import (OUT, ROOT, SEEDS, build, capacity_parameters, compute_rms, stage_data,
                    load_json, save_json, sha, tensor_sha, minibatches, evaluate, SelectionStop,
                    atomic_torch_save, digest_state, write_csv, verify_sources)
from training.mechanistic_retina.losses import expected_bernoulli_nll


def probe(model, data, mask) -> dict:
    with torch.no_grad():
        h1 = model.h1(data.cone_drive[:4], amplitude=model.gates.values(frozenset()).h1)
        activations = model.input_basis(h1.modulated_cones / model.h1_rms)[mask[:4]]
        values = activations.reshape(-1, 4).double().numpy()
        return {'basis_rms': np.sqrt(np.square(values).mean(0)).tolist(),
                'basis_near_zero_fraction': (np.abs(values) < .01).mean(0).tolist(),
                'basis_correlation': np.corrcoef(values.T).tolist(),
                'basis_weights': model.basis_weights.tolist()}


def fit_job(out_string: str, cell: str, seed: int, condition: str, stage: str) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(out_string)
    protocol = load_json(out / 'checkpoints/protocol.json')
    ref = protocol['cells'][cell]
    folder = out / 'checkpoints' / cell.replace('#', '_') / str(seed) / condition
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f'{stage}.pt'
    if destination.exists():
        cp = torch.load(destination, weights_only=True, mmap=True)
        assert cp['protocol_sha256'] == protocol['protocol_sha256']
        return {'cell_id': cell, 'seed': seed, 'condition': condition, 'stage': stage,
                'step': cp['step'], 'best_step': cp['best_step']}
    data, validation, input_mask = stage_data(ref, stage)
    metadata = torch.load(ref['metadata_path'], weights_only=True, mmap=True)
    h1_rms = ref['h1_rms'][stage]
    model = build(metadata, seed, condition, h1_rms)
    rms = compute_rms(model, data, input_mask)
    model = build(metadata, seed, condition, h1_rms, rms)
    params = capacity_parameters(model)
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    assert sum(p.numel() for p in params) == 45 and not list(model.input_basis.parameters())
    optimizer = torch.optim.Adam(params, lr=.003, weight_decay=0)
    batches = minibatches(len(data.cone_drive), seed)
    schedule_hash = tensor_sha(batches)
    assert schedule_hash == ref['schedule_hashes'][str(seed)][stage]
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    fixed = {k: v.clone() for k, v in model.named_buffers()}
    maximum = 3000 if stage == 'inner' else int(torch.load(folder / 'inner.pt', weights_only=True)['best_step'])
    status, best_state, curve, start_step = None, None, [], 0
    gradient_seen = {n: False for n, p in model.named_parameters() if p.requires_grad}
    latest = folder / f'{stage}_latest.pt'
    start = time.perf_counter()
    if latest.exists():
        cp = torch.load(latest, weights_only=True)
        assert cp['protocol_sha256'] == protocol['protocol_sha256'] and cp['schedule_sha256'] == schedule_hash
        model.load_state_dict(cp['model'], strict=True)
        optimizer.load_state_dict(cp['optimizer'])
        start_step, curve, best_state = cp['step'] + 1, cp['curve'], cp['best_state']
        status = SelectionStop(**cp['selection_status']) if cp['selection_status'] else None
        gradient_seen = cp['gradient_nonzero_seen']
        start -= cp['elapsed_seconds']
        step = cp['step']
        if status and status.stopped(step):
            start_step = maximum + 1
    for step in range(start_step, maximum + 1):
        batch_loss, grad_norm = None, None
        if step:
            model.train()
            idx = batches[step - 1]
            optimizer.zero_grad(set_to_none=True)
            logits = model(data.cone_drive[idx], observed_counts=data.spike_events[idx]).logits
            loss = expected_bernoulli_nll(logits, data.spike_events[idx], data.valid_mask[idx])
            loss.backward()
            for name, p in model.named_parameters():
                if p.requires_grad:
                    if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                        raise ValueError(f'Absent/nonfinite gradient: {cell}/{seed}/{condition}/{name}')
                    gradient_seen[name] |= bool(torch.count_nonzero(p.grad))
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True))
            optimizer.step()
            model.project_mechanism_parameters()
            batch_loss = float(loss.detach())
        if step % 25 and step != maximum:
            continue
        inner_nll = evaluate(model, validation)[0] if validation is not None else None
        if validation is not None:
            status = SelectionStop(inner_nll, 0, inner_nll, 0) if status is None else status.observe(inner_nll, step)
            if status.best_step == step:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        full_nll = evaluate(model, data)[0] if step in (0, maximum) else None
        with torch.no_grad():
            logits = model(data.cone_drive[:4], observed_counts=data.spike_events[:4]).logits
            fixed_nll = float(expected_bernoulli_nll(logits.double(), data.spike_events[:4].double(), data.valid_mask[:4]))
        assert all(torch.equal(v, dict(model.named_buffers())[k]) for k, v in fixed.items())
        curve.append({'cell_id': cell, 'group': ref['group'], 'seed': seed, 'condition': condition,
                      'phase': stage, 'step': step, 'sampled_train_batch_nll': batch_loss,
                      'fixed_train_batch_nll': fixed_nll, 'inner_validation_nll': inner_nll,
                      'full_train_nll': full_nll, 'gradient_norm_before_clip': grad_norm,
                      'best_step': status.best_step if status else maximum,
                      'alpha': float(model.alpha.detach()), 'elapsed_seconds': time.perf_counter() - start,
                      **probe(model, data, input_mask)})
        cp = {'cell_id': cell, 'seed': seed, 'condition': condition, 'phase': stage,
              'step': step, 'best_step': status.best_step if status else maximum,
              'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'curve': curve,
              'best_state': best_state, 'selection_status': asdict(status) if status else None,
              'rms': rms, 'h1_rms': h1_rms, 'initial_state': initial_state,
              'schedule_sha256': schedule_hash, 'protocol_sha256': protocol['protocol_sha256'],
              'gradient_nonzero_seen': gradient_seen, 'elapsed_seconds': time.perf_counter() - start,
              **{k: metadata[k] for k in ('model_config', 'cone_positions_degs', 'cell_positions_degs', 'cell_types', 'polarities')},
              'backend': model.backend.value, 'trainable_parameters': 45, 'fixed_buffers_unchanged': True}
        atomic_torch_save(cp, latest)
        if status is not None and status.stopped(step):
            break
    if validation is not None:
        model.load_state_dict(best_state, strict=True)
        best_nll = evaluate(model, validation)[0]
        assert best_nll == status.best_nll
        cp['best_replay_inner_nll'] = best_nll
        cp['model_stage'], cp['optimizer_state_stage'] = 'inner_best', 'inner_stop'
    else:
        assert step == maximum
        cp['refit_train_nll'] = evaluate(model, data)[0]
        cp['model_stage'] = cp['optimizer_state_stage'] = 'refit_final'
    cp['model'] = {k: v.clone() for k, v in model.state_dict().items()}
    cp['actually_changed'] = {k: not torch.equal(initial_state[k], p.detach()) for k, p in model.named_parameters() if p.requires_grad}
    cp['optimizer_steps'] = sorted({int(s['step']) for s in optimizer.state.values()})
    assert cp['optimizer_steps'] == ([step] if step else [])
    atomic_torch_save(cp, destination)
    return {'cell_id': cell, 'seed': seed, 'condition': condition, 'stage': stage,
            'step': step, 'best_step': cp['best_step'], 'elapsed_seconds': cp['elapsed_seconds']}


def combine(out: Path, protocol: dict) -> None:
    rows = []
    for cell, ref in protocol['cells'].items():
        for seed in SEEDS:
            for stage in ('inner', 'refit'):
                old = Path(ref['A'][str(seed)]['path']).parent / f'{stage}.pt'
                rows.extend(dict(r, condition='A', group=ref['group'], reused_frozen_A=True)
                            for r in torch.load(old, weights_only=True, mmap=True)['curve'])
    for stage in ('inner', 'refit'):
        for p in sorted((out / 'checkpoints').glob(f'*/*/*/{stage}.pt')):
            rows.extend(dict(r, reused_frozen_A=False) for r in torch.load(p, weights_only=True, mmap=True)['curve'])
    write_csv(out / 'training.csv', rows)


def guarded_fit_job(*args):
    try:
        return fit_job(*args)
    except Exception:
        traceback.print_exc()
        raise


def run(out: Path, workers: int) -> None:
    protocol = load_json(out / 'checkpoints/protocol.json')
    assert sha(out / 'PROTOCOL.md') == protocol['protocol_sha256']
    verify_sources(protocol, out)
    jobs = [(cell, seed, condition) for cell in protocol['cells'] for seed in SEEDS for condition in ('B', 'C')]
    for stage in ('inner', 'refit'):
        if stage == 'refit':
            selected = sorted((out / 'checkpoints').glob('*/*/*/inner.pt'))
            assert len(selected) == 132
            save_json(out / 'checkpoints/selection_lock.json',
                      {'protocol_sha256': protocol['protocol_sha256'], 'files': {str(p.relative_to(out)): sha(p) for p in selected}})
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(guarded_fit_job, str(out), *job, stage) for job in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                save_json(out / 'checkpoints/progress.json', {'stage': stage, 'completed': completed, 'total': 132,
                          'latest': result, 'updated_utc': datetime.now(timezone.utc).isoformat()})
                print('PROGRESS', stage, completed, '/132', result, flush=True)
                if completed % 12 == 0:
                    combine(out, protocol)
        assert len(list((out / 'checkpoints').glob(f'*/*/*/{stage}.pt'))) == 132
        combine(out, protocol)
    paths = sorted((out / 'checkpoints').glob('*/*/*/refit.pt'))
    save_json(out / 'checkpoints/training_complete.json', {'inner_fits': 132, 'fresh_refits': 132,
              'reused_A_refits': 66, 'external_evaluation_before_freeze': False,
              'files': {str(p.relative_to(out)): sha(p) for p in paths},
              'completed_utc': datetime.now(timezone.utc).isoformat()})
    print('ALL TRAINING FROZEN', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    run(args.out, args.workers)
