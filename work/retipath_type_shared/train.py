from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from common import (ROOT, DEST, SEEDS, MAX_UPDATES, EVAL_EVERY, metadata_from_registry,
                    build_initial, training_banks, digest, read_json, exclusive_json,
                    batch_loss, evaluate, utc, write_csv, save_checkpoint)
from retipath_phase2_common import SelectionStop


def state_hash(state: dict) -> str:
    digestor = hashlib.sha256()
    for name, value in sorted(state.items()):
        digestor.update(name.encode())
        digestor.update(value.detach().contiguous().numpy().tobytes())
    return digestor.hexdigest()


def fit(seed: int, phase: str, selected: int | None = None) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = DEST / 'checkpoints' / str(seed)
    folder.mkdir(exist_ok=True)
    completion = folder / f'{phase}_complete.json'
    if completion.exists():
        return read_json(completion)
    metadata = torch.load(DEST / 'checkpoints/provenance/metadata.pt', weights_only=True)
    banks, _ = training_banks()
    bank = banks['inner_fit' if phase == 'selection' else 'refit']
    model, rms = build_initial(metadata, seed, phase)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    initial_hash = state_hash(model.state_dict())
    fixed_hash = state_hash(dict(model.named_buffers()))
    params = model.trainable()
    optimizer = torch.optim.Adam(params, lr=.003, weight_decay=0)
    generator = torch.Generator().manual_seed(seed + 1_000_003)
    target = MAX_UPDATES if phase == 'selection' else selected
    if target is None:
        raise ValueError('Frozen selected step required')
    protocol_sha = digest(DEST / 'PROTOCOL.md')
    rows, per_cell, status, best_state = [], [], None, None
    seen = {n: False for n, p in model.named_parameters() if p.requires_grad}
    start_step, prior_seconds = 0, 0.
    latest = folder / f'{phase}_latest.pt'
    if latest.exists():
        cp = torch.load(latest, weights_only=True)
        assert cp['protocol_sha256'] == protocol_sha and cp['initial_state_sha256'] == initial_hash
        model.load_state_dict(cp['model'], strict=True)
        optimizer.load_state_dict(cp['optimizer'])
        generator.set_state(cp['sampler_rng'])
        torch.set_rng_state(cp['torch_rng'])
        rows, per_cell, seen = cp['curve'], cp['per_cell_curve'], cp['gradient_seen']
        status = SelectionStop(**cp['selection_status']) if cp['selection_status'] else None
        best_state = cp['best_state']
        start_step, prior_seconds = cp['step'] + 1, cp['elapsed_seconds']
        if status is not None and status.stopped(cp['step']):
            start_step = target + 1
    start_time = time.perf_counter() - prior_seconds
    with (folder / f'{phase}.log').open('a', encoding='utf-8') as log:
        log.write(json.dumps({'event': 'start_or_exact_resume', 'utc': utc(), 'next_step': start_step}) + '\n')
        log.flush()
        for step in range(start_step, target + 1):
            batch_nll, norm = None, None
            if step:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, _ = batch_loss(model, bank, generator)
                loss.backward()
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                            raise FloatingPointError(n)
                        seen[n] |= bool(torch.count_nonzero(p.grad))
                norm = float(torch.nn.utils.clip_grad_norm_(params, 5, error_if_nonfinite=True))
                optimizer.step()
                model.project()
                batch_nll = float(loss.detach())
            if step % EVAL_EVERY and step != target:
                continue
            train_nll, train_cells, _ = evaluate(model, bank)
            val_nll, val_cells = None, {}
            if phase == 'selection':
                val_nll, val_cells, _ = evaluate(model, banks['inner_validation'])
                status = SelectionStop(val_nll, step, val_nll, step) if status is None else status.observe(val_nll, step)
                if status.best_step == step:
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            assert fixed_hash == state_hash(dict(model.named_buffers()))
            assert all(torch.equal(p, initial[n]) for n, p in model.named_parameters() if not p.requires_grad)
            row = {'seed': seed, 'phase': phase, 'step': step, 'train_nll': train_nll,
                   'inner_validation_nll': val_nll, 'sampled_batch_nll': batch_nll,
                   'gradient_norm': norm, 'selected_step': status.best_step if status else target,
                   'best_inner_nll': status.best_nll if status else None,
                   'elapsed_seconds': time.perf_counter() - start_time}
            rows.append(row)
            per_cell.extend({'seed': seed, 'phase': phase, 'step': step, 'cell_id': c,
                             'train_nll': value, 'inner_validation_nll': val_cells.get(c)}
                            for c, value in train_cells.items())
            cp = {'seed': seed, 'phase': phase, 'step': step, 'selected_step': row['selected_step'],
                  'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rms': rms,
                  'curve': rows, 'per_cell_curve': per_cell, 'selection_status': asdict(status) if status else None,
                  'best_state': best_state, 'sampler_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
                  'gradient_seen': seen, 'protocol_sha256': protocol_sha, 'initial_state_sha256': initial_hash,
                  'fixed_buffer_sha256': fixed_hash, 'elapsed_seconds': row['elapsed_seconds'],
                  'optimizer_steps': sorted({int(s['step']) for s in optimizer.state.values()})}
            save_checkpoint(latest, cp)
            log.write(json.dumps(row) + '\n')
            log.flush()
            if status is not None and status.stopped(step):
                break
    if phase == 'selection':
        model.load_state_dict(best_state, strict=True)
        replay, _, _ = evaluate(model, banks['inner_validation'])
        assert replay == status.best_nll
        cp['model'] = model.state_dict()
        cp['model_step'] = status.best_step
        cp['optimizer_stage'] = 'selection_stop; not used for refit'
    else:
        assert cp['step'] == target
        cp['model_step'] = target
        cp['optimizer_stage'] = 'refit_endpoint'
    cp['actually_changed'] = {n: not torch.equal(p, initial[n]) for n, p in model.named_parameters() if p.requires_grad}
    cp['completed_utc'] = utc()
    destination = folder / ('selected.pt' if phase == 'selection' else 'refit.pt')
    save_checkpoint(destination, cp)
    summary = {'seed': seed, 'phase': phase, 'stop_step': cp['step'], 'selected_step': cp['selected_step'],
               'stop_reason': 'patience' if status and status.stopped(cp['step']) else 'budget' if phase == 'selection' else 'frozen_endpoint',
               'best_inner_nll': status.best_nll if status else None, 'seconds': cp['elapsed_seconds'],
               'checkpoint_sha256': digest(destination), 'completed_utc': cp['completed_utc']}
    exclusive_json(completion, summary)
    return summary


def main() -> None:
    lock = read_json(DEST / 'checkpoints/provenance/pretraining_lock.json')
    assert digest(DEST / 'PROTOCOL.md') == lock['protocol_sha256']
    for path, expected in lock['frozen_sha256'].items():
        assert digest(ROOT / path) == expected, path
    execution = DEST / 'checkpoints/provenance/execution_lock.json'
    if not execution.exists():
        exclusive_json(execution, {'started_utc': utc(), 'worker_count': 3, 'python': __import__('sys').executable,
                                   'torch': torch.__version__, 'protocol_sha256': lock['protocol_sha256']})
    selected = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fit, seed, 'selection'): seed for seed in SEEDS}
        for future in as_completed(futures):
            result = future.result()
            selected[str(result['seed'])] = result
            print(json.dumps(result), flush=True)
    selection_lock = DEST / 'checkpoints/provenance/selection_lock.json'
    if not selection_lock.exists():
        exclusive_json(selection_lock, {'created_utc': utc(), 'selection': selected})
    else:
        assert read_json(selection_lock)['selection'] == selected
    refits = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fit, seed, 'refit', selected[str(seed)]['selected_step']): seed for seed in SEEDS}
        for future in as_completed(futures):
            result = future.result()
            refits[str(result['seed'])] = result
            print(json.dumps(result), flush=True)
    exclusive_json(DEST / 'checkpoints/provenance/evaluation_lock.json', {'created_utc': utc(), 'refits': refits,
                   'ranges': [[16, 20], [20, 60]], 'protocol_sha256': lock['protocol_sha256']})
    rows, per_cell = [], []
    for seed in SEEDS:
        for name in ('selected.pt', 'refit.pt'):
            cp = torch.load(DEST / 'checkpoints' / str(seed) / name, weights_only=True)
            rows.extend(cp['curve'])
            per_cell.extend(cp['per_cell_curve'])
    write_csv(DEST / 'training_per_seed.csv', rows)
    write_csv(DEST / 'training_per_cell.csv', per_cell)
    print('TRAINING COMPLETE; all checkpoints frozen before evaluation', flush=True)


if __name__ == '__main__':
    main()
