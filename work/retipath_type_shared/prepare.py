from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

import torch
from torch.nn import functional as F

from common import (ROOT, DEST, BASELINE, REGISTRY, SEEDS, metadata_from_registry,
                    build_initial, training_banks, digest, read_json, exclusive_json,
                    batch_loss, utc, write_csv, save_checkpoint)
from model import TypeSharedRetiPath, SHARED, SPECIFIC


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if DEST.exists():
        raise FileExistsError(f'Preserve existing run: {DEST}')
    metadata, registry = metadata_from_registry()
    banks, data = training_banks()
    model, rms = build_initial(metadata, SEEDS[0], 'selection')
    bank = banks['inner_fit']
    generator = torch.Generator().manual_seed(SEEDS[0] + 1_000_003)
    windows, rows = bank.sample(4, generator)
    with torch.no_grad():
        for i, p in enumerate(model.trainable()):
            p.add_(0.03 * torch.sin(torch.arange(p.numel()).reshape(p.shape) + i))
        model.project()
    drives = model.stimulus_drives(bank.inputs[windows])
    fast_losses, reference_losses, errors = [], [], {}
    for cell, selected in rows.items():
        obs = bank.cells[cell]
        fast = model.logits_from_drive(drives[cell], obs.history[selected], cell)
        reference = model.cell_models[cell](bank.inputs[windows], observed_counts=obs.targets[selected]).logits
        torch.testing.assert_close(fast, reference, rtol=2e-5, atol=2e-6)
        errors[cell] = float((fast-reference).abs().max().detach())
        fast_losses.append(F.binary_cross_entropy_with_logits(fast[obs.mask[selected]], obs.targets[selected][obs.mask[selected]]))
        reference_losses.append(F.binary_cross_entropy_with_logits(reference[obs.mask[selected]], obs.targets[selected][obs.mask[selected]]))
    params = model.trainable()
    fast_grads = torch.autograd.grad(torch.stack(fast_losses).mean(), params)
    reference_grads = torch.autograd.grad(torch.stack(reference_losses).mean(), params)
    gradient_error = 0.
    for fast, reference in zip(fast_grads, reference_grads, strict=True):
        torch.testing.assert_close(fast, reference, rtol=3e-4, atol=1e-7)
        gradient_error = max(gradient_error, float((fast-reference).abs().max()))
    for cells in model.groups.values():
        for name in SHARED:
            assert len({id(model.cell_models[c].get_parameter(name)) for c in cells}) == 1
    for name in SHARED:
        assert len({id(model.cell_models[cells[0]].get_parameter(name)) for cells in model.groups.values()}) == 4
    for name in SPECIFIC:
        assert len({id(model.cell_models[c].get_parameter(name)) for c in model.cells}) == 22
    x = bank.inputs[windows[:1]].clone()
    with torch.no_grad():
        d0 = model.stimulus_drives(x)
        x[:, 80:] += .7
        d1 = model.stimulus_drives(x)
        for cell in model.cells:
            torch.testing.assert_close(d0[cell][:, :80], d1[cell][:, :80], rtol=0, atol=0)
    optimizer = torch.optim.Adam(params, lr=.003, weight_decay=0)
    assert {id(p) for p in params} == {id(p) for g in optimizer.param_groups for p in g['params']}
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    fixed = {n: b.clone() for n, b in model.named_buffers()}
    start = time.perf_counter()
    loss, _ = batch_loss(model, bank, generator)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
    norm = float(torch.nn.utils.clip_grad_norm_(params, 5, error_if_nonfinite=True))
    optimizer.step()
    model.project()
    seconds = time.perf_counter() - start
    assert all(torch.equal(b, dict(model.named_buffers())[n]) for n, b in fixed.items())
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters() if not p.requires_grad)
    changed = {n: not torch.equal(p, before[n]) for n, p in model.named_parameters() if p.requires_grad}
    (DEST / 'checkpoints/provenance').mkdir(parents=True)
    (DEST / 'figures').mkdir()
    save_checkpoint(DEST / 'checkpoints/provenance/metadata.pt', metadata)
    write_csv(DEST / 'parameters.csv', model.inventory())
    protocol = (Path(__file__).parent / 'protocol_template.md').read_text(encoding='utf-8')
    with (DEST / 'PROTOCOL.md').open('x', encoding='utf-8') as stream:
        stream.write(protocol)
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in Path(__file__).parent.glob('*.py')}
    previous = read_json(BASELINE / 'checkpoints/provenance/pretraining_lock.json')
    for path, sha in previous['frozen_sha256'].items():
        assert digest(ROOT / path) == sha
        frozen[path] = sha
    for path in (REGISTRY, BASELINE / 'per_cell_scores.csv', BASELINE / 'comparison_summary.csv',
                 BASELINE / 'checkpoints/provenance/evaluation_lock.json',
                 ROOT / 'work/retipath_phase2_common.py', ROOT / 'work/retipath_final_common.py',
                 ROOT / 'work/retipath_final_prediction.py'):
        frozen[str(path.relative_to(ROOT))] = digest(path)
    baseline_lock = read_json(BASELINE / 'checkpoints/provenance/evaluation_lock.json')
    for seed in SEEDS:
        path = BASELINE / 'checkpoints' / str(seed) / 'refit.pt'
        assert digest(path) == baseline_lock['refit_sha256'][str(seed)]
        frozen[str(path.relative_to(ROOT))] = digest(path)
    evidence = {'status': 'VERIFIED', 'created_utc': utc(), 'forward_max_abs_error': max(errors.values()),
        'forward_per_cell': errors, 'joint_gradient_max_abs_error': gradient_error,
        'type_aliases_and_cell_specific_identity': True, 'future_stimulus_leakage': False,
        'history': 'uses unchanged strictly-past state; numerical replay matched each formal cell',
        'optimizer_complete_and_finite_gradients': True, 'engineering_gradient_norm': norm,
        'engineering_update_seconds': seconds, 'engineering_update_discarded': True,
        'fixed_parameters_and_buffers_unchanged': True, 'changed_after_engineering_update': changed,
        'trainable_total': sum(p.numel() for p in params), 'shared_total': 4*29, 'cell_specific_total': 22*8}
    exclusive_json(DEST / 'correctness.json', evidence)
    lock = {'created_utc': utc(), 'branch': subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
        'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'seeds': list(SEEDS), 'groups': dict(model.groups), 'data': data,
        'current_retipath': {c: row['RetiPath'] for c, row in registry['cells'].items()},
        'frozen_sha256': frozen, 'protocol_sha256': digest(DEST / 'PROTOCOL.md'),
        'shared_parameters': list(SHARED), 'cell_specific_parameters': list(SPECIFIC),
        'trainable_parameters': evidence['trainable_total'], 'maximum_updates': 3000,
        'evaluate_every': 25, 'patience_updates': 400, 'min_delta_for_patience_only': 1e-7,
        'optimizer': 'Adam', 'lr': .003, 'weight_decay': 0., 'clip': 5., 'batch_movies_per_cell': 4,
        'scheduler': None, 'selection_metric': 'equal-cell Bernoulli NLL',
        'evaluation_ranges': [[16, 20], [20, 60]], 'bootstrap_samples': 100000,
        'bootstrap_seed': 20260908, 'tie_tolerance': 1e-7}
    exclusive_json(DEST / 'checkpoints/provenance/pretraining_lock.json', lock)
    print(json.dumps({'prepared': str(DEST), 'trainable': evidence['trainable_total'],
                      'forward_error': evidence['forward_max_abs_error'], 'seconds_per_update': seconds}), flush=True)


if __name__ == '__main__':
    main()
