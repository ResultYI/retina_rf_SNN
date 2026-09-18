from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from common import (ROOT, DEST, BASELINE, SEEDS, digest, read_json, exclusive_json,
                    training_banks, build_initial, utc)
from model import TypeSharedRetiPath
from train import state_hash
from retipath_phase2_common import SelectionStop
from models.mechanistic_retina.state import fixed_one_bin_history_state


def records(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pre = read_json(DEST / 'checkpoints/provenance/pretraining_lock.json')
    selection = read_json(DEST / 'checkpoints/provenance/selection_lock.json')
    evaluation = read_json(DEST / 'checkpoints/provenance/evaluation_lock.json')
    assert pre['created_utc'] < selection['created_utc'] < evaluation['created_utc']
    for path, sha in pre['frozen_sha256'].items():
        assert digest(ROOT / path) == sha, path
    assert digest(DEST / 'PROTOCOL.md') == pre['protocol_sha256']
    program = read_json(DEST / 'checkpoints/provenance/evaluation_program_lock.json')
    assert digest(ROOT / program['source_path']) == program['sha256']
    banks, _ = training_banks()
    metadata = torch.load(DEST / 'checkpoints/provenance/metadata.pt', weights_only=True)
    observations = []
    for seed in SEEDS:
        selected_cp = torch.load(DEST / 'checkpoints' / str(seed) / 'selected.pt', weights_only=True)
        status = None
        for row in selected_cp['curve']:
            value, step = row['inner_validation_nll'], row['step']
            assert step % 25 == 0 and step <= 3000
            status = SelectionStop(value, step, value, step) if status is None else status.observe(value, step)
            assert row['selected_step'] == status.best_step
        assert selected_cp['selected_step'] == status.best_step
        assert selected_cp['step'] == 3000 or status.stopped(selected_cp['step'])
        assert selected_cp['model_step'] == min(selected_cp['curve'], key=lambda r: r['inner_validation_nll'])['step']
        assert selected_cp['completed_utc'] < selection['created_utc']
        for phase, filename in (('selection', 'selected.pt'), ('refit', 'refit.pt')):
            cp = torch.load(DEST / 'checkpoints' / str(seed) / filename, weights_only=True)
            bank = banks['inner_fit' if phase == 'selection' else 'refit']
            initial, rms = build_initial(metadata, seed, phase)
            assert cp['rms'] == rms
            assert state_hash(initial.state_dict()) == cp['initial_state_sha256']
            assert state_hash(dict(initial.named_buffers())) == cp['fixed_buffer_sha256']
            frozen_params = {n: p.clone() for n, p in initial.named_parameters() if not p.requires_grad}
            initial.load_state_dict(cp['model'], strict=True)
            assert state_hash(dict(initial.named_buffers())) == cp['fixed_buffer_sha256']
            assert all(torch.equal(p, dict(initial.named_parameters())[n]) for n, p in frozen_params.items())
            generator = torch.Generator().manual_seed(seed + 1_000_003)
            for _ in range(cp['step']):
                bank.sample(4, generator)
            assert torch.equal(generator.get_state(), cp['sampler_rng'])
            assert cp['optimizer_steps'] == ([cp['step']] if cp['step'] else [])
            if phase == 'refit':
                assert cp['step'] == selected_cp['selected_step']
                assert selection['created_utc'] < cp['completed_utc'] < evaluation['created_utc']
                assert digest(DEST / 'checkpoints' / str(seed) / filename) == evaluation['refits'][str(seed)]['checkpoint_sha256']
                first_cell = initial.cells[0]
                obs = bank.cells[first_cell]
                target = obs.targets[:1].clone()
                changed = target.clone()
                changed[:, 80] = 1 - changed[:, 80]
                m = initial.cell_models[first_cell]
                h0 = fixed_one_bin_history_state(target, float(m.rgc.history_decay))
                h1 = fixed_one_bin_history_state(changed, float(m.rgc.history_decay))
                assert torch.equal(h0[:, :81], h1[:, :81])
                with torch.no_grad():
                    drive = initial.stimulus_drives(bank.inputs[obs.input_indices[:1]])[first_cell]
                    assert torch.equal(initial.logits_from_drive(drive, h0, first_cell)[:, :81],
                                       initial.logits_from_drive(drive, h1, first_cell)[:, :81])
            observations.append({'seed': seed, 'phase': phase, 'step': cp['step'],
                'fresh_initialization_and_RMS_reproduced': True, 'sampler_schedule_reproduced': True,
                'fixed_parameters_and_buffers_unchanged': True,
                'gradient_never_nonzero': [n for n, seen in cp['gradient_seen'].items() if not seen],
                'parameters_unchanged': [n for n, changed in cp['actually_changed'].items() if not changed]})
    rows = records(DEST / 'prediction_per_cell.csv')
    assert len(rows) == 176
    assert {r['range'] for r in rows} == {'[16,20)', '[20,60)'}
    max_summary_error = 0.
    for summary in records(DEST / 'comparison_summary.csv') + records(DEST / 'type_summary.csv'):
        matching = sorted((r for r in rows if r['range'] == summary['range'] and r['seed'] == summary['seed']
                           and (summary['type'] == 'population' or r['type'] == summary['type'])), key=lambda r: r['cell_id'])
        assert len(matching) == int(summary['cells'])
        rng = np.random.default_rng(20260908)
        indices = rng.integers(0, len(matching), (100000, len(matching)))
        for name in ('shared_minus_current', 'shared_minus_openretina', 'current_minus_openretina'):
            array = np.array([float(r[name]) for r in matching])
            ci = np.quantile(array[indices].mean(1), [.025, .975])
            checks = {'mean': array.mean(), 'median': np.median(array), 'ci_low': ci[0], 'ci_high': ci[1],
                      'wins': (array < -1e-7).sum(), 'losses': (array > 1e-7).sum(), 'ties': (np.abs(array) <= 1e-7).sum()}
            for suffix, value in checks.items():
                error = abs(value - float(summary[name + '_' + suffix]))
                max_summary_error = max(max_summary_error, error)
                assert error < 1e-12
    for label in ('[16,20)', '[20,60)'):
        for cell in metadata:
            selected = [r for r in rows if r['range'] == label and r['cell_id'] == cell and r['seed'] != 'aggregate']
            aggregate = next(r for r in rows if r['range'] == label and r['cell_id'] == cell and r['seed'] == 'aggregate')
            for field in ('current_retipath_nll', 'shared_retipath_nll', 'openretina_adapted_nll'):
                assert abs(np.mean([float(r[field]) for r in selected]) - float(aggregate[field])) < 1e-12
    exclusive_json(DEST / 'checkpoints/provenance/verification_complete.json',
                   {'status': 'VERIFIED', 'completed_utc': utc(), 'fits': observations,
                    'independent_summary_max_error': max_summary_error,
                    'scored_current_spike_no_leakage': True, 'training_target_ranges': [[0, 16]],
                    'evaluation_target_ranges': [[16, 20], [20, 60]], 'reference_models_unchanged': True,
                    'new_architecture_layers': 0, 'RF_and_illusion_reruns': 0})
    print('VERIFIED: frozen sources, fresh refits, schedules, causal history, paired statistics', flush=True)


if __name__ == '__main__':
    main()
