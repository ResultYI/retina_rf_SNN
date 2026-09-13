from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from retipath_phase2_common import (
    ROOT, SEEDS, CONDITIONS, SelectionStop, load_json, save_json, fresh_model,
    minibatches, tensor_sha, sha,
)
from retipath_phase2_train import digest_state
from retipath_spatial_ei_report import profile_metrics


def verify(out: Path) -> dict:
    torch.set_num_threads(1)
    config = load_json(out/'checkpoints/protocol.json')
    complete = load_json(out/'checkpoints/training_complete.json')
    analysis = load_json(out/'checkpoints/analysis_complete.json')
    assert complete['inner_fits'] == complete['fresh_refits'] == 198
    assert not complete['development_used_for_selection']
    assert sha(out/'PROTOCOL.md') == config['protocol_sha256']
    assert config['initialization'] == 'fresh'
    for path, digest in config['source_hashes'].items():
        assert sha(ROOT/path) == digest, path
    for ref in config['cells'].values():
        for path, digest in (('input_path', 'input_sha256'),
                             ('metadata_path', 'metadata_sha256'),
                             ('M2_checkpoint_path', 'M2_checkpoint_sha256')):
            assert sha(Path(ref[path])) == ref[digest], ref[path]
    for path, digest in load_json(out/'checkpoints/selection_lock.json')['files'].items():
        assert sha(out/path) == digest, path
    fits = []; curves = []; initial_hashes = {}
    for cell, ref in config['cells'].items():
        for seed in SEEDS:
            schedules = {
                phase: tensor_sha(minibatches(ref[count], seed))
                for phase, count in (('inner', 'inner_train_sequences'),
                                     ('refit', 'full_train_sequences'))
            }
            for condition in CONDITIONS:
                folder = out/'checkpoints'/cell.replace('#', '_')/str(seed)/condition
                inner = torch.load(folder/'inner.pt', weights_only=True)
                for phase in ('inner', 'refit'):
                    cp = inner if phase == 'inner' else torch.load(folder/'refit.pt', weights_only=True)
                    assert cp['schedule_sha256'] == schedules[phase]
                    assert cp['protocol_sha256'] == config['protocol_sha256']
                    assert cp['trainable_parameters'] == (34 if condition == 'A' else 37)
                    assert cp['optimizer_steps'] == ([cp['step']] if cp['step'] else [])
                    assert cp['fixed_buffers_unchanged']
                    assert cp['step'] <= 3000
                    model = fresh_model(cp, seed, condition, cp['rms'])
                    initial = model.state_dict()
                    assert initial.keys() == cp['initial_state'].keys()
                    assert all(torch.equal(v, cp['initial_state'][k]) for k, v in initial.items())
                    upstream = {k: v for k, v in initial.items() if not k.startswith('spatial_ei.')}
                    assert digest_state(upstream) == cp['upstream_initial_sha256']
                    initial_hashes[cell, seed, phase, condition] = digest_state(initial)
                    for name, buffer in model.named_buffers():
                        assert torch.equal(buffer, cp['model'][name]), name
                    assert all(bool(torch.isfinite(v).all()) for v in cp['model'].values())
                    assert [r['step'] for r in cp['curve']] == list(range(0, cp['step']+1, 25))
                    if phase == 'inner':
                        rows = cp['curve']
                        status = SelectionStop(rows[0]['inner_validation_nll'], 0,
                                               rows[0]['inner_validation_nll'], 0)
                        for row in rows[1:]:
                            assert not status.stopped(row['step']-25)
                            status = status.observe(row['inner_validation_nll'], row['step'])
                        assert status.best_step == cp['best_step']
                        assert status.best_nll == cp['best_replay_inner_nll']
                        assert cp['step'] == 3000 or status.stopped(cp['step'])
                        assert all(torch.equal(v, cp['best_state'][k]) for k, v in cp['model'].items())
                    else:
                        assert cp['step'] == cp['best_step'] == inner['best_step']
                        assert cp['curve'][-1]['full_train_nll'] == cp['refit_train_nll']
                    fits.append({'cell_id': cell, 'seed': seed, 'condition': condition,
                                 'phase': phase, 'step': cp['step'], 'selected_step': cp['best_step'],
                                 'zero_gradient_parameters': [k for k, v in cp['gradient_nonzero_seen'].items() if not v],
                                 'unchanged_parameters': [k for k, v in cp['actually_changed'].items() if not v]})
                    curves.extend(cp['curve'])
            for phase in ('inner', 'refit'):
                assert initial_hashes[cell, seed, phase, 'B'] == initial_hashes[cell, seed, phase, 'C']
            assert initial_hashes[cell, seed, 'inner', 'A'] == initial_hashes[cell, seed, 'refit', 'A']
    assert len(fits) == 396
    with (out/'training_per_seed.csv').open(newline='', encoding='utf-8') as stream:
        saved = list(csv.DictReader(stream))
    key = lambda r: (r['cell_id'], int(r['seed']), r['condition'], r['phase'], int(r['step']))
    lookup = {key(r): r for r in curves}
    assert len(saved) == len(lookup) == len(curves)
    for row in saved:
        for name in ('inner_validation_nll', 'full_train_nll', 'sampled_train_batch_nll'):
            expected = lookup[key(row)][name]
            assert (float(row[name]) if row[name] else None) == expected
    prediction = {(r['cell_id'], r['seed'], r['condition']): r for r in analysis['raw_prediction']}
    with (out/'prediction_population.csv').open(newline='', encoding='utf-8') as stream:
        paired_rows = list(csv.DictReader(stream))
    for row in paired_rows:
        if row['row_type'] != 'cell_seed':
            continue
        left, right = row['comparison'].split('-')
        a = prediction[row['cell_id'], int(row['seed']), left]
        b = prediction[row['cell_id'], int(row['seed']), right]
        assert float(row['delta_dev_nll']) == a['dev_nll']-b['dev_nll']
        assert float(row['left_dev_nll']) == a['dev_nll']
        assert float(row['right_dev_nll']) == b['dev_nll']
    cells = list(config['cells'])
    draws = np.random.default_rng(config['bootstrap_seed']).integers(0, len(cells), (10000, len(cells)))
    for left, right in (('C', 'B'), ('C', 'A'), ('B', 'A')):
        differences = np.array([[prediction[c, s, left]['dev_nll']-prediction[c, s, right]['dev_nll']
                                 for c in cells] for s in SEEDS])
        for seed, values in zip((*SEEDS, 'all'), (*differences, differences.mean(0)), strict=True):
            stored = analysis['prediction'][f'{left}-{right}/{seed}']
            assert stored['mean'] == float(values.mean())
            assert stored['median'] == float(np.median(values))
            assert stored['wins'] == int((values < 0).sum())
            np.testing.assert_array_equal(np.quantile(values[draws].mean(-1), [.025, .975]),
                                          [stored['ci_low'], stored['ci_high']])
    with (out/'rf_population.csv').open(newline='', encoding='utf-8') as stream:
        rf = list(csv.DictReader(stream))
    with (out/'mode_contribution.csv').open(newline='', encoding='utf-8') as stream:
        modes = list(csv.DictReader(stream))
    assert len(rf) == len(modes) == 198
    mode_lookup = {(r['cell_id'], int(r['seed']), r['condition']): r for r in modes}
    observations = 0; maximum_relative_error = 0.
    with np.load(out/'figures/delta_P_maps_by_cell_seed_condition.npz', allow_pickle=False) as maps:
        for row in rf:
            if row['status'] != 'VERIFIED':
                continue
            cell, seed, condition = row['cell_id'], int(row['seed']), row['condition']
            safe = cell.replace('#', '_'); positions = maps[safe+'_positions_deg']
            low = np.array(json.loads(row['LOW_profile']))
            high = np.array(json.loads(row['HIGH_profile']))
            delta = np.array(json.loads(row['delta_P']))
            np.testing.assert_allclose(high-low, delta, rtol=0, atol=0)
            assert np.isclose(np.abs(delta).sum()/2, float(row['TV']), rtol=1e-14)
            np.testing.assert_array_equal(maps[safe+'_delta_P'][list(CONDITIONS).index(condition), SEEDS.index(seed)], delta)
            mode = mode_lookup[cell, seed, condition]
            for label, p in (('LOW', low), ('HIGH', high)):
                assert np.isclose(p.sum(), 1, atol=1e-12) and np.all(p >= 0)
                centroid, radius = profile_metrics(p, positions)
                np.testing.assert_allclose(centroid, [float(row[label+'_centroid_x_deg']), float(row[label+'_centroid_y_deg'])], atol=1e-14)
                assert np.isclose(radius, float(row[label+'_radius_mean_profile_deg']), rtol=1e-14)
                raw = json.loads(mode[label+'_observations']); observations += len(raw)
                assert len(raw) == int(row[label+'_n']) <= 20
                assert np.isclose(np.mean([o['q1'] for o in raw]), float(mode[label+'_q1_mean']))
                assert all(np.isclose(o['q1']+o['q2'], 1, atol=1e-12) for o in raw)
                assert all(np.isclose(o['signed1']+o['signed2'], 1, atol=1e-8) for o in raw)
            cached = load_json(out/'checkpoints'/safe/str(seed)/'analysis.json')
            error = cached['checks'][condition]['RF']['sum_relative_l2_error']
            maximum_relative_error = max(maximum_relative_error, error)
            assert error < 1e-8
    required = ['PROTOCOL.md', 'training_per_seed.csv', 'prediction_population.csv',
                'rf_population.csv', 'mode_contribution.csv', 'REPORT.md']
    assert all((out/name).is_file() for name in required)
    assert len(list((out/'figures').glob('delta_P_*.png'))) == 22
    result = {'status': 'VERIFIED', 'fits': 396, 'training_rows': len(curves),
              'prediction_rows': len(paired_rows), 'RF_rows': len(rf), 'mode_rows': len(modes),
              'RF_observations': observations, 'maximum_mode_sum_relative_l2_error': maximum_relative_error,
              'fresh_initialization_and_refit_steps': True, 'matched_BC_initialization': True,
              'matched_schedules': True, 'frozen_inputs_sources_and_M2_checkpoints_unchanged': True,
              'artifact_consistency': True, 'fit_diagnostics': fits,
              'independent_test': 'NONE', 'verification_source_sha256': sha(Path(__file__))}
    save_json(out/'checkpoints/verification.json', result)
    return {k: v for k, v in result.items() if k != 'fit_diagnostics'}


if __name__ == '__main__':
    print(json.dumps(verify(ROOT/'output/experiments/retipath_spatial_ei_phase2_population'), indent=2))
