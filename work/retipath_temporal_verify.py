from __future__ import annotations

import csv
from datetime import datetime

import numpy as np
import torch

from retipath_temporal_common import ROOT, OUT, SEEDS, CONDITIONS, load_json, sha, utc, write_once, make_split
from retipath_phase2_analyze import model_from_checkpoint, rf_modes
from retipath_phase2_common import evaluate


def verify() -> None:
    torch.set_num_threads(1)
    lock = load_json(OUT / 'confirmatory_lock.json')
    digest = sha(OUT / 'confirmatory_lock.json')
    marker = load_json(OUT / 'TEST_CONSUMED.json')
    analysis = load_json(OUT / 'analysis_complete.json')
    assert marker['confirmatory_lock_sha256'] == analysis['confirmatory_lock_sha256'] == digest
    assert datetime.fromisoformat(lock['frozen_utc']) < datetime.fromisoformat(marker['target_access_start_utc'])
    assert lock['live_range_s'] == marker['live_range_s'] == [240, 300]
    assert marker['new_blocks_consumed'] == analysis['new_blocks_consumed'] == 1
    assert marker['training_updates'] == analysis['training_updates'] == analysis['refits'] == 0
    for path, expected in lock['frozen_sha256'].items():
        assert sha(ROOT / path) == expected, path
    cells = [row['cell_id'] for row in lock['cells']]
    assert len(cells) == 17 and len(set(cells)) == 17
    assert len(list((OUT / 'targets').glob('*.npz'))) == len(cells)
    jobs = [load_json(path) for path in sorted((OUT / 'results').glob('*.json'))]
    assert len(jobs) == 51
    assert {(j['cell_id'], j['seed']) for j in jobs} == {(c, s) for c in cells for s in SEEDS}
    raw_csv = list(csv.DictReader((OUT / 'prediction_per_cell_seed.csv').open(encoding='utf-8')))
    csv_nll = {(r['cell_id'], int(r['seed']), r['condition']): float(r['confirmation_nll']) for r in raw_csv}
    lookup = {}
    nll_error = 0.0
    metric_error = 0.0
    normalization_error = 0.0
    max_jacobian_error = 0.0
    observations = 0
    grid = np.array(lock['radial_grid_deg'])
    with np.load(OUT / 'sampling_before_targets.npz', allow_pickle=False) as sampling:
        positions = sampling['positions'].astype(np.float64)
        cones = sampling['cones']
        for info in lock['cells']:
            safe = info['cell_id'].replace('#', '_')
            record = load_json(OUT / 'targets' / f'{safe}.json')
            assert record['confirmatory_lock_sha256'] == digest
            assert record['stored_target_range_s'] == [240, 300]
            assert datetime.fromisoformat(record['started_utc']) >= datetime.fromisoformat(marker['target_access_start_utc'])
            target_path = OUT / 'targets' / f'{safe}.npz'
            assert sha(target_path) == record['target_sha256']
            with np.load(target_path, allow_pickle=False) as target:
                counts = target['counts']
            assert counts.shape == (9000,) and (counts >= 0).all()
            assert counts.sum() == record['selected_spikes']
            split = make_split(cones, counts, info['recording_id'], 240)
            assert list(split.source_image_ids) == info['source_ids']
            assert split.valid_mask.sum().item() == 7200
            for label in ('LOW', 'HIGH'):
                indices = sampling[safe + '_' + label + '_indices']
                assert indices.shape == (info['observation_counts'][label], 2)
                assert 0 < len(indices) <= 20
                assert ((indices[:, 0] >= 0) & (indices[:, 0] < 60)).all()
                assert ((indices[:, 1] >= 30) & (indices[:, 1] < 150)).all()
                values = sampling[safe + '_context'][indices[:, 0], indices[:, 1]]
                assert (values <= info['LOW_threshold']).all() if label == 'LOW' else (values >= info['HIGH_threshold']).all()
        for job in jobs:
            cell, seed = job['cell_id'], job['seed']
            safe = cell.replace('#', '_')
            assert job['confirmatory_lock_sha256'] == digest
            assert job['parameters_unchanged'] and job['training_updates'] == 0
            with np.load(OUT / 'targets' / f'{safe}.npz', allow_pickle=False) as target:
                events = (target['counts'].reshape(60, 150, 1) > 0).astype(np.float64)
            with np.load(OUT / 'results' / f'{safe}_{seed}.npz', allow_pickle=False) as outputs:
                for row in job['prediction']:
                    condition = row['condition']
                    z = outputs[condition + '_logits'].astype(np.float64)
                    assert z.shape == events.shape and np.isfinite(z).all()
                    nll = float((np.logaddexp(0, z) - events * z)[:, 30:].mean())
                    nll_error = max(nll_error, abs(nll - row['confirmation_nll']))
                    assert abs(nll - row['confirmation_nll']) < 2e-14
                    key = cell, seed, condition
                    assert csv_nll[key] == row['confirmation_nll']
                    lookup[key] = row['confirmation_nll']
                    ref = lock['all_phase2_checkpoint_references'][cell][f'{seed}/{condition}/refit']
                    assert row['checkpoint_sha256'] == ref['sha256']
            for row in job['RF']:
                assert row['status'] == 'VERIFIED'
                low = np.array(row['LOW_profile'])
                high = np.array(row['HIGH_profile'])
                delta = high - low
                for label, profile in [('LOW', low), ('HIGH', high)]:
                    assert profile.shape == (289,) and np.isfinite(profile).all() and (profile >= 0).all()
                    normalization_error = max(normalization_error, abs(profile.sum() - 1))
                    assert abs(profile.sum() - 1) < 1e-12
                    center = profile @ positions
                    radius = np.sqrt(profile @ np.sum((positions - center) ** 2, axis=1))
                    metric_error = max(metric_error, abs(radius - row[label + '_radius_mean_profile_deg']))
                    np.testing.assert_allclose(center, [row[label + '_centroid_x_deg'], row[label + '_centroid_y_deg']], atol=1e-14, rtol=0)
                    cdf = np.array(row[label + '_cdf'])
                    assert np.isfinite(cdf).all() and (np.diff(cdf) >= -1e-14).all()
                    assert abs(cdf[-1] - 1) < 1e-12 and cdf[0] >= 0
                    assert row[label + '_gain_median'] > 0
                np.testing.assert_array_equal(delta, row['delta_P'])
                cdf_delta = np.array(row['HIGH_cdf']) - np.array(row['LOW_cdf'])
                expected = {
                    'TV': np.abs(delta).sum() / 2,
                    'centered_radial_CDF_L1_deg': np.trapezoid(np.abs(cdf_delta), grid),
                    'centered_radial_CDF_max': np.abs(cdf_delta).max(),
                    'gain_log_HIGH_over_LOW': np.log(row['HIGH_gain_median'] / row['LOW_gain_median']),
                    'delta_radius_mean_profile_deg': row['HIGH_radius_mean_profile_deg'] - row['LOW_radius_mean_profile_deg'],
                    'delta_radius_observation_median_deg': row['HIGH_radius_observation_median_deg'] - row['LOW_radius_observation_median_deg'],
                    'centroid_shift_deg': np.hypot(row['HIGH_centroid_x_deg'] - row['LOW_centroid_x_deg'], row['HIGH_centroid_y_deg'] - row['LOW_centroid_y_deg']),
                }
                for metric, value in expected.items():
                    metric_error = max(metric_error, abs(value - row[metric]))
                    assert abs(value - row[metric]) < 1e-13
                quality = job['checks'][row['condition']]
                max_jacobian_error = max(max_jacobian_error, quality['sum_relative_l2_error'])
                observations += quality['observations']
        first = lock['cells'][0]
        safe = first['cell_id'].replace('#', '_')
        with np.load(OUT / 'targets' / f'{safe}.npz', allow_pickle=False) as target:
            replay_split = make_split(cones, target['counts'], first['recording_id'], 240)
        selections = {label: sampling[safe + '_' + label + '_indices'].tolist() for label in ('LOW', 'HIGH')}
    rng = np.random.default_rng(lock['bootstrap_seed'])
    draws = rng.integers(0, len(cells), size=(lock['bootstrap_samples'], len(cells)))
    bootstrap_error = 0.0
    for comparison in ['C-A', 'C-B', 'B-A']:
        left, right = comparison.split('-')
        differences = np.array([[lookup[cell, seed, left] - lookup[cell, seed, right] for cell in cells] for seed in SEEDS])
        for label, values in [*zip(SEEDS, differences, strict=True), ('all', differences.mean(0))]:
            expected = analysis['prediction'][f'{comparison}/{label}']
            interval = np.percentile(values[draws].mean(1), [2.5, 97.5])
            fields = {'mean': values.mean(), 'median': np.median(values), 'wins': (values < 0).sum(),
                      'losses': (values > 0).sum(), 'ties': (values == 0).sum(), 'ci_low': interval[0], 'ci_high': interval[1]}
            for name, value in fields.items():
                bootstrap_error = max(bootstrap_error, abs(value - expected[name]))
                assert abs(value - expected[name]) < 1e-14
    replay = []
    original = next(j for j in jobs if j['cell_id'] == first['cell_id'] and j['seed'] == SEEDS[0])
    for condition in CONDITIONS:
        ref = lock['all_phase2_checkpoint_references'][first['cell_id']][f'{SEEDS[0]}/{condition}/refit']
        model, cp = model_from_checkpoint(ROOT / ref['path'])
        model.eval().requires_grad_(False)
        nll, logits = evaluate(model, replay_split)
        with np.load(OUT / 'results' / f'{safe}_{SEEDS[0]}.npz', allow_pickle=False) as outputs:
            np.testing.assert_array_equal(logits.numpy(), outputs[condition + '_logits'])
        assert nll == lookup[first['cell_id'], SEEDS[0], condition]
        rf, _, quality = rf_modes(model, replay_split, selections, positions, grid)
        saved = next(r for r in original['RF'] if r['condition'] == condition)
        assert all(rf[key] == saved[key] for key in rf)
        assert all(torch.equal(v.to(cp['model'][k]), cp['model'][k]) for k, v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        replay.append({'cell_id': first['cell_id'], 'seed': SEEDS[0], 'condition': condition,
                       'logits_bitwise_equal': True, 'all_RF_fields_equal': True, 'parameters_unchanged': True})
    assert observations == analysis['RF_observations'] == 6120
    for path, expected in lock['frozen_sha256'].items():
        assert sha(ROOT / path) == expected, path
    write_once(OUT / 'verification.json', {
        'status': 'VERIFIED', 'verified_utc': utc(), 'confirmatory_lock_sha256': digest,
        'source_and_checkpoint_hashes_unchanged': len(lock['frozen_sha256']),
        'all_phase2_checkpoint_files_frozen': 396, 'scored_models': len(lookup), 'scored_cells': len(cells),
        'lock_before_marker_before_target_parse': True, 'new_blocks_consumed': 1,
        'target_range_s': [240, 300], 'bins_per_cell': 7200, 'training_updates': 0, 'refits': 0,
        'independent_saved_logits_NLL_max_abs_error': nll_error,
        'independent_paired_bootstrap_max_abs_error': bootstrap_error,
        'RF_profile_normalization_max_abs_error': normalization_error,
        'RF_derived_metric_max_abs_error': metric_error, 'RF_observations': observations,
        'full_Jacobian_mode_sum_max_relative_error': max_jacobian_error,
        'frozen_checkpoint_replay': replay,
        'limits': ['Verification concerns the locked computation, not biological mechanism.',
                   'Five Phase 2 cells have no continuous recording coverage for this block.',
                   'Local provenance excludes prior target-based use; legacy whole-file parsing is disclosed in PROTOCOL.md.'],
        'verification_source_sha256': sha(ROOT / 'work/retipath_temporal_verify.py'),
    })
    print('VERIFIED: 153 NLLs, paired bootstrap, 6120 RF observations, 3 exact checkpoint replays; frozen files unchanged.')


if __name__ == '__main__':
    verify()
