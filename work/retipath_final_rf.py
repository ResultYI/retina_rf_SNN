from __future__ import annotations

import itertools

import numpy as np
import torch

from retipath_final_common import ROOT, OUT, PHASE2, CONFIRM, SEEDS, load_json, sha, write_once, load_retipath, verify_sources
from retipath_final_prediction import load_split
from retipath_phase2_analyze import rf_modes, csv_rows, frozen_observations


def main() -> None:
    torch.set_num_threads(1)
    lock = verify_sources()
    registry = load_json(OUT / 'model_registry.json')['cells']
    destination = OUT / 'dynamic_rf'
    destination.mkdir(exist_ok=True)
    (destination / 'profiles').mkdir(exist_ok=True)
    phase = load_json(PHASE2 / 'checkpoints/protocol.json')
    grid = np.array(phase['radial_grid_deg'])
    rows = []
    replays = []
    continuous = {r['cell_id'] for r in lock['continuous_cells']}
    for start, stop in ((16, 20), (240, 300)):
        for cell in lock['cells']:
            if start == 240 and cell not in continuous:
                continue
            safe = cell.replace('#', '_')
            for seed in SEEDS:
                source = (PHASE2 / 'checkpoints' / safe / str(seed) / 'analysis.json' if start == 16
                          else CONFIRM / 'results' / f'{safe}_{seed}.json')
                original = load_json(source)
                row = next(r for r in original['RF'] if r['condition'] == 'C')
                reference = registry[cell]['RetiPath'][str(seed)]
                assert original.get('protocol_sha256', phase['protocol_sha256']) == phase['protocol_sha256']
                renamed = {k: v for k, v in row.items() if k not in ('condition', 'refit_updates')}
                renamed.update(model='RetiPath', range=f'[{start},{stop})', source_RF=source.as_posix(),
                    source_RF_sha256=sha(source), checkpoint_path=reference['path'], checkpoint_sha256=reference['sha256'],
                    evidence_status='existing final-architecture full-prefix Jacobian, verified and reorganized')
                rows.append(renamed)
                if row['status'] != 'VERIFIED':
                    continue
                low, high = np.array(row['LOW_profile']), np.array(row['HIGH_profile'])
                assert abs(low.sum() - 1) < 1e-12 and abs(high.sum() - 1) < 1e-12
                np.testing.assert_array_equal(high - low, row['delta_P'])
                assert abs(np.abs(high - low).sum() / 2 - row['TV']) < 1e-14
                profile_path = destination / 'profiles' / f'{safe}_{start}_{seed}.npz'
                arrays = dict(LOW_P=low, HIGH_P=high, delta_P=high - low,
                    LOW_centered_CDF=np.array(row['LOW_cdf']), HIGH_centered_CDF=np.array(row['HIGH_cdf']), radial_grid_deg=grid)
                if profile_path.exists():
                    with np.load(profile_path, allow_pickle=False) as saved:
                        for key, value in arrays.items():
                            np.testing.assert_array_equal(saved[key], value)
                else:
                    with profile_path.open('xb') as stream:
                        np.savez_compressed(stream, **arrays)
                if cell == lock['cells'][0] and seed == SEEDS[0]:
                    model, cp = load_retipath(ROOT / reference['path'])
                    split = load_split(cell, start, stop)
                    if start == 16:
                        selections, _ = frozen_observations(phase['cells'][cell], split, model)
                    else:
                        with np.load(CONFIRM / 'sampling_before_targets.npz', allow_pickle=False) as sampling:
                            selections = {label: sampling[safe + '_' + label + '_indices'].tolist() for label in ('LOW', 'HIGH')}
                    new, _, checks = rf_modes(model, split, selections, cp['cone_positions_degs'].double().numpy(), grid)
                    errors = []
                    for key in new:
                        if key == 'status':
                            assert new[key] == row[key]
                        else:
                            expected, actual = np.asarray(row[key]), np.asarray(new[key])
                            errors.append(float(np.max(np.abs(expected - actual))))
                            np.testing.assert_allclose(actual, expected, rtol=32*np.finfo(np.float64).eps,
                                                       atol=32*np.finfo(np.float64).eps)
                    replays.append({'range': f'[{start},{stop})', 'cell_id': cell, 'seed': seed,
                                    'max_RF_replay_abs_error': max(errors), 'full_prefix': True,
                                    'comparison_tolerance': '32 binary64 eps, numerical verification only', **checks})
    csv_rows(destination / 'per_cell_seed.csv', rows)
    summaries, repeatability = [], []
    valid = {(r['range'], r['cell_id'], r['seed']): r for r in rows if r['status'] == 'VERIFIED'}
    metrics = ['LOW_gain_median', 'HIGH_gain_median', 'gain_log_HIGH_over_LOW', 'TV', 'centroid_shift_deg',
               'LOW_centroid_x_deg', 'LOW_centroid_y_deg', 'HIGH_centroid_x_deg', 'HIGH_centroid_y_deg',
               'LOW_radius_mean_profile_deg', 'HIGH_radius_mean_profile_deg',
               'LOW_radius_observation_median_deg', 'HIGH_radius_observation_median_deg',
               'delta_radius_mean_profile_deg', 'delta_radius_observation_median_deg',
               'centered_radial_CDF_L1_deg', 'centered_radial_CDF_max']
    for label in ['[16,20)', '[240,300)']:
        cells = sorted({r['cell_id'] for r in rows if r['range'] == label and r['status'] == 'VERIFIED'})
        for seed in [*SEEDS, 'aggregate']:
            values = np.array([[np.mean([valid[label, cell, s][m] for s in SEEDS]) if seed == 'aggregate'
                                else valid[label, cell, seed][m] for m in metrics] for cell in cells])
            summaries.append({'model': 'RetiPath', 'range': label, 'seed': seed, 'n_cells': len(cells),
                **{metric: float(values[:, i].mean()) for i, metric in enumerate(metrics)},
                'HIGH_smaller_cells': int((values[:, metrics.index('delta_radius_observation_median_deg')] < 0).sum()),
                'HIGH_larger_cells': int((values[:, metrics.index('delta_radius_observation_median_deg')] > 0).sum())})
        for cell in cells:
            deltas = [np.array(valid[label, cell, seed]['delta_P']) for seed in SEEDS]
            cosines = [float(deltas[i] @ deltas[j] / (np.linalg.norm(deltas[i]) * np.linalg.norm(deltas[j])))
                       for i, j in itertools.combinations(range(3), 2)]
            repeatability.append({'model': 'RetiPath', 'range': label, 'cell_id': cell,
                'pairwise_seed_delta_P_cosines': cosines, 'all_positive': all(x > 0 for x in cosines)})
    csv_rows(destination / 'population.csv', summaries)
    csv_rows(destination / 'seed_repeatability.csv', repeatability)
    verify_sources()
    write_once(destination / 'complete.json', {'status': 'VERIFIED', 'rows': len(rows), 'valid_rows': len(valid),
        'replays': replays, 'new_contexts': 0, 'new_model_or_metrics': False, 'new_training': False,
        'interpretation': 'Gain, normalized spatial profile and size direction are separate model-internal evidence.'})
    print('DYNAMIC RF COMPLETE: original final-model evidence retained; replay roundoff recorded on both consumed blocks.')


if __name__ == '__main__':
    main()
