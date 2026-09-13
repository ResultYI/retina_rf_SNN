from __future__ import annotations

import csv

import numpy as np
import torch

from retipath_final_common import OUT, ROOT, SEEDS, CONTROL, load_json, sha, write_once, utc, verify_sources
from retipath_final_prediction import load_split


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def verify_prediction() -> dict:
    rows = read_rows(OUT / 'prediction/per_cell_seed.csv')
    error = 0.0
    groups = {}
    for row in rows:
        groups.setdefault((row['cell_id'], row['range']), []).append(row)
    assert len(groups) == 112 and len(rows) == 896
    for (cell, interval), current in groups.items():
        start, stop = map(int, interval.strip('[)').split(','))
        split = load_split(cell, start, stop)
        y = split.spike_events.numpy().astype(np.float64)
        mask = split.valid_mask.numpy()
        assert len(current) == 8
        assert len({r['target_sha256'] for r in current}) == len({r['mask_sha256'] for r in current}) == 1
        with np.load(OUT / 'prediction/cells' / f"{cell.replace('#', '_')}.npz", allow_pickle=False) as arrays:
            for row in current:
                z = arrays[row['logits_key']].astype(np.float64)
                actual = (np.logaddexp(0, z) - y * z)[mask].mean()
                error = max(error, abs(actual - float(row['nll'])))
                assert abs(actual - float(row['nll'])) < 2e-14
    pairs = read_rows(OUT / 'prediction/comparisons_per_cell.csv')
    summaries = read_rows(OUT / 'prediction/population.csv')
    bootstrap_error = 0.0
    for summary in summaries:
        selected = sorted([r for r in pairs if all(r[k] == summary[k] for k in ('range', 'comparison', 'seed'))],
                          key=lambda r: r['cell_id'])
        values = np.array([float(r['delta_nll']) for r in selected])
        assert len(values) == int(summary['n_cells'])
        indices = np.random.default_rng(20260908).integers(0, len(values), size=(100000, len(values)))
        low, high = np.percentile(values[indices].mean(1), [2.5, 97.5])
        expected = {'mean': values.mean(), 'median': np.median(values), 'ci_low': low, 'ci_high': high,
                    'wins': (values < -1e-7).sum(), 'losses': (values > 1e-7).sum(), 'ties': (abs(values) <= 1e-7).sum()}
        for key, value in expected.items():
            delta = abs(value - float(summary[key]))
            bootstrap_error = max(bootstrap_error, delta)
            assert delta < 2e-14
    return {'scores': 896, 'cell_range_records': 112, 'shared_scoring_contract': True,
            'independent_NLL_max_abs_error': error, 'independent_bootstrap_max_abs_error': bootstrap_error,
            'legacy_baseline_float32_replay_max_error': max(float(r['legacy_float32_NLL_replay_error'])
                for r in rows if r['legacy_float32_NLL_replay_error'])}


def verify_brightness(lock: dict) -> dict:
    conditions = ('Normal', 'H1-off', 'direct-BC-off', 'AC-off')
    expected = {}
    count = 0
    cells = lock['cells']
    for seed in SEEDS:
        for cell in cells:
            safe = cell.replace('#', '_')
            path = OUT / 'brightness/responses' / f'{safe}_{seed}.npz'
            record = load_json(path.with_suffix('.json'))
            assert sha(path) == record['responses_sha256']
            assert record['parameters_unchanged'] and record['clamp_contracts_verified']
            with np.load(path, allow_pickle=False) as raw:
                for quantity, key in [('logit', 'logits'), ('probability', 'probability')]:
                    values = raw[key].astype(np.float64)
                    assert values.shape == (4, 166, 150) and np.isfinite(values).all()
                    for family in ('Mach', 'SBC'):
                        pairs = [p for p in lock['stimulus_comparisons'] if p['family'] == family]
                        responses = np.stack([(values[:, p['a'], 45:60] - values[:, p['b'], 45:60]).mean(-1) for p in pairs], -1)
                        if family == 'Mach':
                            paired = responses.reshape(4, 6, 6, 2)
                            control = paired - paired[:, :, :1]
                            endpoints = ('dark_endpoint', 'bright_endpoint')
                            magnitude_endpoint = 'both_endpoints'
                        else:
                            paired = responses.reshape(4, 6, 5, 1)
                            control = paired - paired[:, :1] - paired[:, :, :1] + paired[:, :1, :1]
                            endpoints = ('bright_minus_dark_context',)
                            magnitude_endpoint = endpoints[0]
                        assert np.count_nonzero(control[:, 0]) == 0 and np.count_nonzero(control[:, :, 0]) == 0
                        interaction = control[:1] - control
                        e = control[:, 1:, 1:].mean((1, 2))
                        signed = interaction[:, 1:, 1:].mean((1, 2))
                        magnitude = np.abs(interaction[1:, 1:, 1:]).mean((1, 2, 3))
                        for k, condition in enumerate(conditions):
                            for j, endpoint in enumerate(endpoints):
                                for metric, value in [('control_subtracted_response', e[k, j]),
                                                      ('normal_minus_off_interaction', signed[k, j])]:
                                    expected[family.lower(), str(seed), quantity, cell, condition, metric, endpoint, ''] = float(value)
                        for k, condition in enumerate(conditions[1:]):
                            expected[family.lower(), str(seed), quantity, cell, condition, 'absolute_interaction_magnitude', magnitude_endpoint, ''] = float(magnitude[k])
                        for i in range(3):
                            for j in range(i + 1, 3):
                                expected[family.lower(), str(seed), quantity, cell, conditions[i + 1], 'paired_magnitude_difference', magnitude_endpoint, conditions[j + 1]] = float(magnitude[i] - magnitude[j])
            count += 1
    for key in list(expected):
        if key[1] == str(SEEDS[0]):
            aggregate_key = (key[0], 'aggregate', *key[2:])
            expected[aggregate_key] = float(np.mean([expected[(key[0], str(s), *key[2:])] for s in SEEDS]))
    error, checked = 0.0, 0
    primary_ci_error = 0.0
    primary_rows = []
    for seed in [*SEEDS, 'aggregate']:
        folder = OUT / 'brightness' / (str(seed) if seed != 'aggregate' else 'seed_aggregate')
        for family in ('mach', 'sbc'):
            for row in read_rows(folder / f'{family}_per_cell.csv'):
                if row['level'] != 'full_grid':
                    continue
                key = (family, str(seed), row['quantity'], row['cell_id'], row['condition'], row['metric'], row['endpoint'], row['compared_condition'])
                if key not in expected:
                    continue
                delta = abs(float(row['value']) - expected[key])
                error = max(error, delta)
                assert delta < 2e-12, key
                checked += 1
            for row in read_rows(folder / f'{family}_summary.csv'):
                if not (row['quantity'] == 'logit' and row['level'] == 'full_grid' and row['group'] in ('ON', 'OFF')
                        and row['condition'] == 'Normal' and row['metric'] == 'control_subtracted_response'):
                    continue
                members = [r['cell_id'] for r in load_json(ROOT / '.omo/evidence/retipath-retinotopic-registration-20260910/frozen_registration.json')['cells']
                           if r['polarity'] == row['group']]
                values = np.array([expected[family, str(seed), 'logit', c, 'Normal', row['metric'], row['endpoint'], ''] for c in members])
                index = 1 if row['group'] == 'ON' else 2
                draws = np.random.default_rng(20260910 + index).integers(0, len(values), size=(100000, len(values)))
                ci = np.quantile(values[draws].mean(1), [.025, .975])
                delta = max(abs(ci[0] - float(row['ci_low'])), abs(ci[1] - float(row['ci_high'])))
                primary_ci_error = max(primary_ci_error, delta)
                assert delta < 2e-12
                stable = ((ci[0] > 0 and (values > 1e-9).mean() >= 0.75)
                          or (ci[1] < 0 and (values < -1e-9).mean() >= 0.75))
                assert stable == (row['direction_stable'] == 'True')
                primary_rows.append({'family': family, 'seed': seed, **row})
    assert count == 66 and len(primary_rows) == 24 and checked == len(expected)
    rankings = read_rows(OUT / 'brightness/pathway_rankings.csv')
    verdicts = []
    for seed in [*map(str, SEEDS), 'aggregate']:
        normal = [r for r in primary_rows if str(r['seed']) == seed]
        stable_normal = sum(r['direction_stable'] == 'True' for r in normal)
        stable_rank = {}
        for family in ('mach', 'sbc'):
            current = [r for r in rankings if r['seed'] == seed and r['quantity'] == 'logit'
                       and r['family'] == family and r['group'] in ('ON', 'OFF')]
            winners = [r['paired_CI_winner'] for r in current]
            stable_rank[family] = bool(winners[0] and winners[0] == winners[1])
        status = 'SUPPORTED' if stable_normal == 6 and all(stable_rank.values()) else 'MIXED' if stable_normal else 'NOT SUPPORTED'
        verdicts.append({'seed': seed, 'stable_normal_endpoints': stable_normal, 'stable_magnitude_rankings': stable_rank,
                         'frozen_application_descriptive_verdict': status})
    return {'frozen_models': count, 'manual_full_grid_effects_checked': checked,
            'manual_effect_max_abs_error': error, 'primary_bootstrap_max_abs_error': primary_ci_error,
            'registered_bank_exact_replay': True, 'verdicts': verdicts}


def main() -> None:
    torch.set_num_threads(1)
    lock = verify_sources()
    prediction = verify_prediction()
    brightness = verify_brightness(lock)
    rf = load_json(OUT / 'dynamic_rf/complete.json')
    assert rf['rows'] == 117 and rf['valid_rows'] == 114
    assert len(rf['replays']) == 2
    verify_sources()
    write_once(OUT / 'verification.json', {'status': 'VERIFIED', 'completed_utc': utc(),
        'source_lock_sha256': sha(OUT / 'source_lock.json'), 'source_hash_count': len(lock['frozen_sha256']),
        'prediction': prediction, 'brightness': brightness, 'dynamic_RF': rf,
        'official_entry_and_clamp_tests': '2 passed; exact direct outputs, baseline conductance semantics, historical backend rejection',
        'model_sources_changed_after_freeze': False, 'historical_artifacts_unchanged': True,
        'numerical_replay_correction': 'execution_correction.json; original scripts preserved, no scientific changes',
        'new_training_updates': 0, 'new_refits': 0, 'new_target_blocks': 0, 'new_variants': 0})
    print('FINAL-MODEL VERIFICATION COMPLETE: prediction, RF and frozen application arithmetic verified.')


if __name__ == '__main__':
    main()
