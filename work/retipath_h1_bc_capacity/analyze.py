from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from pathlib import Path
import traceback

import numpy as np
import torch

from common import (ROOT, OUT, SEEDS, load_json, save_json, sha, stage_data, restore,
                    evaluate, write_csv, digest_state, verify_sources)
from retipath_final_common import OUT as FINAL, identity, load_retipath
from retipath_final_prediction import load_split
from retipath_phase2_analyze import frozen_observations
from retipath_spatial_ei_report import profile_metrics, radial_cdf
from retipath_phase2_common import paired_bootstrap
from training.mechanistic_retina.losses import expected_bernoulli_nll


def basis_stats(model, data, mask, common: dict) -> list[dict]:
    parts = []
    with torch.no_grad():
        for start in range(0, len(data.cone_drive), 8):
            h1 = model.h1(data.cone_drive[start:start+8], amplitude=model.gates.values(frozenset()).h1)
            values = model.input_basis(h1.modulated_cones / model.h1_rms)
            parts.append(values[mask[start:start+8]].reshape(-1, 4).numpy())
    values = np.concatenate(parts).astype(np.float64)
    rms = np.sqrt(np.square(values).mean(0))
    corr = np.corrcoef(values.T)
    gram = values.T @ values / len(values)
    quantiles = np.quantile(values, [.01, .1, .5, .9, .99], axis=0)
    rows = []
    for k in range(4):
        rows.append(common | {'record_type': 'activation', 'basis_index': k+1, 'samples': len(values),
                    'mean': float(values[:, k].mean()), 'std': float(values[:, k].std()), 'rms': float(rms[k]),
                    'q01': float(quantiles[0, k]), 'q10': float(quantiles[1, k]), 'q50': float(quantiles[2, k]),
                    'q90': float(quantiles[3, k]), 'q99': float(quantiles[4, k]),
                    'near_zero_fraction_abs_below_001': float((np.abs(values[:, k]) < .01).mean()),
                    'activation_rms_below_001': bool(rms[k] < .01)})
    for i in range(4):
        for j in range(i+1, 4):
            distance = math.sqrt(max(0., 2-2*gram[i, j]/(rms[i]*rms[j])))
            rows.append(common | {'record_type': 'channel_pair', 'basis_index': i+1, 'basis_j': j+1,
                        'correlation': float(corr[i, j]), 'unit_rms_function_difference': distance,
                        'correlation_above_0995': bool(corr[i, j] > .995)})
    weights = model.basis_weights.detach().numpy()
    for p, pathway in enumerate(('sustained', 'transient')):
        for k in range(4):
            rows.append(common | {'record_type': 'mixing', 'pathway': pathway, 'basis_index': k+1,
                        'weight': float(weights[p, k]), 'weight_below_001': bool(weights[p, k] < .01)})
    return rows


def rf_metrics(model, dev, selections: dict, positions: np.ndarray, grid: np.ndarray) -> tuple[dict, dict]:
    model.double().eval().requires_grad_(False)
    before = digest_state(model.state_dict())
    contexts, max_error = {}, 0.
    for label in ('LOW', 'HIGH'):
        profiles, cdfs, gains, radii = [], [], [], []
        for sequence, target in selections[label]:
            x = dev.cone_drive[sequence:sequence+1, :target+1].double().clone().requires_grad_()
            history = dev.spike_events[sequence:sequence+1, :target+1].double().detach()
            z = model(x, observed_counts=history).logits[0, -1, 0]
            jac, = torch.autograd.grad(z, x)
            j = jac.detach().numpy()[0]
            energy = np.square(j).sum()
            assert np.isfinite(j).all() and energy > 0
            p = np.square(j).sum(0) / energy
            assert abs(p.sum()-1) < 1e-12
            profiles.append(p)
            cdfs.append(radial_cdf(p, positions, grid))
            gains.append(math.sqrt(energy))
            radii.append(profile_metrics(p, positions)[1])
        profile = np.mean(profiles, axis=0)
        center, radius = profile_metrics(profile, positions)
        contexts[label] = {'gain_median': float(np.median(gains)), 'centroid_x_deg': float(center[0]),
                           'centroid_y_deg': float(center[1]), 'radius_mean_profile_deg': radius,
                           'radius_observation_median_deg': float(np.median(radii)),
                           'profile': profile, 'cdf': np.mean(cdfs, axis=0)}
        sequence, target = selections[label][0]
        with torch.no_grad():
            full = model(dev.cone_drive[sequence:sequence+1].double(),
                         observed_counts=dev.spike_events[sequence:sequence+1].double()).logits[0, target, 0]
            prefix = model(dev.cone_drive[sequence:sequence+1, :target+1].double(),
                           observed_counts=dev.spike_events[sequence:sequence+1, :target+1].double()).logits[0, target, 0]
        max_error = max(max_error, float((full-prefix).abs()))
        assert max_error < 1e-10
    assert digest_state(model.state_dict()) == before
    low, high = contexts['LOW'], contexts['HIGH']
    delta = high['profile']-low['profile']
    cdf_delta = high['cdf']-low['cdf']
    result = {'status': 'VERIFIED', 'LOW_n': len(selections['LOW']), 'HIGH_n': len(selections['HIGH']),
              'gain_log_HIGH_over_LOW': math.log(high['gain_median']/low['gain_median']),
              'TV': float(np.abs(delta).sum()/2),
              'centroid_shift_deg': math.hypot(high['centroid_x_deg']-low['centroid_x_deg'],
                                             high['centroid_y_deg']-low['centroid_y_deg']),
              'delta_radius_mean_profile_deg': high['radius_mean_profile_deg']-low['radius_mean_profile_deg'],
              'delta_radius_observation_median_deg': high['radius_observation_median_deg']-low['radius_observation_median_deg'],
              'centered_radial_CDF_max': float(np.abs(cdf_delta).max()),
              'centered_radial_CDF_L1_deg': float(np.trapezoid(np.abs(cdf_delta), grid)), 'delta_P': delta.tolist()}
    for label, values in contexts.items():
        result.update({label + '_' + k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in values.items()})
    return result, {'full_prefix_error': max_error, 'weights_unchanged': True,
                    'observations': sum(map(len, selections.values()))}


def analyze_job(out_string: str, cell: str, seed: int) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(out_string)
    protocol = load_json(out / 'checkpoints/protocol.json')
    lock = load_json(out / 'checkpoints/training_complete.json')
    ref = protocol['cells'][cell]
    folder = out / 'checkpoints' / cell.replace('#', '_') / str(seed)
    destination = folder / 'analysis.json'
    if destination.exists():
        saved = load_json(destination)
        assert saved['protocol_sha256'] == protocol['protocol_sha256']
        return saved
    rows, basis_rows, rf_rows, checks = [], [], [], {}
    arrays = {}
    a_ref = ref['A'][str(seed)]
    assert sha(Path(a_ref['path'])) == a_ref['sha256']
    saved_A = load_json(FINAL / 'prediction/cells' / (cell.replace('#', '_') + '.json'))
    for start, stop in ((16, 20), (20, 60)):
        split = load_split(cell, start, stop)
        contract = identity(split)
        label = f'[{start},{stop})'
        a_row = next(r for r in saved_A['rows'] if r['range'] == label and r['model'] == 'RetiPath' and int(r['seed']) == seed)
        assert a_row['checkpoint_sha256'] == a_ref['sha256']
        assert all(a_row[k] == value for k, value in contract.items()), 'DATA MISMATCH: A cache contract'
        with np.load(FINAL / 'prediction/cells' / (cell.replace('#', '_') + '.npz'), allow_pickle=False) as saved:
            a_logits = torch.from_numpy(saved[a_row['logits_key']].copy())
        a_nll = float(expected_bernoulli_nll(a_logits.double(), split.spike_events.double(), split.valid_mask))
        assert abs(a_nll-a_row['nll']) < 1e-12
        scores = {'A': a_nll}
        for condition in ('B', 'C'):
            path = folder / condition / 'refit.pt'
            assert sha(path) == lock['files'][str(path.relative_to(out))]
            model, cp = restore(path)
            assert cp['phase'] == 'refit' and cp['step'] == cp['best_step']
            assert cp['protocol_sha256'] == protocol['protocol_sha256']
            model.eval().requires_grad_(False)
            before = digest_state(model.state_dict())
            scores[condition], logits = evaluate(model, split)
            assert digest_state(model.state_dict()) == before
            arrays[f'{start}_{condition}'] = logits.numpy()
        rows.append({'record_type': 'per_cell', 'range': label, 'cell_id': cell, 'group': ref['group'], 'seed': seed,
                     **{c + '_nll': nll for c, nll in scores.items()},
                     'C_minus_B': scores['C']-scores['B'], 'B_minus_A': scores['B']-scores['A'],
                     'C_minus_A': scores['C']-scores['A'], 'scored_bins': int(split.valid_mask.sum()), **contract})
    for condition in ('B', 'C'):
        for phase in ('inner', 'refit'):
            data, validation, mask = stage_data(ref, phase)
            model, cp = restore(folder / condition / f'{phase}.pt')
            model.eval().requires_grad_(False)
            train_nll = evaluate(model, data)[0]
            if phase == 'refit':
                assert train_nll == cp['refit_train_nll']
            else:
                assert evaluate(model, validation)[0] == cp['best_replay_inner_nll']
            common = {'cell_id': cell, 'group': ref['group'], 'seed': seed, 'condition': condition, 'phase': phase}
            basis_rows.extend(basis_stats(model, data, mask, common | {'checkpoint': 'selected' if phase == 'inner' else 'final'}))
            model.load_state_dict(cp['initial_state'], strict=True)
            basis_rows.extend(basis_stats(model, data, mask, common | {'checkpoint': 'initial'}))
            checks[f'{condition}/{phase}'] = {'score_replay_exact': True, 'selected_step': cp['best_step'],
                'stop_step': cp['step'], 'optimizer_steps': cp['optimizer_steps'], 'parameters': cp['trainable_parameters'],
                'fixed_buffers_unchanged': cp['fixed_buffers_unchanged'],
                'gradient_nonzero_seen': cp['gradient_nonzero_seen'], 'actually_changed': cp['actually_changed']}
            if phase == 'refit':
                tail = [r for r in cp['curve'] if r['step'] >= cp['step']/2]
                for k in range(4):
                    basis_rows.append(common | {'checkpoint': 'last_half_refit', 'record_type': 'persistent', 'basis_index': k+1,
                        'observations': len(tail), 'activation_always_rms_below_001': all(r['basis_rms'][k] < .01 for r in tail),
                        'sustained_weight_always_below_001': all(r['basis_weights'][0][k] < .01 for r in tail),
                        'transient_weight_always_below_001': all(r['basis_weights'][1][k] < .01 for r in tail),
                        'always_correlated_above_0995_with_any_channel': all(any(r['basis_correlation'][k][j] > .995 for j in range(4) if j != k) for r in tail)})
    dev = load_split(cell, 16, 20)
    a_model, a_cp = load_retipath(Path(a_ref['path']))
    selections, provenance = frozen_observations(ref, dev, a_model)
    for condition in ('A', 'B', 'C'):
        common = {'cell_id': cell, 'group': ref['group'], 'seed': seed, 'condition': condition, 'range': '[16,20)'}
        if not all(selections.values()):
            rf_rows.append(common | {'status': 'UNVERIFIED', 'reason': 'missing existing HIGH/LOW observations'})
            continue
        if condition == 'A':
            old_path = Path(a_ref['path']).parent.parent / 'analysis.json'
            old = load_json(old_path)
            assert old['checks']['C']['refit_checkpoint_sha256'] == a_ref['sha256']
            assert old['checks']['C']['observation_provenance'] == provenance
            result = next(r for r in old['RF'] if r['condition'] == 'C')
            rf_rows.append(result | common | {'reused_frozen_A': True})
        else:
            model, cp = restore(folder / condition / 'refit.pt')
            rf, quality = rf_metrics(model, dev, selections, cp['cone_positions_degs'].double().numpy(),
                                     np.array(protocol['radial_grid_deg']))
            rf_rows.append(common | rf | {'reused_frozen_A': False})
            checks[condition + '/RF'] = quality
    checks['RF_context_provenance'] = provenance
    result = {'protocol_sha256': protocol['protocol_sha256'], 'prediction': rows, 'basis': basis_rows,
              'RF': rf_rows, 'checks': checks}
    with (folder / 'prediction_logits.npz').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    save_json(destination, result)
    return result


def aggregate(out: Path, results: list[dict]) -> None:
    predictions = [r for result in results for r in result['prediction']]
    cells = sorted({r['cell_id'] for r in predictions})
    for label in ('[16,20)', '[20,60)'):
        for cell in cells:
            values = [r for r in predictions if r['range'] == label and r['cell_id'] == cell]
            assert len(values) == 3
            predictions.append({**values[0], 'seed': 'aggregate',
                **{key: float(np.mean([r[key] for r in values])) for key in ('A_nll', 'B_nll', 'C_nll', 'C_minus_B', 'B_minus_A', 'C_minus_A')}})
    summaries = []
    for label in ('[16,20)', '[20,60)'):
        for group in ('population', *sorted({r['group'] for r in predictions})):
            cohort = sorted({r['cell_id'] for r in predictions if r['range'] == label and (group == 'population' or r['group'] == group)})
            draws = np.random.default_rng(2026091399).integers(0, len(cohort), size=(10000, len(cohort)))
            for seed in (*SEEDS, 'aggregate'):
                selected = sorted([r for r in predictions if r['range'] == label and r['cell_id'] in cohort and r['seed'] == seed], key=lambda r: r['cell_id'])
                assert len(selected) == len(cohort)
                for comparison in ('C_minus_B', 'B_minus_A', 'C_minus_A'):
                    summaries.append({'record_type': 'summary', 'range': label, 'group': group, 'seed': seed,
                        'comparison': comparison, 'n_cells': len(cohort),
                        **{c + '_nll': float(np.mean([r[c + '_nll'] for r in selected])) for c in ('A', 'B', 'C')},
                        **paired_bootstrap(np.array([r[comparison] for r in selected]), draws)})
    write_csv(out / 'prediction.csv', predictions + summaries)
    write_csv(out / 'basis_diagnostics.csv', [r for result in results for r in result['basis']])
    rf_rows = [r for result in results for r in result['RF']]
    for cell in cells:
        for condition in ('A', 'B', 'C'):
            rows = [r for r in rf_rows if r['cell_id'] == cell and r['condition'] == condition]
            if not all(r['status'] == 'VERIFIED' for r in rows):
                continue
            aggregate_row = dict(rows[0], seed='aggregate')
            for k in rows[0]:
                if k in ('cell_id', 'group', 'seed', 'condition', 'range', 'status', 'refit_updates', 'reused_frozen_A'):
                    continue
                if isinstance(rows[0][k], (int, float, list)):
                    mean = np.mean([r[k] for r in rows], axis=0)
                    aggregate_row[k] = mean.tolist() if isinstance(mean, np.ndarray) else float(mean)
            aggregate_row.pop('refit_updates', None)
            rf_rows.append(aggregate_row)
    write_csv(out / 'rf_secondary.csv', rf_rows)


def guarded_analyze_job(*args):
    try:
        return analyze_job(*args)
    except Exception:
        traceback.print_exc()
        raise


def run(out: Path, workers: int) -> None:
    protocol = load_json(out / 'checkpoints/protocol.json')
    assert sha(out / 'PROTOCOL.md') == protocol['protocol_sha256']
    verify_sources(protocol, out)
    complete = load_json(out / 'checkpoints/training_complete.json')
    assert len(complete['files']) == 132
    for path, digest in complete['files'].items():
        assert sha(out / path) == digest
    save_json(out / 'checkpoints/evaluation_lock.json', {'protocol_sha256': protocol['protocol_sha256'],
              'training_complete_sha256': sha(out / 'checkpoints/training_complete.json'),
              'analysis_code_sha256': sha(Path(__file__)),
              'dependency_hashes': {name: sha(ROOT / name) for name in (
                  'work/retipath_final_common.py', 'work/retipath_final_prediction.py',
                  'work/retipath_phase2_analyze.py', 'work/retipath_spatial_ei_report.py')},
              'ranges': [[16, 20], [20, 60]], 'new_targets': False})
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(guarded_analyze_job, str(out), cell, seed) for cell in protocol['cells'] for seed in SEEDS]
        for future in as_completed(futures):
            results.append(future.result())
            print('ANALYSIS', len(results), '/66', flush=True)
    aggregate(out, results)
    save_json(out / 'checkpoints/evaluation_complete.json', {'cell_seed_groups': 66,
              'files': {p.name: sha(p) for p in (out / 'prediction.csv', out / 'basis_diagnostics.csv', out / 'rf_secondary.csv')}})
    print('ALL FROZEN ANALYSES COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    run(args.out, args.workers)
