from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from derive_development import OUT, ROOT, OLD, BIAS, PRIMARY, DERIVED, ADAPTER, MOVIE, CONDITIONS, now, nll, read_csv, sha, write_csv, write_json
from spatial_features import features


def mean(a: np.ndarray) -> float | None:
    return float(a.mean()) if a.size else None


def ranks(a: np.ndarray) -> np.ndarray:
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    return (np.cumsum(counts)-(counts-1)/2)[inv]


def summary(values: list, indices: np.ndarray | None = None) -> dict:
    good = all(v is not None for v in values)
    row = dict(cells=len(values), defined_cells=sum(v is not None for v in values), available=good, mean=None, median=None, positive=None, negative=None, zero=None,
        mean_ci_low=None, mean_ci_high=None, median_ci_low=None, median_ci_high=None)
    if good:
        a = np.asarray(values)
        row.update(mean=float(a.mean()), median=float(np.median(a)), positive=int((a > 0).sum()), negative=int((a < 0).sum()), zero=int((a == 0).sum()))
        if indices is not None:
            b = a[indices]
            for key, dist in (('mean', b.mean(axis=1)), ('median', np.median(b, axis=1))):
                lo, hi = np.quantile(dist, [.025, .975], method='linear')
                row.update({key+'_ci_low': float(lo), key+'_ci_high': float(hi)})
    return row


def matching(f: dict, delta: np.ndarray, feature: str) -> tuple[dict, list]:
    q, strata = f['S_quintile'], f[feature+'_quintile']
    mass = np.array([min(int(((q == 1) & (strata == s)).sum()), int(((q == 5) & (strata == s)).sum())) for s in range(1, 6)])
    total = int(mass.sum())
    result = dict(E=0.0 if total else None, overlap_mass=total, Q1_bins=int((q == 1).sum()), Q5_bins=int((q == 5).sum()), retained_strata=int((mass > 0).sum()))
    for key in ('U', 'LSC', 'C', 'R'):
        result['residual_'+key+'_difference'] = 0.0 if total else None
    for qkey in ('Q1', 'Q5'):
        result[qkey+'_overlap_fraction'] = total/result[qkey+'_bins'] if result[qkey+'_bins'] else None
    details = []
    for s in range(1, 6):
        low, high = (q == 1) & (strata == s), (q == 5) & (strata == s)
        w = float(mass[s-1]/total) if total else 0.0
        e = float(delta[high].mean()-delta[low].mean()) if mass[s-1] else None
        row = dict(stratum=s, low_bins=int(low.sum()), high_bins=int(high.sum()), overlap_mass=int(mass[s-1]), weight=w, E=e)
        for key in ('U', 'LSC', 'C', 'R'):
            row['Q1_'+key], row['Q5_'+key] = mean(f[key][low]), mean(f[key][high])
            if w:
                result['residual_'+key+'_difference'] += w*(row['Q5_'+key]-row['Q1_'+key])
        if w:
            result['E'] += w*e
        details.append(row)
    return result, details


def analyze(split: str, stimuli: dict, predictions: dict, lock: dict) -> dict:
    tables = {k: [] for k in ('mismatch_quantile_pathway_loss', 'central_drive_quantile_pathway_loss', 'per_cell_context_selectivity',
        'central_drive_matched_control', 'lsc_matched_control', 'matched_strata', 'continuous_associations', 'signed_context_descriptive')}
    group = {r['cell_id']: r['group'] for r in read_csv(ROOT / 'output/audits/macaque_fixed_alignment_experiment_20260905/per_cell_results.csv')}
    output = OUT / (split+'_stimulus_features.parquet')
    assert not output.exists()
    writer = None
    total = 0
    try:
        for cid, stimulus in stimuli.items():
            feature_path = OUT / ('development_features_'+cid.replace('#', '_')+'.npz')
            assert sha(feature_path) == lock['features_sha256'][feature_path.name]
            with np.load(feature_path) as bundle:
                f = {k: bundle[k] for k in bundle.files} if split == 'development' else features(stimulus, {k: bundle[k+'_edges'] for k in ('S', 'U', 'LSC')})
            saved = predictions[cid]
            mask = saved['valid_mask']
            assert torch.equal(mask, stimulus['valid_mask']) and tuple(saved['source_image_ids']) == tuple(stimulus['source_image_ids']) and tuple(saved['trial_indices']) == tuple(stimulus['trial_indices'])
            y = saved['target'][mask].double()
            z0 = saved['normal'][mask].double()
            normal = (torch.nn.functional.softplus(z0)-y*z0).numpy()
            deltas, losses = {}, {}
            row = dict(split=split, cell_id=cid, group=group[cid], scored_bins=int(mask.sum()))
            for pathway in ('BC', 'AC'):
                z = saved['recalibrated'][pathway][mask]
                assert z.dtype == torch.float64 and z.shape == y.shape
                loss = (torch.nn.functional.softplus(z)-y*z).numpy()
                delta = loss-normal
                losses[pathway], deltas[pathway] = loss, delta
                q, u = f['S_quintile'], f['U_quintile']
                e = float(delta[q == 5].mean()-delta[q == 1].mean()) if (q == 1).any() and (q == 5).any() else None
                denominator = float(delta.mean())
                floor = 64*np.finfo(np.float64).eps*max(1., float(np.abs(delta).mean()))
                defined = denominator > floor
                row.update({f'E_{pathway}': e, f'B_{pathway}': denominator, f'B_floor_{pathway}': floor, f'normalized_defined_{pathway}': defined,
                    f'N_{pathway}': e/denominator if defined and e is not None else None,
                    f'G_{pathway}': float(delta[u == 5].mean()-delta[u == 1].mean()) if (u == 1).any() and (u == 5).any() else None})
                for feature, dest in (('S', 'mismatch_quantile_pathway_loss'), ('U', 'central_drive_quantile_pathway_loss')):
                    for k in range(1, 6):
                        selected = f[feature+'_quintile'] == k
                        values = delta[selected]
                        qr = dict(split=split, cell_id=cid, group=group[cid], pathway=pathway, quintile=k, bins=int(selected.sum()), mean_delta_loss=mean(values),
                            median_delta_loss=float(np.median(values)) if values.size else None, normal_nll=mean(normal[selected]), recalibrated_off_nll=mean(loss[selected]), event_rate=mean(y.numpy()[selected]))
                        qr.update({'mean_'+key: mean(f[key][selected]) for key in ('C', 'U', 'R', 'V', 'LSC')})
                        tables[dest].append(qr)
                for feature, dest in (('U', 'central_drive_matched_control'), ('LSC', 'lsc_matched_control')):
                    m, detail = matching(f, delta, feature)
                    if split == 'development' and feature == 'U':
                        assert m['E'] is not None
                    row[f'E_{pathway}_{feature}'] = m['E']
                    if feature == 'U':
                        row[f'N_{pathway}_U'] = m['E']/denominator if defined and m['E'] is not None else None
                    prefix = dict(split=split, cell_id=cid, group=group[cid], pathway=pathway, matching_feature=feature)
                    tables[dest].append({**prefix, **m, 'unconditional_denominator': denominator, 'normalized_E': m['E']/denominator if defined and m['E'] is not None else None})
                    tables['matched_strata'].extend({**prefix, **r} for r in detail)
                for feature in ('S', 'U'):
                    a, b = ranks(f[feature]), ranks(delta)
                    rho = float(np.corrcoef(a, b)[0, 1]) if np.ptp(a) and np.ptp(b) else None
                    tables['continuous_associations'].append(dict(split=split, cell_id=cid, group=group[cid], pathway=pathway, feature=feature, spearman=rho))
            row['D'] = row['N_AC']-row['N_BC'] if row['N_AC'] is not None and row['N_BC'] is not None else None
            row['D_U'] = row['N_AC_U']-row['N_BC_U'] if row['N_AC_U'] is not None and row['N_BC_U'] is not None else None
            tables['per_cell_context_selectivity'].append(row)
            for sign, selected in (('negative', f['Q'] < 0), ('positive', f['Q'] > 0)):
                d = deltas['AC'][selected]
                tables['signed_context_descriptive'].append(dict(split=split, cell_id=cid, group=group[cid], sign=sign, bins=int(selected.sum()), zero_Q_bins=int((f['Q'] == 0).sum()),
                    AC_mean_delta_loss=mean(d), AC_median_delta_loss=float(np.median(d)) if d.size else None))
            count = len(y)
            columns = {k: v for k, v in f.items() if v.ndim == 1 and len(v) == count}
            columns.update(cell_id=np.repeat(cid, count), group=np.repeat(group[cid], count), split=np.repeat(split, count), target=y.numpy(), normal_loss=normal,
                BC_off_recal_loss=losses['BC'], AC_off_recal_loss=losses['AC'], delta_BC=deltas['BC'], delta_AC=deltas['AC'], time_seconds=f['live_frame']/150, decoded_frame=f['live_frame']+751)
            table = pa.table(columns)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression='zstd')
            writer.write_table(table)
            total += count
            print(split, cid, count, flush=True)
    finally:
        if writer is not None:
            writer.close()
    assert total == (65760 if split == 'development' else 657600)
    indices = None
    if split == 'development':
        indices = np.random.default_rng(2026090501).integers(0, 22, size=(100000, 22))
        digest = hashlib.sha256(indices.tobytes()).hexdigest()
        assert digest == 'e7c2af0a9e6c1c54358f3f64513ad92790ea4bcc7647e92f2c9ab3a9da901528'
        write_json(OUT / 'bootstrap_identity.json', dict(existing_seed=2026090501, draws=100000, columns=22, index_sha256=digest))
    rows = tables['per_cell_context_selectivity']
    quantities = ('E_BC', 'E_AC', 'N_BC', 'N_AC', 'D', 'E_BC_U', 'E_AC_U', 'N_BC_U', 'N_AC_U', 'D_U', 'E_BC_LSC', 'E_AC_LSC', 'G_BC', 'G_AC')
    boot = {'E_BC', 'E_AC', 'D', 'E_BC_U', 'E_AC_U', 'D_U'}
    population = [dict(split=split, quantity=k, **summary([r[k] for r in rows], indices if k in boot else None)) for k in quantities]
    tables['population_context_selectivity'] = population
    tables['cell_group_summary'] = []
    for g in ('MC_ON', 'MC_OFF', 'PC_ON', 'PC_OFF'):
        for k in ('E_BC', 'E_AC', 'N_BC', 'N_AC', 'D'):
            s = summary([r[k] for r in rows if r['group'] == g])
            tables['cell_group_summary'].append(dict(split=split, group=g, quantity=k, cells=s['cells'], defined_cells=s['defined_cells'], mean=s['mean'], median=s['median']))
    tables['continuous_population'] = [dict(split=split, feature=f, pathway=p, **{k: v for k, v in summary([r['spearman'] for r in tables['continuous_associations'] if r['feature'] == f and r['pathway'] == p]).items() if '_ci_' not in k}) for f in ('S', 'U') for p in ('BC', 'AC')]
    for name, values in tables.items():
        write_csv(OUT / (split+'_'+name+'.csv'), values)
    if split == 'development':
        outliers = []
        for k in ('E_AC', 'D'):
            available = all(r[k] is not None for r in rows)
            removed = sorted(rows, key=lambda r: (-abs(r[k]), r['cell_id']))[0] if available else None
            outliers.append(dict(quantity=k, available=available, removed_cell=removed['cell_id'] if removed else None,
                removed_value=removed[k] if removed else None, mean_after_single_deletion=float(np.mean([r[k] for r in rows if r is not removed])) if removed else None))
        write_csv(OUT / 'outlier_sensitivity.csv', outliers)
        pop = {r['quantity']: r for r in population}
        checks = dict(AC_raw_positive=pop['E_AC']['mean'] > 0, AC_mean_CI_positive=pop['E_AC']['mean_ci_low'] > 0,
            AC_after_deletion_positive=outliers[0]['mean_after_single_deletion'] > 0,
            U_substantial_positive=pop['E_AC_U']['mean_ci_low'] > 0 and pop['E_AC_U']['mean'] >= .5*pop['E_AC']['mean'],
            normalized_D_direction=pop['D']['available'] and pop['D']['mean_ci_low'] > 0 and outliers[1]['mean_after_single_deletion'] > 0,
            LSC_not_fully_explained=pop['E_AC_LSC']['available'] and pop['E_AC_LSC']['mean'] > 0)
        verdict = 'GO — AC CONTEXT SELECTIVITY SUPPORTED' if all(checks.values()) else ('MIXED — FUNCTIONAL SELECTIVITY WEAK OR NONSPECIFIC' if checks['AC_raw_positive'] else 'NO-GO — NO DISTINCT AC CONTEXT SELECTIVITY')
        write_json(OUT / 'development_verdict_lock.json', dict(verdict=verdict, conditions=checks, primary_numbers=population, protocol_sha256=sha(OUT / 'PROTOCOL.md'), locked_utc=now(),
            decision_data='development [16,20) only', consumed_analysis_started=False, feature_lock_sha256=sha(OUT / 'feature_lock.json'),
            development_artifact_sha256={p.name: sha(p) for p in OUT.glob('development_*') if p.is_file() and p.name != 'development_verdict_lock.json'}, outlier_sha256=sha(OUT / 'outlier_sensitivity.csv')))
        write_json(OUT / 'development_verdict_sha256.json', dict(sha256=sha(OUT / 'development_verdict_lock.json'), timestamp_utc=now()))
        print(verdict, flush=True)
    return tables


def main() -> None:
    split = sys.argv[1]
    assert split in ('development', 'consumed')
    assert json.loads((OUT / 'preflight.json').read_text())['all_passed']
    assert sha(OUT / 'PROTOCOL.md') == json.loads((OUT / 'protocol_hash.json').read_text())['sha256']
    lock = json.loads((OUT / 'feature_lock.json').read_text())
    assert sha(OUT / 'feature_thresholds.csv') == lock['thresholds_sha256']
    torch.set_num_threads(2)
    stimuli = torch.load(OUT / 'development_stimulus.pt', weights_only=True)
    for p, h in json.loads((DERIVED / 'derivation_lock.json').read_text())['sha256'].items():
        assert sha(DERIVED / p) == h
    if split == 'development':
        predictions = torch.load(DERIVED / 'recalibrated_development_logits.pt', weights_only=True)['cells']
    else:
        assert sha(OUT / 'development_verdict_lock.json') == json.loads((OUT / 'development_verdict_sha256.json').read_text())['sha256']
        decision = json.loads((OUT / 'development_verdict_lock.json').read_text())
        for p, h in decision['development_artifact_sha256'].items():
            assert sha(OUT / p) == h
        write_json(OUT / 'consumed_start.json', dict(started_utc=now(), development_verdict_sha256=sha(OUT / 'development_verdict_lock.json'), independent_confirmation=False))
        from preflight import REPOSITORY
        from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
        from data.schottdorf_lee_catalog import mc_pc_recordings
        from data.schottdorf_lee_multirecording import load_schottdorf_cell, load_schottdorf_movie_drive
        config = SchottdorfAdapterConfig(**{**asdict(ADAPTER), 'train_sequence_count': 20, 'validation_sequence_count': 40})
        movie = load_schottdorf_movie_drive(MOVIE, config)
        records = mc_pc_recordings(REPOSITORY / 'data')
        fits = {(r['cell_id'], r['condition']): r for r in read_csv(BIAS / 'bias_fit_per_cell.csv')}
        prior = {(r['cell_id'], r['pathway']): r for r in read_csv(BIAS / 'pathway_bias_recalibration_per_cell.csv')}
        predictions, exact = {}, []
        for cid in stimuli:
            data = load_schottdorf_cell(tuple(r for r in records if r.cell_id == cid), movie, config).validation
            saved = torch.load(OLD / 'cells' / (cid.replace('#', '_')+'-test-predictions.pt'), weights_only=True)
            assert torch.equal(data.spike_events, saved['target']) and torch.equal(data.valid_mask, saved['valid_mask'])
            assert tuple(data.source_image_ids) == tuple(saved['source_image_ids']) and tuple(data.trial_indices) == tuple(saved['trial_indices'])
            stimuli[cid].update(cone_drive=data.cone_drive, valid_mask=data.valid_mask, source_image_ids=data.source_image_ids, trial_indices=data.trial_indices)
            cell = dict(normal=saved['logits']['aligned'], recalibrated={}, target=data.spike_events, valid_mask=data.valid_mask, source_image_ids=data.source_image_ids, trial_indices=data.trial_indices)
            for p, (bias_key, _, _) in CONDITIONS.items():
                z = saved['logits'][bias_key+'-off'].double()+float(fits[cid, bias_key]['fitted_bias'])
                value = nll(z[data.valid_mask], data.spike_events[data.valid_mask].double())
                assert value == float(prior[cid, bias_key]['heldout_off_nll_recal'])
                assert nll(cell['normal'][data.valid_mask].double(), data.spike_events[data.valid_mask].double()) == float(prior[cid, bias_key]['heldout_normal_nll'])
                cell['recalibrated'][p] = z
                exact.append(dict(cell_id=cid, pathway=p, recalibrated_nll_exact=True, normal_nll_exact=True))
            predictions[cid] = cell
        write_json(OUT / 'consumed_identity.json', dict(checks=exact, all_passed=True, model_inference_rerun=False))
    tables = analyze(split, stimuli, predictions, lock)
    if split == 'consumed':
        for name, rows in tables.items():
            write_csv(OUT / (name+'.csv'), read_csv(OUT / ('development_'+name+'.csv'))+rows)
        dev = read_csv(OUT / 'development_per_cell_context_selectivity.csv')
        comparison, signs = [], []
        for k in ('E_AC', 'E_BC', 'D'):
            a, b = [], []
            for r in tables['per_cell_context_selectivity']:
                original = next(d for d in dev if d['cell_id'] == r['cell_id'])
                x, y = float(original[k]) if original[k] != '' else None, r[k]
                same = bool(np.sign(x) == np.sign(y)) if x is not None and y is not None else None
                signs.append(dict(cell_id=r['cell_id'], quantity=k, development=x, consumed_descriptive=y, sign_consistent=same))
                a.append(x); b.append(y)
            da, db = summary(a), summary(b)
            selected = [r for r in signs if r['quantity'] == k]
            comparison.append(dict(quantity=k, development_mean=da['mean'], consumed_descriptive_mean=db['mean'], population_direction_consistent=bool(np.sign(da['mean']) == np.sign(db['mean'])) if da['available'] and db['available'] else None,
                sign_consistent_cells=sum(r['sign_consistent'] is True for r in selected), defined_pairs=sum(r['sign_consistent'] is not None for r in selected), total_cells=22, independent_confirmation=False, verdict_unchanged=True))
        write_csv(OUT / 'development_vs_consumed_descriptive.csv', comparison)
        write_csv(OUT / 'development_vs_consumed_signs.csv', signs)
        with pq.ParquetWriter(OUT / 'stimulus_features.parquet', pq.read_schema(OUT / 'development_stimulus_features.parquet'), compression='zstd') as writer:
            for label in ('development', 'consumed'):
                for batch in pq.ParquetFile(OUT / (label+'_stimulus_features.parquet')).iter_batches():
                    writer.write_batch(batch)
        assert sha(OUT / 'development_verdict_lock.json') == json.loads((OUT / 'development_verdict_sha256.json').read_text())['sha256']
    write_json(OUT / (split+'_status.json'), dict(status='COMPLETE', completed_utc=now(), cells=22))


if __name__ == '__main__':
    main()
