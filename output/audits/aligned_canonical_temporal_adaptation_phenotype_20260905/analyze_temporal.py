from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import json
import sys
import zipfile

import numpy as np
import torch

from temporal_features import OUT, features, save_json, sha, write_csv

ROOT = OUT.parents[2]
OLD = ROOT / 'output/audits/macaque_aligned_heldout_pathway_evaluation_20260905'
COMPARATORS = ('LN', 'CNN')
MODELS = ('aligned', 'LN', 'CNN')


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def average(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def rank(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts)-(counts-1)/2)[inverse]


def summarize(values: list, indices: np.ndarray | None = None) -> dict:
    result = dict(cells=len(values), available=all(v is not None for v in values), mean=None, median=None, positive=None, negative=None, zero=None,
        mean_ci_low=None, mean_ci_high=None, median_ci_low=None, median_ci_high=None)
    if not result['available']:
        return result
    a = np.asarray(values)
    result.update(mean=float(a.mean()), median=float(np.median(a)), positive=int((a > 0).sum()), negative=int((a < 0).sum()), zero=int((a == 0).sum()))
    if indices is not None:
        sample = a[indices]
        for name, distribution in (('mean', sample.mean(axis=1)), ('median', np.median(sample, axis=1))):
            lo, hi = np.quantile(distribution, [.025, .975], method='linear')
            result.update({name+'_ci_low': float(lo), name+'_ci_high': float(hi)})
    return result


def matched(f: dict, d: np.ndarray, two_dimensional: bool) -> tuple[dict, list[dict]]:
    q = f['A_quintile']
    bins = (f['U_quintile']-1)*5+f['C_quintile'] if two_dimensional else f['U_quintile']
    nstrata = 25 if two_dimensional else 5
    masses = np.array([min(int(((q == 1) & (bins == s)).sum()), int(((q == 5) & (bins == s)).sum())) for s in range(1, nstrata+1)])
    mass = int(masses.sum())
    high_n, low_n = int((q == 5).sum()), int((q == 1).sum())
    result = dict(E=0.0 if mass else None, overlap_mass=mass, low_bins=low_n, high_bins=high_n,
        low_overlap_fraction=mass/low_n if low_n else None, high_overlap_fraction=mass/high_n if high_n else None, retained_strata=int((masses > 0).sum()))
    for key in ('U', 'M', 'C'):
        result['residual_'+key+'_difference'] = 0.0 if mass else None
    rows = []
    for s in range(1, nstrata+1):
        low, high = (q == 1) & (bins == s), (q == 5) & (bins == s)
        weight = float(masses[s-1]/mass) if mass else 0.0
        delta = float(d[high].mean()-d[low].mean()) if masses[s-1] else None
        row = dict(stratum=s, U_stratum=(s-1)//5+1 if two_dimensional else s, C_stratum=(s-1)%5+1 if two_dimensional else None,
            low_bins=int(low.sum()), high_bins=int(high.sum()), overlap_mass=int(masses[s-1]), weight=weight, excess_NLL=delta)
        for key in ('U', 'M', 'C'):
            row['Q1_'+key] = average(f[key][low])
            row['Q5_'+key] = average(f[key][high])
            if weight:
                result['residual_'+key+'_difference'] += weight*(row['Q5_'+key]-row['Q1_'+key])
        if weight:
            result['E'] += weight*delta
        rows.append(row)
    return result, rows


def add_array(archive: zipfile.ZipFile, key: str, value: np.ndarray) -> None:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, allow_pickle=False)
    archive.writestr(key+'.npy', stream.getvalue())


def analyze(split: str, stimuli: dict, predictions: dict, groups: dict, lock: dict) -> dict:
    tables = {k: [] for k in ('adaptation_quantile_nll', 'temporal_change_quantile_nll', 'per_cell_adaptation_excess_loss', 'per_cell_temporal_change_excess_loss',
        'current_drive_matched_control', 'current_drive_matched_strata', 'two_dimensional_matched_control', 'two_dimensional_matched_strata', 'continuous_trends', 'precision_comparison', 'phenotype_mask_summary')}
    with zipfile.ZipFile(OUT / f'per_bin_temporal_features_{split}.npz', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for cid, stimulus in stimuli.items():
            path = OUT / ('development_features_'+cid.replace('#', '_')+'.npz')
            assert sha(path) == lock['features_sha256'][path.name]
            with np.load(path) as bundle:
                f = {k: bundle[k] for k in bundle.files} if split == 'development' else features(stimulus, {k: bundle[k+'_edges'] for k in ('A', 'C', 'U')})
            saved = predictions[cid]
            assert torch.equal(stimulus['valid_mask'], saved['valid_mask'])
            assert tuple(stimulus['source_image_ids']) == tuple(saved['source_image_ids']) and tuple(stimulus['trial_indices']) == tuple(saved['trial_indices'])
            mask = f['phenotype_mask']
            target = saved['target'].numpy().astype(np.float64)[mask]
            losses = {}
            for label in MODELS:
                z = saved['logits'][label].numpy().astype(np.float64)[mask]
                assert z.shape == target.shape == f['A'].shape and np.isfinite(z).all()
                losses[label] = np.logaddexp(0, z)-target*z
                from training.mechanistic_retina.losses import expected_bernoulli_nll
                native = float(expected_bernoulli_nll(saved['logits'][label], saved['target'], torch.from_numpy(mask)))
                tables['precision_comparison'].append(dict(split=split, cell_id=cid, model=label, phenotype_float64_nll=float(losses[label].mean()), phenotype_native_float32_nll=native, delta=float(losses[label].mean()-native)))
            tables['phenotype_mask_summary'].append(dict(split=split, cell_id=cid, sequences=mask.shape[0], original_scored_bins=int(f['original_valid_mask'].sum()),
                phenotype_bins=int(mask.sum()), dropped_original_scored_bins=int(f['original_valid_mask'].sum()-mask.sum()), local_first=45, local_last=149))
            for feature, name in (('A', 'adaptation_quantile_nll'), ('C', 'temporal_change_quantile_nll')):
                q = f[feature+'_quintile']
                for k in range(1, 6):
                    selection = q == k
                    row = dict(split=split, cell_id=cid, group=groups[cid], feature=feature, quintile=k, bins=int(selection.sum()), event_rate=average(target[selection]),
                        mean_signed_drive=average(f['M'][selection]), mean_absolute_drive=average(f['U'][selection]), mean_recent_change=average(f['C'][selection]), mean_feature=average(f[feature][selection]))
                    row.update({label+'_nll': average(losses[label][selection]) for label in MODELS})
                    row.update({'aligned_minus_'+c: average((losses['aligned']-losses[c])[selection]) for c in COMPARATORS})
                    tables[name].append(row)
                for c in COMPARATORS:
                    d = losses['aligned']-losses[c]
                    low, high = q == 1, q == 5
                    raw = float(d[high].mean()-d[low].mean()) if low.any() and high.any() else None
                    row = dict(split=split, cell_id=cid, group=groups[cid], comparator=c, raw_E=raw, Q1_bins=int(low.sum()), Q5_bins=int(high.sum()),
                        Q1_event_rate=average(target[low]), Q5_event_rate=average(target[high]))
                    for key in ('M', 'U', 'C'):
                        row['raw_'+key+'_difference'] = float(f[key][high].mean()-f[key][low].mean()) if low.any() and high.any() else None
                    if feature == 'A':
                        control, detail = matched(f, d, False)
                        if split == 'development':
                            assert control['overlap_mass'] > 0
                        row['matched_E'] = control['E']
                        identity = dict(split=split, cell_id=cid, group=groups[cid], comparator=c)
                        tables['current_drive_matched_control'].append({**identity, **control})
                        tables['current_drive_matched_strata'].extend({**identity, **r} for r in detail)
                        if lock['run_two_dimensional_descriptive']:
                            control2, detail2 = matched(f, d, True)
                            row['two_dimensional_E'] = control2['E']
                            tables['two_dimensional_matched_control'].append({**identity, **control2})
                            tables['two_dimensional_matched_strata'].extend({**identity, **r} for r in detail2)
                        tables['per_cell_adaptation_excess_loss'].append(row)
                    else:
                        tables['per_cell_temporal_change_excess_loss'].append(row)
                    x, y = rank(f[feature]), rank(d)
                    rho = float(np.corrcoef(x, y)[0, 1]) if np.ptp(x) and np.ptp(y) else None
                    tables['continuous_trends'].append(dict(split=split, cell_id=cid, group=groups[cid], feature=feature, comparator=c, spearman=rho))
            prefix = cid.replace('#', '_')+'__'
            for key, value in f.items():
                add_array(archive, prefix+key, value)
            add_array(archive, prefix+'target', target)
            add_array(archive, prefix+'cell_id', np.array(cid))
            add_array(archive, prefix+'group', np.array(groups[cid]))
            add_array(archive, prefix+'time_seconds', f['live_frame']/150)
            add_array(archive, prefix+'decoded_frame', f['live_frame']+751)
            for label, loss in losses.items():
                add_array(archive, prefix+label+'_loss', loss)
            print(split, cid, len(target), flush=True)
    expected = 57540 if split == 'development' else 575400
    assert sum(r['phenotype_bins'] for r in tables['phenotype_mask_summary']) == expected
    indices = None
    if split == 'development':
        indices = np.random.default_rng(2026090501).integers(0, 22, size=(100000, 22))
        digest = hashlib.sha256(indices.tobytes()).hexdigest()
        assert digest == 'e7c2af0a9e6c1c54358f3f64513ad92790ea4bcc7647e92f2c9ab3a9da901528'
        save_json('bootstrap_identity.json', dict(existing_seed=2026090501, reused=True, draws=100000, columns=22, index_sha256=digest))
    population, c_population, groups_table, outliers, trends_population, two_population = [], [], [], [], [], []
    for c in COMPARATORS:
        cells = [r for r in tables['per_cell_adaptation_excess_loss'] if r['comparator'] == c]
        assert len(cells) == 22
        for estimate, field in (('raw', 'raw_E'), ('U_matched', 'matched_E')):
            population.append(dict(split=split, comparator=c, estimate=estimate, **summarize([r[field] for r in cells], indices)))
            for group in ('MC_ON', 'MC_OFF', 'PC_ON', 'PC_OFF'):
                values = [r[field] for r in cells if r['group'] == group]
                s = summarize(values)
                groups_table.append(dict(split=split, comparator=c, estimate=estimate, group=group, cells=s['cells'], mean=s['mean'], median=s['median'], available=s['available']))
        c_values = [r['raw_E'] for r in tables['per_cell_temporal_change_excess_loss'] if r['comparator'] == c]
        c_population.append(dict(split=split, comparator=c, **summarize(c_values)))
        if lock['run_two_dimensional_descriptive']:
            two_population.append(dict(split=split, comparator=c, **summarize([r['two_dimensional_E'] for r in cells])))
        for feature in ('A', 'C'):
            s = summarize([r['spearman'] for r in tables['continuous_trends'] if r['comparator'] == c and r['feature'] == feature])
            trends_population.append(dict(split=split, comparator=c, feature=feature, **{k: v for k, v in s.items() if '_ci_' not in k}))
        if split == 'development':
            removed = sorted(cells, key=lambda r: (-abs(r['raw_E']), r['cell_id']))[0]
            outliers.append(dict(comparator=c, removed_cell=removed['cell_id'], removed_E=removed['raw_E'], remaining_cells=21,
                mean_after_single_deletion=float(np.mean([r['raw_E'] for r in cells if r is not removed]))))
    tables.update(population_adaptation_excess_loss=population, population_temporal_change_excess_loss=c_population, cell_group_summary=groups_table,
        continuous_trend_population=trends_population, two_dimensional_matched_population=two_population,
        current_drive_matched_population=[r for r in population if r['estimate'] == 'U_matched'])
    for name, rows in tables.items():
        if rows:
            filename = split+'_'+name+'.csv'
            if split == 'development' and name == 'phenotype_mask_summary':
                assert read_csv(OUT / filename) == [{k: str(v) for k, v in r.items()} for r in rows]
            else:
                write_csv(filename, rows)
    if split == 'development':
        write_csv('outlier_sensitivity.csv', outliers)
        criteria = []
        for c in COMPARATORS:
            raw = next(r for r in population if r['comparator'] == c and r['estimate'] == 'raw')
            control = next(r for r in population if r['comparator'] == c and r['estimate'] == 'U_matched')
            deletion = next(r for r in outliers if r['comparator'] == c)
            checks = dict(raw_mean_ci_positive=raw['mean_ci_low'] > 0, after_deletion_positive=deletion['mean_after_single_deletion'] > 0,
                matched_mean_ci_positive=control['mean_ci_low'] > 0, substantial_retention=control['mean'] >= .5*raw['mean'])
            criteria.append(dict(comparator=c, checks=checks, go=all(checks.values()), mixed_candidate=raw['mean'] > 0 and (control['mean'] > 0 or raw['mean_ci_low'] > 0)))
        label = 'GO — TEMPORAL/ADAPTATION PHENOTYPE SUPPORTED' if any(r['go'] for r in criteria) else ('MIXED — WEAK TEMPORAL PHENOTYPE' if any(r['mixed_candidate'] for r in criteria) else 'NO-GO — NO TEMPORAL/ADAPTATION-SPECIFIC DEFICIT')
        verdict = dict(verdict=label, criteria=criteria, population=population, decision_data='development [16,20) only', consumed_analysis_started=False, utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(OUT / 'PROTOCOL.md'))
        save_json('development_verdict.json', verdict)
        save_json('development_verdict_lock.json', dict(verdict_sha256=sha(OUT / 'development_verdict.json'), locked_utc=datetime.now(timezone.utc).isoformat(),
            development_artifact_sha256={p.name: sha(p) for p in OUT.glob('development_*') if p.is_file() and p.name != 'development_verdict_lock.json'},
            per_bin_sha256=sha(OUT / 'per_bin_temporal_features_development.npz'), feature_lock_sha256=sha(OUT / 'feature_definition_lock.json'), outlier_sha256=sha(OUT / 'outlier_sensitivity.csv')))
        print(label, flush=True)
    return tables


def main() -> None:
    split = sys.argv[1]
    assert split in ('development', 'consumed')
    assert not (OUT / f'per_bin_temporal_features_{split}.npz').exists()
    assert json.loads((OUT / 'preflight.json').read_text())['all_passed']
    assert sha(OUT / 'PROTOCOL.md') == json.loads((OUT / 'protocol_hash.json').read_text())['sha256']
    lock = json.loads((OUT / 'feature_definition_lock.json').read_text())
    assert sha(OUT / 'feature_thresholds.csv') == lock['thresholds_sha256']
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT))
    groups = {r['cell_id']: r['group'] for r in read_csv(ROOT / 'output/audits/macaque_fixed_alignment_experiment_20260905/per_cell_results.csv')}
    stimuli = torch.load(OUT / 'development_stimulus.pt', weights_only=True)
    if split == 'development':
        predictions = torch.load(OUT / 'development_outputs.pt', weights_only=True)
    else:
        decision_lock = json.loads((OUT / 'development_verdict_lock.json').read_text())
        assert sha(OUT / 'development_verdict.json') == decision_lock['verdict_sha256']
        for p, digest in decision_lock['development_artifact_sha256'].items():
            assert sha(OUT / p) == digest
        assert sha(OUT / 'per_bin_temporal_features_development.npz') == decision_lock['per_bin_sha256']
        assert sha(OUT / 'feature_definition_lock.json') == decision_lock['feature_lock_sha256']
        save_json('consumed_descriptive_start.json', dict(start_utc=datetime.now(timezone.utc).isoformat(), development_verdict_sha256=decision_lock['verdict_sha256'], independent_confirmation=False))
        from replay_preflight import ADAPTER, MOVIE, PRIMARY
        sys.path.insert(0, str(PRIMARY))
        from preflight import REPOSITORY
        from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
        from data.schottdorf_lee_catalog import mc_pc_recordings
        from data.schottdorf_lee_multirecording import load_schottdorf_cell, load_schottdorf_movie_drive
        config = SchottdorfAdapterConfig(**{**asdict(ADAPTER), 'train_sequence_count': 20, 'validation_sequence_count': 40})
        movie = load_schottdorf_movie_drive(MOVIE, config)
        records = mc_pc_recordings(REPOSITORY / 'data')
        predictions = {}
        for cid in stimuli:
            data = load_schottdorf_cell(tuple(r for r in records if r.cell_id == cid), movie, config).validation
            old_path = OLD / 'cells' / (cid.replace('#', '_')+'-test-predictions.pt')
            saved = torch.load(old_path, weights_only=True)
            assert torch.equal(data.spike_events, saved['target']) and torch.equal(data.spike_counts, saved['spike_counts'])
            assert torch.equal(data.valid_mask, saved['valid_mask']) and tuple(data.source_image_ids) == tuple(saved['source_image_ids']) and tuple(data.trial_indices) == tuple(saved['trial_indices'])
            stimuli[cid].update(cone_drive=data.cone_drive, valid_mask=data.valid_mask, source_image_ids=data.source_image_ids, trial_indices=data.trial_indices)
            predictions[cid] = saved
    tables = analyze(split, stimuli, predictions, groups, lock)
    if split == 'consumed':
        for name, rows in tables.items():
            if rows:
                write_csv(name+'.csv', read_csv(OUT / ('development_'+name+'.csv'))+rows)
        comparisons, per_cell_comparisons = [], []
        for feature, name, estimates in (('A', 'per_cell_adaptation_excess_loss', ('raw_E', 'matched_E')), ('C', 'per_cell_temporal_change_excess_loss', ('raw_E',))):
            dev = read_csv(OUT / ('development_'+name+'.csv'))
            for c in COMPARATORS:
                for estimate in estimates:
                    dev_lookup = {r['cell_id']: float(r[estimate]) for r in dev if r['comparator'] == c}
                    consumed_rows = [r for r in tables[name] if r['comparator'] == c]
                    consistent, valid = 0, 0
                    for r in consumed_rows:
                        a, b = dev_lookup[r['cell_id']], r[estimate]
                        same = bool(np.sign(a) == np.sign(b)) if b is not None else None
                        consistent += int(same) if same is not None else 0
                        valid += int(same is not None)
                        per_cell_comparisons.append(dict(cell_id=r['cell_id'], feature=feature, comparator=c, estimate=estimate, development_E=a, consumed_descriptive_E=b, sign_consistent=same))
                    a_mean = float(np.mean(list(dev_lookup.values())))
                    b_mean = average(np.array([r[estimate] for r in consumed_rows])) if valid == 22 else None
                    comparisons.append(dict(feature=feature, comparator=c, estimate=estimate, development_mean=a_mean, consumed_descriptive_mean=b_mean,
                        mean_direction_consistent=bool(np.sign(a_mean) == np.sign(b_mean)) if b_mean is not None else None, sign_consistent_cells=consistent, available_cells=valid, total_cells=22, independent_confirmation=False, verdict_unchanged=True))
        write_csv('development_vs_consumed_descriptive.csv', comparisons)
        write_csv('development_vs_consumed_per_cell.csv', per_cell_comparisons)
        assert sha(OUT / 'development_verdict.json') == json.loads((OUT / 'development_verdict_lock.json').read_text())['verdict_sha256']
    save_json(split+'_analysis_status.json', dict(status='COMPLETE', completed_utc=datetime.now(timezone.utc).isoformat(), cells=22, protocol_unchanged=True))


if __name__ == '__main__':
    main()
