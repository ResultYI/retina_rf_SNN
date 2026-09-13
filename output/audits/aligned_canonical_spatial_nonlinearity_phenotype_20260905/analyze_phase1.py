from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

from stimulus_features import OUT, features, sha, write_csv

ROOT = OUT.parents[2]
OLD = ROOT / 'output/audits/macaque_aligned_heldout_pathway_evaluation_20260905'
MODELS = ('aligned', 'LN', 'CNN')
COMPARATORS = ('LN', 'CNN')


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts-1)/2)[inverse]


def summary(values: np.ndarray, indices: np.ndarray | None = None) -> dict:
    result = dict(cells=len(values), mean=float(np.mean(values)), median=float(np.median(values)), positive=int((values > 0).sum()), negative=int((values < 0).sum()), zero=int((values == 0).sum()))
    if indices is not None:
        sampled = values[indices]
        for name, statistic in (('mean', np.mean(sampled, axis=1)), ('median', np.median(sampled, axis=1))):
            lo, hi = np.quantile(statistic, [.025, .975], method='linear')
            result[name+'_ci_low'], result[name+'_ci_high'] = float(lo), float(hi)
    return result


def add_array(archive: zipfile.ZipFile, name: str, value: np.ndarray) -> None:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, allow_pickle=False)
    archive.writestr(name+'.npy', stream.getvalue())


def analyze(split: str, inputs: dict, predictions: dict, groups: dict, frozen: dict) -> dict:
    quantiles, per_cell, strata, trends = [], [], [], []
    numerical = []
    with zipfile.ZipFile(OUT / f'per_bin_features_{split}.npz', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for cid, stimulus in inputs.items():
            if split == 'development':
                path = OUT / ('development_features_'+cid.replace('#', '_')+'.npz')
                assert sha(path) == frozen['features_sha256'][path.name]
                with np.load(path) as bundle:
                    f = {k: bundle[k] for k in bundle.files}
            else:
                with np.load(OUT / ('development_features_'+cid.replace('#', '_')+'.npz')) as bundle:
                    edges = dict(lsc=bundle['lsc_edges'], mean=bundle['mean_edges'])
                f = features(stimulus, edges)
            saved = predictions[cid]
            mask = stimulus['valid_mask'].numpy().astype(bool)
            assert torch.equal(stimulus['valid_mask'], saved['valid_mask'])
            assert tuple(stimulus['source_image_ids']) == tuple(saved['source_image_ids'])
            assert tuple(stimulus['trial_indices']) == tuple(saved['trial_indices'])
            target = saved['target'].numpy().astype(np.float64)[mask]
            losses = {}
            for label in MODELS:
                z = saved['logits'][label].numpy().astype(np.float64)[mask]
                assert z.shape == target.shape == f['LSC'].shape and np.isfinite(z).all()
                losses[label] = np.logaddexp(0, z) - target*z
                from training.mechanistic_retina.losses import expected_bernoulli_nll
                native = float(expected_bernoulli_nll(saved['logits'][label], saved['target'], saved['valid_mask']))
                numerical.append(dict(split=split, cell_id=cid, model=label, float64_nll=float(losses[label].mean()), native_float32_nll=native, delta=float(losses[label].mean()-native)))
            q, mb = f['LSC_quantile'], f['mean_stratum']
            for k in range(1, 6):
                selected = q == k
                assert selected.any(), f'Empty declared quantile: {split} {cid} Q{k}'
                row = dict(split=split, cell_id=cid, group=groups[cid], quantile=k, bins=int(selected.sum()), event_rate=float(target[selected].mean()), local_mean=float(f['local_mean'][selected].mean()), LSC=float(f['LSC'][selected].mean()))
                row.update({label+'_nll': float(losses[label][selected].mean()) for label in MODELS})
                row.update({f'aligned_minus_{c}': float((losses['aligned']-losses[c])[selected].mean()) for c in COMPARATORS})
                quantiles.append(row)
            mass = np.array([min(int(((q == 1) & (mb == s)).sum()), int(((q == 5) & (mb == s)).sum())) for s in range(1, 6)])
            if split == 'development':
                assert mass.sum() > 0
            for c in COMPARATORS:
                d = losses['aligned']-losses[c]
                raw = float(d[q == 5].mean()-d[q == 1].mean())
                matched = 0.0 if mass.sum() else None
                matched_m_delta = 0.0 if mass.sum() else None
                for s in range(1, 6):
                    low, high = (q == 1) & (mb == s), (q == 5) & (mb == s)
                    eligible = bool(mass[s-1] > 0)
                    weight = float(mass[s-1]/mass.sum()) if mass.sum() else 0.0
                    delta = float(d[high].mean()-d[low].mean()) if eligible else None
                    mean_low = float(f['local_mean'][low].mean()) if low.any() else None
                    mean_high = float(f['local_mean'][high].mean()) if high.any() else None
                    if eligible:
                        matched += weight*delta
                        matched_m_delta += weight*(mean_high-mean_low)
                    strata.append(dict(split=split, cell_id=cid, comparator=c, mean_stratum=s, low_bins=int(low.sum()), high_bins=int(high.sum()), overlap_count=int(mass[s-1]), weight=weight, eligible=eligible, excess_nll=delta, low_local_mean=mean_low, high_local_mean=mean_high))
                per_cell.append(dict(split=split, cell_id=cid, group=groups[cid], comparator=c, raw_E=raw, matched_E=matched,
                    low_bins=int((q == 1).sum()), high_bins=int((q == 5).sum()), overlap_mass=int(mass.sum()),
                    low_overlap_fraction=float(mass.sum()/(q == 1).sum()), high_overlap_fraction=float(mass.sum()/(q == 5).sum()), eligible_mean_strata=int((mass > 0).sum()),
                    raw_mean_drive_difference=float(f['local_mean'][q == 5].mean()-f['local_mean'][q == 1].mean()), matched_mean_drive_difference=matched_m_delta,
                    Q1_event_rate=float(target[q == 1].mean()), Q5_event_rate=float(target[q == 5].mean())))
                rho = float(np.corrcoef(ranks(f['LSC']), ranks(d))[0, 1])
                trends.append(dict(split=split, cell_id=cid, group=groups[cid], comparator=c, spearman=rho))
            prefix = cid.replace('#', '_')+'__'
            for key, value in f.items():
                add_array(archive, prefix+key, value)
            add_array(archive, prefix+'target', target)
            add_array(archive, prefix+'time_seconds', f['live_frame']/150)
            add_array(archive, prefix+'decoded_frame', f['live_frame']+751)
            for label, values in losses.items():
                add_array(archive, prefix+label+'_loss', values)
            print(split, cid, 'scored bins', len(target), flush=True)
    indices = None
    if split == 'development':
        indices = np.random.default_rng(2026090501).integers(0, 22, size=(100000, 22))
        digest = hashlib.sha256(indices.tobytes()).hexdigest()
        assert digest == 'e7c2af0a9e6c1c54358f3f64513ad92790ea4bcc7647e92f2c9ab3a9da901528'
        save_json('bootstrap_identity.json', dict(seed=2026090501, existing_seed_reused=True, draws=100000, cells=22, index_sha256=digest))
    population, group_rows, trend_population, outliers = [], [], [], []
    for c in COMPARATORS:
        selected = [r for r in per_cell if r['comparator'] == c]
        assert len(selected) == 22
        for kind in ('raw', 'matched'):
            available = all(r[kind+'_E'] is not None for r in selected)
            if available:
                values = np.asarray([r[kind+'_E'] for r in selected])
                population.append(dict(split=split, comparator=c, estimate=kind, available=True, **summary(values, indices)))
            else:
                population.append(dict(split=split, comparator=c, estimate=kind, available=False, cells=22, mean=None, median=None, positive=None, negative=None, zero=None))
            for group in ('MC_ON', 'MC_OFF', 'PC_ON', 'PC_OFF'):
                members = [r for r in selected if r['group'] == group]
                vals = [r[kind+'_E'] for r in members]
                valid = all(v is not None for v in vals)
                group_rows.append(dict(split=split, comparator=c, estimate=kind, group=group, cells=len(vals), mean=float(np.mean(vals)) if valid else None, median=float(np.median(vals)) if valid else None))
        correlations = np.asarray([r['spearman'] for r in trends if r['comparator'] == c])
        trend_population.append(dict(split=split, comparator=c, **summary(correlations)))
        if split == 'development':
            removed = sorted(selected, key=lambda r: (-abs(r['raw_E']), r['cell_id']))[0]
            outliers.append(dict(comparator=c, removed_cell=removed['cell_id'], removed_E=removed['raw_E'], remaining_cells=21, mean_E_after_single_deletion=float(np.mean([r['raw_E'] for r in selected if r is not removed]))))
    tables = dict(quantile_nll=quantiles, per_cell_excess_loss=per_cell, population_excess_loss=population, mean_matched_control=per_cell,
        mean_matched_strata=strata, cell_group_summary=group_rows, continuous_trend=trends, continuous_trend_population=trend_population, precision_comparison=numerical)
    for name, rows in tables.items():
        write_csv(split+'_'+name+'.csv', rows)
    if split == 'development':
        write_csv('outlier_sensitivity.csv', outliers)
        criteria = []
        for c in COMPARATORS:
            raw = next(r for r in population if r['comparator'] == c and r['estimate'] == 'raw')
            matched = next(r for r in population if r['comparator'] == c and r['estimate'] == 'matched')
            loo = next(r for r in outliers if r['comparator'] == c)
            checks = dict(raw_mean_ci_positive=raw['mean_ci_low'] > 0, single_deletion_positive=loo['mean_E_after_single_deletion'] > 0,
                matched_mean_ci_positive=matched['mean_ci_low'] > 0, substantial_retention=matched['mean'] >= .5*raw['mean'])
            criteria.append(dict(comparator=c, checks=checks, go=all(checks.values()), mixed_candidate=raw['mean'] > 0 and matched['mean'] > 0))
        decision = 'GO — SPATIAL-NONLINEARITY PHENOTYPE SUPPORTED' if any(r['go'] for r in criteria) else ('MIXED — WEAK PHENOTYPE' if any(r['mixed_candidate'] for r in criteria) else 'NO-GO — NO EVIDENCE FOR THIS MODEL DEFICIT')
        verdict = dict(verdict=decision, criteria=criteria, decision_data='development live [16,20) only', consumed_analysis_started=False, utc=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=sha(OUT / 'PROTOCOL.md'), development_population=population,
            development_artifact_sha256={p.name: sha(p) for p in OUT.glob('development_*') if p.is_file()}, per_bin_sha256=sha(OUT / 'per_bin_features_development.npz'), outlier_sha256=sha(OUT / 'outlier_sensitivity.csv'))
        save_json('development_verdict.json', verdict)
        save_json('development_verdict_hash.json', dict(sha256=sha(OUT / 'development_verdict.json'), locked_utc=datetime.now(timezone.utc).isoformat()))
        print(decision, flush=True)
    return tables


def main() -> None:
    split = sys.argv[1]
    assert split in ('development', 'consumed')
    assert not (OUT / f'per_bin_features_{split}.npz').exists()
    assert json.loads((OUT / 'preflight.json').read_text())['all_passed']
    assert sha(OUT / 'PROTOCOL.md') == json.loads((OUT / 'protocol_hash.json').read_text())['sha256']
    frozen = json.loads((OUT / 'feature_definition_lock.json').read_text())
    assert sha(OUT / 'feature_thresholds.csv') == frozen['thresholds_sha256']
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT))
    groups = {r['cell_id']: r['group'] for r in read_csv(ROOT / 'output/audits/macaque_fixed_alignment_experiment_20260905/per_cell_results.csv')}
    stimulus = torch.load(OUT / 'development_stimulus.pt', weights_only=True)
    if split == 'development':
        outputs = torch.load(OUT / 'development_outputs.pt', weights_only=True)
    else:
        assert sha(OUT / 'development_verdict.json') == json.loads((OUT / 'development_verdict_hash.json').read_text())['sha256']
        verdict = json.loads((OUT / 'development_verdict.json').read_text())
        for p, digest in verdict['development_artifact_sha256'].items():
            assert sha(OUT / p) == digest
        save_json('consumed_descriptive_start.json', dict(start_utc=datetime.now(timezone.utc).isoformat(), locked_development_verdict_sha256=sha(OUT / 'development_verdict.json'), independent_confirmation=False))
        from phase1_preflight import ADAPTER, MOVIE, PRIMARY
        sys.path.insert(0, str(PRIMARY))
        from preflight import REPOSITORY
        from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
        from data.schottdorf_lee_catalog import mc_pc_recordings
        from data.schottdorf_lee_multirecording import load_schottdorf_cell, load_schottdorf_movie_drive
        config = SchottdorfAdapterConfig(**{**asdict(ADAPTER), 'train_sequence_count': 20, 'validation_sequence_count': 40})
        movie = load_schottdorf_movie_drive(MOVIE, config)
        records = mc_pc_recordings(REPOSITORY / 'data')
        outputs = {}
        for cid in stimulus:
            data = load_schottdorf_cell(tuple(r for r in records if r.cell_id == cid), movie, config).validation
            saved = torch.load(OLD / 'cells' / (cid.replace('#', '_')+'-test-predictions.pt'), weights_only=True)
            assert torch.equal(data.spike_events, saved['target']) and torch.equal(data.spike_counts, saved['spike_counts'])
            assert torch.equal(data.valid_mask, saved['valid_mask']) and tuple(data.source_image_ids) == tuple(saved['source_image_ids']) and tuple(data.trial_indices) == tuple(saved['trial_indices'])
            stimulus[cid].update(cone_drive=data.cone_drive, valid_mask=data.valid_mask, source_image_ids=data.source_image_ids, trial_indices=data.trial_indices)
            outputs[cid] = saved
    tables = analyze(split, stimulus, outputs, groups, frozen)
    if split == 'consumed':
        for name, rows in tables.items():
            old = read_csv(OUT / ('development_'+name+'.csv'))
            keys = list(old[0])
            combined = old + [{k: r.get(k, '') for k in keys} for r in rows]
            write_csv(name+'.csv', combined)
        comparisons = []
        dev = read_csv(OUT / 'development_population_excess_loss.csv')
        for r in tables['population_excess_loss']:
            d = next(x for x in dev if x['comparator'] == r['comparator'] and x['estimate'] == r['estimate'])
            comparisons.append(dict(comparator=r['comparator'], estimate=r['estimate'], development_mean=float(d['mean']), development_median=float(d['median']),
                consumed_descriptive_mean=r['mean'], consumed_descriptive_median=r['median'], mean_direction_consistent=bool(np.sign(float(d['mean'])) == np.sign(r['mean'])) if r['mean'] is not None else None,
                independent_confirmation=False, verdict_unchanged=True))
        write_csv('development_vs_consumed_descriptive.csv', comparisons)
        assert sha(OUT / 'development_verdict.json') == json.loads((OUT / 'development_verdict_hash.json').read_text())['sha256']
    save_json(split+'_analysis_status.json', dict(status='COMPLETE', completed_utc=datetime.now(timezone.utc).isoformat(), cells=22, protocol_unchanged=True))


if __name__ == '__main__':
    main()
