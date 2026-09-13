from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

OUT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def features(stimulus: dict, thresholds: dict | None = None) -> dict[str, np.ndarray]:
    original = stimulus['valid_mask'].numpy().astype(bool)
    assert original.ndim == 3 and original.shape[1:] == (150, 1)
    assert not original[:, :30].any() and original[:, 30:].all()
    mask = original.copy()
    mask[:, :45] = False
    support = stimulus['support'].numpy().astype(np.float64)
    assert np.isin(support, [0, 1]).all() and support.sum() > 0
    weights = support/support.sum()
    drive = stimulus['cone_drive'].numpy().astype(np.float64)
    assert drive.shape[:2] == original.shape[:2]
    mean = np.sum(drive*weights, axis=2)
    adaptation = np.full(mean.shape, np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(mean, 45, axis=1)
    adaptation[:, 45:] = np.sqrt(np.mean(windows[:, :-1]**2, axis=2))
    change = np.full(mean.shape, np.nan)
    change[:, 5:] = np.abs(mean[:, 5:]-mean[:, :-5])
    selected = mask[..., 0]
    arrays = dict(M=mean[selected], U=np.abs(mean[selected]), A=adaptation[selected], C=change[selected])
    assert all(np.isfinite(v).all() for v in arrays.values())
    for key in ('A', 'C', 'U'):
        edges = np.quantile(arrays[key], [.2, .4, .6, .8], method='linear') if thresholds is None else thresholds[key]
        assert np.all(np.diff(edges) > 0), f'Invalid {key} edges'
        arrays[key+'_edges'] = np.asarray(edges)
        arrays[key+'_quintile'] = 1+np.searchsorted(edges, arrays[key], side='right')
    recording, source, local, global_trial, frame, sequence, timebin = [], [], [], [], [], [], []
    for index, (identity, trial) in enumerate(zip(stimulus['source_image_ids'], stimulus['trial_indices'], strict=True)):
        matched = re.fullmatch(r'(lSS\d+)-live-frames-(\d+)-(\d+)-trial-(\d+)', identity)
        assert matched is not None
        rid, start, stop, local_trial = matched.groups()
        assert int(stop) == int(start)+149
        times = np.flatnonzero(selected[index])
        n = len(times)
        recording.extend([rid]*n)
        source.extend([identity]*n)
        local.extend([int(local_trial)]*n)
        global_trial.extend([int(trial)]*n)
        frame.extend((int(start)+times).tolist())
        sequence.extend([index]*n)
        timebin.extend(times.tolist())
    arrays.update(recording=np.asarray(recording), source_image_id=np.asarray(source), local_trial_one_based=np.asarray(local),
        global_trial_zero_based=np.asarray(global_trial), live_frame=np.asarray(frame), sequence_index=np.asarray(sequence), time_bin=np.asarray(timebin),
        phenotype_mask=mask, original_valid_mask=original, weights=weights, center=stimulus['center'].numpy())
    return arrays


def main() -> None:
    assert not (OUT / 'feature_definition_lock.json').exists()
    assert json.loads((OUT / 'preflight.json').read_text())['all_passed']
    assert sha(OUT / 'PROTOCOL.md') == json.loads((OUT / 'protocol_hash.json').read_text())['sha256']
    stimuli = torch.load(OUT / 'development_stimulus.pt', weights_only=True)
    hashes, thresholds, masks, triggers = {}, [], [], []
    for cid, stimulus in stimuli.items():
        f = features(stimulus)
        for key in ('A', 'C', 'U'):
            assert set(f[key+'_quintile']) == set(range(1, 6)), (cid, key)
            for n, edge in enumerate(f[key+'_edges'], 1):
                thresholds.append(dict(cell_id=cid, feature=key, probability=n/5, threshold=float(edge)))
        q, u = f['A_quintile'], f['U_quintile']
        mass = sum(min(int(((q == 1) & (u == s)).sum()), int(((q == 5) & (u == s)).sum())) for s in range(1, 6))
        assert mass > 0, f'STOP current-drive matching definition UNVERIFIED: {cid}'
        delta = float(f['C'][q == 5].mean()-f['C'][q == 1].mean())
        sd = float(f['C'].std(ddof=0))
        assert sd > 0 or delta == 0
        standardized = abs(delta)/sd if sd > 0 else 0.0
        triggers.append(dict(cell_id=cid, C_Q1=float(f['C'][q == 1].mean()), C_Q5=float(f['C'][q == 5].mean()), C_difference=delta,
            C_population_sd=sd, absolute_standardized_difference=standardized, reaches_prespecified_half_sd=standardized >= .5))
        path = OUT / ('development_features_'+cid.replace('#', '_')+'.npz')
        np.savez_compressed(path, **f)
        hashes[path.name] = sha(path)
        masks.append(dict(split='development', cell_id=cid, sequences=stimulus['valid_mask'].shape[0], original_scored_bins=int(f['original_valid_mask'].sum()),
            phenotype_bins=int(f['phenotype_mask'].sum()), dropped_original_scored_bins=int(f['original_valid_mask'].sum()-f['phenotype_mask'].sum()), local_first=45, local_last=149))
    assert sum(r['phenotype_bins'] for r in masks) == 57540
    write_csv('feature_thresholds.csv', thresholds)
    write_csv('development_phenotype_mask_summary.csv', masks)
    write_csv('development_C_difference_trigger.csv', triggers)
    save_json('feature_definition_lock.json', dict(status='LOCKED_BEFORE_PER_BIN_LOSS', utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(OUT / 'PROTOCOL.md'), features_sha256=hashes, thresholds_sha256=sha(OUT / 'feature_thresholds.csv'),
        mask_summary_sha256=sha(OUT / 'development_phenotype_mask_summary.csv'), trigger_sha256=sha(OUT / 'development_C_difference_trigger.csv'),
        run_two_dimensional_descriptive=any(r['reaches_prespecified_half_sd'] for r in triggers), outputs_used_for_features=False,
        stimulus_sha256=sha(OUT / 'development_stimulus.pt')))
    print('LOCKED', len(stimuli), 'cells, 57540 phenotype bins; 2D trigger', any(r['reaches_prespecified_half_sd'] for r in triggers))


if __name__ == '__main__':
    main()
