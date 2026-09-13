from __future__ import annotations

import csv
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


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def features(stimulus: dict, edges: dict | None = None) -> dict[str, np.ndarray]:
    mask = stimulus['valid_mask'].numpy().astype(bool)
    assert mask.ndim == 3 and mask.shape[-1] == 1
    mask = mask[..., 0]
    drive = stimulus['cone_drive'].numpy().astype(np.float64)[mask]
    support = stimulus['support'].numpy().astype(np.float64)
    assert np.isin(support, [0, 1]).all() and support.sum() > 0
    w = support / support.sum()
    mean = np.sum(drive * w, axis=1)
    contrast = np.sqrt(np.sum(w * (drive - mean[:, None]) ** 2, axis=1))
    rms = np.sqrt(np.sum(w * drive ** 2, axis=1))
    if edges is None:
        edges = dict(lsc=np.quantile(contrast, [.2, .4, .6, .8], method='linear'), mean=np.quantile(mean, [.2, .4, .6, .8], method='linear'))
    assert np.all(np.diff(edges['lsc']) > 0) and np.all(np.diff(edges['mean']) > 0)
    quantile = 1 + np.searchsorted(edges['lsc'], contrast, side='right')
    mean_bin = 1 + np.searchsorted(edges['mean'], mean, side='right')
    frame, recording, local_trial, global_trial, sequence, timebin = [], [], [], [], [], []
    for index, (identity, trial) in enumerate(zip(stimulus['source_image_ids'], stimulus['trial_indices'], strict=True)):
        match = re.fullmatch(r'(lSS\d+)-live-frames-(\d+)-(\d+)-trial-(\d+)', identity)
        assert match is not None
        rid, start, stop, local = match.groups()
        assert int(stop) == int(start) + 149
        selected = np.flatnonzero(mask[index])
        n = len(selected)
        frame.extend((int(start) + selected).tolist())
        recording.extend([rid] * n)
        local_trial.extend([int(local)] * n)
        global_trial.extend([int(trial)] * n)
        sequence.extend([index] * n)
        timebin.extend(selected.tolist())
    return dict(LSC=contrast, local_mean=mean, local_RMS=rms, LSC_quantile=quantile, mean_stratum=mean_bin,
        live_frame=np.asarray(frame), recording=np.asarray(recording), local_trial_one_based=np.asarray(local_trial),
        global_trial_zero_based=np.asarray(global_trial), sequence_index=np.asarray(sequence), time_bin=np.asarray(timebin),
        lsc_edges=np.asarray(edges['lsc']), mean_edges=np.asarray(edges['mean']), weights=w)


def main() -> None:
    assert not (OUT / 'feature_definition_lock.json').exists()
    assert json.loads((OUT / 'preflight.json').read_text())['all_passed']
    assert sha(OUT / 'PROTOCOL.md') == json.loads((OUT / 'protocol_hash.json').read_text())['sha256']
    source = torch.load(OUT / 'development_stimulus.pt', weights_only=True)
    rows, hashes, support_rows = [], {}, []
    for cid, stimulus in source.items():
        f = features(stimulus)
        assert set(f['LSC_quantile']) == set(range(1, 6)), cid
        overlap = [min(int(((f['LSC_quantile'] == 1) & (f['mean_stratum'] == s)).sum()), int(((f['LSC_quantile'] == 5) & (f['mean_stratum'] == s)).sum())) for s in range(1, 6)]
        assert sum(overlap) > 0, f'STOP primary definition: no common mean-stratum support for {cid}'
        path = OUT / ('development_features_' + cid.replace('#', '_') + '.npz')
        np.savez_compressed(path, **f)
        hashes[path.name] = sha(path)
        for k in range(4):
            rows.append(dict(cell_id=cid, quantile_boundary=(k+1)/5, lsc_threshold=float(f['lsc_edges'][k]), mean_threshold=float(f['mean_edges'][k])))
        support_rows.append(dict(cell_id=cid, samples=int(stimulus['support'].sum()), center=stimulus['center'].tolist(),
            support_sha256=hashlib.sha256(stimulus['support'].numpy().tobytes()).hexdigest(), weight_sha256=hashlib.sha256(f['weights'].tobytes()).hexdigest(),
            coordinate_sha256=hashlib.sha256(stimulus['cone_positions'].numpy().tobytes()).hexdigest(), overlap_counts=overlap))
    write_csv('feature_thresholds.csv', rows)
    from datetime import datetime, timezone
    (OUT / 'feature_definition_lock.json').write_text(json.dumps(dict(status='LOCKED_BEFORE_PER_BIN_LOSS', utc=datetime.now(timezone.utc).isoformat(),
        thresholds_sha256=sha(OUT / 'feature_thresholds.csv'), features_sha256=hashes, support_geometry=support_rows, outputs_accessed_for_feature_definition=False), indent=2), encoding='utf-8')
    print('STIMULUS FEATURE LOCK PASSED', len(source), 'cells')


if __name__ == '__main__':
    main()
