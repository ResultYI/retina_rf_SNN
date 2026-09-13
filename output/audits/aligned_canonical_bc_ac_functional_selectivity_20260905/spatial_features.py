from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import torch

from derive_development import OUT, DERIVED, now, sha, write_csv, write_json


def features(stimulus: dict, edges: dict | None = None) -> dict[str, np.ndarray]:
    mask = stimulus['valid_mask'].numpy().astype(bool)
    assert mask.shape[1:] == (150, 1) and not mask[:, :30].any() and mask[:, 30:].all()
    drive = stimulus['cone_drive'].numpy().astype(np.float64)[mask[..., 0]]
    bc, ac = stimulus['bc'].numpy().astype(np.float64), stimulus['ac'].numpy().astype(np.float64)
    assert np.isin(bc, [0, 1]).all() and np.isin(ac, [0, 1]).all() and np.all(bc <= ac)
    ctx = ac-bc
    assert bc.sum() > 0 and ctx.sum() > 0 and np.isfinite(drive).all()
    wb, wr, wa = bc/bc.sum(), ctx/ctx.sum(), ac/ac.sum()
    central, context = np.sum(drive*wb, axis=1), np.sum(drive*wr, axis=1)
    broad_mean = np.sum(drive*wa, axis=1)
    f = dict(C=central, R=context, S=np.abs(context-central), Q=context-central, U=np.abs(central), V=np.abs(context),
        LSC=np.sqrt(np.sum(wa*(drive-broad_mean[:, None])**2, axis=1)))
    for key in ('S', 'U', 'LSC'):
        boundary = np.quantile(f[key], [.2, .4, .6, .8], method='linear') if edges is None else edges[key]
        assert np.all(np.diff(boundary) > 0)
        f[key+'_edges'] = np.asarray(boundary)
        f[key+'_quintile'] = 1+np.searchsorted(boundary, f[key], side='right')
    recording, source, local, trials, sequence, times, frames = [], [], [], [], [], [], []
    for i, (identity, trial) in enumerate(zip(stimulus['source_image_ids'], stimulus['trial_indices'], strict=True)):
        match = re.fullmatch(r'(lSS\d+)-live-frames-(\d+)-(\d+)-trial-(\d+)', identity)
        assert match is not None
        rid, start, stop, local_trial = match.groups()
        assert int(stop) == int(start)+149
        selected = np.flatnonzero(mask[i, :, 0])
        n = len(selected)
        recording.extend([rid]*n); source.extend([identity]*n); local.extend([int(local_trial)]*n)
        trials.extend([int(trial)]*n); sequence.extend([i]*n); times.extend(selected.tolist()); frames.extend((int(start)+selected).tolist())
    f.update(recording=np.asarray(recording), source_image_id=np.asarray(source), local_trial_one_based=np.asarray(local), global_trial_zero_based=np.asarray(trials),
        sequence_index=np.asarray(sequence), time_bin=np.asarray(times), live_frame=np.asarray(frames), weights_BC=wb, weights_context=wr, weights_AC=wa,
        BC_support=bc, AC_support=ac, center=stimulus['center'].numpy(), mask=mask)
    return f


def main() -> None:
    assert not (OUT / 'feature_lock.json').exists()
    assert json.loads((OUT / 'preflight.json').read_text())['all_passed']
    for p, h in json.loads((DERIVED / 'derivation_lock.json').read_text())['sha256'].items():
        assert sha(DERIVED / p) == h
    assert sha(OUT / 'PROTOCOL.md') == json.loads((OUT / 'protocol_hash.json').read_text())['sha256']
    stimuli = torch.load(OUT / 'development_stimulus.pt', weights_only=True)
    thresholds, geometry, hashes = [], [], {}
    for cid, stimulus in stimuli.items():
        f = features(stimulus)
        for key in ('S', 'U', 'LSC'):
            assert set(f[key+'_quintile']) == set(range(1, 6))
            thresholds.extend(dict(cell_id=cid, feature=key, probability=(i+1)/5, threshold=float(v)) for i, v in enumerate(f[key+'_edges']))
        q, u = f['S_quintile'], f['U_quintile']
        overlap = sum(min(int(((q == 1) & (u == s)).sum()), int(((q == 5) & (u == s)).sum())) for s in range(1, 6))
        assert overlap > 0, f'STOP U overlap {cid}'
        p = OUT / ('development_features_'+cid.replace('#', '_')+'.npz')
        np.savez_compressed(p, **f)
        hashes[p.name] = sha(p)
        geometry.append(dict(cell_id=cid, BC_samples=int(f['BC_support'].sum()), AC_samples=int(f['AC_support'].sum()), context_samples=int((f['AC_support']-f['BC_support']).sum()),
            center=f['center'].tolist(), support_exact=True, U_overlap_mass=overlap,
            tensor_sha256={k: hashlib.sha256(f[k].tobytes()).hexdigest() for k in ('BC_support', 'AC_support', 'weights_BC', 'weights_context', 'weights_AC', 'center')}))
    write_csv(OUT / 'feature_thresholds.csv', thresholds)
    write_json(OUT / 'feature_lock.json', dict(status='LOCKED_BEFORE_SELECTIVITY', utc=now(), features_sha256=hashes, thresholds_sha256=sha(OUT / 'feature_thresholds.csv'),
        geometry=geometry, derived_lock_sha256=sha(DERIVED / 'derivation_lock.json'), stimulus_sha256=sha(OUT / 'development_stimulus.pt'), output_or_spikes_used_for_features=False))
    print('FEATURE LOCK', len(stimuli), 'cells')


if __name__ == '__main__':
    main()
