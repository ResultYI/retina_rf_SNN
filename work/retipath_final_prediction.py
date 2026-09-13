from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from pathlib import Path

import numpy as np
import torch

from retipath_final_common import (
    ROOT, OUT, POP, CONFIRM, OLD_BENCH, SEEDS, RANGES, CONTROL,
    load_json, load_retipath, verify_sources, identity, sha, write_once,
)
from retipath_phase2_common import evaluate, verify_split
from retipath_phase2_analyze import model_from_checkpoint, csv_rows
from retipath_temporal_common import block_counts, make_split
from data.retinal_recording import RealSequenceSplit
from baselines.center_surround_ln import CenterSurroundLN
from baselines.compact_causal_cnn import CompactCausalCNN
from training.mechanistic_retina.losses import expected_bernoulli_nll

DEST = OUT / 'prediction'
OLD_INPUT = ROOT / 'output/audits/macaque_aligned_heldout_pathway_evaluation_20260905'
EARLIER = ROOT / '.omo/evidence/context-gain-confirmation-20260908'
_CACHE: dict = {}


def load_tensor(path: Path) -> dict:
    return torch.load(path, weights_only=True, map_location='cpu', mmap=True)


def load_split(cell: str, start: int, stop: int) -> RealSequenceSplit:
    safe = cell.replace('#', '_')
    match start:
        case 16:
            split = RealSequenceSplit(**load_tensor(POP / 'inputs' / f'{safe}.pt')['development'])
        case 20:
            if 'twenty' not in _CACHE:
                _CACHE['twenty'] = load_tensor(OLD_INPUT / 'cnn_test_inputs.pt')
            shared = _CACHE['twenty']
            entry = shared['cells'][cell]
            old = load_tensor(OLD_INPUT / 'cells' / f'{safe}-test-predictions.pt')
            assert torch.equal(entry['events'], old['target'])
            assert old['protocol_sha256'] == shared['protocol_sha256']
            split = RealSequenceSplit(cone_drive=shared['movie_sequences'][entry['sequence_indices']],
                spike_counts=old['spike_counts'], spike_events=old['target'], valid_mask=old['valid_mask'],
                source_image_ids=old['source_image_ids'], trial_indices=old['trial_indices'])
        case 60:
            raw = dict(load_tensor(POP / 'confirmatory' / f'{safe}_split.pt'))
            raw.pop('cone_positions_degs')
            split = RealSequenceSplit(**raw)
        case 120:
            lock = load_json(OUT / 'source_lock.json')
            info = next(r for r in lock['continuous_cells'] if r['cell_id'] == cell)
            with np.load(DEST / 'input_120_180.npz', allow_pickle=False) as data:
                cones, counts = data['cones'], data[safe + '_counts']
            split = make_split(cones, counts, info['recording_id'], start)
        case 180:
            with np.load(EARLIER / 'sampling_before_targets.npz', allow_pickle=False) as data:
                cones = torch.from_numpy(data['cones'].copy())
            with np.load(EARLIER / f'{safe}.npz', allow_pickle=False) as data:
                events = torch.from_numpy(data['observed_events'].copy())
            old = load_json(ROOT / 'output/experiments/context_gain_temporal_confirmation_20260908/confirmatory_lock.json')
            info = next(r for r in old['cells'] if r['cell_id'] == cell)
            mask = torch.ones_like(events, dtype=torch.bool)
            mask[:, :30] = False
            split = RealSequenceSplit(cone_drive=cones, spike_counts=events, spike_events=events,
                valid_mask=mask, source_image_ids=tuple(info['source_ids']), trial_indices=(0,) * 60)
        case 240:
            old = load_json(CONFIRM / 'confirmatory_lock.json')
            info = next(r for r in old['cells'] if r['cell_id'] == cell)
            with np.load(CONFIRM / 'sampling_before_targets.npz', allow_pickle=False) as data:
                cones = data['cones']
            with np.load(CONFIRM / 'targets' / f'{safe}.npz', allow_pickle=False) as data:
                counts = data['counts']
            split = make_split(cones, counts, info['recording_id'], start)
        case _:
            raise ValueError('Time range is outside the migration protocol')
    verify_split(split, start * 150, stop * 150)
    return split


def prepare_consumed_120(lock: dict) -> None:
    from data.schottdorf_lee_2021 import SchottdorfAdapterConfig, _load_calibrated_lm_drive

    path = DEST / 'input_120_180.npz'
    if path.exists():
        assert sha(path) == load_json(DEST / 'input_120_180.json')['sha256']
        return
    movie = ROOT / 'data/real/schottdorf_lee_2021_macaque/1x10_256.mpg'
    drive, _ = _load_calibrated_lm_drive(movie, 180 * 150, SchottdorfAdapterConfig())
    arrays = {'cones': drive.reshape(180, 150, 289)[120:180]}
    for info in lock['continuous_cells']:
        with (ROOT / info['recording_path']).open(encoding='utf-8') as stream:
            counts, bounds = block_counts(stream, 120, 180)
        assert bounds['stored_target_range_s'] == [120, 180]
        arrays[info['cell_id'].replace('#', '_') + '_counts'] = counts
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    write_once(path.with_suffix('.json'), {'sha256': sha(path), 'range_s': [120, 180],
        'new_targets': False, 'source_lock_sha256': sha(OUT / 'source_lock.json')})


def run_cell(cell: str) -> list[dict]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    safe = cell.replace('#', '_')
    destination = DEST / 'cells' / f'{safe}.json'
    if destination.exists():
        saved = load_json(destination)
        assert saved['source_lock_sha256'] == sha(OUT / 'source_lock.json')
        return saved['rows']
    registry = load_json(OUT / 'model_registry.json')['cells'][cell]
    models = []
    for name in ('LN', 'CNN'):
        ref = registry[name]
        cp = load_tensor(ROOT / ref['path'])
        history = cp['history']
        cls = CenterSurroundLN if name == 'LN' else CompactCausalCNN
        model = cls(history['dt_ms'], history['tau_ms'], 61001)
        model.load_state_dict(cp['model'], strict=True)
        model.eval().requires_grad_(False)
        models.append((name, 'fixed', ref, model, cp))
    for seed in SEEDS:
        for name in ('RetiPath', CONTROL):
            ref = registry[name][str(seed)]
            model, cp = (load_retipath(ROOT / ref['path']) if name == 'RetiPath'
                         else model_from_checkpoint(ROOT / ref['path']))
            model.eval().requires_grad_(False)
            models.append((name, seed, ref, model, cp))
    old_rows = {(r['range'], r['cell_id']): r for r in csv.DictReader((OLD_BENCH / 'per_cell_per_range.csv').open(encoding='utf-8'))}
    eligible = {r['cell_id'] for r in load_json(OUT / 'source_lock.json')['continuous_cells']}
    rows = []
    tensors = {}
    for start, stop in RANGES:
        if start >= 60 and cell not in eligible:
            continue
        split = load_split(cell, start, stop)
        shared = identity(split)
        label = f'[{start},{stop})'
        if start < 240:
            old = old_rows[label, cell]
            assert all(shared[k] == old[k] for k in shared), (cell, label, 'DATA MISMATCH')
        for name, seed, ref, model, cp in models:
            with torch.inference_mode():
                if name in ('LN', 'CNN'):
                    logits = torch.cat([model(split.cone_drive[i:i+8], split.spike_events[i:i+8])
                                        for i in range(0, len(split.cone_drive), 8)])
                    nll = float(expected_bernoulli_nll(logits.double(), split.spike_events.double(), split.valid_mask))
                else:
                    nll, logits = evaluate(model, split)
            assert torch.isfinite(logits).all()
            assert all(torch.equal(v, cp['model'][k]) for k, v in model.state_dict().items())
            assert all(p.grad is None for p in model.parameters())
            legacy_error = None
            if start < 240 and name in ('LN', 'CNN'):
                legacy_nll = float(expected_bernoulli_nll(logits, split.spike_events, split.valid_mask))
                legacy_error = abs(legacy_nll - float(old[name + '_nll']))
                assert legacy_error < 1e-7
            if start == 240 and name in ('RetiPath', CONTROL):
                prior = load_json(CONFIRM / 'results' / f'{safe}_{seed}.json')
                tag = 'C' if name == 'RetiPath' else 'A'
                row = next(r for r in prior['prediction'] if r['condition'] == tag)
                assert nll == row['confirmation_nll']
            key = f'{start}_{name}_{seed}'
            tensors[key] = logits.numpy()
            rows.append({'range': label, 'cell_id': cell, 'group': registry['group'], 'model': name, 'seed': seed,
                'nll': nll, 'scored_bins': int(split.valid_mask.sum()), 'sequences': len(split.cone_drive),
                **shared, 'checkpoint_path': ref['path'], 'checkpoint_sha256': ref['sha256'],
                'legacy_float32_NLL_replay_error': legacy_error, 'logits_key': key, 'parameters_unchanged': True})
    with destination.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **tensors)
    write_once(destination, {'cell_id': cell, 'source_lock_sha256': sha(OUT / 'source_lock.json'), 'rows': rows})
    print(f'PREDICTION {cell}: {len(rows)} frozen scores', flush=True)
    return rows


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    summaries, per_cell = [], []
    lookup = {(r['range'], r['cell_id'], r['model'], str(r['seed'])): r['nll'] for r in rows}
    for start, stop in RANGES:
        label = f'[{start},{stop})'
        cells = sorted({r['cell_id'] for r in rows if r['range'] == label})
        draws = np.random.default_rng(20260908).integers(0, len(cells), size=(100000, len(cells)))
        for comparator in ('CNN', 'LN', CONTROL):
            differences = []
            for seed in SEEDS:
                values = np.array([lookup[label, cell, 'RetiPath', str(seed)] -
                    lookup[label, cell, comparator, str(seed) if comparator == CONTROL else 'fixed'] for cell in cells])
                differences.append(values)
            for seed, values in [*zip(SEEDS, differences, strict=True), ('aggregate', np.mean(differences, axis=0))]:
                low, high = np.quantile(values[draws].mean(1), [.025, .975])
                summaries.append({'range': label, 'comparison': 'RetiPath - ' + comparator, 'seed': seed,
                    'n_cells': len(cells), 'mean': float(values.mean()), 'median': float(np.median(values)),
                    'wins': int((values < -1e-7).sum()), 'losses': int((values > 1e-7).sum()),
                    'ties': int((np.abs(values) <= 1e-7).sum()), 'ci_low': float(low), 'ci_high': float(high),
                    'bootstrap_samples': 100000, 'bootstrap_seed': 20260908, 'evidence_status': 'consumed-range descriptive reuse'})
                per_cell.extend({'range': label, 'cell_id': cell, 'comparison': 'RetiPath - ' + comparator,
                                 'seed': seed, 'delta_nll': float(value)} for cell, value in zip(cells, values, strict=True))
    return summaries, per_cell


def main() -> None:
    lock = verify_sources()
    DEST.mkdir(exist_ok=True)
    (DEST / 'cells').mkdir(exist_ok=True)
    prepare_consumed_120(lock)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_cell, cell) for cell in lock['cells']]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda r: (r['range'], r['cell_id'], r['model'], str(r['seed'])))
    assert len(rows) == 896
    summaries, pairs = summarize(rows)
    csv_rows(DEST / 'per_cell_seed.csv', rows)
    csv_rows(DEST / 'comparisons_per_cell.csv', pairs)
    csv_rows(DEST / 'population.csv', summaries)
    verify_sources()
    write_once(DEST / 'complete.json', {'status': 'VERIFIED', 'scores': len(rows), 'summaries': len(summaries),
        'new_target_blocks': 0, 'training_updates': 0, 'source_lock_sha256': sha(OUT / 'source_lock.json')})
    print('PREDICTION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
