from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import TextIO

import numpy as np
import torch

from retipath_phase2_common import ROOT, SEEDS, CONDITIONS, sha, tensor_sha, load_json, paired_bootstrap
from data.retinal_recording import RealSequenceSplit

PHASE2 = ROOT/'output/experiments/retipath_spatial_ei_phase2_population'
OUT = ROOT/'output/experiments/retipath_spatial_ei_temporal_confirmation'
ORIGINAL_CONTEXT = ROOT/'.omo/evidence/context-rf-gain-20260907/calculation.py'
sys.path.append(str(ORIGINAL_CONTEXT.parent))
from calculation import indices as original_indices
assert Path(original_indices.__code__.co_filename).resolve() == ORIGINAL_CONTEXT


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_once(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def block_counts(stream: TextIO, start_s: int, stop_s: int) -> tuple[np.ndarray, dict]:
    if stop_s <= start_s or start_s < 0:
        raise ValueError('invalid temporal block')
    single = False
    for line in stream:
        if line.startswith('Video Start\t'):
            single = True
        if line.rstrip('\r\n') == 'No\tTime':
            break
    else:
        raise ValueError('DATA MISMATCH: missing single-trial table')
    if not single:
        raise ValueError('DATA MISMATCH: not a continuous 10-minute recording')
    ticks = []; previous = None; end_guard = False
    for line in stream:
        if line.startswith('Total spikes\t'):
            break
        fields = line.rstrip('\r\n').split('\t')
        if len(fields) != 2 or not fields[1].strip():
            continue
        tick = int(fields[1])
        if previous is not None and tick < previous:
            raise ValueError('DATA MISMATCH: nonchronological spike table')
        previous = tick
        if tick >= stop_s*10000:
            end_guard = True
            break
        if tick >= start_s*10000:
            ticks.append(tick)
    times_ms = np.asarray(ticks, dtype=np.float64)*0.1
    global_bins = np.floor(times_ms*150.0/1000.0).astype(np.int64)
    local_bins = global_bins-start_s*150
    if np.any(local_bins < 0) or np.any(local_bins >= (stop_s-start_s)*150):
        raise ValueError('DATA MISMATCH: binned target outside locked block')
    counts = np.bincount(local_bins, minlength=(stop_s-start_s)*150).astype(np.int64)
    return counts, {'selected_spikes': len(ticks), 'end_boundary_guard_seen': end_guard,
                    'stored_target_range_s': [start_s, stop_s], 'resolution_ms': .1,
                    'binning': 'floor((ticks*0.1)*150/1000), then subtract global start bin',
                    'video_start_shift': 'none, matching existing live-time adapter'}


def make_split(cones: np.ndarray, counts: np.ndarray, recording_id: str, start_s: int) -> RealSequenceSplit:
    count_rows = counts.reshape(len(cones), 150, 1)
    events = (count_rows > 0).astype(np.float32)
    valid = np.ones_like(events, dtype=bool); valid[:, :30] = False
    source_ids = tuple(f'{recording_id}-live-frames-{(start_s+i)*150:06d}-{(start_s+i+1)*150-1:06d}-trial-1'
                       for i in range(len(cones)))
    return RealSequenceSplit(cone_drive=torch.from_numpy(cones), spike_counts=torch.from_numpy(count_rows),
                             spike_events=torch.from_numpy(events), valid_mask=torch.from_numpy(valid),
                             source_image_ids=source_ids, trial_indices=(0,)*len(cones))


def select_observations(mask: np.ndarray, source_ids: tuple[str, ...]) -> np.ndarray:
    selected = original_indices(mask, source_ids)
    if len(selected):
        selected = selected[np.linspace(0, len(selected)-1, min(20, len(selected))).astype(np.int64)]
    return selected


def prediction_stats(rows: list[dict], cells: list[str], seed: int) -> tuple[list[dict], dict]:
    lookup = {(r['cell_id'], r['seed'], r['condition']): r for r in rows}
    draws = np.random.default_rng(seed).integers(0, len(cells), size=(10000, len(cells)))
    output = []; summaries = {}
    for left, right in (('C', 'A'), ('C', 'B'), ('B', 'A')):
        comparison = f'{left}-{right}'; role = 'primary' if comparison == 'C-A' else 'secondary'
        differences = []
        for s in SEEDS:
            values = []
            for cell in cells:
                a, b = lookup[cell, s, left], lookup[cell, s, right]
                delta = a['confirmation_nll']-b['confirmation_nll']; values.append(delta)
                output.append({'row_type': 'cell_seed', 'role': role, 'comparison': comparison, 'cell_id': cell,
                               'seed': s, 'left_nll': a['confirmation_nll'], 'right_nll': b['confirmation_nll'],
                               'delta_nll': delta, 'scored_bins': a['scored_bins']})
            values = np.asarray(values); differences.append(values)
            stat = paired_bootstrap(values, draws); summaries[f'{comparison}/{s}'] = stat
            output.append({'row_type': 'seed_summary', 'role': role, 'comparison': comparison, 'seed': s, 'n_cells': len(cells), **stat})
        cell_means = np.mean(differences, axis=0)
        for cell, value in zip(cells, cell_means, strict=True):
            output.append({'row_type': 'cell_mean_over_seeds', 'role': role, 'comparison': comparison,
                           'cell_id': cell, 'seed': 'all', 'delta_nll': float(value)})
        stat = paired_bootstrap(cell_means, draws)
        stat['seed_means'] = [float(v.mean()) for v in differences]
        stat['replicated_by_phase2_rule'] = stat['ci_high'] < 0 and all(v < 0 for v in stat['seed_means'])
        summaries[f'{comparison}/all'] = stat
        output.append({'row_type': 'population_summary', 'role': role, 'comparison': comparison,
                       'seed': 'all', 'n_cells': len(cells), **stat})
    return output, summaries
