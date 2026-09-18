from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.mechanistic_retina.mechanism_observation import Intervention, InterventionSpec, observe_mechanism


CONDITIONS = ('NORMAL', 'BLOCK_H1_FEEDBACK')
SAMPLE_RATE_HZ = 150


def _keys(value: dict, required: Set[str], optional: Set[str] = frozenset()) -> None:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError(f'Expected keys {sorted(required)}; optional {sorted(optional)}')


def _placeholders(value, path: str = 'config') -> None:
    if isinstance(value, str) and value.startswith('REQUIRED'):
        raise ValueError(f'{path}: REQUIRED external definition is missing')
    if isinstance(value, dict):
        for key, item in value.items():
            _placeholders(item, f'{path}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _placeholders(item, f'{path}[{index}]')


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{name}: expected a finite number')
    return float(value)


def _bins(ms, name: str) -> int:
    value = _number(ms, name) * SAMPLE_RATE_HZ / 1000
    if value < 0 or not np.isclose(value, round(value), rtol=0, atol=1e-8):
        raise ValueError(f'{name}: must be nonnegative and exactly representable at 150 Hz; no automatic rounding')
    return round(value)


def _radii(geometry: dict) -> list[float]:
    key = 'spot_radii_deg' if 'spot_radii_deg' in geometry else 'spot_diameters_deg'
    values = geometry[key]
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError('At least two externally defined spot sizes are required')
    return [_number(v, key) / (2 if key == 'spot_diameters_deg' else 1) for v in values]


def _point(value) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError('center_deg: expected [x_deg, y_deg], in input-grid coordinates')
    return np.array([_number(v, 'center_deg') for v in value], dtype=np.float64)


def validate_config(config: dict) -> None:
    _placeholders(config)
    _keys(config, {'schema_version', 'provenance', 'sample_rate_hz', 'geometry', 'contrast_weber',
                   'polarity', 'temporal_ms', 'history', 'measurement'})
    if type(config['schema_version']) is not int or config['schema_version'] != 1 or config['sample_rate_hz'] != SAMPLE_RATE_HZ:
        raise ValueError('Only schema_version=1 and the frozen 150 Hz sampling contract are supported')
    provenance = config['provenance']
    _keys(provenance, {'geometry', 'registration', 'contrast', 'timing', 'history', 'response_metric'})
    if any(not isinstance(v, str) or not v.strip() for v in provenance.values()):
        raise ValueError('Every provenance field needs an explicit external reference/definition')
    geometry = config['geometry']
    _keys(geometry, {'center_deg', 'annulus'}, {'spot_radii_deg', 'spot_diameters_deg'})
    if ('spot_radii_deg' in geometry) == ('spot_diameters_deg' in geometry):
        raise ValueError('Provide exactly one of spot_radii_deg / spot_diameters_deg')
    centers = geometry['center_deg']
    if isinstance(centers, dict):
        if not centers or any(not isinstance(k, str) or not k for k in centers):
            raise ValueError('center_deg mapping needs explicit cell IDs')
        for point in centers.values():
            _point(point)
    else:
        _point(centers)
    radii = _radii(geometry)
    if radii[0] <= 0 or any(b <= a for a, b in zip(radii, radii[1:])):
        raise ValueError('Spot sizes must be positive, strictly increasing external values')
    if _number(config['contrast_weber'], 'contrast_weber') <= 0:
        raise ValueError('contrast_weber must be a positive magnitude; specify its signs in polarity')
    signs = config['polarity']
    if not isinstance(signs, list) or not signs or any(type(s) is not int or s not in (-1, 1) for s in signs) or len(set(signs)) != len(signs):
        raise ValueError('polarity must explicitly list +1 and/or -1, without duplicates')
    timing = config['temporal_ms']
    _keys(timing, {'onset', 'duration', 'recovery'})
    bins = {k: _bins(v, f'temporal_ms.{k}') for k, v in timing.items()}
    if bins['duration'] == 0:
        raise ValueError('Stimulus duration must be positive')
    total = sum(bins.values())
    measurement = config['measurement']
    _keys(measurement, {'window_ms', 'reference_spot_index', 'large_spot_index'})
    window = measurement['window_ms']
    if not isinstance(window, list) or len(window) != 2:
        raise ValueError('measurement.window_ms must be [start, end], relative to sequence start')
    left, right = [_bins(v, 'measurement.window_ms') for v in window]
    if not 0 <= left < right <= total:
        raise ValueError('Response window must be nonempty and inside the configured sequence')
    ref, large = measurement['reference_spot_index'], measurement['large_spot_index']
    if type(ref) is not int or type(large) is not int or not 0 <= ref < large < len(radii):
        raise ValueError('Explicit reference_spot_index < large_spot_index must index the configured spots')
    annulus = geometry['annulus']
    if annulus is not None:
        _keys(annulus, {'inner_radius_deg', 'outer_radius_deg'})
        inner = _number(annulus['inner_radius_deg'], 'annulus.inner_radius_deg')
        outer = _number(annulus['outer_radius_deg'], 'annulus.outer_radius_deg')
        if not radii[ref] <= inner < outer:
            raise ValueError('Annulus must be outside the explicitly configured reference spot')
    history = config['history']
    _keys(history, {'condition', 'occupancy'})
    if history['condition'] == 'FIX_HISTORY_ZERO':
        if history['occupancy'] is not None:
            raise ValueError('FIX_HISTORY_ZERO requires occupancy=null; supplied history is never silently discarded')
    elif history['condition'] == 'FIXED_OCCUPANCY':
        values = history['occupancy']
        if not isinstance(values, list) or len(values) != total or any(type(v) is not int or v not in (0, 1) for v in values):
            raise ValueError('FIXED_OCCUPANCY requires one explicit 0/1 value per bin, shared across all conditions')
    else:
        raise ValueError('history.condition must be FIX_HISTORY_ZERO or FIXED_OCCUPANCY; no default')


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    validate_config(config)
    return config


@dataclass(frozen=True)
class StimulusBatch:
    stimulus: np.ndarray
    history: np.ndarray
    cases: list[dict]
    grid: dict


def build_stimuli(config: dict, positions_deg: np.ndarray, cell_id: str) -> StimulusBatch:
    """Point-sample external disks on the supplied input coordinates, never on pathway supports."""
    validate_config(config)
    positions = np.asarray(positions_deg, dtype=np.float64)
    if positions.shape != (289, 2) or not np.isfinite(positions).all() or len(np.unique(positions, axis=0)) != 289:
        raise ValueError('Expected the frozen 17x17 input coordinates [289,2] in degrees')
    axes = [np.unique(positions[:, dim]) for dim in (0, 1)]
    if any(len(axis) != 17 or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-5, atol=1e-8) for axis in axes):
        raise ValueError('Only the existing regular Cartesian 17x17 grid is supported')
    geometry = config['geometry']
    configured_center = geometry['center_deg']
    if isinstance(configured_center, dict):
        if cell_id not in configured_center:
            raise ValueError(f'center_deg.{cell_id}: REQUIRED explicit external registration is missing')
        configured_center = configured_center[cell_id]
    center = _point(configured_center)
    radii = _radii(geometry)
    annulus = geometry['annulus']
    maximum_radius = max(radii + ([annulus['outer_radius_deg']] if annulus is not None else []))
    edges = np.array([[axis[0] - (axis[1] - axis[0]) / 2, axis[-1] + (axis[-1] - axis[-2]) / 2] for axis in axes])
    if np.any(center - maximum_radius < edges[:, 0] - 1e-8) or np.any(center + maximum_radius > edges[:, 1] + 1e-8):
        raise ValueError('External stimulus extends beyond the fixed input field; refusing clipping or support expansion')
    distance = np.linalg.norm(positions - center, axis=1)
    masks = [(f'spot_{i}', distance <= radius, {'radius_deg': radius, 'diameter_deg': 2 * radius}) for i, radius in enumerate(radii)]
    if annulus is not None:
        mask = (distance > annulus['inner_radius_deg']) & (distance <= annulus['outer_radius_deg'])
        masks += [('annulus', mask, dict(annulus)),
                  ('reference_plus_annulus', mask | masks[config['measurement']['reference_spot_index']][1], dict(annulus))]
    onset, duration, recovery = [_bins(config['temporal_ms'][k], k) for k in ('onset', 'duration', 'recovery')]
    total = onset + duration + recovery
    cases = [{'id': 'background', 'kind': 'background', 'contrast_sign': 0, 'indices': [], 'pixel_count': 0}]
    stimulus = [np.zeros((total, 289), dtype=np.float32)]
    seen = {}
    for sign in config['polarity']:
        for name, mask, dimensions in masks:
            indices = np.flatnonzero(mask).tolist()
            if not indices:
                raise ValueError(f'{name}: external size covers no grid centers; no nearest-pixel substitution')
            case_id = f'{name}:{sign:+d}'
            digest = hashlib.sha256(mask.tobytes()).hexdigest()
            cases.append({'id': case_id, 'kind': name, 'contrast_sign': sign, 'center_deg': center.tolist(),
                          **dimensions, 'indices': indices, 'pixel_count': len(indices), 'mask_sha256': digest,
                          'same_mask_as': seen.get((sign, digest))})
            seen.setdefault((sign, digest), case_id)
            values = np.zeros((total, 289), dtype=np.float32)
            values[onset:onset + duration, indices] = sign * config['contrast_weber']
            if not np.isfinite(values).all():
                raise ValueError('Configured contrast is not representable as finite float32')
            stimulus.append(values)
    history = np.zeros((len(cases), total, 1), dtype=np.float32)
    if config['history']['condition'] == 'FIXED_OCCUPANCY':
        history[:] = np.asarray(config['history']['occupancy'], dtype=np.float32)[None, :, None]
    return StimulusBatch(np.stack(stimulus), history, cases, {
        'center_deg': center.tolist(), 'pixel_spacing_deg': [float(axis[1] - axis[0]) for axis in axes],
        'field_edges_deg': edges.tolist(), 'positions_deg': positions.tolist(),
        'sampling': 'pixel-center inclusion; disk d<=r; annulus inner<d<=outer; no support masks',
        'time_bins': {'onset': onset, 'duration': duration, 'recovery': recovery},
    })


def summarize_logits(config: dict, batch: StimulusBatch, logits: np.ndarray, *, cell_id: str, seed: int) -> dict:
    """Use a matched zero-stimulus reference, shared normalizer, and externally indexed size pair."""
    validate_config(config)
    values = np.asarray(logits, dtype=np.float64)
    expected = (2, len(batch.cases), batch.stimulus.shape[1], 1)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f'Expected finite NORMAL/BLOCK logits {expected}')
    left, right = [_bins(v, 'measurement.window_ms') for v in config['measurement']['window_ms']]
    response = (values - values[:, :1])[:, :, left:right, 0].mean(-1)
    raw_rows, pairs = [], []
    case_index = {case['id']: i for i, case in enumerate(batch.cases)}
    ratio = lambda numerator, denominator: float(numerator / denominator) if denominator != 0 else None
    for sign in config['polarity']:
        spot_indices = [i for i, case in enumerate(batch.cases) if case['kind'].startswith('spot_') and case['contrast_sign'] == sign]
        denominator = float(np.abs(response[0, spot_indices]).max())
        for condition_index, condition in enumerate(CONDITIONS):
            for i, case in enumerate(batch.cases):
                if case['contrast_sign'] != sign:
                    continue
                raw_rows.append({'cell': cell_id, 'seed': seed, 'condition': condition, 'case_id': case['id'],
                                 'contrast_sign': sign, 'history_condition': config['history']['condition'],
                                 'R_logit': float(response[condition_index, i]),
                                 'normalization_denominator': denominator,
                                 'normalized_size_response': ratio(response[condition_index, i], denominator) if i in spot_indices else None,
                                 'normalization_status': 'DEFINED' if denominator > 0 else 'UNDEFINED_ZERO_NORMAL_CURVE'})
        ref = case_index[f"spot_{config['measurement']['reference_spot_index']}:{sign:+d}"]
        surrounds = [('size', case_index[f"spot_{config['measurement']['large_spot_index']}:{sign:+d}"])]
        if config['geometry']['annulus'] is not None:
            surrounds.append(('annulus', case_index[f'reference_plus_annulus:{sign:+d}']))
        for label, surround in surrounds:
            rn, rb = response[:, ref]
            ln, lb = response[:, surround]
            sn, sb = rn - ln, rb - lb
            ssi_n, ssi_b = ratio(sn, abs(rn)), ratio(sb, abs(rb))
            pairs.append({'cell': cell_id, 'seed': seed, 'contrast_sign': sign, 'pair': label,
                          'reference_case': batch.cases[ref]['id'], 'surround_case': batch.cases[surround]['id'],
                          'S_NORMAL': float(sn), 'S_BLOCK': float(sb), 'Delta_S_H1': float(sn - sb),
                          'Delta_reference': float(rb - rn), 'Delta_surround': float(lb - ln),
                          'SSI_NORMAL': ssi_n, 'SSI_BLOCK': ssi_b,
                          'Delta_SSI_H1': ssi_n - ssi_b if ssi_n is not None and ssi_b is not None else None,
                          'SSI_status_NORMAL': 'DEFINED' if rn != 0 else 'UNDEFINED_ZERO_REFERENCE',
                          'SSI_status_BLOCK': 'DEFINED' if rb != 0 else 'UNDEFINED_ZERO_REFERENCE'})
    return {'per_cell_per_seed': raw_rows, 'paired_interactions': pairs}


def run_frozen_checkpoint(config_path: str | Path, checkpoint_path: str | Path, expected_sha256: str,
                          output_dir: str | Path) -> dict:
    """Explicit single-checkpoint entry point; no cohort discovery, selection, training, or target loader."""
    config_bytes = Path(config_path).read_bytes()
    config = json.loads(config_bytes.decode('utf-8-sig'))
    validate_config(config)
    checkpoint_path, output_dir = Path(checkpoint_path), Path(output_dir)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Checkpoint SHA256 mismatch; use the canonical migration manifest identity')
    from work.retipath_h1_center_surround_assay import loader

    model, checkpoint = loader()(checkpoint_path)
    if checkpoint['backend'] != 'spatial_conductance' or checkpoint['entry_point'] != 'models.mechanistic_retina.retipath_canonical_gain.CanonicalGainRetiPath':
        raise ValueError('Expected the canonical frozen conductance model')
    if model.training or any(p.requires_grad for p in model.parameters()) or not np.isclose(model.config.dt_ms, 1000 / SAMPLE_RATE_HZ, rtol=0, atol=1e-10):
        raise ValueError('Checkpoint must remain inference-only and use the frozen sampling interval')
    batch = build_stimuli(config, checkpoint['cone_positions_degs'].cpu().numpy(), checkpoint['cell_id'])
    if len(checkpoint['polarities']) != 1:
        raise ValueError('Expected one frozen per-cell checkpoint')
    output_dir.mkdir(parents=True, exist_ok=False)
    lock = {'config': config, 'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
            'checkpoint_path': str(checkpoint_path.resolve()), 'checkpoint_sha256': digest,
            'assay_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cell': checkpoint['cell_id'], 'seed': checkpoint['seed'], 'model_polarity': checkpoint['polarities'][0],
            'grid': batch.grid, 'cases': batch.cases, 'conditions': list(CONDITIONS),
            'pathway_specs': [InterventionSpec(Intervention(c)).to_record() for c in CONDITIONS],
            'history_spec': InterventionSpec(Intervention.FIX_HISTORY_ZERO).to_record() if config['history']['condition'] == 'FIX_HISTORY_ZERO' else config['history'],
            'history_condition': config['history']['condition'], 'frozen_before_responses': True,
            'normalization': 'R / max_spots(abs(R_NORMAL)); same denominator for both conditions',
            'SSI': '(R_reference - R_surround) / abs(R_reference); zero denominator is undefined',
            'training_updates': 0, 'checkpoint_selection': False, 'natural_movie_target_access': False}
    (output_dir / 'lock.json').write_text(json.dumps(lock, indent=2, allow_nan=False), encoding='utf-8')
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        traces = [observe_mechanism(model, torch.from_numpy(batch.stimulus), observed_counts=torch.from_numpy(batch.history),
                                    intervention=InterventionSpec(Intervention(condition))) for condition in CONDITIONS]
    logits = np.stack([trace.logits.cpu().numpy() for trace in traces])
    results = summarize_logits(config, batch, logits, cell_id=checkpoint['cell_id'], seed=checkpoint['seed'])
    if not all(torch.equal(before[key], value) for key, value in model.state_dict().items()) or hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen state changed during observation')
    with (output_dir / 'raw_results.npz').open('xb') as handle:
        np.savez_compressed(handle, logits=logits, probability=np.stack([t.probability.cpu().numpy() for t in traces]),
                            stimulus=batch.stimulus, history=batch.history, cases_json=np.array(json.dumps(batch.cases)),
                            conditions=np.array(CONDITIONS), cell=np.array(checkpoint['cell_id']), seed=np.array(checkpoint['seed']))
    (output_dir / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    return results
