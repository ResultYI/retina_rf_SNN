import copy
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.mechanistic_retina.h1_external_geometry_assay import (
    build_stimuli, load_config, run_frozen_checkpoint, summarize_logits, validate_config,
)


def toy_config():
    return {
        'schema_version': 1,
        'provenance': {key: 'Synthetic unit-test values only; not physiological defaults'
                       for key in ('geometry', 'registration', 'contrast', 'timing', 'history', 'response_metric')},
        'sample_rate_hz': 150,
        'geometry': {'center_deg': [0, 0], 'spot_radii_deg': [0.125, 0.25, 0.5],
                     'annulus': {'inner_radius_deg': 0.25, 'outer_radius_deg': 0.5}},
        'contrast_weber': 0.25, 'polarity': [1, -1],
        'temporal_ms': {'onset': 20, 'duration': 40, 'recovery': 20},
        'history': {'condition': 'FIX_HISTORY_ZERO', 'occupancy': None},
        'measurement': {'window_ms': [20, 60], 'reference_spot_index': 1, 'large_spot_index': 2},
    }


def toy_grid():
    axis = np.arange(-8, 9, dtype=np.float64) / 8
    xx, yy = np.meshgrid(axis, axis)
    return np.column_stack((xx.ravel(), yy.ravel()))


def test_unfilled_template_stops_before_checkpoint_access():
    path = Path(__file__).resolve().parents[1] / 'configs/h1_external_geometry_assay.template.json'
    with pytest.raises(ValueError, match='REQUIRED'):
        load_config(path)
    with pytest.raises(ValueError, match='REQUIRED'):
        run_frozen_checkpoint(path, 'DO_NOT_OPEN_A_CHECKPOINT', 'unused', 'DO_NOT_CREATE_RESULTS')


def test_geometry_units_input_order_pulse_and_history():
    config = toy_config()
    batch = build_stimuli(config, toy_grid(), 'synthetic')
    assert batch.stimulus.shape == (11, 12, 289)
    assert batch.history.shape == (11, 12, 1) and not batch.history.any()
    assert [case['pixel_count'] for case in batch.cases[1:6]] == [5, 13, 49, 36, 49]
    assert not batch.stimulus[:, :3].any() and not batch.stimulus[:, 9:].any()
    np.testing.assert_array_equal(batch.stimulus[1:6], -batch.stimulus[6:])
    np.testing.assert_array_equal(batch.stimulus[3], batch.stimulus[5])
    assert batch.cases[5]['same_mask_as'] == 'spot_2:+1'
    diameter_config = copy.deepcopy(config)
    diameter_config['geometry']['spot_diameters_deg'] = [2 * r for r in diameter_config['geometry'].pop('spot_radii_deg')]
    diameters = build_stimuli(diameter_config, toy_grid(), 'synthetic')
    np.testing.assert_array_equal(batch.stimulus, diameters.stimulus)
    permuted = build_stimuli(config, toy_grid()[::-1], 'synthetic')
    np.testing.assert_array_equal(batch.stimulus[:, :, ::-1], permuted.stimulus)
    fixed = copy.deepcopy(config)
    fixed['geometry']['center_deg'] = {'synthetic': [0, 0]}
    fixed['history'] = {'condition': 'FIXED_OCCUPANCY', 'occupancy': [0, 1] * 6}
    fixed_batch = build_stimuli(fixed, toy_grid(), 'synthetic')
    np.testing.assert_array_equal(fixed_batch.stimulus, batch.stimulus)
    np.testing.assert_array_equal(fixed_batch.history[:, :, 0], np.tile([0, 1] * 6, (11, 1)))
    with pytest.raises(ValueError, match='registration'):
        build_stimuli(fixed, toy_grid(), 'not_registered')


def test_missing_ambiguous_and_unrepresentable_definitions_stop():
    for path, value in [
        (('contrast_weber',), 'REQUIRED'),
        (('temporal_ms', 'duration'), 1),
        (('history', 'condition'), 'OBSERVED_NATURAL_MOVIE'),
        (('geometry', 'spot_diameters_deg'), [0.25, 0.5, 1.0]),
        (('geometry', 'bc_support'), [1, 2]),
        (('measurement', 'reference_spot_index'), 'preferred_by_response'),
    ]:
        config = toy_config()
        node = config
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with pytest.raises(ValueError):
            validate_config(config)
    config = toy_config()
    config['geometry']['spot_radii_deg'][-1] = 1.1
    with pytest.raises(ValueError, match='beyond'):
        build_stimuli(config, toy_grid(), 'synthetic')
    config = toy_config()
    config['geometry']['center_deg'] = [0.0625, 0.0625]
    config['geometry']['spot_radii_deg'][0] = 0.01
    with pytest.raises(ValueError, match='no grid centers'):
        build_stimuli(config, toy_grid(), 'synthetic')


def test_signed_responses_shared_normalizer_and_paired_result_schema():
    config = toy_config()
    batch = build_stimuli(config, toy_grid(), 'synthetic')
    baseline = np.stack((np.arange(12) / 10 + 0.4, np.arange(12) / 20 - 0.3))
    logits = np.broadcast_to(baseline[:, None, :, None], (2, 11, 12, 1)).copy()
    response = {'spot_0': [1, 1.2], 'spot_1': [2, 2.1], 'spot_2': [1, 1.6],
                'annulus': [-0.5, -0.3], 'reference_plus_annulus': [1, 1.6]}
    for i, case in enumerate(batch.cases[1:], 1):
        logits[:, i, 3:9, 0] += np.array(response[case['kind']])[:, None] * case['contrast_sign']
    result = summarize_logits(config, batch, logits, cell_id='synthetic', seed=7)
    assert len(result['per_cell_per_seed']) == 20 and len(result['paired_interactions']) == 4
    row = next(r for r in result['per_cell_per_seed'] if r['case_id'] == 'spot_2:+1' and r['condition'] == 'BLOCK_H1_FEEDBACK')
    assert row['cell'] == 'synthetic' and row['seed'] == 7
    assert row['R_logit'] == pytest.approx(1.6)
    assert row['normalized_size_response'] == pytest.approx(0.8)
    for pair in result['paired_interactions']:
        sign = pair['contrast_sign']
        assert pair['S_NORMAL'] == pytest.approx(sign)
        assert pair['S_BLOCK'] == pytest.approx(sign * 0.5)
        assert pair['Delta_S_H1'] == pytest.approx(sign * 0.5)
        assert pair['Delta_S_H1'] == pytest.approx(pair['Delta_surround'] - pair['Delta_reference'])
        assert pair['SSI_NORMAL'] == pytest.approx(sign * 0.5)
        assert pair['SSI_BLOCK'] == pytest.approx(sign * 0.5 / 2.1)
    json.dumps(result, allow_nan=False)
    flat = summarize_logits(config, batch, np.zeros_like(logits), cell_id='synthetic', seed=7)
    assert all(r['normalized_size_response'] is None for r in flat['per_cell_per_seed'])
    assert all(p['SSI_NORMAL'] is None and p['Delta_SSI_H1'] is None for p in flat['paired_interactions'])
    json.dumps(flat, allow_nan=False)
