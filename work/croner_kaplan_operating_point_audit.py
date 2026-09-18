"""Bounded numerical measurement audit; never reads external physiology targets."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.retipath_canonical_gain import CanonicalGainRetiPath
from croner_kaplan_mc_center_measurement import complex_dog_fit

OUT = ROOT / 'output/evaluations/croner_kaplan_operating_point_audit_20260915'
MANIFEST = ROOT / 'output/evaluations/retipath_gain_canonicalization_20260915/migration_manifest.json'
CELLS = ('67#6', '67#7')
SEED = 2026091301
EPS = [10. ** -k for k in range(2, 10)]
T = 450
OMEGA = 2 * np.pi * 4 / 150


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(payload)


def patterns():
    y, x = np.meshgrid(np.arange(17), np.arange(17), indexing='ij')
    rows = []
    for yy, xx in ((8, 8), (4, 8), (8, 4), (12, 12)):
        a = np.zeros((17, 17)); a[yy, xx] = 1
        rows.append(a.ravel())
    rows += [np.ones(289), np.cos(2 * np.pi * x / 17).ravel(),
             np.cos(2 * np.pi * y / 17).ravel(), ((-1.) ** (x + y)).ravel(),
             np.random.default_rng(19950107).choice([-1., 1.], 289)]
    return np.stack(rows)


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    m = json.loads(MANIFEST.read_text(encoding='utf-8'))
    refs = [{k: r[k] for k in ('cell', 'seed', 'path', 'sha256')}
            for r in m['checkpoints'] if r['cell'] in CELLS and r['seed'] == SEED]
    assert len(refs) == 2
    sources = [Path(__file__), ROOT / 'work/croner_kaplan_mc_center_measurement.py',
               ROOT / 'work/croner_kaplan_center_identifiability.py']
    sources += [ROOT / 'models/mechanistic_retina' / f for f in (
        'retipath_canonical_gain.py', 'retipath_spatial_ei.py', 'h1_pathway.py', 'graph.py',
        'bipolar_subunits.py', 'amacrine_pathways.py', 'state.py', 'pathway_temporal.py',
        'temporal_parameters.py', 'local_bc_nonlinearity.py', 'pathway_gates.py', 'rgc_state.py')]
    lock = {
        'status': 'FROZEN_BEFORE_CHECKPOINT_DIAGNOSTICS', 'cells': CELLS, 'seed': SEED,
        'choice_reason': 'One predetermined OFF and one ON MC cell; no external reference or model radius used for selection.',
        'checkpoints': refs, 'epsilon_Weber': EPS, 'precisions': ['float32', 'float64'],
        'device': 'cuda', 'tf32': False, 'batch_pixels': 16, 'natural_data_access': False,
        'external_reference_access': False, 'B_E_Q_computation': False,
        'initial_state': 'Unmodified forward defaults; H1/AC/adaptation/history deviations zero, membrane V0=2/9.',
        'A': 'Full signed final-logit autograd Jacobian on 150 zero bins, zero history; diagnostic only.',
        'B': 'Symmetric finite differences for each of 289 unit-pixel impulses at bin 0; collect all 450 output bins. Shift covariance checked, giving complete finite-prefix kernel.',
        'B_direction_check': 'Direct final-logit symmetric FD on cosine and negative-sine temporal directions for nine fixed spatial patterns; compare with 150-bin kernel projection.',
        'C': 'Zero-mean 4Hz cosine, nine fixed spatial patterns, 450 bins = 12 cycles; 8-cycle pre-roll then 4-cycle F1, also first and preceding 4-cycle windows.',
        'pre_roll_reason': '2 seconds is 8 times the prior AC maximum tau=250ms; assess remaining transients by adjacent 4-cycle windows; no automatic extension.',
        'pattern_names': ['pixel_8_8', 'pixel_4_8', 'pixel_8_4', 'pixel_12_12', 'full_field', 'cos_x', 'cos_y', 'checkerboard', 'fixed_random_signs'],
        'patterns': patterns().tolist(),
        'numerical_rules': {
            'response_floor_multiplier': 128, 'floor': '128*machine_eps*max(1,max_abs_logits); peak paired response must exceed it.',
            'adjacent_normalized_complex_profile_l2_max': .001,
            'adjacent_radius_relative_change_max': .001,
            'minimum_consecutive_epsilon_points': 3,
            'B_direction_error_relative_l2_max': .01,
            'C_vs_B_long_error_relative_l2_max': .01,
            'C_adjacent_cycle_window_error_relative_l2_max': .01,
            'finite150_vs_long450_error_relative_l2_max': .01,
            'selection': 'Common stable three-point window across both diagnostic cells in float64; choose largest epsilon in its widest contiguous region, never by physiology agreement.',
            'scope': 'Numerical tolerances only, not physiological acceptance thresholds.'},
        'fit': 'Reuse existing complex pixel-integral DoG with fixed functional center; rc used only for epsilon stability, no external lookup.',
        'history_check': 'zero vs fixed events every 17 bins; compare upstream state, logit difference and full input gradient; STOP if stimulus derivative changes materially.',
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        'no_training_or_checkpoint_writes': True,
    }
    write_json(OUT / 'protocol.json', lock)
    print('PREPARED: 2 cells, 1 seed, 8 epsilons, two precisions; no checkpoint loaded.', flush=True)


def construct(ref):
    path = ROOT / ref['path']
    assert sha(path) == ref['sha256']
    cp = torch.load(path, map_location='cpu', weights_only=True)
    assert cp['gain_schema'] == 'retipath_canonical_gain_v1' and cp['cell_id'] == ref['cell']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        model = CanonicalGainRetiPath(MechanisticRetinaConfig(**cp['model_config']),
            cp['cone_positions_degs'], cp['cell_positions_degs'], tuple(cp['cell_types']), tuple(cp['polarities']),
            rms_e=torch.tensor(cp['rms']['e']), rms_i=torch.tensor(cp['rms']['i']))
    model.load_state_dict(cp['model'], strict=True)
    model.eval().requires_grad_(False)
    assert model.config.lag_steps == 16 and abs(model.config.dt_ms - 1000 / 150) < 1e-12
    axis = (np.arange(17) - 8) * .05390625
    xx, yy = np.meshgrid(axis, axis[::-1], indexing='xy')
    expected_positions = np.stack((xx.ravel(), yy.ravel()), -1)
    assert np.allclose(cp['cone_positions_degs'].numpy(), expected_positions, atol=1e-7, rtol=0)
    return model, tuple(cp['cell_positions_degs'].double().numpy()[0])


def response(model, x, history=None):
    if history is None:
        history = x.new_zeros((*x.shape[:2], 1))
    return model(x, observed_counts=history)


def rel(a, b):
    denominator = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / denominator) if denominator > 0 else None


def diagnostic_fit(h, center):
    if not np.isfinite(h).all() or np.linalg.norm(h) == 0:
        return {'rc_deg': None, 'optimizer_success': False, 'relative_l2_residual': None, 'active_bounds': []}
    # Checkpoint rows descend in y; the independent fitter rows ascend in y.
    return complex_dog_fit(h.reshape(17, 17)[::-1], center)


def projection(kernel):
    return np.exp(-1j * OMEGA * np.arange(len(kernel))) @ kernel


def audit_zero(model, dtype):
    x = torch.zeros((1, 150, 289), device='cuda', dtype=dtype, requires_grad=True)
    a = response(model, x)
    jac = torch.autograd.grad(a.logits[0, -1, 0], x)[0].detach().cpu().double().numpy()[0]
    h = x.new_zeros((1, 150, 1)); h[:, ::17] = 1
    x2 = x.detach().clone().requires_grad_(True)
    b = response(model, x2, h)
    jac2 = torch.autograd.grad(b.logits[0, -1, 0], x2)[0].detach().cpu().double().numpy()[0]
    delta = b.logits - a.logits
    predicted = -model.gates.history * model.rgc.history_gain * b.rgc_history_state
    upstream = ['h1_state', 'rgc_membrane', 'rgc_adaptation',
                'bc_sustained_current', 'bc_transient_current',
                'amacrine_local_state', 'amacrine_transient_state']
    state_errors = {name: float((getattr(a, name) - getattr(b, name)).detach().abs().max()) for name in upstream}
    zero = x.detach()
    parts = model.spatial_components(zero)
    ge, gi = model.spatial_ei.conductances(parts.effective_e, parts.effective_i)
    v = model.spatial_ei.voltage(ge, gi)
    row = {'alpha': float(model.alpha), 'symmetry_factor': float((1 + model.alpha) / 2),
        'h1_zero_max': float(parts.h1.state.abs().max()), 'BC_direct_zero_max': float(parts.direct.abs().max()),
        'AC_zero_max': float(parts.ac_states.abs().max()), 'gE_zero': float(ge[0, 0, 0, 0]),
        'gI_zero': float(gi[0, 0, 0, 0]), 'V0': model.spatial_ei.v_0,
        'membrane_zero_max': float(a.rgc_membrane.abs().max()),
        'voltage_minus_V0_max': float((v - model.spatial_ei.v_0).abs().max()),
        'logit_zero_range': [float(a.logits.min()), float(a.logits.max())],
        'history_gradient_max_abs_difference': float(np.max(np.abs(jac2 - jac))),
        'history_gradient_numerical_tolerance': 128 * torch.finfo(dtype).eps * max(1., float(np.abs(jac).max())),
        'history_additive_logit_max_abs_error': float((delta - predicted).abs().max()),
        'history_upstream_max_abs_difference': state_errors,
        'tau_h1_bc_ac': [float(model.h1.tau_ms), model.feature_bank.tau_ms.cpu().tolist(), model.amacrine.tau_ms.cpu().tolist()],
    }
    if row['history_gradient_max_abs_difference'] > row['history_gradient_numerical_tolerance'] or any(state_errors.values()):
        raise RuntimeError('STOP: history changed stimulus-driven state or Jacobian')
    return jac[::-1].copy(), row


def impulse_kernel(model, dtype, eps):
    blocks = []; peak = 0.; magnitude = 1.
    with torch.inference_mode():
        for begin in range(0, 289, 16):
            count = min(16, 289 - begin)
            x = torch.zeros((count, T, 289), device='cuda', dtype=dtype)
            x[torch.arange(count), 0, torch.arange(begin, begin + count)] = eps
            z = response(model, torch.cat((x, -x))).logits[..., 0]
            delta = z[:count] - z[count:]
            peak = max(peak, float(delta.abs().max()))
            magnitude = max(magnitude, float(z.abs().max()))
            blocks.append((delta.double() / (2 * eps)).cpu().numpy().T)
    return np.concatenate(blocks, 1), peak, magnitude


def directional_checks(model, dtype, eps, kernel, pat):
    p = torch.tensor(pat, device='cuda', dtype=dtype)
    times = torch.arange(T, device='cuda', dtype=dtype)
    phase = times * OMEGA
    x = eps * torch.cos(phase)[None, :, None] * p[:, None, :]
    with torch.inference_mode():
        z = response(model, x).logits[..., 0]
        base = response(model, x.new_zeros((1, T, 289))).logits[..., 0]
        dz = (z - base).double().cpu().numpy()
        angle = np.exp(-1j * OMEGA * np.arange(T))
        f1 = [2 / (150 * eps) * (dz[:, start:start+150] @ angle[start:start+150]) for start in (0, 150, 300)]
        ages = torch.arange(149, -1, -1, device='cuda', dtype=dtype)
        temporal = torch.stack((torch.cos(OMEGA * ages), -torch.sin(OMEGA * ages)))
        directions = (temporal[:, None, :, None] * p[None, :, None, :]).reshape(-1, 150, 289)
        bz = response(model, torch.cat((eps * directions, -eps * directions))).logits[:, -1, 0]
        d = ((bz[:len(directions)] - bz[len(directions):]).double() / (2 * eps)).cpu().numpy().reshape(2, -1)
        dsym = d[0] + 1j * d[1]
        shifted = torch.zeros((4, 150, 289), device='cuda', dtype=dtype)
        for row, (pixel, at) in enumerate(((144, 25), (144, 75), (76, 25), (76, 75))):
            shifted[row, at, pixel] = eps
        zs = response(model, torch.cat((shifted, -shifted))).logits[:, -1, 0]
        actual = ((zs[:4] - zs[4:]).double() / (2 * eps)).cpu().numpy()
    expected = np.array([kernel[149-at, pixel] for pixel, at in ((144, 25), (144, 75), (76, 25), (76, 75))])
    b150, blong = pat @ projection(kernel[:150]), pat @ projection(kernel)
    return {'C_vs_B450_relative_l2': rel(f1[2], blong), 'C_last_vs_previous_relative_l2': rel(f1[2], f1[1]),
        'C_first_vs_last_relative_l2': rel(f1[0], f1[2]), 'B_direction_relative_l2': rel(dsym, b150),
        'shift_covariance_max_abs_error': float(np.max(np.abs(actual - expected))),
        'shift_covariance_relative_l2': rel(actual, expected)}, np.stack([b150, blong, dsym, *f1])


def run():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT / p) == digest for p, digest in lock['source_sha256'].items())
    pat = np.array(lock['patterns']); arrays = {}; rows = []; zero_rows = []; start = time.monotonic()
    for ref in lock['checkpoints']:
        original, center = construct(ref)
        for dtype in (torch.float32, torch.float64):
            model = copy.deepcopy(original).to(device='cuda', dtype=dtype).eval().requires_grad_(False)
            prefix = ref['cell'].replace('#', '_') + '_' + str(dtype).split('.')[-1]
            state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            a, zero = audit_zero(model, dtype)
            zero_rows.append({'cell': ref['cell'], 'dtype': str(dtype), **zero})
            arrays[prefix + '_A_kernel150'] = a
            previous = None
            for eps in EPS:
                kernel, peak, magnitude = impulse_kernel(model, dtype, eps)
                h150, h450 = projection(kernel[:150]), projection(kernel)
                fit = diagnostic_fit(h150, center)
                fit_long = diagnostic_fit(h450, center)
                checks, directional = directional_checks(model, dtype, eps, kernel, pat)
                floor = 128 * torch.finfo(dtype).eps * magnitude
                row = {'cell': ref['cell'], 'dtype': str(dtype), 'epsilon': eps,
                    'paired_response_peak': peak, 'numerical_floor': floor, 'response_floor_ratio': peak / floor,
                    'rc150': fit['rc_deg'], 'rc450': fit_long['rc_deg'],
                    'fit_success': fit['optimizer_success'] and fit_long['optimizer_success'],
                    'fit150_relative_residual': fit['relative_l2_residual'],
                    'fit450_relative_residual': fit_long['relative_l2_residual'],
                    'active_bounds150': fit['active_bounds'], 'active_bounds450': fit_long['active_bounds'],
                    'B150_vs_B450_relative_l2': rel(h150, h450),
                    'B150_vs_scaled_A_relative_l2': rel(kernel[:150], zero['symmetry_factor'] * a), **checks}
                valid_radius = fit['rc_deg'] is not None and fit_long['rc_deg'] is not None
                if previous is not None and valid_radius:
                    ph150, ph450, pr, prl = previous
                    row['adjacent_profile_l2'] = max(rel(h150 / np.linalg.norm(h150), ph150 / np.linalg.norm(ph150)),
                                                     rel(h450 / np.linalg.norm(h450), ph450 / np.linalg.norm(ph450)))
                    row['adjacent_rc_relative_change'] = max(abs(fit['rc_deg'] / pr - 1), abs(fit_long['rc_deg'] / prl - 1))
                previous = (h150, h450, fit['rc_deg'], fit_long['rc_deg']) if valid_radius else None
                key = prefix + '_' + format(eps, '.0e')
                arrays[key + '_B_kernel450'] = kernel
                arrays[key + '_directions_B150_B450_Dsym_Cfirst_Cprev_Clast'] = directional
                rows.append(row)
                harmonic = checks['C_vs_B450_relative_l2']
                label = 'UNDEFINED' if harmonic is None else f'{harmonic:.3g}'
                print(f'DIAGNOSTIC {prefix} eps={eps:.0e} floor={peak/floor:.2g} harmonic={label}', flush=True)
            assert all(torch.equal(v.detach().cpu(), state_before[k]) for k, v in model.state_dict().items())
            del model
            torch.cuda.empty_cache()
        assert sha(ROOT / ref['path']) == ref['sha256']
    np.savez_compressed(OUT / 'diagnostic_arrays.npz', **arrays)
    write_json(OUT / 'results.json', {'status': 'DIAGNOSTICS_COMPLETE', 'rows': rows, 'zero_history_checks': zero_rows,
        'protocol_sha256': sha(OUT / 'protocol.json'), 'runtime_seconds': time.monotonic() - start,
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(), 'checkpoint_count': 2,
        'state_dicts_unchanged': True, 'checkpoint_hashes_unchanged': True,
        'external_target_reads': 0, 'formal_nine_cell_measurement': False, 'B_E_Q_computed': False})
    assert all(sha(ROOT / p) == digest for p, digest in lock['source_sha256'].items())
    print('COMPLETE: numerical diagnostics only; checkpoints and source unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
