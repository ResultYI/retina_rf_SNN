"""Analysis-only complex DoG and temporal projection; no model or data loading."""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import least_squares

from croner_kaplan_center_identifiability import PITCH, gaussian


FREQUENCY_HZ = 4.0
DT_SECONDS = 1.0 / 150.0
STARTS = ((.02, .15), (.06, .6), (.16, 2.0))
LOWER = (.005, .001)
UPPER = (.4, 8.0)


def temporal_projection(jacobian: np.ndarray, *, prefix_bins: int) -> np.ndarray:
    """Input is full chronological [time,y,x], not increasing-lag order."""
    j = np.asarray(jacobian)
    if j.shape != (prefix_bins, 17, 17) or prefix_bins < 1:
        raise ValueError('full causal prefix [prefix_bins,17,17] required')
    if np.iscomplexobj(j) or not np.isfinite(j).all():
        raise ValueError('signed finite real Jacobian required')
    ages = np.arange(prefix_bins - 1, -1, -1, dtype=float)
    phase = np.exp(-2j * np.pi * FREQUENCY_HZ * ages * DT_SECONDS)
    # J differentiates bin values: there is no extra dt or FFT normalization.
    return np.einsum('t,tyx->yx', phase, j.astype(np.float64))


def complex_dog_fit(coefficients: np.ndarray, center_deg: tuple[float, float]) -> dict:
    h = np.asarray(coefficients, dtype=np.complex128)
    if h.shape != (17, 17) or not np.isfinite(h).all():
        raise ValueError('finite full 17x17 complex coefficients required')
    if np.asarray(center_deg).shape != (2,) or not np.isfinite(center_deg).all():
        raise ValueError('fixed finite (x,y) center required')
    scale = float(np.max(np.abs(h)))
    if scale == 0:
        raise ValueError('zero transfer has no identifiable radius')
    y = h.ravel() / scale

    def solve(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rc, gap = np.exp(theta)
        design = np.column_stack((gaussian(rc, center_deg, 17, PITCH, 'pixel_integral'),
                                  -gaussian(rc + gap, center_deg, 17, PITCH, 'pixel_integral')))
        # Variable projection solves two unconstrained complex gains exactly.
        gains = np.linalg.lstsq(design, y, rcond=None)[0]
        error = design @ gains - y
        return error, gains, design

    def residual(theta: np.ndarray) -> np.ndarray:
        error = solve(theta)[0]
        return np.concatenate((error.real, error.imag))

    fits = [least_squares(residual, np.log((rc, rs - rc)),
                         bounds=(np.log(LOWER), np.log(UPPER)),
                         ftol=1e-11, xtol=1e-11, gtol=1e-11,
                         max_nfev=250, x_scale='jac') for rc, rs in STARTS]
    best = min(fits, key=lambda f: float(f.fun @ f.fun))
    rc, gap = np.exp(best.x)
    error, gains, design = solve(best.x)
    gains *= scale
    return {
        'rc_deg': float(rc), 'rs_diagnostic_deg': float(rc + gap),
        'A_c_real_imag': [float(gains[0].real), float(gains[0].imag)],
        'A_s_real_imag': [float(gains[1].real), float(gains[1].imag)],
        'optimizer_success': bool(best.success), 'optimizer_message': best.message,
        'active_bounds': best.active_mask.tolist(),
        'relative_l2_residual': float(np.linalg.norm(error) / np.linalg.norm(y)),
        'gain_design_condition_number': float(np.linalg.cond(design)),
        'radius_residual_jacobian_singular_values': np.linalg.svd(best.jac, compute_uv=False).tolist(),
        'total_nfev': sum(f.nfev for f in fits),
        'starts': [{'rc_deg': float(np.exp(f.x[0])), 'sse': float(f.fun @ f.fun),
                    'success': bool(f.success), 'nfev': f.nfev} for f in fits],
    }


def synthetic_check() -> dict:
    n = 150
    ages = np.arange(n - 1, -1, -1)
    omega = 2 * np.pi * FREQUENCY_HZ * DT_SECONDS
    fixtures = []
    for rc, ratio, center in ((.052, 3., (0., 0.)), (.1, 7., (.25 * PITCH, -.25 * PITCH)),
                               (.195, 3., (0., 0.))):
        ac, ass = 1.1 + .7j, .31 - .43j
        gc = gaussian(rc, center, 17, PITCH, 'pixel_integral').reshape(17, 17)
        gs = gaussian(rc * ratio, center, 17, PITCH, 'pixel_integral').reshape(17, 17)
        h = ac * gc - ass * gs
        # Four complete cycles give independent real cosine/sine temporal bases.
        tc = 2 / n * (ac.real * np.cos(omega * ages) - ac.imag * np.sin(omega * ages))
        ts = 2 / n * (ass.real * np.cos(omega * ages) - ass.imag * np.sin(omega * ages))
        j = tc[:, None, None] * gc - ts[:, None, None] * gs
        projected = temporal_projection(j, prefix_bins=n)
        projection_error = float(np.max(np.abs(projected - h)))
        assert projection_error < 1e-13
        fit = complex_dog_fit(projected, center)
        relative_error = abs(fit['rc_deg'] / rc - 1)
        assert fit['optimizer_success'] and relative_error < 1e-6
        assert abs(fit['rs_diagnostic_deg'] / (rc * ratio) - 1) < 1e-5
        assert abs(complex(*fit['A_c_real_imag']) - ac) < 1e-5
        assert abs(complex(*fit['A_s_real_imag']) - ass) < 1e-5
        fixtures.append({'true_rc_deg': rc, 'true_rs_deg': rc * ratio,
                         'center_deg': center, 'projection_max_abs_error': projection_error,
                         'rc_relative_error': relative_error, 'fit': fit})
    pulse = np.zeros((n, 17, 17))
    pulse[n - 1 - 30, 8, 8] = -2
    assert abs(temporal_projection(pulse, prefix_bins=n)[8, 8] + 2 * np.exp(-1j * omega * 30)) < 1e-14
    rejected = False
    try:
        temporal_projection(pulse[-16:], prefix_bins=n)
    except ValueError:
        rejected = True
    assert rejected
    return {'status': 'PASS_SYNTHETIC_ONLY', 'fixtures': fixtures,
            'delayed_signed_impulse_phase': True, 'truncated_prefix_rejected': True,
            'official_checkpoint_loads': 0, 'model_rf_reads': 0,
            'limitation': 'Tests algebra on known DoG ground truth; not model measurement or physiology validation.'}


if __name__ == '__main__':
    print(json.dumps(synthetic_check(), indent=2, allow_nan=False))
