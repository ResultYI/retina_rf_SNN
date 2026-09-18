"""Standalone signed-kernel DoG measurement audit; imports no project/model code."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/evaluations/croner_kaplan_center_identifiability_20260915'
PITCH = 0.05390625
RADII = [.015, .020, .027, .035, .045, .052, .060, .075, .100, .139, .195, .240]
NUISANCE = [(3., .2), (3., .85), (7., .2), (7., .55), (7., .85), (20., .55)]
PHASES = [0., .25, .5]
NOISE = [0., .005, .02]
REPS = 5
SEED = 19950107


def coordinates(n: int, pitch: float) -> np.ndarray:
    return (np.arange(n, dtype=float) - (n - 1) / 2) * pitch


def gaussian(rc: float, xy: tuple[float, float], n: int, pitch: float, operator: str) -> np.ndarray:
    """Unit-integral Gaussian coefficients, either pixel mass or midpoint quadrature."""
    axis = coordinates(n, pitch)
    parts = []
    for c in xy:
        if operator == 'pixel_integral':
            a = .5 * (erf((axis + pitch / 2 - c) / rc) - erf((axis - pitch / 2 - c) / rc))
        else:
            a = pitch / (np.sqrt(np.pi) * rc) * np.exp(-((axis - c) / rc) ** 2)
        parts.append(a)
    return np.outer(parts[1], parts[0]).ravel()


def predict(theta: np.ndarray, case: dict, *, operator: str | None = None, center_only: bool = False) -> np.ndarray:
    op = operator or case['operator']
    if center_only:
        ic, rc = np.exp(theta)
        return ic * gaussian(rc, case['xy'], case['n'], case['pitch'], op)
    ic, ins, rc, gap = np.exp(theta)
    return (ic * gaussian(rc, case['xy'], case['n'], case['pitch'], op)
            - ins * gaussian(rc + gap, case['xy'], case['n'], case['pitch'], op))


def fit(y: np.ndarray, case: dict, *, operator: str | None = None, center_only: bool = False) -> dict:
    if center_only:
        initials = [(1., r) for r in (.02, .06, .16)]
        lower, upper = [1e-6, .005], [100., .4]
    else:
        initials = [(1., .5, r, s - r) for r, s in ((.02, .15), (.06, .6), (.16, 2.))]
        lower, upper = [1e-6, 1e-6, .005, .001], [100., 100., .4, 8.]
    scale = max(float(np.abs(y).max()), 1e-12)
    results = []
    for start in initials:
        result = least_squares(lambda t: (predict(t, case, operator=operator, center_only=center_only) - y) / scale,
                               np.log(start), bounds=(np.log(lower), np.log(upper)),
                               ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=250, x_scale='jac')
        results.append(result)
    best = min(results, key=lambda r: float(np.sum(r.fun ** 2)))
    pars = np.exp(best.x)
    rc = pars[1] if center_only else pars[2]
    ic = pars[0]
    rs = np.nan if center_only else pars[2] + pars[3]
    ins = 0. if center_only else pars[1]
    at_rc_bound = min(abs(rc - .005) / .005, abs(rc - .4) / .4) < 1e-3
    return dict(rc_hat=float(rc), rs_hat=float(rs), ic_hat=float(ic), is_hat=float(ins),
                optimizer_success=bool(best.success), rc_at_bound=at_rc_bound,
                relative_error=float((rc - case['rc']) / case['rc']),
                nfev=sum(x.nfev for x in results), normalized_sse=float(np.sum(best.fun ** 2)))


def cases() -> list[dict]:
    bank = []
    for rc in RADII:
        for phase in PHASES:
            for op, n in [('point', 17), ('pixel_integral', 17), ('pixel_integral', 51)]:
                pitch = PITCH if n == 17 else PITCH / 3
                bank.append(dict(family='center_only', rc=rc, rs=0., rho=0., phase=phase,
                                 xy=(phase * pitch, phase * pitch), n=n, pitch=pitch, operator=op))
            for ratio, rho in NUISANCE:
                bank.append(dict(family='dog_interior', rc=rc, rs=rc * ratio, rho=rho, phase=phase,
                                 xy=(phase * PITCH, phase * PITCH), n=17, pitch=PITCH, operator='pixel_integral'))
        for ratio, rho in NUISANCE:
            bank.append(dict(family='dog_edge', rc=rc, rs=rc * ratio, rho=rho, phase=.25,
                             xy=(.25 * PITCH, 6.25 * PITCH), n=17, pitch=PITCH, operator='pixel_integral'))
    for i, case in enumerate(bank):
        case['case_id'] = i
    return bank


def self_check() -> dict:
    rc = .07
    assert np.isclose(np.exp(-(rc / rc) ** 2), 1 / np.e)
    n = 17
    axis = coordinates(n, PITCH)
    assert np.isclose(axis[-1] + PITCH / 2, .458203125)
    case = dict(rc=rc, xy=(.25 * PITCH, .25 * PITCH), n=n, pitch=PITCH, operator='pixel_integral')
    # Independent Gauss-Legendre quadrature checks the erf pixel integral.
    z, w = np.polynomial.legendre.leggauss(20)
    q = sum(wi * np.exp(-((axis + zi * PITCH / 2 - case['xy'][0]) / rc) ** 2)
            for zi, wi in zip(z, w)) * PITCH / (2 * np.sqrt(np.pi) * rc)
    truth = gaussian(rc, case['xy'], n, PITCH, 'pixel_integral')
    quadrature_error = float(np.max(np.abs(np.outer(q, q).ravel() - truth)))
    assert quadrature_error < 1e-12
    fft = np.fft.fft2(truth.reshape(n, n), norm='ortho')
    assert np.isclose(np.sum(truth ** 2), np.sum(np.abs(fft) ** 2), atol=1e-13)
    result = fit(truth, case, center_only=True)
    assert abs(result['relative_error']) < 1e-7
    toy = dict(case, rs=.49, rho=.55)
    y = truth - toy['rho'] * gaussian(toy['rs'], toy['xy'], n, PITCH, 'pixel_integral')
    fitted = fit(y, toy)
    assert abs(fitted['relative_error']) < 1e-6
    return dict(quadrature_max_abs_error=quadrature_error, parseval=True,
                center_relative_error=result['relative_error'], dog_relative_error=fitted['relative_error'])


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    protected = [ROOT / p for p in (
        'models/mechanistic_retina/retipath_canonical_gain.py', 'models/mechanistic_retina/h1_pathway.py',
        'evaluation/mechanistic_retina/h1_external_geometry_assay.py',
        'data/schottdorf_lee_2021.py', 'configs/h1_external_geometry_assay.template.json',
        'docs/H1_CRONER_KAPLAN1995_FEASIBILITY.md')]
    protocol = dict(radii_deg=RADII, nuisance_ratio_gain=NUISANCE, phases=PHASES,
                    noise_fraction_of_peak=NOISE, noise_replicates=REPS, seed=SEED,
                    center_known=True, fit_starts='three fixed starts; no ground-truth initialization',
                    max_nfev_per_start=250, relative_recovery_tolerance=.10, minimum_success_rate=.90,
                    maximum_abs_median_relative_bias=.05, noiseless_tolerance=.001,
                    lower_bound_rc=.005, upper_bound_rc=.4, operator='signed pixel coefficients, exact erf integration',
                    fit_definition='unweighted L2; Parseval-equivalent to all complex discrete Fourier coefficients',
                    noise_model='iid Gaussian coefficient errors; not biological or RetiPath noise',
                    fine_reference='51x51, same physical FOV, synthetic center-only oracle; no preprocessing changes',
                    numerical_versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    protected_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},
                    cases=cases())
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(json.dumps(dict(stage='LOCKED',cases=len(protocol['cases']),path=str(OUT)), ensure_ascii=False))


def run() -> None:
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    assert protocol['source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert not (OUT / 'recovery.csv').exists()
    checks = self_check()
    bank = protocol['cases']
    arrays = {}
    clean = {}
    for case in bank:
        c = gaussian(case['rc'], case['xy'], case['n'], case['pitch'], case['operator'])
        surround = (case['rho'] * gaussian(case['rs'], case['xy'], case['n'], case['pitch'], case['operator'])
                    if case['rho'] else np.zeros_like(c))
        y = c - surround
        key = f"case_{case['case_id']:04d}"
        arrays[key] = y
        clean[case['case_id']] = (y, surround)
    np.savez_compressed(OUT / 'synthetic_bank.npz', **arrays)
    del arrays
    started = time.monotonic()
    count = 0
    with (OUT / 'recovery.csv').open('x', newline='', encoding='utf-8') as f:
        writer = None
        for index, case in enumerate(bank):
            y, surround = clean[case['case_id']]
            peak = float(np.max(np.abs(y)))
            levels = [0., .02] if case['family'] == 'dog_edge' else NOISE
            for noise in levels:
                for rep in range(1 if noise == 0 else REPS):
                    noise_seed = SEED + case['case_id'] * 100 + int(noise * 1000) * 10 + rep
                    sigma = noise * peak
                    obs = y + sigma * np.random.default_rng(noise_seed).standard_normal(y.shape)
                    result = fit(obs, case, center_only=case['family'] == 'center_only')
                    row = {k: case[k] for k in ('case_id', 'family', 'operator', 'n', 'pitch', 'rc', 'rs', 'rho', 'phase')}
                    row.update(center_x=case['xy'][0], center_y=case['xy'][1], noise=noise, rep=rep,
                               noise_seed=noise_seed, noise_sd=sigma, peak=peak, **result)
                    ok = result['optimizer_success'] and not result['rc_at_bound'] and np.isfinite(result['rc_hat'])
                    row['recovered'] = bool(ok and abs(result['relative_error']) <= (.001 if noise == 0 else .10))
                    row['oracle_rc_hat'] = np.nan
                    row['oracle_relative_error'] = np.nan
                    row['surround_captured_fraction'] = float(surround.sum() / case['rho']) if case['rho'] else np.nan
                    row['naive_point_rc_hat'] = np.nan
                    if case['rho']:
                        oracle = fit(obs + surround, case, center_only=True)
                        row['oracle_rc_hat'] = oracle['rc_hat']
                        row['oracle_relative_error'] = oracle['relative_error']
                    if noise == 0 and case['operator'] == 'pixel_integral' and case['n'] == 17:
                        row['naive_point_rc_hat'] = fit(obs, case, operator='point',
                                                      center_only=case['family'] == 'center_only')['rc_hat']
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    count += 1
            f.flush()
            if (index + 1) % 24 == 0:
                print(json.dumps(dict(stage='RECOVERY',cases=index+1,total=len(bank),fits=count,
                                      elapsed_s=round(time.monotonic()-started)),ensure_ascii=False), flush=True)
    for name, sha in protocol['protected_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    print(json.dumps(dict(stage='COMPLETE',fits=count,checks=checks,
                          elapsed_s=round(time.monotonic()-started),protected_unchanged=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'check'])
    action = parser.parse_args().action
    if action == 'prepare':
        prepare()
    elif action == 'run':
        run()
    else:
        print(json.dumps(self_check()))
