"""Analysis of saved transfers and synthetic DoGs; no checkpoint/model/data loader."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from croner_kaplan_center_identifiability import PITCH, gaussian

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/evaluations/croner_kaplan_grating_measurement_audit_20260915'
OLD = ROOT / 'output/evaluations/croner_kaplan_operating_point_audit_20260915'
RADII = [.035, .045, .052, .060, .075, .100, .139, .195, .240]
NUISANCE = [(3., .2), (3., .85), (7., .2), (7., .55), (7., .85), (20., .55)]
PHASES = [0., .25, .5]
NOISE = [0., .005, .02]
SEED = 19950107
FREQUENCIES = np.geomspace(.05, 8., 33)
SUBSETS = {'full33': np.arange(33), 'sparse9': np.arange(0, 33, 4),
           'upper4': np.flatnonzero(FREQUENCIES <= 4.)}
STARTS = [(.02, .15, .5), (.06, .6, .5), (.16, 2., .5)]
LOWER = np.array([.005, .001, 1e-6])
UPPER = np.array([.4, 8., 100.])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n', encoding='utf-8')


def grating(nu, theta_deg, center=(0., 0.)):
    axis = (np.arange(17) - 8) * PITCH
    x, y = np.meshgrid(axis, axis, indexing='xy')
    theta = np.deg2rad(theta_deg)
    qx, qy = np.asarray(nu) * np.cos(theta), np.asarray(nu) * np.sin(theta)
    attenuation = np.sinc(PITCH * qx) * np.sinc(PITCH * qy)
    phase = qx[:, None] * (x.ravel() - center[0]) + qy[:, None] * (y.ravel() - center[1])
    return attenuation[:, None] * np.exp(-2j * np.pi * phase)


def fit_amplitude(nu, amplitude):
    y = np.asarray(amplitude, dtype=float)
    scale = float(y.max())
    if not np.isfinite(y).all() or scale <= 0:
        raise ValueError('finite nonzero amplitudes required')
    yn = y / scale

    def solve(t):
        rc, gap, rho = np.exp(t)
        base = np.abs(np.exp(-(np.pi * rc * nu) ** 2) - rho * np.exp(-(np.pi * (rc + gap) * nu) ** 2))
        strength = max(0., float(base @ yn / (base @ base)))
        return strength * base - yn, strength

    fits = [least_squares(lambda t: solve(t)[0], np.log([rc, rs-rc, rho]),
                         bounds=(np.log(LOWER), np.log(UPPER)),
                         ftol=1e-11, xtol=1e-11, gtol=1e-11,
                         max_nfev=250, x_scale='jac') for rc, rs, rho in STARTS]
    best = min(fits, key=lambda f: float(f.fun @ f.fun))
    rc, gap, rho = np.exp(best.x)
    residual, strength = solve(best.x)
    near = np.minimum(np.abs(np.exp(best.x) / LOWER - 1), np.abs(np.exp(best.x) / UPPER - 1)) < .001
    return {'rc_hat': float(rc), 'rs_hat': float(rc + gap), 'gap_hat': float(gap),
            'surround_center_integrated_gain_ratio': float(rho), 'overall_scale': strength * scale,
            'normalized_fit_residual': float(np.linalg.norm(residual) / np.linalg.norm(yn)),
            'solver_success': bool(best.success), 'rc_at_bound': bool(near[0]),
            'gap_at_bound': bool(near[1]), 'gain_ratio_at_bound': bool(near[2]),
            'nfev': sum(f.nfev for f in fits), 'start_rc_min': float(min(np.exp(f.x[0]) for f in fits)),
            'start_rc_max': float(max(np.exp(f.x[0]) for f in fits)),
            'radius_jacobian_smallest_singular_value': float(np.linalg.svd(best.jac, compute_uv=False)[-1])}


def predicted(nu, fit):
    return fit['overall_scale'] * np.abs(np.exp(-(np.pi * fit['rc_hat'] * nu) ** 2)
        - fit['surround_center_integrated_gain_ratio'] * np.exp(-(np.pi * fit['rs_hat'] * nu) ** 2))


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    old = json.loads((OLD / 'protocol.json').read_text())
    source_paths = [Path(__file__), ROOT / 'work/croner_kaplan_center_identifiability.py',
                    OLD / 'protocol.json', OLD / 'results.json', OLD / 'diagnostic_arrays.npz']
    protected = [ROOT / p for p in ['models/mechanistic_retina/retipath_canonical_gain.py',
        'models/mechanistic_retina/retipath_spatial_ei.py', 'models/mechanistic_retina/h1_pathway.py',
        'models/mechanistic_retina/bipolar_subunits.py', 'models/mechanistic_retina/amacrine_pathways.py',
        'evaluation/mechanistic_retina/h1_external_geometry_assay.py', 'data/schottdorf_lee_2021.py']]
    lock = {'status': 'FROZEN_BEFORE_GRATING_RESPONSE_AND_SYNTHETIC_RECOVERY',
        'semantics': {
            'temporal_frequency_hz': 4., 'source_level': 'SECONDARY_SOURCE_SUPPORTED',
            'original_methods_status': 'UNVERIFIED: publisher Methods not accessible; no claim that paper omitted them',
            'original_directly_confirmed': 'Bibliographic identity and original abstract describe ganglion-cell S-potential recording; grating/F1 details not directly confirmed from original Methods.',
            'observable': 'spatial-frequency response amplitude; model-side finite-prefix symmetric small-signal logit transfer, not finite-contrast spike-rate F1 replication',
            'sources': [
                {'url': 'https://doi.org/10.1016/0042-6989(94)E0066-T', 'level': 'ORIGINAL', 'methods_verified': False},
                {'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC3214635/#S2', 'level': 'SECONDARY_FOR_CRONER_KAPLAN', 'supports': 'Methods: P-cell discussion explicitly refers to sine-wave gratings drifting at 4 Hz; not verification of every M-cell protocol detail'},
                {'url': 'https://journals.physiology.org/doi/10.1152/jn.2001.85.1.235', 'level': 'SECONDARY_FOR_CRONER_KAPLAN', 'supports': 'Published text attributes spatial-frequency amplitude DoG to Croner-Kaplan; phase-antagonistic assumption near 4 Hz. Original 1995 F1 normalization not verified here'}],
            'original_orientation': 'UNRESOLVED', 'original_aperture': 'UNRESOLVED',
            'original_frequency_sampling': 'UNRESOLVED', 'formal_orientation_aggregation': None,
            'formula': 'R(nu)=A*abs(Ic*exp(-(pi*rc*nu)^2)-Is*exp(-(pi*rs*nu)^2))',
            'gains': 'Ic=1 gauge, Is=rho>0, free overall A>=0 solved analytically; no independent complex gains or phases',
        },
        'operating_point': {'background_Weber': 0., 'history': 'FIX_HISTORY_ZERO',
            'state': 'unmodified default initial states, V0=2/9', 'epsilon_Weber': .01,
            'dtype': 'float64', 'prefix_bins': 150, 'sample_rate_hz': 150.,
            'definition': 'finite-prefix symmetric small-signal logit transfer'},
        'diagnostics': [{'cell': c, 'seed': 2026091301,
            'kernel_key': c.replace('#','_') + '_float64_1e-02_B_kernel450'} for c in ['67#6','67#7']],
        'fixed_centers': old['engineering_amendment_coordinate_order']['functional_centers_from_already_selected_checkpoint_metadata'],
        'pixel_pitch_deg': PITCH, 'grid_shape': [17,17], 'field_edge_extent_deg': 17*PITCH,
        'grating': 'p(nu,theta)=sinc(pitch*nu*cos(theta))*sinc(pitch*nu*sin(theta))*exp(-i*2*pi*nu*((x-cx)*cos(theta)+(y-cy)*sin(theta))); Z=sum(H*p), no conjugation',
        'orientation_wavevector_deg': [0.,90.], 'aperture': 'entire existing square input field; outside field contributes zero, no FOV extrapolation',
        'frequency_cpd': FREQUENCIES.tolist(), 'frequency_subset_indices': {k:v.tolist() for k,v in SUBSETS.items()},
        'frequency_choice': '33 log-spaced .05..8 cpd, below 1/(2*pitch); chosen before response using task radius range and pixel Nyquist, not literature-exact. sparse9 keeps endpoints; upper4 tests lost high-frequency coverage.',
        'synthetic': {'radii_deg': RADII, 'primary_MC_range_deg': [.052,.195],
            'nuisance_rs_over_rc_and_Is_over_Ic': NUISANCE, 'phase_pixel': PHASES,
            'center': '(phase*pitch,phase*pitch); known fixed registration',
            'main_operator': 'unit-integral concentric DoG -> analytic Gaussian pixel integrals in current FOV -> pixel-area-average complex grating -> complex response -> amplitude',
            'orientation': '0 degrees; 90 is redundant for isotropic DoG, square FOV and equal x/y phase; synthetic symmetry fixture checks it',
            'noise_fraction_complex_response_peak_rms': NOISE, 'replicates_per_nonzero_noise': 5,
            'noise_seed': SEED, 'noise': 'independent complex Gaussian noise before abs; each real/imag SD=noise*max_abs_clean_response/sqrt(2). Reuse same full-frequency realization in all subsets. This is measurement noise, not biological noise or former pixel-coefficient noise.',
            'oracle': 'Noiseless ideal infinite-field analytic frequency DoG for all radius/nuisance/subsets; separates solver/frequency-grid recovery from combined finite-field/pixel observation mismatch. Never substitutes for main recovery.'},
        'fit': {'radius_bounds_deg': [.005,.4], 'gap_bounds_deg': [.001,8.], 'gain_ratio_bounds': [1e-6,100.],
            'starts_rc_rs_rho': STARTS, 'max_nfev_each': 250, 'tolerances': 1e-11,
            'loss': 'unweighted amplitude L2, divide observations by peak for numerical conditioning only; fit function stays analytic DoG without pixel-MTF correction',
            'no_deconvolution_or_frequency_deletion': True},
        'criteria': {'noiseless_rc_relative_error': .001, 'noisy_rc_relative_error': .10,
            'minimum_success_rate_each_phase': .90, 'max_abs_median_relative_bias_each_phase': .05,
            'success_requires': 'solver success and rc not at bound; separately retain gap/gain bounds, not hidden',
            'bound_nearness_relative': .001, 'collapse': 'adjacent fitted median radii differ by <=5% (old criterion); also retain reversed ordering, <=half log separation, empirical noise interval overlap',
            'floor': 'lowest tested rc such that every tested radius through .195 passes all three phases for the given noise/subset; .240 boundary probe excluded',
            'model_diagnostic': 'report residual, bounds, orientation/subset drift descriptively; no physiology PASS threshold and no best-orientation/subset choice'},
        'access': {'checkpoint_loads': 0, 'external_radius_reference_reads': 0, 'natural_movie_or_spike_reads': 0,
                   'B_E_Q': False, 'formal_nine_cell_measurement': False, 'training': False},
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths},
        'protected_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in protected}}
    write_json(OUT / 'protocol.json', lock)
    print('LOCKED: saved transfers only; 0/90 degrees; 33 frequencies and two fixed subsets.', flush=True)


def fixtures():
    nu = np.array([.05,1.,8.]);theta=37.;center=(.01,-.02)
    p = grating(nu,theta,center)
    axis=(np.arange(17)-8)*PITCH;x,y=np.meshgrid(axis,axis,indexing='xy')
    q,w=np.polynomial.legendre.leggauss(16);check=np.zeros_like(p)
    for dx,wx in zip(q,w):
        for dy,wy in zip(q,w):
            distance=(x.ravel()+dx*PITCH/2-center[0])*np.cos(np.deg2rad(theta))+(y.ravel()+dy*PITCH/2-center[1])*np.sin(np.deg2rad(theta))
            check += wx*wy/4*np.exp(-2j*np.pi*nu[:,None]*distance)
    error=float(np.max(abs(check-p)));assert error<1e-13
    shape=gaussian(.08,center,17,PITCH,'pixel_integral')
    phase_shift=abs(grating(nu,theta,center)@shape)-abs(grating(nu,theta)@shape)
    assert np.max(abs(phase_shift))<1e-14
    shape=gaussian(.08,(.25*PITCH,.25*PITCH),17,PITCH,'pixel_integral')
    assert np.max(abs(abs(grating(nu,0)@shape)-abs(grating(nu,90)@shape)))<1e-14
    return {'pixel_average_quadrature_max_error':error,'global_phase_amplitude_invariance':True,'synthetic_orthogonal_symmetry':True}


def run():
    lock=json.loads((OUT/'protocol.json').read_text())
    if any((OUT/f).exists() for f in ['synthetic_recovery.csv','diagnostic_model_fits.csv','diagnostic_curves.npz']):
        raise FileExistsError('refusing to overwrite completed or partial diagnostic output')
    assert all(sha(ROOT/p)==v for p,v in lock['source_sha256'].items())
    assert all(sha(ROOT/p)==v for p,v in lock['protected_sha256'].items())
    checks=fixtures();start=time.monotonic();rows=[];arrays={};case_id=0
    pattern=grating(FREQUENCIES,0)
    for rc in RADII:
        for ratio,rho in NUISANCE:
            ideal=np.abs(np.exp(-(np.pi*rc*FREQUENCIES)**2)-rho*np.exp(-(np.pi*rc*ratio*FREQUENCIES)**2))
            for sub,idx in SUBSETS.items():
                f=fit_amplitude(FREQUENCIES[idx],ideal[idx]);err=(f['rc_hat']/rc-1)
                rows.append({'family':'ideal_frequency_oracle','case_id':case_id,'true_rc':rc,'true_rs':rc*ratio,'true_rho':rho,
                    'phase':None,'noise':0.,'rep':0,'noise_seed':None,'subset':sub,**f,'bias_deg':f['rc_hat']-rc,
                    'relative_error':err,'recovery_success':f['solver_success'] and not f['rc_at_bound'] and abs(err)<=.001})
            for phase in PHASES:
                center=(phase*PITCH,phase*PITCH)
                h=gaussian(rc,center,17,PITCH,'pixel_integral')-rho*gaussian(rc*ratio,center,17,PITCH,'pixel_integral')
                z=pattern@h
                arrays[f'synthetic_{case_id}_clean_complex']=z
                for noise in NOISE:
                    for rep in range(1 if noise==0 else 5):
                        noise_seed=SEED+case_id*1000+NOISE.index(noise)*100+rep
                        rng=np.random.default_rng(noise_seed)
                        noisy=z+noise*max(abs(z))/np.sqrt(2)*(rng.standard_normal(33)+1j*rng.standard_normal(33))
                        y=abs(noisy)
                        for sub,idx in SUBSETS.items():
                            f=fit_amplitude(FREQUENCIES[idx],y[idx]);err=f['rc_hat']/rc-1
                            ok=f['solver_success'] and not f['rc_at_bound'] and abs(err)<=(.001 if noise==0 else .1)
                            rows.append({'family':'pixel_field','case_id':case_id,'true_rc':rc,'true_rs':rc*ratio,'true_rho':rho,
                                'phase':phase,'noise':noise,'rep':rep,'noise_seed':noise_seed,'subset':sub,**f,
                                'bias_deg':f['rc_hat']-rc,'relative_error':err,'recovery_success':ok})
                case_id+=1
        print(f'SYNTHETIC rc={rc:.3f}: fixed nuisance/noise/subset bank completed.',flush=True)
    with (OUT/'synthetic_recovery.csv').open('x',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    model_rows=[]
    with np.load(OLD/'diagnostic_arrays.npz') as saved:
        for spec in lock['diagnostics']:
            kernel=saved[spec['kernel_key']][:150]
            chronological_phase=np.exp(-2j*np.pi*4/150*np.arange(150))
            # Saved pixels descend in y; synthetic grating arrays ascend in y.
            h=(chronological_phase@kernel).reshape(17,17)[::-1].ravel()
            center=lock['fixed_centers'][spec['cell']]
            for theta in lock['orientation_wavevector_deg']:
                z=grating(FREQUENCIES,theta,center)@h;y=abs(z)
                key=spec['cell'].replace('#','_')+f'_theta{theta:g}'
                arrays[key+'_complex_response']=z
                for sub,idx in SUBSETS.items():
                    f=fit_amplitude(FREQUENCIES[idx],y[idx]);arrays[key+'_'+sub+'_fitted_amplitude']=predicted(FREQUENCIES,f)
                    model_rows.append({'cell':spec['cell'],'seed':spec['seed'],'theta_deg':theta,'subset':sub,
                        'frequency_count':len(idx),**f})
    with (OUT/'diagnostic_model_fits.csv').open('x',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=list(model_rows[0]));writer.writeheader();writer.writerows(model_rows)
    arrays['frequency_cpd']=FREQUENCIES
    arrays['fixture_quadrature_max_error']=np.array(checks['pixel_average_quadrature_max_error'])
    arrays['runtime_seconds']=np.array(time.monotonic()-start)
    for k,idx in SUBSETS.items():arrays['indices_'+k]=idx
    np.savez_compressed(OUT/'diagnostic_curves.npz',**arrays)
    assert all(sha(ROOT/p)==v for p,v in lock['source_sha256'].items())
    assert all(sha(ROOT/p)==v for p,v in lock['protected_sha256'].items())
    print(f'COMPLETE: {len(rows)} synthetic fits, {len(model_rows)} saved-transfer fits; no checkpoint loaded.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
