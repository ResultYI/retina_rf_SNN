from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.mechanistic_retina.retipath import RetiPath
from models.mechanistic_retina.retipath_spatial_ei import SpatialEIBackend
from retipath_phase2_common import SEEDS, load_json
from retipath_spatial_ei_pilot import model_args, sha, tensor_sha
from retipath_temporal_common import write_once, utc

OUT = ROOT / 'output/evaluations/retipath_final_model_evidence_20260913'
PHASE2 = ROOT / 'output/experiments/retipath_spatial_ei_phase2_population'
CONFIRM = ROOT / 'output/experiments/retipath_spatial_ei_temporal_confirmation'
POP = ROOT / 'output/experiments/local_bc_subunit_nonlinearity_population_20260907'
OLD_BENCH = ROOT / 'output/evaluations/final_m2_vs_cnn_temporal_benchmark_20260908'
APP_CODE = ROOT / '.omo/evidence/retipath-brightness-application-20260910'
REG_CODE = ROOT / '.omo/evidence/retipath-retinotopic-registration-20260910'
APP_OLD = ROOT / 'output/applications/retipath_brightness_illusions_20260910'
STIM_CODE = ROOT / '.omo/evidence/parametric_illusion_benchmark'
RANGES = ((16, 20), (20, 60), (60, 120), (120, 180), (180, 240), (240, 300))
CONTROL = 'without conductance integration'


def load_retipath(path: Path) -> tuple[RetiPath, dict]:
    cp = torch.load(path, weights_only=True, map_location='cpu')
    if (cp['backend'] != SpatialEIBackend.CONDUCTANCE.value or cp['phase'] != 'refit'
            or cp['step'] != cp['best_step']):
        raise ValueError('Formal RetiPath requires a frozen full-training conductance refit endpoint')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cp['seed'])
        model = RetiPath(*model_args(cp), rms_e=torch.tensor(cp['rms']['e']), rms_i=torch.tensor(cp['rms']['i']))
    model.load_state_dict(cp['model'], strict=True)
    model.eval().requires_grad_(False)
    return model, cp


def verify_sources() -> dict:
    lock = load_json(OUT / 'source_lock.json')
    corrections = OUT / 'execution_correction.json'
    overrides = load_json(corrections)['source_replacements'] if corrections.exists() else {}
    for path, digest in lock['frozen_sha256'].items():
        if path in overrides:
            record = overrides[path]
            if record['original_sha256'] != digest or sha(OUT / record['preserved_original']) != digest:
                raise ValueError(f'Original frozen source was not preserved: {path}')
            digest = record['corrected_sha256']
        if sha(Path(path) if Path(path).is_absolute() else ROOT / path) != digest:
            raise ValueError(f'Frozen source changed: {path}')
    return lock


def application_imports() -> None:
    sys.path[:0] = [str(REG_CODE), str(APP_CODE), str(STIM_CODE)]


def identity(split) -> dict:
    import hashlib

    return {'target_sha256': tensor_sha(split.spike_events), 'mask_sha256': tensor_sha(split.valid_mask),
            'stimulus_sha256': tensor_sha(split.cone_drive),
            'source_ids_sha256': hashlib.sha256('\n'.join(split.source_image_ids).encode()).hexdigest()}
