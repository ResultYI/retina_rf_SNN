from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'work')]
from model import BasisForm, H1BCCapacityRetiPath, PointwiseInputBasis, capacity_parameters
from retipath_phase2_common import (SEEDS, SelectionStop, load_json, load_train,
    inner_input_mask, make_inner_dev, minibatches, compute_rms, evaluate, sha, tensor_sha)
from retipath_spatial_ei_pilot import model_args
from retipath_phase2_train import digest_state
from models.mechanistic_retina.retipath import RetiPath

OUT = ROOT / 'output/experiments/retipath_h1_bc_capacity_diagnosis'
REGISTRY = ROOT / 'output/evaluations/retipath_final_model_evidence_20260913/model_registry.json'
PHASE2 = ROOT / 'output/experiments/retipath_spatial_ei_phase2_population'
SLOPES = [.5, 1., 2., 4.]
OFFSETS = [1., 0., -1., -4.]


def durable_replace(source: Path, destination: Path) -> None:
    if os.name == 'nt':
        import ctypes
        move = ctypes.WinDLL('kernel32', use_last_error=True).MoveFileExW
        move.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        move.restype = ctypes.c_int
        if not move(str(source.resolve()), str(destination.resolve()), 1 | 8):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.replace(source, destination)


def atomic_torch_save(payload: dict, path: Path) -> None:
    if path.stem.endswith('_latest') and path.exists():
        previous = path.with_name(path.stem.removesuffix('_latest') + '_previous.pt')
        backup = previous.with_suffix('.pending')
        with path.open('rb') as source, backup.open('wb') as stream:
            shutil.copyfileobj(source, stream)
            stream.flush()
            os.fsync(stream.fileno())
        durable_replace(backup, previous)
    temporary = path.with_suffix('.pending')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    durable_replace(temporary, path)


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.pending')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
        stream.flush()
        os.fsync(stream.fileno())
    durable_replace(temporary, path)


def verify_sources(protocol: dict, out: Path) -> None:
    path = out / 'checkpoints/engineering_corrections.json'
    overrides = load_json(path)['source_replacements'] if path.exists() else {}
    for name, digest in protocol['source_hashes'].items():
        if name in overrides:
            record = overrides[name]
            assert record['original_sha256'] == digest
            assert sha(out / record['preserved_original']) == digest
            digest = record['corrected_sha256']
        assert sha(ROOT / name) == digest, name


def build(metadata: dict, seed: int, condition: str, h1_rms: float = 1., rms: dict | None = None):
    torch.manual_seed(seed)
    rms = rms or {'e': [[1., 1.]], 'i': [[1., 1.]]}
    kwargs = {'rms_e': torch.tensor(rms['e']), 'rms_i': torch.tensor(rms['i'])}
    if condition == 'A':
        return RetiPath(*model_args(metadata), **kwargs)
    if condition not in ('B', 'C'):
        raise ValueError(condition)
    bank = PointwiseInputBasis(BasisForm.LINEAR if condition == 'B' else BasisForm.MONOTONE_SOFTPLUS,
                              torch.tensor(SLOPES), None if condition == 'B' else torch.tensor(OFFSETS))
    return H1BCCapacityRetiPath(*model_args(metadata), **kwargs, input_basis=bank, h1_rms=h1_rms)


def h1_scale(model, split, mask: torch.Tensor) -> float:
    squared = torch.tensor(0., dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, len(split.cone_drive), 8):
            h1 = model.h1(split.cone_drive[start:start+8], amplitude=model.gates.values(frozenset()).h1)
            squared += (h1.modulated_cones.double().square() * mask[start:start+8, :, None]).sum()
    result = float((squared / (int(mask.sum()) * split.cone_drive.shape[-1])).sqrt().float())
    if not result > 0:
        raise ValueError('Undefined H1 train RMS')
    return result


def stage_data(ref: dict, phase: str):
    full = load_train(ref)
    if phase == 'inner':
        inner = make_inner_dev(full)
        return inner.train, inner.development, inner_input_mask(inner)
    if phase != 'refit':
        raise ValueError(phase)
    return full, None, torch.ones(full.cone_drive.shape[:2], dtype=torch.bool)


def restore(path: Path):
    cp = torch.load(path, weights_only=True, map_location='cpu')
    model = build(cp, cp['seed'], cp['condition'], cp['h1_rms'], cp['rms'])
    model.load_state_dict(cp['model'], strict=True)
    return model, cp


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix('.pending')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
        stream.flush()
        os.fsync(stream.fileno())
    durable_replace(temporary, path)
