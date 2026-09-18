from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'work'), str(ROOT / 'work/retipath_author_baseline')]
from data_adapter import MovieBank, digest, read_json, training_banks, exclusive_json
from retipath_phase2_common import SEEDS, compute_rms, inner_input_mask, load_train, make_inner_dev
from model import TypeSharedRetiPath

DEST = ROOT / 'output/evaluations/retipath_type_shared_capacity_diagnosis'
BASELINE = ROOT / 'output/evaluations/retipath_author_baseline_assessment'
REGISTRY = ROOT / 'output/evaluations/retipath_final_model_evidence_20260913/model_registry.json'
MAX_UPDATES, EVAL_EVERY, PATIENCE = 3000, 25, 400
META_KEYS = ('model_config', 'cone_positions_degs', 'cell_positions_degs', 'cell_types', 'polarities')


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def metadata_from_registry() -> tuple[dict, dict]:
    registry = read_json(REGISTRY)
    metadata = {}
    assert len(registry['cells']) == 22 and tuple(registry['seeds']) == SEEDS
    for cell, row in sorted(registry['cells'].items()):
        for seed in SEEDS:
            ref = row['RetiPath'][str(seed)]
            assert digest(ROOT / ref['path']) == ref['sha256']
        cp = torch.load(ROOT / row['RetiPath'][str(SEEDS[0])]['path'], weights_only=True, mmap=True)
        assert cp['backend'] == 'spatial_conductance' and cp['phase'] == 'refit' and cp['trainable_parameters'] == 37
        metadata[cell] = {k: cp[k] for k in META_KEYS}
    return metadata, registry


def build_initial(metadata: dict, seed: int, phase: str) -> tuple[TypeSharedRetiPath, dict]:
    model = TypeSharedRetiPath(metadata, seed)
    rms = {}
    for cell in model.cells:
        path = ROOT / 'output/experiments/local_bc_subunit_nonlinearity_population_20260907/inputs' / (cell.replace('#', '_') + '.pt')
        full = load_train({'input_path': str(path)})
        if phase == 'selection':
            inner = make_inner_dev(full)
            split, mask = inner.train, inner_input_mask(inner)
        elif phase == 'refit':
            split, mask = full, torch.ones(full.cone_drive.shape[:2], dtype=torch.bool)
        else:
            raise ValueError(phase)
        rms[cell] = compute_rms(model.cell_models[cell], split, mask)
        model.cell_models[cell].spatial_ei.rms_e.copy_(torch.tensor(rms[cell]['e']))
        model.cell_models[cell].spatial_ei.rms_i.copy_(torch.tensor(rms[cell]['i']))
    return model, rms


def batch_loss(model: TypeSharedRetiPath, bank: MovieBank, generator: torch.Generator) -> tuple[torch.Tensor, tuple]:
    windows, rows = bank.sample(4, generator)
    drives = model.stimulus_drives(bank.inputs[windows])
    losses = []
    for cell, selected in rows.items():
        obs = bank.cells[cell]
        logits = model.logits_from_drive(drives[cell], obs.history[selected], cell)
        mask = obs.mask[selected]
        losses.append(F.binary_cross_entropy_with_logits(logits[mask], obs.targets[selected][mask]))
    return torch.stack(losses).mean(), (windows, rows)


def evaluate(model: TypeSharedRetiPath, bank: MovieBank) -> tuple[float, dict, dict]:
    model.eval()
    parts = {c: [] for c in bank.cells}
    with torch.no_grad():
        for first in range(0, len(bank.inputs), 4):
            drives = model.stimulus_drives(bank.inputs[first:first+4])
            for cell in bank.cells:
                parts[cell].append(drives[cell])
        losses, logits_all = {}, {}
        for cell, obs in bank.cells.items():
            logits = model.logits_from_drive(torch.cat(parts[cell])[obs.input_indices], obs.history, cell)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(cell)
            losses[cell] = float(F.binary_cross_entropy_with_logits(logits.double()[obs.mask], obs.targets.double()[obs.mask]))
            logits_all[cell] = logits
    return sum(losses.values()) / len(losses), losses, logits_all


def save_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.pending')
    torch.save(value, temporary)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
