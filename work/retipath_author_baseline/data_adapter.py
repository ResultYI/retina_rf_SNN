from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "work")]
from data.retinal_recording import RealSequenceSplit
from retipath_phase2_common import load_train, make_inner_dev
from retipath_spatial_ei_pilot import tensor_sha
from models.mechanistic_retina.state import fixed_one_bin_history_state, decay_from_tau

FINAL = ROOT / "output/evaluations/retipath_final_model_evidence_20260913"
DEST = ROOT / "output/evaluations/retipath_author_baseline_assessment"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def exclusive_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


@dataclass
class CellObservations:
    targets: torch.Tensor
    history: torch.Tensor
    mask: torch.Tensor
    input_indices: torch.Tensor
    source_ids: tuple[str, ...]
    trial_indices: tuple[int, ...]
    by_input: tuple[torch.Tensor, ...]


@dataclass
class MovieBank:
    inputs: torch.Tensor
    cells: dict[str, CellObservations]
    phase: str

    def means(self) -> dict[str, float]:
        return {cell: float(obs.targets[obs.mask].mean()) for cell, obs in self.cells.items()}

    def sample(self, size: int, generator: torch.Generator) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        windows = torch.randint(len(self.inputs), (size,), generator=generator)
        rows = {}
        for cell, obs in self.cells.items():
            rows[cell] = torch.stack([obs.by_input[w][torch.randint(len(obs.by_input[w]), (), generator=generator)]
                                      for w in windows.tolist()])
        return windows, rows


def make_bank(splits: dict[str, RealSequenceSplit], phase: str) -> MovieBank:
    window_pattern = re.compile(r"-live-frames-(\d+)-(\d+)-trial-(\d+)$")
    shared: dict[tuple[int, int], tuple[int, torch.Tensor]] = {}
    cells = {}
    decay = decay_from_tau(1000 / 150, 30.0)
    for cell, split in splits.items():
        indices = []
        for row, source_id in enumerate(split.source_image_ids):
            match = window_pattern.search(source_id)
            if match is None:
                raise ValueError(f"DATA MISMATCH: unrecognized recording identity {source_id}")
            key = (int(match[1]), int(match[2]))
            stimulus = split.cone_drive[row]
            if key not in shared:
                shared[key] = (len(shared), stimulus)
            index, reference = shared[key]
            if not torch.equal(reference, stimulus):
                raise ValueError(f"DATA MISMATCH: same movie time has different inputs: {cell}, {source_id}")
            indices.append(index)
        mapping = torch.tensor(indices)
        cells[cell] = CellObservations(split.spike_events,
            fixed_one_bin_history_state(split.spike_events, decay), split.valid_mask,
            mapping, tuple(split.source_image_ids), tuple(split.trial_indices),
            tuple(torch.nonzero(mapping == i).flatten() for i in range(len(shared))))
    if any(len(obs.by_input) != len(shared) or any(len(rows) == 0 for rows in obs.by_input) for obs in cells.values()):
        raise ValueError("DATA MISMATCH: an expected training movie window is missing; no zero-fill is allowed")
    return MovieBank(torch.stack([v[1] for v in shared.values()]), cells, phase)


def training_banks() -> tuple[dict[str, MovieBank], dict]:
    registry = read_json(FINAL / "model_registry.json")
    full, fit, val = {}, {}, {}
    metadata = {}
    for cell in sorted(registry["cells"]):
        path = ROOT / "output/experiments/local_bc_subunit_nonlinearity_population_20260907/inputs" / (cell.replace("#", "_") + ".pt")
        train = load_train({"input_path": str(path)})
        inner = make_inner_dev(train)
        full[cell], fit[cell], val[cell] = train, inner.train, inner.development
        metadata[cell] = {"input_path": str(path.relative_to(ROOT)), "input_sha256": digest(path),
            "source_ids": list(train.source_image_ids), "trial_indices": list(train.trial_indices),
            "recording_ids": sorted({sid.split("-live-frames-")[0] for sid in train.source_image_ids}),
            "inner_boundaries": [asdict(b) for b in inner.boundaries],
            "train_target_sha256": tensor_sha(train.spike_events),
            "train_input_sha256": tensor_sha(train.cone_drive),
            "train_mask_sha256": tensor_sha(train.valid_mask),
            "inner_fit_scored_bins": int(inner.train.valid_mask.sum()),
            "inner_validation_scored_bins": int(inner.development.valid_mask.sum())}
    return {"inner_fit": make_bank(fit, "inner_fit"), "inner_validation": make_bank(val, "inner_validation"),
            "refit": make_bank(full, "refit")}, metadata
