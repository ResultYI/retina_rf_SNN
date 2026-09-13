from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from data.retinal_recording import RealSequenceSplit
from evaluation.mechanistic_retina.factorized_ln_split import make_inner_dev, InnerDevSplit
from models.mechanistic_retina.retipath_spatial_ei import SpatialEIRetiPath, SpatialEIBackend
from training.mechanistic_retina.losses import expected_bernoulli_nll
from retipath_spatial_ei_pilot import model_args, sha, tensor_sha

POP = ROOT / 'output/experiments/local_bc_subunit_nonlinearity_population_20260907'
PILOT = ROOT / 'output/experiments/retipath_spatial_ei_pilot'
SEEDS = (2026091301, 2026091302, 2026091303)
CONDITIONS = {'A': SpatialEIBackend.RETIPATH, 'B': SpatialEIBackend.CURRENT,
              'C': SpatialEIBackend.CONDUCTANCE}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load_train(ref: dict) -> RealSequenceSplit:
    archive = torch.load(ref['input_path'], weights_only=True, mmap=True)
    if set(archive) != {'train', 'development'}:
        raise ValueError('unexpected cached data splits')
    train = RealSequenceSplit(**archive['train'])
    verify_split(train, 0, 2400)
    return train


def load_development(ref: dict) -> RealSequenceSplit:
    archive = torch.load(ref['input_path'], weights_only=True, mmap=True)
    dev = RealSequenceSplit(**archive['development'])
    verify_split(dev, 2400, 3000)
    return dev


def verify_split(split: RealSequenceSplit, low: int, high: int) -> None:
    if split.cone_drive.shape[1:] != (150, 289):
        raise ValueError('DATA MISMATCH: sequence or frontend shape')
    for source_id in split.source_image_ids:
        match = re.search(r'live-frames-(\d+)-(\d+)-trial-', source_id)
        if match is None or not low <= int(match[1]) <= int(match[2]) < high:
            raise ValueError('DATA MISMATCH: cached source range')
    if not torch.equal(split.spike_events, (split.spike_counts > 0).to(split.spike_events)):
        raise ValueError('DATA MISMATCH: occupancy target')
    expected = torch.ones_like(split.valid_mask); expected[:, :30] = False
    if not torch.equal(split.valid_mask, expected):
        raise ValueError('DATA MISMATCH: sequence warmup mask')


def fresh_model(metadata: dict, seed: int, condition: str, rms: dict | None = None,
                upstream: dict | None = None) -> SpatialEIRetiPath:
    torch.manual_seed(seed)
    kwargs = {} if condition == 'A' else {'rms_e': torch.tensor(rms['e']), 'rms_i': torch.tensor(rms['i'])}
    model = SpatialEIRetiPath(*model_args(metadata), backend=CONDITIONS[condition], **kwargs)
    if upstream is not None:
        state = model.state_dict(); state.update(upstream); model.load_state_dict(state, strict=True)
    return model


def inner_input_mask(inner: InnerDevSplit) -> torch.Tensor:
    boundaries = {b.trial_index: b for b in inner.boundaries}
    masks = []
    for source_id, trial in zip(inner.train.source_image_ids, inner.train.trial_indices, strict=True):
        match = re.search(r'live-frames-(\d+)-', source_id)
        if match is None:
            raise ValueError('missing native source identity')
        masks.append(int(match[1]) + torch.arange(150) < boundaries[trial].fit_stop)
    return torch.stack(masks)


def compute_rms(model: SpatialEIRetiPath, train: RealSequenceSplit, input_mask: torch.Tensor) -> dict:
    squared_e = torch.zeros(1, 2, dtype=torch.float64)
    squared_i = torch.zeros_like(squared_e)
    with torch.no_grad():
        for start in range(0, len(train.cone_drive), 8):
            parts = model.spatial_components(train.cone_drive[start:start+8])
            mask = input_mask[start:start+8, :, None, None]
            squared_e += (parts.u_e.double().square()*mask).sum((0, 1))
            squared_i += (parts.u_i.double().square()*mask).sum((0, 1))
    count = int(input_mask.sum())
    e, i = (squared_e/count).sqrt().float(), (squared_i/count).sqrt().float()
    if not bool(torch.isfinite(e).all() & torch.isfinite(i).all() & (e > 0).all() & (i > 0).all()):
        raise ValueError('invalid initial fit-only RMS')
    return {'e': e.tolist(), 'i': i.tolist(), 'input_bins_including_warmup': count}


def minibatches(sequence_count: int, seed: int, steps: int = 3000) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 1_000_003)
    return torch.stack([torch.randint(sequence_count, (4,), generator=generator) for _ in range(steps)])


def evaluate(model: SpatialEIRetiPath, split: RealSequenceSplit) -> tuple[float, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(split.cone_drive[i:i+8], observed_counts=split.spike_events[i:i+8]).logits
                            for i in range(0, len(split.cone_drive), 8)])
        nll = expected_bernoulli_nll(logits.double(), split.spike_events.double(), split.valid_mask)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('nonfinite model output')
    return float(nll), logits


@dataclass(frozen=True)
class SelectionStop:
    best_nll: float
    best_step: int
    plateau_nll: float
    last_improvement_step: int

    def observe(self, nll: float, step: int) -> SelectionStop:
        better = nll < self.best_nll
        significant = nll < self.plateau_nll - 1e-7
        return SelectionStop(nll if better else self.best_nll, step if better else self.best_step,
                             nll if significant else self.plateau_nll,
                             step if significant else self.last_improvement_step)

    def stopped(self, step: int) -> bool:
        return step - self.last_improvement_step >= 400


def jacobian_modes(model: SpatialEIRetiPath, stimulus: torch.Tensor,
                   observed_history: torch.Tensor) -> tuple[np.ndarray, np.ndarray, dict]:
    x = stimulus.detach().clone().requires_grad_()
    history = observed_history.detach()
    captured = []
    handle = None
    if model.spatial_ei is not None:
        handle = model.spatial_ei.register_forward_pre_hook(lambda module, args: captured.append(args))
    try:
        output = model(x, observed_counts=history)
    finally:
        if handle is not None:
            handle.remove()
    target = output.logits[0, -1, 0]
    full, = torch.autograd.grad(target, x, retain_graph=True)
    if model.spatial_ei is not None:
        u_e, u_i = captured[0]
        adj_e, adj_i = torch.autograd.grad(target, (u_e, u_i), retain_graph=True)
        components = []
        for k in range(2):
            mask = u_e.new_zeros(2); mask[k] = 1
            part, = torch.autograd.grad((u_e, u_i), x,
                                       grad_outputs=(adj_e*mask, adj_i*mask), retain_graph=True)
            components.append(part)
    else:
        adjoint, = torch.autograd.grad(target, output.total_current, retain_graph=True)
        currents = model.spatial_components(x).currents.sum(-1)
        reconstructed = currents.sum(-1)
        torch.testing.assert_close(reconstructed, output.total_current, rtol=1e-10, atol=1e-11)
        components = [torch.autograd.grad(currents[..., k], x, grad_outputs=adjoint, retain_graph=True)[0]
                      for k in range(2)]
    modes = torch.stack(components)
    error = modes.sum(0)-full
    torch.testing.assert_close(modes.sum(0), full, rtol=1e-8, atol=1e-10)
    quality = {'sum_max_abs_error': float(error.abs().max()),
               'sum_relative_l2_error': float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(full))}
    return full.detach().numpy()[0], modes.detach().numpy()[:, 0], quality


def mode_shares(jacobian: np.ndarray, modes: np.ndarray) -> dict:
    mode_energy = np.square(modes).sum(axis=(1, 2))
    total_energy = np.square(jacobian).sum()
    if total_energy <= 0 or mode_energy.sum() <= 0:
        raise ValueError('undefined zero RF/mode energy')
    q = mode_energy/mode_energy.sum()
    signed = (modes*jacobian[None]).sum(axis=(1,2))/total_energy
    cross = 2*np.sum(modes[0]*modes[1])/total_energy
    return {'q': q, 'signed_projection': signed, 'cross_over_total_energy': float(cross),
            'mode_energy': mode_energy, 'total_energy': float(total_energy)}


def paired_bootstrap(values: np.ndarray, bootstrap_indices: np.ndarray) -> dict:
    means = values[bootstrap_indices].mean(-1)
    lo, hi = np.quantile(means, [.025,.975])
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'wins': int((values<0).sum()), 'losses': int((values>0).sum()),
            'ties': int((values==0).sum()), 'ci_low': float(lo), 'ci_high': float(hi)}
