from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
AUTHOR_SHA = "7b7750d167ed965e068770874f42895bc61dc415"
AUTHOR = WORK / f"open-retina-{AUTHOR_SHA}"
os.environ.setdefault("OPENRETINA_CACHE_DIRECTORY", str(WORK / "author_cache"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path[:0] = [str(AUTHOR), str(ROOT)]

from omegaconf import OmegaConf
from openretina.models.core_readout import BaseCoreReadout
from openretina.modules.core.base_core import SimpleCoreWrapper
from openretina.modules.readout.multi_readout import MultiSampledGaussianReadout
from models.mechanistic_retina.state import decay_from_tau, fixed_one_bin_history_state


def author_configuration() -> dict:
    base = OmegaConf.load(AUTHOR / "configs/model/core_gaussian_readout.yaml")
    outer = OmegaConf.load(AUTHOR / "configs/vystrcilova_2024_nm_cnn.yaml")
    merged = OmegaConf.merge(base, outer.model)
    return OmegaConf.to_container(merged, resolve=False)


def deterministic(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)


def identity_drive(value: torch.Tensor) -> torch.Tensor:
    return value


class OpenRetinaAdapted(BaseCoreReadout):
    """Official marmoset core/readout with causal padding and Bernoulli/history interfaces."""

    def __init__(self, cells: tuple[str, ...], mean_occupancy: dict[str, float], seed: int) -> None:
        deterministic(seed)
        cfg = author_configuration()
        core_args = {k: v for k, v in cfg["core"].items() if not k.startswith("_") and k != "channels"}
        core_args["channels"] = (1, *cfg["hidden_channels"])
        core = SimpleCoreWrapper(**core_args)
        # Preserve every official kernel; spatial zero padding accommodates the fixed 17x17 field.
        for layer, kernel in zip(core.features, cfg["core"]["spatial_kernel_sizes"], strict=True):
            layer.conv.padding = (0, kernel // 2, kernel // 2)
        readout_args = {k: v for k, v in cfg["readout"].items() if not k.startswith("_") and k != "in_shape"}
        readout = MultiSampledGaussianReadout(
            in_shape=(64, 150, 17, 17), n_neurons_dict={self.key(c): 1 for c in cells},
            nonlinearity_function=identity_drive, **readout_args,
        )
        super().__init__(core, readout, learning_rate=cfg["learning_rate"],
                         loss=nn.BCEWithLogitsLoss(reduction="sum"))
        self.cells = cells
        self.history_raw = nn.ParameterDict({self.key(c): nn.Parameter(torch.tensor(math.log(math.expm1(0.02))))
                                            for c in cells})
        self.register_buffer("history_decay", torch.tensor(decay_from_tau(1000 / 150, 30.0)))
        with torch.no_grad():
            for cell in cells:
                p = mean_occupancy[cell]
                if not 0 < p < 1:
                    raise ValueError(f"Invalid training occupancy for {cell}: {p}")
                self.readout[self.key(cell)].bias.fill_(math.log(p / (1 - p)))

    @staticmethod
    def key(cell: str) -> str:
        return cell.replace("#", "_")

    def place(self, device: str = "cuda") -> OpenRetinaAdapted:
        self.core.to(device)
        # CPU grid_sample has deterministic backward; the official readout implementation is unchanged.
        self.readout.cpu()
        self.history_raw.cpu()
        return self

    def core_features(self, cones: torch.Tensor) -> torch.Tensor:
        if cones.ndim != 3 or cones.shape[-1] != 289:
            raise ValueError("Input must be [batch,time,289] without spatial resampling")
        x = cones.reshape(cones.shape[0], cones.shape[1], 17, 17).unsqueeze(1)
        x = F.pad(x, (0, 0, 0, 0, 30, 0))
        return self.core(x.to(next(self.core.parameters()).device)).cpu()

    def stimulus_drive(self, features: torch.Tensor, cell: str) -> torch.Tensor:
        return self.readout(features, data_key=self.key(cell))

    def history_feature(self, events: torch.Tensor) -> torch.Tensor:
        return fixed_one_bin_history_state(events.cpu(), float(self.history_decay))

    def history_coefficient(self, cell: str) -> torch.Tensor:
        return F.softplus(self.history_raw[self.key(cell)])

    def logits_from_drive(self, drive: torch.Tensor, history: torch.Tensor, cell: str) -> torch.Tensor:
        return drive - self.history_coefficient(cell) * history

    def forward(self, cones: torch.Tensor, events: torch.Tensor, cell: str) -> torch.Tensor:
        return self.logits_from_drive(self.stimulus_drive(self.core_features(cones), cell),
                                      self.history_feature(events), cell)

    def regularization(self, cells: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.regularizer().cpu(), sum(self.readout.regularizer(self.key(c)) for c in cells)

    def project_readout(self) -> None:
        with torch.no_grad():
            for cell in self.cells:
                self.readout[self.key(cell)].mu.clamp_(-1, 1)

    def cpu_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
