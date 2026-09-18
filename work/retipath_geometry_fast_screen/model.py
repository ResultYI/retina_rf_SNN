from __future__ import annotations

import torch
from torch import nn

from models.mechanistic_retina.bipolar_subunits import PathFeatureBank
from models.mechanistic_retina.retipath import RetiPath
from models.mechanistic_retina.retipath_spatial_ei import spatial_ei_parameters
from models.mechanistic_retina.support_partition import partition_spatial_basis
from retipath_spatial_ei_pilot import model_args


SIGMAS = {"midget": (0.05, 0.14), "parasol": (0.09, 0.20)}
SCALE_FRACTION = 0.10


class BoundedFeatureBank(PathFeatureBank):
    def __init__(self, original: PathFeatureBank, cp: dict, learn_scales: bool):
        super().__init__(*model_args(cp))
        self.load_state_dict(original.state_dict(), strict=True)
        positions = cp["cone_positions_degs"].float()
        center = cp["cell_positions_degs"].float()
        assert center.shape == (1, 2)
        spacings = []
        for axis in range(2):
            grid = positions[:, axis].unique(sorted=True)
            assert len(grid) == 17
            step = (grid[-1] - grid[0]) / 16
            torch.testing.assert_close(grid.diff(), step.expand(16), atol=1e-7, rtol=1e-5)
            spacings.append(step)
        self.register_buffer("grid_spacing_deg", torch.stack(spacings))
        self.register_buffer("reference_center_deg", center)
        self.register_buffer("relative_positions_deg", positions - center)
        self.register_buffer("original_sigmas_deg", positions.new_tensor(SIGMAS[cp["cell_types"][0]]))
        assert self.original_sigmas_deg[0] * 1.1 < self.original_sigmas_deg[1] * 0.9
        self.raw_center = nn.Parameter(torch.zeros(2))
        if learn_scales:
            self.raw_scale = nn.Parameter(torch.zeros(2))
        else:
            self.register_buffer("raw_scale", torch.zeros(2))

    @property
    def center_delta_deg(self):
        return self.grid_spacing_deg * self.raw_center.tanh()

    @property
    def scale_factors(self):
        return 1 + SCALE_FRACTION * self.raw_scale.tanh()

    def geometry_basis(self):
        # Gaussian likelihood ratio preserves the saved basis exactly at zero.
        # Masked normalization below preserves the original disk edge topology.
        relative = self.relative_positions_deg
        delta = self.center_delta_deg
        distance_change = -2 * (relative * delta).sum(-1) + delta.square().sum()
        inverse_scale2 = self.scale_factors.square().reciprocal()
        exponent = -0.5 * (
            relative.square().sum(-1)[None] * (inverse_scale2[:, None] - 1)
            + distance_change[None] * inverse_scale2[:, None]
        ) / self.original_sigmas_deg[:, None].square()
        return partition_spatial_basis(self.spatial_basis * exponent.exp()[None], self.supports)

    def basis_kernels(self):
        return torch.einsum("n,prl,npsc->npsrlc", self.polarity_sign,
                            self.temporal_basis.repeat(2, 1, 1), self.geometry_basis())


def build(cp: dict, condition: str):
    assert condition in ("A", "B", "C")
    assert cp["backend"] == "spatial_conductance" and cp["phase"] == "refit"
    assert cp["step"] == cp["best_step"]
    torch.manual_seed(cp["seed"])
    model = RetiPath(*model_args(cp), rms_e=torch.tensor(cp["rms"]["e"]),
                     rms_i=torch.tensor(cp["rms"]["i"]))
    model.load_state_dict(cp["model"], strict=True)
    if condition != "A":
        model.feature_bank = BoundedFeatureBank(model.feature_bank, cp, condition == "C")
    params = list(spatial_ei_parameters(model))
    if condition != "A":
        params.append(model.feature_bank.raw_center)
    if condition == "C":
        params.append(model.feature_bank.raw_scale)
    assert len({id(p) for p in params}) == len(params)
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    assert sum(p.numel() for p in params) == {"A": 37, "B": 39, "C": 41}[condition]
    return model, params
