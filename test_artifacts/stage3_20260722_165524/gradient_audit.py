from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.reconstruction import fit_augmented_reconstruction_scale
from loss.retina import RetinaLosses, RetinaObjective
from scripts.run_experiment import _build_network
from training.checkpointing import load_checkpoint
from training.config import load_config
from training.data import AugmentedClip, augment_clip, prepare_data
from training.trainer import RetinaTrainer


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def main() -> int:
    config = load_config(ARTIFACT_DIR / "budget_smoke_config.yaml")
    device = torch.device("cuda")
    prepared = prepare_data(config.data)
    model, decoder = _build_network(config, prepared, device)
    trainer = RetinaTrainer(
        model,
        decoder,
        RetinaObjective(
            rho_energy=config.objective.rho_energy,
            variance_floor=config.objective.variance_floor,
            phenotype_temperature=config.objective.phenotype_temperature,
            homeostasis_rate_min=config.objective.homeostasis_rate_min,
        ),
        config,
        fit_augmented_reconstruction_scale(
            prepared.train,
            config.data,
            seed=config.seed + 3,
        ),
    )
    sampling_generator = torch.Generator().manual_seed(config.seed + 1)
    augmentation_generator = torch.Generator().manual_seed(config.seed + 2)
    trainer.restore(
        load_checkpoint(ARTIFACT_DIR / "run" / "checkpoint_last.pt", device),
        sampling_generator,
        augmentation_generator,
    )

    audit_seed = config.seed + 20_000
    augmented = augment_clip(
        prepared.train[0],
        config.data,
        torch.Generator().manual_seed(audit_seed),
    )
    clip = AugmentedClip(
        noisy_input=augmented.noisy_input.unsqueeze(0).to(device),
        clean_target=augmented.clean_target.unsqueeze(0).to(device),
        metadata=augmented.metadata,
    )
    model_parameters = tuple(model.parameters())
    decoder_parameters = tuple(decoder.parameters())
    groups = {
        "all": (*model_parameters, *decoder_parameters),
        "rgc_phenotype": tuple(model.rgc.parameters()),
        "temporal": tuple(
            parameter
            for name, parameter in model.named_parameters()
            if any(token in name for token in ("tau", "gain", "mix"))
        ),
        "decoder": decoder_parameters,
    }
    selectors: dict[str, Callable[[RetinaLosses], torch.Tensor]] = {
        "reconstruction": lambda losses: losses.normalized_reconstruction,
        "weighted_wiring": lambda losses: config.objective.wiring_weight
        * losses.wiring,
        "weighted_diversity": lambda losses: config.objective.diversity_weight
        * (losses.variance_floor + losses.phenotype_repulsion + losses.homeostasis),
        "energy": lambda losses: losses.energy_penalty,
    }
    group_norms: dict[str, dict[str, float]] = {name: {} for name in groups}
    component_values: dict[str, float] = {}
    energy_violation = 0.0
    for component, selector in selectors.items():
        trainer.optimizer.zero_grad(set_to_none=True)
        losses, _, _ = trainer.forward_clip(
            clip.noisy_input,
            clip.clean_target,
            checkpointed=False,
        )
        value = selector(losses)
        value.backward()
        component_values[component] = float(value.detach())
        if component == "energy":
            energy_violation = float(losses.energy_violation.detach())
        for group, parameters in groups.items():
            group_norms[group][component] = _gradient_norm(parameters)
    trainer.optimizer.zero_grad(set_to_none=True)

    all_norms = group_norms["all"]
    reconstruction_norm = all_norms["reconstruction"]
    denominator = max(reconstruction_norm, torch.finfo(torch.float32).eps)
    result = {
        "source_id": prepared.train[0].source_id,
        "augmentation_seed": audit_seed,
        "optimizer_step": trainer.optimizer_step,
        "component_values": component_values,
        "component_group_grad_norms": group_norms,
        "reconstruction_grad_norm": reconstruction_norm,
        "weighted_wiring_grad_norm": all_norms["weighted_wiring"],
        "weighted_diversity_grad_norm": all_norms["weighted_diversity"],
        "energy_grad_norm": all_norms["energy"],
        "wiring_to_reconstruction_ratio": all_norms["weighted_wiring"]
        / denominator,
        "diversity_to_reconstruction_ratio": all_norms["weighted_diversity"]
        / denominator,
        "energy_to_reconstruction_ratio": all_norms["energy"] / denominator,
        "energy_violation": energy_violation,
    }
    result["finite"] = all(
        math.isfinite(value)
        for norms in group_norms.values()
        for value in norms.values()
    ) and all(math.isfinite(value) for value in component_values.values())
    (ARTIFACT_DIR / "gradient_audit.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
