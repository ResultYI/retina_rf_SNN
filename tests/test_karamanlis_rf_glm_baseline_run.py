from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from baselines.local_point_process_glm import LocalPointProcessGLM
import evaluation.mechanistic_retina.karamanlis_baseline_run as baseline_run
from evaluation.mechanistic_retina.karamanlis_baseline_run import (
    KaramanlisBaselineRunConfig,
)
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import (
    PopulationGLMTrainingRequest,
    PopulationGLMTrainingResult,
)


_SESSION = Path(
    "data/real/karamanlis_2024/sessions/20220301_252MEA_marmoset_left_n1"
)
_GRAPH = Path("output/real_data/karamanlis_2024_population_locality_graph_v1")
_RETINAL = Path(
    "output/real_data/karamanlis_2024_population_rf_geometry_cell_gains_seed20260302/model-best.pt"
)
_GLM = Path(
    "output/real_data/karamanlis_2024_population_rf_geometry_cell_gains_glm_baseline_v1"
)
_ARTIFACTS_AVAILABLE = all(
    path.exists() for path in (_SESSION, _GRAPH, _RETINAL, _GLM / "glm-trained.pt")
)


@pytest.mark.skipif(not _ARTIFACTS_AVAILABLE, reason="real GLM artifacts unavailable")
def test_rf_gain_baseline_runner_replays_reviewed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the real 60-cell RF/gain checkpoint and its train-only fitted GLM.
    saved_glm = torch.load(_GLM / "glm-trained.pt", map_location="cpu", weights_only=True)
    contract = saved_glm["contract"]

    def replay_fit(
        request: PopulationGLMTrainingRequest,
    ) -> PopulationGLMTrainingResult:
        assert request.graph_radius_deg is None
        assert request.support_mask is not None
        model = LocalPointProcessGLM(
            request.cone_positions,
            request.cell_positions,
            request.graph_radius_deg,
            request.temporal_lags,
            support_mask=request.support_mask,
        )
        model.load_state_dict(saved_glm["model"], strict=True)
        return PopulationGLMTrainingResult(
            model=model,
            gradients_finite=bool(contract["gradients_finite"]),
            actually_updated=tuple(contract["actually_updated"]),
            train_nll_initial=float(contract["train_nll_initial"]),
            train_nll_trained=float(contract["train_nll_trained"]),
            solver_iterations=int(contract["solver_iterations"]),
            converged=bool(contract["solver_converged"]),
        )

    monkeypatch.setattr(baseline_run, "fit_population_glm", replay_fit)

    # When: the current executable entrypoint rebuilds data, retinal, and report.
    result = baseline_run.run_karamanlis_prediction_baselines(
        KaramanlisBaselineRunConfig(
            session_dir=_SESSION,
            graph_dir=_GRAPH,
            checkpoint_path=_RETINAL,
            output_dir=tmp_path / "baseline",
        )
    )

    # Then: every reviewed metric and RF-lineage field is reproduced exactly.
    expected = json.loads((_GLM / "results.json").read_text(encoding="utf-8"))
    actual = json.loads(
        (result.artifact_dir / "results.json").read_text(encoding="utf-8")
    )
    assert actual == expected
    assert result.glm_nll == pytest.approx(0.14343951642513275, abs=0.0)
    assert actual["comparison_scope"]["matched_capacity"] is False
    assert actual["lineage"]["cell_count"] == 60
    assert actual["glm_contract"]["support_count_mean"] == pytest.approx(31.6)
