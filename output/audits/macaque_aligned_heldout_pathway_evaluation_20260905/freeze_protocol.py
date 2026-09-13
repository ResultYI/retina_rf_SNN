# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(name: str, alternate: str | None = None) -> list[dict[str, str]]:
    paths = (OUT / name, OUT / "scan_escalated" / name) if alternate is None else (OUT / name, OUT / alternate)
    rows = {}
    for path in paths:
        for row in read_csv(path):
            rows[tuple(sorted(row.items()))] = row
    return list(rows.values())


def main() -> None:
    assert not (OUT / "TEST_PROTOCOL.md").exists(), "Protocol is immutable"
    initial = json.loads((OUT / "evidence_manifest.initial.json").read_text())
    usage = json.loads((OUT / "production_usage.json").read_text())
    records = read_csv(OUT / "recording_trial_inventory.csv")
    config = merge("executed_adapter_configs.csv")
    binary = merge("binary_usage_default.csv", "binary_usage_escalated.csv")
    ids = merge("binary_source_ids_default.csv", "binary_source_ids_escalated.csv")
    hits = merge("data_usage_search_hits.csv")
    inventory = merge("repository_search_inventory.csv")
    assert not any(float(r["end_seconds_at_150Hz"]) > 20 for r in config)
    assert not any(int(r["stop_inclusive"]) >= 3000 for r in ids)
    source_pattern = re.compile(r"live-frames-(\d+)-(\d+)")
    assert not any(int(m.group(2)) >= 3000 for r in hits for m in source_pattern.finditer(r["evidence"]))
    searches = [json.loads((p / "search_summary.json").read_text()) for p in (OUT, OUT / "scan_escalated")]
    inaccessible = [[Path(r["path"]) for r in s["inaccessible"]] for s in searches]
    unresolved = [(str(a), str(b)) for a in inaccessible[0] for b in inaccessible[1]
                  if a == b or a in b.parents or b in a.parents]
    assert not unresolved
    assert usage["cell_count"] == 22 and usage["recording_count"] == 37 and usage["trial_count"] == 137
    rows = []

    def add(path: str, lineage: str, identity: str, bounds: str, category: str, evidence: str,
            stimulus: str = "False", spikes: str = "False", predictions: str = "False",
            metrics: str = "False", training: str = "False", selection: str = "False",
            architecture: str = "False", metadata: str = "False") -> None:
        rows.append(dict(artifact_path=path, lineage=lineage, recording_or_cell=identity,
            accessed_live_range=bounds, stimulus_accessed=stimulus, spikes_or_targets_accessed=spikes,
            model_predictions_computed=predictions, NLL_or_metric_computed=metrics,
            used_for_training=training, used_for_model_selection=selection,
            used_for_architecture_decision=architecture, only_hash_or_metadata_access=metadata,
            classification=category, evidence=evidence))

    for r in config:
        path = r["artifact_path"]
        fixture = "/test_" in path or path.startswith(".test-tmp")
        add(path, "mocked loader test" if fixture else str(Path(path).parent), "see artifact",
            f"[0,{r['end_seconds_at_150Hz']}) s; train={r['train_sequences']}; validation={r['validation_sequences']}",
            "CLEAN_FOR_CONFIRMATORY_EVALUATION" if fixture else "CONSUMED_FOR_DEVELOPMENT",
            f"Structured executed artifact field {r['field']}; actual adapter coverage. Mocked fixtures do not use the real later movie." if fixture
            else f"Structured saved run/result configuration {r['field']}; conservative development consumption for the full recorded prefix.",
            stimulus="mocked" if fixture else "True", spikes="mocked" if fixture else "True",
            predictions="True", metrics="True", training="see run stage", selection="see run stage", architecture="UNKNOWN")
    for r in binary:
        if int(r["source_ID_count"]) == 0:
            continue
        add(r["artifact_path"], str(Path(r["artifact_path"]).parent), "exact per-sequence IDs in binary_source_ids_default/escalated.csv",
            f"live frames [{r['minimum_live_frame']},{int(r['maximum_live_frame_inclusive']) + 1})",
            "CONSUMED_FOR_DEVELOPMENT", f"Existing binary payload source IDs; SHA256={r['sha256']}; field paths={r['fields']}",
            stimulus="True", spikes="True", predictions="False" if "/inputs/" in r["artifact_path"] else "True",
            metrics="False" if "/inputs/" in r["artifact_path"] else "True", training="see producer", selection="see producer", architecture="UNKNOWN")
    for r in read_csv(OUT / "model_artifact_inventory.csv"):
        add(r["path"], r["model"], r["cell_id"], "fit [0,16) s; previously inspected validation [16,20) s",
            "CONSUMED_FOR_DEVELOPMENT", "Checkpoint metadata checked; fresh full-train refit; original target/mask/order verified separately in saved_prediction_identity.json.",
            stimulus="True", spikes="True", predictions="True", metrics="True", training="True", selection="inner-development only", architecture="UNKNOWN")
    metadata_witnesses = (
        (".omo/evidence/schottdorf_lee_timing_contract_final_check.md", "full recording [0,60) or [0,600) s", "Full raw timestamps, header/repeat/epoch consistency and edge-bin source validation; no fitted-model outputs or likelihood evaluation.", "True"),
        (".omo/evidence/schottdorf_lee_frame_zero_resolution.md", "full movie container / onset metadata", "File/duration/onset provenance; frozen decoded frame 751 remains acquisition-provenance UNKNOWN.", "False"),
        (".local_archives/20260905-pre-audit-cleanup/recoverable-content.zip", "recovered scientific payloads; executed prefixes <=20 s", "All 1310 archive members inventoried; relevant text/JSON and saved binary source IDs read, not merely the cleanup index.", "False"),
        ("output/retina_rf_SNN_code_review_20260831_122458.zip", "source archive; no later-time execution evidence", "All 526 members inventoried and relevant source payloads scanned. Source capability is not proof of execution.", "False"),
    )
    for path, bounds, evidence, targets in metadata_witnesses:
        add(path, "source / archive validation", "all applicable recordings", bounds, "METADATA_ONLY", evidence,
            stimulus="True", spikes=targets, metadata="True")
    for r in records:
        add(r["path"], "raw recording identity", r["recording_id"] + " / " + r["cell_id"],
            f"header/timestamps for {r['trial_count']} trials of {r['live_duration_s_per_trial']} s",
            "METADATA_ONLY", "Current audit parsed original recording headers and trials only; no new target statistics, model inference or metric. Original parser's catalog override/duplicate handling preserved.",
            spikes="True", metadata="True")
        add(r["path"], "candidate held-out", r["recording_id"] + " / " + r["cell_id"], "[20,60) s; live [3000,9000)",
            "CLEAN_FOR_CONFIRMATORY_EVALUATION", "No evidence of candidate target/prediction/metric use for development after text, run-config, saved source-ID, archive and temporary evidence review; complete common trial support verified.")
    write_csv("data_usage_inventory.csv", rows)
    gate = dict(status="CLEAN_FOR_CONFIRMATORY_EVALUATION", timestamp_utc=datetime.now(timezone.utc).isoformat(),
        candidate_live_seconds=[20, 60], candidate_live_frames=[3000, 9000], cells=22, recordings=37, trials=137,
        unique_text_paths=len({r['artifact_path'] for r in inventory}), unique_binary_paths=len({r['artifact_path'] for r in binary}),
        unique_source_ID_rows=len(ids), merged_config_rows=len(config), merged_range_hits=len(hits),
        post20_executed_configs=0, post20_saved_source_IDs=0, overlapping_inaccessible_directories=unresolved,
        scope="Operational no-evidence criterion inside current repository, recoverable archives and temporary artifacts; not proof about undocumented actions outside this evidence boundary.",
        new_model_outputs_before_protocol=0, source_validation_is_not_model_development=True)
    (OUT / "data_usage_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    report = f"""# Previously-unused data audit

Decision: **CLEAN_FOR_CONFIRMATORY_EVALUATION**, live **[20,60) s**, for all 22 cells, 37 recordings and 137 recording-local trials.

The current loader, 22 saved CNN input bundles, 88 saved validation target/mask/order bundles and 88 final checkpoint metadata records agree: training uses live [0,2400) frames ([0,16) s), development validation [2400,3000) ([16,20) s). Each 150-bin sequence masks its first 30 bins and scores 120. Exact recording/trial/segment identities are in production_source_ranges.csv; this conclusion does not come from defaults alone.

The two filesystem scans cover {gate['unique_text_paths']} unique text paths and {gate['unique_binary_paths']} relevant binary paths; the 1836 archive members were inspected as payloads. Structured executed configs stop at 20 s (short mocked tests at 0.8 s); {len(ids)} merged binary source-ID field records and text IDs contain no live frame >=3000. Binary metadata was loaded without running a model. Search artifacts preserve file hashes, line/field witnesses and excluded caches. Ordinary and approved read access had complementary inaccessible temporary directories; exact and ancestor overlaps are zero. Neither scan alone was treated as complete.

Historical factorized/LN/GLM, independent-seed, architecture/prediction/RF and reset/pre-roll runs inherit the recorded 0–20 s prefix or explicit 16–20 s validation IDs. Artificial illusion and temporal-probe stimuli do not consume later natural-movie targets. Original third-party notebooks and fitting source demonstrate capability, not execution in this project. All candidate-relevant execution evidence is bounded; no unresolved candidate-overlapping run was identified. Non-model timing checks parsed full timestamps and validated recording duration/repeated headers; those are METADATA_ONLY, not target-based model-performance inspection.

There are 20 repeated 6×1min recordings and 17 10min recordings. A repeated trial is a separate presentation of the same first-minute live movie, not a later successive minute. Raw format overrides the catalog for lSS01184; the frozen parser and its duplicate-payload handling are retained. recording_trial_inventory.csv gives every exact local trial count and recording identity. All contain [20,60); movie frame-count metadata permits decoded [3751,9751) frames. We retain decoded live-start 751 and the existing stimulus/spike timeline; acquisition frame-zero provenance remains UNKNOWN and is not repaired here.

Classification is scoped to the user's no-evidence definition and the repository/archives available now. It cannot establish that undocumented external human activity never occurred. Metadata access is reported explicitly, including raw timestamp parsing. No candidate prediction, NLL, center fitting or model selection was performed by this audit. The prior range-selection rule was recorded before any prediction, and the preferred interval passed without invoking a fallback.

Evidence: data_usage_inventory.csv; repository_search_inventory.csv and scan_escalated/; executed_adapter_configs.csv; archive_members.csv; data_usage_search_hits.csv; binary_usage_default/escalated.csv; binary_source_ids_default/escalated.csv; production_usage.json; production_source_ranges.csv; saved_prediction_identity.json; recording_trial_inventory.csv; model_artifact_inventory.csv; evidence_manifest.initial.json.
"""
    (OUT / "DATA_USAGE_REPORT.md").write_text(report, encoding="utf-8")
    cells = json.loads((ROOT / initial['lineages']['zero_center_reference'] / 'results.json').read_text())["cells"]
    model_rows = read_csv(OUT / "model_artifact_inventory.csv")
    lines = ["# Preregistered held-out natural-movie protocol", "", f"Frozen UTC: {gate['timestamp_utc']}",
        f"Branch: {initial['branch']}; HEAD: {initial['HEAD']}", "",
        "## Data gate and immutable range", "", "Eligibility passed before this protocol and before any new model output; see DATA_USAGE_REPORT.md and data_usage_gate.json. No result-based range selection. Live [20,60) seconds; live integer frames [3000,9000); decoded integer movie frames [3751,9751), zero-based half-open. Frozen live origin 751 retains its acquisition-provenance UNKNOWN. Movie: data/real/schottdorf_lee_2021_macaque/1x10_256.mpg; raw recordings are the exact paths below, with original hashes in evidence_manifest.initial.json.", "",
        "## Exact cells, recordings and local trials", "", "| Cell | Recording | Kind | One-based local trials |", "|---|---|---|---|"]
    lines += [f"| {r['cell_id']} | {r['recording_id']} | {r['recording_kind']} | {','.join(map(str, range(1,int(r['trial_count'])+1)))} |" for r in records]
    lines += ["", "Cell order: " + ", ".join(c["cell_id"] for c in cells), "",
        "Recording order is the frozen catalog order within each cell. Within each recording: segment 20 through 59, then local trial in ascending order for each segment; preserve recording-local IDs and cell-global trial offsets. Source ID: recording-live-frames-{150*segment:06d}-{150*segment+149:06d}-trial-{one_based_trial}. Expected 5480 sequences, 657600 valid response bins across the 137 recording-local trials. No cell/recording/trial exclusion.", "",
        "## Production evaluation contract", "", "Native 150 Hz; 150 bins/sequence; reset each 1 s sequence; score local 30..149 only. No cross-sequence pre-roll or state continuation. Original calibrated 17×17 central L+M Weber drive from the 51-pixel crop and 3-pixel pooling; original blank calibration, signs and coordinates. Same frozen parser, floor binning and Bernoulli event target q=1[count>0]. Full unscored warmup is supplied as input and history. Observed spike history is strictly past, implemented by the existing model; no current/future target leakage and no generated-spike replacement. All four models use identical events/masks/source/trial order. Construct the new split using existing loader semantics over the first 60 live seconds, selecting segments 20..59; this only extends data coverage, with no training invocation.", "",
        "NLL is the existing CPU Torch 2.6.0 expected_bernoulli_nll: sum_valid(softplus(z)-q*z)/number_valid, nats per valid bin. Float32 model and original tensor reduction; no probability clipping or new loss. Within a cell weight all valid recording/trial bins as production; population means weight each of 22 cells equally. Cells share a stimulus, and cell-bootstrap intervals are not new-animal or new-movie-sample generalization intervals.", "",
        "## Frozen models and preflight", "", "Aligned checkpoint root was resolved from the fixed-alignment analysis-manifest.json inputs_sha256, not inferred from a directory name. Both Canonical families must have public name Canonical V1, revision 4, h1-shared-bc-direct-broad-ac and bc-central-disk_ac-overlapping-full-disk, 33 trainable scalars and no trainable center. Aligned positions equal the frozen LN final center mapping x_deg=x_LN*(3*4.6/256), y_deg=-y_LN*(3*4.6/256); no refit. All final fresh full-train checkpoints and their initial hashes are fixed below.", "",
        "| Cell | Model | Frozen checkpoint |", "|---|---|---|"]
    lines += [f"| {r['cell_id']} | {r['model']} | {r['path']} |" for r in model_rows]
    lines += ["", "Before any held-out prediction/metric, replay original development validation [16,20) for all four families. Canonical zero/aligned: strict load, exact target/mask/source/trial order, bitwise logits equality and exact original NLL. LN: strict final-refit load and exact original NLL. CNN: strict final load, original GPU float32 forward (Torch 2.10.0+cu126, RTX4070 Laptop, deterministic algorithms, no AMP/TF32, CUBLAS_WORKSPACE_CONFIG=:4096:8), then original CPU Torch2.6.0 NLL on returned full-precision logits, compared exactly to existing results.json. Batch size 8 for all original evaluation functions and held-out conditions. Canonical/LN forward use original CPU Torch2.6.0 with 2 threads. Constructor RNG uses only existing checkpoint/training seed and is overwritten by strict frozen state; no new model seed. No checkpoint conversion, precision relaxation, backend-search rescue or altered pass tolerance. Any required mismatch or unavailable original runtime stops the entire experiment before held-out inference.", "",
        "## Frozen structural interventions", "", "Only aligned Canonical: normal (empty clamps); H1-off={PathwayClamp.H1}; direct-BC-off={DIRECT_BC_SUSTAINED,DIRECT_BC_TRANSIENT}; AC-off={AC_LOCAL,AC_TRANSIENT}. Use the production forward_sequence clamps argument. Direct BC branches are removed at direct pathway-current readout, preserving broad BC drive to AC. All bias/history parameters, remaining pathway gains and all state_dict tensors are unchanged; no refit or recalibration. No combined clamps beyond the two declared grouped branches. Every condition uses the same sequence/history/mask and batch order. Logit effect uses off-normal over the complete scored-bin vector (warmup excluded); store signed mean, mean absolute value, RMS and full vector L2 norm.", "",
        "## Estimands and statistics", "", "Primary paired differences per cell: aligned-zero, aligned-LN, aligned-CNN; negative means lower aligned NLL. Report all per-cell NLLs/differences, mean, median, negative/positive/exact-zero counts and paired-cell mean and median 95% percentile bootstrap intervals. Optional Constant is the explicitly requested analytical constant fit to identical scored test targets: p=their event fraction per cell; report its Bernoulli entropy as a test-fitted descriptive reference, not a frozen externally fitted predictor. No optimizer, checkpoint or tuning for Constant.", "",
        "Bootstrap: numpy default_rng(2026090501), 100000 draws of 22 complete biological-cell pairs with replacement in the frozen cell order; use the same index matrix for all paired model and pathway differences. Percentiles 2.5 and 97.5, NumPy default linear quantile method. This seed is solely for requested statistics, not a model seed. No p-values and no group significance tests.", "",
        "Secondary alignment: frozen radial_offset_degrees from the existing fixed-alignment audit per_cell_results.csv against I=zero-aligned; descriptive Pearson and Spearman (average ranks for ties). Single outlier diagnostic only: drop the cell with largest held-out absolute I (ties by lexical cell_id); recalculate mean I, Pearson and Spearman once. No other deletion search. Compare existing development aligned-zero mean difference, wins, Pearson and Spearman with the same four held-out quantities. Do not rerun RF/temporal/center estimation.", "",
        "Pathways: Delta NLL=off-normal; positive means the frozen removal worsens predictive likelihood. Per pathway report equal-cell mean/median, positive/negative/exact-zero counts, the paired-cell mean and median percentile intervals and MC ON/MC OFF/PC ON/PC OFF descriptive means. Per cell/pathway retain signed mean Delta logit, mean absolute Delta logit, RMS, full valid-vector norm and Delta NLL. Descriptive Pearson/Spearman of mean-absolute-logit effect versus Delta NLL, separately for each pathway, without thresholds or p-values.", "",
        "## Interpretation fixed before results", "", "Prediction: REPLICATED requires lower held-out aligned mean NLL, the mean-difference interval below zero (the conservative route to the user's CI-or-strong-cell-support criterion), positive mean improvement after the single declared deletion, and a clearly positive offset relation supported by Pearson and Spearman including that diagnostic. PARTIAL_REPLICATION applies if mean improvement remains but intervals/cell support or offset relation weaken markedly or the single cell dominates. FAILED_TO_REPLICATE applies if improvement disappears/systematic worsening occurs or the offset relation disappears. Preserve the user's qualitative 'clearly positive' criterion: report all coefficients and make a bounded descriptive judgment, without inventing a new correlation significance threshold or changing statistics after results.", "",
        "Pathways are judged independently: HELD-OUT PREDICTIVE CONSEQUENCE SUPPORTED when mean off-normal NLL is positive and its prespecified 95% interval lies above zero; WEAK / MIXED when positive effects are unresolved or heterogeneous; NO POSITIVE PREDICTIVE SUPPORT when no positive population degradation occurs or removal improves prediction. Always report sign heterogeneity. Clamp findings are predictive consequences of model-internal intervention under frozen remaining parameters/bias and depend on this model family; no biological necessity, unique causal contribution, retrained ablation or real-retina lesion claim. Baseline intervals containing zero are unresolved/not separated by the current interval; no equivalence or superiority claim.", "",
        "## Consumption and stop rules", "", "Immediately upon the first successful held-out target-based NLL, create TEST_CONSUMED.md before viewing/reporting its value: exact range/identities/models/clamps/metrics, branch/HEAD, this SHA256 and UTC time. If later execution fails, preserve partial consumption and stop; do not silently restart or select a new range. The full declared range is conservatively retired after first consumption. No consumption marker if a preflight gate fails before any held-out output. Final REPORT answers only the seven requested questions plus one research-decision sentence. New training, tuning, checkpoint selection, parameter/production edits, RF/tau/delay audits, new synthetic/illusion runs and additional experiments remain zero.", ""]
    protocol = OUT / "TEST_PROTOCOL.md"
    protocol.write_text("\n".join(lines), encoding="utf-8")
    (OUT / "protocol_hash.json").write_text(json.dumps(dict(path=protocol.relative_to(ROOT).as_posix(), sha256=sha(protocol),
        frozen_utc=datetime.now(timezone.utc).isoformat(), data_usage_gate_sha256=sha(OUT / "data_usage_gate.json"),
        data_usage_inventory_sha256=sha(OUT / "data_usage_inventory.csv"), new_model_outputs_before_freeze=0), indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    print("PROTOCOL FROZEN", sha(protocol))


if __name__ == "__main__":
    main()
