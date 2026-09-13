from __future__ import annotations

import ast
import csv
from pathlib import Path
import subprocess

import numpy as np
import torch

from retipath_temporal_common import (
    ROOT, OUT, PHASE2, SEEDS, CONDITIONS, ORIGINAL_CONTEXT, load_json, sha, tensor_sha,
    utc, write_once, block_counts, make_split, select_observations,
)
from retipath_phase2_analyze import model_from_checkpoint
from retipath_spatial_ei_pilot import context_values
from data.schottdorf_lee_catalog import public_recordings, RecordingKind
from data.schottdorf_lee_2021 import SchottdorfAdapterConfig, _load_calibrated_lm_drive


def freeze() -> None:
    torch.set_num_threads(1)
    assert not (OUT/'confirmatory_lock.json').exists() and not (OUT/'TEST_CONSUMED.json').exists()
    scan = load_json(OUT/'provenance_scan.json'); supplemental = load_json(OUT/'provenance_supplement.json')
    assert not supplemental['errors']
    roots = {str(Path(ast.literal_eval(e[e.index("'"):]))) for e in scan['errors']}
    assert roots == set(supplemental['supplemental_roots'])
    assert not scan['post240_source_objects'] and not supplemental['post240_source_objects']
    assert not scan['candidate_mentions'] and not supplemental['candidate_mentions']
    consumed = [[0, 16], [16, 20], [20, 60]]+[r['live_range_s'] for r in scan['consumption_markers']]
    start = next(s for s in range(240, 600, 60) if not any(a < s+60 and b > s for a,b in consumed))
    stop = start+60
    assert [start, stop] == [240, 300]
    phase = load_json(PHASE2/'checkpoints/protocol.json')
    old = load_json(ROOT/'output/experiments/local_bc_subunit_nonlinearity_population_20260907/confirmatory_range_lock.json')
    original_records = {r['biological_cell']: r for r in old['recordings']}
    catalog = {r.cell_id: r for r in public_recordings(ROOT/'data/real/schottdorf_lee_2021_repository/data')
               if r.recording_kind is RecordingKind.TEN_MINUTE and r.cell_id in phase['cells']}
    assert set(catalog) == set(original_records) and len(catalog) == 17
    frozen = {}
    for path,digest in phase['source_hashes'].items():
        assert sha(ROOT/path) == digest, path
        frozen[path] = digest
    checkpoint_refs = {}
    for cell in phase['cells']:
        refs = {}
        for seed in SEEDS:
            for condition in CONDITIONS:
                folder = PHASE2/'checkpoints'/cell.replace('#','_')/str(seed)/condition
                for stage in ('inner','refit'):
                    path = folder/f'{stage}.pt'; digest = sha(path)
                    key = str(path.relative_to(ROOT)); frozen[key] = digest
                    refs[f'{seed}/{condition}/{stage}'] = {'path': key, 'sha256': digest}
                inner = torch.load(folder/'inner.pt', weights_only=True)
                refit = torch.load(folder/'refit.pt', weights_only=True)
                assert refit['step'] == refit['best_step'] == inner['best_step']
                assert refit['protocol_sha256'] == phase['protocol_sha256']
                assert refit['condition'] == condition and refit['seed'] == seed and refit['cell_id'] == cell
                refs[f'{seed}/{condition}/refit']['selected_updates'] = refit['step']
        checkpoint_refs[cell] = refs
    movie = ROOT/old['movie']['path']
    assert sha(movie) == old['movie']['sha256']
    frozen[str(movie.relative_to(ROOT))] = old['movie']['sha256']
    thresholds_path = ROOT/'output/experiments/context_dependent_rf_gain_20260907/context_thresholds.csv'
    thresholds = {r['cell_id']: r for r in csv.DictReader(thresholds_path.open())}
    cells = []
    for cell in phase['cells']:
        if cell not in catalog:
            continue
        recording = catalog[cell]; prior = original_records[cell]
        assert prior['duration_s'] >= stop and prior['header_kind_verified']
        assert sha(recording.path) == prior['sha256']
        with recording.path.open(encoding='utf-8') as stream:
            header = []
            for line in stream:
                header.append(line)
                if line.rstrip('\r\n') == 'No\tTime':
                    break
            else:
                raise ValueError('DATA MISMATCH: single-trial metadata missing')
        assert any(line.startswith('Video Start\t') for line in header)
        frozen[str(recording.path.relative_to(ROOT))] = prior['sha256']
        cells.append({'cell_id': cell, 'group': phase['cells'][cell]['group'], 'recording_id': recording.recording_id,
                      'recording_path': str(recording.path.relative_to(ROOT)), 'recording_sha256': prior['sha256'],
                      'duration_s': prior['duration_s'],
                      'LOW_threshold': float(thresholds[cell]['LOW_threshold']),
                      'HIGH_threshold': float(thresholds[cell]['HIGH_threshold'])})
    print('Provenance and all 396 Phase 2 checkpoint identities verified; preparing stimulus-only sampling.', flush=True)
    drive, positions = _load_calibrated_lm_drive(movie, stop*150, SchottdorfAdapterConfig())
    cones = drive[start*150:stop*150].reshape(60,150,289).copy()
    sampling = {'cones': cones, 'positions': positions}
    replay = []
    for info in cells:
        cell = info['cell_id']; safe = cell.replace('#','_')
        ref = phase['cells'][cell]
        assert sha(Path(ref['input_path'])) == ref['input_sha256']
        archive = torch.load(ref['input_path'], weights_only=True, mmap=True)['train']
        matched = [i for i,s in enumerate(archive['source_image_ids']) if s.startswith(info['recording_id']+'-')]
        assert len(matched) == 16
        np.testing.assert_array_equal(archive['cone_drive'][matched].numpy(), drive[:2400].reshape(16,150,289))
        with (ROOT/info['recording_path']).open(encoding='utf-8') as stream:
            old_counts, _ = block_counts(stream, 0, 16)
        np.testing.assert_array_equal(old_counts.reshape(16,150,1), archive['spike_counts'][matched].numpy())
        model, cp = model_from_checkpoint(ROOT/checkpoint_refs[cell][f'{SEEDS[0]}/A/refit']['path'])
        np.testing.assert_array_equal(cp['cone_positions_degs'].numpy(), positions)
        support = model.feature_bank.bc_support[0] > 0
        assert tensor_sha(support) == thresholds[cell]['support_sha256']
        split = make_split(cones, np.zeros(9000,dtype=np.int64), info['recording_id'], start)
        context = context_values(split, support)
        info['source_ids'] = list(split.source_image_ids)
        info['context_counts'] = {}; info['observation_counts'] = {}
        sampling[safe+'_context'] = context
        for label, mask in (('LOW', context <= info['LOW_threshold']), ('HIGH', context >= info['HIGH_threshold'])):
            assert not mask[:, :45].any()
            selected = select_observations(mask, split.source_image_ids)
            sampling[safe+'_'+label+'_indices'] = selected
            info['context_counts'][label] = int(mask.sum())
            info['observation_counts'][label] = len(selected)
        replay.append({'cell_id': cell, 'old_consumed_range_s': [0,16], 'cone_max_abs_error': 0., 'spike_count_exact': True})
    with (OUT/'sampling_before_targets.npz').open('xb') as stream:
        np.savez_compressed(stream, **sampling)
    evidence = [OUT/'provenance_scan.json', OUT/'provenance_supplement.json', OUT/'sampling_before_targets.npz',
                thresholds_path, ORIGINAL_CONTEXT,
                PHASE2/'PROTOCOL.md', PHASE2/'checkpoints/protocol.json', PHASE2/'checkpoints/selection_lock.json',
                PHASE2/'checkpoints/analysis_complete.json', PHASE2/'checkpoints/verification.json',
                PHASE2/'prediction_population.csv', PHASE2/'rf_population.csv',
                ROOT/'data/schottdorf_lee_catalog.py', ROOT/'data/schottdorf_lee_2021.py',
                ROOT/'data/schottdorf_lee_multirecording.py', ROOT/'data/schottdorf_lee_spikes.py',
                ROOT/'work/retipath_temporal_scan.py', ROOT/'work/retipath_temporal_common.py', Path(__file__),
                ROOT/'work/retipath_temporal_evaluate.py', ROOT/'work/retipath_phase2_analyze.py',
                ROOT/'work/retipath_spatial_ei_report.py', ROOT/'work/retipath_spatial_ei_pilot.py',
                ROOT/'tests/test_retipath_temporal_confirmation.py']
    frozen.update({str(p.relative_to(ROOT)): sha(p) for p in evidence})
    protocol = '''# RetiPath spatial E/I frozen temporal confirmation

## Pre-target freeze and provenance
The earliest locally unconsumed complete 60 s interval beginning at 240 s is [240,300). Prior training/development and confirmation consumption covers [0,240). The initial accessible-repository scan and a narrowly scoped, permission-approved supplemental read cover all initial directory gaps. Raw targets, binary tensors and git history are excluded from provenance search. This is absence of local target-based analysis/scoring consumption evidence, not proof about undocumented external activity. Earlier legacy code parsed complete recording payloads before retaining its selected interval; raw-file parsing alone is distinguished from use in fitting, model selection, target statistics or scoring. This confirmation does not claim that these source-file bytes have never been opened.

Freeze all 22 Phase 2 cells, three seeds (2026091301/2/3), and all 396 selected-inner/fresh-refit checkpoint files. Actual confirmation eligibility is determined only by recording coverage: 17 continuous-10-minute cells; 68#4,69#4,69#21,70#1,70#15 lack this interval and are excluded before targets. No replacement cells or block. Scores cannot establish generalization for the five unavailable cells.

## Frozen model and scoring contract
A=current RetiPath architecture; B=spatial E/I current integration; C=matched conductance integration. Use the Phase 2 full-[0,16) fresh-refit endpoint for each cell/seed/condition. All model code, checkpoint weights and buffers, alpha bounds, fixed RMS, K=2 basis, reversal potentials, membrane scales and history remain unchanged. No optimization, refit, checkpoint selection, new normalization, parameter projection or model edits.

Use the existing calibrated cone frontend: experimental150 Hz, decoded frame=live frame+751, crop51/pool3, 289 cone coordinates and blank-frame calibration. Container fps is not the experimental rate. Live frames[36000,45000), decoded[36751,45751). Sixty independent150-bin sequences; first30 bins warmup, remaining120 scored,7200 Bernoulli occupancy bins/cell. Reset all states by their original forward contract (B/C voltage V0). Observed strictly-past occupancy history is fixed within each sequence; no sequence or block boundary state carry.

Targets are accessed only after durable confirmatory_lock.json and TEST_CONSUMED.json writes. The new reader streams chronological single-trial spike times, retains only this block, and stops at the end-boundary guard; a boundary sentinel is used only to stop parsing, never scored/stored as a later target. Tick resolution0.1ms; preserve legacy global-bin floor arithmetic, then slice the locked range. Reader and frontend were checked against already-consumed [0,16) before locking. The block is retired before first access, including failed attempts; recovery may only reuse this block.

## Prediction primary and secondary
Primary C-A Bernoulli NLL; secondary C-B and B-A. Report cell/seed values and per-seed equal-cell mean,median,wins/losses/ties and paired-cell percentile95% bootstrap CI (10000 draws, frozen bootstrap seed2026091399). Seed aggregate averages the three differences within each cell before bootstrapping17 cells; never treat51 cell-seeds as independent cells. Reuse Phase2's directional replication rule: aggregate CI upper bound<0 and all three seed means<0; report each seed's CI separately. No outcome-dependent changes or extra seeds.

## RF secondary, unchanged Phase2 definitions
Use train-derived LOW q20 and HIGH q80 thresholds from the frozen threshold CSV, based on mean spatial RMS contrast over strictly previous45 bins within the fixed BC support; reset independently per sequence. Apply the original chronological sampler (at most100 points) then Phase2 floor(linspace) thinning to at most20 observations/context. Context indices and stimulus are saved and hashed before new targets; every seed and A/B/C share them. No context search, threshold change, substitute observation or RF effect cutoff. Missing context pair means RF UNVERIFIED, with prediction retained.

Use float64 full causal-prefix logit Jacobian with observed history fixed, no16-lag truncation or state detach. Reuse Phase2 RF evaluator unchanged. Gain=||J||2; per-observation P(x)=sum_lag J²/sum_lag,x J²; context profiles are observation means. Save HIGH/LOW gain, normalized profiles, centroids, second-moment radii around own centroids, TV, centered radial CDF and HIGH-LOW delta P. Same radial grid0..1.5degree/0.005degree and same per-observation centering. Preserve both mean-profile radius and median-of-observation radius conventions. Use the existing pairwise-seed delta-P cosine descriptively; size direction uses the signs of existing HIGH-LOW radius differences and compares the same available Phase2 cells. No new RF metric, RF loss or effect threshold. Profile dynamics and prediction are separate; context association is not causal adaptation.

## Deliver and stop
Save protocol, confirmatory lock, local provenance closure, consumption marker, necessary sampling/target/output arrays, prediction and RF CSVs, per-cell delta-P maps, verification and a Chinese report answering the five requested questions. No new training checkpoints. Recommendation does not automatically change the default model. Consume one block only, then stop.
'''
    with (OUT/'PROTOCOL.md').open('x', encoding='utf-8') as stream:
        stream.write(protocol)
    frozen[str((OUT/'PROTOCOL.md').relative_to(ROOT))] = sha(OUT/'PROTOCOL.md')
    closure = {'status':'NO_LOCAL_TARGET_BASED_CONSUMPTION_EVIDENCE', 'scanned_objects': len(scan['objects'])+len(supplemental['objects']),
               'initial_directory_gaps_closed': len(roots), 'unresolved_errors': [], 'known_consumed_ranges_s': consumed,
               'candidate_range_s':[start,stop], 'new_target_reads':0}
    write_once(OUT/'provenance_closure.json', closure)
    frozen[str((OUT/'provenance_closure.json').relative_to(ROOT))] = sha(OUT/'provenance_closure.json')
    lock = {'status':'FROZEN_BEFORE_NEW_TARGETS', 'frozen_utc':utc(),
            'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
            'HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'live_range_s':[start,stop], 'live_frames_half_open':[start*150,stop*150],
            'decoded_frames_half_open':[start*150+751,stop*150+751],
            'phase2_cohort':list(phase['cells']), 'cells':cells,
            'excluded_no_coverage':sorted(set(phase['cells'])-set(catalog)), 'seeds':list(SEEDS),
            'all_phase2_checkpoint_references':checkpoint_refs, 'scoring_checkpoint_stage':'refit',
            'bootstrap_seed':phase['bootstrap_seed'], 'bootstrap_samples':10000,
            'primary':'C-A', 'secondary':['C-B','B-A'], 'radial_grid_deg':phase['radial_grid_deg'],
            'frozen_sha256':frozen, 'pre_target_checks':{'synthetic_tests_passed':2,'consumed_input_replay':replay},
            'new_target_reads':0,'training_updates':0,'refits':0}
    write_once(OUT/'confirmatory_lock.json', lock)
    print('FROZEN BEFORE NEW TARGETS', [start,stop],len(cells),'cells',sha(OUT/'confirmatory_lock.json'),flush=True)


if __name__ == '__main__':
    freeze()
