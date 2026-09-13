from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import itertools
from pathlib import Path

import numpy as np
import torch

from retipath_temporal_common import (
    ROOT, OUT, PHASE2, SEEDS, CONDITIONS, load_json, sha, tensor_sha, utc, write_once,
    block_counts, make_split, prediction_stats,
)
from retipath_phase2_common import evaluate
from retipath_phase2_analyze import model_from_checkpoint, rf_modes, csv_rows


def verify_frozen(lock: dict) -> None:
    for path,digest in lock['frozen_sha256'].items():
        assert sha(ROOT/path) == digest, path


def prepare_targets(lock: dict, lock_sha: str) -> None:
    marker = OUT/'TEST_CONSUMED.json'
    if marker.exists():
        previous = load_json(marker)
        assert previous['confirmatory_lock_sha256'] == lock_sha
        assert previous['live_range_s'] == lock['live_range_s']
    else:
        write_once(marker, {'status':'TEST_CONSUMED', 'target_access_start_utc':utc(),
                           'live_range_s':lock['live_range_s'], 'live_frames_half_open':lock['live_frames_half_open'],
                           'decoded_frames_half_open':lock['decoded_frames_half_open'],
                           'confirmatory_lock_sha256':lock_sha,
                           'sampling_sha256':sha(OUT/'sampling_before_targets.npz'),
                           'retirement_rule':'Retired before first new target parse, including failures; only this block may be resumed.',
                           'training_updates':0, 'refits':0, 'new_blocks_consumed':1})
    (OUT/'targets').mkdir(exist_ok=True)
    for info in lock['cells']:
        safe = info['cell_id'].replace('#','_'); path = OUT/'targets'/f'{safe}.npz'
        record = path.with_suffix('.json')
        if path.exists() and record.exists():
            assert sha(path) == load_json(record)['target_sha256']
            continue
        assert not path.exists() and not record.exists()
        started = utc()
        with (ROOT/info['recording_path']).open(encoding='utf-8') as stream:
            counts, provenance = block_counts(stream, *lock['live_range_s'])
        with path.open('xb') as stream:
            np.savez_compressed(stream, counts=counts)
        write_once(record, {'cell_id':info['cell_id'], 'recording_path':info['recording_path'],
                            'started_utc':started, 'completed_utc':utc(), 'confirmatory_lock_sha256':lock_sha,
                            'target_sha256':sha(path), **provenance})


def evaluate_job(cell: str, seed: int) -> dict:
    torch.set_num_threads(1)
    lock = load_json(OUT/'confirmatory_lock.json'); lock_sha = sha(OUT/'confirmatory_lock.json')
    info = next(r for r in lock['cells'] if r['cell_id'] == cell); safe = cell.replace('#','_')
    destination = OUT/'results'/f'{safe}_{seed}.json'
    if destination.exists():
        saved = load_json(destination)
        assert saved['confirmatory_lock_sha256'] == lock_sha
        return saved
    with np.load(OUT/'sampling_before_targets.npz',allow_pickle=False) as arrays:
        cones = arrays['cones']; positions = arrays['positions'].astype(np.float64)
        selections = {label:arrays[safe+'_'+label+'_indices'].tolist() for label in ('LOW','HIGH')}
    target_path = OUT/'targets'/f'{safe}.npz'
    assert sha(target_path) == load_json(target_path.with_suffix('.json'))['target_sha256']
    with np.load(target_path,allow_pickle=False) as arrays:
        counts = arrays['counts']
    split = make_split(cones, counts, info['recording_id'], lock['live_range_s'][0])
    assert list(split.source_image_ids) == info['source_ids'] and int(split.valid_mask.sum()) == 7200
    prediction = []; rf = []; checks = {}; outputs = {}
    for condition in CONDITIONS:
        ref = lock['all_phase2_checkpoint_references'][cell][f'{seed}/{condition}/refit']
        model, cp = model_from_checkpoint(ROOT/ref['path'])
        model.eval().requires_grad_(False)
        nll, logits = evaluate(model, split)
        assert all(torch.equal(v,cp['model'][k]) for k,v in model.state_dict().items())
        common = {'cell_id':cell, 'group':info['group'], 'seed':seed, 'condition':condition}
        prediction.append(common | {'confirmation_nll':nll, 'scored_bins':int(split.valid_mask.sum()),
                                     'selected_refit_updates':cp['step'], 'checkpoint_sha256':ref['sha256']})
        outputs[condition+'_logits'] = logits.numpy()
        if all(selections.values()):
            row, _, quality = rf_modes(model, split, selections, positions, np.array(lock['radial_grid_deg']))
            rf.append(common | row)
            checks[condition] = quality
        else:
            rf.append(common | {'status':'UNVERIFIED','reason':'missing frozen HIGH/LOW context pair',
                                 'LOW_n':len(selections['LOW']),'HIGH_n':len(selections['HIGH'])})
            checks[condition] = {'observations':0, 'reason':'missing context pair'}
        assert all(torch.equal(v.to(cp['model'][k]),cp['model'][k]) for k,v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
    with (OUT/'results'/f'{safe}_{seed}.npz').open('xb') as stream:
        np.savez_compressed(stream, **outputs)
    result = {'cell_id':cell, 'seed':seed, 'confirmatory_lock_sha256':lock_sha,
              'target_sha256':sha(target_path), 'prediction':prediction, 'RF':rf, 'checks':checks,
              'parameters_unchanged':True, 'training_updates':0}
    write_once(destination,result)
    print('EVALUATED',cell,seed,flush=True)
    return result


def rf_summary(rows: list[dict], cells: list[str]) -> dict:
    valid = {(r['cell_id'],r['seed'],r['condition']):r for r in rows if r['status']=='VERIFIED'}
    old = list(csv.DictReader((PHASE2/'rf_population.csv').open(encoding='utf-8')))
    prior = {(r['cell_id'],int(r['seed']),r['condition']):r for r in old if r['status']=='VERIFIED'}
    metrics = ('TV','centered_radial_CDF_L1_deg','centered_radial_CDF_max','centroid_shift_deg',
               'gain_log_HIGH_over_LOW','LOW_gain_median','HIGH_gain_median',
               'delta_radius_observation_median_deg','delta_radius_mean_profile_deg')
    output = {}
    for condition in CONDITIONS:
        per_seed = {}
        for seed in SEEDS:
            current = [valid[c,seed,condition] for c in cells if (c,seed,condition) in valid]
            per_seed[str(seed)] = {'n_cells':len(current),
                **{metric:{'mean':float(np.mean([r[metric] for r in current])),
                           'median':float(np.median([r[metric] for r in current]))} for metric in metrics},
                'radius_smaller_HIGH_cells':sum(r['delta_radius_observation_median_deg']<0 for r in current),
                'radius_larger_HIGH_cells':sum(r['delta_radius_observation_median_deg']>0 for r in current)}
        repeat = []; comparison = []
        for cell in cells:
            if not all((cell,s,condition) in valid for s in SEEDS):
                continue
            delta = [np.array(valid[cell,s,condition]['delta_P']) for s in SEEDS]
            cosine = []
            for i,j in itertools.combinations(range(3),2):
                norm = np.linalg.norm(delta[i])*np.linalg.norm(delta[j])
                cosine.append(float(delta[i]@delta[j]/norm) if norm>0 else None)
            radii = [valid[cell,s,condition]['delta_radius_observation_median_deg'] for s in SEEDS]
            repeat.append({'cell_id':cell, 'delta_P_pairwise_cosines':cosine,
                           'all_three_cosines_positive':all(v is not None and v>0 for v in cosine),
                           'radius_delta_by_seed':radii, 'HIGH_smaller_all_seeds':all(v<0 for v in radii),
                           'HIGH_larger_all_seeds':all(v>0 for v in radii)})
            if all((cell,s,condition) in prior for s in SEEDS):
                reference = [float(prior[cell,s,condition]['delta_radius_observation_median_deg']) for s in SEEDS]
                comparison.append({'cell_id':cell, 'phase2_radius_delta_mean':float(np.mean(reference)),
                                   'confirmation_radius_delta_mean':float(np.mean(radii)),
                                   'radius_sign_matches_phase2':bool(np.sign(np.mean(reference))==np.sign(np.mean(radii))),
                                   'phase2_TV_mean':float(np.mean([float(prior[cell,s,condition]['TV']) for s in SEEDS])),
                                   'confirmation_TV_mean':float(np.mean([valid[cell,s,condition]['TV'] for s in SEEDS])),
                                   'phase2_radial_CDF_L1_mean':float(np.mean([float(prior[cell,s,condition]['centered_radial_CDF_L1_deg']) for s in SEEDS])),
                                   'confirmation_radial_CDF_L1_mean':float(np.mean([valid[cell,s,condition]['centered_radial_CDF_L1_deg'] for s in SEEDS]))})
        output[condition] = {'per_seed':per_seed,'repeatability':repeat,'phase2_same_cell_comparison':comparison,
                              'valid_cells':len(repeat),'delta_P_all_pairs_positive_cells':sum(r['all_three_cosines_positive'] for r in repeat),
                              'median_pairwise_delta_P_cosine':float(np.median([v for r in repeat for v in r['delta_P_pairwise_cosines'] if v is not None])),
                              'HIGH_smaller_all_seeds_cells':sum(r['HIGH_smaller_all_seeds'] for r in repeat),
                              'HIGH_larger_all_seeds_cells':sum(r['HIGH_larger_all_seeds'] for r in repeat)}
    return output


def run() -> None:
    lock = load_json(OUT/'confirmatory_lock.json'); lock_sha = sha(OUT/'confirmatory_lock.json')
    assert lock['status']=='FROZEN_BEFORE_NEW_TARGETS' and lock['live_range_s']==[240,300]
    verify_frozen(lock)
    prepare_targets(lock, lock_sha)
    (OUT/'results').mkdir(exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)
    cells = [r['cell_id'] for r in lock['cells']]
    results = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        jobs = [pool.submit(evaluate_job,cell,seed) for cell in cells for seed in SEEDS]
        for future in as_completed(jobs):
            results.append(future.result())
    results.sort(key=lambda r:(r['cell_id'],r['seed']))
    prediction = [r for job in results for r in job['prediction']]
    rf = [r for job in results for r in job['RF']]
    assert len(prediction)==len(rf)==153
    comparisons, statistics = prediction_stats(prediction,cells,lock['bootstrap_seed'])
    rf_statistics = rf_summary(rf,cells)
    csv_rows(OUT/'prediction_per_cell_seed.csv',prediction)
    csv_rows(OUT/'prediction_population.csv',comparisons)
    csv_rows(OUT/'rf_per_cell_seed.csv',rf)
    verify_frozen(lock)
    assert sha(OUT/'confirmatory_lock.json') == lock_sha
    write_once(OUT/'analysis_complete.json',{'completed_utc':utc(),'confirmatory_lock_sha256':lock_sha,
               'prediction':statistics,'RF':rf_statistics,'scored_models':153,'scored_cells':17,
               'RF_observations':sum(c.get('observations',0) for r in results for c in r['checks'].values()),
               'new_blocks_consumed':1,'training_updates':0,'refits':0,'model_files_unchanged':True})
    print('FROZEN CONFIRMATION SCORING COMPLETE',flush=True)


if __name__ == '__main__':
    run()
