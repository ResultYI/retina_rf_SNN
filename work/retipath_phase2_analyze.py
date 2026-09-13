from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from retipath_phase2_common import (
    ROOT, SEEDS, CONDITIONS, load_json, save_json, load_train, load_development, fresh_model,
    evaluate, jacobian_modes, mode_shares, paired_bootstrap, sha, tensor_sha,
)
from retipath_spatial_ei_pilot import context_values
from retipath_spatial_ei_report import profile_metrics, radial_cdf


def model_from_checkpoint(path: Path):
    cp=torch.load(path,weights_only=True)
    model=fresh_model(cp,cp['seed'],cp['condition'],cp['rms'])
    model.load_state_dict(cp['model'],strict=True)
    return model,cp


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:
        return None
    return float(np.corrcoef(x,y)[0,1])


def rank_average(x: np.ndarray) -> np.ndarray:
    values, inverse, counts=np.unique(x,return_inverse=True,return_counts=True)
    starts=np.cumsum(counts)-counts
    return (starts+(counts-1)/2)[inverse]


def frozen_observations(ref: dict, dev, model) -> tuple[dict, dict]:
    thresholds=ROOT/'output/experiments/context_dependent_rf_gain_20260907/context_thresholds.csv'
    rows=list(csv.DictReader(thresholds.open()))
    row=next(r for r in rows if r['cell_id']==ref['cell_id'])
    support=model.feature_bank.bc_support[0]>0
    assert row['support_sha256']==tensor_sha(support)
    assert row['input_sha256']==ref['input_sha256']
    path=ROOT/'.omo/evidence/context-rf-gain-20260907'/f"{ref['cell_id'].replace('#','_')}.npz"
    selected={}
    context=context_values(dev,support)
    with np.load(path,allow_pickle=False) as original:
        for label in ('LOW','HIGH'):
            indices=original[f'indices_{label}']
            if len(indices):
                indices=indices[np.linspace(0,len(indices)-1,min(20,len(indices))).astype(np.int64)]
                values=context[indices[:,0],indices[:,1]]
                if label=='LOW': assert np.all(values<=float(row['LOW_threshold']))
                else: assert np.all(values>=float(row['HIGH_threshold']))
                assert bool(dev.valid_mask[indices[:,0],indices[:,1]].all())
            selected[label]=indices.tolist()
    return selected,{'thresholds_sha256':sha(thresholds),'original_indices_path':str(path),
                     'original_indices_sha256':sha(path),'thresholds':[float(row['LOW_threshold']),float(row['HIGH_threshold'])]}


def rf_modes(model,dev,selections: dict,positions: np.ndarray,grid: np.ndarray) -> tuple[dict,dict,dict]:
    model.double().eval()
    for p in model.parameters(): p.requires_grad_(False)
    states={k:tensor_sha(v) for k,v in model.state_dict().items()}
    profiles={};contexts={};quality={'sum_max_abs_error':0.,'sum_relative_l2_error':0.,'observations':0}
    for label in ('LOW','HIGH'):
        observations=[];p_rows=[];cdf_rows=[]
        for sequence,target in selections[label]:
            x=dev.cone_drive[sequence:sequence+1,:target+1].double()
            h=dev.spike_events[sequence:sequence+1,:target+1].double()
            jac,modes,check=jacobian_modes(model,x,h)
            assert np.isfinite(jac).all() and np.isfinite(modes).all()
            assert check['sum_relative_l2_error']<1e-8
            shares=mode_shares(jac,modes)
            energy=shares['total_energy'];gain=math.sqrt(energy)
            assert abs(shares['q'].sum()-1)<1e-12
            assert abs(shares['signed_projection'].sum()-1)<1e-8
            p=np.square(jac).sum(0)/energy
            assert abs(p.sum()-1)<1e-12
            center,radius=profile_metrics(p,positions)
            p_rows.append(p);cdf_rows.append(radial_cdf(p,positions,grid))
            observations.append({'sequence':sequence,'bin':target,'prefix_bins':target+1,
                'gain':gain,'centroid_deg':center.tolist(),'radius_deg':radius,
                'q1':float(shares['q'][0]),'q2':float(shares['q'][1]),
                'signed1':float(shares['signed_projection'][0]),'signed2':float(shares['signed_projection'][1]),
                'cross_over_total_energy':shares['cross_over_total_energy'],
                'energy_before_last16_fraction':float(np.square(jac[:-16]).sum()/energy) if len(jac)>16 else 0.})
            quality['sum_max_abs_error']=max(quality['sum_max_abs_error'],check['sum_max_abs_error'])
            quality['sum_relative_l2_error']=max(quality['sum_relative_l2_error'],check['sum_relative_l2_error'])
            quality['observations']+=1
        profile=np.mean(p_rows,axis=0);center,radius=profile_metrics(profile,positions)
        profiles[label]=profile
        contexts[label]={'n':len(observations),'gain_median':float(np.median([o['gain'] for o in observations])),
                        'centroid_x_deg':float(center[0]),'centroid_y_deg':float(center[1]),
                        'radius_mean_profile_deg':radius,'radius_observation_median_deg':float(np.median([o['radius_deg'] for o in observations])),
                        'profile':profile,'cdf':np.mean(cdf_rows,axis=0),'observations':observations,
                        'q1_mean':float(np.mean([o['q1'] for o in observations])),
                        'q2_mean':float(np.mean([o['q2'] for o in observations])),
                        'signed1_mean':float(np.mean([o['signed1'] for o in observations])),
                        'signed2_mean':float(np.mean([o['signed2'] for o in observations])),
                        'cross_over_total_energy_mean':float(np.mean([o['cross_over_total_energy'] for o in observations]))}
    assert all(states[k]==tensor_sha(v) for k,v in model.state_dict().items())
    low,high=contexts['LOW'],contexts['HIGH']
    delta=profiles['HIGH']-profiles['LOW'];cdf_delta=high['cdf']-low['cdf']
    rf={'status':'VERIFIED','LOW_n':low['n'],'HIGH_n':high['n'],
        'gain_log_HIGH_over_LOW':math.log(high['gain_median']/low['gain_median']),
        'TV':float(np.abs(delta).sum()/2),
        'centroid_shift_deg':math.hypot(high['centroid_x_deg']-low['centroid_x_deg'],high['centroid_y_deg']-low['centroid_y_deg']),
        'delta_radius_mean_profile_deg':high['radius_mean_profile_deg']-low['radius_mean_profile_deg'],
        'delta_radius_observation_median_deg':high['radius_observation_median_deg']-low['radius_observation_median_deg'],
        'centered_radial_CDF_max':float(np.abs(cdf_delta).max()),
        'centered_radial_CDF_L1_deg':float(np.trapezoid(np.abs(cdf_delta),grid)),
        'delta_P':delta.tolist()}
    mode={'status':'VERIFIED','LOW_n':low['n'],'HIGH_n':high['n'],
          'delta_q1':high['q1_mean']-low['q1_mean'],'delta_q2':high['q2_mean']-low['q2_mean'],
          'delta_signed1':high['signed1_mean']-low['signed1_mean'],
          'delta_signed2':high['signed2_mean']-low['signed2_mean'],
          'TV':rf['TV'],'centered_radial_CDF_L1_deg':rf['centered_radial_CDF_L1_deg']}
    for label,values in contexts.items():
        for key in ('gain_median','centroid_x_deg','centroid_y_deg','radius_mean_profile_deg','radius_observation_median_deg','profile','cdf'):
            val=values[key]
            rf[f'{label}_{key}']=val.tolist() if isinstance(val,np.ndarray) else val
        for key in ('q1_mean','q2_mean','signed1_mean','signed2_mean','cross_over_total_energy_mean','observations'):
            mode[f'{label}_{key}']=values[key]
    return rf,mode,quality


def analyze_job(out_string: str,cell_id: str,seed: int) -> dict:
    torch.set_num_threads(1)
    out=Path(out_string);config=load_json(out/'checkpoints/protocol.json');ref=config['cells'][cell_id]
    destination=out/'checkpoints'/cell_id.replace('#','_')/str(seed)/'analysis.json'
    if destination.exists():
        saved=load_json(destination)
        assert saved['protocol_sha256']==config['protocol_sha256']
        return saved
    train=load_train(ref);dev=load_development(ref)
    grid=np.array(config['radial_grid_deg']);prediction=[];rf_rows=[];mode_rows=[];checks={}
    for condition in CONDITIONS:
        path=destination.parent/condition/'refit.pt'
        model,cp=model_from_checkpoint(path)
        assert cp['phase']=='refit' and cp['step']==cp['best_step']
        assert cp['protocol_sha256']==config['protocol_sha256']
        train_nll,_=evaluate(model,train);dev_nll,logits=evaluate(model,dev)
        assert train_nll==cp['refit_train_nll']
        common={'cell_id':cell_id,'group':ref['group'],'seed':seed,'condition':condition,'refit_updates':cp['step']}
        inner=torch.load(path.with_name('inner.pt'),weights_only=True)
        prediction.append(common|{'train_nll':train_nll,'dev_nll':dev_nll,'inner_best_nll':inner['selection_status']['best_nll'],
                                  'inner_stop_step':inner['step'],'reached_maximum':inner['step']==3000})
        selections,provenance=frozen_observations(ref,dev,model)
        if not all(selections.values()):
            rf_rows.append(common|{'status':'UNVERIFIED','reason':'missing existing HIGH/LOW observations'})
            mode_rows.append(common|{'status':'UNVERIFIED','reason':'missing existing HIGH/LOW observations'})
            checks[condition]={'RF_status':'UNVERIFIED','observation_provenance':provenance}
            continue
        rf,mode,quality=rf_modes(model,dev,selections,cp['cone_positions_degs'].double().numpy(),grid)
        rf_rows.append(common|rf);mode_rows.append(common|mode)
        checks[condition]={'train_replay_error':0.,'dev_logits_sha256':tensor_sha(logits),
                           'refit_checkpoint_sha256':sha(path),'RF':quality,'observation_provenance':provenance}
    result={'cell_id':cell_id,'seed':seed,'protocol_sha256':config['protocol_sha256'],
            'prediction':prediction,'RF':rf_rows,'mode':mode_rows,'checks':checks}
    save_json(destination,result)
    print('ANALYZED',cell_id,seed,flush=True)
    return result


def csv_rows(path: Path,rows: list[dict]) -> None:
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def prediction_statistics(prediction: list[dict],cells: list[str],config: dict) -> tuple[list[dict],dict]:
    lookup={(r['cell_id'],r['seed'],r['condition']):r for r in prediction}
    rng=np.random.default_rng(config['bootstrap_seed'])
    draws=rng.integers(0,len(cells),size=(10000,len(cells)))
    rows=[];summaries={}
    for left,right in (('C','B'),('C','A'),('B','A')):
        differences=[]
        for seed in SEEDS:
            values=[]
            for cell in cells:
                a,b=lookup[cell,seed,left],lookup[cell,seed,right]
                delta=a['dev_nll']-b['dev_nll'];values.append(delta)
                rows.append({'row_type':'cell_seed','cell_id':cell,'seed':seed,'comparison':f'{left}-{right}',
                    'left_train_nll':a['train_nll'],'right_train_nll':b['train_nll'],
                    'left_dev_nll':a['dev_nll'],'right_dev_nll':b['dev_nll'],
                    'delta_train_nll':a['train_nll']-b['train_nll'],'delta_dev_nll':delta,
                    'left_refit_updates':a['refit_updates'],'right_refit_updates':b['refit_updates'],
                    'left_inner_stop':a['inner_stop_step'],'right_inner_stop':b['inner_stop_step']})
            values=np.array(values);differences.append(values)
            summary=paired_bootstrap(values,draws)
            summaries[f'{left}-{right}/{seed}']=summary
            rows.append({'row_type':'seed_summary','seed':seed,'comparison':f'{left}-{right}',**summary})
        per_cell=np.mean(differences,axis=0)
        for cell,value in zip(cells,per_cell,strict=True):
            rows.append({'row_type':'cell_mean_over_seeds','cell_id':cell,'seed':'all','comparison':f'{left}-{right}',
                         'delta_dev_nll':float(value)})
        summary=paired_bootstrap(per_cell,draws)
        summary['seed_means']=[float(v.mean()) for v in differences]
        summary['stable_by_frozen_rule']=summary['ci_high']<0 and all(v<0 for v in summary['seed_means'])
        summaries[f'{left}-{right}/all']=summary
        rows.append({'row_type':'population_summary','seed':'all','comparison':f'{left}-{right}',**summary})
    return rows,summaries


def rf_statistics(rf_rows: list[dict],mode_rows: list[dict],cells: list[str]) -> dict:
    rf={(r['cell_id'],r['seed'],r['condition']):r for r in rf_rows if r['status']=='VERIFIED'}
    modes={(r['cell_id'],r['seed'],r['condition']):r for r in mode_rows if r['status']=='VERIFIED'}
    summary={}
    for condition in CONDITIONS:
        rows=[r for r in rf.values() if r['condition']==condition]
        per_seed={str(seed):{metric:float(np.mean([r[metric] for r in rows if r['seed']==seed]))
                  for metric in ('TV','centered_radial_CDF_L1_deg','centered_radial_CDF_max',
                                 'delta_radius_observation_median_deg','gain_log_HIGH_over_LOW')}
                  for seed in SEEDS}
        repeat=[];paired=[]
        for cell in cells:
            if not all((cell,seed,condition) in rf for seed in SEEDS):
                continue
            delta=[np.array(rf[cell,seed,condition]['delta_P']) for seed in SEEDS]
            cosine=[]
            for i,j in itertools.combinations(range(3),2):
                denominator=np.linalg.norm(delta[i])*np.linalg.norm(delta[j])
                cosine.append(float(np.dot(delta[i],delta[j])/denominator) if denominator>0 else None)
            dq=[modes[cell,seed,condition]['delta_q1'] for seed in SEEDS]
            repeat.append({'cell_id':cell,'delta_P_pairwise_cosines':cosine,
                           'all_three_cosines_positive':all(v is not None and v>0 for v in cosine),
                           'delta_q1_by_seed':dq,'delta_q1_same_sign':all(v>0 for v in dq) or all(v<0 for v in dq)})
            paired.append((np.mean([rf[cell,seed,condition]['TV'] for seed in SEEDS]),
                           np.mean([rf[cell,seed,condition]['centered_radial_CDF_L1_deg'] for seed in SEEDS]),
                           np.mean([abs(modes[cell,seed,condition]['delta_q1']) for seed in SEEDS]),
                           np.mean([modes[cell,seed,condition]['delta_q2'] for seed in SEEDS]),
                           np.mean([rf[cell,seed,condition]['delta_radius_observation_median_deg'] for seed in SEEDS])))
        a=np.array(paired)
        summary[condition]={'per_seed_equal_cell_means':per_seed,'valid_cells':len(repeat),'repeatability':repeat,
            'LOW_q1_mean':float(np.mean([r['LOW_q1_mean'] for r in modes.values() if r['condition']==condition])),
            'HIGH_q1_mean':float(np.mean([r['HIGH_q1_mean'] for r in modes.values() if r['condition']==condition])),
            'delta_P_all_pairs_positive_cells':sum(r['all_three_cosines_positive'] for r in repeat),
            'median_pairwise_delta_P_cosine':float(np.median([v for r in repeat for v in r['delta_P_pairwise_cosines'] if v is not None])),
            'delta_q1_same_sign_cells':sum(r['delta_q1_same_sign'] for r in repeat),
            'TV_vs_abs_delta_q1_Pearson':correlation(a[:,0],a[:,2]),
            'TV_vs_abs_delta_q1_Spearman':correlation(rank_average(a[:,0]),rank_average(a[:,2])),
            'centered_radial_vs_abs_delta_q1_Pearson':correlation(a[:,1],a[:,2]),
            'delta_radius_vs_delta_q2_Pearson':correlation(a[:,4],a[:,3]),
            'mean_delta_q2':float(a[:,3].mean()),
            'mean_abs_delta_q1':float(a[:,2].mean()),
            'association_unit':'22 cells or available paired subset; first average seeds within cell; descriptive only'}
    return summary


def run_analysis(out: Path,workers: int) -> None:
    complete=load_json(out/'checkpoints/training_complete.json')
    assert complete['inner_fits']==198 and complete['fresh_refits']==198 and not complete['development_used_for_selection']
    config=load_json(out/'checkpoints/protocol.json');cells=list(config['cells'])
    lock=load_json(out/'checkpoints/selection_lock.json')
    assert len(lock['files'])==198
    for path,digest in lock['files'].items(): assert sha(out/path)==digest
    for path,digest in config['source_hashes'].items(): assert sha(ROOT/path)==digest
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(analyze_job,str(out),cell,seed) for cell in cells for seed in SEEDS]
        for future in as_completed(jobs):
            results.append(future.result())
    results.sort(key=lambda r:(r['cell_id'],r['seed']))
    prediction=[row for r in results for row in r['prediction']]
    rf=[row for r in results for row in r['RF']];mode=[row for r in results for row in r['mode']]
    assert len(prediction)==len(rf)==len(mode)==198
    paired,stats=prediction_statistics(prediction,cells,config)
    rf_stats=rf_statistics(rf,mode,cells)
    csv_rows(out/'prediction_population.csv',paired)
    csv_rows(out/'rf_population.csv',rf);csv_rows(out/'mode_contribution.csv',mode)
    summary={'protocol_sha256':config['protocol_sha256'],'prediction':stats,'RF':rf_stats,
             'raw_prediction':prediction,'analysis_source_sha256':sha(Path(__file__)),
             'model_files_unchanged':True,'original_input_hashes_unchanged':True,
             'complete_cell_seed_condition_rows':198,'independent_test':'NONE'}
    for path,digest in config['source_hashes'].items(): assert sha(ROOT/path)==digest
    for ref in config['cells'].values(): assert sha(Path(ref['input_path']))==ref['input_sha256']
    save_json(out/'checkpoints/analysis_complete.json',summary)
    from retipath_phase2_report import make_figures,write_report
    make_figures(out,config,prediction,rf,mode,stats,rf_stats)
    write_report(out,config,prediction,stats,rf_stats)
    print('PHASE 2 COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'output/experiments/retipath_spatial_ei_phase2_population')
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args();run_analysis(args.out,args.workers)
