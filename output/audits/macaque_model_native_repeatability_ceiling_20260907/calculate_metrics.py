# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "torch"]
# ///
# How to run: D:/anaconda/python.exe -B -u calculate_metrics.py primary
from __future__ import annotations

import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Final

import numpy as np
import torch
from statistics_core import pearson,repeatability,spearman

OUT: Final=Path(__file__).resolve().parent
MODELS: Final=('aligned','LN','CNN')
MARK: Final='NEW_DETERMINISTIC_DERIVED_ARTIFACT'


def sha(path: Path) -> str:
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main() -> None:
    stage=sys.argv[1]
    assert stage in {'primary','secondary'}
    protocol=json.loads((OUT/'protocol_hash.json').read_text())['sha256']
    assert sha(OUT/'PROTOCOL.md')==protocol
    assert json.loads((OUT/'preflight.json').read_text())['all_passed']
    data=torch.load(OUT/'inference_inputs.pt',weights_only=True,map_location='cpu')
    mask=data['mask'].numpy()
    partitions=tuple(tuple(i-1 for i in row['A']) for row in json.loads((OUT/'split_half_partitions.json').read_text()))
    assert len(partitions)==10
    tables={}
    cells=list(data['cells'])
    if stage=='primary':
        for backend in ('cpu','cnn'): assert (OUT/f'primary_{backend}_inference_checks.json').exists()
        assert not (OUT/'primary_lock.json').exists()
        split_rows=[]
        repeat_tables={component:[] for component in ('occupancy','count','extra_count')}
        model_rows=[]
        loro=[]
        for cid,record in data['cells'].items():
            counts=record['counts'].numpy().astype(np.float64)[:,mask]
            occ=(counts>0).astype(np.float64)
            extra=np.maximum(counts-1,0)
            reps={name:repeatability(values,partitions) for name,values in [('occupancy',occ),('count',counts),('extra_count',extra)]}
            base={'cell_id':cid,'group':record['group'],'recording_id':record['recording_id']}
            multi=float((counts>1).mean())
            beyond=float(extra.sum()/counts.sum())
            for name,rep in reps.items():
                repeat_tables[name].append({**base,'r3':rep.r3,'raw_split_r_mean':rep.raw_r_mean,'Rel6':rep.rel6,'ceiling':rep.ceiling,'status':rep.status,'multi_spike_bin_fraction':multi,'multi_spike_fraction_among_occupied':float((counts>1).sum()/(counts>0).sum()),'beyond_first_spike_fraction':beyond,'event_rate_hz':float(occ.mean()*150),'firing_rate_hz':float(counts.mean()*150),'count_minus_occupancy_Rel6':reps['count'].rel6-reps['occupancy'].rel6})
                for s,r in enumerate(rep.split_r):split_rows.append({**base,'component':name,'split':s+1,'r':r})
            mean=occ.mean(axis=0)
            noise=float(occ.var(axis=0,ddof=1).mean()/6)
            totalvar=float(mean.var(ddof=0))
            signal=totalvar-noise
            for model in MODELS:
                pred=torch.load(OUT/'predictions'/f'{cid.replace("#","_")}_{model}_primary.pt',weights_only=True)
                assert pred['protocol_sha256']==protocol
                probabilities=pred['probabilities'].numpy().astype(np.float64)
                assert np.array_equal(probabilities,np.repeat(probabilities[:1],6,axis=0))
                p=probabilities[0,mask]
                r=pearson(p,mean)
                ceiling=reps['occupancy'].ceiling
                ncorr=r/ceiling if np.isfinite(ceiling) and ceiling>0 else float('nan')
                mse=float(np.mean((mean-p)**2))
                msesignal=mse-noise
                fev=1-msesignal/signal if signal>0 and np.isfinite(signal) else float('nan')
                model_rows.append({**base,'model':model,'contract':'STIMULUS_ONLY_ZERO_HISTORY','raw_Pearson':r,'raw_Spearman':spearman(p,mean),'NCorr':ncorr,'FEV':fev,'Rel6':reps['occupancy'].rel6,'correlation_ceiling':ceiling,'normalization_status':reps['occupancy'].status,'FEV_status':'DEFINED' if signal>0 else 'UNDEFINED_NONPOSITIVE_SIGNAL_VARIANCE','NCorr_gap':1-ncorr,'FEV_gap':1-fev,'raw_correlation_gap':ceiling-r,'mean_predicted_probability':float(p.mean()),'mean_repeat_occupancy':float(mean.mean()),'repeat_mean_variance':totalvar,'repeat_mean_noise_variance':noise,'signal_variance':signal,'MSE':mse,'MSE_signal':msesignal,'count_Rel6':reps['count'].rel6,'extra_count_Rel6':reps['extra_count'].rel6,'multi_spike_bin_fraction':multi,'beyond_first_spike_fraction':beyond,'event_rate_hz':float(occ.mean()*150)})
                values=[pearson(p,np.delete(occ,i,axis=0).mean(axis=0)) for i in range(6)]
                for i,value in enumerate(values):
                    loro.append({**base,'model':model,'left_out_repeat':i+1,'raw_Pearson':value,'six_repeat_raw_Pearson':r,'min':min(values),'max':max(values),'range':max(values)-min(values),'std':float(np.std(values))})
        tables['split_half_correlations.csv']=split_rows
        for component,rows in repeat_tables.items():tables[f'per_cell_{component}_repeatability.csv']=rows
        tables['stimulus_only_model_scores.csv']=model_rows
        tables['per_cell_model_native_ceiling.csv']=model_rows
        tables['leave_one_repeat_out_sensitivity.csv']=loro
        populations=[]
        for model in MODELS:
            rows=[r for r in model_rows if r['model']==model]
            row={'model':model,'cells':len(rows)}
            for metric in ('raw_Pearson','NCorr','FEV','NCorr_gap','FEV_gap'):
                values=np.array([r[metric] for r in rows]); values=values[np.isfinite(values)]
                row[metric+'_n']=len(values);row[metric+'_mean']=float(values.mean()) if len(values) else float('nan');row[metric+'_median']=float(np.median(values)) if len(values) else float('nan')
            populations.append(row)
        tables['population_model_native_ceiling.csv']=populations
        pairs=[]
        index_cache={}
        for left,right in [('aligned','LN'),('aligned','CNN'),('CNN','LN')]:
            for metric in ('raw_Pearson','NCorr','FEV'):
                lv={r['cell_id']:r[metric] for r in model_rows if r['model']==left}
                rv={r['cell_id']:r[metric] for r in model_rows if r['model']==right}
                common=[c for c in cells if np.isfinite(lv[c]) and np.isfinite(rv[c])]
                n=len(common)
                assert n>0
                delta=np.array([lv[c]-rv[c] for c in common])
                if n not in index_cache:index_cache[n]=np.random.default_rng(20260907).integers(0,n,size=(100000,n))
                boot=delta[index_cache[n]].mean(axis=1)
                low,high=np.quantile(boot,[.025,.975],method='linear')
                pairs.append({'left':left,'right':right,'metric':metric,'n_cells':n,'mean_difference':float(delta.mean()),'CI95_low':float(low),'CI95_high':float(high),'bootstrap_draws':100000,'bootstrap_seed':20260907,'cell_ids':';'.join(common)})
        tables['paired_model_comparisons.csv']=pairs
        groups=[]
        for group in ('MC_ON','MC_OFF','PC_ON','PC_OFF'):
            for model in MODELS:
                subset=[r for r in model_rows if r['group']==group and r['model']==model]
                row={'group':group,'model':model,'cells':len(subset)}
                for metric in ('Rel6','count_Rel6','extra_count_Rel6','raw_Pearson','NCorr','FEV'):
                    values=np.array([r[metric] for r in subset]); values=values[np.isfinite(values)]
                    row[metric+'_mean']=float(values.mean()) if len(values) else float('nan')
                    row[metric+'_median']=float(np.median(values)) if len(values) else float('nan')
                groups.append(row)
        tables['cell_group_summary.csv']=groups
        associations=[]
        for model in MODELS:
            rows=[r for r in model_rows if r['model']==model]
            for gap in ('NCorr_gap','FEV_gap'):
                for burden in ('multi_spike_bin_fraction','beyond_first_spike_fraction','extra_count_Rel6','event_rate_hz'):
                    x=np.array([r[burden] for r in rows]);y=np.array([r[gap] for r in rows]);valid=np.isfinite(x)&np.isfinite(y)
                    associations.append({'model':model,'gap':gap,'burden':burden,'n_cells':int(valid.sum()),'Pearson':pearson(x[valid],y[valid]),'Spearman':spearman(x[valid],y[valid])})
        tables['multispike_gap_relations.csv']=associations
    else:
        primary=json.loads((OUT/'primary_lock.json').read_text())
        assert all(sha(OUT/name)==digest for name,digest in primary['output_sha256'].items())
        for backend in ('cpu','cnn'):assert (OUT/f'conditional_{backend}_inference_checks.json').exists()
        conditional_rows=[]
        for cid,record in data['cells'].items():
            occ=record['events'].numpy().astype(np.float64)[:,mask];mean=occ.mean(axis=0)
            for model in MODELS:
                primary_pred=torch.load(OUT/'predictions'/f'{cid.replace("#","_")}_{model}_primary.pt',weights_only=True)['probabilities'].numpy().astype(np.float64)[:,mask]
                cond=torch.load(OUT/'predictions'/f'{cid.replace("#","_")}_{model}_conditional.pt',weights_only=True)['probabilities'].numpy().astype(np.float64)[:,mask]
                for repeat in range(6):
                    r=pearson(cond[repeat],occ[repeat]);r0=pearson(primary_pred[repeat],occ[repeat]);rm=pearson(cond[repeat],mean);rm0=pearson(primary_pred[repeat],mean)
                    conditional_rows.append({'cell_id':cid,'group':record['group'],'model':model,'repeat':repeat+1,'contract':'FORMAL_CONDITIONAL_HISTORY','Pearson_to_corresponding_occupancy':r,'zero_history_Pearson_to_corresponding_occupancy':r0,'delta_to_corresponding_occupancy':r-r0,'Pearson_to_six_repeat_mean':rm,'zero_history_Pearson_to_six_repeat_mean':rm0,'delta_to_six_repeat_mean':rm-rm0,'ceiling_normalized':False})
        tables['conditional_history_model_scores.csv']=conditional_rows
    for name,rows in tables.items():
        with (OUT/name).open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=[*rows[0],'artifact_class','protocol_sha256']);writer.writeheader()
            writer.writerows({**r,'artifact_class':MARK,'protocol_sha256':protocol} for r in rows)
    if stage=='primary':
        files=[OUT/name for name in tables]+list((OUT/'predictions').glob('*_primary.pt'))+[OUT/'primary_cpu_inference_checks.json',OUT/'primary_cnn_inference_checks.json']
        with (OUT/'primary_lock.json').open('x') as stream:
            json.dump({'artifact_class':MARK,'frozen_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':protocol,'before_conditional_inference':True,'output_sha256':{p.relative_to(OUT).as_posix():sha(p) for p in files}},stream,indent=2)
        print(json.dumps(populations,indent=2))
    print(stage,'metrics complete; files',len(tables),flush=True)


if __name__=='__main__':main()
