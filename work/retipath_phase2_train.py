from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from retipath_phase2_common import (
    ROOT, POP, PILOT, SEEDS, CONDITIONS, SelectionStop, load_json, save_json, load_train,
    fresh_model, compute_rms, inner_input_mask, minibatches, evaluate, make_inner_dev, sha, tensor_sha,
)
from models.mechanistic_retina.retipath_spatial_ei import spatial_ei_parameters
from training.mechanistic_retina.losses import expected_bernoulli_nll


def digest_state(state: dict) -> str:
    digest = hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze(out: Path, initialization: str) -> None:
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    (out/'checkpoints').mkdir()
    (out/'figures').mkdir()
    pilot = load_json(PILOT/'correctness.json')
    for path,digest in pilot['protocol']['source_hashes'].items():
        if path.startswith('models'):
            assert sha(ROOT/path)==digest, path
    lock = load_json(POP/'population_training_lock.json')
    manifest = load_json(POP/'evidence_manifest.json')
    assert len(lock['cells'])==22
    refs = {}
    for row in lock['cells']:
        safe = row['cell_id'].replace('#','_')
        key = f'checkpoints/{safe}/M2/best.pt'
        assert key in manifest['artifacts_sha256']
        inp = POP/row['input_path']; meta = ROOT/row['checkpoint_path']; old = POP/key
        assert sha(inp)==row['input_sha256'] and sha(meta)==row['checkpoint_sha256']
        ref = {'cell_id':row['cell_id'], 'group':row['group'], 'is_pilot':row['is_pilot'],
               'input_path':str(inp), 'input_sha256':sha(inp), 'metadata_path':str(meta),
               'metadata_sha256':sha(meta), 'M2_checkpoint_path':str(old),
               'M2_checkpoint_sha256':manifest['artifacts_sha256'][key]}
        train = load_train(ref)
        inner = make_inner_dev(train)
        metadata = torch.load(meta,weights_only=True,mmap=True)
        ref.update({'config':metadata['model_config'], 'inner_boundaries':[asdict(b) for b in inner.boundaries],
                    'inner_train_sequences':len(inner.train.cone_drive), 'inner_validation_sequences':len(inner.development.cone_drive),
                    'inner_train_score_bins':int(inner.train.valid_mask.sum()),
                    'inner_validation_score_bins':int(inner.development.valid_mask.sum()),
                    'inner_fit_input_bins':int(inner_input_mask(inner).sum()),
                    'full_train_sequences':len(train.cone_drive), 'full_train_score_bins':int(train.valid_mask.sum()),
                    'schedule_hashes':{str(seed):{'inner':tensor_sha(minibatches(len(inner.train.cone_drive),seed)),
                                                'refit':tensor_sha(minibatches(len(train.cone_drive),seed))} for seed in SEEDS}})
        refs[row['cell_id']] = ref
    source_paths = [Path(__file__), ROOT/'work/retipath_phase2_common.py', ROOT/'tests/test_retipath_phase2.py',
                    ROOT/'evaluation/mechanistic_retina/factorized_ln_split.py', ROOT/'training/mechanistic_retina/losses.py',
                    ROOT/'training/mechanistic_retina/optimizer.py']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    hashes.update({k:v for k,v in pilot['protocol']['source_hashes'].items() if k.startswith('models')})
    hashes[str((POP/'evidence_manifest.json').relative_to(ROOT))]=sha(POP/'evidence_manifest.json')
    hashes[str((POP/'population_training_lock.json').relative_to(ROOT))]=sha(POP/'population_training_lock.json')
    frozen = {'created_utc':datetime.now(timezone.utc).isoformat(), 'initialization':initialization,
              'initialization_reason':'train-only checkpoint selection and existing fresh-refit semantics; no development-selected weights in fresh mode',
              'branch':subprocess.check_output(['git','branch','--show-current'],text=True,cwd=ROOT).strip(),
              'HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
              'seeds':list(SEEDS), 'maximum_updates':3000, 'patience_updates':400, 'evaluation_interval':25,
              'min_delta_for_patience_only':1e-7, 'lr':.003, 'batch_size':4, 'weight_decay':0,
              'gradient_norm_clip':5, 'scheduler':None, 'optimizer':'Adam', 'betas':[.9,.999], 'eps':1e-8,
              'primary':'C-B Bernoulli NLL after fresh full-train refit', 'bootstrap_seed':2026091399,
              'bootstrap_samples':10000, 'bootstrap_unit':'biological cell; average seeds within cell for overall result',
              'RF_max_observations_per_context':20, 'radial_grid_deg':[i*.005 for i in range(301)],
              'cells':refs, 'source_hashes':hashes, 'pretraining_tests':'11 passed: pilot regression, exact integration, full mode sum, patience and paired bootstrap'}
    text = '''# RetiPath spatial E/I Phase 2 冻结协议

## 范围和来源

22 macaque cells，A/B/C与已完成pilot同一架构/forward；本阶段不改任何模型模块。A为当前RetiPath架构，B为两个现有空间basis的E/I电流整合，C与B相同表示和参数数目、仅换为电导整合。K是有效空间尺度，不是树突区室。固定alpha bounds [0.05,2]、EL/EE/EI=0/1/−1/3、gL=1、Cm=G0*tau_ref、V0=2/9；全部其余固定与可学习量沿用pilot。文献依据与本轮设计的区分见原pilot PROTOCOL.md；本阶段不增加新机制、RF loss、baseline或数据。

默认fresh初始化：只复用原checkpoint中的配置、位置、类型和极性，不加载已经由外部development选择的M2可学习权重。相同cell/seed的A/B/C上游初始权重相同，alpha初值1，B/C新增aE/aI/output_scale初值1。这改变的是本阶段用户指定的选模/refit流程，并不修改pilot架构。A表示同架构的fresh fit，不是复用pilot四cell的已训练A权重。固定几何和架构属于历史选择，不声称本development完全未经历史使用。

## 数据与模型选择

只在既有train [0,16)s内调用原make_inner_dev：每trial 80/20，dev_start=1920 bin，guard=[1860,1920)，即[12.4,12.8)s；inner fit在[0,12.4)，inner validation评分在[12.8,16)。guard只作为validation的上下文，guard前的validation输入/observed history置零。保留原150-bin独立sequence、30-bin warmup/120评分mask，实际mask为二者交集。原calibrated frontend、alignment和Bernoulli occupancy不变。

对每个cell的每个固定seed分别训练A/B/C。Adam lr=.003,batch4,wd0,clip_norm5，无scheduler，最多3000 updates。step0和每25步评估inner validation；选精确最小NLL，并列保留更早step。patience是400 optimizer updates，即16个评估间隔；仅当相对上次有效改善超过1e−7时重置patience，该min_delta不改变精确best的选择。相同cell/seed/阶段共享同一完整minibatch schedule；不同条件允许按各自patience停止，实际使用同一schedule前缀。三个seed一次冻结，不按结果追加seed或训练。

inner RMS只从该seed初始上游作用于inner-fit的实际输入bins计算，含原warmup但排除guard/inner validation和末尾占位零。随后固定。选定step后丢弃inner权重与optimizer，使用相同seed全新初始化，在完整[0,16)重新计算初始RMS并固定，fresh Adam训练到该条件选定step。B/C每阶段共享同一RMS。不得将inner optimizer/权重带到refit。step0可被选中，此时refit为零update。

所有198次模型选择完成且冻结后，才开始任何完整train refit；全部198次refit完成后才读取/评分已消费[16,20)作为development comparison。原输入归档mmap包含train/development两split，选模路径只解引用train tensor，不计算development统计。无新时间block访问。

## 预测统计

Primary C−B NLL (nats/scored bin)，secondary C−A、B−A。保存每seed、每cell原始A/B/C和配对差。每seed独立报告22-cell mean/median/wins、paired percentile95% bootstrap CI（10000次，按cell成对抽样、相同比较共享索引）。整体先在每cell内对3 seeds等权平均差，再按22个cell bootstrap；不把66个cell-seed当成独立生物样本。seed mean的范围/方向一致性明确报告。整体CI完全低于0且3个seed mean均低于0作为本阶段“稳定预测收益”的描述规则；每seed CI和wins另列，不伪称独立确认或生物机制发现。

## RF与模式贡献

沿用pilot的45严格过去帧HIGH/LOW、train q20/q80及同源development索引：从既有至多100观测用floor(linspace(0,n−1,min(20,n)))取至多20个，所有seeds/条件相同。不搜索新context。若某cell某context无既有观测，则RF标为UNVERIFIED并报告覆盖，不替换cell或搜索观测，prediction仍保留。

用refit checkpoint，observed history固定，float64计算logit对从sequence reset到目标bin完整因果prefix的Jacobian J。无16-lag截断或膜状态detach。G=||J||2，每观测P(x)=sum_lag J²/sum_lag,x J²；context内等权平均P。报告gain、centroid、second-moment radius、TV和每观测自身重心下径向CDF的context平均差。使用pilot同一0..1.5degree/0.005degree网格。每cell保存seed平均HIGH−LOW ΔP图（A/B/C并列），同时保存每seed完整ΔP数组。另给seed间ΔP cosine与差异幅度，描述normalized dynamics的重复性；没有新增RF阈值、面积或loss。

B/C使用真实integrator输入uE/uI处的链式VJP分解J1/J2；膜与adaptation及其完整因果依赖都在下游adjoint中。A使用真实total-current下游adjoint，通过已有BC/AC空间成分拆分两个模式。检验J1+J2在float64容差内恢复完整J。q_k=||J_k||²/(||J1||²+||J2||²)，非负且sum=1。因为模式Jacobian可重叠，q不是总RF能量的无交叉项分割：同时报告signed contribution <Jk,J>/||J||²（可负、sum=1）与2<J1,J2>/||J||²。HIGH/LOW分别平均逐观测q，报告Δq。模式编号沿用原basis顺序，不按结果重排。

检查每seed/cell的TV、中心化径向差与|Δq1|的关联，报告散点、Pearson和Spearman描述量（无额外显著性检验），并报告ΔP和Δq跨seed一致性。存在关联不证明因果重分配，q基本不变也不能排除mode内部profile变化；预测优势和空间变化分别判读。

## 保存与完成条件

交付PROTOCOL.md、training_per_seed.csv、prediction_population.csv、rf_population.csv、mode_contribution.csv、figures/和REPORT.md。checkpoints/仅保存本run必要的初始/选步/refit/optimizer与恢复状态，供精确复放和预算核实。原pilot和其他历史结果不覆盖。直接验证fresh初始化、matched schedules、guard、patience单位、B/C同参数、J1+J2、预测重放、完整prefix和profile单位化；不追加无关审计。最终只回答用户指定五个问题，然后停止。

## 实际冻结记录

```json
'''
    (out/'PROTOCOL.md').write_text(text+json.dumps(frozen,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
    frozen['protocol_sha256']=sha(out/'PROTOCOL.md')
    save_json(out/'checkpoints/protocol.json',frozen)
    print('FROZEN',out,'22 cells x 3 conditions x 3 seeds',flush=True)


def stage_data(ref: dict, phase: str):
    full=load_train(ref)
    if phase=='inner':
        inner=make_inner_dev(full)
        return inner.train,inner.development,inner_input_mask(inner)
    return full,None,torch.ones(full.cone_drive.shape[:2],dtype=torch.bool)


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary=path.with_suffix('.pending')
    torch.save(payload,temporary)
    temporary.replace(path)


def fit_job(out_string: str, cell_id: str, seed: int, phase: str) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(out_string); config=load_json(out/'checkpoints/protocol.json'); ref=config['cells'][cell_id]
    train,validation,input_mask=stage_data(ref,phase)
    metadata=torch.load(ref['metadata_path'],weights_only=True,mmap=True)
    upstream=None
    if config['initialization']=='continuation':
        assert sha(Path(ref['M2_checkpoint_path']))==ref['M2_checkpoint_sha256']
        upstream=torch.load(ref['M2_checkpoint_path'],weights_only=True)['model']
    initial=fresh_model(metadata,seed,'A',upstream=upstream)
    rms=compute_rms(initial,train,input_mask)
    upstream_hash=digest_state(initial.state_dict())
    batches=minibatches(len(train.cone_drive),seed)
    schedule_hash=tensor_sha(batches)
    assert schedule_hash==ref['schedule_hashes'][str(seed)]['inner' if phase=='inner' else 'refit']
    base=out/'checkpoints'/cell_id.replace('#','_')/str(seed)
    base.mkdir(parents=True,exist_ok=True)
    results={}
    for condition in CONDITIONS:
        folder=base/condition; folder.mkdir(exist_ok=True)
        destination=folder/f'{phase}.pt'
        if destination.exists():
            saved=torch.load(destination,weights_only=True)
            assert saved['protocol_sha256']==config['protocol_sha256']
            results[condition]={'step':saved['step'],'best_step':saved['best_step']}
            continue
        model=fresh_model(metadata,seed,condition,rms,upstream)
        filtered={k:v for k,v in model.state_dict().items() if not k.startswith('spatial_ei.')}
        assert digest_state(filtered)==upstream_hash
        params=spatial_ei_parameters(model)
        assert {id(p) for p in params}=={id(p) for p in model.parameters() if p.requires_grad}
        optimizer=torch.optim.Adam(params,lr=.003,weight_decay=0)
        fixed={k:v.clone() for k,v in model.named_buffers()}
        initial_state={k:v.clone() for k,v in model.state_dict().items()}
        maximum=3000 if phase=='inner' else int(torch.load(folder/'inner.pt',weights_only=True)['best_step'])
        status=None; curve=[]; step_start=0; best_state=None
        gradient_seen={k:False for k,p in model.named_parameters() if p.requires_grad}
        start_time=time.perf_counter()
        latest=folder/f'{phase}_latest.pt'
        if latest.exists():
            resume=torch.load(latest,weights_only=True)
            assert resume['protocol_sha256']==config['protocol_sha256'] and resume['schedule_sha256']==schedule_hash
            model.load_state_dict(resume['model'],strict=True); optimizer.load_state_dict(resume['optimizer'])
            curve=resume['curve']; step_start=resume['step']+1; best_state=resume['best_state']
            status=SelectionStop(**resume['selection_status']) if resume['selection_status'] else None
            gradient_seen=resume['gradient_nonzero_seen']; start_time-=resume['elapsed_seconds']
            payload=resume; step=resume['step']; full_train_nll=curve[-1]['full_train_nll']
            if status is not None and status.stopped(step):
                step_start=maximum+1
        for step in range(step_start,maximum+1):
            loss_value=None; grad_norm=None
            if step:
                model.train(); idx=batches[step-1]; optimizer.zero_grad(set_to_none=True)
                logits=model(train.cone_drive[idx],observed_counts=train.spike_events[idx]).logits
                loss=expected_bernoulli_nll(logits,train.spike_events[idx],train.valid_mask[idx])
                loss.backward()
                for name,p in model.named_parameters():
                    if p.requires_grad:
                        if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                            raise ValueError(f'nonfinite or absent gradient: {cell_id}/{seed}/{condition}/{name}')
                        gradient_seen[name] |= bool(torch.count_nonzero(p.grad))
                grad_norm=float(torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True))
                optimizer.step(); model.project_mechanism_parameters(); loss_value=float(loss.detach())
            if step%25 and step!=maximum:
                continue
            inner_nll=None; full_train_nll=None
            if validation is not None:
                inner_nll,_=evaluate(model,validation)
                status=SelectionStop(inner_nll,0,inner_nll,0) if status is None else status.observe(inner_nll,step)
                if status.best_step==step:
                    best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            elif step in (0,maximum):
                full_train_nll,_=evaluate(model,train)
            assert all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in fixed.items())
            row={'cell_id':cell_id,'seed':seed,'condition':condition,'phase':phase,'step':step,
                 'sampled_train_batch_nll':loss_value,'inner_validation_nll':inner_nll,
                 'full_train_nll':full_train_nll,'gradient_norm_before_clip':grad_norm,
                 'best_step':status.best_step if status else maximum,'best_inner_nll':status.best_nll if status else None,
                 'alpha':float(model.alpha.detach()),'elapsed_seconds':time.perf_counter()-start_time}
            curve.append(row)
            payload={'cell_id':cell_id,'seed':seed,'condition':condition,'phase':phase,'step':step,
                     'best_step':status.best_step if status else maximum,'model':model.state_dict(),
                     'optimizer':optimizer.state_dict(),'curve':curve,'best_state':best_state,
                     'selection_status':asdict(status) if status else None,'rms':rms,
                     'upstream_initial_sha256':upstream_hash,'initial_state':initial_state,
                     'schedule_sha256':schedule_hash,'protocol_sha256':config['protocol_sha256'],
                     'gradient_nonzero_seen':gradient_seen,'elapsed_seconds':time.perf_counter()-start_time,
                     'model_config':metadata['model_config'],'cone_positions_degs':metadata['cone_positions_degs'],
                     'cell_positions_degs':metadata['cell_positions_degs'],'cell_types':metadata['cell_types'],
                     'polarities':metadata['polarities'],'backend':model.backend.value,
                     'trainable_parameters':sum(p.numel() for p in params),'fixed_buffers_unchanged':True}
            atomic_torch_save(payload,latest)
            if status is not None and status.stopped(step):
                break
        if validation is not None:
            model.load_state_dict(best_state,strict=True)
            replay,_=evaluate(model,validation)
            assert replay==status.best_nll
            payload['best_replay_inner_nll']=replay
            payload['model_stage']='inner_best'
            payload['optimizer_state_stage']='inner_stop'
        else:
            payload['refit_train_nll']=full_train_nll
            payload['model_stage']='refit_final'
            payload['optimizer_state_stage']='refit_final'
            assert step==maximum
        payload['actually_changed']={k:not torch.equal(initial_state[k],p.detach()) for k,p in model.named_parameters() if p.requires_grad}
        payload['optimizer_steps']=sorted({int(s['step']) for s in optimizer.state.values()})
        assert payload['optimizer_steps']==([step] if step else [])
        atomic_torch_save(payload,destination)
        results[condition]={'step':step,'best_step':payload['best_step']}
        print(phase,cell_id,seed,condition,'stop',step,'selected',payload['best_step'],flush=True)
    return {'cell_id':cell_id,'seed':seed,'phase':phase,'conditions':results}


def combine_training(out: Path) -> None:
    rows=[]
    for path in sorted((out/'checkpoints').glob('*/*/*/inner.pt')):
        rows.extend(torch.load(path,weights_only=True)['curve'])
    for path in sorted((out/'checkpoints').glob('*/*/*/refit.pt')):
        rows.extend(torch.load(path,weights_only=True)['curve'])
    if rows:
        destination=out/'training_per_seed.csv'; temporary=destination.with_suffix('.pending')
        with temporary.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        temporary.replace(destination)


def run(out: Path, workers: int) -> None:
    config=load_json(out/'checkpoints/protocol.json')
    assert sha(out/'PROTOCOL.md')==config['protocol_sha256']
    for path,digest in config['source_hashes'].items():
        assert sha(ROOT/path)==digest,path
    for ref in config['cells'].values():
        assert sha(Path(ref['input_path']))==ref['input_sha256']
        assert sha(Path(ref['metadata_path']))==ref['metadata_sha256']
    jobs=[(cell,seed) for cell in config['cells'] for seed in SEEDS]
    for phase in ('inner','refit'):
        if phase=='refit':
            selected=list((out/'checkpoints').glob('*/*/*/inner.pt'))
            assert len(selected)==198
            lock={str(p.relative_to(out)):sha(p) for p in selected}
            save_json(out/'checkpoints/selection_lock.json',{'protocol_sha256':config['protocol_sha256'],
                      'selected_before_refit_and_development':True,'files':lock})
        completed=0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(fit_job,str(out),cell,seed,phase):(cell,seed) for cell,seed in jobs}
            for future in as_completed(futures):
                result=future.result(); completed+=1
                print('PROGRESS',phase,completed,'/66 cell-seed groups',flush=True)
                save_json(out/'checkpoints/progress.json',{'phase':phase,'completed_cell_seed_groups':completed,
                          'total_cell_seed_groups':66,'latest':result,'updated_utc':datetime.now(timezone.utc).isoformat()})
                combine_training(out)
        assert len(list((out/'checkpoints').glob(f'*/*/*/{phase}.pt')))==198
    save_json(out/'checkpoints/training_complete.json',{'inner_fits':198,'fresh_refits':198,
              'development_used_for_selection':False,'completed_utc':datetime.now(timezone.utc).isoformat()})
    print('ALL MODEL SELECTION AND FRESH REFITS COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('freeze','run'))
    parser.add_argument('--initialization',choices=('fresh','continuation'),default='fresh')
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--out',type=Path,default=ROOT/'output/experiments/retipath_spatial_ei_phase2_population')
    args=parser.parse_args()
    if args.mode=='freeze':
        out=args.out
        if out.exists(): out=out/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        freeze(out,args.initialization)
    else:
        run(args.out,args.workers)
