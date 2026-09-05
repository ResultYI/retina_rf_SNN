#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0"]
# ///
# How to run: D:/anaconda/python.exe -B mixer_audit.py from the repository root.
# Body below was executed through stdin; no files are written.
from pathlib import Path
from typing import Final
import hashlib
import json
import torch
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.optimizer import build_phase1_optimizer

ROOT: Final = Path('output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830')
PATHS: Final = sorted(ROOT.glob('cells/*/model-trained.pt'),key=lambda p:tuple(map(int,p.parent.name.split('_'))))
SOURCES: Final = sorted(Path('models/mechanistic_retina').glob('*.py')) + [Path('training/mechanistic_retina/optimizer.py')]
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
pre_hashes = {str(p):sha256(p) for p in PATHS + SOURCES}
print('PRE_CHECKPOINT_HASHES',json.dumps({p.parent.name:pre_hashes[str(p)] for p in PATHS},sort_keys=True),flush=True)
print('PRE_SOURCE_HASHES',json.dumps({str(p):pre_hashes[str(p)] for p in SOURCES},sort_keys=True),flush=True)
assert len(PATHS) == 22
rows=[]
key_sets=[]
torch.set_num_threads(1)
for path in PATHS:
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    key_sets.append(tuple(checkpoint))
    config=MechanisticRetinaConfig(**checkpoint['model_config'])
    model=build_mechanistic_retina(config,checkpoint['cone_positions_degs'],checkpoint['cell_positions_degs'],tuple(checkpoint['cell_types']),tuple(checkpoint['polarities']))
    load=model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    parameters=dict(model.named_parameters())
    buffers=dict(model.named_buffers())
    raw=model.shared_subunits.raw_connections
    optimizer=build_phase1_optimizer(model,learning_rate=checkpoint['training_contract']['learning_rate'])
    member=any(raw is p for group in optimizer.param_groups for p in group['params'])
    cone_count=checkpoint['cone_positions_degs'].shape[0]
    cell_count=checkpoint['cell_positions_degs'].shape[0]
    t=torch.arange(32,dtype=torch.float32).reshape(1,32,1)
    c=torch.arange(cone_count,dtype=torch.float32).reshape(1,1,cone_count)
    stimulus=0.3*torch.sin(0.29*t+0.017*c)+0.15*torch.cos(0.11*t-0.031*c)
    history=torch.zeros(1,32,cell_count)
    saved=raw.detach().clone()
    differences=[]
    matrices=[]
    with torch.no_grad():
        baseline=model(stimulus,observed_counts=history).logits.clone()
        matrix=model.shared_subunits.connection_matrix().clone()
        for delta in (0.2,-0.2,2.0):
            raw.copy_(saved+delta)
            changed=model(stimulus,observed_counts=history).logits
            differences.append(float((changed-baseline).abs().max()))
            assert torch.equal(changed,baseline)
            matrices.append(model.shared_subunits.connection_matrix().tolist())
            raw.copy_(saved)
            state=model.state_dict()
            assert set(state)==set(checkpoint['model'])
            assert all(torch.equal(value,checkpoint['model'][name]) for name,value in state.items())
    row={'cell':checkpoint['cell_id'],'architecture':str(config.architecture_mode),'N':cell_count,'parameter':'shared_subunits.raw_connections' in parameters,'buffer':'shared_subunits.raw_connections' in buffers,'requires_grad':raw.requires_grad,'optimizer_member':member,'optimizer_parameter_count':sum(len(g['params']) for g in optimizer.param_groups),'optimizer_state_entries':len(optimizer.state),'raw_saved':saved.tolist(),'edge_index':model.shared_subunits.edge_index.tolist(),'matrix':matrix.tolist(),'logit_range':[float(baseline.min()),float(baseline.max())],'delta_logit_maxabs':differences,'perturbed_matrices':matrices,'state_restored':True,'strict_load':not load.missing_keys and not load.unexpected_keys,'optimizer_like_fields':[k for k in checkpoint if 'optim' in k.lower()]}
    rows.append(row)
    print('CELL',json.dumps(row,sort_keys=True),flush=True)
post_hashes={str(p):sha256(p) for p in PATHS + SOURCES}
changed_paths=[p for p in pre_hashes if pre_hashes[p]!=post_hashes[p]]
aggregate={'cells':len(rows),'n1':sum(r['N']==1 for r in rows),'all_raw_buffers':all(r['buffer'] and not r['parameter'] for r in rows),'raw_requires_grad_count':sum(r['requires_grad'] for r in rows),'optimizer_member_count':sum(r['optimizer_member'] for r in rows),'optimizer_top_level_field_count':sum(bool(r['optimizer_like_fields']) for r in rows),'perturbation_runs':sum(len(r['delta_logit_maxabs']) for r in rows),'maximum_logit_diff':max(v for r in rows for v in r['delta_logit_maxabs']),'all_state_restored':all(r['state_restored'] for r in rows),'changed_source_or_checkpoint_paths':changed_paths,'all_same_top_level_keys':len(set(key_sets))==1,'top_level_keys':list(key_sets[0]),'source_files_hashed':len(SOURCES),'checkpoint_files_hashed':len(PATHS)}
print('AGGREGATE',json.dumps(aggregate,sort_keys=True),flush=True)
assert not changed_paths
