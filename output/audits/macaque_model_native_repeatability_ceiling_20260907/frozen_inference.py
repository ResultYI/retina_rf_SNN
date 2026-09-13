# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy", "pydantic"]
# ///
# How to run: D:/anaconda/python.exe -B -u frozen_inference.py primary cpu
# CNN: D:/anaconda/envs/snn_env/python.exe -B -u frozen_inference.py primary cnn
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Final

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]
sys.path.insert(0,str(ROOT))
from baselines.center_surround_ln import CenterSurroundLN
from baselines.compact_causal_cnn import CompactCausalCNN
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina


def sha(path: Path) -> str:
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main() -> None:
    stage, backend=sys.argv[1:]
    assert stage in {'primary','conditional'} and backend in {'cpu','cnn'}
    assert json.loads((OUT/'preflight.json').read_text())['all_passed']
    protocol=json.loads((OUT/'protocol_hash.json').read_text())['sha256']
    assert sha(OUT/'PROTOCOL.md')==protocol
    if stage=='conditional':
        frozen=json.loads((OUT/'primary_lock.json').read_text())
        assert all(sha(OUT/name)==digest for name,digest in frozen['output_sha256'].items())
    torch.set_num_threads(2)
    use_cuda=backend=='cnn'
    if use_cuda:
        runtime=json.loads((ROOT/'.omo/evidence/compact_causal_cnn_baseline/runtime.json').read_text())
        assert str(torch.__version__)==runtime['torch'] and torch.cuda.get_device_name()==runtime['gpu']
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
    else:
        assert str(torch.__version__).startswith('2.6.0')
    inputs=torch.load(OUT/'inference_inputs.pt',weights_only=True,map_location='cpu')
    lock=json.loads((OUT/'input_lock.json').read_text())
    targets=lock['checkpoint_identity']
    drive=inputs['movie_sequences'].repeat_interleave(6,dim=0).to('cuda' if use_cuda else 'cpu')
    assert drive.shape==(360,150,289)
    output_dir=OUT/'predictions'
    output_dir.mkdir(exist_ok=True)
    checks=[]
    for cid,data in inputs['cells'].items():
        events=data['events'].reshape(6,60,150,1).permute(1,0,2,3).reshape(360,150,1).to(drive.device)
        observed=events if stage=='conditional' else torch.zeros_like(events)
        if stage=='primary': assert torch.count_nonzero(observed)==0
        labels=['CNN'] if use_cuda else ['aligned','LN']
        for label in labels:
            destination=output_dir/f'{cid.replace("#","_")}_{label}_{stage}.pt'
            assert not destination.exists(),destination
            ref=next(r for r in targets if r['cell_id']==cid and r['model']==label)
            path=ROOT/ref['path']
            assert sha(path)==ref['sha256']
            checkpoint=torch.load(path,weights_only=True,map_location='cpu')
            assert checkpoint['cell_id']==cid
            match label:
                case 'aligned':
                    torch.manual_seed(checkpoint['seed'])
                    model=build_mechanistic_retina(MechanisticRetinaConfig(**checkpoint['model_config']),checkpoint['cone_positions_degs'],checkpoint['cell_positions_degs'],tuple(checkpoint['cell_types']),tuple(checkpoint['polarities']))
                case 'LN':
                    model=CenterSurroundLN(checkpoint['history']['dt_ms'],checkpoint['history']['tau_ms'],checkpoint['seed'])
                case 'CNN':
                    model=CompactCausalCNN(checkpoint['history']['dt_ms'],checkpoint['history']['tau_ms'],61001).cuda()
            model.load_state_dict(checkpoint['model'],strict=True)
            model.eval()
            with torch.no_grad():
                chunks=[]
                for start in range(0,360,8):
                    if label=='aligned':
                        logits=model.forward_sequence(drive[start:start+8],observed_counts=observed[start:start+8]).logits
                    else:
                        logits=model(drive[start:start+8],observed[start:start+8])
                    chunks.append(logits.cpu())
                logits=torch.cat(chunks).reshape(60,6,150).permute(1,0,2).reshape(6,9000)
                probabilities=torch.stack([row.sigmoid() for row in logits])
            assert torch.isfinite(logits).all()
            repeat_exact=all(torch.equal(logits[0],logits[r]) for r in range(1,6))
            if stage=='primary': assert repeat_exact,'Zero-history repeated predictions differ'
            assert all(torch.equal(v.cpu(),checkpoint['model'][k]) for k,v in model.state_dict().items())
            assert all(p.grad is None for p in model.parameters())
            assert sha(path)==ref['sha256']
            torch.save({'artifact_class':'NEW_DETERMINISTIC_DERIVED_ARTIFACT','cell_id':cid,'model':label,'contract':'STIMULUS_ONLY_ZERO_HISTORY' if stage=='primary' else 'FORMAL_CONDITIONAL_HISTORY','protocol_sha256':protocol,'checkpoint_sha256':ref['sha256'],'logits':logits,'probabilities':probabilities},destination)
            checks.append({'cell_id':cid,'model':label,'checkpoint_sha256':ref['sha256'],'strict_load':True,'state_unchanged':True,'no_gradients':True,'repeat_predictions_exact':repeat_exact,'zero_history_input':stage=='primary','torch':str(torch.__version__),'device':str(drive.device),'batch_size':8,'output':destination.relative_to(OUT).as_posix(),'sha256':sha(destination)})
            print(stage,backend,len(checks),cid,label,flush=True)
    with (OUT/f'{stage}_{backend}_inference_checks.json').open('x') as stream:
        json.dump({'artifact_class':'NEW_DETERMINISTIC_DERIVED_ARTIFACT','protocol_sha256':protocol,'checks':checks,'training':0,'parameter_fitting':0},stream,indent=2)


if __name__=='__main__': main()
