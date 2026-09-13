# /// script
# requires-python = ">=3.12"
# dependencies = ["torch"]
# ///
# How to run: D:/anaconda/python.exe -B -u normalize_sigmoid_execution.py
from pathlib import Path
import hashlib
import json
import torch


def main() -> None:
    out=Path(__file__).resolve().parent
    assert not (out/'primary_lock.json').exists()
    assert not (out/'sigmoid_execution_correction.json').exists()
    torch.set_num_threads(2)
    rows=[]
    for path in sorted((out/'predictions').glob('*_primary.pt')):
        saved=torch.load(path,weights_only=True)
        logits=saved['logits']
        assert torch.equal(logits,logits[:1].repeat(6,1))
        corrected=torch.stack([row.sigmoid() for row in logits])
        assert torch.equal(corrected,corrected[:1].repeat(6,1))
        rows.append({'path':path.name,'before_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'maximum_probability_roundoff':float((corrected-saved['probabilities']).abs().max()),'changed_entries':int((corrected!=saved['probabilities']).sum()),'logits_unchanged':True})
        saved['probabilities']=corrected
        torch.save(saved,path)
        rows[-1]['after_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    for backend in ('cpu','cnn'):
        path=out/f'primary_{backend}_inference_checks.json'
        checks=json.loads(path.read_text())
        for row in checks['checks']:
            row['sha256']=hashlib.sha256((out/row['output']).read_bytes()).hexdigest()
            row['probability_repeat_exact']=True
            row['sigmoid_execution']='PER_REPEAT_IDENTICAL_SHAPE'
        path.write_text(json.dumps(checks,indent=2))
    with (out/'sigmoid_execution_correction.json').open('x') as stream:
        json.dump({'artifact_class':'NEW_DETERMINISTIC_DERIVED_ARTIFACT','before_primary_metrics_lock':True,'reason':'Identical logits had vectorized sigmoid lane/tail roundoff; each repeat now uses identical one-dimensional torch sigmoid execution. No logits, parameters, estimators, alignment, probability calibration or model selection changed.','checks':rows},stream,indent=2)
    print('Probability trace identity corrected and recorded',len(rows))


if __name__=='__main__': main()
