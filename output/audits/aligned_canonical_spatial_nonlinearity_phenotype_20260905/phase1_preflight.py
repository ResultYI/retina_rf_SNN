from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT / 'output/audits/macaque_aligned_heldout_pathway_evaluation_20260905'
OLD_MANIFEST = json.loads((OLD / 'evidence_manifest.json').read_text())
IDENTITIES = [r for r in OLD_MANIFEST['checkpoint_identity'] if r['model'] in ('aligned', 'LN', 'CNN')]
ROOTS = {r['model']: (ROOT / r['path']).parents[2] for r in IDENTITIES}
PRIMARY, LN, CNN = (ROOTS[k] for k in ('aligned', 'LN', 'CNN'))
sys.path.insert(0, str(PRIMARY))
sys.path.insert(0, str(ROOT))
from preflight import ADAPTER, MOVIE, alignment, factory, folder, load_data
from baselines.center_surround_ln import CenterSurroundLN
from data.schottdorf_lee_multirecording import load_schottdorf_movie_drive
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import evaluate_retinal_model
from training.mechanistic_retina.center_surround_ln import evaluate_center_surround_ln
from training.mechanistic_retina.losses import expected_bernoulli_nll


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    assert not (OUT / 'preflight.json').exists()
    assert sha(OUT / 'PROTOCOL.md') == json.loads((OUT / 'protocol_hash.json').read_text())['sha256']
    initial = json.loads((OUT / 'evidence_manifest.initial.json').read_text())
    for p, digest in initial['input_sha256'].items():
        assert sha(ROOT / p) == digest, p
    torch.set_num_threads(2)
    assert str(torch.__version__) == '2.6.0+cpu'
    rows: list[dict] = []
    status = dict(status='IN_PROGRESS', all_passed=False, cells=rows, started_utc=now(), training=0, parameter_fitting=0)
    save_json('preflight.json', status)
    try:
        source = {label: {r['cell_id']: r for r in json.loads((root / 'results.json').read_text())['cells']} for label, root in ROOTS.items()}
        movie = load_schottdorf_movie_drive(MOVIE, ADAPTER)
        stimulus, outputs = {}, {}
        for cid in source['aligned']:
            data = load_data(cid, movie)
            split = data.validation
            bundle = torch.load(CNN / 'inputs' / (cid.replace('#', '_') + '.pt'), weights_only=True)['validation']
            assert all(torch.equal(getattr(split, k), bundle[k]) for k in ('cone_drive', 'spike_counts', 'spike_events', 'valid_mask'))
            assert tuple(bundle['source_image_ids']) == split.source_image_ids
            assert tuple(bundle['trial_indices']) == split.trial_indices
            outputs[cid] = dict(target=split.spike_events, valid_mask=split.valid_mask, source_image_ids=split.source_image_ids, trial_indices=split.trial_indices, logits={})
            for label, filename in (('aligned', 'model-trained.pt'), ('LN', 'ln-trained.pt')):
                cp_path = folder(cid, ROOTS[label]) / filename
                cp = torch.load(cp_path, weights_only=True)
                assert cp['cell_id'] == cid
                if label == 'aligned':
                    assert cp['revision'] == 4 and cp['stage'] == 'trained'
                    assert cp['steps'] == cp['training_contract']['fresh_full_train_refit_steps'] == source[label][cid]['best_step']
                    assert not cp['training_contract']['original_validation_used_for_selection']
                    assert torch.equal(cp['cell_positions_degs'], alignment(cid))
                    assert torch.equal(cp['cone_positions_degs'], data.cone_positions_degs)
                    model = factory(cid, data, cp['cell_positions_degs'])
                    model.load_state_dict(cp['model'], strict=True)
                    metrics, logits = evaluate_retinal_model(model, split)
                    expected = source[label][cid]['validation_nll_trained']
                    support = cp['model']['feature_bank.bc_support'][0]
                    assert ((support == 0) | (support == 1)).all() and support.sum() > 0
                    stimulus[cid] = dict(cone_drive=split.cone_drive, valid_mask=split.valid_mask, source_image_ids=split.source_image_ids, trial_indices=split.trial_indices,
                        support=support, center=cp['cell_positions_degs'], cone_positions=cp['cone_positions_degs'])
                else:
                    assert cp['best_step'] == cp['refit_steps'] == source[label][cid]['refit_steps']
                    model = CenterSurroundLN(cp['history']['dt_ms'], cp['history']['tau_ms'], cp['seed'])
                    model.load_state_dict(cp['model'], strict=True)
                    metrics, logits = evaluate_center_surround_ln(model, split)
                    expected = source[label][cid]['ln_nll']
                saved = torch.load(folder(cid, ROOTS[label]) / 'validation-predictions.pt', weights_only=True)
                checks = dict(strict_load=True, input_exact=True, target_exact=torch.equal(saved['target'], split.spike_events), mask_exact=torch.equal(saved['valid_mask'], split.valid_mask),
                    source_order_exact=tuple(saved['source_image_ids']) == split.source_image_ids, trial_order_exact=tuple(saved['trial_indices']) == split.trial_indices,
                    logits_exact=torch.equal(logits, saved['logits_trained']), nll_exact=metrics.population_nll == expected,
                    state_unchanged=all(torch.equal(v, cp['model'][k]) for k, v in model.state_dict().items()), no_gradients=all(p.grad is None for p in model.parameters()))
                rows.append(dict(cell_id=cid, model=label, checks=checks, passed=all(checks.values()), nll=metrics.population_nll, saved_nll=expected, max_logit_error=float((logits-saved['logits_trained']).abs().max()), checkpoint_sha256=sha(cp_path)))
                save_json('preflight.json', status)
                assert all(checks.values()), f'STOP exact replay: {cid} {label}'
                outputs[cid]['logits'][label] = logits
            print(f'CPU exact replay {len(stimulus)}/22 {cid}', flush=True)
        subprocess.run(['D:/anaconda/envs/snn_env/python.exe', '-B', '-u', str(OUT / 'replay_cnn_gpu.py')], cwd=ROOT, check=True)
        gpu = torch.load(OUT / 'cnn_development_replay.pt', weights_only=True)
        for cid, record in gpu.items():
            saved = torch.load(folder(cid, CNN) / 'validation-predictions.pt', weights_only=True)
            current = outputs[cid]
            value = float(expected_bernoulli_nll(record['logits'], current['target'], current['valid_mask']))
            checks = dict(strict_load=record['strict_load'], input_exact=True, target_exact=torch.equal(saved['target'], current['target']), mask_exact=torch.equal(saved['valid_mask'], current['valid_mask']),
                source_order_exact=tuple(saved['source_image_ids']) == current['source_image_ids'], trial_order_exact=tuple(saved['trial_indices']) == current['trial_indices'],
                logits_exact=record['logits_bitwise_equal'] and torch.equal(record['logits'], saved['logits_trained']), nll_exact=value == source['CNN'][cid]['cnn_nll'], state_unchanged=record['checkpoint_unchanged'])
            rows.append(dict(cell_id=cid, model='CNN', checks=checks, passed=all(checks.values()), nll=value, saved_nll=source['CNN'][cid]['cnn_nll'], max_logit_error=float((record['logits']-saved['logits_trained']).abs().max()), checkpoint_sha256=sha(folder(cid, CNN) / 'cnn-trained.pt')))
            save_json('preflight.json', status)
            assert all(checks.values()), f'STOP exact replay: {cid} CNN'
            current['logits']['CNN'] = record['logits']
        assert len(rows) == 66 and sum(int(s['valid_mask'].sum()) for s in stimulus.values()) == 65760
        torch.save(stimulus, OUT / 'development_stimulus.pt')
        torch.save(outputs, OUT / 'development_outputs.pt')
        status.update(status='PASSED', all_passed=True, completed_utc=now(), cells_count=22, model_pairs=66, scored_bins=65760)
        save_json('preflight.json', status)
    except Exception as exc:
        status.update(status='STOP_PREFLIGHT_FAILED', error=str(exc), traceback=traceback.format_exc(), stopped_utc=now())
        save_json('preflight.json', status)
        raise


if __name__ == '__main__':
    main()
