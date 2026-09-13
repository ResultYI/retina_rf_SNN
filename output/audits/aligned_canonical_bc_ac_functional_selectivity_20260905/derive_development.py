from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BIAS = ROOT / 'output/audits/macaque_pathway_bias_recalibration_20260905'
OLD = ROOT / 'output/audits/macaque_aligned_heldout_pathway_evaluation_20260905'
IDENTITIES = [r for r in json.loads((OLD / 'evidence_manifest.json').read_text())['checkpoint_identity'] if r['model'] == 'aligned']
PRIMARY = (ROOT / IDENTITIES[0]['path']).parents[2]
DERIVED = OUT / 'derived_recalibrated_development'
sys.path.insert(0, str(PRIMARY))
sys.path.insert(0, str(ROOT))
from preflight import ADAPTER, MOVIE, alignment, factory, folder, load_data
from data.schottdorf_lee_multirecording import load_schottdorf_movie_drive
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import evaluate_retinal_model
from models.mechanistic_retina.contracts import PathwayClamp
from training.mechanistic_retina.losses import expected_bernoulli_nll

CONDITIONS = {'BC': ('direct-BC', 'direct_BC_off', frozenset({PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT})),
              'AC': ('AC', 'AC_off', frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}))}


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def thash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def nll(z: torch.Tensor, y: torch.Tensor) -> float:
    return float((torch.nn.functional.softplus(z)-y*z).mean())


def main() -> None:
    assert not DERIVED.exists()
    initial = json.loads((OUT / 'continuation_initial_manifest.json').read_text())
    assert sha(OUT / 'PROTOCOL.md') == json.loads((OUT / 'protocol_hash.json').read_text())['sha256']
    for path, digest in initial['input_sha256'].items():
        assert sha(ROOT / path) == digest, path
    bias_path = BIAS / 'bias_fit_per_cell.csv'
    bias_lock = json.loads((BIAS / 'bias_fit_lock.json').read_text())
    assert sha(bias_path) == bias_lock['bias_fit_per_cell_sha256']
    fits = read_csv(bias_path)
    lookup = {(r['cell_id'], r['condition']): r for r in fits}
    assert len(lookup) == len(fits) == 88
    assert len([r for r in fits if r['condition'] in ('direct-BC', 'AC')]) == 44
    torch.set_num_threads(2)
    assert str(torch.__version__) == '2.6.0+cpu'
    state = dict(status='IN_PROGRESS', all_passed=False, cells=[], started_utc=now(), fitting_performed=False, model_inference_rerun=True)
    write_json(OUT / 'preflight.json', state)
    try:
        movie = load_schottdorf_movie_drive(MOVIE, ADAPTER)
        originals = {r['cell_id']: r for r in json.loads((PRIMARY / 'results.json').read_text())['cells']}
        cells, stimuli, rows, provenance = {}, {}, [], []
        for identity in IDENTITIES:
            cid = identity['cell_id']
            cp_path = ROOT / identity['path']
            assert sha(cp_path) == identity['sha256']
            cp = torch.load(cp_path, weights_only=True)
            data = load_data(cid, movie)
            split = data.validation
            pred_path, raw_path = cp_path.parent / 'validation-predictions.pt', cp_path.parent / 'causal-replay-tensors.pt'
            pred, archived = torch.load(pred_path, weights_only=True), torch.load(raw_path, weights_only=True)
            native_input = torch.load(ROOT / '.omo/evidence/compact_causal_cnn_baseline/inputs' / (cid.replace('#', '_')+'.pt'), weights_only=True)['validation']
            assert all(torch.equal(getattr(split, k), native_input[k]) for k in ('cone_drive', 'spike_counts', 'spike_events', 'valid_mask'))
            assert torch.equal(pred['target'], split.spike_events) and torch.equal(pred['valid_mask'], split.valid_mask)
            assert tuple(pred['source_image_ids']) == split.source_image_ids and tuple(pred['trial_indices']) == split.trial_indices
            assert tuple(native_input['source_image_ids']) == split.source_image_ids and tuple(native_input['trial_indices']) == split.trial_indices
            assert cp['cell_id'] == cid and torch.equal(cp['cell_positions_degs'], alignment(cid))
            assert torch.equal(cp['cone_positions_degs'], data.cone_positions_degs)
            model = factory(cid, data, cp['cell_positions_degs'])
            model.load_state_dict(cp['model'], strict=True)
            metrics, normal = evaluate_retinal_model(model, split)
            assert torch.equal(normal, pred['logits_trained']) and torch.equal(normal, archived['normal']['logits'])
            assert metrics.population_nll == originals[cid]['validation_nll_trained']
            mask, target = split.valid_mask, split.spike_events
            cell = dict(normal=normal, raw={}, recalibrated={}, biases={}, target=target, valid_mask=mask, source_image_ids=split.source_image_ids, trial_indices=split.trial_indices)
            checks = []
            for label, (bias_key, raw_key, clamps) in CONDITIONS.items():
                with torch.no_grad():
                    raw = model.forward_sequence(split.cone_drive, observed_counts=target, clamps=clamps).logits
                reference = archived[raw_key]['logits']
                assert torch.equal(raw, reference), f'STOP raw replay {cid} {label}'
                assert float(expected_bernoulli_nll(raw, target, mask)) == float(expected_bernoulli_nll(reference, pred['target'], pred['valid_mask']))
                fit = lookup[cid, bias_key]
                bias = float(fit['fitted_bias'])
                assert float.fromhex(bias.hex()) == bias and fit['condition'] == bias_key and fit['cell_id'] == cid
                recal = raw.double()+bias
                reconstructed = reference.double()+float(lookup[cid, bias_key]['fitted_bias'])
                assert torch.equal(recal, reconstructed)
                error = float(((recal-raw.double())-bias).abs().max())
                bound = 4*torch.finfo(torch.float64).eps*max(1., float(raw.abs().max()), abs(bias))
                assert error <= bound and torch.isfinite(recal).all()
                cell['raw'][label], cell['recalibrated'][label], cell['biases'][label] = reference, recal, bias
                rows.append(dict(cell_id=cid, pathway=bias_key, normal_nll=nll(normal[mask].double(), target[mask].double()), raw_off_nll=nll(raw[mask].double(), target[mask].double()),
                    recalibrated_nll=nll(recal[mask], target[mask].double()), frozen_bias=bias, frozen_bias_hex=bias.hex(), scored_bins=int(mask.sum()),
                    provenance='NEW_DETERMINISTIC_DERIVED_ARTIFACT', fitting_performed=False))
                checks.append(dict(pathway=bias_key, raw_bitwise_exact=True, raw_nll_exact=True, bias_hash_exact=True, addition_bitwise_exact=True, subtraction_max_error=error, subtraction_rounding_bound=bound))
                provenance.append(dict(cell_id=cid, pathway=bias_key, source_raw_logit_path=raw_path.relative_to(ROOT).as_posix(), source_raw_logit_sha256=sha(raw_path), raw_tensor_key=raw_key+'.logits', raw_tensor_sha256=thash(reference),
                    frozen_bias_source_path=bias_path.relative_to(ROOT).as_posix(), frozen_bias_source_sha256=sha(bias_path), bias_lock_sha256=sha(BIAS / 'bias_fit_lock.json'), bias_value=bias, bias_hex=bias.hex(),
                    target_mask_source=pred_path.relative_to(ROOT).as_posix(), target_mask_source_sha256=sha(pred_path), target_sha256=thash(target), mask_sha256=thash(mask), checkpoint_path=identity['path'], checkpoint_sha256=identity['sha256']))
            assert all(torch.equal(v, cp['model'][k]) for k, v in model.state_dict().items()) and all(p.grad is None for p in model.parameters())
            bc, ac = cp['model']['feature_bank.bc_support'][0], cp['model']['feature_bank.ac_support'][0]
            assert ((bc == 0) | (bc == 1)).all() and ((ac == 0) | (ac == 1)).all()
            assert (bc <= ac).all() and bc.sum() > 0 and (ac-bc).sum() > 0
            stimuli[cid] = dict(cone_drive=split.cone_drive, valid_mask=mask, source_image_ids=split.source_image_ids, trial_indices=split.trial_indices,
                bc=bc, ac=ac, center=cp['cell_positions_degs'], cone_positions=cp['cone_positions_degs'])
            cells[cid] = cell
            state['cells'].append(dict(cell_id=cid, strict_load=True, normal_input_target_mask_order_logits_nll_exact=True, state_unchanged=True, checks=checks, passed=True))
            write_json(OUT / 'preflight.json', state)
            print('RAW REPLAY + DERIVATION', len(cells), cid, flush=True)
        assert len(cells) == 22 and len(rows) == 44
        for p, h in initial['input_sha256'].items():
            assert sha(ROOT / p) == h
        DERIVED.mkdir()
        torch.save(dict(provenance='NEW_DETERMINISTIC_DERIVED_ARTIFACT', cells=cells), DERIVED / 'recalibrated_development_logits.pt')
        write_csv(DERIVED / 'recalibrated_development_nll.csv', rows)
        torch.save(stimuli, OUT / 'development_stimulus.pt')
        write_json(DERIVED / 'derivation_manifest.json', dict(classification='NEW_DETERMINISTIC_DERIVED_ARTIFACT', formula='recalibrated_logits = raw_off_logits + frozen_training_bias',
            dtype='raw float32 promoted to float64 before addition', fitting_performed=False, development_target_used_for_bias_estimation=False, model_inference_rerun=True, model_parameters_modified=False,
            created_utc=now(), branch=initial['branch'], HEAD=initial['HEAD'], protocol_sha256=sha(OUT / 'PROTOCOL.md'), sources=provenance,
            output_sha256={p.name: sha(p) for p in DERIVED.iterdir() if p.is_file()}, original_inputs_unchanged=True))
        write_json(DERIVED / 'derivation_lock.json', dict(locked_utc=now(), sha256={p.name: sha(p) for p in DERIVED.iterdir() if p.is_file()}))
        state.update(status='VERIFIED DERIVATION INPUTS — development recalibrated logits newly derived deterministically from frozen training-only biases', all_passed=True,
            classification='NEW_DETERMINISTIC_DERIVED_ARTIFACT', historical_recalibrated_replay=False, completed_utc=now(), derived_cells=22, derived_pathways=44)
        write_json(OUT / 'preflight.json', state)
    except Exception as exc:
        state.update(status='UNVERIFIED', error=str(exc), traceback=traceback.format_exc())
        write_json(OUT / 'preflight.json', state)
        raise


if __name__ == '__main__':
    main()
