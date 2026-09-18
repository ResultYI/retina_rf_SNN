from __future__ import annotations

import time
from pathlib import Path
import torch

from common import (OUT, SEEDS, load_json, save_json, stage_data, build, compute_rms,
                    capacity_parameters, sha)
from training.mechanistic_retina.losses import expected_bernoulli_nll


def run() -> None:
    torch.set_num_threads(1)
    protocol = load_json(OUT / 'checkpoints/protocol.json')
    groups, results = set(), {}
    for cell, ref in protocol['cells'].items():
        if ref['group'] in groups:
            continue
        groups.add(ref['group'])
        data, _, mask = stage_data(ref, 'inner')
        metadata = torch.load(ref['metadata_path'], weights_only=True, mmap=True)
        old = torch.load(ref['A'][str(SEEDS[0])]['path'], weights_only=True, mmap=True)
        reference = build(metadata, SEEDS[0], 'A')
        reference = build(metadata, SEEDS[0], 'A', rms=compute_rms(reference, data, mask))
        with torch.no_grad():
            a = reference(data.cone_drive[:4], observed_counts=data.spike_events[:4]).logits
        for condition in ('B', 'C'):
            model = build(metadata, SEEDS[0], condition, ref['h1_rms']['inner'])
            rms = compute_rms(model, data, mask)
            model = build(metadata, SEEDS[0], condition, ref['h1_rms']['inner'], rms)
            fixed = {n: v.clone() for n, v in model.named_buffers()}
            times = []
            for _ in range(3):
                start = time.perf_counter()
                model.zero_grad(set_to_none=True)
                logits = model(data.cone_drive[:4], observed_counts=data.spike_events[:4]).logits
                loss = expected_bernoulli_nll(logits, data.spike_events[:4], data.valid_mask[:4])
                loss.backward()
                times.append(time.perf_counter()-start)
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in capacity_parameters(model))
            assert all(torch.equal(v, dict(model.named_buffers())[n]) for n, v in fixed.items())
            error = float((a-logits.detach()).abs().max()) if condition == 'B' else None
            if condition == 'B':
                torch.testing.assert_close(a, logits, atol=3e-6, rtol=2e-5)
            results[cell + '/' + condition] = {'real_inner_batch_finite_loss': float(loss.detach()),
                'finite_gradients_all_trainables': True, 'buffers_unchanged': True,
                'B_initial_logit_max_error_against_A': error, 'median_forward_backward_seconds': sorted(times)[1],
                'fixed_parameters': [n for n, _ in model.named_buffers() if n.startswith('input_basis.') or n == 'h1_rms'],
                'trainable_parameters': {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}}
    checks = load_json(OUT / 'correctness.json')
    checks['real_batch_preflight'] = results
    checks['preflight_sha256'] = sha(Path(__file__))
    checks['optimizer_updates_before_formal_training'] = 0
    save_json(OUT / 'correctness.json', checks)
    print('REAL BATCH CHECKS PASSED:', len(results), 'conditions across four types; zero optimizer updates', flush=True)


if __name__ == '__main__':
    run()
