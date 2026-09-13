from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import csv
import io

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from retipath_final_common import (
    ROOT, OUT, REG_CODE, SEEDS, application_imports, load_json, load_retipath, sha, write_once, verify_sources,
)
from retipath_phase2_analyze import csv_rows

application_imports()
from application_io import CLAMPS, CONDITIONS, GROUPS, Cell
from contract import Comparison
from run_application import check_clamp
from response_metrics import response_columns, Column, Key
import analyze_application
from comparison_metrics import Statistics, PairedSummary, ranking

DEST = OUT / 'brightness'


def infer_cell(cell: str, seed: int) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    safe = cell.replace('#', '_')
    destination = DEST / 'responses' / f'{safe}_{seed}.npz'
    record = destination.with_suffix('.json')
    if destination.exists():
        saved = load_json(record)
        assert saved['source_lock_sha256'] == sha(OUT / 'source_lock.json')
        assert saved['responses_sha256'] == sha(destination)
        return saved
    lock = load_json(OUT / 'source_lock.json')
    ref = load_json(OUT / 'model_registry.json')['cells'][cell]['RetiPath'][str(seed)]
    model, cp = load_retipath(ROOT / ref['path'])
    with np.load(OUT / 'registered_stimulus_bank.npz', allow_pickle=False) as bank:
        index = bank['cell_ids'].tolist().index(cell)
        patches = torch.from_numpy(bank['patches'][index].copy())
        pulse = torch.from_numpy(bank['pulse'].copy())
        history = torch.from_numpy(bank['history'].copy())
    drive = pulse[None, :, None] * patches.flatten(1)[:, None, :]
    logits = torch.empty(4, 166, 150)
    probabilities = torch.empty_like(logits)
    calls = 0
    with torch.inference_mode():
        for first in range(0, 166, 8):
            last = min(first + 8, 166)
            x, h = drive[first:last], history[first:last]
            normal = model.forward_sequence(x, observed_counts=h)
            for condition, clamps in enumerate(CLAMPS):
                output = normal if condition == 0 else model.forward_sequence(x, observed_counts=h, clamps=clamps)
                calls += 1
                check_clamp(normal, output, CONDITIONS[condition])
                assert all(torch.isfinite(value).all() for value in output.tensors())
                assert torch.count_nonzero(output.rgc_history_state) == 0
                assert torch.equal(output.spike_probability, torch.sigmoid(output.logits))
                logits[condition, first:last] = output.logits[..., 0]
                probabilities[condition, first:last] = output.spike_probability[..., 0]
    assert torch.equal(logits[:, 0], logits[:, 1])
    assert all(torch.equal(v, cp['model'][name]) for name, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert sha(ROOT / ref['path']) == ref['sha256']
    with destination.open('xb') as stream:
        np.savez_compressed(stream, logits=logits.numpy(), probability=probabilities.numpy())
    result = {'cell_id': cell, 'seed': seed, 'model': 'RetiPath', 'source_lock_sha256': sha(OUT / 'source_lock.json'),
        'checkpoint_sha256': ref['sha256'], 'responses_sha256': sha(destination), 'forward_calls': calls,
        'conditions': list(CONDITIONS), 'clamp_contracts_verified': True, 'identical_zero_controls': True,
        'finite_outputs': True, 'observed_history_zero': True, 'parameters_unchanged': True, 'parameter_gradients': 0,
        'registration_center': [lock['registration'][index]['center_x_deg'], lock['registration'][index]['center_y_deg']]}
    write_once(record, result)
    print(f'BRIGHTNESS {cell} seed {seed}: frozen bank and four pathway conditions', flush=True)
    return result


def rank_summary(folder, seed) -> list[dict]:
    results = []
    for family in ('mach', 'sbc'):
        rows = list(csv.DictReader((folder / f'{family}_summary.csv').open(encoding='utf-8')))
        for quantity in ('logit', 'probability'):
            for group in GROUPS:
                subset = [r for r in rows if r['quantity'] == quantity and r['group'] == group and r['level'] == 'full_grid']
                wrapped = []
                for row in subset:
                    key = Key(quantity, row['level'], row['endpoint'], row['condition'], row['metric'],
                              compared_condition=row['compared_condition'])
                    n = int(row['n_cells'])
                    mean = float(row['mean'])
                    negative, positive, zero = int(row['negative']), int(row['positive']), int(row['zero'])
                    same = positive if mean > 1e-9 else negative if mean < -1e-9 else zero
                    stats = Statistics(mean, float(row['median']), float(row['ci_low']), float(row['ci_high']),
                        negative, zero, positive, n, same, same / n, 0.0, row['direction_stable'] == 'True')
                    wrapped.append(PairedSummary(family, group, key, stats, stats, 0.0, 0.0, 0.0))
                order = ranking(wrapped, registered=True)
                results.append({'model': 'RetiPath', 'seed': seed, 'family': family, 'quantity': quantity, 'group': group,
                    'order': ' > '.join(order.order), 'paired_CI_winner': order.winner,
                    'direct_BC_greatest': order.direct_bc_greatest, 'numerical_tie': order.numerical_tie})
    return results


def main() -> None:
    lock = verify_sources()
    DEST.mkdir(exist_ok=True)
    (DEST / 'responses').mkdir(exist_ok=True)
    checks = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(infer_cell, cell, seed) for cell in lock['cells'] for seed in SEEDS]
        for future in as_completed(futures):
            checks.append(future.result())
    assert len(checks) == 66 and sum(r['forward_calls'] for r in checks) == 5544
    comparisons = tuple(Comparison(**p) for p in lock['stimulus_comparisons'])
    old = load_json(REG_CODE / 'frozen_registration.json')
    original_cells = tuple(Cell(**r) for r in old['cells'])
    all_columns = {}
    rankings = []
    with threadpool_limits(limits=2):
        for seed in SEEDS:
            current = []
            for cell in lock['cells']:
                with np.load(DEST / 'responses' / f"{cell.replace('#', '_')}_{seed}.npz", allow_pickle=False) as arrays:
                    current.append({name: arrays[name].astype(np.float64) for name in ('logits', 'probability')})
            columns = {'mach': [], 'sbc': []}
            for quantity, stored in [('logit', 'logits'), ('probability', 'probability')]:
                raw = np.stack([row[stored] for row in current])
                for family in ('Mach', 'SBC'):
                    columns[family.lower()].extend(response_columns(raw, comparisons, quantity, family))
            folder = DEST / str(seed)
            folder.mkdir(exist_ok=False)
            analyze_application.OUT = folder
            with redirect_stdout(io.StringIO()):
                for family in ('mach', 'sbc'):
                    analyze_application.write_family(family, tuple(replace(c, seed=seed) for c in original_cells), tuple(columns[family]))
            all_columns[seed] = columns
            rankings.extend(rank_summary(folder, seed))
            print(f'BRIGHTNESS METRICS: seed {seed}, original endpoint/grid/bootstrap definitions', flush=True)
        folder = DEST / 'seed_aggregate'
        folder.mkdir(exist_ok=False)
        analyze_application.OUT = folder
        for family in ('mach', 'sbc'):
            per_seed = [all_columns[seed][family] for seed in SEEDS]
            assert all([c.key for c in columns] == [c.key for c in per_seed[0]] for columns in per_seed)
            means = tuple(Column(c.key, np.mean([per_seed[k][i].values for k in range(3)], axis=0))
                          for i, c in enumerate(per_seed[0]))
            with redirect_stdout(io.StringIO()):
                analyze_application.write_family(family, original_cells, means)
        rankings.extend(rank_summary(folder, 'aggregate'))
    csv_rows(DEST / 'pathway_rankings.csv', rankings)
    verify_sources()
    write_once(DEST / 'complete.json', {'status': 'VERIFIED', 'models': 66, 'stimuli_per_model': 166,
        'pathway_conditions': list(CONDITIONS), 'forward_calls': 5544, 'checks': checks,
        'model_name': 'RetiPath', 'training_updates': 0, 'new_stimuli': 0, 'centers_changed': False,
        'seed_aggregation': 'Average each already-defined per-checkpoint effect within cell, then bootstrap biological cells.'})
    print('BRIGHTNESS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
