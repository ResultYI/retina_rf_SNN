from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import (ROOT, DEST, BASELINE, REGISTRY, SEEDS, digest, read_json, exclusive_json,
                    evaluate, utc, write_csv)
from data_adapter import make_bank
from model import TypeSharedRetiPath
from retipath_final_prediction import load_split
from retipath_final_common import identity


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def stats(values: np.ndarray) -> dict:
    indices = np.random.default_rng(20260908).integers(0, len(values), (100000, len(values)))
    lo, hi = np.quantile(values[indices].mean(-1), [.025, .975])
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'wins': int((values < -1e-7).sum()), 'losses': int((values > 1e-7).sum()),
            'ties': int((np.abs(values) <= 1e-7).sum()), 'ci_low': float(lo), 'ci_high': float(hi)}


def summarize(rows: list[dict], label: str, seed: str | int, group: str) -> dict:
    selected = sorted((r for r in rows if r['range'] == label and str(r['seed']) == str(seed)
                       and (group == 'population' or r['type'] == group)), key=lambda r: r['cell_id'])
    result = {'range': label, 'seed': seed, 'type': group, 'cells': len(selected)}
    for model in ('current_retipath', 'shared_retipath', 'openretina_adapted'):
        result[model + '_nll'] = float(np.mean([r[model + '_nll'] for r in selected]))
    for name in ('shared_minus_current', 'shared_minus_openretina', 'current_minus_openretina'):
        values = np.array([r[name] for r in selected])
        result.update({name + '_' + k: v for k, v in stats(values).items()})
    reduction = -result['shared_minus_current_mean']
    original = result['current_minus_openretina_mean']
    result['gap_reduction_nats_per_bin'] = reduction
    result['gap_reduction_percent'] = 100 * reduction / original if original > 0 else None
    return result


def evaluate_all() -> tuple[list[dict], list[dict], list[dict], dict]:
    lock = read_json(DEST / 'checkpoints/provenance/evaluation_lock.json')
    pre = read_json(DEST / 'checkpoints/provenance/pretraining_lock.json')
    assert lock['ranges'] == [[16, 20], [20, 60]]
    assert digest(DEST / 'PROTOCOL.md') == pre['protocol_sha256']
    for path, sha in pre['frozen_sha256'].items():
        assert digest(ROOT / path) == sha, path
    registry = read_json(REGISTRY)
    metadata = torch.load(DEST / 'checkpoints/provenance/metadata.pt', weights_only=True)
    models = {}
    for seed in SEEDS:
        path = DEST / 'checkpoints' / str(seed) / 'refit.pt'
        assert digest(path) == lock['refits'][str(seed)]['checkpoint_sha256']
        cp = torch.load(path, weights_only=True)
        assert cp['step'] == cp['selected_step'] == cp['model_step']
        model = TypeSharedRetiPath(metadata, seed, cp['rms'])
        model.load_state_dict(cp['model'], strict=True)
        model.eval().requires_grad_(False)
        for group in model.groups.values():
            for name in pre['shared_parameters']:
                assert all(torch.equal(cp['model']['cell_models.'+group[0]+'.'+name],
                                       cp['model']['cell_models.'+c+'.'+name]) for c in group)
        models[seed] = model
    old_rows = read_csv(BASELINE / 'per_cell_scores.csv')
    old_index = {(r['range'], r['cell_id'], r['seed']): r for r in old_rows}
    saved = DEST / 'checkpoints/evaluation_logits'
    saved.mkdir(exist_ok=False)
    rows, replay_errors = [], []
    cache_errors, formal_errors, numpy_errors = [], [], []
    for start, stop in ((16, 20), (20, 60)):
        label = f'[{start},{stop})'
        splits = {c: load_split(c, start, stop) for c in sorted(registry['cells'])}
        bank = make_bank(splits, label)
        per_seed, logits_seed = {}, {}
        for seed, model in models.items():
            _, per_seed[seed], logits_seed[seed] = evaluate(model, bank)
            for cell in bank.cells:
                obs = bank.cells[cell]
                subset = torch.arange(min(4, len(obs.targets)))
                reference = model.cell_models[cell](bank.inputs[obs.input_indices[subset]],
                            observed_counts=obs.targets[subset]).logits
                fast = logits_seed[seed][cell][subset]
                torch.testing.assert_close(fast, reference, rtol=3e-5, atol=5e-6)
                formal_errors.append(float((fast-reference).abs().max()))
        for cell, split in splits.items():
            safe = cell.replace('#', '_')
            contract = identity(split)
            author_logits = np.load(BASELINE / 'checkpoints/evaluation_logits' / f'{safe}_{start}.npz')
            final = ROOT / 'output/evaluations/retipath_final_model_evidence_20260913/prediction/cells'
            old_model_rows = read_json(final / f'{safe}.json')['rows']
            current_logits = np.load(final / f'{safe}.npz')
            archive = {}
            mask = split.valid_mask.numpy()
            target = split.spike_events.numpy().astype(np.float64)
            for seed in SEEDS:
                old = old_index[(label, cell, str(seed))]
                assert all(contract[k] == old[k] for k in contract), ('DATA MISMATCH', cell, label)
                assert old['retipath_checkpoint_sha256'] == pre['current_retipath'][cell][str(seed)]['sha256']
                assert digest(ROOT / pre['current_retipath'][cell][str(seed)]['path']) == old['retipath_checkpoint_sha256']
                original = next(r for r in old_model_rows if r['range'] == label and r['model'] == 'RetiPath' and int(r['seed']) == seed)
                for array, expected in ((author_logits[str(seed)], float(old['openretina_adapted_nll'])),
                                        (current_logits[original['logits_key']], float(old['retipath_nll']))):
                    value = array.astype(np.float64)
                    recomputed = float((np.logaddexp(0, value) - target * value)[mask].mean())
                    cache_errors.append(abs(recomputed-expected))
                    assert abs(recomputed-expected) < 1e-12
                current = float(old['retipath_nll'])
                author = float(old['openretina_adapted_nll'])
                shared = per_seed[seed][cell]
                array = logits_seed[seed][cell].numpy()
                value = array.astype(np.float64)
                numpy_nll = float((np.logaddexp(0, value) - target * value)[mask].mean())
                numpy_errors.append(abs(numpy_nll-shared))
                assert abs(numpy_nll-shared) < 1e-12
                group = metadata[cell]['polarities'][0] + ' ' + metadata[cell]['cell_types'][0]
                rows.append({'range': label, 'cell_id': cell, 'type': group, 'seed': seed,
                    'current_retipath_nll': current, 'shared_retipath_nll': shared, 'openretina_adapted_nll': author,
                    'shared_minus_current': shared-current, 'shared_minus_openretina': shared-author,
                    'current_minus_openretina': current-author, 'scored_bins': int(mask.sum()), **contract,
                    'shared_checkpoint_sha256': lock['refits'][str(seed)]['checkpoint_sha256'],
                    'current_checkpoint_sha256': old['retipath_checkpoint_sha256'],
                    'openretina_checkpoint_sha256': old['openretina_checkpoint_sha256']})
                archive[str(seed)] = array
            author_logits.close()
            current_logits.close()
            with (saved / f'{safe}_{start}.npz').open('xb') as stream:
                np.savez_compressed(stream, **archive)
            selected = [r for r in rows if r['range'] == label and r['cell_id'] == cell]
            aggregate = {k: selected[0][k] for k in ('range', 'cell_id', 'type', 'scored_bins', *contract.keys())}
            aggregate['seed'] = 'aggregate'
            for field in ('current_retipath_nll', 'shared_retipath_nll', 'openretina_adapted_nll',
                          'shared_minus_current', 'shared_minus_openretina', 'current_minus_openretina'):
                aggregate[field] = float(np.mean([r[field] for r in selected]))
            rows.append(aggregate)
    groups = sorted(next(iter(models.values())).groups)
    summaries = [summarize(rows, label, seed, 'population') for label in ('[16,20)', '[20,60)') for seed in (*SEEDS, 'aggregate')]
    types = [summarize(rows, label, seed, group) for label in ('[16,20)', '[20,60)')
             for seed in (*SEEDS, 'aggregate') for group in groups]
    evidence = {'completed_utc': utc(), 'ranges': [[16, 20], [20, 60]], 'cells_per_range': 22,
                'rows': len(rows), 'cached_score_max_error': max(cache_errors),
                'independent_numpy_score_max_error': max(numpy_errors),
                'trained_formal_forward_max_error': max(formal_errors),
                'all_evaluation_checkpoints_unchanged': True}
    for seed in SEEDS:
        assert digest(DEST / 'checkpoints' / str(seed) / 'refit.pt') == lock['refits'][str(seed)]['checkpoint_sha256']
    return rows, summaries, types, evidence


def plot_curves(rows: list[dict], summary: list[dict]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey='row', constrained_layout=True)
    for column, seed in enumerate(SEEDS):
        axis = axes[0, column]
        selected = [r for r in rows if int(r['seed']) == seed and r['phase'] == 'selection']
        axis.plot([int(r['step']) for r in selected], [float(r['train_nll']) for r in selected], label='inner fit')
        axis.plot([int(r['step']) for r in selected], [float(r['inner_validation_nll']) for r in selected], label='inner validation')
        best = min(selected, key=lambda r: float(r['inner_validation_nll']))
        axis.axvline(int(best['step']), color='gray', linestyle='--', label='selected step')
        axis.set(title=f'Seed {str(seed)[-4:]}', xlabel='Optimizer updates')
        axis.grid(alpha=.2)
        refit = [r for r in rows if int(r['seed']) == seed and r['phase'] == 'refit']
        axes[1, column].plot([int(r['step']) for r in refit], [float(r['train_nll']) for r in refit],
                             color='tab:green', label='full [0,16) train')
        axes[1, column].set(title='Fresh refit', xlabel='Optimizer updates')
        axes[1, column].grid(alpha=.2)
    axes[0, 0].set_ylabel('Equal-cell Bernoulli NLL (nats/bin)')
    axes[1, 0].set_ylabel('Equal-cell Bernoulli NLL (nats/bin)')
    axes[0, -1].legend(fontsize=8)
    axes[1, -1].legend(fontsize=8)
    fig.savefig(DEST / 'figures/training.png', dpi=180)
    plt.close(fig)


def report(summaries: list[dict], types: list[dict]) -> None:
    aggregate = [r for r in summaries if r['seed'] == 'aggregate']
    train = read_csv(DEST / 'training_per_seed.csv')
    plot_curves(train, summaries)
    complexity = []
    for seed in SEEDS:
        for phase in ('selection', 'refit'):
            end = read_json(DEST / 'checkpoints' / str(seed) / f'{phase}_complete.json')
            complexity.append({'model': 'type-shared RetiPath', 'seed': seed, **end,
                'trainable_parameters': 292, 'type_shared_parameters': 116, 'cell_specific_parameters': 176,
                'current_retipath_trainable_parameters': 814, 'new_configurations': 1,
                'independent_cell_losses_per_update': 22, 'sequences_per_cell_per_update': 4,
                'cost_scope': 'CPU one thread per worker; three concurrent workers; durations overlap'})
    for old in read_csv(BASELINE / 'complexity.csv'):
        complexity.append({'model': old['model'], 'seed': old['seed'], 'phase': 'frozen reference',
            'trainable_parameters': old['total_trainable_parameters_per_seed'],
            'new_updates': 0, 'selection_opportunities': old['selection_opportunities'],
            'historical_refit_updates': old['historical_refit_updates_sum_over_cells'] or old['refit_updates'],
            'sharing': old['sharing'], 'effective_window': old['stimulus_causal_window']})
    write_csv(DEST / 'complexity.csv', complexity)
    lines = ['# RetiPath capacity diagnosis', '',
             '1. **参数共享本身能否提高预测？**', '',
             '以下均为已消费时间段的描述性比较，单位 nats/scored bin；先逐cell平均3 seeds，再对cells配对重采样。负Δ有利共享模型。', '',
             '| 区间 | 当前逐cell | type-shared | OpenRetina-adapted | shared−current（95% CI） | wins/losses/ties |',
             '|---|---:|---:|---:|---|---|']
    for r in aggregate:
        prefix = 'shared_minus_current_'
        lines.append(f"| {r['range']} | {r['current_retipath_nll']:.6f} | {r['shared_retipath_nll']:.6f} | {r['openretina_adapted_nll']:.6f} | {r[prefix+'mean']:+.6f} [{r[prefix+'ci_low']:+.6f}, {r[prefix+'ci_high']:+.6f}] | {r[prefix+'wins']}/{r[prefix+'losses']}/{r[prefix+'ties']} |")
    primary = next(r for r in aggregate if r['range'] == '[20,60)')
    lo, hi = primary['shared_minus_current_ci_low'], primary['shared_minus_current_ci_high']
    conclusion = '共享训练方案提高预测' if hi < 0 else '共享训练方案降低预测' if lo > 0 else '共享训练方案与当前模型的差异未分辨'
    lines += ['', f'主要[20,60)结果：{conclusion}。这里同时改变了参数约束、联合平衡采样和全population选模，不能把差异完全归因于单一统计正则化效应。结构、机制和前端保持原定义。', '',
              '2. **哪些cell type受益？**', '',
              '| Type | N | development Δ（95% CI） | [20,60) Δ（95% CI） | primary wins/losses/ties |',
              '|---|---:|---|---|---|']
    for group in sorted({r['type'] for r in types}):
        dev = next(r for r in types if r['type'] == group and r['range'] == '[16,20)' and r['seed'] == 'aggregate')
        main = next(r for r in types if r['type'] == group and r['range'] == '[20,60)' and r['seed'] == 'aggregate')
        def interval(r):
            return f"{r['shared_minus_current_mean']:+.6f} [{r['shared_minus_current_ci_low']:+.6f}, {r['shared_minus_current_ci_high']:+.6f}]"
        lines.append(f"| {group} | {main['cells']} | {interval(dev)} | {interval(main)} | {main['shared_minus_current_wins']}/{main['shared_minus_current_losses']}/{main['shared_minus_current_ties']} |")
    lines += ['', '按type分析为小样本描述；CI跨零只表示未分辨。每seed、median及全部配对结果见type_summary.csv。', '',
              '3. **与OpenRetina的gap缩小多少？**', '',
              '| 区间 | 原gap | 剩余gap shared−OpenRetina（95% CI） | gap减少量 | 减少比例 |',
              '|---|---:|---|---:|---:|']
    for r in aggregate:
        lines.append(f"| {r['range']} | {r['current_minus_openretina_mean']:+.6f} | {r['shared_minus_openretina_mean']:+.6f} [{r['shared_minus_openretina_ci_low']:+.6f}, {r['shared_minus_openretina_ci_high']:+.6f}] | {r['gap_reduction_nats_per_bin']:+.6f} | {r['gap_reduction_percent']:+.1f}% |")
    lines += ['', '| 区间 | Seed | shared−current | shared−OpenRetina |', '|---|---|---:|---:|']
    for r in summaries:
        if r['seed'] != 'aggregate':
            lines.append(f"| {r['range']} | {r['seed']} | {r['shared_minus_current_mean']:+.6f} | {r['shared_minus_openretina_mean']:+.6f} |")
    if primary['shared_minus_openretina_ci_low'] > 0:
        next_step = ('仍值得保留H1→BC capacity layer作为下一项独立容量检验，但本轮只说明当前共享方案未消除预测gap，'
                     '没有定位到H1→BC，也没有证明新增层必要。若选模接近预算上限仍在改善，训练充分性必须与新增层效应分开处理；不能直接升级架构。')
    elif primary['shared_minus_openretina_ci_high'] < 0:
        next_step = '本轮已经显示共享模型的预测优势，因此没有仅凭这个能力差距立即启动H1→BC加层的充分理由；新层必要性仍未验证。'
    else:
        next_step = '与OpenRetina的差异未分辨，不能称等效；本轮不足以认定有必要立即启动H1→BC加层，容量瓶颈尚未定位。'
    lines += ['', '4. **下一步是否还有必要测试H1→BC capacity layer？**', '',
              next_step + ' 本轮没有启动新层或其他机制。', '',
              '| Seed | selection stop | selected / refit step | 停止原因 | 最佳inner NLL | 最后inner NLL |',
              '|---|---:|---:|---|---:|---:|']
    for seed in SEEDS:
        end = read_json(DEST / 'checkpoints' / str(seed) / 'selection_complete.json')
        last = [r for r in train if int(r['seed']) == seed and r['phase'] == 'selection'][-1]
        lines.append(f"| {seed} | {end['stop_step']} | {end['selected_step']} | {end['stop_reason']} | {end['best_inner_nll']:.6f} | {float(last['inner_validation_nll']):.6f} |")
    lines += ['', '训练使用一个预定配置：Adam .003、最多3000 updates、每25步评价、patience400、无scheduler；每step为4个movie窗口×22个独立cell条件似然。原逐cell模型各自选step，共享模型每seed选一个population step；OpenRetina保留上一轮作者优化/正则与选择预算。预算和训练方式不同，结果不能单独区分表示容量、共享偏差及优化充分性。达到预算上限不称充分收敛，validation平台期也不意味着全局最优。', '',
              '可学习参数292=4×29共享+22×8逐cell；当前模型814。固定geometry、RMS、膜尺度和未启用operator另列，不以参数比例宣称效率。参数逐项清单见parameters.csv；运行成本见complexity.csv。', '',
              f'![训练内曲线]({(DEST / "figures/training.png").as_posix()})', '',
              '全部selected/fresh-refit checkpoints在评价前冻结；两个参照的scores由原始logits独立重算核对。所有输出检查通过后完成；不读取新block、不重跑RF/错视、不修改正式模型。']
    with (DEST / 'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    exclusive_json(DEST / 'checkpoints/provenance/evaluation_program_lock.json',
                   {'created_utc': utc(), 'source_path': str(Path(__file__).relative_to(ROOT)),
                    'sha256': digest(Path(__file__)), 'before_evaluation_target_access': True})
    rows, summaries, types, evidence = evaluate_all()
    write_csv(DEST / 'prediction_per_cell.csv', rows)
    write_csv(DEST / 'comparison_summary.csv', summaries)
    write_csv(DEST / 'type_summary.csv', types)
    exclusive_json(DEST / 'checkpoints/provenance/evaluation_complete.json', evidence)
    report(summaries, types)
    print(json.dumps([r for r in summaries if r['seed'] == 'aggregate']), flush=True)


if __name__ == '__main__':
    main()
