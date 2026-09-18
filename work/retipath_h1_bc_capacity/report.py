from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import torch

from common import OUT, ROOT, SEEDS, load_json, save_json, sha, verify_sources


def number(value: str) -> float:
    return float(value)


def verification(out: Path) -> dict:
    protocol = load_json(out / 'checkpoints/protocol.json')
    checks = load_json(out / 'correctness.json')
    assert sha(out / 'PROTOCOL.md') == protocol['protocol_sha256']
    verify_sources(protocol, out)
    complete = load_json(out / 'checkpoints/training_complete.json')
    selection = load_json(out / 'checkpoints/selection_lock.json')
    evaluations = load_json(out / 'checkpoints/evaluation_complete.json')
    evaluation_lock = load_json(out / 'checkpoints/evaluation_lock.json')
    assert sha(Path(__file__).with_name('analyze.py')) == evaluation_lock['analysis_code_sha256']
    assert sha(out / 'checkpoints/training_complete.json') == evaluation_lock['training_complete_sha256']
    for name, digest in evaluation_lock['dependency_hashes'].items():
        assert sha(ROOT / name) == digest
    for path, digest in {**complete['files'], **selection['files']}.items():
        assert sha(out / path) == digest
    for name, digest in evaluations['files'].items():
        assert sha(out / name) == digest
    fits, changed, unseen, steps, elapsed = [], [], [], 0, 0.
    rf_observations, rf_prefix_error, rf_missing = 0, 0., set()
    for cell, ref in protocol['cells'].items():
        for seed in SEEDS:
            analysis_path = out / 'checkpoints' / cell.replace('#', '_') / str(seed) / 'analysis.json'
            analysis = load_json(analysis_path)
            assert analysis['protocol_sha256'] == protocol['protocol_sha256']
            assert len(analysis['prediction']) == 2 and len(analysis['RF']) == 3
            for name, quality in analysis['checks'].items():
                if name.endswith('/RF'):
                    assert quality['weights_unchanged'] and quality['full_prefix_error'] < 1e-10
                    rf_observations += quality['observations']
                    rf_prefix_error = max(rf_prefix_error, quality['full_prefix_error'])
            rf_missing.update(r['cell_id'] for r in analysis['RF'] if r['status'] != 'VERIFIED')
            old = Path(ref['A'][str(seed)]['path'])
            assert sha(old) == ref['A'][str(seed)]['sha256']
            for stage in ('inner', 'refit'):
                a_cp = torch.load(old.parent / f'{stage}.pt', weights_only=True, mmap=True)
                for condition in ('B', 'C'):
                    folder = out / 'checkpoints' / cell.replace('#', '_') / str(seed) / condition
                    cp = torch.load(folder / f'{stage}.pt', weights_only=True, mmap=True)
                    assert cp['schedule_sha256'] == a_cp['schedule_sha256'] == ref['schedule_hashes'][str(seed)][stage]
                    assert cp['h1_rms'] == ref['h1_rms'][stage]
                    for key, value in a_cp['initial_state'].items():
                        if key not in ('spatial_ei.rms_e', 'spatial_ei.rms_i'):
                            assert torch.equal(value, cp['initial_state'][key]), key
                    for key in ('h1_rms', 'input_basis.slopes', 'spatial_ei.rms_e', 'spatial_ei.rms_i'):
                        assert torch.equal(cp['model'][key], cp['initial_state'][key])
                    if condition == 'C':
                        assert torch.equal(cp['model']['input_basis.offsets'], cp['initial_state']['input_basis.offsets'])
                    assert cp['trainable_parameters'] == 45 and cp['fixed_buffers_unchanged']
                    trainable_names = set(cp['gradient_nonzero_seen'])
                    assert all(torch.equal(value, cp['initial_state'][key])
                               for key, value in cp['model'].items() if key not in trainable_names)
                    assert cp['optimizer_steps'] == ([cp['step']] if cp['step'] else [])
                    assert cp['step'] <= 3000
                    if stage == 'refit':
                        inner = torch.load(folder / 'inner.pt', weights_only=True, mmap=True)
                        assert cp['step'] == inner['best_step']
                    else:
                        rows = cp['curve']
                        best = min(rows, key=lambda r: r['inner_validation_nll'])
                        assert best['step'] == cp['best_step']
                    tag = f'{cell}/{seed}/{condition}/{stage}'
                    unseen.extend(tag + '/' + k for k, v in cp['gradient_nonzero_seen'].items() if not v)
                    changed.extend(tag + '/' + k for k, v in cp['actually_changed'].items() if not v)
                    steps += cp['step']
                    elapsed += cp['elapsed_seconds']
                    fits.append({'cell_id': cell, 'group': ref['group'], 'seed': seed, 'condition': condition,
                                 'phase': stage, 'step': cp['step'], 'selected_step': cp['best_step'],
                                 'elapsed_seconds': cp['elapsed_seconds']})
    assert len(fits) == 264
    checks['synthetic_interface_real_target_reads'] = checks.pop('real_target_reads', 0)
    checks.update({'status': 'VERIFIED', 'training_complete': True, 'evaluation_complete': True,
        'optimizer_updates': steps, 'new_optimizer_updates': steps, 'new_fit_worker_hours': elapsed/3600,
        'new_selection_fits': 132, 'new_fresh_refits': 132, 'reused_A_refits': 66,
        'same_original_initial_parameters_and_schedules_as_A': True,
        'shared_basis_fixed_and_not_optimized': True, 'formal_model_sources_unchanged': True,
        'all_nontrainable_state_entries_unchanged': True,
        'old_A_checkpoints_unchanged': True, 'fresh_refit_exact_selected_step': True,
        'evaluation_ranges_s': [[16, 20], [20, 60]], 'RF_range_s': [16, 20], 'new_time_blocks': 0,
        'target_access_ranges_s': [[0, 16], [16, 20], [20, 60]],
        'external_target_access_only_after_refit_freeze': True,
        'update_count_scope': 'effective updates on retained deterministic fit trajectories',
        'physical_updates_including_lost_crash_computation': 'UNVERIFIED',
        'RF_new_full_prefix_observations': rf_observations,
        'RF_max_full_sequence_prefix_logit_error': rf_prefix_error,
        'RF_missing_existing_context_cells': sorted(rf_missing),
        'parameters_without_observed_nonzero_gradient': unseen,
        'parameters_not_changed_at_saved_endpoint': changed, 'fits': fits})
    save_json(out / 'correctness.json', checks)
    return checks


def training_plot(out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = list(csv.DictReader((out / 'training.csv').open(encoding='utf-8')))
    cells = sorted({r['cell_id'] for r in rows})
    figure, axes = plt.subplots(6, 4, figsize=(15, 18), sharex=True)
    colors = {'A': '#444444', 'B': '#2878b5', 'C': '#db6529'}
    for cell, axis in zip(cells, axes.flat):
        for condition in ('A', 'B', 'C'):
            for seed in SEEDS:
                points = sorted([r for r in rows if r['cell_id'] == cell and r['condition'] == condition
                                 and r['phase'] == 'inner' and int(r['seed']) == seed], key=lambda r: int(r['step']))
                x = [int(r['step']) for r in points]
                y = [float(r['inner_validation_nll']) for r in points]
                axis.plot(x, y, color=colors[condition], alpha=.65, linewidth=.9)
                selected = int(np.argmin(y))
                axis.scatter([x[selected]], [y[selected]], color=colors[condition], s=9)
        axis.set_title(cell, fontsize=10)
        axis.grid(alpha=.2)
        axis.set_xlim(0, 3000)
    for axis in list(axes.flat)[len(cells):]:
        axis.set_visible(False)
    for column in range(4):
        last = ((len(cells)-1-column)//4)*4+column
        axes.flat[last].tick_params(labelbottom=True)
    figure.suptitle('Inner validation NLL | A current, B fixed linear, C fixed nonlinear | 3 seeds\nDots: selected step; train batch curves retained in training.csv', fontsize=13)
    figure.supxlabel('Optimizer updates')
    figure.supylabel('Bernoulli NLL (nats/scored bin)')
    figure.legend(handles=[Line2D([0], [0], color=colors[c], label=label) for c, label in
                  (('A', 'A current'), ('B', 'B fixed linear'), ('C', 'C fixed nonlinear'))],
                  loc='upper center', bbox_to_anchor=(.5, .945), ncol=3, frameon=False)
    figure.tight_layout(rect=[.02, .02, 1, .925])
    figure.savefig(out / 'checkpoints/inner_validation_curves.png', dpi=140)
    plt.close(figure)


def run(out: Path = OUT) -> None:
    checks = verification(out)
    predictions = list(csv.DictReader((out / 'prediction.csv').open(encoding='utf-8')))
    basis = list(csv.DictReader((out / 'basis_diagnostics.csv').open(encoding='utf-8')))
    rf = list(csv.DictReader((out / 'rf_secondary.csv').open(encoding='utf-8')))
    def summary(label, comparison, seed='aggregate', group='population'):
        return next(r for r in predictions if r['record_type'] == 'summary' and r['range'] == label
                    and r['comparison'] == comparison and str(r['seed']) == str(seed) and r['group'] == group)
    def result(row):
        return '前者更低' if number(row['ci_high']) < 0 else ('前者更高' if number(row['ci_low']) > 0 else '未分辨')
    def interval(row):
        return f"{number(row['mean']):+.6f} [{number(row['ci_low']):+.6f}, {number(row['ci_high']):+.6f}]"
    text = '# RetiPath H1→BC capacity diagnosis\n\n'
    text += '仅为已消费时间区间的 descriptive comparison。A=正式 RetiPath；B=固定线性 bank；C=固定单调非线性 bank。负的 NLL 差有利于前者。单位均为 nats/scored Bernoulli bin。\n\n'
    text += '## 1. 更灵活的 H1→BC 表示是否提高 prediction？\n\n'
    for label in ('[16,20)', '[20,60)'):
        r = summary(label, 'C_minus_A')
        text += f"{label} 的 C−A 为 {interval(r)}，{r['wins']} wins / {r['losses']} losses / {r['ties']} ties，结论：{result(r)}。\n\n"
        seed_rows = [summary(label, 'C_minus_A', seed) for seed in SEEDS]
        text += f"三个 seed 的 C−A mean 范围 [{min(number(s['mean']) for s in seed_rows):+.6f}, {max(number(s['mean']) for s in seed_rows):+.6f}]，其中 {sum(number(s['ci_high'])<0 for s in seed_rows)}/3 个 seed 的 CI 完全低于0。\n\n"
    text += '| 区间 | A absolute NLL | B absolute NLL | C absolute NLL |\n|---|---:|---:|---:|\n'
    for label in ('[16,20)', '[20,60)'):
        r = summary(label, 'C_minus_A')
        text += f"| {label} | {number(r['A_nll']):.6f} | {number(r['B_nll']):.6f} | {number(r['C_nll']):.6f} |\n"
    text += '\n## 2. C 是否明确优于 matched linear B？\n\n'
    text += '| 区间 | seed | C−B mean [paired-cell 95% CI] | median | wins/losses | 判读 |\n|---|---|---|---:|---|---|\n'
    for label in ('[16,20)', '[20,60)'):
        for seed in (*SEEDS, 'aggregate'):
            r = summary(label, 'C_minus_B', seed)
            text += f"| {label} | {seed} | {interval(r)} | {number(r['median']):+.6f} | {r['wins']}/{r['losses']} | {result(r)} |\n"
    text += '\n先在每 cell 内平均三个 seed 的 loss，再按 cell 重采样；没有平均概率/模型权重，也没有将66个cell-seed作为生物重复。CI跨零只表示未分辨。完整 per-seed/per-cell 和所有 secondary 统计见 prediction.csv。\n\n'
    text += '## 3. 收益主要来自线性 capacity 还是新增 nonlinearity？\n\n'
    for label in ('[16,20)', '[20,60)'):
        text += f"{label}：B−A {interval(summary(label, 'B_minus_A'))}；C−B {interval(summary(label, 'C_minus_B'))}。\n\n"
    cb_main = summary('[20,60)', 'C_minus_B')
    ba_main = summary('[20,60)', 'B_minus_A')
    if number(cb_main['ci_high']) < 0:
        text += '在[20,60)上，C−B支持固定单调nonlinear expansion具有额外预测价值。'
    elif number(cb_main['ci_low']) > 0:
        text += '在[20,60)上，C比B更差，本轮固定nonlinear expansion未带来额外预测价值。'
    else:
        text += '在[20,60)上，新增nonlinearity的额外预测价值未分辨，不能把C−A点估计的变化直接归因于nonlinearity。'
    if number(ba_main['ci_high']) < 0:
        text += 'B−A同时支持线性重新参数化后的训练收益。'
    else:
        text += 'B−A也没有确立线性重新参数化的预测优势。'
    text += '\n\n'
    text += 'B 的四个 channel 经非负归一化组合严格折叠为局部线性缩放，本身未增加空间特征；B−A反映这种参数化及优化路径的影响。C−B检验本轮固定 nonlinear bank 的额外预测价值，不能推广为任意非线性或生理机制的效应。\n\n'
    text += '| [20,60) cell type | N | B−A mean [95% CI] | C−B mean [95% CI] | C−A mean [95% CI] |\n|---|---:|---|---|---|\n'
    for group in ('PC ON', 'PC OFF', 'MC ON', 'MC OFF'):
        r = summary('[20,60)', 'C_minus_A', group=group)
        text += f"| {group} | {r['n_cells']} | {interval(summary('[20,60)', 'B_minus_A', group=group))} | {interval(summary('[20,60)', 'C_minus_B', group=group))} | {interval(r)} |\n"
    text += '\nPC=midget，MC=parasol；ON/OFF与geometry均取原manifest。分组为描述性结果，不把小组结果当作总体确认。\n\n'
    text += '## 4. 新 basis 是否实际使用，还是明显冗余？\n\n'
    for condition in ('B', 'C'):
        activation = [r for r in basis if r['condition'] == condition and r['phase'] == 'refit' and r['checkpoint'] == 'final' and r['record_type'] == 'activation']
        mixing = [r for r in basis if r['condition'] == condition and r['phase'] == 'refit' and r['checkpoint'] == 'final' and r['record_type'] == 'mixing']
        pairs = [r for r in basis if r['condition'] == condition and r['phase'] == 'refit' and r['checkpoint'] == 'final' and r['record_type'] == 'channel_pair']
        persistent = [r for r in basis if r['condition'] == condition and r['record_type'] == 'persistent']
        text += f"{condition}：final activation RMS 中位数 {np.median([number(r['rms']) for r in activation]):.4f}；RMS<0.01 为 {sum(r['activation_rms_below_001']=='True' for r in activation)}/{len(activation)}。最终 mixing<0.01 为 {sum(r['weight_below_001']=='True' for r in mixing)}/{len(mixing)}；pair correlation>0.995 为 {sum(r['correlation_above_0995']=='True' for r in pairs)}/{len(pairs)}，unit-RMS function difference 中位数 {np.median([number(r['unit_rms_function_difference']) for r in pairs]):.4f}。refit 后半段在固定训练诊断子集上持续与其他 channel 高相关的 channel 为 {sum(r['always_correlated_above_0995_with_any_channel']=='True' for r in persistent)}/{len(persistent)}。\n\n"
    text += '激活统计覆盖bank计算的全部289个位置，到输出的贡献仍受原BC support和mixing约束。这些分母为描述性的cell-seed-channel或pair，不用于增加统计样本量。固定 bank 的函数本身不随训练退化；相关性描述实际 H1 输入分布上存在多少功能重叠。B 的共线性由设计保证；C 的高相关不等于函数完全相同。所有激活分位数、相关性、mixing和预定长期标记见 basis_diagnostics.csv。\n\n'
    text += '| C basis | activation RMS 中位数 | sustained mixing 中位数 | transient mixing 中位数 |\n|---|---:|---:|---:|\n'
    for k in range(1, 5):
        selected = [r for r in basis if r['condition'] == 'C' and r['phase'] == 'refit' and r['checkpoint'] == 'final' and r['basis_index'] == str(k)]
        text += f"| {k} | {np.median([number(r['rms']) for r in selected if r['record_type']=='activation']):.4f} | {np.median([number(r['weight']) for r in selected if r['record_type']=='mixing' and r['pathway']=='sustained']):.4f} | {np.median([number(r['weight']) for r in selected if r['record_type']=='mixing' and r['pathway']=='transient']):.4f} |\n"
    text += '\n'
    text += 'A/B/C分别37/45/45个可学习参数/cell；每seed的22cell合计814/990/990。B/C新增参数仅2×4 mixing logits，a/b和H1 RMS均固定；所有旧可学习参数继续学习，所有旧固定量保留。\n\n'
    text += '## 5. dynamic RF 能力是否变化？\n\n'
    text += '| 条件 | 有效 cells | HIGH/LOW log-gain 中位数 | TV 中位数 | centroid shift (deg) 中位数 | HIGH−LOW radius (deg) 中位数 | centered CDF L1 (deg) 中位数 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for condition in ('A', 'B', 'C'):
        values = [r for r in rf if r['condition'] == condition and r['seed'] == 'aggregate' and r['status'] == 'VERIFIED']
        fields = ('gain_log_HIGH_over_LOW', 'TV', 'centroid_shift_deg', 'delta_radius_mean_profile_deg', 'centered_radial_CDF_L1_deg')
        text += f"| {condition} | {len(values)} | " + ' | '.join(f'{np.median([number(r[k]) for r in values]):.6g}' for k in fields) + ' |\n'
    text += '\n上述 aggregate 是 cell 内三个 seed 的指标均值，再跨 cell 取中位数；完整 P/cdf 另保留。TV度量归一化profile变化，仍可包括centroid移动；centered radial CDF以每个观测自身重心计算，区别于gain和纯平移。RF变化不是更正确的RF证明，HIGH/LOW分组关联不作因果适应结论。\n\n'
    valid = {(r['cell_id'], r['condition'], r['seed']): r for r in rf if r['status'] == 'VERIFIED'}
    rf_cells = sorted({r['cell_id'] for r in rf if r['seed'] == 'aggregate' and r['status'] == 'VERIFIED'})
    missing = sorted({r['cell_id'] for r in rf if r['status'] != 'VERIFIED'})
    if missing:
        text += '未获得既有双context观测的 cells：' + ', '.join(missing) + '，其 RF 为 UNVERIFIED，未搜索替代观测。\n\n'
    for baseline in ('A', 'B'):
        differences = {key: [number(valid[c, 'C', 'aggregate'][key])-number(valid[c, baseline, 'aggregate'][key]) for c in rf_cells]
                       for key in ('TV', 'centered_radial_CDF_L1_deg')}
        text += f"C−{baseline} 的逐cell TV差中位数 {np.median(differences['TV']):+.6g}，centered CDF L1差中位数 {np.median(differences['centered_radial_CDF_L1_deg']):+.6g} deg。"
    text += '\n\n'
    for condition in ('A', 'B', 'C'):
        seed_tv = [np.median([number(r['TV']) for r in rf if r['condition'] == condition and r['seed'] == str(seed) and r['status'] == 'VERIFIED']) for seed in SEEDS]
        seed_cdf = [np.median([number(r['centered_radial_CDF_L1_deg']) for r in rf if r['condition'] == condition and r['seed'] == str(seed) and r['status'] == 'VERIFIED']) for seed in SEEDS]
        text += f"{condition} 三个seed的TV中位数为 " + ', '.join(f'{v:.6g}' for v in seed_tv) + '；centered CDF L1中位数为 ' + ', '.join(f'{v:.6g}' for v in seed_cdf) + ' deg。\n\n'
    for condition in ('A', 'B', 'C'):
        values = [r for r in rf if r['condition'] == condition and r['seed'] == 'aggregate' and r['status'] == 'VERIFIED']
        signs = [number(r['delta_radius_mean_profile_deg']) for r in values]
        text += f"{condition} 的 HIGH−LOW radius：{sum(v>0 for v in signs)} cells增加，{sum(v<0 for v in signs)} cells减小。"
    text += '\n\n## 6. 是否足以纳入正式 RetiPath？\n\n'
    ca, cb = summary('[20,60)', 'C_minus_A'), summary('[20,60)', 'C_minus_B')
    if number(ca['ci_high']) < 0 and number(cb['ci_high']) < 0:
        text += '本轮支持固定 nonlinear bank 在已消费数据上的预测价值，但仍属单一预设bank、三个seed的descriptive诊断；尚不能直接升级为正式主模型。'
    else:
        text += '本轮没有同时确立相对当前正式模型及matched linear control的预测优势，不足以把该module纳入正式RetiPath。'
    text += '\n\n训练的数值正确性已验证，验证平台期与预算限制需要分别看待：\n\n'
    text += '| 条件 | inner fits | 到3000上限 | selected step 中位数 [min,max] | selected≥2750 |\n|---|---:|---:|---|---:|\n'
    for condition in ('B', 'C'):
        fits = [f for f in checks['fits'] if f['phase'] == 'inner' and f['condition'] == condition]
        selected = [f['selected_step'] for f in fits]
        text += f"| {condition} | {len(fits)} | {sum(f['step']==3000 for f in fits)} | {np.median(selected):g} [{min(selected)},{max(selected)}] | {sum(s>=2750 for s in selected)} |\n"
    text += f"\n保留的拟合轨迹共 {checks['new_optimizer_updates']:,} 有效 optimizer updates，保存记录的累计 worker 时间 {checks['new_fit_worker_hours']:.2f} h（并行进程时间相加，不是墙钟时间）。达到预算不代表充分收敛；未增加训练预算、seed、选择机会或新block。历史A选模机会与本轮一致，其冻结结果复用，不改变正式checkpoint。\n\n"
    if (out / 'checkpoints/runtime_events.json').exists():
        text += '电脑发生死机/重启；完整checkpoint从原参数、Adam和选模状态继续。6个最新文件损坏并已保留，只有这6个原有拟合使用原seed和原batch顺序从step0重算；受损的原轨迹无法逐位对照，初始化、数据、数值训练函数和更新顺序均未改变。保存函数增加fsync、write-through替换和上一份snapshot；数值训练函数AST保持不变。随后降到4个低优先级并行任务。未增加有效优化路径或选模机会；实际物理update数与总计算时间因重启前丢失片段而为UNVERIFIED，保存记录的worker时间不包含这些额外工程开销。详见 checkpoints/engineering_corrections.json 和 checkpoints/recovery_20260914/recovery.json。\n\n'
    text += '![Inner validation curves](checkpoints/inner_validation_curves.png)\n\n全部检查与参数清单见 correctness.json；固定科学定义与来源见 PROTOCOL.md。正式模型保持不变，本轮到此停止。\n'
    (out / 'REPORT.md').write_text(text, encoding='utf-8')
    training_plot(out)
    protocol = load_json(out / 'checkpoints/protocol.json')
    source_names = set(protocol['source_hashes'])
    source_names.update(str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob('*.py'))
    source_names.update(load_json(out / 'checkpoints/evaluation_lock.json')['dependency_hashes'])
    snapshot = out / 'checkpoints/source_snapshot'
    source_hashes = {}
    for name in sorted(source_names):
        source, destination = ROOT / name, snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[name] = sha(destination)
    save_json(out / 'checkpoints/source_snapshot.json', {'files': source_hashes,
              'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
              'historical_frozen_source_and_serialization_corrections': 'engineering_corrections.json'})
    save_json(out / 'checkpoints/completion.json', {'status': 'VERIFIED', 'new_time_blocks': 0,
        'files': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
        'report_code_sha256': sha(Path(__file__))})
    print('REPORT AND VERIFICATION COMPLETE', flush=True)


if __name__ == '__main__':
    run()
