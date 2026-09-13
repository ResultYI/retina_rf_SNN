from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from retipath_temporal_common import OUT, SEEDS, load_json, sha, utc, write_once
from retipath_phase2_analyze import csv_rows


COLORS = {'A': '#70747a', 'B': '#dc8c24', 'C': '#237dad'}


def table(headers: list[str], rows: list[list[str]]) -> str:
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(row) + ' |' for row in rows]])


def save_figure(fig: plt.Figure, path: Path) -> None:
    assert not path.exists(), path
    fig.savefig(path, dpi=160, facecolor='white')
    plt.close(fig)


def report() -> None:
    analysis = load_json(OUT / 'analysis_complete.json')
    verification = load_json(OUT / 'verification.json')
    lock = load_json(OUT / 'confirmatory_lock.json')
    assert verification['status'] == 'VERIFIED'
    cells = [r['cell_id'] for r in lock['cells']]
    jobs = [load_json(p) for p in sorted((OUT / 'results').glob('*.json'))]
    prediction = [r for j in jobs for r in j['prediction']]
    rf = [r for j in jobs for r in j['RF']]
    plook = {(r['cell_id'], r['seed'], r['condition']): r for r in prediction}
    rlook = {(r['cell_id'], r['seed'], r['condition']): r for r in rf}
    aggregate = {}
    scalar_metrics = [key for key, value in rf[0].items()
                      if isinstance(value, (int, float)) and key not in ['seed', 'LOW_n', 'HIGH_n']]
    for cell in cells:
        for condition in 'ABC':
            rows = [rlook[cell, seed, condition] for seed in SEEDS]
            aggregate[cell, condition] = {'row_type': 'cell_mean_over_seeds', 'cell_id': cell,
                'condition': condition, 'seed': 'all', 'n_cells': 1,
                **{metric: float(np.mean([r[metric] for r in rows])) for metric in scalar_metrics}}
    rf_summaries = list(aggregate.values())
    for condition in 'ABC':
        for seed in [*SEEDS, 'all']:
            rows = ([rlook[cell, seed, condition] for cell in cells] if seed != 'all'
                    else [aggregate[cell, condition] for cell in cells])
            rf_summaries.append({'row_type': 'population_summary' if seed == 'all' else 'seed_summary',
                'cell_id': 'all', 'condition': condition, 'seed': seed, 'n_cells': len(rows),
                **{metric: float(np.mean([r[metric] for r in rows])) for metric in scalar_metrics}})
    csv_rows(OUT / 'rf_population.csv', rf_summaries)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'semibold', 'savefig.bbox': 'tight'})
    fig, axes = plt.subplots(2, 3, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 2.8]}, layout='constrained')
    for col, comparison in enumerate(['C-A', 'C-B', 'B-A']):
        ax = axes[0, col]
        for index, seed in enumerate([*SEEDS, 'all']):
            row = analysis['prediction'][f'{comparison}/{seed}']
            ax.errorbar(row['mean'], index, xerr=[[row['mean'] - row['ci_low']], [row['ci_high'] - row['mean']]],
                        fmt='o', color='#1a5e84' if seed == 'all' else '#7d9eb4', capsize=3)
        ax.axvline(0, color='#777777', lw=0.8)
        ax.set_yticks(range(4), ['Seed 1', 'Seed 2', 'Seed 3', 'Seed aggregate'])
        ax.invert_yaxis()
        ax.set_title(comparison + (' (primary)' if col == 0 else ' (secondary)'))
        ax.set_xlabel('Equal-cell mean NLL difference; 95% CI')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        left, right = comparison.split('-')
        ax = axes[1, col]
        values = np.array([[plook[cell, seed, left]['confirmation_nll'] - plook[cell, seed, right]['confirmation_nll']
                            for cell in cells] for seed in SEEDS])
        for index, seed in enumerate(SEEDS):
            ax.scatter(values[index], np.arange(len(cells)) + (index - 1) * 0.16, s=18, alpha=0.7, label=f'Seed {index + 1}')
        ax.scatter(values.mean(0), range(len(cells)), s=26, marker='|', color='black', label='Seed mean')
        ax.axvline(0, color='#777777', lw=0.8)
        ax.set_yticks(range(len(cells)), cells)
        ax.invert_yaxis()
        ax.set_xlabel('Per-cell NLL difference (nats / bin)')
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.025), ncol=4, fontsize=9)
    fig.suptitle('Frozen temporal confirmation [240,300) s | 17 coverage-eligible cells | lower NLL is better', fontsize=13)
    save_figure(fig, OUT / 'figures/prediction_confirmation.png')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for ax, metric, label, scale in zip(axes.flat,
            ['TV', 'centered_radial_CDF_L1_deg', 'gain_log_HIGH_over_LOW', 'delta_radius_observation_median_deg'],
            ['Normalized profile TV', 'Centered radial CDF L1 (millidegrees)', 'log(gain HIGH / LOW)',
             'HIGH - LOW radius (millidegrees; observation medians)'], [1, 1000, 1, 1000], strict=True):
        for index, condition in enumerate('ABC'):
            values = np.array([aggregate[cell, condition][metric] for cell in cells]) * scale
            offsets = np.linspace(-0.15, 0.15, len(cells))
            ax.scatter(index + offsets, values, c=COLORS[condition], alpha=0.75, s=28)
            ax.plot([index - 0.2, index + 0.2], [values.mean()] * 2, color='black', lw=2)
        ax.axhline(0, color='#999999', lw=0.7)
        ax.set_xticks(range(3), ['A: RetiPath', 'B: current', 'C: conductance'])
        ax.set_ylabel(label)
    fig.suptitle('RF context dynamics | each dot = one cell, averaged over three seeds\nBlack line = equal-cell mean; no RF effect cutoff', fontsize=13)
    save_figure(fig, OUT / 'figures/rf_confirmation.png')

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained')
    for ax, condition in zip(axes, 'ABC', strict=True):
        matched = analysis['RF'][condition]['phase2_same_cell_comparison']
        x = np.array([r['phase2_radius_delta_mean'] for r in matched]) * 1000
        y = np.array([r['confirmation_radius_delta_mean'] for r in matched]) * 1000
        ax.scatter(x, y, color=COLORS[condition], s=35)
        ax.axhline(0, color='#888888', lw=0.8)
        ax.axvline(0, color='#888888', lw=0.8)
        matches = sum(r['radius_sign_matches_phase2'] for r in matched)
        ax.set_title(f'{condition}: same sign in {matches}/{len(matched)} cells')
        ax.set_xlabel('Phase 2 [16,20): HIGH - LOW radius (mdeg)')
        ax.set_ylabel('Confirmation [240,300): HIGH - LOW radius (mdeg)')
    fig.suptitle('Size direction across time blocks | median-of-observation radius, seed mean\n16 matched cells; 68#10 lacks a Phase 2 HIGH context', fontsize=13)
    save_figure(fig, OUT / 'figures/radius_direction.png')

    with np.load(OUT / 'sampling_before_targets.npz', allow_pickle=False) as sampling:
        positions = sampling['positions'].astype(np.float64)
    grid = np.array(lock['radial_grid_deg'])
    spacing = np.diff(np.unique(positions[:, 0])).mean()
    extent = [positions[:, 0].min() - spacing / 2, positions[:, 0].max() + spacing / 2,
              positions[:, 1].min() - spacing / 2, positions[:, 1].max() + spacing / 2]
    profile_dir = OUT / 'figures/delta_P'
    profile_dir.mkdir(exist_ok=True)
    for cell in cells:
        profiles = np.array([[[rlook[cell, seed, condition][label + '_profile'] for label in ['LOW', 'HIGH']]
                               for seed in SEEDS] for condition in 'ABC'])
        cdfs = np.array([[[rlook[cell, seed, condition][label + '_cdf'] for label in ['LOW', 'HIGH']]
                          for seed in SEEDS] for condition in 'ABC'])
        delta = profiles[:, :, 1] - profiles[:, :, 0]
        safe = cell.replace('#', '_')
        with (profile_dir / f'{safe}.npz').open('xb') as stream:
            np.savez_compressed(stream, conditions=np.array(list('ABC')), seeds=np.array(SEEDS),
                contexts=np.array(['LOW', 'HIGH']), positions_deg=positions, radial_grid_deg=grid,
                P=profiles, delta_P=delta, centered_radial_CDF=cdfs)
        means = profiles.mean(1)
        dmeans = delta.mean(1)
        pmax = means.max()
        dmax = np.abs(dmeans).max()
        fig, axes = plt.subplots(3, 4, figsize=(15, 10), layout='constrained')
        for row, condition in enumerate('ABC'):
            for col in range(3):
                ax = axes[row, col]
                values = means[row, col] if col < 2 else dmeans[row]
                im = ax.imshow(values.reshape(17, 17), origin='upper', extent=extent,
                    cmap='viridis' if col < 2 else 'RdBu_r', vmin=0 if col < 2 else -dmax,
                    vmax=pmax if col < 2 else dmax, interpolation='nearest')
                ax.set_title(f'{condition}: ' + ['LOW P(x)', 'HIGH P(x)', 'HIGH - LOW P(x)'][col])
                ax.set_xlabel('x (deg)')
                ax.set_ylabel('y (deg)')
                fig.colorbar(im, ax=ax, shrink=0.75, format='%.2g')
            ax = axes[row, 3]
            for index, label in enumerate(['LOW', 'HIGH']):
                ax.plot(grid, cdfs[row, :, index].mean(0), label=label, color=['#4b85b0', '#bd6144'][index])
            ax.set_title(f'{condition}: centered radial CDF')
            ax.set_xlabel('Radius from own centroid (deg)')
            ax.set_ylabel('Cumulative normalized sensitivity')
            ax.set_ylim(0, 1.02)
            ax.legend(fontsize=8)
        fig.suptitle(f'{cell} | [240,300) s | three-seed mean\nShared P and delta-P color scales within this cell; all bins shown, no area threshold', fontsize=13)
        save_figure(fig, profile_dir / f'{safe}.png')

    summaries = analysis['prediction']
    pred_rows = []
    for comparison in ['C-A', 'C-B', 'B-A']:
        for seed in [*SEEDS, 'all']:
            row = summaries[f'{comparison}/{seed}']
            pred_rows.append([comparison, str(seed), f"{row['mean']:+.6f}", f"{row['median']:+.6f}",
                              f"{row['wins']}/{row['losses']}/{row['ties']}", f"[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}]"])
    cell_rows = []
    for cell in cells:
        nll = {c: np.mean([plook[cell, s, c]['confirmation_nll'] for s in SEEDS]) for c in 'ABC'}
        cell_rows.append([cell, *[f'{nll[c]:.6f}' for c in 'ABC'],
                          *[f'{nll[l] - nll[r]:+.6f}' for l, r in [('C', 'A'), ('C', 'B'), ('B', 'A')]]])
    rf_rows = []
    for condition in 'ABC':
        row = next(r for r in rf_summaries if r['row_type'] == 'population_summary' and r['condition'] == condition)
        rf_rows.append([condition, *[f'{row[m]:.6g}' for m in ['LOW_gain_median', 'HIGH_gain_median', 'TV',
            'centroid_shift_deg', 'centered_radial_CDF_L1_deg', 'centered_radial_CDF_max',
            'delta_radius_observation_median_deg', 'delta_radius_mean_profile_deg']]])
    rf_seed_rows = []
    for condition in 'ABC':
        stats = analysis['RF'][condition]
        for seed, row in stats['per_seed'].items():
            rf_seed_rows.append([condition, seed, f"{row['TV']['mean']:.6f}",
                f"{row['centered_radial_CDF_L1_deg']['mean']:.8f}",
                f"{row['delta_radius_observation_median_deg']['mean']:+.8f}",
                f"{row['radius_smaller_HIGH_cells']}/{row['radius_larger_HIGH_cells']}"])
    text = [
        '# RetiPath spatial E/I：一次冻结时间确认',
        '',
        '结论：在具有连续记录的 17 个细胞、唯一新评价 block `[240,300)` 上，C−A 主终点和 C−B 次终点均复制。归一化 RF profile 的 context 差异继续存在；跨时间块统一的 RF-size 方向没有复制。预测用途支持采用 C，但不构成真实电导机制或真实 RF 变化正确性的证明。',
        '',
        '## 1. C−A prediction benefit 是否复制？',
        '',
        '**复制。** Seed aggregate 的 equal-cell mean 为 −0.020678 nats/bin，median −0.013469，14 wins / 3 losses，paired-cell bootstrap 95% CI [−0.031098, −0.010876]。三个 seed 的 mean 与各自 CI 均低于零。A/B/C 的 population 平均 NLL 分别为 0.417165 / 0.424126 / 0.396488。',
        '',
        '以下负差表示左侧模型更好；wins/losses/ties 以 cell 为单位。`all` 先在每个 cell 内平均三个 seed，再汇总与重采样；没有把 51 个 cell-seed 当成独立样本。Bootstrap 使用冻结的 10000 次 paired-cell draws，仅量化这 17 个细胞的抽样不确定性。',
        '',
        table(['比较', 'Seed', 'Mean', 'Median', 'W/L/T', 'Mean 的 95% CI'], pred_rows),
        '',
        '## 2. C−B conductance benefit 是否复制？',
        '',
        '**群体平均收益复制，逐细胞有异质性。** C−B mean −0.027639，median −0.019695，10 wins / 7 losses，95% CI [−0.043571, −0.012327]；三个 seed 均支持负的群体平均差。C 同时优于 A，所以并非仅优于较弱的 B。B−A aggregate mean +0.006961，95% CI [+0.000350, +0.013561]。',
        '',
        'C−B 是匹配空间表示、输入映射与参数数量后的整合机制对照，支持这一已拟合模型家族中的 conductance integration 预测收益。C−A 还包含空间表示及后端的共同变化，不能全部归因于电导。',
        '',
        '逐细胞的 seed-mean NLL 如下；全部 cell × seed × A/B/C 原值在 `prediction_per_cell_seed.csv`，差值与 bootstrap 汇总在 `prediction_population.csv`。',
        '',
        table(['Cell', 'A NLL', 'B NLL', 'C NLL', 'C−A', 'C−B', 'B−A'], cell_rows),
        '',
        '![Prediction](figures/prediction_confirmation.png)',
        '',
        '## 3. Normalized RF spatial dynamics 是否复制？',
        '',
        '**模型内归一化空间形状的 context 差异继续存在。** C 的 mean TV=0.030564，mean centered radial CDF L1=0.00191831 deg；去除整体 gain 后仍有差异，且按每个观测自身重心居中后仍有径向 profile 差异。17/17 cells 的 ΔP 在三个 seed 的全部两两比较中余弦为正，所有两两余弦的 median=0.999707。A/B 也各有 17/17 同方向细胞，故这种现象并非 C 独有。',
        '',
        '这复制的是“模型的 normalized profile 随既定 HIGH/LOW context 分组变化”这一现象，不是每个像素在两个时间块有完全相同变化，也不证明真实 RF 更正确、或对比度分组关联构成因果适应。未新增 RF 显著性检验或 effect threshold。',
        '',
        '以下均先平均 seed、再 equal-cell 平均；gain 是每 context 的观测中位数，P 是先逐观测归一化再取 context 均值。半径两种汇总沿用 Phase 2：各观测半径的中位数，以及 context mean P 围绕其自身重心的半径。二者不能混用。所有角度单位为 deg。',
        '',
        table(['模型', 'LOW gain', 'HIGH gain', 'TV', '重心位移', 'Centered CDF L1', 'Centered CDF max', 'Δradius 观测中位数', 'Δradius mean P'], rf_rows),
        '',
        table(['模型', 'Seed', 'Mean TV', 'Mean centered CDF L1 (deg)', 'Mean Δradius (deg, 观测中位数)', 'HIGH 更小/更大 cells'], rf_seed_rows),
        '',
        '在两个时间块都有 HIGH/LOW 的相同 16 cells 中，C 的 mean TV 从 Phase 2 的 0.009779 变为本块 0.031916，centered CDF L1 从 0.00061890 变为 0.00200115 deg。68#10 在原 development 缺 HIGH，因此只从跨块 RF 配对比较排除，本块预测与 RF 均保留。幅度变化只作描述，不作为新增通过标准。',
        '',
        '`rf_per_cell_seed.csv` 保存每个 cell/seed/condition 的 HIGH/LOW gain、P、centroid、两种 second-moment radius、TV、完整 centered CDF 与 ΔP。`rf_population.csv` 保存逐 cell 的 seed mean 与逐 seed/population 汇总。`figures/delta_P/` 中每 cell 一张 A/B/C 的 LOW、HIGH、ΔP map 与 CDF 图，配套 NPZ 保存所有三个 seed 的数组，轴顺序为 condition × seed × context × pixel/grid。',
        '',
        '![RF](figures/rf_confirmation.png)',
        '',
        '## 4. 是否存在稳定的 RF-size 方向？',
        '',
        '**跨时间块没有稳定的统一方向。** 本块 C 在所有三个 seed 中均为 13 cells HIGH 更大、4 cells HIGH 更小；mean Δradius=+0.00085559 deg（观测中位数）。这在本块 seeds 间稳定，但与 Phase 2 的同 cell 方向仅 5/16 一致。相同 16 cells 的平均 Δradius 从 −0.00022517 变成 +0.00088364 deg。不能据此宣称 HIGH 下普遍收缩，也不能宣称普遍扩大。',
        '',
        '![Size direction](figures/radius_direction.png)',
        '',
        '## 5. C 是否足以升级为下一代 RetiPath 主模型？',
        '',
        '**就本次覆盖的 macaque spike-prediction 用途，证据支持升级 C。** 主终点 C−A 与匹配对照 C−B 都在一个未用于拟合、选择或先前评分的时间块复制，且三个 seeds 方向稳定；因此推荐保留 conductance 版本作为下一代预测主模型。RF-size 方向不稳定限制的是机制解释，不抵消已观察到的预测收益。',
        '',
        '这不是对全部 22 cells、其他刺激/物种或真实生物电导机制的无条件确认；5 个无连续覆盖细胞的这一时间段泛化仍为 **UNVERIFIED**。本次没有修改默认模型、架构、K、参数、alpha bounds、任何旧 checkpoint 或 Phase 2 成果；推荐与实际切换分开，等待用户后续决定。',
        '',
        '## 冻结与验证证据',
        '',
        f"- 本地 branch `{lock['branch']}`；HEAD `{lock['HEAD']}`。来源直接解析 Phase 2 实际 checkpoint，全部 396 个 selected-inner/fresh-refit 文件被冻结；评分仅使用 153 个 coverage-eligible full-[0,16) refit 终点。",
        f"- Confirmatory lock 写入 `{lock['frozen_utc']}`，SHA256 `{sha(OUT / 'confirmatory_lock.json')}`；TEST_CONSUMED 写入后才解析本块 targets。详细科学定义在 `PROTOCOL.md`，锁定文件与来源哈希在 `confirmatory_lock.json`。",
        '- 唯一 block `[240,300)`；150 Hz Bernoulli occupancy，60 个独立 150-bin sequences、30-bin warmup、每 cell 7200 个评分 bins。Observed strictly-past history 固定；完整可用因果 prefix Jacobian，没有 16-lag 截断或状态 detach。',
        '- 22 个 Phase 2 cells 中，68#4、69#4、69#21、70#1、70#15 没有连续记录覆盖；17-cell cohort 在 targets 前按覆盖冻结。全部 17 cells 的已有 [0,16) spike bins 与 cone drive 在锁定前精确重放。',
        '- 本地 provenance 初扫和仅针对权限缺口的补扫共覆盖 8354 个对象，缺口已关闭；既有 target-based 使用覆盖 [0,240)，未见 [240,300) 先前拟合、选择、统计或评分证据。旧代码曾先解析完整原始记录再切片，因此不声称原始文件字节或所有时间戳从未进入内存；也无法排除未记录的外部活动。当前读取器仅保留本块，并在结束边界 sentinel 停止，sentinel 不保存、不评分。',
        f"- 验证结果 **VERIFIED**：{verification['source_and_checkpoint_hashes_unchanged']} 项冻结来源哈希不变；153 个保存 logits 的 NLL 独立重算最大误差 {verification['independent_saved_logits_NLL_max_abs_error']:.3g}；paired-cell bootstrap 最大误差 {verification['independent_paired_bootstrap_max_abs_error']:.3g}。",
        f"- 6120 个 RF 观测；P 归一化最大误差 {verification['RF_profile_normalization_max_abs_error']:.3g}，派生指标复算最大误差 {verification['RF_derived_metric_max_abs_error']:.3g}；完整 Jacobian 分解重加和最大相对误差 {verification['full_Jacobian_mode_sum_max_relative_error']:.3g}。对预定排序首个 cell/seed 的 A/B/C 重放，logits 逐位相等，全部 RF 字段相等，参数未变化。",
        '- 0 optimizer updates、0 refits、0 checkpoint reselection、仅 1 个新评价 block。本轮完成并停止。',
        '',
    ]
    with (OUT / 'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(text))
    artifacts = [OUT / 'REPORT.md', OUT / 'rf_population.csv', *sorted((OUT / 'figures').rglob('*'))]
    write_once(OUT / 'delivery.json', {'completed_utc': utc(), 'confirmatory_lock_sha256': sha(OUT / 'confirmatory_lock.json'),
        'verification_sha256': sha(OUT / 'verification.json'), 'report_source_sha256': sha(Path(__file__)),
        'figure_png_count': len(list((OUT / 'figures').rglob('*.png'))),
        'per_cell_delta_P_maps': len(cells), 'artifacts_sha256': {str(p.relative_to(OUT)): sha(p) for p in artifacts if p.is_file()}})
    print('REPORT WRITTEN: 20 figures, 17 per-cell profile arrays, prediction and RF summaries.')


if __name__ == '__main__':
    report()
