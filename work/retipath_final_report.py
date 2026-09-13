from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from retipath_final_common import OUT, CONFIRM, SEEDS, CONTROL, load_json, sha, write_once, utc, application_imports


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(row) + ' |' for row in rows]])


def save(fig, path: Path) -> None:
    assert not path.exists()
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def prediction_plot(population: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout='constrained')
    ranges = ['[16,20)', '[20,60)', '[60,120)', '[120,180)', '[180,240)', '[240,300)']
    for ax, comparator in zip(axes, ['CNN', 'LN', CONTROL], strict=True):
        for i, interval in enumerate(ranges):
            row = next(r for r in population if r['range'] == interval and r['seed'] == 'aggregate'
                       and r['comparison'] == 'RetiPath - ' + comparator)
            mean, low, high = [float(row[k]) for k in ['mean', 'ci_low', 'ci_high']]
            ax.errorbar(mean, i, xerr=[[mean - low], [high - mean]], fmt='o', color='#237dad', capsize=4)
            for j, seed in enumerate(SEEDS):
                point = next(r for r in population if r['range'] == interval and r['seed'] == str(seed)
                             and r['comparison'] == 'RetiPath - ' + comparator)
                ax.plot(float(point['mean']), i + (j - 1) * 0.14, '.', color='#999999', ms=4)
        ax.axvline(0, color='#777777', lw=0.8)
        ax.set_yticks(range(6), [x + ('  N=22' if i < 2 else '  N=17') for i, x in enumerate(ranges)])
        ax.invert_yaxis()
        ax.set_xlabel('RetiPath minus comparator NLL (nats/bin)')
        ax.set_title(comparator if comparator != CONTROL else 'without conductance\nintegration (internal control)')
    fig.suptitle('RetiPath | Frozen prediction benchmark\nBlue: seed aggregate and paired-cell 95% CI; gray: individual seeds; all ranges previously consumed', fontsize=12)
    save(fig, OUT / 'prediction/benchmark.png')


def rf_figures(rows: list[dict]) -> None:
    valid = [r for r in rows if r['status'] == 'VERIFIED']
    ranges = ['[16,20)', '[240,300)']
    metrics = ['gain_log_HIGH_over_LOW', 'TV', 'centroid_shift_deg',
               'delta_radius_observation_median_deg', 'centered_radial_CDF_L1_deg']
    labels = ['log(gain HIGH / LOW)', 'Normalized spatial profile TV', 'Centroid shift (mdeg)',
              'HIGH - LOW radius (mdeg)', 'Centered radial CDF L1 (mdeg)']
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), layout='constrained')
    for ax, metric, label in zip(axes, metrics, labels, strict=True):
        for j, interval in enumerate(ranges):
            cells = sorted({r['cell_id'] for r in valid if r['range'] == interval})
            values = np.array([np.mean([float(r[metric]) for r in valid if r['range'] == interval and r['cell_id'] == cell]) for cell in cells])
            if metric.endswith('_deg'):
                values *= 1000
            ax.scatter(j + np.linspace(-0.14, 0.14, len(cells)), values, s=18, alpha=0.8,
                       color=['#60859b', '#23617f'][j])
            ax.plot([j - 0.2, j + 0.2], [values.mean()] * 2, color='black', lw=2)
        ax.axhline(0, color='#999999', lw=0.7)
        ax.set_xticks([0, 1], ['[16,20)\n21 cells', '[240,300)\n17 cells'])
        ax.set_ylabel(label)
    fig.suptitle('RetiPath | Dynamic effective RF\nGain, normalized profile, centroid and size direction are distinct; points = cell means across seeds', fontsize=12)
    save(fig, OUT / 'dynamic_rf/overview.png')
    with np.load(CONFIRM / 'sampling_before_targets.npz', allow_pickle=False) as data:
        positions = data['positions']
    with (OUT / 'dynamic_rf/profiles/coordinates.npz').open('xb') as stream:
        np.savez_compressed(stream, cone_positions_deg=positions)
    folder = OUT / 'dynamic_rf/figures'
    folder.mkdir(exist_ok=True)
    pitch = np.diff(np.unique(positions[:, 0])).mean()
    extent = [float(positions[:, 0].min() - pitch / 2), float(positions[:, 0].max() + pitch / 2),
              float(positions[:, 1].min() - pitch / 2), float(positions[:, 1].max() + pitch / 2)]
    for interval in ranges:
        start = int(interval[1:].split(',')[0])
        for cell in sorted({r['cell_id'] for r in valid if r['range'] == interval}):
            selected = [r for r in valid if r['range'] == interval and r['cell_id'] == cell]
            means = {key: np.mean([json.loads(r[key]) for r in selected], axis=0)
                     for key in ['LOW_profile', 'HIGH_profile', 'delta_P', 'LOW_cdf', 'HIGH_cdf']}
            with np.load(OUT / 'dynamic_rf/profiles' / f"{cell.replace('#', '_')}_{start}_{SEEDS[0]}.npz") as data:
                grid = data['radial_grid_deg']
            fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), layout='constrained')
            pmax = max(means['LOW_profile'].max(), means['HIGH_profile'].max())
            dmax = np.abs(means['delta_P']).max()
            for i, key in enumerate(['LOW_profile', 'HIGH_profile', 'delta_P']):
                im = axes[i].imshow(means[key].reshape(17, 17), origin='upper', extent=extent, interpolation='nearest',
                    cmap='viridis' if i < 2 else 'RdBu_r', vmin=0 if i < 2 else -dmax, vmax=pmax if i < 2 else dmax)
                axes[i].set_title(['LOW P(x)', 'HIGH P(x)', 'HIGH - LOW P(x)'][i])
                axes[i].set_xlabel('x (deg)')
                axes[i].set_ylabel('y (deg)')
                fig.colorbar(im, ax=axes[i], shrink=0.75, format='%.2g')
            axes[3].plot(grid, means['LOW_cdf'], label='LOW')
            axes[3].plot(grid, means['HIGH_cdf'], label='HIGH')
            axes[3].set_title('Centered radial profile')
            axes[3].set_xlabel('Radius from own centroid (deg)')
            axes[3].set_ylabel('Cumulative sensitivity')
            axes[3].legend(fontsize=8)
            fig.suptitle(f'RetiPath | {cell} | {interval} s | three-seed mean; no area threshold', fontsize=12)
            save(fig, folder / f"{cell.replace('#', '_')}_{start}.png")


def brightness_figures() -> None:
    application_imports()
    import plot_application as original

    original.OUT = OUT / 'brightness/seed_aggregate'
    original.FIGURES = OUT / 'brightness/figures'
    original.FIGURES.mkdir(exist_ok=False)
    original.mach_figures(original.read_summary('mach'))
    original.sbc_figures(original.read_summary('sbc'))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for col, family in enumerate(['mach', 'sbc']):
        rows = read_rows(original.OUT / f'{family}_summary.csv')
        for row_index, quantity in enumerate(['logit', 'probability']):
            ax = axes[row_index, col]
            for k, condition in enumerate(['H1-off', 'direct-BC-off', 'AC-off']):
                points = [next(r for r in rows if r['quantity'] == quantity and r['group'] == group
                    and r['level'] == 'full_grid' and r['condition'] == condition and r['metric'] == 'absolute_interaction_magnitude')
                          for group in ['ALL', 'ON', 'OFF']]
                means = np.array([float(r['mean']) for r in points])
                lows = np.array([float(r['ci_low']) for r in points])
                highs = np.array([float(r['ci_high']) for r in points])
                ax.errorbar(np.arange(3) + (k - 1) * 0.18, means, yerr=[means - lows, highs - means],
                            fmt='o', capsize=3, label=condition, color=original.COLORS[condition])
            ax.set_xticks(range(3), ['ALL', 'ON', 'OFF'])
            ax.set_title(f"{'Mach' if family == 'mach' else 'SBC'} | {quantity}")
            ax.set_ylabel('Mean absolute interaction magnitude')
            ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.035), ncol=3)
    fig.suptitle('RetiPath | Frozen pathway-drive interventions\nPer-checkpoint magnitudes averaged within cell across three seeds; paired-cell 95% CI', fontsize=12)
    save(fig, original.FIGURES / 'pathway_magnitude.png')


def main() -> None:
    check = load_json(OUT / 'verification.json')
    assert check['status'] == 'VERIFIED'
    prediction = read_rows(OUT / 'prediction/population.csv')
    rf = read_rows(OUT / 'dynamic_rf/per_cell_seed.csv')
    rf_population = read_rows(OUT / 'dynamic_rf/population.csv')
    repeat = read_rows(OUT / 'dynamic_rf/seed_repeatability.csv')
    ranks = read_rows(OUT / 'brightness/pathway_rankings.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    prediction_plot(prediction)
    rf_figures(rf)
    brightness_figures()
    pred_table = []
    for row in prediction:
        if row['seed'] == 'aggregate':
            pred_table.append([row['range'], row['n_cells'], row['comparison'], f"{float(row['mean']):+.6f}",
                f"{float(row['median']):+.6f}", f"{row['wins']}/{row['losses']}/{row['ties']}",
                f"[{float(row['ci_low']):+.6f}, {float(row['ci_high']):+.6f}]"])
    rf_table = []
    for row in rf_population:
        if row['seed'] == 'aggregate':
            rf_table.append([row['range'], row['n_cells'], *[f'{float(row[k]):.6g}' for k in
                ['LOW_gain_median', 'HIGH_gain_median', 'TV', 'centroid_shift_deg',
                 'delta_radius_observation_median_deg', 'delta_radius_mean_profile_deg', 'centered_radial_CDF_L1_deg']]])
    app_table = []
    for family in ['mach', 'sbc']:
        for row in read_rows(OUT / 'brightness/seed_aggregate' / f'{family}_summary.csv'):
            if (row['quantity'] == 'logit' and row['group'] in ['ON', 'OFF'] and row['level'] == 'full_grid'
                    and row['condition'] == 'Normal' and row['metric'] == 'control_subtracted_response'):
                app_table.append([family, row['group'], row['endpoint'], f"{float(row['mean']):+.6f}",
                    f"{float(row['median']):+.6f}", f"[{float(row['ci_low']):+.6f}, {float(row['ci_high']):+.6f}]",
                    f"{row['negative']}/{row['zero']}/{row['positive']}", row['direction_stable']])
    ranking_table = [[r['family'], r['quantity'], r['group'], r['order'], r['paired_CI_winner'] or 'UNVERIFIED']
                     for r in ranks if r['seed'] == 'aggregate' and r['group'] in ['ALL', 'ON', 'OFF']]
    per_seed_negative = all(float(r['ci_high']) < 0 for r in prediction)
    assert per_seed_negative
    text = [
        '# RetiPath：正式模型的三项核心结果', '',
        '正式 RetiPath 已固定为两个有效空间模式的 E/I conductance architecture。三个核心结果已完成：在统一评价合同下优于现有 LN/CNN；具备超出整体 gain 的有效 RF 空间 profile 变化；冻结 Mach/SBC 应用呈现稳定 signed response，但 pathway ranking 的完整稳定性仍有限。全程没有训练、refit、新 variant、新机制或新 target block。', '',
        '迁移依据与历史结果边界见 [MIGRATION_PROTOCOL.md](MIGRATION_PROTOCOL.md)。历史 artifacts、文件名与 checkpoint 均保留原样。新正式入口与既有冻结实现的直接输出逐位相等；后续调用 `models.mechanistic_retina.retipath.RetiPath`，配置见项目 `configs/retipath_final.json`。旧入口仅维持兼容，不作为正式模型默认选择。', '',
        '## 1. Prediction benchmark', '',
        '**RetiPath 在六个已消费区间的 equal-cell mean NLL 均优于 LN、CNN 和内部历史架构对照；三个 seeds 各自以及 seed aggregate 的所有 mean 95% CI 均低于零。** 个别细胞仍可由 comparator 胜出，不能表述为每个细胞都更好。', '',
        '下表为 seed aggregate，单位 nats/scored bin；先在每个 cell 内平均三个冻结模型的 NLL 差，再汇总/paired-cell bootstrap，没有平均权重或 ensemble probability。W/L/T 为 RetiPath 胜/负/数值平局。每个 seed 的完整统计保存在 `prediction/population.csv`，逐 cell/seed 模型分数与差值保存在同目录另外两张 CSV。', '',
        markdown_table(['区间(s)', 'Cells', '比较', 'Mean ΔNLL', 'Median ΔNLL', 'W/L/T', 'Mean 95% CI'], pred_table), '',
        '![Prediction benchmark](prediction/benchmark.png)', '',
        '统一合同：相同 calibrated 289-cone inputs、150 Hz Bernoulli occupancy、150-bin 独立 sequence、30-bin warmup、120-bin scoring、strictly-past observed history 和相同 mask。Float32 forward，float64 NLL；与旧 LN/CNN 的 float32 汇总重放最大差为 5.96×10⁻⁸，属于数值汇总精度范围。', '',
        'LN/CNN 各使用一个既有 train-selected/fresh-refit checkpoint；RetiPath 与内部对照保留三个 Phase 2 refit seeds。统一的是评价输入和评分合同，不是训练预算、参数数量、先验或随机种子。RetiPath 37 个 trainable parameters，LN 128，CNN 2990；这些计数不能单独证明参数效率或机制可识别。', '',
        '`without conductance integration` 仅作内部 historical architecture control；它同时缺少新的空间表示，故这一比较不能解释成仅切掉电导的严格消融。该对照未被当作外部 baseline。所有六段在本轮开始前均已消费，因此本基准是 frozen descriptive re-evaluation，不是又一次独立 confirmation。22/17 的 cohort 差异按连续记录覆盖决定；区间不合并为额外生物独立样本。', '',
        '## 2. Dynamic effective RF', '',
        '**Gain modulation 与 normalized spatial-profile change 均存在；RF-size 没有跨时间块统一方向的必要性，也没有被强行赋予统一方向。** 下表分别列出 gain、TV、centroid shift、两种 radius 汇总和 centered radial profile 差异。角度单位为 degree。', '',
        markdown_table(['区间(s)', 'RF cells', 'LOW gain', 'HIGH gain', 'TV', 'Centroid shift', 'Δradius 观测中位数', 'Δradius mean P', 'Centered CDF L1'], rf_table), '',
        '![Dynamic effective RF](dynamic_rf/overview.png)', '',
        'Gain 为完整因果 prefix 的 logit Jacobian 范数；P 先逐观测归一化，再按 context 求均值。TV 比较 normalized P，因此整体乘性 gain 不足以解释它；centered radial CDF 先按各观测自身重心居中，仍有差异，说明变化也不只来自重心移动。Centroid 的绝对位置、HIGH/LOW radius、完整 P/ΔP/CDF 均保存在 `dynamic_rf/per_cell_seed.csv` 与 `dynamic_rf/profiles/`，38 张逐 cell/block profile 图在 `dynamic_rf/figures/`。', '',
        '在 development 的21个可用细胞中，20个的 ΔP 在三个 seeds 的全部两两比较中同向；后续已确认区间为17/17。Development 的68#10缺 HIGH 配对，保留为 UNVERIFIED，不补选 context。跨块相同16个细胞中，radius 变化符号仅5个一致：后段13个 HIGH 更大、4个更小。这里支持的是模型内部 context-dependent spatial profile，不是统一收缩/扩张规律，更不是对真实 RF 的恢复。', '',
        '这一部分整理的是与正式架构完全相同的既有 RF 数值，保留原 checkpoint、context 索引与来源哈希；没有把历史架构 RF 重新标为正式模型结果。两段各做了一组新入口重放。原始 RF 数值完整保留，最大重放差记录于 verification.json；舍入容差只用于工程复核，未增加 RF effect threshold。', '',
        '## 3. Frozen brightness-illusion application', '',
        '**按原冻结 logit 评价规则，三个 seeds 与 seed aggregate 均为 MIXED。** Mach 四个、SBC 两个主要 Normal signed summaries 都方向稳定；Mach 的最大 intervention 是 direct-BC-off，ON/OFF 均有 paired-CI 支持。SBC 的平均 magnitude 排名为 AC-off > direct-BC-off > H1-off，但 OFF 组尚无 paired-CI 稳定赢家，所以不能将整体应用写成 SUPPORTED。', '',
        '本轮对22 cells × 3 seeds × 166个注册刺激 × 4种 pathway 条件全部重新 frozen inference。原 Mach/SBC bank、22个 retinotopic centers、网格、active bins [45,60)、zero observed history 和评价公式完全不变。以下 logit mean/median/CI 与负/零/正 cell counts 都是在原固定网格内按原定义汇总，ON/OFF 不反号、不合并为知觉方向。', '',
        markdown_table(['Family', 'Group', 'Endpoint', 'Mean', 'Median', 'Mean 95% CI', '−/0/+', '方向稳定'], app_table), '',
        '![Mach response](brightness/figures/mach_normal_width.png)', '',
        '![SBC response](brightness/figures/sbc_normal_surface.png)', '',
        '三条 pathway 的 signed interaction 与 magnitude 分开报告。Intervention 是原路径驱动关闭：direct-BC-off/AC-off 将相应输入驱动置零，保留正式模型的基线电导；没有擅自把基线电导置零。Magnitude 先在每个 checkpoint 上取原逐点 absolute interaction，再在 cell 内平均 seeds，避免将正负抵消后的数值冒充 magnitude。', '',
        markdown_table(['Family', 'Quantity', 'Group', '平均 magnitude 排名', 'Paired-CI winner'], ranking_table), '',
        '![Pathway magnitude](brightness/figures/pathway_magnitude.png)', '',
        '![Mach signed interaction](brightness/figures/mach_pathway_interaction.png)', '',
        '![SBC signed interaction](brightness/figures/sbc_pathway_interaction.png)', '',
        '原规则的“方向稳定”只要求组内符号一致性和 paired-cell CI，不要求预先指定的知觉 sign。本结果不能被直接写成解释人类亮度错觉或定位真实视网膜必要通路。Logit 与 probability 的全部差异、七个 groups、全网格与 pathway pairwise differences 均保存在 `brightness/<seed>/` 及 `brightness/seed_aggregate/`；概率结果作为完整 secondary evidence，未用来改换 verdict。', '',
        '## 完整性与交付边界', '',
        f"- 来源锁定 {check['source_hash_count']} 个输入/来源条目；176 个本轮评分 checkpoint，历史 selected/refit checkpoint 也保留。模型实现和历史 artifacts 未改写。正式入口仅固定后端与命名，没有增加参数或计算机制。",
        f"- 896 个模型-区间评分、112 个 cell-range 输入合同；独立 NLL 复算最大误差 {check['prediction']['independent_NLL_max_abs_error']:.3g}，bootstrap 汇总复算误差 {check['prediction']['independent_bootstrap_max_abs_error']:.3g}。",
        f"- RF 共117个 cell-seed/block记录，其中114个配对可用；两个完整 prefix 重放保留浮点误差记录。Mach/SBC 共66个冻结模型、5544次批量 forward；独立复算6336个 full-grid效应，最大误差 {check['brightness']['manual_effect_max_abs_error']:.3g}。",
        '- 科学来源锁保持原样。新 RF 整理脚本的逐位汇总检查曾过严，旧加载器也出现相同 binary64 舍入差；工程修正及修正前源码保存在 execution_correction.json 与 provenance/pre_correction/，没有更改模型、旧 RF 数值或科学指标。',
        '- 0 training、0 refit、0 new variant、0 new physiological mechanism、0 new target block。三项正式模型 evidence 已完成；本轮到此停止。', '',
    ]
    with (OUT / 'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(text))
    write_once(OUT / 'report_generation.json', {'completed_utc': utc(), 'report_source_sha256': sha(Path(__file__)),
        'plot_source_sha256': sha(OUT.parents[2] / '.omo/evidence/retipath-brightness-application-20260910/plot_application.py'),
        'figures': len(list(OUT.rglob('*.png'))), 'all_model_display_names_migrated': True})
    print('FINAL REPORT AND FIGURES WRITTEN')


if __name__ == '__main__':
    main()
