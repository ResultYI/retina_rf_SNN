from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import subprocess

import numpy as np

from temporal_features import OUT, save_json, sha

ROOT = OUT.parents[2]


def read(name: str) -> list[dict]:
    with (OUT / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def num(value: str | float | None) -> str:
    return f'{float(value):+.6f}' if value is not None and value != '' else 'UNAVAILABLE'


def ci(row: dict, kind: str = 'mean') -> str:
    return f"[{num(row[kind+'_ci_low'])}, {num(row[kind+'_ci_high'])}]"


def main() -> None:
    protocol = json.loads((OUT / 'protocol_hash.json').read_text())
    assert sha(OUT / 'PROTOCOL.md') == protocol['sha256']
    lock = json.loads((OUT / 'development_verdict_lock.json').read_text())
    assert sha(OUT / 'development_verdict.json') == lock['verdict_sha256']
    for p, h in lock['development_artifact_sha256'].items():
        assert sha(OUT / p) == h
    assert sha(OUT / 'per_bin_temporal_features_development.npz') == lock['per_bin_sha256']
    assert sha(OUT / 'feature_definition_lock.json') == lock['feature_lock_sha256']
    for split in ('development', 'consumed'):
        assert json.loads((OUT / (split+'_analysis_status.json')).read_text())['status'] == 'COMPLETE'
    verdict = json.loads((OUT / 'development_verdict.json').read_text())
    qrows = read('development_adaptation_quantile_nll.csv')
    population = {(r['comparator'], r['estimate']): r for r in read('development_population_adaptation_excess_loss.csv')}
    outliers = {r['comparator']: r for r in read('outlier_sensitivity.csv')}
    precision = max(abs(float(r['delta'])) for r in read('precision_comparison.csv'))

    def qmean(q: int, name: str) -> float:
        return float(np.mean([float(r[name]) for r in qrows if int(r['quintile']) == q]))

    text = ['# Phase T1 — temporal/adaptation-dependent residual phenotype', '',
        '## 1. High recent adaptation-load stimuli 上是否存在更大的 absolute prediction loss？', '',
        '没有。Development `[16,20)` 的 Canonical equal-cell absolute NLL 在 A-Q5 较 A-Q1 更低。以下是22-cell等权均值；NLL单位为 nats/phenotype bin。', '',
        '| Development | A-Q1 | A-Q5 | Q5−Q1 |', '|---|---:|---:|---:|']
    for label in ('aligned', 'LN', 'CNN'):
        low, high = qmean(1, label+'_nll'), qmean(5, label+'_nll')
        text.append(f'| {label} NLL | {low:.6f} | {high:.6f} | {high-low:+.6f} |')
    low, high = qmean(1, 'event_rate'), qmean(5, 'event_rate')
    text += [f'| Observed event rate | {low:.6f} | {high:.6f} | {high-low:+.6f} |', '',
        '66/66 strict exact replay 通过，原 scored-mask logits、NLL、输入、target、mask 和 source/trial order 均精确一致。本 phenotype 仅保留 local t45..149：548 sequences、57,540 bins；原65,760-bin模型评价不变。A 使用严格过去45bins，不含当前bin、不跨sequence；C固定k5。', '',
        f'分层分析使用 float64 loss。两个区间全部cell/model的 phenotype-mask float64与原生float32汇总最大数值差为 {precision:.8g} nats/bin，不改变原NLL exact replay门槛。', '',
        '## 2. 这种变化是否相对 LN/CNN 特异？', '',
        '没有发现预注册方向的 comparator-relative deficit：Canonical−LN 的 A high-vs-low E 为负且mean interval位于零以下；Canonical−CNN的点估计也为负，interval跨零。不能将 absolute loss、event rate变化解释为 adaptation 缺失。']
    for c, number in (('LN', 3), ('CNN', 4)):
        r = population[c, 'raw']
        deleted = outliers[c]
        text += ['', f'## {number}. Canonical−{c} 的 adaptation high-vs-low excess NLL 是多少？', '',
            f"Mean E **{num(r['mean'])}**，paired-cell bootstrap 95% CI **{ci(r)}**；median {num(r['median'])}，median95% CI {ci(r, 'median')}。正/负/零细胞：{r['positive']}/{r['negative']}/{r['zero']}。", '',
            f"唯一预注册删一细胞检查：删除最大|E|的 {deleted['removed_cell']} 后，21-cell mean E 为 {num(deleted['mean_after_single_deletion'])}。"]
    text += ['', '## 5. 控制 current |drive| 后 effect 还剩多少？', '',
        '| Comparator | Raw mean E | U-matched mean E | Mean95% CI | Median | 正/负/零 cells |', '|---|---:|---:|---|---:|---:|']
    for c in ('LN', 'CNN'):
        r = population[c, 'U_matched']
        text.append(f"| Canonical−{c} | {num(population[c, 'raw']['mean'])} | {num(r['mean'])} | {ci(r)} | {num(r['median'])} | {r['positive']}/{r['negative']}/{r['zero']} |")
    controls = [r for r in read('development_current_drive_matched_control.csv') if r['comparator'] == 'LN']
    overlap = np.asarray([float(r['low_overlap_fraction']) for r in controls])
    u_residual = np.asarray([float(r['residual_U_difference']) for r in controls])
    c_residual = np.asarray([float(r['residual_C_difference']) for r in controls])
    text += ['', f'22个细胞均有U-stratum overlap；overlap mass/Q1（Q5相同）中位数 {np.median(overlap):.1%}，范围 {overlap.min():.1%}–{overlap.max():.1%}。加权后Q5−Q1的连续U差异中位数 {np.median(u_residual):.6f}，范围 [{u_residual.min():.6f}, {u_residual.max():.6f}] Weber drive；C差异中位数 {np.median(c_residual):.6f}。五分位分层没有完全消除连续当前drive或recent-change共变。', '',
        'Stimulus-only条件触发了预注册U×C二维描述性控制（全22cells，无bootstrap、不参与verdict）：', '',
        '| Comparator | U×C-matched mean E | Median | 正/负 cells |', '|---|---:|---:|---:|']
    for r in read('development_two_dimensional_matched_population.csv'):
        text.append(f"| Canonical−{r['comparator']} | {num(r['mean'])} | {num(r['median'])} | {r['positive']}/{r['negative']} |")
    text += ['', '## 6. Fast temporal change C 是否表现出类似 phenotype？', '',
        '未发现正向的 population point estimate。C仅为secondary；这些结果不能被自动称为 adaptation deficit。', '',
        '| Comparator | C mean E | Median | 正/负/零 cells |', '|---|---:|---:|---:|']
    for r in read('development_population_temporal_change_excess_loss.csv'):
        text.append(f"| Canonical−{r['comparator']} | {num(r['mean'])} | {num(r['median'])} | {r['positive']}/{r['negative']}/{r['zero']} |")
    text += ['', '| Per-cell Spearman(feature, relative loss) | Mean | Median | 正/负 cells |', '|---|---:|---:|---:|']
    for r in read('development_continuous_trend_population.csv'):
        text.append(f"| {r['feature']} / Canonical−{r['comparator']} | {num(r['mean'])} | {num(r['median'])} | {r['positive']}/{r['negative']} |")
    text += ['', '## 7. 哪些 cell classes 最明显？', '',
        '相对CNN，MC_OFF的raw/matched均值为正；PC_OFF均值弱正而中位数为负。相对LN四类均值均为负。仅作描述，不据此筛选primary cohort。每格为mean / median adaptation E。', '',
        '| Class | N | LN raw | LN U-matched | CNN raw | CNN U-matched |', '|---|---:|---:|---:|---:|---:|']
    grows = read('development_cell_group_summary.csv')
    for group in ('MC_ON', 'MC_OFF', 'PC_ON', 'PC_OFF'):
        g = {(r['comparator'], r['estimate']): r for r in grows if r['group'] == group}
        values = [f"{num(g[c, e]['mean'])} / {num(g[c, e]['median'])}" for c, e in (('LN', 'raw'), ('LN', 'U_matched'), ('CNN', 'raw'), ('CNN', 'U_matched'))]
        text.append(f"| {group} | {g['LN', 'raw']['cells']} | "+' | '.join(values)+' |')
    text += ['', '## 8. Consumed descriptive reuse 与 development 方向是否一致？', '',
        '下表使用已consumed `[20,60)`、575,400 phenotype bins，以及development冻结的A/C/U数值阈值。没有重新定义quintiles；仅作描述性复用，不是独立确认。', '',
        '| Feature / comparison / estimate | Development mean E | Consumed mean E | 均值方向一致 | 逐细胞符号一致 |', '|---|---:|---:|---|---:|']
    for r in read('development_vs_consumed_descriptive.csv'):
        text.append(f"| {r['feature']} / {r['comparator']} / {r['estimate']} | {num(r['development_mean'])} | {num(r['consumed_descriptive_mean'])} | {'是' if r['mean_direction_consistent'] == 'True' else '否'} | {r['sign_consistent_cells']}/{r['total_cells']} |")
    text += ['', '## 9. 最终 verdict 是 GO / MIXED / NO-GO？', '', f"**{verdict['verdict']}**", '',
        '**NO-GO FOR TEMPORAL/ADAPTATION FRONT-END**', '',
        '判定只使用development：两个comparator的raw A mean均不为正，U-matched均值也均为负，因此不满足GO或MIXED的冻结条件。Development verdict在读取consumed phenotype前已写入并SHA256封存，之后未改变。', '',
        '该结论限于本轮冻结特征、窗口、数据和模型，不能证明真实视网膜没有adaptation机制。A只是stimulus-derived adaptation-load proxy。本轮front-end/core/生产数据/模型参数修改、training、parameter fitting、新seed和Phase T2实施均为0。', '',
        '目前没有足够证据进入 Phase T2 的 dynamic front-end mechanism experiment。', '']
    assert verdict['verdict'] == 'NO-GO — NO TEMPORAL/ADAPTATION-SPECIFIC DEFICIT'
    (OUT / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    save_json('per_bin_temporal_features_schema.json', dict(format='NumPy compressed NPZ; allow_pickle=False', files=['per_bin_temporal_features_development.npz', 'per_bin_temporal_features_consumed.npz'],
        key_format='cell_id replacing # with _ + __ + field', row_order='original sequence/time order restricted to phenotype mask',
        row_fields='A,C,M,U,A_quintile,C_quintile,U_quintile,source_image_id,recording,local_trial_one_based,global_trial_zero_based,sequence_index,time_bin,live_frame,decoded_frame,time_seconds,target,aligned_loss,LN_loss,CNN_loss',
        metadata_fields='weights,center,A_edges,C_edges,U_edges,cell_id,group', full_shape_mask_fields='phenotype_mask,original_valid_mask',
        A='sqrt(mean of M[t-45:t]^2), strictly past45,300ms', C='abs(M[t]-M[t-5]),33.333ms', phenotype_mask='original mask AND local t>=45; no padding/cross-sequence history', feature_unit='dimensionless Weber drive', loss_unit='nats/phenotype bin',
        frame_zero_provenance='UNKNOWN; frozen decoded origin751 preserved', consumed_thresholds='exact numerical development thresholds', bootstrap=json.loads((OUT / 'bootstrap_identity.json').read_text())))
    initial = json.loads((OUT / 'evidence_manifest.initial.json').read_text())
    changed = [p for p, h in initial['input_sha256'].items() if sha(ROOT / p) != h]
    assert not changed
    save_json('evidence_manifest.json', dict(status='COMPLETE', created_utc=datetime.now(timezone.utc).isoformat(), branch=initial['branch'], HEAD=initial['HEAD'],
        git_status_before=initial['git_status_before'], git_status_after=subprocess.check_output(['git', 'status', '--short', '--untracked-files=all'], cwd=ROOT, text=True),
        input_sha256=initial['input_sha256'], input_count=len(initial['input_sha256']), input_hash_changes=changed, checkpoint_identity=initial['checkpoint_identity'], aligned_geometry=initial['aligned_geometry'],
        exact_replay_pairs=66, all_exact_replay=True, original_development_scored_bins=65760, development_phenotype_bins=57540, consumed_phenotype_bins=575400,
        protocol_sha256=protocol['sha256'], protocol_unchanged=True, development_verdict_sha256=lock['verdict_sha256'], development_verdict=verdict['verdict'], verdict_data='development[16,20) only', consumed_is_independent_confirmation=False,
        training=0, parameter_fitting=0, new_seed=0, front_end_modifications=0, core_modifications=0, production_modifications=0, checkpoint_modifications=0, new_pathway_clamp_analysis=0, phase_T2_implementation=0,
        max_phenotype_float64_vs_native_nll_difference=precision,
        output_sha256={p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'evidence_manifest.json' and '__pycache__' not in p.parts}))
    print('REPORT complete; unchanged inputs', len(initial['input_sha256']))


if __name__ == '__main__':
    main()
