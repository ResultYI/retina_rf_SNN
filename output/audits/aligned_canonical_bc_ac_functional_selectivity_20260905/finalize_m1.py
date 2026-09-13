from __future__ import annotations

import json
import subprocess

import numpy as np
import pyarrow.parquet as pq

from derive_development import OUT, ROOT, DERIVED, IDENTITIES, now, read_csv, sha, write_json


def number(value: str | float | None) -> str:
    return f'{float(value):+.6f}' if value is not None and value != '' else 'UNDEFINED'


def interval(row: dict, kind: str = 'mean') -> str:
    return '['+number(row[kind+'_ci_low'])+', '+number(row[kind+'_ci_high'])+']'


def main() -> None:
    protocol = json.loads((OUT / 'protocol_hash.json').read_text())
    assert sha(OUT / 'PROTOCOL.md') == protocol['sha256']
    verdict_sha = json.loads((OUT / 'development_verdict_sha256.json').read_text())['sha256']
    assert sha(OUT / 'development_verdict_lock.json') == verdict_sha
    decision = json.loads((OUT / 'development_verdict_lock.json').read_text())
    for p, h in decision['development_artifact_sha256'].items():
        assert sha(OUT / p) == h
    assert sha(OUT / 'feature_lock.json') == decision['feature_lock_sha256']
    for p, h in json.loads((DERIVED / 'derivation_lock.json').read_text())['sha256'].items():
        assert sha(DERIVED / p) == h
    for split in ('development', 'consumed'):
        assert json.loads((OUT / (split+'_status.json')).read_text())['status'] == 'COMPLETE'
    pop = {r['quantity']: r for r in read_csv(OUT / 'development_population_context_selectivity.csv')}
    cells = read_csv(OUT / 'development_per_cell_context_selectivity.csv')
    outlier = {r['quantity']: r for r in read_csv(OUT / 'outlier_sensitivity.csv')}
    pf = json.loads((OUT / 'preflight.json').read_text())
    max_subtraction = max(c['subtraction_max_error'] for r in pf['cells'] for c in r['checks'])
    text = ['# Phase M1 — direct-BC versus downstream AC functional selectivity', '',
        '## 1. AC predictive consequence是否随 central-vs-broad context mismatch 增大？', '',
        'Development `[16,20)` 未提供清晰支持：AC的high-S minus low-S mean E接近零，interval跨零，median为负；不是稳定的正向context selectivity。所有primary统计使用22-cell等权、原始scored bins（65,760 bins），E单位nats/bin。', '',
        'S=|R−C|，C是固定BC support的均匀平均drive，R是固定AC\\BC context的均匀平均drive。Δloss_P=loss(P-off,recalibrated)−loss(normal,uncalibrated)。下文“raw E”指尚未归一化的selectivity，仍由recalibrated off logits计算；不是未校准pathway-off penalty。', '',
        '**证据来源分开记录：**历史/frozen evidence是development raw normal/off logits及training-only biases。本轮按明确授权首次计算raw_off.double()+frozen_bias，产生 **NEW_DETERMINISTIC_DERIVED_ARTIFACT**：development recalibrated logits/NLL，以及后续M1统计。它们不是历史recalibrated replay；确定性派生本身不降低其作为本轮输入的有效性。', '',
        f'22个aligned checkpoints strict-load，normal/BC-off/AC-off共66组raw重放与原artifact逐位一致；44个bias源/lock精确验证，44组新派生加法逐位一致。减法核查最大舍入残差 {max_subtraction:.8g}，均在预注册float64舍入界限内。未调用solver、未拟合bias。', '',
        '## 2. direct-BC 是否也表现相同关系？', '',
        'direct-BC的raw mismatch E均值同样为正但区间跨零；不能据raw scale宣称AC更selective。Architecture-derived central-magnitude comparison仅为secondary：', '',
        '| U-Q5−U-Q1 consequence | Mean G | Median | 正/负 cells |', '|---|---:|---:|---:|']
    for key, label in (('G_BC', 'direct-BC'), ('G_AC', 'AC')):
        r = pop[key]
        text.append(f"| {label} | {number(r['mean'])} | {number(r['median'])} | {r['positive']}/{r['negative']} |")
    text += ['', '| Per-cell Spearman | Mean | Median | 正/负 cells |', '|---|---:|---:|---:|']
    for r in read_csv(OUT / 'development_continuous_population.csv'):
        text.append(f"| {r['feature']} / {r['pathway']} | {number(r['mean'])} | {number(r['median'])} | {r['positive']}/{r['negative']} |")
    for n, key, label in ((3, 'E_AC', 'AC'), (4, 'E_BC', 'direct-BC')):
        r = pop[key]
        text += ['', f'## {n}. {label} raw E 是多少？', '',
            f"Mean E **{number(r['mean'])}**，paired-cell bootstrap mean95% CI **{interval(r)}**；median {number(r['median'])}，median95% CI {interval(r, 'median')}。正/负/零cells：{r['positive']}/{r['negative']}/{r['zero']}。"]
        if key == 'E_AC':
            r = outlier[key]
            text += ['', f"唯一E_AC outlier检查：删除最大|E_AC|的 {r['removed_cell']} 后，mean为 {number(r['mean_after_single_deletion'])}。"]
    d = pop['D']
    text += ['', '## 5. Normalized AC-vs-BC differentiation D 是多少？', '',
        f"Mean D **{number(d['mean'])}**，95% CI **{interval(d)}**；median {number(d['median'])}，median95% CI {interval(d, 'median')}；正/负cells {d['positive']}/{d['negative']}。不支持AC normalized selectivity更强。", '',
        f"Mean N_AC={number(pop['N_AC']['mean'])}，mean N_BC={number(pop['N_BC']['mean'])}。归一化使用各cell/pathway全部analysis bins平均penalty；44个denominator均正且超过预注册数值零界限，无epsilon、clip或排除cell。BC denominator范围 [{min(float(r['B_BC']) for r in cells):.6f}, {max(float(r['B_BC']) for r in cells):.6f}]；AC [{min(float(r['B_AC']) for r in cells):.6f}, {max(float(r['B_AC']) for r in cells):.6f}]。", '',
        f"唯一D outlier检查：删除 {outlier['D']['removed_cell']} 后，mean D={number(outlier['D']['mean_after_single_deletion'])}。", '',
        '## 6. 控制 central |drive| 后 AC effect还剩多少？', '',
        '| U-matched quantity | Mean | Mean95% CI | Median | 正/负 cells |', '|---|---:|---|---:|---:|']
    for key in ('E_AC_U', 'E_BC_U', 'D_U'):
        r = pop[key]
        text.append(f"| {key} | {number(r['mean'])} | {interval(r)} | {number(r['median'])} | {r['positive']}/{r['negative']} |")
    control = [r for r in read_csv(OUT / 'development_central_drive_matched_control.csv') if r['pathway'] == 'AC']
    overlap = np.asarray([float(r['Q1_overlap_fraction']) for r in control])
    residual = np.asarray([float(r['residual_U_difference']) for r in control])
    text += ['', f'AC的raw弱正均值在U-matching后转负。22cells均有overlap；overlap mass/Q1（Q5相同）中位数 {np.median(overlap):.1%}，范围 {overlap.min():.1%}–{overlap.max():.1%}。连续U的加权Q5−Q1差异中位数 {np.median(residual):.6f} Weber drive，范围 [{residual.min():.6f}, {residual.max():.6f}]，不是精确连续匹配。Normalized U-control仍使用原无条件denominator。', '',
        '## 7. 控制 generic spatial heterogeneity 后还剩多少？', '',
        '| LSC_AC-matched | Mean E | Median | 正/负 cells |', '|---|---:|---:|---:|']
    for key in ('E_AC_LSC', 'E_BC_LSC'):
        r = pop[key]
        text.append(f"| {key} | {number(r['mean'])} | {number(r['median'])} | {r['positive']}/{r['negative']} |")
    control = [r for r in read_csv(OUT / 'development_lsc_matched_control.csv') if r['pathway'] == 'AC']
    overlap = np.asarray([float(r['Q1_overlap_fraction']) for r in control])
    residual = np.asarray([float(r['residual_LSC_difference']) for r in control])
    text += ['', f'这是预注册secondary描述性控制，没有额外bootstrap；AC正向effect没有保留，因而不能支持特异context解释。Overlap mass/Q1中位数 {np.median(overlap):.1%}，范围 {overlap.min():.1%}–{overlap.max():.1%}；残余LSC差异中位数 {np.median(residual):.6f}，范围 [{residual.min():.6f}, {residual.max():.6f}]。有限overlap不允许声称LSC已被完全排除或已确定解释全部effect。', '',
        '## 8. 哪些 cell classes 最明显？', '',
        '仅描述，不做group tests或更改primary cohort。每格为mean / median；normalized值和D无量纲。Signed Q<0/Q>0的AC描述见signed_context_descriptive.csv，不进入判定。', '',
        '| Class | N | E_AC | E_BC | N_AC | N_BC | D |', '|---|---:|---:|---:|---:|---:|---:|']
    grows = read_csv(OUT / 'development_cell_group_summary.csv')
    for g in ('MC_ON', 'MC_OFF', 'PC_ON', 'PC_OFF'):
        lookup = {r['quantity']: r for r in grows if r['group'] == g}
        values = [f"{number(lookup[k]['mean'])} / {number(lookup[k]['median'])}" for k in ('E_AC', 'E_BC', 'N_AC', 'N_BC', 'D')]
        text.append(f"| {g} | {lookup['E_AC']['cells']} | "+' | '.join(values)+' |')
    text += ['', '## 9. Consumed descriptive reuse方向是否一致？', '',
        '| Quantity | Development mean | Consumed mean | Population方向一致 | Cell符号一致 |', '|---|---:|---:|---|---:|']
    for r in read_csv(OUT / 'development_vs_consumed_descriptive.csv'):
        text.append(f"| {r['quantity']} | {number(r['development_mean'])} | {number(r['consumed_descriptive_mean'])} | {'是' if r['population_direction_consistent'] == 'True' else '否'} | {r['sign_consistent_cells']}/{r['total_cells']} |")
    text += ['', 'Consumed `[20,60)`使用相同geometry/bias及development冻结S/U/LSC阈值。44组consumed recalibrated NLL与上一轮已记录值exact；没有新inference或bias拟合。D的population方向转为正，但仅8/22逐细胞符号一致。该范围不是新的independent confirmation，不能修订development判定。', '',
        '## 10. 最终 verdict：GO / MIXED / NO-GO？', '', f"**{decision['verdict']}**", '',
        '按冻结的有序规则，AC raw mean>0，但CI跨零、单一最大贡献cell删除后转负、U/LSC控制后均值为负，normalized D不支持AC更强，因此为MIXED。该结果没有证明distinct AC context selectivity，也不是对真实生物AC功能的唯一解释。', '',
        'Development verdict先写入并SHA256锁定，之后才读取consumed prediction payload。历史停止记录已保留；本轮仅解除用户明确授权的确定性派生阻断。训练、拟合、model/front-end/center/support/生产文件修改、新seed、artificial stimuli及M2均为0。', '',
        '当前没有足够证据进入 Phase M2 的 artificial matched-context circuit prediction experiment。', '']
    assert decision['verdict'] == 'MIXED — FUNCTIONAL SELECTIVITY WEAK OR NONSPECIFIC'
    (OUT / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    write_json(OUT / 'stimulus_features_schema.json', dict(format='Parquet zstd', path='stimulus_features.parquet', rows=pq.ParquetFile(OUT / 'stimulus_features.parquet').metadata.num_rows,
        development_rows=65760, consumed_rows=657600, row_order='split then original cell/sequence/time scored order', geometry_files='development_features_*.npz exact masks/weights/centers',
        C='BC uniform drive', R='AC minus BC uniform context drive', S='abs(R-C)', Q='R-C', U='abs(C)', V='abs(R)', LSC='fullAC uniform spatial standard deviation',
        feature_units='dimensionless Weber drive', loss_units='nats per scored bin', provenance='Historical raw predictions/frozen biases; newly derived development calibrated outputs and selectivity statistics',
        normal_calibrated=False, consumed_thresholds='fixed numerical development thresholds', decoded_frame_origin_provenance='UNKNOWN; frozen751 preserved'))
    initial = json.loads((OUT / 'continuation_initial_manifest.json').read_text())
    changed = [p for p, h in initial['input_sha256'].items() if sha(ROOT / p) != h]
    assert not changed
    write_json(OUT / 'evidence_manifest.json', dict(status='COMPLETE', created_utc=now(), branch=initial['branch'], HEAD=initial['HEAD'], git_status_before=initial['git_status_before'],
        git_status_after=subprocess.check_output(['git', 'status', '--short', '--untracked-files=all'], cwd=ROOT, text=True), input_sha256=initial['input_sha256'], input_count=len(initial['input_sha256']), input_hash_changes=changed,
        aligned_checkpoint_identity=IDENTITIES, protocol_sha256=protocol['sha256'], protocol_unchanged=True, preflight_status=pf['status'], raw_exact_replay_pairs=66,
        derived_artifact_classification='NEW_DETERMINISTIC_DERIVED_ARTIFACT', derived_conditions=44, derivation_manifest_sha256=sha(DERIVED / 'derivation_manifest.json'),
        historical_recalibrated_development_replay=False, arithmetic_subtraction_max_error=max_subtraction, development_verdict=decision['verdict'], development_verdict_sha256=verdict_sha,
        decision_data='development [16,20) only', consumed_independent_confirmation=False, training=0, fitting=0, new_seed=0, production_modifications=0, checkpoint_modifications=0, M2=0,
        output_sha256={p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*') if p.is_file() and p != OUT / 'evidence_manifest.json' and '__pycache__' not in p.parts}))
    print('M1 REPORT + MANIFEST', len(initial['input_sha256']), 'unchanged inputs')


if __name__ == '__main__':
    main()
