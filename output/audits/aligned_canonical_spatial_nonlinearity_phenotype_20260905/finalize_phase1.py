from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from stimulus_features import OUT, sha

ROOT = OUT.parents[2]


def read(name: str) -> list[dict]:
    with (OUT / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    frozen = json.loads((OUT / 'protocol_hash.json').read_text())
    assert sha(OUT / 'PROTOCOL.md') == frozen['sha256']
    verdict_hash = json.loads((OUT / 'development_verdict_hash.json').read_text())['sha256']
    assert sha(OUT / 'development_verdict.json') == verdict_hash
    verdict = json.loads((OUT / 'development_verdict.json').read_text())
    for path, digest in verdict['development_artifact_sha256'].items():
        assert sha(OUT / path) == digest
    assert json.loads((OUT / 'development_analysis_status.json').read_text())['status'] == 'COMPLETE'
    assert json.loads((OUT / 'consumed_analysis_status.json').read_text())['status'] == 'COMPLETE'
    quantile = read('development_quantile_nll.csv')
    pop = {(r['comparator'], r['estimate']): r for r in read('development_population_excess_loss.csv')}
    loo = {r['comparator']: r for r in read('outlier_sensitivity.csv')}
    groups = read('development_cell_group_summary.csv')
    per_cell = [r for r in read('development_mean_matched_control.csv') if r['comparator'] == 'LN']

    def number(value: str | float) -> str:
        return f'{float(value):+.6f}'

    def interval(row: dict) -> str:
        return f"[{number(row['mean_ci_low'])}, {number(row['mean_ci_high'])}]"

    def qmean(k: int, field: str) -> float:
        return float(np.mean([float(r[field]) for r in quantile if int(r['quantile']) == k]))

    text = ['# Phase 1 — spatial-contrast-dependent residual phenotype', '',
        '## 1. Aligned Canonical 是否在 high-LSC natural stimuli 上出现更大的 prediction residual？', '',
        'Development live `[16,20)` 的 absolute Bernoulli NLL 在 Q5 较 Q1 增加；这本身不是 BC nonlinear deficit 的证据。以下均为22个细胞等权均值，单位 nats/scored bin。', '',
        '| Development | Q1 | Q5 | Q5−Q1 |', '|---|---:|---:|---:|']
    for label in ('aligned', 'LN', 'CNN'):
        low, high = qmean(1, label+'_nll'), qmean(5, label+'_nll')
        text.append(f'| {label} NLL | {low:.6f} | {high:.6f} | {high-low:+.6f} |')
    low, high = qmean(1, 'event_rate'), qmean(5, 'event_rate')
    text.append(f'| Observed event rate | {low:.6f} | {high:.6f} | {high-low:+.6f} |')
    precision = max(abs(float(r['delta'])) for r in read('precision_comparison.csv'))
    text += ['', '66/66 cell/model exact replay 通过：strict load、输入、target、mask、顺序、bitwise logits、原始 float32 NLL 一致。LSC 是 aligned checkpoint 固定 BC support 内均匀加权的原始 Weber-drive 空间标准差，未使用输出选择特征。', '',
        f'分层分析使用 float64 per-bin loss；它与原生 float32 NLL 汇总的最大数值差为 {precision:.8g} nats/bin（两个区间、全部模型/细胞）。这不改变 exact replay 门槛。', '',
        '## 2. 是普遍变难，还是相对 LN/CNN 特异变差？', '',
        '三个模型的 absolute loss 均增加。Canonical 相对 LN 的 raw excess loss 增加，但 mean interval 跨零；相对 CNN 的 raw excess loss 减少且 interval 在零以下，因此不支持一致的 comparator-relative deficit。', '',
        '| Development per-cell Spearman(LSC, relative loss) | mean | median | 正/负 cells |', '|---|---:|---:|---:|']
    for r in read('development_continuous_trend_population.csv'):
        text.append(f"| Canonical−{r['comparator']} | {number(r['mean'])} | {number(r['median'])} | {r['positive']}/{r['negative']} |")
    for c, n in (('LN', 3), ('CNN', 4)):
        r = pop[c, 'raw']
        text += ['', f'## {n}. Canonical−{c} 的 high-vs-low excess NLL 是多少？', '',
            f"Mean E **{number(r['mean'])}**，paired-cell bootstrap 95% CI **{interval(r)}**；median {number(r['median'])}，median 95% CI [{number(r['median_ci_low'])}, {number(r['median_ci_high'])}]；正/负/零细胞 {r['positive']}/{r['negative']}/{r['zero']}。",
            '', f"唯一预注册删一细胞检查：删除最大 |E| 的 {loo[c]['removed_cell']} 后，21-cell mean 为 {number(loo[c]['mean_E_after_single_deletion'])}。"]
    text += ['', '## 5. 控制 local mean intensity 后 effect 还剩多少？', '',
        '| Comparator | Raw mean E | Matched mean E | Matched mean 95% CI | Matched median | 正/负 cells |', '|---|---:|---:|---|---:|---:|']
    for c in ('LN', 'CNN'):
        raw, m = pop[c, 'raw'], pop[c, 'matched']
        text.append(f"| Canonical−{c} | {number(raw['mean'])} | {number(m['mean'])} | {interval(m)} | {number(m['median'])} | {m['positive']}/{m['negative']} |")
    overlap = np.array([float(r['low_overlap_fraction']) for r in per_cell])
    remaining_mean = np.array([float(r['matched_mean_drive_difference']) for r in per_cell])
    text += ['', 'LN 的 matched 点估计未消失；CNN 从 raw 负值变为 matched 正值。但两者 matched mean intervals 均跨零。', '',
        f'控制按预注册 M 五分位进行，同一层内比较全局 Q5/Q1，并按 min(nQ1,nQ5) 加权。仅2–3个 M strata/细胞有共同支持；overlap mass / Q1（Q5相同）的范围为 {overlap.min():.1%}–{overlap.max():.1%}，中位数 {np.median(overlap):.1%}。匹配后仍有连续 local mean 差异：细胞中位数 {np.median(remaining_mean):.6f}，范围 [{remaining_mean.min():.6f}, {remaining_mean.max():.6f}] Weber drive。故不能宣称完全排除了 mean-intensity 共变。', '',
        '## 6. 哪些 cell classes 最明显？', '',
        '正向 LN 点估计主要见于 MC_ON、其次 PC_ON；OFF 群体未呈现同样方向。下表仅描述，无 group tests 或 subset selection。每格为 mean / median E。', '',
        '| Class | N | LN raw | LN matched | CNN raw | CNN matched |', '|---|---:|---:|---:|---:|---:|']
    for g in ('MC_ON', 'MC_OFF', 'PC_ON', 'PC_OFF'):
        selected = {(r['comparator'], r['estimate']): r for r in groups if r['group'] == g}
        cells = selected['LN', 'raw']['cells']
        values = [f"{number(selected[c, k]['mean'])} / {number(selected[c, k]['median'])}" for c, k in (('LN', 'raw'), ('LN', 'matched'), ('CNN', 'raw'), ('CNN', 'matched'))]
        text.append(f"| {g} | {cells} | " + ' | '.join(values) + ' |')
    text += ['', '## 7. 结果是 GO / MIXED / NO-GO？', '', f"**{verdict['verdict']}**。Development 中没有 comparator 同时满足预注册 raw mean CI >0、删一细胞后 mean >0、matched mean CI >0 与至少50%效应保留。LN 的 raw/matched 均值同为正，因此按冻结规则判为 MIXED。", '',
        'Development 判定已先写入并封存，之后才计算已 consumed `[20,60)` 的描述性复用：', '',
        '| Comparator / estimate | Development mean E | Consumed descriptive mean E | 方向一致 |', '|---|---:|---:|---|']
    for r in read('development_vs_consumed_descriptive.csv'):
        text.append(f"| {r['comparator']} / {r['estimate']} | {number(r['development_mean'])} | {number(r['consumed_descriptive_mean'])} | {'是' if r['mean_direction_consistent'] == 'True' else '否'} |")
    text += ['', '该区间沿用 development 的 LSC/M 阈值，未重新分位；不是新的 independent confirmation，不参与或改变判定。', '',
        '## 8. 是否有足够证据进入 minimal BC pre-pooling nonlinearity 实验？', '',
        '尚未达到本轮预注册 GO 条件。现有结果是 comparator-dependent 的弱 phenotype，不能据此认定 BC rectification 缺失或已证明具体机制。交回用户/Chat 决定是否进入 Phase 2；本轮停止，未实现 rectification，训练、参数拟合、模型/生产代码修改、新 seed 均为0。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    schema = dict(format='NumPy compressed NPZ; allow_pickle=False', files=['per_bin_features_development.npz', 'per_bin_features_consumed.npz'],
        key_format='cell_id with # replaced by _ + __ + field', rows='scored bins in original sequence/time order',
        fields='LSC,local_mean,local_RMS,LSC_quantile,mean_stratum,live_frame,decoded_frame,time_seconds,recording,local_trial_one_based,global_trial_zero_based,sequence_index,time_bin,target,aligned_loss,LN_loss,CNN_loss',
        non_row_metadata_fields=['lsc_edges', 'mean_edges', 'weights'], target='binary spike event', loss_unit='nats per scored bin', feature_unit='dimensionless Weber drive',
        source_frame_origin_provenance='UNKNOWN; frozen decoded origin751 preserved', bootstrap=json.loads((OUT / 'bootstrap_identity.json').read_text()))
    (OUT / 'per_bin_features_schema.json').write_text(json.dumps(schema, indent=2), encoding='utf-8')
    initial = json.loads((OUT / 'evidence_manifest.initial.json').read_text())
    changed = [p for p, digest in initial['input_sha256'].items() if sha(ROOT / p) != digest]
    assert not changed
    status = subprocess.check_output(['git', 'status', '--short', '--untracked-files=all'], cwd=ROOT, text=True)
    manifest = dict(status='COMPLETE', created_utc=datetime.now(timezone.utc).isoformat(), branch=initial['branch'], HEAD=initial['HEAD'],
        git_status_before=initial['git_status_before'], git_status_after=status, input_sha256=initial['input_sha256'], input_count=len(initial['input_sha256']), input_hash_changes=changed,
        checkpoint_identity=initial['checkpoint_identity'], preflight_pairs=66, all_exact_replay=True, protocol_sha256=frozen['sha256'], protocol_unchanged=True,
        development_verdict_sha256=verdict_hash, development_verdict=verdict['verdict'], decision_data='development [16,20) only', secondary_data='consumed [20,60), descriptive only',
        training=0, parameter_fitting=0, new_seed=0, production_modifications=0, checkpoint_modifications=0, phase2_implementation=0,
        analysis_statistics_precision='float64 softplus(logits)-y*logits; frozen logits float32',
        max_loss_reduction_difference_from_native_float32=max(abs(float(r['delta'])) for r in read('precision_comparison.csv')),
        output_sha256={p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'evidence_manifest.json' and '__pycache__' not in p.parts})
    (OUT / 'evidence_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('REPORT AND MANIFEST', len(manifest['output_sha256']), 'outputs;', len(initial['input_sha256']), 'unchanged inputs')


if __name__ == '__main__':
    main()
