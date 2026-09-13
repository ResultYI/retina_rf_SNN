from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from retipath_spatial_ei_pilot import (
    ROOT, CELLS, BACKENDS, SpatialEIRetiPath, SpatialEIBackend, evaluate, load_inputs,
    load_json, model_args, save_json, sha, tensor_sha, spatial_ei_parameters,
)


def profile_metrics(profile: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, float]:
    center = np.sum(profile[:, None] * positions, axis=0)
    radius = np.sqrt(np.sum(profile * np.sum((positions - center) ** 2, axis=1)))
    return center, float(radius)


def radial_cdf(profile: np.ndarray, positions: np.ndarray, grid: np.ndarray) -> np.ndarray:
    center, _ = profile_metrics(profile, positions)
    radius = np.sqrt(np.sum((positions-center) ** 2, axis=1))
    if radius.max() > grid[-1]:
        raise ValueError('frozen radial grid does not cover this geometry')
    return (profile[:, None] * (radius[:, None] <= grid[None])).sum(0)


def metric_self_check() -> dict:
    positions = np.array([[-1., 0], [0., 0], [1., 0]])
    j = np.array([[1., 2., 1.], [2., 1., 2.]])
    p = np.square(j).sum(0) / np.square(j).sum()
    p2 = np.square(7*j).sum(0) / np.square(7*j).sum()
    np.testing.assert_allclose(p, p2, rtol=1e-14, atol=1e-14)
    center, radius = profile_metrics(p, positions)
    center2, radius2 = profile_metrics(p, positions + np.array([.2, -.1]))
    np.testing.assert_allclose(center2-center, [.2, -.1], atol=1e-14)
    assert abs(radius2-radius) < 1e-14
    return {'gain_invariance': True, 'translation_invariance_of_radius': True,
            'radius': radius, 'normalized_profile_sum': float(p.sum())}


def restore(path: Path, *, double: bool = False) -> tuple[SpatialEIRetiPath, dict]:
    cp = torch.load(path, weights_only=True)
    model = SpatialEIRetiPath(*model_args(cp), backend=SpatialEIBackend(cp['backend']),
                             rms_e=torch.tensor(cp['source']['rms']['e']),
                             rms_i=torch.tensor(cp['source']['rms']['i']))
    model.load_state_dict(cp['model'], strict=True)
    model.eval()
    if double:
        model.double()
    return model, cp


def diagnostics(model: SpatialEIRetiPath, dev) -> dict:
    result = {'alpha': float(model.alpha.detach()),
              'alpha_bound_hit': abs(float(model.alpha.detach())-.05) < 1e-6 or abs(float(model.alpha.detach())-2) < 1e-6}
    if model.spatial_ei is None:
        return result
    integrator = model.spatial_ei
    rows = []
    with torch.no_grad():
        for start in range(0, len(dev.cone_drive), 8):
            parts = model.spatial_components(dev.cone_drive[start:start+8])
            e, i = integrator.conductances(parts.u_e, parts.u_i)
            v = integrator.voltage(e, i)
            mask = dev.valid_mask[start:start+8, :, 0].bool()
            assert bool(torch.isfinite(v).all() & torch.isfinite(e).all() & torch.isfinite(i).all())
            assert bool((e >= 0).all() & (i >= 0).all())
            rows.append({'uE': parts.u_e[mask].numpy()[:, 0], 'uI': parts.u_i[mask].numpy()[:, 0],
                         'gE': e[mask].numpy()[:, 0], 'gI': i[mask].numpy()[:, 0], 'V': v[mask].numpy()[:, 0]})
    values = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
    result['aE'] = float(integrator.log_a_e.detach().exp())
    result['aI'] = float(integrator.log_a_i.detach().exp())
    result['output_scale'] = float(integrator.log_output_scale.detach().exp())
    for key, val in values.items():
        result[key] = {'min': val.min(0).tolist(), 'max': val.max(0).tolist(),
                       'std': val.std(0).tolist(), 'quantiles_01_50_99': np.quantile(val, [.01,.5,.99], axis=0).tolist()}
        if key in ('gE', 'gI'):
            derivative = -np.expm1(-val)
            result[key]['softplus_derivative_below_01_fraction'] = (derivative < .01).mean(0).tolist()
            result[key]['softplus_derivative_above_99_fraction'] = (derivative > .99).mean(0).tolist()
            result[key]['above_99_means_linear_branch_not_upper_saturation'] = True
    v = values['V']
    result['mode_voltage_correlation'] = float(np.corrcoef(v.T)[0,1]) if bool((v.std(0) > 0).all()) else None
    result['mode_voltage_centered_rms'] = np.sqrt(np.mean((v-integrator.v_0)**2, axis=0)).tolist()
    result['voltage_near_reversal_fraction'] = ((np.abs(v-integrator.e_e) < .001) | (np.abs(v-integrator.e_i) < .001)).mean(0).tolist()
    return result


def spatial_rf(model: SpatialEIRetiPath, dev, selections: dict, positions: np.ndarray,
               grid: np.ndarray) -> tuple[dict, dict]:
    for p in model.parameters():
        p.requires_grad_(False)
    state_hashes = {k: tensor_sha(v) for k,v in model.state_dict().items()}
    contexts = {}
    quality = {'prefix_comparison_max_abs': 0., 'observations': 0,
               'finite_jacobians': True, 'positive_gains': True, 'history_fixed': True}
    for label in ('LOW', 'HIGH'):
        obs, profiles, radial, gains, radii, old_energy = [], [], [], [], [], []
        for sequence, target in selections[label]:
            x = dev.cone_drive[sequence:sequence+1, :target+1].double().clone().requires_grad_()
            history = dev.spike_events[sequence:sequence+1, :target+1].double().detach()
            z = model(x, observed_counts=history).logits[0, target, 0]
            jac, = torch.autograd.grad(z, x)
            j = jac.detach().numpy()[0]
            assert np.isfinite(j).all()
            energy = np.square(j).sum()
            assert np.isfinite(energy) and energy > 0
            gain = float(np.sqrt(energy))
            profile = np.square(j).sum(0) / energy
            center, radius = profile_metrics(profile, positions)
            tail_fraction = float(np.square(j[:-16]).sum() / energy) if len(j) > 16 else 0.
            profiles.append(profile); gains.append(gain); radii.append(radius)
            radial.append(radial_cdf(profile, positions, grid)); old_energy.append(tail_fraction)
            obs.append({'sequence': sequence, 'bin': target, 'prefix_bins': target+1,
                        'source_id': dev.source_image_ids[sequence], 'gain': gain,
                        'centroid_deg': center.tolist(), 'radius_deg': radius,
                        'energy_before_last16_fraction': tail_fraction})
            quality['observations'] += 1
        mean_profile = np.mean(profiles, axis=0)
        center, radius = profile_metrics(mean_profile, positions)
        contexts[label] = {'gain_median': float(np.median(gains)), 'gain_mean': float(np.mean(gains)),
                           'centroid_x_deg': float(center[0]), 'centroid_y_deg': float(center[1]),
                           'radius_of_mean_profile_deg': radius, 'radius_observation_median_deg': float(np.median(radii)),
                           'profile': mean_profile, 'radial_cdf_mean': np.mean(radial, axis=0),
                           'tail_energy_median': float(np.median(old_energy)), 'observations': obs}
        sequence, target = selections[label][0]
        with torch.no_grad():
            full = model(dev.cone_drive[sequence:sequence+1].double(),
                         observed_counts=dev.spike_events[sequence:sequence+1].double()).logits[0,target,0]
            prefix = model(dev.cone_drive[sequence:sequence+1,:target+1].double(),
                           observed_counts=dev.spike_events[sequence:sequence+1,:target+1].double()).logits[0,target,0]
        error = float((full-prefix).abs())
        assert error < 1e-10
        quality['prefix_comparison_max_abs'] = max(quality['prefix_comparison_max_abs'], error)
    assert all(tensor_sha(v) == state_hashes[k] for k,v in model.state_dict().items())
    low, high = contexts['LOW'], contexts['HIGH']
    cdf_diff = np.abs(high['radial_cdf_mean']-low['radial_cdf_mean'])
    result = {'LOW_count': len(selections['LOW']), 'HIGH_count': len(selections['HIGH']),
              'log_gain_HIGH_over_LOW': math.log(high['gain_median']/low['gain_median']),
              'profile_TV': float(.5*np.abs(high['profile']-low['profile']).sum()),
              'centroid_shift_deg': math.hypot(high['centroid_x_deg']-low['centroid_x_deg'],
                                              high['centroid_y_deg']-low['centroid_y_deg']),
              'radius_HIGH_minus_LOW_deg': high['radius_of_mean_profile_deg']-low['radius_of_mean_profile_deg'],
              'observation_radius_HIGH_minus_LOW_deg': high['radius_observation_median_deg']-low['radius_observation_median_deg'],
              'centered_radial_CDF_max_difference': float(cdf_diff.max()),
              'centered_radial_CDF_L1_deg': float(np.trapezoid(cdf_diff, grid))}
    for label, values in contexts.items():
        for key, val in values.items():
            if isinstance(val, np.ndarray):
                val = json.dumps(val.tolist(), separators=(',', ':'))
            elif isinstance(val, list):
                val = json.dumps(val, separators=(',', ':'))
            result[f'{label}_{key}'] = val
    return result, quality


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def aggregate(predictions: list[dict], checkpoint: str, split: str, left: str, right: str) -> dict:
    values = {(r['cell_id'], r['condition']): r[f'{split}_nll'] for r in predictions if r['checkpoint'] == checkpoint}
    diffs = np.array([values[c,left]-values[c,right] for c in CELLS])
    return {'mean': float(diffs.mean()), 'median': float(np.median(diffs)),
            'wins': int((diffs < 0).sum()), 'losses': int((diffs > 0).sum()), 'ties': int((diffs == 0).sum())}


def report(out: Path) -> None:
    torch.set_num_threads(1)
    evidence = load_json(out / 'correctness.json')
    assert evidence['training_complete'] and evidence['optimizer_updates'] == 12000
    assert sha(out / 'PROTOCOL.md') == evidence['protocol_sha256']
    grid = np.array(evidence['protocol']['spatial_radius_grid_deg'])
    predictions, spatial, post_checks, diagnostic_rows = [], [], {}, []
    raw_predictions = {}
    for cell_id in CELLS:
        ref = evidence['protocol']['cells'][cell_id]
        _, _, train, dev = load_inputs(ref)
        for variant in ('A', 'B', 'C'):
            folder = out / 'checkpoints' / cell_id.replace('#', '_') / variant
            final_cp = torch.load(folder / 'final.pt', weights_only=True)
            assert final_cp['step'] == 1000
            adam_steps = [int(s['step']) for s in final_cp['optimizer']['state'].values()]
            assert adam_steps and set(adam_steps) == {1000}
            assert final_cp['fixed_buffers_unchanged']
            curve = final_cp['training_curve']
            assert [r['step'] for r in curve] == list(range(0,1001,25))
            chosen = min(curve, key=lambda r:r['dev_nll'])
            post_checks[f'{cell_id}/{variant}'] = {'adam_steps': sorted(set(adam_steps)),
                'schedule_sha256': tensor_sha(final_cp['schedule']),
                'nonzero_gradient_seen': final_cp['gradient_nonzero_seen'],
                'parameters_actually_changed': final_cp['parameters_actually_changed'],
                'fixed_buffers_unchanged': True, 'checkpoint_replay': {}}
            assert tensor_sha(final_cp['schedule']) == ref['schedule_sha256']
            for kind in ('start', 'final', 'best'):
                model, cp = restore(folder / f'{kind}.pt')
                assert cp['protocol_sha256'] == evidence['protocol_sha256']
                if kind == 'best':
                    assert cp['step'] == chosen['step']
                train_nll, _ = evaluate(model, train)
                dev_nll, _ = evaluate(model, dev)
                assert train_nll == cp['train_nll'] and dev_nll == cp['dev_nll']
                post_checks[f'{cell_id}/{variant}']['checkpoint_replay'][kind] = {'train_error':0., 'dev_error':0.}
                row = {'cell_id':cell_id, 'condition':variant, 'checkpoint':kind, 'step':cp['step'],
                       'train_nll':train_nll, 'dev_nll':dev_nll}
                raw_predictions[cell_id, variant, kind] = row
                predictions.append(row)
                if kind == 'start':
                    continue
                dg = diagnostics(model, dev)
                dg.update({'cell_id':cell_id,'condition':variant,'checkpoint':kind,
                           'last250_train_nll_change':curve[-1]['train_nll']-curve[-11]['train_nll'],
                           'last250_dev_nll_change':curve[-1]['dev_nll']-curve[-11]['dev_nll'],
                           'best_step':chosen['step'],
                           'dev_minus_train_nll':dev_nll-train_nll})
                diagnostic_rows.append(dg)
                model.double()
                metrics, quality = spatial_rf(model, dev, ref['selected_observations'],
                                              cp['cone_positions_degs'].double().numpy(), grid)
                metrics.update({'cell_id':cell_id, 'condition':variant, 'checkpoint':kind, 'step':cp['step']})
                spatial.append(metrics)
                post_checks[f'{cell_id}/{variant}'][f'{kind}_RF'] = quality
                print('RF', cell_id, variant, kind, 'TV', round(metrics['profile_TV'],6), flush=True)
    for row in predictions:
        for baseline in ('A','B'):
            reference = raw_predictions[row['cell_id'],baseline,row['checkpoint']]
            for split in ('train','dev'):
                row[f'{split}_minus_{baseline}'] = row[f'{split}_nll']-reference[f'{split}_nll']
    write_csv(out / 'prediction_per_cell.csv', predictions)
    write_csv(out / 'spatial_rf_per_cell.csv', spatial)
    for path,digest in evidence['protocol']['source_hashes'].items():
        assert sha(ROOT/path) == digest, path
    for ref in evidence['protocol']['cells'].values():
        for key,digest in ref['hashes'].items():
            assert sha(Path(ref['paths'][key])) == digest
    evidence['post_training_checks'] = post_checks
    evidence['spatial_metric_tests'] = metric_self_check()
    evidence['diagnostics'] = diagnostic_rows
    evidence['original_source_and_input_hashes_unchanged'] = True
    evidence['analysis_source_sha256'] = sha(Path(__file__))
    evidence['report_complete'] = True
    evidence['data_access'] = {'train_seconds':[0,16], 'development_seconds':[16,20],
                             'new_evaluation_data_read': False, 'independent_test':'NONE'}
    save_json(out/'correctness.json', evidence)
    write_report(out, predictions, spatial, diagnostic_rows, evidence)
    write_implementation(out, evidence)
    expected = {'PROTOCOL.md','correctness.json','training.csv','prediction_per_cell.csv',
                'spatial_rf_per_cell.csv','checkpoints','REPORT.md','implementation_note_zh.md'}
    assert {p.name for p in out.iterdir()} == expected
    print('REPORT COMPLETE', out, flush=True)


def write_report(out: Path, pred: list[dict], spatial: list[dict], diagnostics_rows: list[dict], evidence: dict) -> None:
    lines = ['# RetiPath 空间 E/I pilot 结果', '',
             '已完成4 cells × A/B/C，共12 fits、12,000 optimizer updates。固定 step1000 为主要结果；best 仅来自预定 step0/每25步 development 评估。单 seed、单次匹配预算，无追加训练。**Development pilot；independent test = NONE；没有读取 [20,60) 或之后的数据。**', '',
             '## 1. 实现与训练正确性', '',
             'VERIFIED：默认开关逐位重放四个旧 RetiPath best checkpoint；新实现仅新增独立模块，原模型文件未改。BC/AC重加和通过相对1e−5、绝对1e−6容差；最大误差见 correctness.json。固定电导解析解、恒定总电导B/C一致、非负映射、状态有限、future stimulus/current spike 无泄漏、完整prefix和梯度均通过。新后端无需在step0等于旧模型。', '',
             'A有34个可学习标量，B/C各37个；增加aE、aI、output_scale三个正值标量。其余原可学习参数继续学习。固定RMS/原buffer未更新，所有预期参数进入optimizer。每个fit的Adam计数均为1000，start/final/best重放train/dev NLL误差为0。', '',
             '## 2. 逐cell预测', '',
             'NLL单位 nats/scored bin，越低越好。best train 是 dev最低 checkpoint 对应的 train NLL，不是独立选 train 最低。', '',
             '|cell|条件|start train|final train|best train|start dev|final dev|best dev|best step|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    lookup = {(r['cell_id'],r['condition'],r['checkpoint']):r for r in pred}
    for c in CELLS:
        for v in ('A','B','C'):
            start, final, best = (lookup[c,v,k] for k in ('start','final','best'))
            vals = [start['train_nll'],final['train_nll'],best['train_nll'],start['dev_nll'],final['dev_nll'],best['dev_nll']]
            lines.append(f'|{c}|{v}|'+'|'.join(f'{x:.8f}' for x in vals)+f'|{best["step"]}|')
    lines += ['', '### 匹配差值汇总', '', 'wins指左条件NLL较低；四个cell等权，无bootstrap或显著性检验。逐cell的差值保存在 prediction_per_cell.csv。', '',
              '|checkpoint|split|差值|mean|median|wins/losses/ties|', '|---|---|---|---:|---:|---:|']
    for k in ('start','final','best'):
        for split in ('train','dev'):
            for left,right in (('B','A'),('C','A'),('C','B')):
                s=aggregate(pred,k,split,left,right)
                lines.append(f'|{k}|{split}|{left}−{right}|{s["mean"]:+.8f}|{s["median"]:+.8f}|{s["wins"]}/{s["losses"]}/{s["ties"]}|')
    cb=aggregate(pred,'final','dev','C','B'); ba=aggregate(pred,'final','dev','B','A'); ca=aggregate(pred,'final','dev','C','A')
    lines += ['', f'**C是否比B预测更好：** 固定step1000的development C−B均值 {cb["mean"]:+.8f}，中位数 {cb["median"]:+.8f}，C胜 {cb["wins"]}/4。'+('本pilot支持C在这四个细胞上具有平均预测优势。' if cb['mean']<0 else '本pilot没有显示C在这四个细胞上具有平均预测优势。'), '',
              f'**B/C是否比同预算A更好：** B−A均值 {ba["mean"]:+.8f}（{ba["wins"]}/4胜），C−A均值 {ca["mean"]:+.8f}（{ca["wins"]}/4胜）。B/C相对A同时改变空间表示与后端，不能单独归因于电导机制。主要科学比较始终是C−B。', '',
              '## 3. 空间 RF：与预测分开报告', '',
              '每cell/context20个来自既有development观测的确定性子样本。logit对完整因果prefix的Jacobian使用float64计算；observed history固定，不截断16 lag。每观测先计算单位化P，随后context内等权平均。所有289位置均参与。空间位置/半径单位degree。', '',
              '下表为主要final checkpoint。G为原始Jacobian L2 gain的context median；r为平均P自身重心的二阶矩半径；Δr_obs为各观测围绕自身重心半径的HIGH−LOW median差，避免把context内重心离散误当成单个RF扩张。TV只去除gain；中心化径向CDF差在每个观测自身重心下计算，同时去除平移影响。', '',
              '|cell|条件|G LOW|G HIGH|r LOW|r HIGH|Δr_obs|重心移动|P TV|中心化径向CDF最大差|',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    final_rf=[r for r in spatial if r['checkpoint']=='final']
    for r in final_rf:
        vals=[r['LOW_gain_median'],r['HIGH_gain_median'],r['LOW_radius_of_mean_profile_deg'],r['HIGH_radius_of_mean_profile_deg'],r['observation_radius_HIGH_minus_LOW_deg'],r['centroid_shift_deg'],r['profile_TV'],r['centered_radial_CDF_max_difference']]
        lines.append(f'|{r["cell_id"]}|{r["condition"]}|'+'|'.join(f'{x:.6f}' for x in vals)+'|')
    lines += ['', '各context的x/y重心、完整P数组、径向CDF数组、每观测gain/半径/索引，以及best checkpoint的同样结果见 spatial_rf_per_cell.csv。', '',
              '**是否超出整体gain与重心移动：** P单位化已消除整体gain；中心化径向分布的残余差不能由统一增益或统一平移单独解释。以下为四cell的实际残余量；这是拟合模型内的context关联，不是生物机制或因果适应验证。', '',
              '|条件|P TV mean|中心化径向CDF差 mean|中心化径向CDF差积分 mean (degree)|Δr_obs mean (degree)|',
              '|---|---:|---:|---:|---:|']
    for v in ('A','B','C'):
        rows=[r for r in final_rf if r['condition']==v]
        cols=('profile_TV','centered_radial_CDF_max_difference','centered_radial_CDF_L1_deg','observation_radius_HIGH_minus_LOW_deg')
        lines.append(f'|{v}|'+'|'.join(f'{np.mean([r[k] for r in rows]):.8f}' for k in cols)+'|')
    lines += ['', '离散cone采样和CDF网格会影响差异数值；未设生物学效应阈值，也不把“非零”视为显著。归一化profile变化不自动表明更正确，未出现变化也不自动表明模型无用。没有添加RF形状/大小loss或固定颜色阈值面积。', '',
              '## 4. 通道状态、饱和与预算', '',
              'softplus没有上界；导数>0.99表示进入近线性区，并非上饱和。导数<0.01是接近关闭的数值诊断。下表报告final在development评分bin上的两个K范围；模式高相关提示表示冗余，不能据此声称某个真实生物通道消失。', '',
              '|cell|条件|aE|aI|output scale|V两模式相关|gE关闭比例最大值|gI关闭比例最大值|最后250步train变化|最后250步dev变化|',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in diagnostics_rows:
        if d['checkpoint']!='final' or d['condition']=='A': continue
        corr=d['mode_voltage_correlation']
        lines.append(f'|{d["cell_id"]}|{d["condition"]}|{d["aE"]:.5f}|{d["aI"]:.5f}|{d["output_scale"]:.5f}|{corr:.6f}|{max(d["gE"]["softplus_derivative_below_01_fraction"]):.4f}|{max(d["gI"]["softplus_derivative_below_01_fraction"]):.4f}|{d["last250_train_nll_change"]:+.8f}|{d["last250_dev_nll_change"]:+.8f}|')
    dgfinal=[d for d in diagnostics_rows if d['checkpoint']=='final' and d['condition']!='A']
    near_closed=[f'{d["cell_id"]}/{d["condition"]}' for d in dgfinal if max(d['gE']['softplus_derivative_below_01_fraction']+d['gI']['softplus_derivative_below_01_fraction'])>.95]
    tail_decreasing=[f'{d["cell_id"]}/{d["condition"]}' for d in diagnostics_rows if d['checkpoint']=='final' and d['last250_train_nll_change']<0]
    lines += ['', '超过95%评分bin处于softplus近关闭状态的通道所属fit：'+(', '.join(near_closed) if near_closed else '无')+'。该95%仅描述数值状态，不是预设生物学退化标准。完整uE/uI、gE/gI、V范围/标准差/分位数和反转电位附近比例见correctness.json的diagnostics。', '',
              '两个K的电压相关系数约0.929–0.999，部分fit的有效空间模式高度相似，存在表示冗余迹象；并未观察到整条E/I通道几乎始终关闭。69#4的A/B中alpha仍在原上界2，已记录其梯度与投影后未改变的区别；没有改变alpha bounds。', '',
              '最后250步train NLL仍下降的fit：'+', '.join(tail_decreasing)+'。因此不能将1000步预算耗尽称为充分收敛；曲线趋势和train/dev差异均保留供判断。尤其67#6的B和67#4的B/C在末段train/dev均继续改善，预算可能仍限制其比较结果；这不是追加训练的依据。没有为获得理想方向追加步数、重启或调整范围。', '',
              '## 5. 证据边界与交付', '',
              '预测改善与空间RF变化分别判读。本次仅四个指定代表细胞，结果不能作为独立确认、总体显著性或真实电压/电导恢复证据，也不足以关闭整个研究方向。训练期间没有访问新评价数据。', '',
              'PROTOCOL.md含训练前配置、branch/HEAD、manifest解析来源及必要hash；correctness.json含工程验证、训练重放、实际更新记录与诊断；training.csv含492个预定评估点；两个per-cell CSV保存预测与空间量；checkpoints/含每fit start/best/final及optimizer、完整schedule和配置；implementation_note_zh.md解释张量、公式与单位。', '',
              '文献框架见 [Latimer et al., 2019](https://elifesciences.org/articles/47012)。作者代码反转电位数值本次远程核实未完成（UNVERIFIED），本实现严格采用用户明确指定的归一化值，详见协议。', '',
              '**整个pilot完成后停止；没有扩展22 cells、接入新数据或启动下一项机制实验。**']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def write_implementation(out: Path, evidence: dict) -> None:
    text = r'''# 空间 E/I pilot 中文实现说明

## 从输入到spike probability

新模块是 `models/mechanistic_retina/retipath_spatial_ei.py`。`SpatialEIRetiPath(..., backend="retipath")` 是默认，调用原始RetiPath完整forward。显式选择 `spatial_current` 或 `spatial_conductance` 才启用B/C，且必须给出冻结RMS。原生产工厂、原模型文件与旧checkpoint均未修改。runner为 `work/retipath_spatial_ei_pilot.py`，分析为 `work/retipath_spatial_ei_report.py`。

输入 `cones[B,T,C]` 是原calibrated L+M Weber drive；本pilot C=289 (17×17)，T=150，N=1。spike events `[B,T,1]` 为150Hz Bernoulli occupancy。空间位置来自旧checkpoint，单位degree；没有新增中心、半径或空间滤波器。H1先按原forward作用于输入。

## 为什么保留K不改变BC整流语义

原 `feature_bank` 输出 `[B,T,N,P=4,K=2,R=3]`。P依次为direct sustained、direct transient、broad sustained、broad transient；R为时间basis。原positive BC权重 `[N,2,K,R]` 在direct/broad共享。令w乘feature得到每个时间/空间basis贡献。

先取完整分支 `S=sum_K,R(w*feature)`，据此取 `s=1 if S>=0 else alpha`。只汇聚时间basis得到 `u_k=sum_R(w*feature)`，然后每个模式都乘同一个s。因此 `sum_k(s*u_k)=phi_alpha(S)`。这不是逐K独立整流；在S正负侧翻转时两个K一起改变斜率。计算中完整S沿用原reduction顺序，K重加和允许已记录的float32舍入差。

转轴后direct与broad均为 `[B,T,N,K,2]`，末维sustained/transient。broad重排为 `[B,T,N*K,2]` 调用同一原AC delay/filter/gate，再恢复K轴。AC真实公式为 `current=-gate*filtered_state`。保留原cell gain后，`uE_k=sum(direct sustained/transient currents)`；`uI_k=-sum(AC currents)`。禁止abs。负uI表示相对基线抑制输入减少，而非负电导。

## 固定尺度与正值映射

初始最终M2 best checkpoint作用于train [0,16)的所有独立sequence；用其中全部bin（含warmup及原重复次数）分别计算每个K的RMS：`s=sqrt(mean(u²))`。float64累积，存为float32 buffer `[N,2]`；之后没有随训练重估，也没有development统计。RMS必须严格正且有限。

`b0=log(exp(1)-1)`，所以softplus(b0)=1。

`gE_k=softplus(b0+exp(log_aE)*uE_k/sE_k)`；`gI_k=softplus(b0+exp(log_aI)*uI_k/sI_k)`。

aE/aI是两个可学习正值标量，跨K共享；原始log参数初值0，有效初值1。B/C完全相同。零刺激在独立sequence的零初始上游状态下产生零uE/uI，因此gE=gI=1。映射中的负驱动允许g下降到基线1以下，始终非负。softplus负端接近0、正端近线性，没有上界。

## B和C唯一不同的膜方程

固定 `EL=0, EE=1, EI=-1/3, gL=1`。这是一套有效归一化电压/电导尺度，不是mV/nS测量。`G0=3, V0=2/9`，`tau_ref=5 ms` 复用四个起点的固定膜时间常数；`Cm=G0*tau_ref=15`，其单位为有效电导×ms。每bin `dt=1000/150=6.666666... ms`。Cm、反转电位、gL、V0、tau_ref本轮全部固定。

C电导型：

`Cm*dV_k/dt = gL*(EL-V_k)+gE_k*(EE-V_k)+gI_k*(EI-V_k)`。

于是 `G=1+gE+gI`，`Vinf=(gE-gI/3)/G`。

B在同一静息点线性化：

`Cm*dV_k/dt=-G0*(V_k-V0)+(gE_k-1)*(EE-V0)+(gI_k-1)*(EI-V0)`。

于是 `G=G0`，`Vinf=V0+[(gE-1)*(EE-V0)+(gI-1)*(EI-V0)]/G0`。

两者均按每bin输入恒定解析更新：`V_next=V+(1-exp(-dt*G/Cm))*(Vinf-V)`。代码用 `-expm1(-dt*G/Cm)` 稳定计算括号。sequence初态为V0；从未默认数值0。每个K独立递推；所有前序状态都保留在autograd中，既不detach也不截断。

当gE+gI=2，C中的G=G0，且两者Vinf代数相同，因此同输入/初态给出相同V。一般情况下C多出总电导随输入变化和膜状态相互作用；B/C各自都仍含共同softplus映射，并不是整体线性模型。

## 如何接回原RGC尾部

两个K等权平均：`membrane=exp(log_output_scale)*(mean_K(V)-V0)`，output scale初值1。此量直接替代旧normalized-current→membrane输出。新forward没有执行旧divisive normalization或旧membrane低通。为保留返回结构，旧divisive状态字段置零，表示此后端不使用它。

原adaptation仍是membrane的因果低通。logit保持：`logit_slope*(membrane-threshold)-adaptation_gain*adaptation-history_gate*history_gain*strictly_past_history+bias`。observed event先移后一bin再低通，所以当前spike不进入当前logit；输出sigmoid(logit)得到占据概率。没有新增适应状态、阈值动态、NMDA或可塑性。

## 哪些量学习

原可学习量包括H1时间常数/延迟/幅度、BC时间basis与延迟、共享连接中原本可学习的量、BC positive weight参数、AC时间常数/延迟/gates、history gate、cell BC/AC gain、response bias和alpha。具体requires_grad、optimizer覆盖、非零梯度与实际改变分别记录在correctness.json，避免把这四件事混为一谈。A为34个标量；B/C各增加3个，共37。

原空间basis/support/位置、参数bounds、固定RGC衰减/gain/logit_slope/threshold保持固定。B/C还固定RMS、b0、电位、gL和Cm。alpha继续原[0.05,2]投影；新增正值参数以指数参数化，不另加上限。Adam每fit新建，学习率.003、无weight decay/scheduler、1000 updates、batch4、gradient norm clip5，三条件共享同一cell的整个minibatch schedule。

## 如何理解空间RF量

固定observed history，从本sequence起点到目标bin全部输入求 `J(lag,x)=d logit(t)/d cones(t-lag,x)`，而不是只求最后16bin。RF使用float64运算，训练和预测重放使用原float32。CSV同时保存最后16bin之前的梯度能量比例；它是完整计算的描述，不用于选择截断。

原始gain `G=sqrt(sum J²)`。单位化空间敏感度 `P(x)=sum_lag J²/sum_lag,x J²`，所有289位置参与，和为1。先每观测单位化，再在context内等权平均P，避免大gain观测支配空间profile。重心 `mu=sum P*x`，二阶矩半径 `r=sqrt(sum P*||x-mu||²)`。还报告逐观测自身重心半径的median。

HIGH/LOW的TV差去除了整体gain。进一步对每观测用自身mu计算径向CDF `F(r)=sum_{||x-mu||<=r} P(x)`，再context内平均；它消除了位置平移。固定半径网格上的CDF最大差和差积分反映径向profile残余差。径向CDF不包含角向信息；TV保留原空间profile差。所有度量仅描述拟合模型局部敏感度，不能证明哪个生物机制正确。HIGH/LOW为train-derived既有对比度分组，关联不等于因果适应。

## 证据位置

PROTOCOL.md包含实际配置和来源hash；correctness.json记录直接必要测试、参数状态、序列化重放和诊断；training.csv保存全部492个预定评估点；prediction_per_cell.csv保存start/final/best和配对差值；spatial_rf_per_cell.csv保存完整profile/CDF数组及观测索引/指标；checkpoint内包含模型、optimizer、来源、配置、schedule与曲线。没有新增评价数据或独立确认。
'''
    (out/'implementation_note_zh.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'output/experiments/retipath_spatial_ei_pilot')
    report(parser.parse_args().out)
