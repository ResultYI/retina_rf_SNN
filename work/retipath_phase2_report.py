from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from retipath_phase2_common import SEEDS,CONDITIONS,sha,load_json,save_json


def make_figures(out: Path,config: dict,prediction: list[dict],rf: list[dict],mode: list[dict],stats: dict,rf_stats: dict) -> None:
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':180})
    cells=list(config['cells']);figures=out/'figures'
    colors={'A':'#60758c','B':'#d59135','C':'#218378'}
    pred={(r['cell_id'],r['seed'],r['condition']):r for r in prediction}
    fig,axes=plt.subplots(1,2,figsize=(13,9),gridspec_kw={'width_ratios':[1.35,1]})
    y=np.arange(len(cells))
    for index,(left,right) in enumerate((('C','B'),('C','A'))):
        values=np.array([[pred[c,s,left]['dev_nll']-pred[c,s,right]['dev_nll'] for s in SEEDS] for c in cells])
        offsets=y+(index-.5)*.2
        axes[0].hlines(offsets,values.min(1),values.max(1),color=colors['C' if index==0 else 'B'],alpha=.6)
        axes[0].scatter(values.mean(1),offsets,color=colors['C' if index==0 else 'B'],s=24,label=f'{left}-{right}')
    axes[0].set_yticks(y,cells);axes[0].invert_yaxis();axes[0].axvline(0,color='#777',lw=.8)
    axes[0].set_xlabel('Development NLL difference (nats/bin)')
    axes[0].set_title('Cell means; lines show range across 3 seeds')
    axes[0].legend()
    labels=[];values=[];low=[];high=[];cs=[]
    for comparison in ('C-B','C-A','B-A'):
        for s in (*SEEDS,'all'):
            item=stats[f'{comparison}/{s}'];labels.append(f'{comparison}  '+(f'seed {SEEDS.index(s)+1}' if s!='all' else 'seed mean'))
            values.append(item['mean']);low.append(item['mean']-item['ci_low']);high.append(item['ci_high']-item['mean'])
            cs.append(colors[comparison[0]])
    for i,(value,lo,hi,color) in enumerate(zip(values,low,high,cs,strict=True)):
        axes[1].errorbar(value,i,xerr=np.array([[lo],[hi]]),fmt='o',color=color,capsize=3)
    axes[1].set_yticks(range(len(labels)),labels);axes[1].invert_yaxis();axes[1].axvline(0,color='#777',lw=.8)
    axes[1].set_title('Paired 95% cell bootstrap intervals')
    axes[1].set_xlabel('Equal-cell mean NLL difference')
    fig.suptitle('Phase 2: train-only selection, fresh refit, consumed development comparison',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(figures/'prediction_population.png');plt.close(fig)
    valid=[r for r in rf if r['status']=='VERIFIED']
    lookup={(r['cell_id'],r['seed'],r['condition']):r for r in valid}
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for k,condition in enumerate(CONDITIONS):
        for i,s in enumerate(SEEDS):
            records=[r for r in valid if r['condition']==condition and r['seed']==s]
            axes[0].scatter(k+(i-1)*.12,np.mean([r['TV'] for r in records]),color=colors[condition],marker=('o','s','^')[i],s=50)
            axes[1].scatter(k+(i-1)*.12,np.mean([r['centered_radial_CDF_L1_deg'] for r in records]),color=colors[condition],marker=('o','s','^')[i],s=50)
        cosines=[v for row in rf_stats[condition]['repeatability'] for v in row['delta_P_pairwise_cosines'] if v is not None]
        axes[2].scatter(np.full(len(cosines),k),cosines,color=colors[condition],s=13,alpha=.35)
        axes[2].scatter(k,np.median(cosines),color='black',marker='_',s=250)
    for ax in axes: ax.set_xticks(range(3),list(CONDITIONS))
    axes[0].set_title('Normalized profile TV');axes[0].set_ylabel('Equal-cell mean, each marker a seed')
    axes[1].set_title('Centered radial CDF distance');axes[1].set_ylabel('CDF difference integral (degree)')
    axes[2].set_title('Within-cell delta P agreement across seeds');axes[2].set_ylabel('Pairwise cosine');axes[2].axhline(0,color='#888',lw=.8)
    coverage=', '.join(f'{c}={rf_stats[c]["valid_cells"]}' for c in CONDITIONS)
    fig.suptitle('Cells with existing HIGH/LOW observations: '+coverage,fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(figures/'rf_normalized_dynamics.png');plt.close(fig)
    mode_lookup={(r['cell_id'],r['seed'],r['condition']):r for r in mode if r['status']=='VERIFIED'}
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for ax,condition in zip(axes,CONDITIONS,strict=True):
        pairs=[]
        for cell in cells:
            if not all((cell,s,condition) in mode_lookup for s in SEEDS):continue
            dq=np.mean([abs(mode_lookup[cell,s,condition]['delta_q1']) for s in SEEDS])
            tv=np.mean([lookup[cell,s,condition]['TV'] for s in SEEDS]);pairs.append((dq,tv))
        a=np.array(pairs);ax.scatter(a[:,0],a[:,1],color=colors[condition],s=35,alpha=.8)
        ax.set_xlabel('Mean across seeds of |HIGH-LOW q1|');ax.set_ylabel('Mean normalized profile TV')
        corr=rf_stats[condition]['TV_vs_abs_delta_q1_Spearman']
        ax.set_title(f'{condition}: cell-level Spearman '+(f'{corr:.3f}' if corr is not None else 'undefined'))
    fig.suptitle('Descriptive cell-level association; RF coverage '+coverage,fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(figures/'mode_redistribution_association.png');plt.close(fig)
    arrays={'cells':np.array(cells),'seeds':np.array(SEEDS),'conditions':np.array(list(CONDITIONS))}
    for cell in cells:
        if not all((cell,s,c) in lookup for s in SEEDS for c in CONDITIONS):
            fig,ax=plt.subplots(figsize=(8,3));ax.axis('off')
            ax.text(.5,.6,f'{cell}: delta P UNVERIFIED',ha='center',fontsize=16)
            ax.text(.5,.35,'Missing existing HIGH/LOW observations; no replacement sampling',ha='center')
            fig.savefig(figures/f'delta_P_{cell.replace("#","_")}.png');plt.close(fig)
            continue
        cp=torch.load(out/'checkpoints'/cell.replace('#','_')/str(SEEDS[0])/'A/refit.pt',weights_only=True)
        positions=cp['cone_positions_degs'].double().numpy()
        delta=np.stack([[lookup[cell,s,c]['delta_P'] for s in SEEDS] for c in CONDITIONS])
        arrays[cell.replace('#','_')+'_delta_P']=delta
        arrays[cell.replace('#','_')+'_positions_deg']=positions
        average=delta.mean(1);limit=float(np.abs(average).max())
        if limit==0:limit=np.finfo(np.float64).eps
        ux,uy=np.unique(positions[:,0]),np.unique(positions[:,1])
        ix,iy=np.searchsorted(ux,positions[:,0]),np.searchsorted(uy,positions[:,1])
        fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharex=True,sharey=True)
        for i,condition in enumerate(CONDITIONS):
            grid=np.zeros((len(uy),len(ux)));grid[iy,ix]=average[i]
            art=axes[i].pcolormesh(ux,uy,grid,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='nearest',rasterized=True)
            axes[i].set_aspect('equal');axes[i].set_title(condition);axes[i].set_xlabel('x (degree)')
        axes[0].set_ylabel('y (degree)')
        fig.suptitle(f'{cell}: HIGH - LOW normalized spatial sensitivity, mean of 3 seeds')
        fig.subplots_adjust(left=.07,right=.86,bottom=.17,top=.8,wspace=.22)
        cax=fig.add_axes((.89,.2,.015,.55));fig.colorbar(art,cax=cax,label='Delta P (shared scale within cell)')
        fig.savefig(figures/f'delta_P_{cell.replace("#","_")}.png');plt.close(fig)
    np.savez_compressed(figures/'delta_P_maps_by_cell_seed_condition.npz',**arrays)


def number(value: float | None,places: int = 6) -> str:
    return 'UNVERIFIED' if value is None else f'{value:.{places}f}'


def write_report(out: Path,config: dict,prediction: list[dict],stats: dict,rf_stats: dict) -> None:
    cb,ca,ba=(stats[f'{comparison}/all'] for comparison in ('C-B','C-A','B-A'))
    stable_cb=cb['stable_by_frozen_rule'];stable_ca=ca['stable_by_frozen_rule']
    lines=['# RetiPath spatial E/I Phase 2 population validation','',
           '22 macaque cells × 3 deterministic seeds × A/B/C。仅在[0,16)内部选step，fresh refit完整[0,16)，再比较已消费[16,20)。本报告为development comparison；independent test = NONE。原pilot实现、alpha bounds、空间basis、固定量和front-end均未改变。','',
           '## 1. C−B是否在population和seeds上稳定？','',
           ('按预先冻结规则，本阶段C−B预测收益稳定。' if stable_cb else '按预先冻结规则，本阶段C−B预测收益尚不稳定。')+
           f'先在每cell内平均三个seed，再作equal-cell汇总：mean={cb["mean"]:+.8f}、median={cb["median"]:+.8f} nats/bin，wins={cb["wins"]}/22，paired cell bootstrap95% CI=[{cb["ci_low"]:+.8f}, {cb["ci_high"]:+.8f}]。负值有利于C。','',
           '冻结的稳定规则是整体CI上端<0且三个seed mean均<0。每seed的CI另列；方向一致不自动意味着每个seed都有排除0的CI。bootstrap按cell成对重采样10000次，不把66个cell-seed视为独立生物样本。','',
           '|比较|seed|equal-cell mean|median|wins/22|95% paired CI|',
           '|---|---|---:|---:|---:|---|']
    for comparison in ('C-B','C-A','B-A'):
        for seed in (*SEEDS,'all'):
            d=stats[f'{comparison}/{seed}']
            lines.append(f'|{comparison}|{seed}|{d["mean"]:+.8f}|{d["median"]:+.8f}|{d["wins"]}|[{d["ci_low"]:+.8f}, {d["ci_high"]:+.8f}]|')
    lines+=['','每cell/seed的A/B/C原始train/dev NLL、选定步数及差值见 prediction_population.csv；其cell_seed行保留所有预定seed，不做结果驱动的seed排除。','',
            f'![预测比较]({(out/"figures/prediction_population.png").as_posix()})','',
            '## 2. C是否真正优于当前A，还是主要只优于B？','',
            f'C−A的equal-cell mean={ca["mean"]:+.8f}、median={ca["median"]:+.8f}，wins={ca["wins"]}/22，95% CI=[{ca["ci_low"]:+.8f}, {ca["ci_high"]:+.8f}]。B−A mean={ba["mean"]:+.8f}，wins={ba["wins"]}/22。','']
    if stable_ca:
        lines.append('按同一预定规则，C在本阶段也稳定优于A，收益不只来自相对B。')
    elif stable_cb:
        lines.append('本阶段更明确的结果是C优于B；C相对A尚未达到相同稳定标准，因此不能把C−B的优势直接称为超过当前RetiPath。')
    else:
        lines.append('C相对A与B的结果应分别按上表判读，目前不足以宣布C全面优于现有方案。')
    lines+=['','这里A是当前RetiPath同架构在本阶段的fresh fit，不是直接复用pilot已由development选出的权重。三条件使用同一seed、同一阶段的minibatch schedule及相同上游初始化，B/C新增参数同初始化。选步后丢弃inner权重和optimizer，再从同seed初始化完整train refit。B/C−A包含空间表示和后端的共同改变；C−B才是matched conductance比较。','',
            '## 3. RF normalized shape dynamics是否稳定存在？','',
            '以下均使用完整因果prefix logit Jacobian、固定observed history和原HIGH/LOW观测。P先按每观测归一化，空间TV去除整体gain，逐观测自身重心下的径向CDF去除平移。指标仍描述拟合模型敏感度，不是生物学正确性或因果适应证据。','',
            '|条件|seed|TV mean|中心化径向CDF差积分 mean (degree)|Δradius median-of-observations mean (degree)|log gain HIGH/LOW mean|',
            '|---|---|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        for seed in SEEDS:
            d=rf_stats[condition]['per_seed_equal_cell_means'][str(seed)]
            lines.append(f'|{condition}|{seed}|{d["TV"]:.8f}|{d["centered_radial_CDF_L1_deg"]:.8f}|{d["delta_radius_observation_median_deg"]:+.8f}|{d["gain_log_HIGH_over_LOW"]:+.6f}|')
    lines+=['','|条件|有效RF cells|ΔP三组seed配对cosine均>0的cells|配对cosine median|',
            '|---|---:|---:|---:|']
    for condition in CONDITIONS:
        d=rf_stats[condition]
        lines.append(f'|{condition}|{d["valid_cells"]}|{d["delta_P_all_pairs_positive_cells"]}|{d["median_pairwise_delta_P_cosine"]:.6f}|')
    c_rf=rf_stats['C'];positive=c_rf['delta_P_all_pairs_positive_cells'];count=c_rf['valid_cells']
    lines+=['',f'C有{positive}/{count}个有效RF细胞的三个seed配对ΔP方向均一致。'+
            ('因此空间变化在这些细胞中有跨seed重复性，但不据此宣称全体细胞均稳定。' if positive<count else '在本阶段全部有效RF细胞中观察到方向重复性。')+
            'TV/CDF差的幅度与跨seed方向一致性应一起看：非零TV本身不是充分的稳定性标准。','',
            'rf_population.csv包含每cell/seed/条件的原始gain、x/y centroid、second-moment radius、TV、centered radial CDF及完整ΔP数组。figures/delta_P_<cell>.png保存每cell三个条件的seed平均HIGH−LOW图；delta_P_maps_by_cell_seed_condition.npz保留各seed原数组和位置。颜色仅显示有符号ΔP，不用于圈RF面积。CDF网格和离散cone采样会影响数值。','',
            f'![空间形状与重复性]({(out/"figures/rf_normalized_dynamics.png").as_posix()})','',
            '## 4. 这种变化是否与两个spatial modes的相对贡献改变一致？','',
            '用真实forward的链式VJP获得J1/J2，并逐观测检验J1+J2恢复完整J。q_k=||J_k||²/(||J1||²+||J2||²)。同时保存signed projection <Jk,J>/||J||²和交叉项，避免将非正交模式的能量误当作互不重叠的加和。HIGH/LOW各自平均逐观测q后计算Δq；mode编号保持原basis顺序。','',
            '|条件|mean abs(Δq1)|Δq1三个seed同号 cells|TV vs abs(Δq1) Pearson|TV vs abs(Δq1) Spearman|径向差 vs abs(Δq1) Pearson|',
            '|---|---:|---:|---:|---:|---:|']
    missing=sorted(set(config['cells'])-{r['cell_id'] for r in c_rf['repeatability']})
    missing_notes=[]
    for cell in missing:
        cache=load_json(out/'checkpoints'/cell.replace('#','_')/str(SEEDS[0])/'analysis.json')
        original=cache['checks']['C']['observation_provenance']['original_indices_path']
        with np.load(original,allow_pickle=False) as indices:
            missing_notes.append(f'{cell}: LOW={len(indices["indices_LOW"])}, HIGH={len(indices["indices_HIGH"])}')
    if missing_notes:
        location=lines.index('## 3. RF normalized shape dynamics是否稳定存在？')+2
        lines[location:location]=['RF覆盖限制：原固定观测索引为 '+'；'.join(missing_notes)+
            '。缺少context配对的RF和mode指标标为UNVERIFIED，没有重新搜索观测；prediction仍包含全部22 cells。','']
    mean_tv=np.mean([v['TV'] for v in c_rf['per_seed_equal_cell_means'].values()])
    mean_radial=np.mean([v['centered_radial_CDF_L1_deg'] for v in c_rf['per_seed_equal_cell_means'].values()])
    inconsistent=[r['cell_id'] for r in c_rf['repeatability'] if not r['all_three_cosines_positive']]
    location=lines.index('## 4. 这种变化是否与两个spatial modes的相对贡献改变一致？')
    lines[location:location]=[f'C的equal-cell/seed mean TV={mean_tv:.8f}，中心化径向CDF差积分={mean_radial:.8f} degree。'+
        '归一化且移除重心平移后仍有可重复变化，但幅度较小；B的变化幅度更大，并没有带来更好的预测。'+
        ('C的ΔP方向例外为 '+', '.join(inconsistent)+'。' if inconsistent else ''),'']
    for condition in CONDITIONS:
        d=rf_stats[condition]
        lines.append(f'|{condition}|{d["mean_abs_delta_q1"]:.8f}|{d["delta_q1_same_sign_cells"]}/{d["valid_cells"]}|{number(d["TV_vs_abs_delta_q1_Pearson"])}|{number(d["TV_vs_abs_delta_q1_Spearman"])}|{number(d["centered_radial_vs_abs_delta_q1_Pearson"])}|')
    lines+=['',f'C的mean abs(Δq1)为{100*c_rf["mean_abs_delta_q1"]:.3f}个百分点，'+
            f'{c_rf["delta_q1_same_sign_cells"]}/{count}个细胞的Δq方向跨seed一致；但TV与abs(Δq1)的Spearman为'+
            f'{number(c_rf["TV_vs_abs_delta_q1_Spearman"])}，中心化径向差的Pearson为'+
            f'{number(c_rf["centered_radial_vs_abs_delta_q1_Pearson"])}。'+
            '本阶段未观察到C的空间变化幅度随模式重分配幅度增加的明显关系。B的对应关联更强。'+
            '因此两种现象均可重复，但尚不能确认模式重分配是C空间动态的主要来源。']
    lines+=['','|条件|LOW mean q1|HIGH mean q1|LOW mean q2|HIGH mean q2|',
            '|---|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        d=rf_stats[condition]
        lines.append(f'|{condition}|{d["LOW_q1_mean"]:.6f}|{d["HIGH_q1_mean"]:.6f}|{1-d["LOW_q1_mean"]:.6f}|{1-d["HIGH_q1_mean"]:.6f}|')
    lines+=['','上表对有效cell与seed等权平均，绝对贡献占比和HIGH−LOW变化分开报告。','',
            '补充有符号描述：第二mode的mean Δq2与半径变化的Pearson关联分别为 '+
            '；'.join(f'{c}: mean Δq2={rf_stats[c]["mean_delta_q2"]:+.6f}, r={number(rf_stats[c]["delta_radius_vs_delta_q2_Pearson"])}' for c in CONDITIONS)+
            '。mode编号为原basis顺序；拟合后的净Jacobian受E/I抵消影响，不能仅凭编号保证第二mode的有效RF始终更宽。']
    lines+=['','关联统计先在cell内平均seed，再按cell计算，仅作描述，没有额外p值或因果解释。模式贡献变化与profile变化可以一致，但q也会受模式间重叠/抵消影响；profile还可能因mode内部Jacobian变化而变化。因此不能仅凭非零Δq宣称其解释了全部ΔP。mode_contribution.csv保存每context的q、signed contribution、cross term和逐观测数值，及HIGH−LOW变化。','',
            f'![模式贡献与空间变化]({(out/"figures/mode_redistribution_association.png").as_posix()})','',
            '## 5. 下一步应保留conductance、修改spatial representation，还是两者都不升级？','']
    if stable_cb and stable_ca:
        decision='保留conductance作为后续候选，当前K=2空间表示暂时保持。C在本阶段同时超过matched B和同协议A，且归一化空间动态可重复；本轮没有必须立即修改空间表示的证据。模式重分配对C的空间变化解释仍不足，不升级这一机制解释，也不把已消费development结果当作独立确认。'
    elif stable_cb:
        decision='保留conductance作为候选，但当前空间表示不升级，优先重新审视空间表示。C−B的证据不能替代C−A；在超过A的证据仍不足时，不应整体替换当前RetiPath。'
    else:
        decision='本阶段conductance和新增空间表示均不升级。保留已完成结果，但本次稳定性证据不足以支持替换当前RetiPath；这不等于凭此关闭研究方向。'
    lines.append('**'+decision+'**')
    lines+=['','训练预算与证据范围：','',
            '|条件|inner达到3000上限的fits/66|refit steps median|min|max|',
            '|---|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        rows=[r for r in prediction if r['condition']==condition];steps=[r['refit_updates'] for r in rows]
        lines.append(f'|{condition}|{sum(r["reached_maximum"] for r in rows)}|{np.median(steps):.1f}|{min(steps)}|{max(steps)}|')
    lines+=['','patience=400 updates，max3000；达到上限不能称为充分收敛。实际全部选步、停止原因、minibatch loss及inner validation轨迹见training_per_seed.csv。没有追加seed、训练预算、RF loss、basis、区室、baseline或新数据实验。','',
            '直接验证：pilot回归与解析积分、patience按update计数、B/C共享初始化与参数数目、train-only guard及RMS、全部fresh refit的train NLL精确重放、J1+J2与完整prefix、固定模型source/input hash。所需状态保存在本run的checkpoints/，与原成果隔离。该审阅为本任务执行者复核，未冒称独立审计。','',
            '本阶段[16,20)是已消费development，历史模型/空间几何也已有研究选择背景。paired CI表达该22-cell集合的cell重采样不确定性，不能替代新的独立时间块或跨数据集确认。','',
            '**Phase 2完成后停止；本轮不实施上述下一步建议。**']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
