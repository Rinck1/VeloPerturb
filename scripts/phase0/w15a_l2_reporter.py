"""Post-completion L2 analysis/publication, independent from scientific execution.

No model is trained or intervened on here. Publication errors cannot change the
completed scientific grid. Window and seed summaries remain descriptive.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import cdist

from w4_artifacts import Artifacts,sha256
from w14_publication_recovery import publish_records
from w15a_cellflow_adapter import ROOT
from w15a_cellflow_registered_grid import ARMS,SEEDS
from w15a_cellflow_common_evaluation import KEYS,freeze_common_grid,weighted_energy
from w15_l1_common_alignment import join as join_l1


PAIRS=(('cf_kin','cf_static','group_kin_vs_static'),
       ('cf_kin','cf_Ssum','group_kin_vs_S_summary'),
       ('cf_kin','cf_directU','group_kin_vs_direct_U'),
       ('cf_kin_cell','cf_kin_cell_shuf','local_cell_vs_single_shuffle'))
HORIZONS=(5,10,15)
DEFAULT_L1=ROOT/'outputs/w15_l1_common_alignment_v1'


def load_json(path):return json.loads(Path(path).read_text())


def relative_gain(real,control):
    return None if control<=1e-12 else float((control-real)/control)


def require_ready(root):
    root=Path(root)
    required=(root/'status.json',root/'summary.json',root/'whole_grid_frozen.json',
              root/'evaluation/status.json',root/'evaluation/summary.json',root/'evaluation/frozen_grid.json')
    if not all(p.is_file() for p in required):return False
    status=load_json(root/'status.json');summary=load_json(root/'summary.json')
    evaluation=load_json(root/'evaluation/status.json');evaluated=load_json(root/'evaluation/summary.json')
    return (status['status']=='complete' and summary['status']=='REGISTERED_L2_DEVELOPMENT_GRID_COMPLETE'
        and evaluation['status']=='complete' and evaluated['status']=='COMMON_SAMPLE_DEVELOPMENT_EVALUATED')


def validate_rows(metrics):
    expected={(arm,key) for arm in ARMS for key in KEYS};seen=set()
    for row in metrics:
        identity=(row['arm'],row['key'])
        if identity in seen:raise ValueError('Duplicate arm/window score')
        seen.add(identity)
        start,stop,clock=KEYS[row['key']]
        if (row['source_hours'],row['target_hours'],row['kernel'])!=(start,stop,clock) or row['seeds']!=3:
            raise ValueError('Window or seed count does not match fixed common grid')
        if not np.isfinite(row['energy']) or row['energy'] < -1e-7:
            raise ValueError('Invalid Energy score')
    if seen!=expected:raise ValueError('Expected complete six-arm twelve-window evaluation')


def comparison_tables(metrics):
    validate_rows(metrics);lookup={(r['arm'],r['key']):r for r in metrics};windows=[];horizons=[];sensitivity=[]
    for real,control,name in PAIRS:
        for key,(start,stop,clock) in KEYS.items():
            a,b=lookup[(real,key)],lookup[(control,key)]
            row=dict(comparison=name,real_arm=real,control_arm=control,key=key,
                source_hours=start,target_hours=stop,horizon_hours=stop-start,kernel=clock,
                population_reference_self=start==5,
                real_energy=a['energy'],control_energy=b['energy'],
                absolute_energy_gain=float(b['energy']-a['energy']),
                relative_energy_gain=relative_gain(a['energy'],b['energy']))
            for metric in ('sampled_prediction_mean_mse','mode_proportion_TV'):
                if metric in a and metric in b:
                    row['real_'+metric]=a[metric];row['control_'+metric]=b[metric]
                    row['absolute_'+metric+'_gain']=float(b[metric]-a[metric])
            windows.append(row)
        for clock in ('decay','linear'):
            for duration in HORIZONS:
                group=[r for r in windows if r['comparison']==name and r['kernel']==clock and r['horizon_hours']==duration]
                real_energy=float(np.mean([r['real_energy'] for r in group]));control_energy=float(np.mean([r['control_energy'] for r in group]))
                row=dict(comparison=name,real_arm=real,control_arm=control,horizon_hours=duration,kernel=clock,
                    windows=len(group),window_keys=';'.join(r['key'] for r in group),
                    real_mean_energy=real_energy,control_mean_energy=control_energy,
                    absolute_energy_gain=control_energy-real_energy,
                    relative_energy_gain=relative_gain(real_energy,control_energy),
                    windows_with_positive_energy_gain=sum(r['absolute_energy_gain']>0 for r in group),
                    self_reference_windows=sum(r['population_reference_self'] for r in group),
                    aggregation='equal_window_mean_then_relative_gain',biological_replicates_inferred=False)
                for metric in ('sampled_prediction_mean_mse','mode_proportion_TV'):
                    if all('real_'+metric in r for r in group):
                        row['real_mean_'+metric]=float(np.mean([r['real_'+metric] for r in group]))
                        row['control_mean_'+metric]=float(np.mean([r['control_'+metric] for r in group]))
                horizons.append(row)
        for duration in HORIZONS:
            a=next(r for r in horizons if r['comparison']==name and r['horizon_hours']==duration and r['kernel']=='decay')
            b=next(r for r in horizons if r['comparison']==name and r['horizon_hours']==duration and r['kernel']=='linear')
            da,li=a['relative_energy_gain'],b['relative_energy_gain']
            pattern=('undefined_zero_control' if da is None or li is None else
                'both_positive' if da>0 and li>0 else 'decay_only_positive' if da>0 else
                'linear_only_positive' if li>0 else 'neither_positive')
            sensitivity.append(dict(comparison=name,horizon_hours=duration,decay_relative_gain=da,
                linear_relative_gain=li,clock_direction_pattern=pattern,
                difference_percentage_points=None if da is None or li is None else 100*(da-li),
                primary_clock='decay',clock_selected_from_results=False))
    return windows,horizons,sensitivity


def population_token_mask(conditions):
    """Prepared interface only: zero STANDARDIZED group token, nothing else.

    This is not U=0, not raw-reference k=0, and not an executed biological test.
    """
    value=np.asarray(conditions['population_token'])
    if value.ndim!=3 or value.shape[1]!=1 or not np.allclose(value,value[:1]):
        raise ValueError('Expected one constant population token for the full source cohort')
    result={key:np.asarray(v).copy() for key,v in conditions.items()}
    result['population_token']=np.zeros_like(value)
    return result


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def canonical_records(records):
    return sorted(records,key=lambda r:(r['arm'],r['seed']))


def verify_completed_grid(root):
    root=Path(root)
    if not require_ready(root):raise ValueError('Wait for complete18-grid and existing evaluation; no early scoring')
    study=load_json(root/'study_manifest.json');completed=load_json(root/'completed_models.json')['completed']
    if study['models']!=18 or study['prediction_windows']!=216 or study['arms']!=list(ARMS) or study['seeds']!=list(SEEDS):
        raise ValueError('Grid does not match the approved fixed18 protocol')
    if len(completed)!=18 or {(r['arm'],r['seed']) for r in completed}!={(a,s) for a in ARMS for s in SEEDS}:
        raise ValueError('Completed model list is not the exact18 grid')
    paths=[Path(r['prediction']) for r in completed]
    reference,frozen=freeze_common_grid(paths,reference_root=study['reference'],expected_seeds=SEEDS,expected_arms=ARMS)
    whole=load_json(root/'whole_grid_frozen.json');evaluation_index=load_json(root/'evaluation/frozen_grid.json')
    if whole['model_count']!=18 or whole['window_count']!=216 or canonical_records(whole['models'])!=canonical_records(frozen):
        raise ValueError('Whole-grid pre-evaluation index changed')
    if canonical_records(evaluation_index['predictions'])!=canonical_records(frozen):
        raise ValueError('Evaluation used a different prediction grid')
    for row in completed:
        training=Path(row['training']);tm=load_json(training/'summary.json')
        if (tm['arm'],tm['seed'])!=(row['arm'],row['seed']):
            raise ValueError('Training and prediction arm/seed identities differ')
        if sha256(training/'model.msgpack')!=row['model_sha256'] or tm['model_sha256']!=row['model_sha256']:
            raise ValueError('Model checkpoint changed after common predictions')
        predicted=load_json(Path(row['prediction'])/'prediction_manifest.json')
        if predicted['extra'].get('model_sha256')!=row['model_sha256']:
            raise ValueError('Predictions do not bind their recorded checkpoint')
    metrics=load_json(root/'evaluation/summary.json')['metrics'];validate_rows(metrics)
    return study,completed,reference,frozen,metrics


def seed_metrics(frozen,reference,centers_path):
    """Descriptive per-training-seed scores, only AFTER complete grid evaluation."""
    targets={}
    for stop,record in reference['target_caches'].items():
        if sha256(record['path'])!=record['sha256']:raise ValueError('Reference target cache changed')
        with np.load(record['path'],allow_pickle=False) as f:targets[stop]=f['target'].copy()
    centers=np.load(centers_path,allow_pickle=False);result=[]
    for record in frozen:
        with np.load(record['path'],allow_pickle=False) as f:
            for key,(start,stop,clock) in KEYS.items():
                x=np.asarray(f[key],dtype=float);y=targets[stop]
                if x.ndim==3:
                    q=np.asarray(f[key+'_q'],dtype=float);q=q/q.sum(1,keepdims=True)
                    w=q.reshape(-1)/len(q);x=x.reshape(-1,x.shape[-1])
                else:w=np.full(len(x),1/len(x))
                px=np.bincount(cdist(x,centers).argmin(1),weights=w,minlength=len(centers))
                py=np.bincount(cdist(y,centers).argmin(1),minlength=len(centers))/len(y)
                result.append(dict(arm=record['arm'],seed=record['seed'],key=key,source_hours=start,
                    target_hours=stop,horizon_hours=stop-start,kernel=clock,energy=weighted_energy(x,w,y),
                    population_reference_self=start==5,
                    sampled_prediction_mean_mse=float(np.mean((w@x-y.mean(0))**2)),
                    mode_proportion_TV=float(.5*np.abs(px-py).sum()),
                    inference_unit='training_seed_not_independent_biological_replicate'))
    return result


def collect_provenance(root,study,completed,reference,frozen):
    root=Path(root);paths=[Path(__file__)]
    paths += [root/name for name in ('study_manifest.json','completed_models.json','whole_grid_frozen.json','status.json','summary.json')]
    paths += [root/'evaluation'/name for name in ('summary.json','status.json','metrics.csv','frozen_grid.json')]
    for r in frozen:paths += [Path(r['path']),Path(r['manifest'])]
    for r in completed:paths += [Path(r['training'])/'summary.json',Path(r['training'])/'model.msgpack']
    for value,expected in study.get('data_hashes',{}).items():
        path=Path(value);path=path if path.is_absolute() else ROOT/path
        if sha256(path)!=expected:raise ValueError('Grid source/context artifact changed')
        paths.append(path)
    paths += [Path(r['path']) for r in reference['target_caches'].values()]
    pack=Path(study['pack']);paths += [pack/'manifest.json',pack/'token_normalizer.npz',pack/'training_target_mode_centers.npy']
    pm=load_json(pack/'manifest.json')
    for row in pm.get('training_groups',[])+pm.get('development_sources',[]):
        if sha256(row['path'])!=row['sha256']:raise ValueError('Source/training pack changed')
        paths.append(Path(row['path']))
    upstream=[]
    for field,expected in (('data_path',pm.get('config',{}).get('data_sha256')),
                           ('transform_path',pm.get('source_transform_sha256'))):
        value=pm.get('config',{}).get(field)
        if value is None:continue
        path=Path(value);path=path if path.is_absolute() else ROOT/path
        entry=dict(path=str(path),expected_sha256=expected,present=path.is_file(),verified=False)
        if path.is_file():
            digest=sha256(path)
            if expected is not None and digest!=expected:raise ValueError('Upstream dataset/geometry hash mismatch')
            paths.append(path);entry['verified']=True
        upstream.append(entry)
    unique=sorted({p.resolve() for p in paths},key=str)
    return dict(artifacts=[dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size) for p in unique],
        reference_hashes=reference['hashes'],scientific_freeze_commit=study['freeze_commit'],
        upstream_declarations=upstream,no_training=True,no_frozen_weight_intervention_executed=True,Traxler_read=False)


def percent(value):return '未定义' if value is None else f'{100*value:+.2f}%'


def render_report(summary,windows,horizons,sensitivity,l1_comparison=None):
    selected={r['comparison']:r for r in horizons if r['kernel']=='decay' and r['horizon_hours']==5}
    gains=[selected[name]['relative_energy_gain'] for name in ('group_kin_vs_static','group_kin_vs_S_summary','group_kin_vs_direct_U')]
    if all(g is not None and g>0 for g in gains):
        finding='群体动力学token在三个5h窗口的平均Energy上优于static、S摘要和直接U。'
    elif gains[0] is not None and gains[0]>0 and gains[1] is not None and gains[1]>0:
        finding='群体动力学token超过static和S摘要，但没有超过直接U；动力学变换的额外价值尚未建立。'
    elif gains[0] is not None and gains[0]>0:
        finding='群体动力学token超过static，但尚未超过S摘要；群体摘要本身仍是可行解释。'
    else:finding='本轮没有看到群体动力学token相对static的主窗口平均改善。'
    lines=['# CellFlow动力学插件：固定L2开发结果','',finding,'',
        '同一128×2基座、固定5000步和学习率；六臂各三训练seed。复用W13源细胞ID和同一目标缓存，'
        '先冻结18模型的216个窗口/时钟预测，再统一评分。以下主结果是三seed等权混合分布，不是挑选最好seed。'
        '动力学为解析剪接残差的PCA投影与源参考相减，不是原版GFG/JVP。','',
        '## 时间跨度与对照','',
        '| 预测跨度 | 窗口数 | 群体kin对static | 对S摘要 | 对直接U | 逐细胞kin对单置换shuffle |',
        '|---|---:|---:|---:|---:|---:|']
    for duration in HORIZONS:
        rows={r['comparison']:r for r in horizons if r['kernel']=='decay' and r['horizon_hours']==duration}
        lines.append(f"| {duration}h | {rows['group_kin_vs_static']['windows']} | "+' | '.join(
            percent(rows[name]['relative_energy_gain']) for _,_,name in PAIRS)+' |')
    lines += ['','正数表示Energy降低，负数表示对照更好。5h为三个相邻窗口等权平均；10h为两个窗口，15h为一个窗口。','',
        '## 六个窗口原样报告','',
        '| 窗口 | 群体自身参考 | kin Energy | static Energy | 对static | 对S摘要 | 对直接U |',
        '|---|---|---:|---:|---:|---:|---:|']
    for start,stop in ((5,10),(10,15),(15,20),(5,15),(10,20),(5,20)):
        key=f'{start:02d}_{stop:02d}_decay';rows={r['comparison']:r for r in windows if r['key']==key}
        row=rows['group_kin_vs_static']
        lines.append(f"| {start}→{stop}h | {'是' if row['population_reference_self'] else '否'} | {row['real_energy']:.6f} | {row['control_energy']:.6f} | "+' | '.join(
            percent(rows[name]['relative_energy_gain']) for name in ('group_kin_vs_static','group_kin_vs_S_summary','group_kin_vs_direct_U'))+' |')
    lines += ['','05h起点以自身作参考，cf_kin、cf_Ssum、cf_directU三个原始群体token按定义全为0；'
        '训练标准化后仅是常量，不一定是数值0，不含该群体额外的当前动力学信息，逐细胞cf_kin_cell仍可非零。'
        'origin5三个窗口仍全部保留，但其群体插件差异不能单独证明source U当前信息增量；'
        'origin10/15包含相对05h的实际变化。此标注由冻结特征定义得出，没有新增筛选门。','',
        '## 时间映射敏感性','',
        '| 跨度 | 对照 | decay主时钟 | linear敏感性 |','|---|---|---:|---:|']
    for row in sensitivity:
        lines.append(f"| {row['horizon_hours']}h | {row['comparison']} | {percent(row['decay_relative_gain'])} | {percent(row['linear_relative_gain'])} |")
    if l1_comparison is not None:
        l1_rows=l1_comparison['horizon_summary']
        lines += ['','## 与L1在相同六个decay窗口上对齐','',
            '| 跨度 | L1 real Energy | L1 static Energy | CellFlow kin Energy | CellFlow static Energy |',
            '|---|---:|---:|---:|---:|']
        for duration in HORIZONS:
            kin=next(r for r in l1_rows if r['common_arm']=='cf_kin' and r['horizon_hours']==duration)
            static=next(r for r in l1_rows if r['common_arm']=='cf_static' and r['horizon_hours']==duration)
            lines.append(f"| {duration}h | {kin['L1_real_energy']:.6f} | {kin['L1_static_energy']:.6f} | {kin['common_energy']:.6f} | {static['common_energy']:.6f} |")
        lines += ['','双方使用相同源ID、目标缓存、三seed等权混合和同一原始Energy公式。'
            'L1保留解析均值锚，L2学习均值，参数量和训练步数也不同，因此velocity增量首先比较各线自己的static。'
            'L1只有真实的六个decay窗口；没有构造或推测L1 linear结果。']
    lines += ['','## 机理与范围','',
        'S摘要检验是否仅需要额外群体信息，直接U检验速度构造是否优于第二模态本身；'
        '逐细胞kin与源群体内shuffle单列，只使用一个固定置换perm_seed0，三训练seed不等于三个置换。'
        'sci-FATE2只有一个条件，因此没有检验群体token跨条件对应关系CF-H3。','',
        '这是同系列开发结果。窗口有重叠，训练seed也不是独立生物重复；不把这些描述性差值当作生物重复置信区间。'
        '模式比例使用训练目标k-means的表达成分，不是已观测细胞命运。','',
        '群体token遮蔽接口已准备，但本轮没有执行任何冻结权重的新干预或额外训练。','',
        f"详细记录：`{summary['artifact_directory']}`；包含逐窗口、跨度、时钟及逐训练seed CSV和全部输入产物哈希。",'']
    return '\n'.join(lines)


def analyze(root,output,*,report_path=None,publish=False,publisher=publish_records,l1_directory=DEFAULT_L1):
    root=Path(root);study,completed,reference,frozen,metrics=verify_completed_grid(root)
    windows,horizons,sensitivity=comparison_tables(metrics)
    run=Artifacts(output,seed=list(SEEDS),inputs=[Path(__file__),root/'study_manifest.json',root/'summary.json',root/'evaluation/metrics.csv'])
    l1_comparison=None
    if l1_directory is not None:
        l1_comparison=join_l1(l1_directory,root/'evaluation',run.directory/'l1_common_comparison',expected_arms=ARMS)
    provenance=collect_provenance(root,study,completed,reference,frozen)
    if l1_directory is not None:
        alignment_paths=[Path(join_l1.__code__.co_filename)]
        alignment_paths += [p for p in Path(l1_directory).rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.md')]
        alignment_paths += [p for p in (run.directory/'l1_common_comparison').rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.md')]
        existing={r['path'] for r in provenance['artifacts']}
        provenance['artifacts'] += [dict(path=str(p.resolve()),sha256=sha256(p),bytes=p.stat().st_size)
            for p in sorted(set(alignment_paths),key=str) if str(p.resolve()) not in existing]
        provenance['L1_alignment_directory']=str(Path(l1_directory).resolve())
    run.json('provenance.json',provenance)
    seeds=seed_metrics(frozen,reference,Path(study['pack'])/'training_target_mode_centers.npy')
    annotated_metrics=[dict(r,population_reference_self=r['source_hours']==5) for r in metrics]
    for name,rows in (('window_metrics.csv',annotated_metrics),('window_comparisons.csv',windows),
        ('horizon_summary.csv',horizons),('clock_sensitivity.csv',sensitivity),('per_seed_metrics.csv',seeds)):
        write_csv(run.directory/name,rows)
    intervention=dict(status='PREPARED_NOT_RUN',method='population_token_mask',
        change='standardized_population_token_to_zero_equals_training_mean_not_raw_U_zero',
        weights='frozen_cf_kin_only',source_context='complete_source_then_original_W13_terminal_IDs',
        unchanged=['source_state','cell_token','bio_time','bio_delta','perturbation','model_weights'],
        new_predictions_executed=False,new_ablation_training=False)
    run.json('prepared_intervention.json',intervention)
    summary=dict(status='L2_POST_COMPLETION_REPORT_COMPLETE',science_status='complete',grid=str(root.resolve()),
        artifact_directory=str(run.directory.resolve()),models=18,prediction_windows=216,
        primary_horizon_hours=5,primary_clock='decay',horizon_comparisons=horizons,
        same_series_development=True,independent_biological_replicates_inferred=False,
        local_permutation_seeds=[0],population_CF_H3='NOT_TESTED_SINGLE_CONDITION',
        population_self_reference_origin_hours=5,local_cell_token_zero_not_implied=True,
        frozen_weight_intervention_executed=False,Traxler_read=False,L1_comparison=l1_comparison,
        L1_linear_fabricated=False)
    report=render_report(summary,windows,horizons,sensitivity,l1_comparison)
    (run.directory/'RESULTS.md').write_text(report)
    if report_path is not None:
        path=Path(report_path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(report);temporary.replace(path)
    run.json('summary.json',summary);run.json('status.json',dict(status='complete',science_status='complete'))
    # Scientific reporting is COMPLETE before optional independent publication.
    publication=dict(status='not_requested',science_status='complete')
    if publish:
        paths=[p for p in run.directory.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.md') and p.name!='publication.json']
        if report_path is not None:paths.append(Path(report_path))
        try:publication=dict(publisher(ROOT,paths,'W15-A: report complete fixed L2 common-sample grid'),science_status='complete')
        except Exception as error:publication=dict(status='failed',error=f'{type(error).__name__}: {error}',science_status='complete',retry='publish existing records only; never retrain')
    run.json('publication.json',publication)
    return dict(**summary,publication=publication)


def watch(root,output,*,report_path=None,publish=False,timeout_seconds=12*3600,poll_seconds=20,training_job_status=None,l1_directory=DEFAULT_L1):
    if poll_seconds<=0 or poll_seconds>60:raise ValueError('Polling must remain bounded at 60 seconds')
    deadline=time.monotonic()+timeout_seconds
    while not require_ready(root):
        if training_job_status is not None and Path(training_job_status).exists():
            state=load_json(training_job_status)
            if state.get('status')=='failed':raise RuntimeError('Scientific grid failed; reporter will not restart training')
        if time.monotonic()>deadline:raise TimeoutError('Complete L2 grid/evaluation not ready before reporting deadline')
        status=Path(root)/'status.json';state=load_json(status) if status.exists() else {}
        print(json.dumps(dict(status='waiting_for_full_scientific_grid',completed_models=state.get('completed_models'),
            stage=state.get('status'),arm=state.get('arm'),current_seed=state.get('current_seed'))),flush=True)
        time.sleep(poll_seconds)
    return analyze(root,output,report_path=report_path,publish=publish,l1_directory=l1_directory)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',required=True);p.add_argument('--output',required=True)
    p.add_argument('--report',default=str(ROOT/'docs/current/w15a_l2_results.md'))
    p.add_argument('--publish',action='store_true');p.add_argument('--wait',action='store_true')
    p.add_argument('--l1-alignment',default=str(DEFAULT_L1))
    p.add_argument('--training-job-status',default=None);p.add_argument('--timeout-seconds',type=int,default=12*3600)
    args=p.parse_args()
    result=(watch(args.grid,args.output,report_path=args.report,publish=args.publish,
        timeout_seconds=args.timeout_seconds,training_job_status=args.training_job_status,l1_directory=args.l1_alignment) if args.wait else
        analyze(args.grid,args.output,report_path=args.report,publish=args.publish,l1_directory=args.l1_alignment))
    print(json.dumps(result,indent=2))
