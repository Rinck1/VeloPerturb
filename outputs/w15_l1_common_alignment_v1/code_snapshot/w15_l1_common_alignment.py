"""Align genuine six-window L1 forecasts with complete L2/L3 common evaluations.

Never creates a fictitious L1 linear-clock forecast. Reuses frozen candidates,
original seed-mixture weights, original source order and original target caches.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts,sha256
from w7_router_scaling import write_csv
from w15a_cellflow_common_evaluation import (DEFAULT_REFERENCE,KEYS,WINDOWS,
    reference_sources,weighted_energy)

ROOT=Path(__file__).resolve().parents[2]
L1_DEFAULT=ROOT/'outputs/w15_l1_routing_20261021_v1'
W13_DEFAULT=ROOT/'outputs/w13_time_program_v1/ode_refinement'
VARIANTS={'soft':'velocity_router','static':'static_router','shuffled':'shuffled_velocity_router'}


def completed_status(directory):
    """Only read status initially; do not inspect partial evaluation summaries."""
    status=json.loads((Path(directory)/'status.json').read_text())
    if status.get('status')!='complete':
        raise ValueError('Complete evaluation required; partial results are not opened')


def original_mixture(x,q,source_cells,seeds=(0,1,2)):
    x=np.asarray(x);q=np.asarray(q)
    if x.ndim!=3 or q.shape!=x.shape[:2] or len(x)!=source_cells*len(seeds):
        raise ValueError('Expected original seed-major candidates with the common source count')
    if not np.isfinite(x).all() or not np.isfinite(q).all() or (q<0).any():
        raise ValueError('Invalid frozen candidate values or weights')
    if not np.allclose(q.sum(1),1.,atol=1e-6):raise ValueError('Invalid source-conditional q')
    # Original W13 concatenates equal-size source cohorts seed-major. This is
    # precisely 1/3 per seed, 1/N per source and original q within each source.
    return x.reshape(-1,x.shape[-1]),q.reshape(-1)/len(q)


def freeze_l1(l1_root=L1_DEFAULT,reference_root=DEFAULT_REFERENCE,w13_root=W13_DEFAULT):
    l1_root,w13_root=Path(l1_root),Path(w13_root)
    completed_status(l1_root);reference=reference_sources(reference_root)
    index_path=l1_root/'frozen_predictions.json';index=json.loads(index_path.read_text())
    if not index['all_predictions_ready_before_evaluation'] or index['optimizer_steps']!=0:
        raise ValueError('L1 forecast freeze contract changed')
    for path,digest in index['data_hashes'].items():
        if sha256(path)!=digest:raise ValueError('Original L1 input changed: '+path)
    mean_record=json.loads((w13_root/'mean_models.json').read_text())
    for path,digest in mean_record['data_hashes'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Original W13 input changed: '+path)
    for record in mean_record['models']:
        if sha256(record['path'])!=record['sha256']:raise ValueError('Original mean model changed')
    old_index=json.loads((w13_root/'frozen_predictions.json').read_text())
    records=[];identities=set()
    for record in index['predictions']:
        variant=record['variant']
        if variant not in VARIANTS:continue
        start,stop=record['source_hour'],record['target_hour'];identity=(variant,start,stop)
        if identity in identities or (start,stop) not in WINDOWS:raise ValueError('Unexpected L1 window identity')
        identities.add(identity)
        if sha256(record['path'])!=record['sha256']:raise ValueError('Frozen L1 prediction changed')
        arm=VARIANTS[variant]
        matches=[r for r in old_index['predictions'] if
                 (r['arm'],r['source_hour'],r['target_hour'])==(arm,start,stop)]
        if len(matches)!=1:raise ValueError('Original W13 forecast is ambiguous')
        old=matches[0]
        if sha256(old['path'])!=old['sha256']:raise ValueError('Original W13 forecast changed')
        with np.load(record['path'],allow_pickle=False) as f,np.load(old['path'],allow_pickle=False) as g:
            x,q=f['candidates'],f['q']
            if not np.array_equal(x,g['candidates']) or not np.array_equal(q,g['q']):
                raise ValueError('L1 shared-scoring adapter requires bitwise original W13 x and q')
            original_mixture(x,q,len(reference['sources'][start]['cell_ids']))
        records.append(dict(variant=variant,arm='L1_'+arm,source_hours=start,target_hours=stop,
            key=f'{start:02d}_{stop:02d}_decay',kernel='decay',path=record['path'],sha256=record['sha256'],
            original_path=old['path'],original_sha256=old['sha256'],original_candidates_bitwise=True,
            source_ids_sha256=reference['sources'][start]['source_ids_sha256']))
    if identities!={(v,a,b) for v in VARIANTS for a,b in WINDOWS}:
        raise ValueError('All original three arms by six windows are required')
    proof=json.loads((l1_root/'reference_reproduction.json').read_text())
    metrics=w13_root/'metrics.csv'
    if proof['checked']!=18 or sha256(metrics)!=proof['reference_sha256']:
        raise ValueError('Original18-row reference proof changed')
    return reference,records,dict(l1_frozen_index_sha256=sha256(index_path),
        W13_metrics_path=str(metrics.resolve()),W13_metrics_sha256=sha256(metrics),
        shared_scorer_path=str((ROOT/'scripts/phase0/w15a_cellflow_common_evaluation.py').resolve()),
        shared_scorer_sha256=sha256(ROOT/'scripts/phase0/w15a_cellflow_common_evaluation.py'))


def reproduce(output,*,l1_root=L1_DEFAULT,reference_root=DEFAULT_REFERENCE,w13_root=W13_DEFAULT,tolerance=1e-5):
    reference,frozen,proof=freeze_l1(l1_root,reference_root,w13_root)
    run=Artifacts(output,seed=[0,1,2],inputs=[Path(__file__),
        Path(l1_root)/'frozen_predictions.json',Path(w13_root)/'metrics.csv'])
    run.json('frozen_grid.json',dict(predictions=frozen,reference_hashes=reference['hashes'],
        target_caches=reference['target_caches'],all18_predictions_verified_before_target_read=True,
        source_order='original_W13_source_samples_seed_major',linear_clock_predictions_created=False,**proof))
    targets={}
    for stop,cache in reference['target_caches'].items():
        if sha256(cache['path'])!=cache['sha256']:raise ValueError('Original target cache changed')
        with np.load(cache['path'],allow_pickle=False) as f:targets[stop]=f['target'].copy()
    originals=list(csv.DictReader(Path(proof['W13_metrics_path']).open()))
    rows=[]
    for record in frozen:
        start,stop=record['source_hours'],record['target_hours']
        if sha256(record['path'])!=record['sha256']:raise ValueError('L1 forecast changed after freeze')
        with np.load(record['path'],allow_pickle=False) as f:
            x,w=original_mixture(f['candidates'],f['q'],len(reference['sources'][start]['cell_ids']))
        value=weighted_energy(x,w,targets[stop])
        old=next(r for r in originals if (r['arm'],int(r['source_hour']),int(r['target_hour']))==
                 (VARIANTS[record['variant']],start,stop))
        rows.append(dict(arm=record['arm'],variant=record['variant'],key=record['key'],
            source_hours=start,target_hours=stop,kernel='decay',seeds=3,
            source_cells_per_seed=len(reference['sources'][start]['cell_ids']),target_cells=len(targets[stop]),
            energy=value,original_W13_energy=float(old['energy_distance']),
            absolute_reproduction_difference=abs(value-float(old['energy_distance']))))
        run.json('status.json',dict(status='reproducing_L1_raw_weighted_energy',completed=len(rows),total=18))
    write_csv(run.directory/'metrics.csv',rows)
    maximum=max(r['absolute_reproduction_difference'] for r in rows)
    if maximum>tolerance:raise ArithmeticError(f'Raw shared-scoring reproduction exceeds tolerance: {maximum}')
    result=dict(status='L1_W13_SHARED_SCORER_REPRODUCED',metrics=rows,reference_hashes=reference['hashes'],
        target_caches=reference['target_caches'],maximum_absolute_energy_difference=maximum,tolerance=tolerance,
        seeds=[0,1,2],windows_per_arm=6,linear_clock_not_available=True,original_candidates_bitwise=True,
        original_mean_anchor_unchanged=True,no_retraining=True,no_reintegration=True,no_resampling=True,
        metrics_sha256=sha256(run.directory/'metrics.csv'),frozen_grid_sha256=sha256(run.directory/'frozen_grid.json'),**proof)
    run.json('summary.json',result)
    (run.directory/'RESULTS.md').write_text('# L1: genuine six-window common-scorer reproduction\n\n'+json.dumps(result,indent=2)+'\n')
    run.json('status.json',dict(status='complete'));return result


def joined_decay_rows(l1,common,expected_arms):
    expected_arms=set(expected_arms)
    if not expected_arms or set(common['arms'])!=expected_arms:
        raise ValueError('Complete explicitly requested common arm grid is required')
    if common['expected_seeds']!=[0,1,2] or l1['seeds']!=[0,1,2]:
        raise ValueError('This comparison requires all three original seeds')
    if common['reference_hashes']!=l1['reference_hashes'] or common['target_caches']!=l1['target_caches']:
        raise ValueError('L1 and common evaluation do not use the exact same source/target reference')
    if l1['windows_per_arm']!=6 or not l1['linear_clock_not_available']:
        raise ValueError('L1 has only its genuine six-window primary clock')
    for arm in expected_arms:
        rows=[r for r in common['metrics'] if r['arm']==arm]
        if len(rows)!=12 or {r['key'] for r in rows}!=set(KEYS):
            raise ValueError('Every common arm must have all twelve complete windows')
    by={(r['variant'],r['key']):r for r in l1['metrics']}
    if len(by)!=18 or set(by)!={(v,f'{a:02d}_{b:02d}_decay') for v in VARIANTS for a,b in WINDOWS}:
        raise ValueError('L1 original18-row grid is incomplete or duplicated')
    rows=[]
    for item in common['metrics']:
        if item['kernel']!='decay':continue
        key=item['key'];a,b,_=KEYS[key]
        original={v:by[(v,key)] for v in VARIANTS}
        if any(r['source_cells_per_seed']!=item['source_cells_per_seed'] or
               r['target_cells']!=item['target_cells'] for r in original.values()):
            raise ValueError('Source or target empirical sample sizes differ')
        static=original['static']['energy'];real=original['soft']['energy']
        rows.append(dict(common_arm=item['arm'],key=key,source_hours=a,target_hours=b,horizon_hours=b-a,
            kernel='decay',common_energy=item['energy'],L1_real_energy=real,L1_static_energy=static,
            L1_shuffled_energy=original['shuffled']['energy'],
            common_gain_vs_L1_static=1-item['energy']/static,
            common_gain_vs_L1_real=1-item['energy']/real,
            source_cells_per_seed=item['source_cells_per_seed'],target_cells=item['target_cells'],seeds=3))
    return rows


def join(l1_directory,common_directory,output,*,expected_arms):
    l1_directory,common_directory=Path(l1_directory),Path(common_directory)
    # Read status only before accepting either side. A live L2/L3 grid is never
    # inspected merely because its first arm or first seed happens to exist.
    completed_status(l1_directory);completed_status(common_directory)
    l1=json.loads((l1_directory/'summary.json').read_text())
    common=json.loads((common_directory/'summary.json').read_text())
    if l1['status']!='L1_W13_SHARED_SCORER_REPRODUCED' or common['status']!='COMMON_SAMPLE_DEVELOPMENT_EVALUATED':
        raise ValueError('Completed shared-scorer outputs required')
    if sha256(l1_directory/'metrics.csv')!=l1['metrics_sha256'] or sha256(l1_directory/'frozen_grid.json')!=l1['frozen_grid_sha256']:
        raise ValueError('Completed L1 reproduction records changed')
    grid=json.loads((common_directory/'frozen_grid.json').read_text())
    if not grid['all_expected_seeds_and_twelve_windows_complete']:
        raise ValueError('Common whole-grid freeze not complete')
    required={(a,s) for a in expected_arms for s in (0,1,2)}
    actual={(r['arm'],r['seed']) for r in grid['predictions']}
    if len(grid['predictions'])!=len(required) or actual!=required:
        raise ValueError('Common arm by seed frozen grid is incomplete')
    for r in grid['predictions']:
        if sha256(r['path'])!=r['sha256'] or sha256(r['manifest'])!=r['manifest_sha256']:
            raise ValueError('A completed common prediction changed')
    rows=joined_decay_rows(l1,common,expected_arms)
    # Unique parent records avoid the same-named summary.json snapshot collision.
    run=Artifacts(output,seed=[0,1,2],inputs=[Path(__file__)])
    run.json('comparison_inputs.json',dict(L1_summary_path=str((l1_directory/'summary.json').resolve()),
        L1_summary_sha256=sha256(l1_directory/'summary.json'),
        common_summary_path=str((common_directory/'summary.json').resolve()),
        common_summary_sha256=sha256(common_directory/'summary.json'),
        common_frozen_grid_sha256=sha256(common_directory/'frozen_grid.json')))
    write_csv(run.directory/'paired_decay_metrics.csv',rows)
    averages=[]
    for arm in expected_arms:
        for horizon in (5,10,15):
            selected=[r for r in rows if r['common_arm']==arm and r['horizon_hours']==horizon]
            e,real,static=[float(np.mean([r[k] for r in selected]))
                           for k in ('common_energy','L1_real_energy','L1_static_energy')]
            averages.append(dict(common_arm=arm,horizon_hours=horizon,windows=len(selected),common_energy=e,
                L1_real_energy=real,L1_static_energy=static,common_gain_vs_L1_real=1-e/real,
                common_gain_vs_L1_static=1-e/static))
    write_csv(run.directory/'horizon_summary.csv',averages)
    result=dict(status='L1_COMMON_GENUINE_DECAY_COMPARISON_COMPLETE',common_arms=list(expected_arms),
        reference_hashes=l1['reference_hashes'],horizon_summary=averages,
        compared_windows_per_arm=6,L1_linear_fabricated=False,all_seeds_and_same_empirical_samples=True,
        scorer='same_raw_weighted_energy_V_statistic',
        interpretation='common_empirical_distribution; L1 retains analytical mean anchor whereas L2/L3 learn mean',
        no_retraining=True,no_reintegration=True,no_new_target_read=True)
    run.json('summary.json',result)
    (run.directory/'RESULTS.md').write_text('# L1/L2/L3: genuine six decay windows\n\n'+json.dumps(result,indent=2)+'\n')
    run.json('status.json',dict(status='complete'));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['reproduce','join'])
    parser.add_argument('--output',required=True);parser.add_argument('--l1-root',default=str(L1_DEFAULT))
    parser.add_argument('--reference',default=str(DEFAULT_REFERENCE));parser.add_argument('--w13-root',default=str(W13_DEFAULT))
    parser.add_argument('--l1-reproduction');parser.add_argument('--common-evaluation');parser.add_argument('--expected-common-arms')
    args=parser.parse_args()
    if args.action=='reproduce':
        result=reproduce(args.output,l1_root=args.l1_root,reference_root=args.reference,w13_root=args.w13_root)
    else:
        if not args.expected_common_arms:parser.error('join requires explicit --expected-common-arms')
        result=join(args.l1_reproduction,args.common_evaluation,args.output,
                    expected_arms=tuple(args.expected_common_arms.split(',')))
    print(json.dumps(result,indent=2))
