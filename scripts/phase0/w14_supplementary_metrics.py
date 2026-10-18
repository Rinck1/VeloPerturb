"""All-seed scores and descriptive cell bootstrap, without changing P1/P2."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial.distance import cdist

from w4_artifacts import Artifacts,sha256
from w7_router_scaling import write_csv
from w14_data import CountReader,SourceTransform,load_npz
from w14_loro import energy,freeze_grid,verify_lock


ROOT=Path(__file__).resolve().parents[2]


def source_kernels(candidates,q,y,block=16):
    """Cell kernels integrate experts/seeds before a source-cell bootstrap."""
    n,k,d=candidates.shape;flat=candidates.reshape(-1,d);weights=q/q.sum(1,keepdims=True)
    a=np.empty((n,n));b=np.empty((n,len(y)))
    for i in range(0,n,block):
        m=min(block,n-i);x=flat[i*k:(i+m)*k]
        distance=cdist(x,flat).reshape(m,k,n,k)
        a[i:i+m]=np.einsum('aibj,ai,bj->ab',distance,weights[i:i+m],weights,optimize=True)
        cross=cdist(x,y).reshape(m,k,len(y))
        b[i:i+m]=np.einsum('aij,ai->aj',cross,weights[i:i+m],optimize=True)
    return a,b,cdist(y,y)


def bootstrap_energy(a,b,d,source_weights,target_weights,dimensions):
    aw=source_weights;bw=target_weights
    return (2*np.einsum('ij,ij->i',aw@b,bw)-np.einsum('ij,ij->i',aw@a,aw)
        -np.einsum('ij,ij->i',bw@d,bw))/np.sqrt(dimensions)


def run(root,output):
    root=Path(root);deadline=time.monotonic()+7*86400
    while time.monotonic()<deadline:
        states=[json.loads((root/f'worker{i}.json').read_text()).get('status')
            if (root/f'worker{i}.json').exists() else 'not_started' for i in range(3)]
        if 'failed' in states:raise RuntimeError('Primary worker failed; do not score partial grid')
        if all(v=='complete' for v in states):break
        time.sleep(20)
    else:raise TimeoutError('W14 predictions incomplete')
    m=verify_lock(root);c=m['config'];frozen=freeze_grid(root,m)
    run=Artifacts(output,seed=[0,1,2],inputs=[root/'study_manifest.json',Path(__file__)])
    run.json('frozen_grid.json',dict(predictions=frozen))
    reader=CountReader(ROOT/c['data_path'],c);individual=[];intervals=[]
    names=['static','real','direct_U','nonlinear_SU']+[f'shuffled_v_perm{i}' for i in c['shuffle_seeds']]
    for rep in c['replicates']:
        transform=SourceTransform.load(root/'folds'/rep/'source_transform.npz')
        for day in c['training_days'][:-1]:
            target=reader.load(rep,day+1);y=transform.state(target['s'])
            ti=np.random.default_rng(0).choice(len(y),min(len(y),512),replace=False);y=y[ti]
            boot_rng=np.random.default_rng(np.random.SeedSequence([20261018,int(rep[1:]),day]))
            source_boot=None;target_boot=boot_rng.multinomial(len(y),np.full(len(y),1/len(y)),size=1000)/len(y)
            for name in names:
                selected=sorted([r for r in frozen if r['job']['heldout']==rep and r['source_day']==day and r['job']['name']==name],
                    key=lambda r:r['job']['seed'])
                packs=[load_npz(r['path']) for r in selected]
                for item,p in zip(selected,packs):
                    value=energy(p['candidates'].reshape(-1,c['dimensions']),p['q'].ravel()/len(p['q']),y,'cpu')
                    individual.append(dict(heldout=rep,source_day=day,arm=name,seed=item['job']['seed'],energy=value))
                x=np.stack([p['candidates'] for p in packs],axis=1).reshape(len(packs[0]['q']),-1,c['dimensions'])
                q=np.stack([p['q'] for p in packs],axis=1).reshape(len(x),-1)/len(packs)
                if source_boot is None:
                    source_boot=boot_rng.multinomial(len(x),np.full(len(x),1/len(x)),size=1000)/len(x)
                a,b,d=source_kernels(x,q,y)
                values=bootstrap_energy(a,b,d,source_boot,target_boot,c['dimensions'])
                lo,hi=np.quantile(values,[.025,.975])
                intervals.append(dict(heldout=rep,source_day=day,arm=name,energy_ci_low=float(lo),energy_ci_high=float(hi),
                    repetitions=1000,scope='conditional_frozen_prediction_cell_bootstrap_not_biological_CI'))
                run.json('status.json',dict(status='scoring',heldout=rep,source_day=day,arm=name,completed=len(intervals)))
    write_csv(run.directory/'per_seed_metrics.csv',individual);write_csv(run.directory/'descriptive_cell_intervals.csv',intervals)
    result=dict(status='W14_SUPPLEMENTARY_METRICS_COMPLETE',per_seed_rows=len(individual),interval_rows=len(intervals),
        bootstrap_repetitions=1000,primary_P1_P2_changed=False,formal_significance_claim=False,
        uncertainty='Cell reweighting of frozen predictions; no refitting/reforecasting; not between-replicate uncertainty')
    run.json('summary.json',result)
    report='# W14补充：逐seed分数与描述性细胞区间\n\n'+json.dumps(result,indent=2)+'\n'
    (run.directory/'RESULTS.md').write_text(report)
    while time.monotonic()<deadline:
        p=ROOT/'outputs/w14_program_v1/status.json';state=json.loads(p.read_text())
        if state['status']=='failed':raise RuntimeError('Primary completion failed; supplemental values retained')
        if state['status']=='complete':break
        time.sleep(20)
    else:raise TimeoutError('Primary W14 completion timeout')
    primary=list(csv.DictReader((root/'evaluation/metrics.csv').open()))
    retention=[];real_gains=[];cohort_gains=[]
    for rep in c['replicates']:
        for day in c['training_days'][:-1]:
            by={r['arm']:r for r in primary if r['heldout']==rep and int(r['source_day'])==day}
            static=float(by['static']['energy']);real=float(by['real']['energy']);averaged=float(by['real']['cohort_average_energy'])
            gain=1-real/static;cohort_gain=1-averaged/static
            real_gains.append(gain);cohort_gains.append(cohort_gain)
            retention.append(dict(heldout=rep,source_day=day,real_gain=gain,cohort_gain=cohort_gain,
                retained_gain_fraction=cohort_gain/gain if gain>1e-12 else None,
                interpretation='not_applicable_when_original_gain_nonpositive' if gain<=1e-12 else 'frozen_recentered_q_intervention'))
    write_csv(run.directory/'routing_retention.csv',retention)
    result['overall_retained_gain_fraction']=float(np.mean(cohort_gains)/np.mean(real_gains)) if np.mean(real_gains)>1e-12 else None
    result['primary_metrics_sha256']=sha256(root/'evaluation/metrics.csv')
    run.json('summary.json',result)
    report='# W14补充：逐seed分数与描述性细胞区间\n\n'+json.dumps(result,indent=2)+'\n'
    (run.directory/'RESULTS.md').write_text(report)
    doc=ROOT/'docs/current/w14_supplementary_results.md'
    if doc.exists():raise ValueError('Prior supplementary report retained')
    doc.write_text(report)
    files=[str(doc.relative_to(ROOT))]+[str((run.directory/n).relative_to(ROOT)) for n in
        ('RESULTS.md','summary.json','per_seed_metrics.csv','descriptive_cell_intervals.csv','routing_retention.csv')]
    subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--only','-m','W14: add all-seed scores and descriptive cell intervals','--',*files],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','master'],cwd=ROOT,check=True)
    run.json('status.json',dict(status='complete',published=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/w14_loro_v1');p.add_argument('--output',required=True)
    a=p.parse_args();run(a.root,a.output)
