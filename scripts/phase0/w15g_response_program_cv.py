"""Development CV of the small response-program probe on training KOs only.

The original source transform remains fixed (it used all outer-training source
KOs). Response axes, protein PCA, references and predictors are re-fit in every
inner fold without its future labels. This is internal development CV, not a
new independent confirmation. Outer validation/test destinations are not read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

import w15g_response_program_probe as probe
from w4_artifacts import Artifacts,sha256


def partition(cohort,displacement,training,heldout,c):
    select=lambda ix:{k:np.asarray(v)[ix] for k,v in cohort.items()}
    train=select(training);source=select(heldout);target=displacement[training]
    mean,axes,singular=probe.response_basis(target,c['programs'])
    emean=train['embedding'].mean(0)
    _,_,pc=np.linalg.svd(train['embedding']-emean,full_matrices=False)
    pc=pc[:min(c['protein_dimensions'],len(training)-1)]
    features=lambda g:np.c_[g['state'],g['U'],(g['embedding']-emean)@pc.T]
    vm=train['velocity'].mean(0);um=train['U'].mean(0);sm=train['state'].mean(0)
    def representation(g):
        v=(g['velocity']-vm)@axes.T
        return dict(direction=probe.unit(v),absolute_direction=probe.unit(g['velocity']@axes.T),
            U_direction=probe.unit((g['U']-um)@axes.T),S_direction=probe.unit((g['state']-sm)@axes.T),
            magnitude=np.pad(np.linalg.norm(v,axis=1,keepdims=True),((0,0),(0,len(axes)-1))))
    return dict(train=train,source=source,target=target,mean=mean,axes=axes,singular=singular,
        x=features(train),xv=features(source),features=representation(train),
        validation_features=representation(source))


def summarize(rows):
    summaries=[]
    for alpha in sorted(set(r['alpha'] for r in rows)):
        names=sorted(set(r['arm'] for r in rows))
        errors={name:np.mean([r['MSE'] for r in rows if r['alpha']==alpha and r['arm']==name]) for name in names}
        v=errors['velocity_direction'];base=errors['S_U']
        shuffled=np.mean([errors[name] for name in names if name.startswith('shuffled_direction')])
        pairs={}
        for row in rows:
            if row['alpha']==alpha:pairs.setdefault(row['KO'],{})[row['arm']]=row['MSE']
        summaries.append(dict(alpha=alpha,KO_count=len(pairs),S_U_MSE=float(base),velocity_direction_MSE=float(v),
            gain_vs_S_U=float(1-v/base),gain_vs_mean_shuffle=float(1-v/shuffled),
            gain_vs_U_program_direction=float(1-v/errors['U_program_direction']),
            positive_direction_correction_gain=float(1-errors['positive_direction_correction']/base),
            direction_KO_wins_vs_U=sum(r['velocity_direction']<r['S_U'] for r in pairs.values()),
            baseline_gain_vs_shared=float(1-base/errors['shared_response']),
            direction_gain_vs_shared=float(1-v/errors['shared_response'])))
    return summaries


def program(config_path,output):
    c=json.loads(Path(config_path).read_text())
    run=Artifacts(output,seed=c['cv_seed'],inputs=[Path(config_path),Path(__file__),
        Path(probe.__file__),Path(c['split']),Path(c['conditions'])])
    try:
        # Only the already-used training future is loaded by prepare; no new held-out future.
        original=probe.prepare(c);cohort=original['train'];target=original['target']
        indices=np.random.default_rng(c['cv_seed']).permutation(len(target))
        folds=np.array_split(indices,c['cv_folds']);prepared=[];frozen=[]
        for number,heldout in enumerate(folds):
            training=np.array(sorted(set(range(len(target)))-set(heldout.tolist())))
            p=partition(cohort,target,training,heldout,c)
            prediction,coefficients=probe.fit_predictions(p,c)
            prepared.append((number,heldout,p))
            for alpha,arms in prediction.items():
                for arm,pred in arms.items():
                    path=run.directory/f'fold{number}_alpha{alpha}_{arm}.npz'
                    np.savez_compressed(path,delta=pred,KO=p['source']['labels'])
                    frozen.append(dict(fold=number,alpha=float(alpha),arm=arm,path=str(path.resolve()),sha256=sha256(path)))
            np.savez_compressed(run.directory/f'fold{number}_axes.npz',axes=p['axes'],response_mean=p['mean'])
        run.json('frozen_predictions.json',dict(predictions=frozen,outer_validation_targets_read=False,test_targets_read=False))
        rows=[];alignments=[]
        for number,heldout,p in prepared:
            restored={}
            for r in [r for r in frozen if r['fold']==number]:
                if sha256(r['path'])!=r['sha256']:raise ValueError('Frozen CV prediction changed')
                restored.setdefault(format(r['alpha'],'g'),{})[r['arm']]=probe.read_npz(r['path'])['delta']
            future=dict(z=cohort['state'][heldout]+target[heldout],conditions=cohort['labels'][heldout])
            metrics,_,alignment=probe.evaluate(p,restored,future)
            rows.extend(dict(fold=number,**r) for r in metrics)
            alignments.append(dict(fold=number,**alignment))
        summary=summarize(rows)
        from w7_router_scaling import write_csv
        write_csv(run.directory/'metrics.csv',rows);write_csv(run.directory/'comparisons.csv',summary)
        write_csv(run.directory/'alignment.csv',alignments)
        result=dict(status='TRAINING_KO_RESPONSE_PROGRAM_CV_COMPLETE',KO_count=len(target),folds=c['cv_folds'],
            comparisons=summary,outer_validation_targets_read=False,test_targets_read=False,
            source_geometry='fixed_outer_training_source_transform',
            response_axes_and_predictors='refit_within_every_inner_fold',
            scope='internal_development_CV_not_independent_confirmation')
        run.json('summary.json',result);run.json('status.json',dict(status='complete'))
        (run.directory/'RESULTS.md').write_text('# Training-KO response-program CV\n\n'+json.dumps(result,indent=2)+'\n')
        return result
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configuration',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(program(args.configuration,args.output),indent=2))
