"""Train-only multimodal definitions and donor-level frozen-grid evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .artifacts import Run, object_hash, save_csv, save_json, sha256
from .latent import load_pack
from .multimodal import density_support


def define_modes(config,donor,output):
    fold=Path(config['root'])/'folds'/donor
    source,sm=load_pack(fold/'train_source.npz',expected_side='source')
    target,tm=load_pack(fold/'train_target.npz',expected_side='target')
    if sm['role']!='train' or tm['role']!='train':raise ValueError('Mode definitions require training donors only')
    c=config['multimodality'];rng=np.random.default_rng(c['seed'])
    types=sorted({x.split('|',1)[1] for x in target['conditions']})
    definitions={};rows=[]
    with Run(output,stage='Kang_training_within_type_multimodal_definitions',kind='engineering',
             config=dict(**c,heldout=donor),inputs=[fold/'train_source.npz',fold/'train_target.npz'],seed=c['seed']) as run:
        for cell_type in types:
            groups=sorted(g for g in set(source['conditions'])&set(target['conditions']) if g.split('|',1)[1]==cell_type)
            ys=[target['z'][target['conditions']==g] for g in groups]
            xs=[source['z'][source['conditions']==g] for g in groups]
            # Equal donor weighting for the effect axis; never mix cell types.
            effect=np.mean([y.mean(0)-x.mean(0) for x,y in zip(xs,ys)],0)
            effect=effect/max(np.linalg.norm(effect),1e-8)
            pooled=np.concatenate(ys);pc=PCA(2).fit(pooled).components_
            vectors=dict(effect_direction=effect,local_pc1=pc[0],local_pc2=pc[1])
            supports={name:[] for name in c['axes']}
            for group,y in zip(groups,ys):
                for name in c['axes']:
                    votes=[]
                    if len(y)//2>=c['minimum_cells_per_half']:
                        for split in range(c['splits']):
                            order=rng.permutation(len(y));a,b=np.array_split(order,2)
                            test=density_support(y[a]@vectors[name],y[b]@vectors[name],c,c['seed']+split)
                            votes.append(test['supported'])
                            rows.append(dict(cell_type=cell_type,group=group,axis=name,split=split,**test))
                    stability=float(np.mean(votes)) if votes else 0.
                    supports[name].append(stability>=c['minimum_split_support'])
            rates={name:float(np.mean(v)) for name,v in supports.items()}
            best=max(c['axes'],key=lambda name:rates[name])
            approved=(rates[best]>=c['minimum_training_donor_support_fraction'] and sum(supports[best])>=4)
            projection=pooled@vectors[best]
            mixture=GaussianMixture(2,n_init=5,reg_covar=1e-5,random_state=c['seed']).fit(projection[:,None])
            order=np.argsort(mixture.means_.ravel())
            definitions[cell_type]=dict(approved=bool(approved),axis_name=best,axis=vectors[best].tolist(),
                means=mixture.means_.ravel()[order].tolist(),variances=mixture.covariances_.ravel()[order].tolist(),
                weights=mixture.weights_[order].tolist(),donor_support=rates[best],
                supporting_donors=[g.split('|',1)[0] for g,v in zip(groups,supports[best]) if v],
                training_groups=groups,axis_selection_rates=rates,
                scope='training_defined_response_components_not_individual_fates')
        nulls=[]
        for seed in range(c['null_simulations']):
            x=rng.normal(size=200)
            nulls.append(dict(simulation=seed,**density_support(x[:100],x[100:],c,c['seed']+seed)))
        save_json(run.directory/'mode_definitions.json',definitions)
        save_csv(run.directory/'axis_split_metrics.csv',rows,fields=list(rows[0]) if rows else ['cell_type','group','axis','split','supported'])
        save_csv(run.directory/'gaussian_sanity_checks.csv',nulls)
        result=dict(status='KANG_TRAINING_MODE_SCREEN_COMPLETE',heldout_donor=donor,
            supported_types=[ct for ct,d in definitions.items() if d['approved']],
            formal_FDR_D4=False,heldout_targets_read=False,gaussian_null_positive_fraction=float(np.mean([r['supported'] for r in nulls])))
        save_json(run.directory/'summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Kang within-type multimodality screen\n\n'+json.dumps(result,indent=2)
            +'\n\nExploratory replicated density screen, not an omnibus or FDR-calibrated test. Axes and modes use training donors only.\n')
    return result


def weighted_energy(x,weights,y,target_weights=None,block=256):
    x,y=np.asarray(x,dtype='float64'),np.asarray(y,dtype='float64')
    weights=np.asarray(weights,dtype='float64')
    target_weights=np.ones(len(y)) if target_weights is None else np.asarray(target_weights,dtype='float64')
    if not len(x) or not len(y) or (weights<0).any() or (target_weights<0).any() or weights.sum()<=0 or target_weights.sum()<=0:
        raise ValueError('Invalid weighted empirical distributions')
    weights=weights/weights.sum();target_weights=target_weights/target_weights.sum()
    def term(a,wa,b,wb):
        result=0.
        for i in range(0,len(a),block):
            for j in range(0,len(b),block):
                result+=wa[i:i+block]@cdist(a[i:i+block],b[j:j+block])@wb[j:j+block]
        return float(result)
    result=2*term(x,weights,y,target_weights)-term(x,weights,x,weights)-term(y,target_weights,y,target_weights)
    if result < -1e-7:raise ArithmeticError('Negative energy outside rounding tolerance')
    return max(0.,result)


def mode_posterior(x,definition):
    projection=np.asarray(x)@np.array(definition['axis'])
    means,var,weights=[np.asarray(definition[k]) for k in ('means','variances','weights')]
    logits=np.log(weights)[None]-.5*(np.log(2*np.pi*var)[None]+(projection[:,None]-means[None])**2/var[None])
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))


def donor_bootstrap(rows,config,control,metric,*,multimodal_only=False,real_arm='gfg_joint_router'):
    real=real_arm;diff=[];donors=[]
    primary=config['cell_types']['primary']
    for donor in config['donors']:
        selected=[r for r in rows if r['donor']==donor and r['cell_type']==primary]
        if multimodal_only:selected=[r for r in selected if r['mode_approved'] and r['adequate_test_count']]
        values=[]
        for seed in config['experiments']['seeds']:
            pair={r['arm']:r[metric] for r in selected if r['seed']==seed and r['arm'] in (real,control)}
            if set(pair)!={real,control} or any(v is None for v in pair.values()):break
            values.append(pair[control]-pair[real])
        if len(values)==len(config['experiments']['seeds']):donors.append(donor);diff.append(float(np.mean(values)))
    result=dict(control=control,metric=metric,scope='multimodal_primary' if multimodal_only else 'all_primary',
                n_donors=len(diff),donors=donors,gain=float(np.mean(diff)) if diff else None,
                confidence=config['experiments']['primary_family_confidence'],ci_low=None,ci_high=None,
                inference_unit='donor_after_seed_averaging',adequate_donor_support=len(diff)>=5)
    if len(diff)>=2:
        rng=np.random.default_rng(config['multimodality']['seed']);x=np.array(diff)
        b=x[rng.integers(len(x),size=(config['experiments']['bootstrap_repetitions'],len(x)))].mean(1)
        a=(1-result['confidence'])/2;lo,hi=np.quantile(b,[a,1-a]);result.update(ci_low=float(lo),ci_high=float(hi))
    return result


def all_type_noninferiority(rows,config,*,real_arm='gfg_joint_router',static_arm='static_router'):
    """Equal donor and equal type weights; positive error excess is worse."""
    values=[];margin=config['experiments']['all_type_noninferiority_margin_relative']
    for donor in config['donors']:
        seed_values=[]
        for seed in config['experiments']['seeds']:
            group=[r for r in rows if r['donor']==donor and r['seed']==seed]
            by_type={}
            for r in group:
                if r['arm'] in (static_arm,real_arm):
                    by_type.setdefault(r['cell_type'],{})[r['arm']]=r['energy_distance']
            if not by_type or any(set(v)!={static_arm,real_arm} for v in by_type.values()):
                raise ValueError('Incomplete all-type noninferiority pairing')
            seed_values.append(np.mean([(v[real_arm]-(1+margin)*v[static_arm])/max(v[static_arm],1e-8)
                                        for v in by_type.values()]))
        values.append(float(np.mean(seed_values)))
    rng=np.random.default_rng(config['multimodality']['seed']);x=np.array(values)
    b=x[rng.integers(len(x),size=(config['experiments']['bootstrap_repetitions'],len(x)))].mean(1)
    upper=float(np.quantile(b,config['experiments']['primary_family_confidence']))
    return dict(relative_margin=margin,mean_excess=float(x.mean()),one_sided_upper=upper,noninferior=upper<0,
                n_donors=len(values),resampling='donor_after_equal_type_and_seed_averaging')


def evaluate_grid(config,output,*,experiment_group='experiments',experiment_root=None,registered_arms=None,real_arm='gfg_joint_router',
                  primary_controls=('static_router','gfg_joint_shuffled_U')):
    root=Path(config['root']);experiment_root=Path(experiment_root) if experiment_root else root/experiment_group
    arms=registered_arms or config['experiments']['neural_arms']+config['experiments']['simple_baselines']
    frozen=[]
    # First freeze the WHOLE fixed primary grid; no held-out target is opened above this boundary.
    for donor in config['donors']:
        for seed in config['experiments']['seeds']:
            for arm in arms:
                selection=json.loads((experiment_root/donor/f'seed{seed}'/arm/'selection.json').read_text())
                path=Path(selection['prediction']);manifest=json.loads((path/'prediction_manifest.json').read_text())
                provenance=json.loads((path/'provenance.json').read_text())
                if provenance['status']!='complete' or manifest['future_target_read'] or manifest['protocol_hash']!=object_hash(config):
                    raise ValueError('Entire registered prediction grid must be complete, source-only, and protocol-bound')
                if (manifest['donor'],manifest['seed'],manifest['arm'])!=(donor,seed,arm):raise ValueError('Prediction identity mismatch')
                digest=sha256(path/'predictions.npz')
                if digest!=manifest['prediction_sha256'] or digest!=selection['prediction_sha256'] or digest!=provenance['outputs']['predictions.npz']:
                    raise ValueError('Prediction artifact changed after freezing')
                frozen.append(dict(donor=donor,seed=seed,arm=arm,path=str(path),sha256=digest))
    with Run(output,stage='Kang_frozen_grid_donor_evaluation',kind='development',config=config,
             inputs=[Path(r['path'])/'prediction_manifest.json' for r in frozen],seed=config['experiments']['seeds']) as run:
        save_json(run.directory/'frozen_grid.json',frozen)
        rows=[]
        for donor in config['donors']:
            fold=root/'folds'/donor
            source,sm=load_pack(fold/'validation_source.npz',expected_side='source')
            target,tm=load_pack(fold/'validation_target.npz',expected_side='target')
            if tm['heldout_donor']!=donor or tm['role_semantics']!='outer_heldout' or tm['transform_hash']!=sm['transform_hash']:
                raise ValueError('Held-out target contract mismatch')
            definitions=json.loads((root/'modes'/donor/'mode_definitions.json').read_text())
            modeprov=json.loads((root/'modes'/donor/'provenance.json').read_text())
            if modeprov['status']!='complete' or sha256(root/'modes'/donor/'mode_definitions.json')!=modeprov['outputs']['mode_definitions.json']:
                raise ValueError('Training mode definitions were modified')
            geometry={}
            for record in [r for r in frozen if r['donor']==donor]:
                with np.load(Path(record['path'])/'predictions.npz',allow_pickle=False) as f:p={k:f[k] for k in f.files}
                if not np.array_equal(p['source_ids'],source['cell_ids']) or not np.array_equal(p['conditions'],source['conditions']):
                    raise ValueError('Prediction/source sample keys changed')
                if record['arm'] in ('static_router','gfg_joint_router','gfg_joint_shuffled_U','gfg_frozen_router','raw_SU_router'):
                    if record['seed'] in geometry:np.testing.assert_array_equal(geometry[record['seed']],p['candidates'])
                    else:geometry[record['seed']]=p['candidates']
                for label in sorted(set(source['conditions'])):
                    ct=label.split('|',1)[1];pi=p['conditions']==label;ti=target['conditions']==label
                    y=target['z'][ti];candidates=p['candidates'][pi];q=p['q'][pi]
                    if not len(y):raise ValueError('Registered source cohort has no evaluable target group')
                    x=candidates.reshape(-1,candidates.shape[-1]);w=q.reshape(-1).astype('float64');w/=w.sum()
                    mean=w@x;variance=float(w@(x*x).sum(1)-mean@mean)
                    ratio=variance/max(float(y.var(0).sum()),1e-12)
                    definition=definitions[ct]
                    row=dict(donor=donor,seed=record['seed'],arm=record['arm'],cell_type=ct,
                        source_cells=int(pi.sum()),target_cells=len(y),energy_distance=weighted_energy(x,w,y),
                        mean_mse=float(((mean-y.mean(0))**2).mean()),log_variance_error=abs(float(np.log(max(ratio,1e-12)))),
                        mode_approved=definition['approved'],mode_proportion_error=None,within_mode_energy=None,
                        predicted_high_mode=None,target_high_mode=None,
                        adequate_test_count=min(int(pi.sum()),len(y))>=config['cell_types']['minimum_cells_per_donor_side'])
                    if definition['approved']:
                        px,py=mode_posterior(x,definition),mode_posterior(y,definition)
                        predicted=w@px;truth=py.mean(0)
                        row.update(mode_proportion_error=float(np.abs(predicted-truth).mean()),
                            predicted_high_mode=float(predicted[1]),target_high_mode=float(truth[1]))
                        if min(predicted)>1e-8 and min(truth)>1e-8:
                            row['within_mode_energy']=float(np.mean([weighted_energy(x,w*px[:,k],y,py[:,k]) for k in range(2)]))
                    rows.append(row)
        comparisons=[donor_bootstrap(rows,config,control,metric,multimodal_only=True,real_arm=real_arm)
            for control in primary_controls for metric in config['experiments']['primary_metrics']]
        overall=[donor_bootstrap(rows,config,control,'energy_distance',real_arm=real_arm) for control in arms if control!=real_arm]
        success=all(r['adequate_donor_support'] and r['ci_low'] is not None and r['ci_low']>0 for r in comparisons)
        noninferiority=all_type_noninferiority(rows,config,real_arm=real_arm,static_arm=primary_controls[0])
        result=dict(status='KANG_PRIMARY_GRID_EVALUATED',primary_comparisons=comparisons,all_primary_donors=overall,
            experiment_group=experiment_group,real_arm=real_arm,
            velocity_multimodal_increment='SUPPORTED_IN_THIS_PROTOCOL' if success else 'NOT_ESTABLISHED',
            causal_fate_claim=False,perturbation_generalization=False,celltype_noninferiority=noninferiority,
            combined_multimodal_and_noninferiority_claim=success and noninferiority['noninferior'],
            external_baselines='NOT_INCLUDED_UNTIL_ACTUAL_ADAPTER_RESULTS_EXIST',frozen_predictions=len(frozen))
        save_csv(run.directory/'metrics.csv',rows);save_csv(run.directory/'comparisons.csv',comparisons+overall)
        save_json(run.directory/'summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Kang primary grid\n\n'+json.dumps(result,indent=2)
            +'\n\nEnergy is the exact weighted empirical-mixture V-statistic, not only a sampled endpoint score. '
             'Mode proportions use training-defined expression response components, not observed lineage fates. '
             'Eight donors, not cells, define the independent resampling units.\n')
        # Supplementary definitions requested by the user. Does not change the
        # frozen primary success rule, predictions, model, or training schedule.
        from .kang_shared_metrics import evaluate_shared_metrics
        evaluate_shared_metrics(config,run.directory/'cellflow_metrics',frozen)
    return result
