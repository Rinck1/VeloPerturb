"""Exploratory estimator repair; v1 forecasts/protocol remain immutable.

Regularize baseline U/S rates, subtract baseline fields, and bound forecast
duration. A fixed factorial grid and capacity-matched S/U baselines separate
estimator/centering/readout effects. No future target trains any predictor.
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from w4_artifacts import Artifacts, sha256
from w4_shorttime import read_metadata
from w4_streams import SourceTransform
from w5_scntseq_minutes import (INTERVALS, cosine, load_population, verify_config)


ROOT = Path(__file__).resolve().parents[2]
CODE = [Path(__file__).with_name(name) for name in (
    'w5_scntseq_repair.py','w5_scntseq_minutes.py','w4_streams.py','w4_shorttime.py','w4_artifacts.py')]
BASELINES = ('past_unit','state','state_history','U','state_U','new')


def regularized_gamma(pooled_s, pooled_u, prior=1.):
    s, u = np.asarray(pooled_s).reshape(-1), np.asarray(pooled_u).reshape(-1)
    if (s.shape != u.shape or not len(s) or not np.isfinite(s).all() or not np.isfinite(u).all()
            or (s < 0).any() or (u < 0).any() or not np.isfinite(prior) or prior <= 0
            or s.sum() <= 0 or u.sum() <= 0):
        raise ValueError('Positive source exposure and U information required')
    global_ratio = float(u.sum()/s.sum())
    gamma = (u+prior*global_ratio)/(s+prior)
    return gamma, dict(global_ratio=global_ratio, prior_spliced_units=float(prior),
        prior_only_gene_indices=np.flatnonzero(u == 0).tolist(),
        source_U_observed_gene_indices=np.flatnonzero(u > 0).tolist(),
        zero_gamma_genes=int(np.sum(gamma == 0)), rates_are_measured_physical_decay=False)


def effective_minutes(duration, calibration, kernel):
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(calibration) or calibration <= 0:
        raise ValueError('Positive finite time intervals required')
    if kernel == 'linear':
        return float(duration)
    if kernel == 'cap':
        return float(min(duration,calibration))
    if kernel == 'decay':
        return float(calibration*(-np.expm1(-duration/calibration))/(-np.expm1(-1.)))
    raise ValueError('Unknown forecast time kernel')


def fit_readout(features, rate, positive, ridge=1.):
    matrix = np.stack(features, axis=1).astype(float)
    rate = np.asarray(rate, dtype=float)
    if (matrix.shape[0] != len(rate) or matrix.shape[1] != len(positive)
            or not np.isfinite(matrix).all() or not np.isfinite(rate).all()
            or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError('Invalid fixed low-dimensional readout')
    scale = np.linalg.norm(matrix,axis=0)
    scale = np.where(scale > 1e-12,scale,1.)
    normalized = matrix/scale
    augmented = np.concatenate((normalized, np.sqrt(ridge)*np.eye(matrix.shape[1])),axis=0)
    target = np.r_[rate,np.zeros(matrix.shape[1])]
    lower = np.array([0. if p else -np.inf for p in positive])
    solution = lsq_linear(augmented,target,bounds=(lower,np.full(len(positive),np.inf)),tol=1e-12)
    if not solution.success:
        raise ArithmeticError('Readout solve failed')
    coefficient = solution.x/scale
    coefficient[np.abs(coefficient) < 1e-12] = 0.
    return dict(coefficients=coefficient.tolist(), source_column_norms=scale.tolist(),
        positive_columns=list(positive), ridge=float(ridge), parameter_count=len(positive))


def readout_rate(features, fitted):
    return np.stack(features,axis=1) @ np.asarray(fitted['coefficients'])


def sufficient_features(transform, layers):
    s, u = layers['spliced'], layers['unspliced']
    sn, un = transform.normalized(s,u)
    ps, pu = transform.normalized(s.sum(0),u.sum(0))
    total = float(layers['total'].sum())
    if total <= 0:
        raise ValueError('Empty labeling library')
    new = np.asarray(layers['new'].sum(0)).ravel()[transform.genes]*(transform.target_sum/total)
    return dict(state=transform.state_normalized(sn).mean(0),
        mean_u=np.mean(un/(1+sn),axis=0), mean_s=np.mean(sn/(1+sn),axis=0),
        pool_u=(pu/(1+ps))[0], pool_s=(ps/(1+ps))[0],
        normalized_pool_s=ps[0], normalized_pool_u=pu[0],
        U=np.log1p(pu)[0] @ transform.components.T, new=np.log1p(new) @ transform.components.T)


def velocity_feature(source, baseline, gamma, components, aggregation, centering):
    def field(population):
        return (population[f'{aggregation}_u']-gamma*population[f'{aggregation}_s']) @ components.T
    velocity = field(source)
    if centering == 'baseline0':
        velocity = velocity-field(baseline)
    elif centering != 'raw':
        raise ValueError('Unknown field centering')
    return velocity


def model_specs(config):
    specs = {'identity':dict(kind='identity')}
    for kind,kernel in product(BASELINES,config['kernels']):
        specs[f'{kind}__{kernel}'] = dict(kind=kind,kernel=kernel)
    for estimator,aggregation,centering,model,kernel in product(config['estimators'],config['aggregations'],
            config['centerings'],config['models'],config['kernels']):
        name = '__'.join((estimator,aggregation,centering,model,kernel))
        specs[name] = dict(kind=model,estimator=estimator,aggregation=aggregation,centering=centering,kernel=kernel)
    return specs


def design(spec, populations, minute, previous, gammas, transform):
    population, baseline = populations[minute], populations[0]
    offset = population['state']-baseline['state']
    past = (population['state']-populations[previous]['state'])/(minute-previous)
    u, new = population['U']-baseline['U'], population['new']-baseline['new']
    kind = spec['kind']
    if kind == 'identity':
        return [], []
    if kind == 'past_unit':
        return [past], [False]
    controls = dict(state=([offset],[False]), state_history=([offset,past],[False,False]),
        U=([u],[False]), state_U=([offset,u],[False,False]), new=([new],[False]))
    if kind in controls:
        return controls[kind]
    velocity = velocity_feature(population,baseline,gammas[spec['estimator']],transform.components,
        spec['aggregation'],spec['centering'])
    if kind == 'velocity':
        return [velocity], [True]
    if kind == 'state_velocity':
        return [offset,velocity], [False,True]
    raise ValueError('Unknown forecast model')


def setup(config_path, *, repo=ROOT):
    config = json.loads(Path(config_path).read_text())
    if (config['kind'] != 'exploratory_scntseq_estimator_repair' or not config['previously_scored_series']
            or config['router_enabled'] or config['gfg_jvp_enabled'] or config['original_results_overwritten']
            or config['calibration_interval'] != [15,30] or config['test_intervals'] != [list(x) for x in INTERVALS]
            or config['estimators'] != ['tail','pooled_shrink'] or config['aggregations'] != ['mean','pool']
            or config['centerings'] != ['raw','baseline0'] or config['kernels'] != ['linear','cap','decay']
            or config['models'] != ['velocity','state_velocity']):
        raise ValueError('Unexpected repair grid')
    original = Path(repo)/config['original_predictions']
    manifest = json.loads((original/'prediction_manifest.json').read_text())
    original_config, _, data = verify_config(manifest['config_path'],manifest['freeze_commit'],repo=repo)
    if str(data) != config['data_path'] or sha256(data) != config['data_sha256']:
        raise ValueError('Use the same unchanged paired release')
    transform_path = original/'source_transform.npz'
    if sha256(transform_path) != manifest['transform_sha256']:
        raise ValueError('Original source geometry changed')
    return config,original_config,data,transform_path


def predict(config_path,output, *, repo=ROOT):
    config,original_config,data,transform_path = setup(config_path,repo=repo)
    inputs = [Path(config_path),data,transform_path,Path(repo)/'PLANS/w5_scntseq_repair_v2.md']+CODE
    run = Artifacts(output,seed=config['seed'],inputs=inputs)
    transform = SourceTransform.load(transform_path)
    obs,genes = read_metadata(data)
    np.testing.assert_array_equal(genes,transform.gene_names)
    populations,cell_counts = {},{}
    for minute in (0,15,30):
        layers,ids = load_population(data,obs,minute,original_config,source=True)
        populations[minute] = sufficient_features(transform,layers); cell_counts[str(minute)] = len(ids)
        if minute == 0:
            np.testing.assert_array_equal(ids,transform.source_ids)
    gamma,details = regularized_gamma(populations[0]['normalized_pool_s'],populations[0]['normalized_pool_u'],
        config['prior_spliced_units'])
    gammas = dict(tail=transform.gamma.copy(),pooled_shrink=gamma)
    np.savez_compressed(run.directory/'gamma_estimates.npz',genes=transform.gene_names[transform.genes],**gammas)
    run.json('estimator.json',dict(old_zero_gamma_genes=int((transform.gamma == 0).sum()),
        repaired_zero_gamma_genes=int((gamma == 0).sum()),source_fit_minutes=[0],**details))
    specs = model_specs(config)
    if config['primary_candidate'] not in specs or any(x not in specs for x in config['primary_comparators']):
        raise ValueError('Primary candidate/comparators outside fixed grid')
    names = list(specs)
    rate = (populations[30]['state']-populations[15]['state'])/15
    fits = {}
    for name,spec in specs.items():
        features,positive = design(spec,populations,15,0,gammas,transform)
        if spec['kind'] == 'identity':
            fits[name] = dict(coefficients=[],parameter_count=0)
        elif spec['kind'] == 'past_unit':
            fits[name] = dict(coefficients=[1.],parameter_count=0)
        else:
            fits[name] = fit_readout(features,rate,positive,config['readout_ridge'])
    run.json('calibration.json',dict(interval=[15,30],models=specs,fits=fits,
        calibration_uses_future_test_targets=False,all_models_reported=True))
    predictions,states = [],[]
    for start,stop,previous in ((30,60,15),(60,120,30)):
        if start == 60:
            layers,ids = load_population(data,obs,60,original_config,source=True)
            populations[60] = sufficient_features(transform,layers); cell_counts['60'] = len(ids)
        mu = populations[start]['state']
        values = []
        for name,spec in specs.items():
            features,_ = design(spec,populations,start,previous,gammas,transform)
            if spec['kind'] == 'identity':
                values.append(mu.copy())
            else:
                duration = effective_minutes(stop-start,15,spec['kernel'])
                values.append(mu+duration*readout_rate(features,fits[name]))
        values = np.stack(values)
        path = run.directory/f'prediction_{start}_to_{stop}.npz'
        np.savez_compressed(path,arms=np.array(names),predictions=values,source_state=mu)
        run.json(f'prediction_{start}_to_{stop}_manifest.json',dict(status='INTERVAL_PREDICTION_FROZEN',
            future_test_target_used=False,expression_minutes_read=[0,15,30] if start == 30 else [0,15,30,60],
            prediction_sha256=sha256(path)))
        predictions.append(values);states.append(mu)
    np.savez_compressed(run.directory/'predictions.npz',arms=np.array(names),intervals=np.array(INTERVALS),
        predictions=np.stack(predictions),source_states=np.stack(states))
    run.json('prediction_manifest.json',dict(status='REPAIR_PREDICTIONS_FROZEN',kind=config['kind'],
        config_path=str(Path(config_path).resolve()),config_sha256=sha256(config_path),
        transform_path=str(transform_path),transform_sha256=sha256(transform_path),
        prediction_sha256=sha256(run.directory/'predictions.npz'),
        calibration_sha256=sha256(run.directory/'calibration.json'), source_cells=cell_counts,arms=names,
        primary_candidate=config['primary_candidate'],future_test_targets_used_in_fitting=False,
        original_results_changed=False,independent_confirmation=False,
        code_hashes={str(p.resolve()):sha256(p) for p in CODE}))
    run.json('status.json',dict(status='complete'))
    print(json.dumps(dict(status='REPAIR_PREDICTIONS_FROZEN',models=len(names),output=str(output))),flush=True)


def evaluate(directory,output, *, repo=ROOT):
    directory = Path(directory)
    manifest = json.loads((directory/'prediction_manifest.json').read_text())
    config_path = Path(manifest['config_path'])
    if (manifest['status'] != 'REPAIR_PREDICTIONS_FROZEN' or manifest['future_test_targets_used_in_fitting']
            or sha256(config_path) != manifest['config_sha256']):
        raise ValueError('Changed repair prediction contract')
    config,original_config,data,transform_path = setup(config_path,repo=repo)
    for filename,key in (('predictions.npz','prediction_sha256'),('calibration.json','calibration_sha256')):
        if sha256(directory/filename) != manifest[key]:
            raise ValueError('Repair artifact changed')
    if manifest['code_hashes'] != {str(p.resolve()):sha256(p) for p in CODE}:
        raise ValueError('Repair code changed')
    run = Artifacts(output,seed=config['seed'],inputs=[config_path,data,transform_path,
        directory/'prediction_manifest.json',directory/'predictions.npz',directory/'calibration.json']+CODE)
    transform = SourceTransform.load(transform_path)
    obs,genes = read_metadata(data);np.testing.assert_array_equal(genes,transform.gene_names)
    with np.load(directory/'predictions.npz',allow_pickle=False) as saved:
        names = saved['arms'].tolist();pred = saved['predictions'];states = saved['source_states']
        np.testing.assert_array_equal(saved['intervals'],INTERVALS)
    errors,truth,rows = np.empty((2,len(names))),[],[]
    for i,(start,stop) in enumerate(INTERVALS):
        layers,ids = load_population(data,obs,stop,original_config,source=False)
        target = transform.state(layers['spliced']).mean(0);truth.append(target)
        errors[i] = np.square(pred[i]-target).sum(1)
        base = errors[i,names.index('identity')]
        for j,name in enumerate(names):
            rows.append(dict(source_minute=start,target_minute=stop,arm=name,
                mse=float(errors[i,j]/len(target)),gain_over_identity=float((base-errors[i,j])/max(base,1e-12)),
                displacement_cosine=cosine(pred[i,j]-states[i],target-states[i])))
    candidate = names.index(config['primary_candidate'])
    identity = names.index('identity')
    gains = (errors[:,identity,None]-errors)/np.maximum(errors[:,identity,None],1e-12)
    checks = {name:bool(errors[:,candidate].mean() < errors[:,names.index(name)].mean())
              for name in config['primary_comparators']}
    result = dict(status='EXPLORATORY_REPAIR_EVALUATED',exploratory=True,independent_confirmation=False,
        original_results_changed=False,primary_candidate=config['primary_candidate'],
        primary_gain_by_interval=gains[:,candidate].tolist(),primary_mean_mse=float(errors[:,candidate].mean()/transform.components.shape[0]),
        primary_beats_identity_both=bool((gains[:,candidate] > 0).all()),primary_mean_error_checks=checks,
        primary_repair_rule_met=bool((gains[:,candidate] > 0).all()) and all(checks.values()),
        mean_mse={name:float(errors[:,j].mean()/transform.components.shape[0]) for j,name in enumerate(names)},
        mean_interval_gain={name:float(gains[:,j].mean()) for j,name in enumerate(names)},
        compared_models=len(names),test_intervals=[list(x) for x in INTERVALS],
        biological_significance_test=False,router_enabled=False,gfg_jvp_enabled=False)
    with (run.directory/'metrics.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(run.directory/'targets_and_errors.npz',arms=np.array(names),truth=np.stack(truth),errors=errors)
    run.json('summary.json',result)
    (run.directory/'RESULTS.md').write_text('# scNT-seq estimator repair v2\n\n'
        'All arms are exploratory on the already scored series; no v1 score is changed. '
        'The primary candidate was fixed before this grid run.\n\n'+json.dumps(result,indent=2)+'\n')
    run.json('status.json',dict(status='complete'))
    print(json.dumps({k:v for k,v in result.items() if k not in ('mean_mse','mean_interval_gain')}),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='stage',required=True)
    p = sub.add_parser('predict');p.add_argument('--config',default='configs/w5_scntseq_repair_v2.json');p.add_argument('--output',required=True)
    e = sub.add_parser('evaluate');e.add_argument('--predictions',required=True);e.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.stage == 'predict':predict(args.config,args.output)
    else:evaluate(args.predictions,args.output)
