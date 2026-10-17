"""Fixed exploratory U-total/composition factorial; unchanged v2 geometry/readout.

Equal-cell pooling removes S-library weighting, not every technical/biological
confound. All forecasts retain the same original S state and target estimand.
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from w4_artifacts import Artifacts, sha256
from w4_shorttime import read_metadata
from w4_streams import SourceTransform
from w5_scntseq_minutes import INTERVALS, load_population
from w5_scntseq_repair import effective_minutes, fit_readout, readout_rate, setup


ROOT = Path(__file__).resolve().parents[2]
CODE = [Path(__file__).with_name(name) for name in (
    'w5_scntseq_U_attribution.py', 'w5_scntseq_repair.py', 'w5_scntseq_minutes.py',
    'w4_streams.py', 'w4_shorttime.py', 'w4_artifacts.py')]


def pooled_statistics(transform, spliced, unspliced):
    """Both poolings use all-gene S exposure and retain the U/S ratio."""
    depth_s = np.asarray(spliced.sum(1)).ravel().astype(float)
    depth_u = np.asarray(unspliced.sum(1)).ravel().astype(float)
    if (depth_s <= 0).any() or not np.isfinite(depth_s).all() or (depth_u < 0).any():
        raise ValueError('Positive finite S libraries required')
    sn, un = transform.normalized(spliced, unspliced)
    ps, pu = transform.normalized(spliced.sum(0), unspliced.sum(0))
    values = dict(
        depth_weighted=dict(S=ps[0], U=pu[0], ratio=float(depth_u.sum()/depth_s.sum())),
        equal_cell=dict(S=sn.mean(0), U=un.mean(0), ratio=float(np.mean(depth_u/depth_s))))
    if any(v['ratio'] <= 0 for v in values.values()):
        raise ValueError('Positive U exposure required')
    return values, transform.state_normalized(sn).mean(0)


def counterfactual_u(source, baseline, composition, total):
    """Composition normalized over ALL genes, even though stored U is HVG-only."""
    if composition not in ('current', 'baseline') or total not in ('current', 'baseline'):
        raise ValueError('Unknown factorial setting')
    origin = source if composition == 'current' else baseline
    ratio = source['ratio'] if total == 'current' else baseline['ratio']
    if origin['ratio'] <= 0 or ratio <= 0:
        raise ValueError('Positive U/S required')
    return origin['U']*(ratio/origin['ratio'])


def field_components(source, baseline, gamma, rotation, composition, total):
    u = counterfactual_u(source, baseline, composition, total)
    derivative = (u/(1+source['S'])-baseline['U']/(1+baseline['S'])) @ rotation.T
    kinetic = (-gamma*(source['S']/(1+source['S'])-baseline['S']/(1+baseline['S']))) @ rotation.T
    return dict(U_derivative=derivative, S_kinetic=kinetic, velocity=derivative+kinetic)


def model_specs(config):
    specs = {}
    for aggregation, composition, total, field in product(config['aggregations'], config['compositions'],
            config['totals'], config['fields']):
        name = f'{aggregation}__composition_{composition}__total_{total}__{field}'
        specs[name] = dict(aggregation=aggregation, composition=composition, total=total, field=field)
    for aggregation in config['aggregations']:
        specs[f'{aggregation}__S_kinetic'] = dict(aggregation=aggregation, composition='baseline',
            total='baseline', field='S_kinetic')
    return specs


def design(spec, populations, minute, gamma, rotation):
    source, baseline = populations[minute], populations[0]
    field = field_components(source['pools'][spec['aggregation']], baseline['pools'][spec['aggregation']],
        gamma, rotation, spec['composition'], spec['total'])[spec['field']]
    return [source['state']-baseline['state'], field], [False, spec['field'] == 'velocity']


def verify(path, *, repo=ROOT):
    config = json.loads(Path(path).read_text())
    expected = dict(kind='exploratory_scntseq_U_factorial_attribution',
        aggregations=['depth_weighted', 'equal_cell'], compositions=['current', 'baseline'],
        totals=['current', 'baseline'], fields=['velocity', 'U_derivative'],
        calibration_interval=[15,30], test_intervals=[list(x) for x in INTERVALS],
        readout_ridge=1., time_kernel='decay', previously_scored_series=True,
        router_enabled=False, gfg_jvp_enabled=False)
    if any(config.get(k) != v for k,v in expected.items()):
        raise ValueError('Unexpected attribution design')
    repaired = Path(repo)/config['repair_predictions']
    manifest = json.loads((repaired/'prediction_manifest.json').read_text())
    for filename,key in (('predictions.npz','prediction_sha256'), ('calibration.json','calibration_sha256')):
        if sha256(repaired/filename) != manifest[key]:
            raise ValueError('Original repair changed')
    if sha256(manifest['config_path']) != manifest['config_sha256']:
        raise ValueError('Original repair configuration changed')
    repair, original, data, transform = setup(manifest['config_path'], repo=repo)
    if sha256(transform) != manifest['transform_sha256']:
        raise ValueError('Original geometry changed')
    return config, repaired, repair, original, data, transform


def predict(config_path, output, *, repo=ROOT):
    config,repaired,repair,original,data,transform_path = verify(config_path,repo=repo)
    run = Artifacts(output,seed=config['seed'],inputs=[Path(config_path),Path(repo)/config['protocol_path'],
        data,transform_path,repaired/'prediction_manifest.json',repaired/'gamma_estimates.npz']+CODE)
    transform = SourceTransform.load(transform_path)
    obs,genes = read_metadata(data); np.testing.assert_array_equal(genes,transform.gene_names)
    with np.load(repaired/'gamma_estimates.npz',allow_pickle=False) as saved:
        gamma = saved['pooled_shrink'].copy()
    populations, diagnostic_rows = {}, []
    def load(minute):
        layers,ids = load_population(data,obs,minute,original,source=True)
        pools,state = pooled_statistics(transform,layers['spliced'],layers['unspliced'])
        s = np.asarray(layers['spliced'].sum(1)).ravel().astype(float)
        u = np.asarray(layers['unspliced'].sum(1)).ravel().astype(float)
        correlation = float(spearmanr(s,u/s).statistic) if np.ptp(s) > 0 and np.ptp(u/s) > 0 else None
        diagnostic_rows.append(dict(minute=minute,cells=len(ids),median_S=float(np.median(s)),
            median_U=float(np.median(u)),pooled_U_to_S=pools['depth_weighted']['ratio'],
            equal_cell_U_to_S=pools['equal_cell']['ratio'],median_cell_U_to_S=float(np.median(u/s)),
            spearman_S_depth_vs_cell_U_to_S=correlation))
        return dict(pools=pools,state=state)
    for minute in (0,15,30): populations[minute] = load(minute)
    specs = model_specs(config)
    rate = (populations[30]['state']-populations[15]['state'])/15
    fits = {}
    for name,spec in specs.items():
        columns,positive = design(spec,populations,15,gamma,transform.components)
        fits[name] = fit_readout(columns,rate,positive,ridge=config['readout_ridge'])
    run.json('calibration.json',dict(interval=[15,30],models=specs,fits=fits))
    with np.load(repaired/'predictions.npz',allow_pickle=False) as saved:
        candidate = saved['predictions'][:,saved['arms'].tolist().index(repair['primary_candidate'])].copy()
    names = ['identity','v2_candidate']+list(specs)
    predictions,states = [],[]
    for i,(start,stop) in enumerate(INTERVALS):
        if start == 60: populations[60] = load(60)
        state = populations[start]['state']
        duration = effective_minutes(stop-start,15,config['time_kernel'])
        values = [state.copy(), candidate[i]]
        for name,spec in specs.items():
            columns,_ = design(spec,populations,start,gamma,transform.components)
            values.append(state+duration*readout_rate(columns,fits[name]))
        values = np.stack(values)
        # Algebraic real/real pooling arm must reproduce the fixed v2 candidate.
        actual = names.index('depth_weighted__composition_current__total_current__velocity')
        np.testing.assert_allclose(values[actual],candidate[i],atol=1e-10,rtol=1e-10)
        part = run.directory/f'prediction_{start}_to_{stop}.npz'
        np.savez_compressed(part,arms=np.array(names),predictions=values,source_state=state)
        run.json(f'prediction_{start}_to_{stop}_manifest.json',dict(status='INTERVAL_PREDICTION_FROZEN',
            future_target_used=False,prediction_sha256=sha256(part)))
        predictions.append(values); states.append(state)
    np.savez_compressed(run.directory/'predictions.npz',arms=np.array(names),intervals=np.array(INTERVALS),
        predictions=np.stack(predictions),source_states=np.stack(states))
    with (run.directory/'source_depth_diagnostics.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(diagnostic_rows[0]));writer.writeheader();writer.writerows(diagnostic_rows)
    run.json('prediction_manifest.json',dict(status='U_ATTRIBUTION_PREDICTIONS_FROZEN',
        config_path=str(Path(config_path).resolve()),config_sha256=sha256(config_path),
        prediction_sha256=sha256(run.directory/'predictions.npz'),
        calibration_sha256=sha256(run.directory/'calibration.json'),code_hashes={p.name:sha256(p) for p in CODE},
        future_targets_used_in_fitting=False,arms=names,original_results_changed=False))
    run.json('status.json',dict(status='complete'))
    print(json.dumps(dict(status='U_ATTRIBUTION_PREDICTIONS_FROZEN',arms=len(names))),flush=True)


def evaluate(predictions,output, *, repo=ROOT):
    directory = Path(predictions)
    manifest = json.loads((directory/'prediction_manifest.json').read_text())
    if manifest['future_targets_used_in_fitting'] or manifest['status'] != 'U_ATTRIBUTION_PREDICTIONS_FROZEN':
        raise ValueError('Invalid attribution prediction contract')
    config_path = Path(manifest['config_path'])
    if sha256(config_path) != manifest['config_sha256'] or manifest['code_hashes'] != {p.name:sha256(p) for p in CODE}:
        raise ValueError('Attribution code/config changed')
    config,_,_,original,data,transform_path = verify(config_path,repo=repo)
    for filename,key in (('predictions.npz','prediction_sha256'),('calibration.json','calibration_sha256')):
        if sha256(directory/filename) != manifest[key]: raise ValueError('Prediction artifacts changed')
    run = Artifacts(output,seed=config['seed'],inputs=[config_path,data,transform_path,
        directory/'prediction_manifest.json',directory/'predictions.npz',directory/'calibration.json']+CODE)
    transform = SourceTransform.load(transform_path)
    obs,_ = read_metadata(data)
    with np.load(directory/'predictions.npz',allow_pickle=False) as saved:
        names,pred,states = saved['arms'].tolist(),saved['predictions'],saved['source_states']
        np.testing.assert_array_equal(saved['intervals'],INTERVALS)
    rows,errors,targets = [],[],[]
    for i,(start,stop) in enumerate(INTERVALS):
        layers,_ = load_population(data,obs,stop,original,source=False)
        target = transform.state(layers['spliced']).mean(0)
        mse = np.square(pred[i]-target).mean(1); errors.append(mse); targets.append(target)
        base = mse[names.index('identity')]
        for name,error in zip(names,mse):
            rows.append(dict(source_minute=start,target_minute=stop,arm=name,mse=float(error),
                gain_over_identity=float((base-error)/base)))
    by_arm = {name:[float(e[j]) for e in errors] for j,name in enumerate(names)}
    contrasts = []
    for aggregation,field in product(config['aggregations'],config['fields']):
        current = by_arm[f'{aggregation}__composition_current__total_current__{field}']
        for factor,control in (
                ('remove_current_composition',f'{aggregation}__composition_baseline__total_current__{field}'),
                ('remove_current_total',f'{aggregation}__composition_current__total_baseline__{field}')):
            excess = np.array(by_arm[control])-np.array(current)
            contrasts.append(dict(aggregation=aggregation,field=field,contrast=factor,
                control_mse_excess_by_interval=excess.tolist(),positive_means_current_factor_helped=True))
    result = dict(status='EXPLORATORY_U_ATTRIBUTION_COMPLETE',exploratory=True,
        independent_confirmation=False,biological_significance_test=False,
        technical_vs_biological_total_identified=False,original_results_changed=False,
        models=len(names),mse_by_interval=by_arm,factor_contrasts=contrasts,
        fixed_readout_parameters=2,router_enabled=False)
    with (run.directory/'metrics.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(run.directory/'targets_and_errors.npz',arms=np.array(names),targets=np.stack(targets),mse=np.stack(errors))
    run.json('summary.json',result)
    (run.directory/'RESULTS.md').write_text('# U factorial attribution\n\n'
        'Same original state/target for every arm. Equal-cell aggregation removes S-library weights; '
        'it does not causally separate technical depth from biological RNA content. All arms reported.\n\n'
        +json.dumps(result,indent=2)+'\n')
    run.json('status.json',dict(status='complete'))
    print(json.dumps(result),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser();sub = parser.add_subparsers(dest='stage',required=True)
    p = sub.add_parser('predict');p.add_argument('--config',default='configs/w5_scntseq_U_attribution_v1.json');p.add_argument('--output',required=True)
    e = sub.add_parser('evaluate');e.add_argument('--predictions',required=True);e.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.stage == 'predict':predict(args.config,args.output)
    else:evaluate(args.predictions,args.output)
