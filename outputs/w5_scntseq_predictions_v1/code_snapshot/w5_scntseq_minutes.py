"""Frozen low-capacity scNT-seq rolling forecasts, without neural GFG/JVP.

Only 0-min counts fit the state/kinetic transform. Scalars are calibrated once
on 15->30. Freeze 30->60 before reading 60 as the next source; 120 is read by
the separate evaluator only. This one-series study has no biological p-values.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from w4_artifacts import Artifacts, sha256
from w4_shorttime import read_count_rows, read_metadata
from w4_streams import SourceTransform


ROOT = Path(__file__).resolve().parents[2]
ARMS = ('identity', 'past_S', 'past_S_calibrated', 'vel_mean', 'vel_pool',
        'direct_U_raw', 'direct_U_delta0', 'new_raw', 'new_delta0')
INTERVALS = ((30, 60), (60, 120))
CODE = [Path(__file__).with_name(name) for name in (
    'w5_scntseq_minutes.py', 'w4_streams.py', 'w4_shorttime.py', 'w4_artifacts.py')]


def verify_config(path, commit, *, repo=ROOT):
    repo, path = Path(repo).resolve(), Path(path).resolve()
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Full frozen commit required')
    config = json.loads(path.read_text())
    if (config['kind'] != 'scntseq_minutes_mechanistic_validation'
            or config['times'] != [0,15,30,60,120] or config['calibration_interval'] != [15,30]
            or config['test_intervals'] != [list(x) for x in INTERVALS] or config['arms'] != list(ARMS)
            or config['primary_velocity_arm'] != 'vel_mean' or config['router_enabled']
            or config['gfg_jvp_enabled'] or config['fit_scale_after_calibration']):
        raise ValueError('Unexpected minute-scale protocol or model grid')
    protocol = repo/config['protocol_path']
    for filename in (path, protocol):
        frozen = subprocess.check_output(['git','-C',str(repo),'show', f'{commit}:{filename.relative_to(repo)}'])
        if frozen != filename.read_bytes():
            raise ValueError('Protocol/config changed after freeze')
    if not re.search(r'^状态：\*\*已冻结\*\*', protocol.read_text(), flags=re.MULTILINE):
        raise ValueError('External protocol is not frozen')
    data = Path(config['data_path'])
    if sha256(data) != config['data_sha256']:
        raise ValueError('Paired count release changed')
    return config, protocol, data


def time_rows(obs, minute, config):
    for name in ('time_minutes', 'spliced_counts_common_genes', 'unspliced_counts_common_genes'):
        if name not in obs:
            raise ValueError(f'Missing technical column {name}')
    keep = ((obs['time_minutes'].astype(int).to_numpy() == minute)
        & (obs['spliced_counts_common_genes'].to_numpy() >= config['minimum_spliced_counts'])
        & (obs['unspliced_counts_common_genes'].to_numpy() >= config['minimum_unspliced_counts']))
    rows = np.flatnonzero(keep)
    if len(rows) < 2:
        raise ValueError(f'Insufficient technically qualified cells at {minute} minutes')
    return rows


def load_population(data, obs, minute, config, *, source):
    rows = time_rows(obs, minute, config)
    fields = ('spliced','unspliced','new','total') if source else ('spliced',)
    return {name:read_count_rows(data, name, rows) for name in fields}, obs.index[rows].to_numpy(dtype=str)


def population_features(transform, layers, *, baseline=None, previous_state=None, elapsed_minutes=None):
    s, u = layers['spliced'], layers['unspliced']
    sn, un = transform.normalized(s, u)
    state = transform.state_normalized(sn).mean(0)
    ps, pu = transform.normalized(s.sum(0), u.sum(0))
    raw_u = np.log1p(pu)[0] @ transform.components.T
    total_depth = float(layers['total'].sum())
    if total_depth <= 0:
        raise ValueError('Empty GEO total library for labeling features')
    new = np.asarray(layers['new'].sum(0)).ravel()[transform.genes] * (transform.target_sum/total_depth)
    raw_new = np.log1p(new) @ transform.components.T
    past = np.zeros_like(state)
    if previous_state is not None:
        if elapsed_minutes is None or elapsed_minutes <= 0:
            raise ValueError('Positive historical interval required')
        past = (state-previous_state)/elapsed_minutes
    vectors = dict(identity=np.zeros_like(state), past_S=past, past_S_calibrated=past.copy(),
        vel_mean=transform.projected_rate(sn, un-transform.gamma*sn).mean(0),
        vel_pool=transform.projected_rate(ps, pu-transform.gamma*ps)[0],
        direct_U_raw=raw_u, direct_U_delta0=np.zeros_like(state) if baseline is None else raw_u-baseline['direct_U_raw'],
        new_raw=raw_new, new_delta0=np.zeros_like(state) if baseline is None else raw_new-baseline['new_raw'])
    if set(vectors) != set(ARMS) or any(not np.isfinite(v).all() for v in vectors.values()):
        raise ValueError('Invalid projected source features')
    return state, vectors


def nonnegative_scale(feature, observed_rate, floor=1e-12):
    feature, observed_rate = np.asarray(feature), np.asarray(observed_rate)
    if feature.shape != observed_rate.shape or not np.isfinite(feature).all() or not np.isfinite(observed_rate).all():
        raise ValueError('Finite aligned calibration vectors required')
    denominator = float(feature @ feature)
    if denominator <= floor:
        return 0., True
    return max(0., float(feature @ observed_rate)/denominator), False


def forecast(state, vectors, scales, elapsed_minutes):
    if elapsed_minutes <= 0 or set(scales) != set(ARMS) or any(not np.isfinite(a) or a < 0 for a in scales.values()):
        raise ValueError('Invalid forecast duration or scales')
    return np.stack([state+elapsed_minutes*scales[arm]*vectors[arm] for arm in ARMS])


def predict(config_path, output, freeze_commit, *, repo=ROOT):
    config, protocol, data = verify_config(config_path, freeze_commit, repo=repo)
    run = Artifacts(output, seed=config['seed'], inputs=[Path(config_path), protocol, data]+CODE)
    obs, genes = read_metadata(data)
    if sorted(obs['time_minutes'].astype(int).unique()) != config['times']:
        raise ValueError('Unexpected sampling times')
    base_layers, base_ids = load_population(data, obs, 0, config, source=True)
    transform = SourceTransform.fit(base_layers['spliced'], base_layers['unspliced'], cell_ids=base_ids,
        allowed_ids=set(base_ids), gene_names=genes, n_hvg=config['n_hvg'], dimensions=config['dimensions'],
        clusters=config['clusters'], target_sum=config['target_sum'], seed=config['seed'])
    transform.save(run.directory/'source_transform.npz')
    base_state, base_features = population_features(transform, base_layers)
    layers15, ids15 = load_population(data, obs, 15, config, source=True)
    state15, features15 = population_features(transform, layers15, baseline=base_features,
        previous_state=base_state, elapsed_minutes=15)
    layers30, ids30 = load_population(data, obs, 30, config, source=True)
    state30, features30 = population_features(transform, layers30, baseline=base_features,
        previous_state=state15, elapsed_minutes=15)
    observed_rate = (state30-state15)/15
    scales, details = dict(identity=0., past_S=1.), {}
    for arm in ARMS:
        if arm in scales:
            continue
        scales[arm], degenerate = nonnegative_scale(features15[arm], observed_rate, config['scale_denominator_floor'])
        details[arm] = dict(scale=scales[arm], degenerate_feature=degenerate,
            calibration_feature_norm=float(np.linalg.norm(features15[arm])),
            alignment_dot=float(features15[arm] @ observed_rate))
    run.json('calibration.json', dict(interval=[15,30], scales=scales, details=details,
        baseline_cells=len(base_ids), calibration_source_cells=len(ids15), calibration_target_cells=len(ids30),
        calibration_not_test=True, gamma_zero_genes=int(np.sum(transform.gamma == 0))))
    # This file is written BEFORE any 60-min expression is loaded.
    first = forecast(state30, features30, scales, 30)
    np.savez_compressed(run.directory/'prediction_30_to_60.npz', arms=np.array(ARMS), prediction=first, source_state=state30)
    run.json('prediction_30_to_60_manifest.json', dict(status='INTERVAL_PREDICTION_FROZEN',
        source_minute=30, target_minute=60, expression_minutes_read=[0,15,30],
        future_expression_used=False, prediction_sha256=sha256(run.directory/'prediction_30_to_60.npz')))
    layers60, ids60 = load_population(data, obs, 60, config, source=True)
    state60, features60 = population_features(transform, layers60, baseline=base_features,
        previous_state=state30, elapsed_minutes=30)
    second = forecast(state60, features60, scales, 60)
    np.savez_compressed(run.directory/'prediction_60_to_120.npz', arms=np.array(ARMS), prediction=second, source_state=state60)
    run.json('prediction_60_to_120_manifest.json', dict(status='INTERVAL_PREDICTION_FROZEN',
        source_minute=60, target_minute=120, expression_minutes_read=[0,15,30,60],
        future_expression_used=False, calibration_updated=False,
        prediction_sha256=sha256(run.directory/'prediction_60_to_120.npz')))
    np.savez_compressed(run.directory/'predictions.npz', arms=np.array(ARMS), intervals=np.array(INTERVALS),
        predictions=np.stack((first, second)), source_states=np.stack((state30,state60)))
    run.json('prediction_manifest.json', dict(status='MINUTE_PREDICTIONS_FROZEN', kind=config['kind'],
        config_path=str(Path(config_path).resolve()), freeze_commit=freeze_commit,
        protocol_sha256=sha256(protocol), arms=list(ARMS), intervals=[list(x) for x in INTERVALS],
        prediction_sha256=sha256(run.directory/'predictions.npz'),
        transform_sha256=sha256(run.directory/'source_transform.npz'),
        calibration_sha256=sha256(run.directory/'calibration.json'),
        source_cells={'0':len(base_ids),'15':len(ids15),'30':len(ids30),'60':len(ids60)},
        per_interval_future_expression_used=False, scale_fit_minutes=[15,30],
        source_transform_fit_minutes=[0], future_120_expression_read=False,
        source60_read_only_after_first_prediction_frozen=True, router_enabled=False, gfg_jvp_enabled=False,
        code_hashes={str(p.resolve()):sha256(p) for p in CODE}))
    run.json('status.json', dict(status='complete'))
    print(json.dumps(dict(status='MINUTE_PREDICTIONS_FROZEN', output=str(output), scales=scales)), flush=True)


def cosine(a, b):
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return None if denominator <= 1e-12 else float(np.clip(a @ b/denominator, -1, 1))


def evaluate(predictions, output, *, repo=ROOT):
    directory = Path(predictions)
    manifest = json.loads((directory/'prediction_manifest.json').read_text())
    config_path = Path(manifest['config_path'])
    config, protocol, data = verify_config(config_path, manifest['freeze_commit'], repo=repo)
    if (manifest['status'] != 'MINUTE_PREDICTIONS_FROZEN' or manifest['per_interval_future_expression_used']
            or manifest['future_120_expression_read'] or manifest['protocol_sha256'] != sha256(protocol)):
        raise ValueError('Invalid minute-scale prediction contract')
    for filename,key in (('predictions.npz','prediction_sha256'), ('source_transform.npz','transform_sha256'),
                         ('calibration.json','calibration_sha256')):
        if sha256(directory/filename) != manifest[key]:
            raise ValueError(f'Frozen artifact changed: {filename}')
    if manifest['code_hashes'] != {str(p.resolve()):sha256(p) for p in CODE}:
        raise ValueError('Prediction/evaluation code changed')
    for start,stop in INTERVALS:
        part = json.loads((directory/f'prediction_{start}_to_{stop}_manifest.json').read_text())
        if part['future_expression_used'] or sha256(directory/f'prediction_{start}_to_{stop}.npz') != part['prediction_sha256']:
            raise ValueError('An interval prediction changed')
    run = Artifacts(output, seed=config['seed'], inputs=[config_path, protocol, data,
        directory/'prediction_manifest.json', directory/'predictions.npz', directory/'source_transform.npz',
        directory/'calibration.json']+CODE)
    obs, genes = read_metadata(data)
    transform = SourceTransform.load(directory/'source_transform.npz')
    np.testing.assert_array_equal(genes, transform.gene_names)
    with np.load(directory/'predictions.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['arms'], ARMS); np.testing.assert_array_equal(saved['intervals'], INTERVALS)
        pred, states = saved['predictions'], saved['source_states']
    scales = json.loads((directory/'calibration.json').read_text())['scales']
    truth, rows, error = [], [], np.empty((len(INTERVALS), len(ARMS)))
    for i,(start,stop) in enumerate(INTERVALS):
        future, identities = load_population(data, obs, stop, config, source=False)
        state = transform.state(future['spliced']).mean(0)
        truth.append(state)
        for j,arm in enumerate(ARMS):
            error[i,j] = np.square(pred[i,j]-state).sum()
        identity_error = error[i,ARMS.index('identity')]
        for j,arm in enumerate(ARMS):
            rows.append(dict(source_minute=start, target_minute=stop, arm=arm,
                source_cells=manifest['source_cells'][str(start)], target_cells=len(identities),
                mse=float(error[i,j]/config['dimensions']),
                gain_over_identity=float((identity_error-error[i,j])/max(identity_error,1e-12)),
                displacement_cosine=cosine(pred[i,j]-states[i],state-states[i]),
                predicted_displacement_norm=float(np.linalg.norm(pred[i,j]-states[i])),
                true_displacement_norm=float(np.linalg.norm(state-states[i])), scale=scales[arm]))
    vi, ii = ARMS.index('vel_mean'), ARMS.index('identity')
    positive = bool((error[:,vi] < error[:,ii]).all())
    better_past = all(error[:,vi].mean() < error[:,ARMS.index(arm)].mean() for arm in ('past_S','past_S_calibrated'))
    better_u = all(error[:,vi].mean() < error[:,ARMS.index(arm)].mean() for arm in ('direct_U_raw','direct_U_delta0'))
    result = dict(status='SCNT_MINUTES_EVALUATED', kind='single_series_descriptive_mechanistic_validation',
        freeze_commit=manifest['freeze_commit'], intervals=[list(x) for x in INTERVALS],
        primary_velocity_arm='vel_mean', primary_descriptive_rule_met=positive and better_past,
        velocity_beats_identity_in_both_intervals=positive, velocity_beats_both_past_baselines=bool(better_past),
        velocity_beats_both_direct_U_baselines=bool(better_u),
        mean_mse={arm:float(error[:,j].mean()/config['dimensions']) for j,arm in enumerate(ARMS)},
        mean_interval_gain_over_identity={arm:float(np.mean((error[:,ii]-error[:,j])/np.maximum(error[:,ii],1e-12)))
            for j,arm in enumerate(ARMS)},
        biological_replicate_count=None, significance_test_performed=False,
        new_RNA_is_net_velocity=False, router_enabled=False, gfg_jvp_enabled=False,
        calibration_interval_is_test=False, inference_scope='two_rolling_intervals_one_stimulus_series')
    np.savez_compressed(run.directory/'targets_and_errors.npz', intervals=np.array(INTERVALS), arms=np.array(ARMS),
        truth=np.stack(truth), errors=error)
    with (run.directory/'metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    run.json('summary.json', result)
    (run.directory/'RESULTS.md').write_text('# scNT-seq minute-scale rolling forecasts\n\n'
        'One nonnegative scale per calibrated arm, fitted only on 15→30. '
        'Test: 30→60 and 60→120. No biological p-values from cells/genes.\n\n'
        +json.dumps(result, indent=2)+'\n')
    run.json('status.json', dict(status='complete'))
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('predict')
    p.add_argument('--config', default='configs/w5_scntseq_minutes_v1.json')
    p.add_argument('--freeze-commit', required=True); p.add_argument('--output', required=True)
    e = sub.add_parser('evaluate')
    e.add_argument('--predictions', required=True); e.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.stage == 'predict':
        predict(args.config, args.output, args.freeze_commit)
    else:
        evaluate(args.predictions, args.output)


if __name__ == '__main__':
    main()
