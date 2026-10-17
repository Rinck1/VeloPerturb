"""Train/evaluate source-only no-JVP heads on the existing day4->5 development split.

The six-arm H1 implementation is unchanged. This optional extension freezes
all head/ridge predictions before a separate command opens development targets.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from w4_artifacts import Artifacts, sha256
from w4_arms import CORE_ARMS, average_predictions_then_errors, fit_ridge_training_cv
from w4_kinetic_head import (FEATURES, apply_calibrator, fit_calibrator, pooled_feature_values,
                             source_partition_pairs)
from w4_shorttime import (CONTROL_LABELS, DEVELOPMENT_TFS, condition_embeddings, qc_mask,
                          read_count_rows, read_metadata, validate_development_times)
from w4_streams import SourceTransform, add_swap_feature, group_mean, population_features


ROOT = Path(__file__).resolve().parents[2]
ARMS = ('learned_velocity', 'learned_velocity_Uswap', 'learned_direct_U', 'learned_S_only')
BRANCHES = dict(learned_velocity='vel_pool', learned_velocity_Uswap='vel_pool_Uswap',
                learned_direct_U='direct_U_pool', learned_S_only='pooled_S')
CODE = [Path(__file__).with_name(name) for name in (
    'w4_learned_head.py', 'w4_kinetic_head.py', 'w4_streams.py', 'w4_shorttime.py', 'w4_arms.py', 'w4_artifacts.py')]


def config_file(path):
    config = json.loads(Path(path).read_text())
    if (config['kind'] != 'development_only' or config['features'] != list(FEATURES)
            or config['arms'] != list(ARMS) or config['head_condition_input']
            or config['head_future_supervision'] or config['router_enabled'] or config['primary_replication_changed']
            or not config['seeds'] or len(set(config['seeds'])) != len(config['seeds'])):
        raise ValueError('This runner is source-trained development only, not H1/router execution')
    return config


def baseline_artifacts(config):
    directory = ROOT/config['baseline_predictions']
    manifest = json.loads((directory/'prediction_manifest.json').read_text())
    if (manifest['kind'] != 'development' or (manifest['source_day'], manifest['target_day']) != (4, 5)
            or manifest['heldout_future_expression_read'] or manifest['heldout_TFs'] != list(DEVELOPMENT_TFS)):
        raise ValueError('Expected the existing frozen day4->5 development baseline')
    if sha256(directory/'predictions.npz') != manifest['prediction_sha256']:
        raise ValueError('Frozen baseline predictions changed')
    transform = directory/'transform_seed0.npz'
    if sha256(transform) != manifest['transform_sha256']['0']:
        raise ValueError('Frozen source transform changed')
    # The old runner's evaluation/reporting code has evolved. Bind the actual
    # feature/ridge dependencies, not that entire historical CLI file.
    for path in (Path(manifest['source_path']), Path(manifest['target_path']), CODE[2], CODE[4], CODE[5]):
        expected = [digest for filename, digest in manifest['data_hashes'].items()
                    if (ROOT/filename).resolve() == path.resolve()]
        if len(expected) != 1 or sha256(path) != expected[0]:
            raise ValueError(f'Baseline input or source dependency changed: {path}')
    return directory, manifest, transform


def add_pooled_s(transform, features):
    result = dict(features)
    sn, _ = transform.normalized(features['pooled_s'], features['pooled_u'])
    result['pooled_S'] = transform.state_normalized(sn)
    return result


def predict(config_path, output, *, device='cpu'):
    config = config_file(config_path)
    if device != 'cpu':
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy(device)
    baseline, old, transform_path = baseline_artifacts(config)
    source, target = Path(old['source_path']), Path(old['target_path'])
    embeddings = ROOT/config['embeddings']
    embedding_digest = [h for p,h in old['data_hashes'].items() if (ROOT/p).resolve() == embeddings.resolve()]
    if embedding_digest != [sha256(embeddings)]:
        raise ValueError('Use exactly the baseline frozen condition embeddings')
    obs, genes = read_metadata(source); future_obs, future_genes = read_metadata(target)
    validate_development_times(obs, future_obs)
    np.testing.assert_array_equal(genes, future_genes)
    labels, future_labels = obs['condition'].astype(str).to_numpy(), future_obs['condition'].astype(str).to_numpy()
    train_labels = sorted(old['training_TFs']+old['controls'])
    si = np.flatnonzero(qc_mask(obs) & np.isin(labels, train_labels))
    vi = np.flatnonzero(qc_mask(obs) & np.isin(labels, DEVELOPMENT_TFS))
    ti = np.flatnonzero(qc_mask(future_obs) & np.isin(future_labels, train_labels))
    if set(future_labels[ti]) != set(train_labels) or set(train_labels) & set(DEVELOPMENT_TFS):
        raise ValueError('Invalid training target allowlist')
    transform = SourceTransform.load(transform_path)
    ids = np.array([f'day4:{i}' for i in obs.index[si]])
    query_ids = np.array([f'day4:{i}' for i in obs.index[vi]])
    np.testing.assert_array_equal(ids, transform.source_ids)
    np.testing.assert_array_equal(genes, transform.gene_names)
    run = Artifacts(output, seed=config['seeds'], inputs=[Path(config_path), baseline/'prediction_manifest.json',
        baseline/'predictions.npz', transform_path, source, target, embeddings]+CODE)
    s, u = read_count_rows(source, 'spliced', si), read_count_rows(source, 'unspliced', si)
    sv, uv = read_count_rows(source, 'spliced', vi), read_count_rows(source, 'unspliced', vi)
    train = add_pooled_s(transform, population_features(transform, s, u, labels[si], train_labels, ids))
    test = add_pooled_s(transform, population_features(transform, sv, uv, labels[vi], DEVELOPMENT_TFS, query_ids))
    train, train_swap = add_swap_feature(transform, train, train, seed=0)
    test, test_swap = add_swap_feature(transform, test, train, seed=0)
    predictions = {arm: [] for arm in ARMS}
    histories, selections, checkpoints, diagnostics = {}, {}, {}, {}
    et, ev = condition_embeddings(embeddings, train_labels), condition_embeddings(embeddings, DEVELOPMENT_TFS)
    controls = np.isin(train_labels, CONTROL_LABELS)
    # Future TRAINING TFs are legal ridge labels; the head never receives them.
    future_train = read_count_rows(target, 'spliced', ti)
    delta = group_mean(transform.state(future_train), future_labels[ti], train_labels)-train['state']
    for seed in config['seeds']:
        a, b, identities = source_partition_pairs(transform, s, u, labels[si], train_labels,
            partitions=config['training_partitions_per_condition'], seed=seed)
        da, db, _ = source_partition_pairs(transform, s, u, labels[si], train_labels,
            partitions=config['diagnostic_partitions_per_condition'], seed=seed+100000)
        models = {}
        histories[str(seed)], diagnostics[str(seed)], selections[str(seed)] = {}, {}, {}
        for feature in FEATURES:
            model, history = fit_calibrator(a[feature], b[feature], config, seed=seed, device=device)
            name = f'head_seed{seed}_{feature}.pt'
            torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                hidden=config['hidden'], maximum_relative_correction=config['maximum_relative_correction'],
                feature=feature, seed=seed, source_transform_sha256=sha256(transform_path)), run.directory/name)
            checkpoints[name] = sha256(run.directory/name)
            models[feature], histories[str(seed)][feature] = model, history
            scale = model.scale.detach().cpu().numpy()
            before = float(np.mean(((da[feature]-db[feature])/scale)**2))
            after = .5*(np.mean(((apply_calibrator(model, da[feature])-db[feature])/scale)**2)
                       +np.mean(((apply_calibrator(model, db[feature])-da[feature])/scale)**2))
            diagnostics[str(seed)][feature] = dict(pair_mse_before=before, pair_mse_after=float(after),
                reused_source_cells=True, independent_biological_replicates=False)
        for arm, feature in BRANCHES.items():
            head = models['vel_pool' if feature == 'vel_pool_Uswap' else feature]
            tr, te = apply_calibrator(head, train[feature]), apply_calibrator(head, test[feature])
            x = np.concatenate((train['state'], et, tr), axis=1)
            xe = np.concatenate((test['state'], ev, te), axis=1)
            scaler, ridge, selection = fit_ridge_training_cv(x, delta, controls=controls)
            predictions[arm].append(test['state']+ridge.predict(scaler.transform(xe)))
            selections[str(seed)][arm] = selection
        run.json('status.json', dict(status='training', completed_seed=seed, router_enabled=False))
        print(json.dumps(dict(status='source_heads_and_predictions_complete', seed=seed)), flush=True)
    # Existing six arms are copied, not refitted or selected using new results.
    with np.load(baseline/'predictions.npz', allow_pickle=False) as p:
        original = {arm:p[arm].copy() for arm in CORE_ARMS}
    np.savez_compressed(run.directory/'predictions.npz', labels=np.array(DEVELOPMENT_TFS),
        learned_seeds=np.array(config['seeds']), baseline_seeds=np.array(old['seed']),
        **original, **{arm:np.stack(values) for arm,values in predictions.items()})
    run.json('training.json', dict(histories=histories, source_diagnostics=diagnostics,
        source_partition_conditions=sorted(set(identities)), ridge_selection=selections,
        U_swap=dict(train=train_swap, development_source=test_swap)))
    run.json('prediction_manifest.json', dict(status='PREDICTIONS_FROZEN', kind='development',
        arms=list(CORE_ARMS)+list(ARMS), learned_seeds=config['seeds'],
        baseline_directory=str(baseline.resolve()), source_path=str(source), target_path=str(target),
        source_transform=str(transform_path.resolve()), source_transform_sha256=sha256(transform_path),
        prediction_sha256=sha256(run.directory/'predictions.npz'), checkpoints=checkpoints,
        heldout_TFs=list(DEVELOPMENT_TFS), training_TFs=old['training_TFs'],
        heldout_future_expression_read=False, head_future_supervision=False,
        head_training_scope='outer_training_source_cells_only', head_conditions_input=False,
        router_enabled=False, primary_replication_changed=False, config=config))
    run.json('status.json', dict(status='complete'))


def evaluate(directory, output):
    directory = Path(directory)
    manifest = json.loads((directory/'prediction_manifest.json').read_text())
    if (manifest['kind'] != 'development' or manifest['heldout_future_expression_read']
            or sha256(directory/'predictions.npz') != manifest['prediction_sha256']):
        raise ValueError('Invalid frozen development predictions')
    transform_path, target = Path(manifest['source_transform']), Path(manifest['target_path'])
    if sha256(transform_path) != manifest['source_transform_sha256']:
        raise ValueError('Source transform changed')
    for filename, digest in manifest['checkpoints'].items():
        if sha256(directory/filename) != digest:
            raise ValueError('Frozen source head changed')
    for path in [target]+CODE:
        expected = [h for p,h in manifest['data_hashes'].items() if Path(p).resolve() == path.resolve()]
        if expected != [sha256(path)]:
            raise ValueError(f'Prediction dependency changed: {path}')
    run = Artifacts(output, seed=manifest['learned_seeds'], inputs=[directory/'prediction_manifest.json',
        directory/'predictions.npz', directory/'training.json', transform_path, target]+CODE)
    obs, genes = read_metadata(target)
    transform = SourceTransform.load(transform_path)
    np.testing.assert_array_equal(genes, transform.gene_names)
    labels = list(DEVELOPMENT_TFS)
    rows = np.flatnonzero(qc_mask(obs) & obs['condition'].isin(labels).to_numpy())
    truth = group_mean(transform.state(read_count_rows(target, 'spliced', rows)),
        obs['condition'].astype(str).to_numpy()[rows], labels)
    with np.load(directory/'predictions.npz', allow_pickle=False) as p:
        np.testing.assert_array_equal(p['labels'], labels)
        errors = {arm:average_predictions_then_errors(p[arm], truth) for arm in manifest['arms']}
    result = dict(status='NOJVP_HEAD_DEVELOPMENT_EVALUATED', kind='development',
        mse={arm:float(error.mean()/truth.shape[1]) for arm,error in errors.items()},
        relative_gain_over_static={arm:float(1-error.mean()/errors['static'].mean()) for arm,error in errors.items()},
        mean_TF_relative_gain={arm:float(np.mean((errors['static']-error)/np.maximum(errors['static'],1e-12)))
            for arm,error in errors.items()},
        learned_velocity_gain_over_analytic=float(1-errors['learned_velocity'].mean()/errors['vel_pool'].mean()),
        no_formal_replication_claim=True, head_future_supervision=False, router_enabled=False,
        interpretation='source-population feature denoising, not independent true-velocity validation')
    run.json('summary.json', result)
    np.savez_compressed(run.directory/'targets_and_errors.npz', truth=truth, labels=np.array(labels), **errors)
    with (run.directory/'metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['arm', 'TF', 'sse', 'mse'])
        writer.writeheader()
        for arm, values in errors.items():
            for label, value in zip(labels, values):
                writer.writerow(dict(arm=arm, TF=label, sse=float(value), mse=float(value/truth.shape[1])))
    (run.directory/'RESULTS.md').write_text('# Source-only no-JVP head: day4→5 development\n\n'
        'Fixed three-seed mean predictions. Matched 1600-parameter heads for velocity, direct U, and S-only '
        'population features. No future labels in head training; future training TFs supervise ridge only.\n\n'
        +json.dumps(result, indent=2)+'\n')
    run.json('status.json', dict(status='complete'))
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('predict')
    p.add_argument('--config', default='configs/w4_nojvp_head_development_v1.json')
    p.add_argument('--output', required=True); p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    e = sub.add_parser('evaluate')
    e.add_argument('--predictions', required=True); e.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.stage == 'predict':
        predict(args.config, args.output, device=args.device)
    else:
        evaluate(args.predictions, args.output)


if __name__ == '__main__':
    main()
