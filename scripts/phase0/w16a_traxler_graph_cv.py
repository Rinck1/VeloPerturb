"""Fixed seven-arm, three-fold perturbation study; no held-out target pre-freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from w4_artifacts import Artifacts, sha256

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMS = ('no_graph', 'state_graph', 'U_graph', 'velocity_graph', 'shuffle0', 'shuffle1', 'shuffle2')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


def registered_splits(original, folds, seed):
    from sklearn.model_selection import KFold
    parts = [set(original[k]) for k in ('train', 'val', 'test')]
    if any(parts[i] & parts[j] for i in range(3) for j in range(i)):
        raise ValueError('Original KO sets overlap')
    labels = np.array(sorted(parts[0] | parts[1]))
    return [dict(train=labels[t].tolist(), val=labels[v].tolist(), test=sorted(parts[2]))
            for t, v in KFold(folds, shuffle=True, random_state=seed).split(labels)]


def read_csr_rows(group, rows):
    """Read only requested data/indices segments, not the other cells' counts."""
    from scipy import sparse
    if group.attrs.get('encoding-type') != 'csr_matrix':
        raise ValueError('Explicit CSR row access required')
    rows = np.asarray(rows, dtype=int)
    if rows.ndim != 1 or len(set(rows)) != len(rows) or (rows < 0).any():
        raise ValueError('Unique nonnegative rows required')
    indptr = group['indptr'][:]
    if len(rows) and rows.max() >= len(indptr)-1: raise ValueError('Row outside matrix')
    starts, ends = indptr[rows], indptr[rows+1]
    data = [group['data'][int(a):int(b)] for a, b in zip(starts, ends)]
    indices = [group['indices'][int(a):int(b)] for a, b in zip(starts, ends)]
    return sparse.csr_matrix((np.concatenate(data) if data else np.empty(0),
        np.concatenate(indices) if indices else np.empty(0, dtype=int),
        np.r_[0, np.cumsum(ends-starts)]), shape=(len(rows), int(group.attrs['shape'][1])))


def load_counts(h5dir, cell_ids, layers=('spliced', 'unspliced')):
    import h5py
    from scipy import sparse
    requested = set(map(str, cell_ids))
    if len(requested) != len(cell_ids) or not requested: raise ValueError('Invalid selected cells')
    matrices = {k:[] for k in layers}; loaded = []; genes = None
    for library in sorted({k.split(':', 1)[0] for k in requested}):
        with h5py.File(Path(h5dir)/(library+'.h5ad'), 'r') as f:
            names = f['obs'][f['obs'].attrs['_index']].asstr()[:]
            current = f['var'][f['var'].attrs['_index']].asstr()[:]
            if genes is None: genes = current
            else: np.testing.assert_array_equal(genes, current)
            rows = np.flatnonzero(np.isin(names, list(requested)))
            loaded.extend(names[rows].tolist())
            for key in layers: matrices[key].append(read_csr_rows(f['layers'][key], rows))
    if set(loaded) != requested or len(loaded) != len(requested): raise ValueError('Cell alignment failed')
    return np.asarray(loaded, dtype=str), {k:sparse.vstack(v).tocsr() for k, v in matrices.items()}, genes


def fit_transform(s, u, genes, ids, *, n_genes, dimensions, seed):
    """Sparse full-library normalization; otherwise the original frozen formula."""
    from sklearn.decomposition import PCA
    from veloroute.latent import FrozenSplicingTransform
    from veloroute.artifacts import object_hash
    depth = np.asarray(s.sum(1)).ravel().astype(float)
    if (depth <= 0).any(): raise ValueError('Zero spliced source depth')
    normalized = s.astype(float).multiply((10000./depth)[:, None]).tocsr()
    logs = normalized.copy(); logs.data = np.log1p(logs.data)
    mean = np.asarray(logs.mean(0)).ravel()
    variance = np.maximum(np.asarray(logs.power(2).mean(0)).ravel()-mean*mean, 0.)
    candidates = np.flatnonzero(np.asarray((s > 0).sum(0)).ravel() >= min(10, len(ids)))
    panel = candidates[np.argsort(-variance[candidates], kind='stable')[:n_genes]]
    if len(panel) < dimensions or len(ids) <= dimensions: raise ValueError('Insufficient source support')
    sn = normalized[:, panel].toarray()
    un = u[:, panel].astype(float).multiply((10000./depth)[:, None]).toarray()
    pca = PCA(n_components=dimensions, svd_solver='randomized', random_state=seed)
    z = pca.fit_transform(np.log1p(sn))
    tail = sn >= np.quantile(sn, .8, axis=0)
    denominator = (sn*sn*tail).sum(0)
    gamma = (sn*un*tail).sum(0)/np.maximum(denominator, 1e-8)
    supported = ((un > 0).sum(0) >= 10) & (denominator > 1e-8) & (gamma > 0)
    gamma = np.where(supported, gamma, 0.)
    v = np.where(supported, (un-sn*gamma)/(1+sn), 0.) @ pca.components_.T
    scale = float(np.sqrt(np.mean(v*v)))
    if not np.isfinite(scale) or scale < 1e-10: raise ValueError('No supported velocity')
    ul = np.log1p(un) @ pca.components_.T
    design = np.column_stack((np.ones(len(z)), z)); penalty = np.eye(design.shape[1]); penalty[0, 0] = 0.
    coefficient = np.linalg.solve(design.T@design+penalty, design.T@ul)
    return FrozenSplicingTransform(tuple(map(str, genes)), panel, pca.mean_, pca.components_, gamma,
        supported, scale, ul.mean(0), coefficient, object_hash(sorted(map(str, ids))), 10000.)


def apply_transform(transform, s, u=None):
    depth = np.asarray(s.sum(1)).ravel().astype(float)
    if (depth <= 0).any(): raise ValueError('Zero source/target spliced depth')
    sn = s[:, transform.selected].astype(float).multiply((transform.target_sum/depth)[:, None]).toarray()
    z = (np.log1p(sn)-transform.mean) @ transform.components.T
    result = dict(z=z.astype('float32'))
    if u is not None:
        un = u[:, transform.selected].astype(float).multiply((transform.target_sum/depth)[:, None]).toarray()
        vg = np.where(transform.supported, (un-sn*transform.gamma)/(1+sn), 0.)
        result.update(velocity=((vg @ transform.components.T)/transform.velocity_scale).astype('float32'),
            u_features=(np.log1p(un) @ transform.components.T-transform.u_mean).astype('float32'),
            u_predicted=(np.column_stack((np.ones(len(z)), z)) @ transform.u_coef-transform.u_mean).astype('float32'),
            depth=depth)
    return result


def frame_ids(frame, labels, hour):
    part = frame[frame['KO'].astype(str).isin(labels) & (frame['time'].astype(int) == hour)]
    return (part['library'].astype(str)+':'+part['barcode'].astype(str)).to_numpy(dtype=str)


def build_fold(plan, split, directory, frame):
    from veloroute.latent import save_pack
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    ids, counts, genes = load_counts(plan['h5dir'], frame_ids(frame, split['train'], plan['source_hours']))
    transform = fit_transform(counts['spliced'], counts['unspliced'], genes, ids,
        n_genes=plan['n_genes'], dimensions=plan['n_components'], seed=plan['transform_seed'])
    transform.save(directory/'transform.npz'); fingerprint = sha256(directory/'transform.npz')
    lookups = dict(zip((frame['library'].astype(str)+':'+frame['barcode'].astype(str)), frame['KO'].astype(str)))
    def save(side, role, cell_ids, arrays, hour):
        arrays.update(cell_ids=cell_ids, conditions=np.asarray([lookups[k] for k in cell_ids], dtype=str))
        save_pack(directory/f'{role}_{side}.npz', arrays, dict(side=side, role=role, day=hour,
            task='traxler_6_24_graph_cv', kind='development', transform_hash=fingerprint,
            fit_ids_hash=transform.fit_ids_hash, formal_ready=False, estimator=transform.estimator))
    save('source', 'train', ids, apply_transform(transform, counts['spliced'], counts['unspliced']), plan['source_hours'])
    ids, counts, found_genes = load_counts(plan['h5dir'], frame_ids(frame, split['val'], plan['source_hours']))
    np.testing.assert_array_equal(genes, found_genes)
    save('source', 'validation', ids, apply_transform(transform, counts['spliced'], counts['unspliced']), plan['source_hours'])
    ids, counts, found_genes = load_counts(plan['h5dir'], frame_ids(frame, split['train'], plan['target_hours']), ('spliced',))
    np.testing.assert_array_equal(genes, found_genes)
    save('target', 'train', ids, apply_transform(transform, counts['spliced']), plan['target_hours'])
    write_json(directory/'split.json', split)
    c = read_json(plan['base_configuration'])
    c.update(fold=str(directory), split=str(directory/'split.json'), conditions=plan['conditions'],
             seeds=plan['model_seeds'], steps=plan['steps'])
    write_json(directory/'configuration.json', c)
    write_json(directory/'ready.json', dict(status='complete', train_KOs=len(split['train']),
        validation_KOs=len(split['val']), heldout_target_loaded=False, test_cells_loaded=False,
        transform_sha256=fingerprint, supported_genes=int(transform.supported.sum())))


def code_inputs(configuration, plan):
    names = ('w16a_traxler_graph_cv.py', 'w4_artifacts.py', 'w15o_velocity_neighborhood.py',
        'w15n_velocity_auxiliary.py', 'w15h_traxler_program_router.py', 'w15i_program_router_followups.py',
        'w15a_cellflow_moe.py', 'w15a_cellflow_moe_official.py', 'w15a_cellflow_moe_real_predict.py',
        'w15a_cellflow_train.py', 'w15a_cellflow_adapter.py', 'w15c_renge_l3.py', 'w15g_response_program_probe.py')
    return [Path(configuration), Path(plan['protocol']), Path(plan['base_configuration']),
            Path(plan['conditions']), ROOT/'src/veloroute/latent.py', *[HERE/n for n in names]]


def prepare(configuration):
    import pandas as pd
    plan = read_json(configuration); root = Path(plan['root'])
    splits = registered_splits(read_json(plan['original_split']), plan['folds'], plan['split_seed'])
    if len(set(splits[0]['train']+splits[0]['val'])) != plan['development_KOs'] or len(splits[0]['test']) != plan['reserved_test_KOs']:
        raise ValueError('Registered development/test cohort changed')
    raw = [Path(plan['h5dir'])/(lib+'.h5ad') for lib in ('B1', 'B2', 'B3', 'C1', 'C2', 'C3')]
    run = Artifacts(root/'prepare', seed=plan['split_seed'], inputs=[*code_inputs(configuration, plan),
        Path(plan['cell_manifest']), Path(plan['original_split']), *raw])
    try:
        frame = pd.read_csv(plan['cell_manifest'])
        for i, split in enumerate(splits):
            run.json('status.json', dict(status='preparing_fold', fold=i, completed_folds=i))
            build_fold(plan, split, root/'pack'/f'fold{i}', frame)
            print(json.dumps(dict(status='fold_prepared', fold=i)), flush=True)
        run.json('summary.json', dict(status='TRAXLER_GRAPH_CV_PACKS_COMPLETE', folds=3,
            development_KOs=63, reserved_test_KOs=7, heldout_target_values_loaded=False, test_targets_read=False))
        run.json('status.json', dict(status='complete', completed_folds=3))
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc))); raise


def stratified_order(ids, seed, label):
    ids = np.asarray(ids); libraries = np.array([str(k).split(':', 1)[0] for k in ids])
    order = np.arange(len(ids))
    for lib in sorted(set(libraries)):
        ix = np.flatnonzero(libraries == lib)
        if len(ix) < 2: continue
        token = int.from_bytes(hashlib.sha256((str(label)+'|'+lib).encode()).digest()[:4], 'little')
        rng = np.random.default_rng(seed+token)
        while True:
            shuffled = rng.permutation(len(ix))
            if np.all(shuffled != np.arange(len(ix))): break
        order[ix] = ix[shuffled]
    return order


def graph_grid(group, plan):
    import w15o_velocity_neighborhood as neighborhood
    original = neighborhood.source_graphs(group, plan['state_candidates'], plan['graph_neighbors'])
    result = {k:original[k] for k in ARMS[:4]}; movement = []
    for seed in plan['shuffle_seeds']:
        order = stratified_order(group['cell_ids'], seed, group['label'])
        shuffled = dict(group, shuffled_velocity_program=group['velocity_program'][order])
        result[f'shuffle{seed}'] = neighborhood.source_graphs(shuffled, plan['state_candidates'],
            plan['graph_neighbors'])['shuffled_velocity_graph']
        movement.append(dict(KO=group['label'], shuffle=seed,
            moved_fraction=float(np.mean(order != np.arange(len(order))))))
    return result, movement


def train_fold(configuration, fold):
    from veloroute.gpu_policy import enforce_gpu_policy
    enforce_gpu_policy('cuda')
    import functools
    import jax
    from flax import serialization
    import w15h_traxler_program_router as base
    import w15o_velocity_neighborhood as neighborhood
    from w15i_program_router_followups import cell_programs
    from w15n_velocity_auxiliary import deployment_source
    from cellflow.utils import match_linear
    from w7_router_scaling import write_csv
    plan = read_json(configuration); root = Path(plan['root']); pack = root/'pack'/f'fold{fold}'
    c = read_json(pack/'configuration.json'); output = root/'training'/f'fold{fold}'
    inputs = [*code_inputs(configuration, plan), pack/'configuration.json', pack/'split.json',
              pack/'transform.npz', *[pack/n for n in ('train_source.npz', 'train_target.npz', 'validation_source.npz')]]
    run = Artifacts(output, seed=c['seeds'], inputs=inputs); deadline = time.monotonic()+plan['maximum_hours']*3600
    try:
        training, heldout, norm, split, _ = base.prepare(c)
        training, _ = cell_programs(training, norm, 0)
        grids = {g['label']:graph_grid(g, plan) for g in training}
        graphs = {k:v[0] for k, v in grids.items()}
        write_csv(output/'shuffle_movement.csv', [r for v in grids.values() for r in v[1]])
        np.savez_compressed(output/'source_graphs.npz', **{f'{k}__{a}':g[a] for k, g in graphs.items() for a in ARMS})
        np.savez_compressed(output/'training_programs.npz', **norm)
        run.json('study_manifest.json', dict(plan=plan, config=c, split=split, planned_fits=21,
            future_target_read=False, explicit_velocity_input=False, graph_discarded_at_deployment=True))
        model = base.create_model(c, training[0]['z'].shape[1], 0, training[0]['embedding'])
        matcher = functools.partial(match_linear, epsilon=c['ot_epsilon'], scale_cost='mean', max_iterations=c['ot_max_iterations'])
        step = neighborhood.make_step(model, matcher); integrate = base.make_candidate_integrator(model)
        records = []; done = 0; initializations = {}
        for seed in c['seeds']:
            for arm in ARMS:
                if time.monotonic() > deadline: raise TimeoutError('Fold training deadline')
                directory = output/'fits'/f'seed{seed}'/arm
                fit_run = Artifacts(directory, seed=seed, inputs=[Path(configuration), Path(__file__)])
                run.json('status.json', dict(status='training', fold=fold, seed=seed, arm=arm, completed_fits=done, total_fits=21))
                params, initial, history = neighborhood.fit(model, step, training, graphs, c, plan, arm, seed,
                    lambda row:fit_run.json('status.json', dict(status='training', **row)))
                digest = hashlib.sha256(serialization.to_bytes(initial)).hexdigest()
                if seed in initializations and initializations[seed] != digest: raise ValueError('Matched initialization changed')
                initializations[seed] = digest
                parameters = sum(v.size for v in jax.tree_util.tree_leaves(params))
                if parameters != 389704: raise ValueError('Fixed model parameter count changed')
                checkpoint = directory/'model.msgpack'; checkpoint.write_bytes(serialization.to_bytes(params))
                params = serialization.from_bytes(initial, checkpoint.read_bytes()); model_hash = sha256(checkpoint)
                for source in heldout:
                    if Path(source['label']).name != source['label']: raise ValueError('Unsafe KO identity')
                    prediction = base.predict(model, integrate, params, deployment_source(source, c['programs']), c)
                    path = directory/(source['label']+'.npz'); np.savez_compressed(path, **prediction)
                    records.append(dict(fold=fold, KO=source['label'], seed=seed, arm=arm, path=str(path),
                        sha256=sha256(path), model=str(checkpoint), model_sha256=model_hash,
                        heldout_future_used=False, explicit_velocity_input=False))
                fit_run.json('training_history.json', dict(history=history))
                fit_run.json('summary.json', dict(status='complete', seed=seed, arm=arm, steps=c['steps'],
                    parameters=parameters, initial_parameters_sha256=digest, model_sha256=model_hash,
                    graph_js_first=history[0]['graph_js'], graph_js_last=history[-1]['graph_js']))
                fit_run.json('status.json', dict(status='complete')); done += 1
                run.json('status.json', dict(status='training', completed_fits=done, total_fits=21, fold=fold))
                print(json.dumps(dict(status='cv_fit_complete', fold=fold, seed=seed, arm=arm, completed=done, total=21)), flush=True)
        run.json('frozen_grid.json', dict(predictions=records, heldout_target_read=False, test_targets_read=False))
        run.json('summary.json', dict(status='CV_FOLD_PREDICTIONS_FROZEN', fits=21, predictions=len(records),
            parameters=389704, heldout_target_read=False, test_targets_read=False, GFG=False, JVP=False))
        run.json('status.json', dict(status='complete', completed_fits=21, total_fits=21))
    except Exception as exc:
        run.json('status.json', dict(status='failed', completed_fits=locals().get('done', 0), error=repr(exc))); raise


def exact_metrics(candidates, q, target, axes):
    from scipy.spatial.distance import cdist
    x = np.asarray(candidates, float).reshape(-1, target.shape[1]); y = np.asarray(target, float)
    w = np.asarray(q, float).ravel()/len(q)
    np.testing.assert_allclose(w.sum(), 1., atol=1e-6)
    if (w < 0).any() or not np.isfinite(x).all(): raise ValueError('Invalid empirical mixture')
    w /= w.sum(); mean = w@x; truth = y.mean(0); scale = np.sqrt(x.shape[1])
    self_term = float(w@cdist(x, x)@w)+float(cdist(y, y).mean())
    energy = float(2*w@cdist(x, y).mean(1)-self_term)/scale
    centered = float(2*w@cdist(x-mean, y-truth).mean(1)-self_term)/scale
    variance = max(float(w@(x*x).sum(1)-mean@mean), 0.)
    ratio = variance/max(float(y.var(0).sum()), 1e-12)
    return dict(energy=max(energy, 0.), centered_energy=max(centered, 0.),
        mean_mse=float(np.mean((mean-truth)**2)), program_mean_mse=float(np.mean(((mean-truth)@axes.T)**2)),
        variance_ratio=ratio, log_variance_error=abs(float(np.log(max(ratio, 1e-12)))))


def summarize(rows, plan):
    by = {(r['KO'], r['seed'], r['arm']):r for r in rows}
    labels = sorted({r['KO'] for r in rows}); results = []
    for control in ('no_graph', 'state_graph', 'U_graph', 'shuffle_mean', 'shuffle0', 'shuffle1', 'shuffle2'):
        values = []; per_seed = {}; per_fold = {}
        for ko in labels:
            gains = []
            for seed in plan['model_seeds']:
                real = by[(ko, seed, 'velocity_graph')]
                other = np.mean([by[(ko, seed, f'shuffle{i}')]['energy'] for i in plan['shuffle_seeds']]) if control == 'shuffle_mean' else by[(ko, seed, control)]['energy']
                gain = 1-real['energy']/max(other, 1e-12); gains.append(gain)
                per_seed.setdefault(str(seed), []).append(gain)
                per_fold.setdefault(str(real['fold']), []).append(gain)
            values.append(float(np.mean(gains)))
        values = np.asarray(values); rng = np.random.default_rng(plan['split_seed'])
        draws = values[rng.integers(len(values), size=(plan['bootstrap_repetitions'], len(values)))].mean(1)
        a = (1-plan['confidence'])/2; low, high = np.quantile(draws, [a, 1-a])
        results.append(dict(control=control, energy_gain=float(values.mean()), ci_low=float(low), ci_high=float(high),
            winning_KOs=int((values > 0).sum()), KOs=len(values),
            by_seed={k:float(np.mean(v)) for k, v in per_seed.items()},
            by_fold={k:float(np.mean(v)) for k, v in per_fold.items()}))
    return results


def evaluate(configuration):
    import pandas as pd
    from veloroute.latent import FrozenSplicingTransform
    from w7_router_scaling import write_csv
    plan = read_json(configuration); root = Path(plan['root']); records = []; expected = set()
    for fold in range(plan['folds']):
        directory = root/'training'/f'fold{fold}'
        if read_json(directory/'status.json')['status'] != 'complete': raise ValueError('All folds must complete before scoring')
        records.extend(read_json(directory/'frozen_grid.json')['predictions'])
        split = read_json(root/'pack'/f'fold{fold}'/'split.json')
        expected |= {(fold, s, a, ko) for s in plan['model_seeds'] for a in ARMS for ko in split['val']}
    if {(r['fold'], r['seed'], r['arm'], r['KO']) for r in records} != expected or len(records) != len(expected):
        raise ValueError('Incomplete fixed prediction grid')
    models = {}
    for r in records:
        if r['heldout_future_used'] or sha256(r['path']) != r['sha256']: raise ValueError('Unfrozen prediction')
        models[r['model']] = r['model_sha256']
    if len(models) != plan['planned_fits']: raise ValueError('Wrong number of models')
    for path, digest in models.items():
        if sha256(path) != digest: raise ValueError('Unfrozen model')
    run = Artifacts(root/'evaluation', seed=plan['split_seed'], inputs=[*code_inputs(configuration, plan),
        root/'prepare'/'summary.json'])
    # The per-fold manifests have identical basenames. Hash them into the
    # aggregate instead of colliding in the provenance snapshot directory.
    run.json('whole_grid_frozen.json', dict(predictions=records, models=len(models),
        fold_manifests={str(root/'training'/f'fold{i}'/'frozen_grid.json'):
            sha256(root/'training'/f'fold{i}'/'frozen_grid.json') for i in range(plan['folds'])},
        heldout_target_read=False, test_targets_read=False))
    frame = pd.read_csv(plan['cell_manifest']); rows = []
    try:
        for fold in range(plan['folds']):
            pack = root/'pack'/f'fold{fold}'; split = read_json(pack/'split.json')
            ids, counts, _ = load_counts(plan['h5dir'], frame_ids(frame, split['val'], plan['target_hours']), ('spliced',))
            transform = FrozenSplicingTransform.load(pack/'transform.npz')
            targets = apply_transform(transform, counts['spliced'])['z']
            lookup = dict(zip((frame['library'].astype(str)+':'+frame['barcode'].astype(str)), frame['KO'].astype(str)))
            conditions = np.array([lookup[k] for k in ids])
            with np.load(root/'training'/f'fold{fold}'/'training_programs.npz', allow_pickle=False) as f: axes = f['axes']
            for ko in split['val']:
                y = targets[conditions == ko]
                rng = np.random.default_rng(0); y = y[rng.choice(len(y), min(len(y), 128), replace=False)]
                common = None
                for r in [r for r in records if r['fold'] == fold and r['KO'] == ko]:
                    with np.load(r['path'], allow_pickle=False) as f: p = {k:f[k] for k in f.files}
                    if common is None: common = p['source_ids']
                    else: np.testing.assert_array_equal(common, p['source_ids'])
                    average = p['full_source_q'].astype(float).mean(0)
                    rows.append(dict(fold=fold, KO=ko, seed=r['seed'], arm=r['arm'],
                        effective_experts=float(np.exp(-np.sum(average*np.log(np.maximum(average, 1e-12))))),
                        mean_cell_entropy=float(-np.mean(np.sum(p['full_source_q']*np.log(np.maximum(p['full_source_q'], 1e-12)), axis=1))),
                        **exact_metrics(p['candidates'], p['q'], y, axes)))
            run.json('status.json', dict(status='evaluating', completed_folds=fold+1))
        comparisons = summarize(rows, plan)
        primary = [r for r in comparisons if r['control'] in plan['primary_controls']]
        supported = all(r['ci_low'] > 0 and all(v > 0 for v in r['by_seed'].values()) for r in primary)
        names = ('energy', 'centered_energy', 'mean_mse', 'program_mean_mse', 'variance_ratio', 'log_variance_error', 'effective_experts')
        result = dict(status='TRAXLER_FIXED_GRAPH_CV_COMPLETE', fits=len(models), predictions=len(records),
            KOs=len({r['KO'] for r in rows}), comparisons=comparisons, relationship_candidate_supported=supported,
            means={a:{k:float(np.mean([r[k] for r in rows if r['arm'] == a])) for k in names} for a in ARMS},
            test_targets_read=False, independent_confirmation=False, physical_direction_claim=False,
            GFG=False, JVP=False, source_hours=6, target_hours=24, parameters=389704,
            aggregation='KO_equal_after_seed_averaging', shuffle_scope=plan['shuffle_scope'])
        write_csv(root/'evaluation'/'metrics.csv', rows)
        write_csv(root/'evaluation'/'comparisons.csv', [{k:v for k, v in r.items() if k not in ('by_seed', 'by_fold')} for r in comparisons])
        run.json('summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Fixed Traxler perturbation graph CV\n\n'+json.dumps(result, indent=2)+'\n')
        run.json('status.json', dict(status='complete')); return result
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc))); raise


def gpu_free(index):
    from veloroute.gpu_policy import gpu_uuids
    uuid = gpu_uuids()[index]
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True, timeout=20)
    return uuid not in result


def dispatch(configuration):
    from w15i_followup_queue import worker_environment
    plan = read_json(configuration); root = Path(plan['root'])
    if tuple(plan['arms']) != ARMS or plan['folds'] != 3 or plan['planned_fits'] != 63:
        raise ValueError('Fixed grid changed')
    run = Artifacts(root/'controller', seed=plan['split_seed'], inputs=code_inputs(configuration, plan))
    try:
        prepare(configuration); launched = []; deadline = time.monotonic()+plan['maximum_hours']*3600
        for fold, gpu in enumerate(plan['gpus']):
            while not gpu_free(gpu):
                if time.monotonic() > deadline: raise TimeoutError('Waiting for approved GPU')
                run.json('status.json', dict(status='waiting_for_gpu', fold=fold, physical_gpu=gpu)); time.sleep(15)
            job = Path(plan['jobs_root'])/f'fold{fold}_gpu{gpu}'
            command = [sys.executable, str(HERE/'launch_gpu_job.py'), '--gpu', str(gpu), '--script',
                str(Path(__file__).resolve()), '--output', str(job), '--', '--configuration',
                str(Path(configuration).resolve()), '--stage', 'train', '--fold', str(fold)]
            receipt = json.loads(subprocess.check_output(command, cwd=ROOT,
                env=worker_environment(root/'runtime'/f'gpu{gpu}'), text=True, timeout=45))
            launched.append(receipt); print(json.dumps(receipt), flush=True)
        run.json('launch_receipts.json', dict(jobs=launched, planned_fits=63))
        while True:
            states = [read_json(Path(r['job_directory'])/'status.json') if (Path(r['job_directory'])/'status.json').exists() else {} for r in launched]
            if any(s.get('status') == 'failed' for s in states): raise RuntimeError('Fold worker failed; inspect preserved logs')
            if all(s.get('status') == 'complete' for s in states): break
            if time.monotonic() > deadline: raise TimeoutError('CV controller deadline')
            run.json('status.json', dict(status='training', jobs=states, planned_fits=63)); time.sleep(20)
        result = evaluate(configuration)
        run.json('summary.json', result); run.json('status.json', dict(status='complete'))
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc))); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--configuration', required=True)
    parser.add_argument('--stage', choices=('dispatch', 'prepare', 'train', 'evaluate'), required=True)
    parser.add_argument('--fold', type=int, choices=range(3)); args = parser.parse_args()
    if args.stage == 'train':
        if args.fold is None: parser.error('--train requires --fold')
        train_fold(args.configuration, args.fold)
    else: globals()[args.stage](args.configuration)
