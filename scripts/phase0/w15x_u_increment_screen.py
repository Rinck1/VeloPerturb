"""Nested train-KO screening of U information beyond mature state and c.

Only population response labels are available. No single-cell fate target,
future development/test input, GFG or router is constructed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from w4_artifacts import Artifacts, sha256

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ('positive_kinetic_A', 'free_kinetic_A', 'free_log_U')


def keyed_seed(seed, identity):
    return int(seed)+int.from_bytes(hashlib.sha256(str(identity).encode()).digest()[:4], 'little')


def derangement(n, seed):
    if n < 2: raise ValueError('At least two KO residuals required for shuffling')
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(n)
        if np.all(order != np.arange(n)): return order


def pca_context(values, train, query, components):
    x = np.asarray(values, dtype='float64')
    mean = x[train].mean(0)
    _, _, axes = np.linalg.svd(x[train]-mean, full_matrices=False)
    axes = axes[:min(components, len(train)-1, x.shape[1])]
    return (x[train]-mean)@axes.T, (x[query]-mean)@axes.T


def designs(data, train, query, plan):
    train, query = np.asarray(train), np.asarray(query)
    st, sq = pca_context(data['z0'], train, query, plan['source_PCs'])
    ct, cq = pca_context(data['esm'], train, query, plan['condition_PCs'])
    def raw(indices, state, condition):
        shared = np.concatenate((state, condition, data['technical'][indices]), axis=-1)
        return np.concatenate((data['local'][indices],
            np.broadcast_to(shared[:, None], (len(indices), data['local'].shape[1], shared.shape[1]))), axis=-1)
    xt, xq = raw(train, st, ct), raw(query, sq, cq)
    mean, scale = xt.mean(0), np.maximum(xt.std(0), 1e-6)
    add_intercept = lambda value: np.concatenate((np.ones((*value.shape[:2], 1)), (value-mean)/scale), axis=-1)
    return add_intercept(xt), add_intercept(xq)


def nuisance_fit(data, train, query, plan):
    """No query response is touched when producing nuisance predictions."""
    xt, xq = designs(data, train, query, plan)
    targets = np.stack((data['y'][train], data['A'][train], data['log_U'][train]), axis=-1)
    penalty = np.eye(xt.shape[-1])*plan['ridge']; penalty[0, 0] = 0.
    gram = np.einsum('ngp,ngq->gpq', xt, xt, optimize=True)+penalty[None]
    rhs = np.einsum('ngp,ngt->gpt', xt, targets, optimize=True)
    coef = np.linalg.solve(gram, rhs)
    # Ridge shrinkage of g(S,c) must not manufacture an apparent U increment
    # when U is exactly determined by the included mature-state features.
    inverse = np.linalg.pinv(xt.transpose(1, 0, 2), rcond=1e-10)
    coef[:, :, 1:] = np.einsum('gpn,ngt->gpt', inverse, targets[:, :, 1:], optimize=True)
    fitted = np.einsum('ngp,gpt->ngt', xt, coef, optimize=True)
    predicted = np.einsum('ngp,gpt->ngt', xq, coef, optimize=True)
    return dict(base_train=fitted[:, :, 0], base_query=predicted[:, :, 0],
        response_residual=data['y'][train]-fitted[:, :, 0],
        A_variance=np.var(data['A'][train], axis=0), log_U_variance=np.var(data['log_U'][train], axis=0),
        A_train=data['A'][train]-fitted[:, :, 1], A_query=data['A'][query]-predicted[:, :, 1],
        log_U_train=data['log_U'][train]-fitted[:, :, 2], log_U_query=data['log_U'][query]-predicted[:, :, 2])


def incremental_fit(nuisance, family, plan, identity):
    feature = 'log_U' if family == 'free_log_U' else 'A'
    x, xq = nuisance[feature+'_train'], nuisance[feature+'_query']
    residual = nuisance['response_residual']
    positive = family == 'positive_kinetic_A'
    def coefficient(values):
        scale2 = np.maximum(np.mean(values**2, axis=0), 1e-12)
        beta = np.sum(values*residual, axis=0)/(np.sum(values**2, axis=0)+plan['ridge']*scale2)
        active = np.mean(values**2, axis=0) > np.maximum(1e-12,
            plan['minimum_innovation_fraction']*nuisance[feature+'_variance'])
        beta = np.where(active, beta, 0.)
        return np.maximum(beta, 0.) if positive else beta
    beta = coefficient(x)
    predictions = {'real': nuisance['base_query']+xq*beta}
    for seed in plan['shuffle_seeds']:
        it = derangement(len(x), keyed_seed(seed, identity+'|train'))
        iq = derangement(len(xq), keyed_seed(seed, identity+'|query'))
        b = coefficient(x[it])
        predictions[f'shuffle_{seed}'] = nuisance['base_query']+xq[iq]*b
    return predictions, beta


def split_indices(indices, folds, seed):
    from sklearn.model_selection import KFold
    indices = np.asarray(indices)
    return [(indices[t], indices[v]) for t, v in KFold(folds, shuffle=True, random_state=seed).split(indices)]


def selection_statistics(data, indices, plan, cache=None):
    """Nested selector only ever scores responses INSIDE supplied indices."""
    indices = np.asarray(indices); genes = data['y'].shape[1]
    names = ['S', 'real']+[f'shuffle_{s}' for s in plan['shuffle_seeds']]
    total = {family: {name: np.zeros(genes) for name in names} for family in FAMILIES}
    wins = {family: np.zeros(genes, int) for family in FAMILIES}
    signs = {family: np.zeros(genes, int) for family in FAMILIES}
    cache = {} if cache is None else cache
    for fold, (train, query) in enumerate(split_indices(indices, plan['inner_folds'], plan['seed'])):
        key = (tuple(train), tuple(query))
        if key not in cache: cache[key] = nuisance_fit(data, train, query, plan)
        nuisance = cache[key]
        base_error = np.sum((nuisance['base_query']-data['y'][query])**2, axis=0)
        for family in FAMILIES:
            predictions, beta = incremental_fit(nuisance, family, plan, f'inner|{fold}|{tuple(indices)}')
            total[family]['S'] += base_error
            for name, prediction in predictions.items():
                total[family][name] += np.sum((prediction-data['y'][query])**2, axis=0)
            real_error = np.sum((predictions['real']-data['y'][query])**2, axis=0)
            wins[family] += real_error < base_error
            signs[family] += beta > 0
    result = {}
    for family in FAMILIES:
        errors = total[family]
        gain = 1-errors['real']/np.maximum(errors['S'], 1e-12)
        eligible = data['supported'] & (errors['S'] > 1e-10) & (gain >= plan['minimum_inner_gain'])
        eligible &= wins[family] >= plan['minimum_winning_inner_folds']
        for seed in plan['shuffle_seeds']: eligible &= errors['real'] < errors[f'shuffle_{seed}']
        if family == 'positive_kinetic_A': eligible &= signs[family] >= plan['minimum_winning_inner_folds']
        result[family] = dict(mask=eligible, gain=gain, winning_folds=wins[family], positive_coefficient_folds=signs[family], errors=errors)
    return result


def program_axes(residual, mask, programs):
    n = int(np.sum(mask))
    if not n: return np.empty((0, len(mask)))
    value = residual[:, mask]-residual[:, mask].mean(0)
    _, singular, axes = np.linalg.svd(value, full_matrices=False)
    count = min(programs, n, len(value)-1, int(np.sum(singular > 1e-10)))
    result = np.zeros((count, len(mask))); result[:, mask] = axes[:count]
    return result


def score_fold(data, query, nuisance, predictions, mask, axes, plan, family, fold):
    rows = []; count = int(mask.sum())
    y = data['y'][query]
    train_variance = nuisance['target_variance']
    for arm, predicted in dict(S=nuisance['base_query'], **predictions).items():
        delta = predicted-y
        for i, ko in enumerate(data['labels'][query]):
            valid = count >= plan['minimum_selected_genes']
            rows.append(dict(family=family, fold=fold, KO=str(ko), arm=arm, selected_genes=count,
                selected_standardized_mse=float(np.mean(delta[i, mask]**2/train_variance[mask])) if valid else None,
                all_gene_mse=float(np.mean(delta[i]**2)),
                terminal_mean_mse=float(np.mean((18*delta[i]@data['components'].T)**2)),
                program_mse=float(np.mean((delta[i]@axes.T)**2)) if valid and len(axes) else None))
    return rows


def bootstrap_comparison(real, control, plan):
    real, control = np.asarray(real, float), np.asarray(control, float)
    gain = 1-real.mean()/max(control.mean(), 1e-12)
    rng = np.random.default_rng(plan['seed'])
    ix = rng.integers(len(real), size=(plan['bootstrap_repetitions'], len(real)))
    draws = 1-real[ix].mean(1)/np.maximum(control[ix].mean(1), 1e-12)
    alpha = (1-plan['confidence'])/2
    low, high = np.quantile(draws, [alpha, 1-alpha])
    return dict(gain=float(gain), mean_paired_gain=float(np.mean(1-real/np.maximum(control, 1e-12))),
                ci_low=float(low), ci_high=float(high), winning_KOs=int(np.sum(real < control)))


def summarize(rows, plan):
    results = []
    for family in FAMILIES:
        selected = [r for r in rows if r['family'] == family and r['selected_standardized_mse'] is not None]
        by = {(r['KO'], r['arm']): r for r in selected}
        labels = sorted({r['KO'] for r in selected})
        folds = sorted({r['fold'] for r in selected})
        controls = ['S']+[f'shuffle_{s}' for s in plan['shuffle_seeds']]
        comparisons = []
        for control in controls:
            if labels:
                values = bootstrap_comparison([by[(ko, 'real')]['selected_standardized_mse'] for ko in labels],
                    [by[(ko, control)]['selected_standardized_mse'] for ko in labels], plan)
            else: values = dict(gain=None, mean_paired_gain=None, ci_low=None, ci_high=None, winning_KOs=0)
            comparisons.append(dict(control=control, **values))
        adequate = len(labels) >= plan['minimum_valid_conditions'] and len(folds) >= plan['minimum_valid_outer_folds']
        passed = adequate and all(r['ci_low'] is not None and r['ci_low'] > 0 for r in comparisons)
        other = [r for r in rows if r['family'] == family]
        full = {arm:float(np.mean([r['terminal_mean_mse'] for r in other if r['arm'] == arm]))
                for arm in ['S', 'real']+[f'shuffle_{s}' for s in plan['shuffle_seeds']]}
        results.append(dict(family=family, valid_KOs=len(labels), valid_folds=folds, adequate_support=adequate,
            predictive_screen_passed=passed, comparisons=comparisons, full_PCA_terminal_mean_mse=full))
    return results


def load_data(plan):
    from veloroute.latent import FrozenSplicingTransform, load_pack
    c = json.loads(Path(plan['base_configuration']).read_text()); fold = Path(c['fold'])
    split = json.loads(Path(c['split']).read_text())
    with np.load(plan['training_features'], allow_pickle=False) as f: values = {k:f[k] for k in f.files}
    labels = values['labels'].astype(str)
    if set(labels) != set(split['train']): raise ValueError('Only original training KOs are allowed')
    source, metadata = load_pack(fold/'train_source.npz', expected_side='source')
    if metadata.get('day') != 6 or set(source['conditions']) != set(labels): raise ValueError('Wrong training source')
    transform = FrozenSplicingTransform.load(fold/'transform.npz')
    with np.load(plan['training_counts'], allow_pickle=False) as f:
        s, u, depth = f['S'], f['U'], f['depth']
        np.testing.assert_array_equal(f['cell_ids'], source['cell_ids'])
    sn, un = s*transform.target_sum/depth[:, None], u*transform.target_sum/depth[:, None]
    logs, logu = np.log1p(sn), np.log1p(un)
    source_libraries = np.array([str(k).split(':', 1)[0] for k in source['cell_ids']])
    libraries = sorted(set(source_libraries))
    U, variance, technical = [], [], []
    for ko in labels:
        mask = source['conditions'] == ko
        U.append(logu[mask].mean(0)); variance.append(logs[mask].var(0))
        technical.append([np.log(depth[mask]).mean(), np.log(mask.sum())]+
            [float(np.mean(source_libraries[mask] == lib)) for lib in libraries[:2]])
    with np.load(c['conditions'], allow_pickle=False) as f:
        embeddings = {str(k):v for k, v in zip(f['conditions'], f['embeddings'])}
    local = np.stack((values['log_start'], values['B'], values['log_start']**2, np.asarray(variance)), axis=-1)
    return dict(values, local=local, log_U=np.asarray(U), technical=np.asarray(technical),
        esm=np.stack([embeddings[k] for k in labels]), components=transform.components,
        gene_ids=np.asarray(transform.gene_ids)[transform.selected], supported=transform.supported), c


def program(configuration, output):
    from w7_router_scaling import write_csv
    plan = json.loads(Path(configuration).read_text())
    if tuple(plan['families']) != FAMILIES: raise ValueError('Unregistered increment families')
    c = json.loads(Path(plan['base_configuration']).read_text()); fold = Path(c['fold'])
    inputs = [Path(configuration), Path(__file__), ROOT/'PLANS/w15x_u_increment_screen_20261022.md',
        Path(plan['base_configuration']), Path(plan['training_features']), Path(plan['training_counts']),
        Path(c['split']), Path(c['conditions']), fold/'train_source.npz', fold/'transform.npz']
    run = Artifacts(output, seed=plan['seed'], inputs=inputs)
    deadline = time.monotonic()+plan['maximum_seconds']; data, c = load_data(plan)
    rows, selections, cache = [], [], {}
    outer_counts = {family:np.zeros(len(data['gene_ids']), int) for family in FAMILIES}
    indices = np.arange(len(data['labels']))
    try:
        for fold_index, (train, query) in enumerate(split_indices(indices, plan['outer_folds'], plan['seed'])):
            if time.monotonic() > deadline: raise TimeoutError('Nested U increment screen deadline')
            run.json('status.json', dict(status='nested_selection', outer_fold=fold_index, completed_outer_folds=fold_index))
            selected = selection_statistics(data, train, plan, cache)
            nuisance = nuisance_fit(data, train, query, plan)
            nuisance['target_variance'] = np.maximum(data['y'][train].var(0), 1e-8)
            directory = run.directory/f'outer_fold{fold_index}'; directory.mkdir()
            for family in FAMILIES:
                result = selected[family]; mask = result['mask']; outer_counts[family] += mask
                predictions, beta = incremental_fit(nuisance, family, plan, f'outer|{fold_index}')
                # ALL arms use the training-selected mask; omitted genes use S.
                predictions = {name:np.where(mask[None], pred, nuisance['base_query']) for name, pred in predictions.items()}
                axes = program_axes(nuisance['response_residual'], mask, plan['programs'])
                np.savez_compressed(directory/(family+'.npz'), query_indices=query, training_indices=train,
                    selected_gene_mask=mask, program_axes=axes, S=nuisance['base_query'], **predictions)
                rows.extend(score_fold(data, query, nuisance, predictions, mask, axes, plan, family, fold_index))
                for g in range(len(mask)):
                    selections.append(dict(fold=fold_index, family=family, gene_id=str(data['gene_ids'][g]),
                        selected=bool(mask[g]), inner_gain=float(result['gain'][g]),
                        winning_inner_folds=int(result['winning_folds'][g]), positive_coefficient_folds=int(result['positive_coefficient_folds'][g])))
            run.json('status.json', dict(status='outer_fold_complete', completed_outer_folds=fold_index+1,
                selected_genes={f:int(selected[f]['mask'].sum()) for f in FAMILIES}))
            print(json.dumps(dict(status='U_increment_outer_fold_complete', fold=fold_index,
                selected_genes={f:int(selected[f]['mask'].sum()) for f in FAMILIES})), flush=True)
        summaries = summarize(rows, plan)
        # Frozen final candidates for FUTURE experiments; never rescore OOF on them.
        final = selection_statistics(data, indices, plan, cache)
        full_nuisance = nuisance_fit(data, indices, indices, plan)
        gene_rows, final_counts = [], {}
        for family in FAMILIES:
            result = final[family]
            mask = result['mask'] & (outer_counts[family] >= plan['minimum_final_selection_folds'])
            final_counts[family] = int(mask.sum())
            axes = program_axes(full_nuisance['response_residual'], mask, plan['programs'])
            np.savez_compressed(run.directory/(family+'_candidate_programs.npz'), gene_ids=data['gene_ids'],
                selected_gene_mask=mask, program_axes=axes, selection_fold_counts=outer_counts[family])
            for g, gene in enumerate(data['gene_ids']):
                gene_rows.append(dict(family=family, gene_id=str(gene), selected=bool(mask[g]),
                    training_inner_gain=float(result['gain'][g]), outer_selection_count=int(outer_counts[family][g]),
                    winning_inner_folds=int(result['winning_folds'][g])))
        signal = next(r for r in summaries if r['family'] == 'positive_kinetic_A')
        result = dict(status='NESTED_TRAINING_U_INCREMENT_SCREEN_COMPLETE', training_KOs=len(indices), genes=len(data['gene_ids']),
            summaries=summaries, final_candidate_counts=final_counts,
            candidate_for_velocity_testing=signal['predictive_screen_passed'] and final_counts['positive_kinetic_A'] >= plan['minimum_selected_genes'],
            validation_targets_read=False, test_targets_read=False, router_trained=False, GFG_trained=False,
            final_consensus_used_for_OOF_scoring=False, individual_gene_FDR_claim=False,
            scope='KO_population_time_increment_not_observed_individual_fates',
            representation_fixed_on_outer_training_fold=True)
        write_csv(run.directory/'metrics.csv', rows); write_csv(run.directory/'nested_gene_selections.csv', selections)
        write_csv(run.directory/'gene_candidates.csv', gene_rows)
        run.json('summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Training-only nested U increment screen\n\n'+json.dumps(result, indent=2)+'\n')
        run.json('status.json', dict(status='complete', completed_outer_folds=plan['outer_folds']))
        return result
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc))); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--configuration', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(program(args.configuration, args.output), indent=2), flush=True)
