"""Training-only replicated density screening; clustering is not a mode test."""
import json
import warnings

import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .artifacts import Run, save_csv, save_json
from .latent import load_pack


def density_support(train, test, config, seed):
    a, b = np.asarray(train).reshape(-1, 1), np.asarray(test).reshape(-1, 1)
    one = GaussianMixture(1, n_init=2, reg_covar=1e-5, random_state=seed).fit(a)
    two = GaussianMixture(2, n_init=3, reg_covar=1e-5, random_state=seed).fit(a)
    mean = two.means_.ravel()
    sd = np.sqrt(two.covariances_.ravel())
    grid = np.linspace(min(mean-4*sd), max(mean+4*sd), 1024)[:, None]
    density = np.exp(two.score_samples(grid))
    peaks = len(find_peaks(density, prominence=.01*density.max())[0])
    separation = abs(mean[0]-mean[1])/np.sqrt((sd**2).mean())
    bic = one.bic(a)-two.bic(a)
    heldout = two.score(b)-one.score(b)
    support = (peaks == 2 and bic >= config['bic_gain_minimum'] and heldout > 0
               and separation >= config['ashman_separation_minimum']
               and two.weights_.min() >= config['minimum_component_fraction'])
    return dict(supported=bool(support), density_peaks=peaks, bic_gain=float(bic),
                heldout_loglik_gain=float(heldout), separation=float(separation),
                smaller_component=float(two.weights_.min()))


def audit_training_multimodality(fold, config, output):
    from pathlib import Path
    fold = Path(fold)
    source, sm = load_pack(fold/'train_source.npz', expected_side='source')
    target, tm = load_pack(fold/'train_target.npz', expected_side='target')
    if sm['role'] != 'train' or tm['role'] != 'train':
        raise ValueError('Multimodal candidate discovery is training-only')
    with Run(output, stage='training_multimodal_candidate_screen', kind='engineering', config=config,
             inputs=[fold/'train_source.npz', fold/'train_target.npz'], seed=config['seed']) as run:
        rng = np.random.default_rng(config['seed'])
        rows, summaries = [], []
        for label in sorted(set(target['conditions'])):
            z = target['z'][target['conditions'] == label]
            s = source['z'][source['conditions'] == label]
            if len(z)//2 < config['minimum_cells_per_half']:
                summaries.append(dict(condition=label, cells=len(z), supported=False, best_axis='insufficient_cells', stability=0.))
                continue
            axis_votes = {axis: [] for axis in config['axes']}
            for split in range(config['splits']):
                order = rng.permutation(len(z))
                a, b = z[order[:len(z)//2]], z[order[len(z)//2:]]
                pca = PCA(2).fit(a)
                effect = a.mean(0)-s.mean(0)
                effect /= max(np.linalg.norm(effect), 1e-8)
                vectors = {'local_pc1': pca.components_[0], 'local_pc2': pca.components_[1], 'effect_direction': effect}
                for name in config['axes']:
                    result = density_support(a@vectors[name], b@vectors[name], config, split)
                    axis_votes[name].append(result['supported'])
                    rows.append(dict(condition=label, split=split, axis=name, **result))
            stability = {name: float(np.mean(votes)) for name, votes in axis_votes.items()}
            best = max(stability, key=stability.get)
            summaries.append(dict(condition=label, cells=len(z), supported=stability[best] >= config['minimum_split_support'],
                                  best_axis=best, stability=stability[best]))
        # One-dimensional standard-Gaussian sanity check, NOT condition-matched null calibration.
        nulls = []
        for simulation in range(config['null_simulations']):
            x = rng.normal(size=(100, 1))
            result = density_support(x[:50], x[50:], config, simulation)
            nulls.append(dict(simulation=simulation, **result))
        save_csv(run.directory/'axis_splits.csv', rows)
        save_csv(run.directory/'conditions.csv', summaries)
        save_csv(run.directory/'gaussian_null_checks.csv', nulls)
        result = dict(status='TRAINING_MULTIMODAL_SCREEN_COMPLETE', conditions=len(summaries),
            supported_conditions=[r['condition'] for r in summaries if r['supported']],
            gaussian_null_positive_fraction=float(np.mean([r['supported'] for r in nulls])),
            formal_D4=False, confirmation_evaluated=False,
            limitations=['Replicated density screen, not a calibrated omnibus/FDR test.',
                        'Guide/editing/cell-cycle confounds are not excluded by this screen.'])
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Training-only multimodal candidates\n\n'+json.dumps(result, indent=2))
    return result
