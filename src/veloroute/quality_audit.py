"""Training-only kinetic/guide quality descriptions, never a formal GO flag."""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse, stats

from .artifacts import Run, load_config, object_hash, read_csv, save_csv, save_json
from .latent import FrozenSplicingTransform


def cosine_rows(a, b):
    denominator = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    return np.divide((a*b).sum(-1), denominator, out=np.zeros_like(denominator), where=denominator > 1e-12)


def thinning(counts, probability, rng):
    value = counts.tocsr(copy=True)
    rounded = np.rint(value.data)
    if not np.allclose(value.data, rounded) or np.any(rounded < 0):
        raise ValueError('Molecule thinning needs nonnegative integer counts')
    value.data = rng.binomial(rounded.astype(np.int64), probability).astype(value.dtype)
    value.eliminate_zeros()
    return value


def run_training_quality(config_path, output):
    config = load_config(config_path)
    if config['roles'] != ['train'] or config['days'] != [4, 5] or not 0 < config['thinning_probability'] < 1:
        raise ValueError('This audit is training-only, day4/day5, with bounded thinning')
    release, fold = Path(config['release']), Path(config['fold'])
    inputs = [config_path, fold/'transform.npz', fold/'fit_cell_ids.csv']+[release/f'day{day}/day{day}.h5ad' for day in config['days']]
    with Run(output, stage='training_only_quality_audit', kind='engineering', config=config, inputs=inputs, seed=20260913) as run:
        transform = FrozenSplicingTransform.load(fold/'transform.npz')
        parts = []
        for day in config['days']:
            data = ad.read_h5ad(release/f'day{day}/day{day}.h5ad')
            part = data[(data.obs.role == 'train') & data.obs.technical_qc_pass & data.obs.guide_call_pass].copy()
            del data
            if tuple(part.var_names) != transform.gene_ids:
                raise ValueError('Audit reference gene order mismatch')
            parts.append(part)
        train = ad.concat(parts, join='inner', merge='same')
        fit_ids = {row['cell_id'] for row in read_csv(fold/'fit_cell_ids.csv')}
        if set(train.obs_names) != fit_ids or object_hash(sorted(fit_ids)) != transform.fit_ids_hash:
            raise ValueError('Audit cells differ from frozen training-only fit scope')
        s, u = train.layers['spliced'].tocsr(), train.layers['unspliced'].tocsr()
        def project(s_counts, u_counts):
            values = [transform.transform(s_counts[start:start+128], u_counts[start:start+128], gene_ids=transform.gene_ids)
                      for start in range(0, len(train), 128)]
            return {key: np.concatenate([value[key] for value in values]) for key in ('z', 'velocity')}
        original = project(s, u)
        stability_rows = []
        for seed in config['thinning_seeds']:
            rng = np.random.default_rng(seed)
            sampled = project(thinning(s, config['thinning_probability'], rng), thinning(u, config['thinning_probability'], rng))
            cosines = cosine_rows(original['velocity'], sampled['velocity'])
            for day in config['days']:
                for label in sorted(set(train.obs.condition)):
                    ix = ((train.obs.day == day) & (train.obs.condition == label)).to_numpy()
                    stability_rows.append({'day': day, 'condition': label, 'seed': seed, 'cells': int(ix.sum()),
                        'median_velocity_direction_cosine': float(np.median(cosines[ix])),
                        'fraction_positive_direction_cosine': float(np.mean(cosines[ix] > 0))})
        save_csv(run.directory/'thinning_stability.csv', stability_rows)
        norm = np.linalg.norm(original['velocity'], axis=1)
        technical = []
        for day in config['days']:
            ix = (train.obs.day == day).to_numpy()
            for feature in ('spliced_umi', 'unspliced_umi', 'mito_fraction', 'guide_top_fraction'):
                values = train.obs[feature].to_numpy()[ix]
                association = stats.spearmanr(norm[ix], values).statistic
                technical.append({'day': day, 'feature': feature, 'velocity_norm_spearman': float(association)})
        save_csv(run.directory/'technical_associations.csv', technical)
        scale = transform.target_sum/np.asarray(s.sum(1)).ravel()
        sf = s[:, transform.selected].toarray()*scale[:, None]
        uf = u[:, transform.selected].toarray()*scale[:, None]
        residual = uf-sf*transform.gamma
        r2 = 1-residual.var(0)/np.maximum(uf.var(0), 1e-12)
        mse_r2 = 1-np.mean(residual**2, axis=0)/np.maximum(uf.var(0), 1e-12)
        genes = np.array(transform.gene_ids)[transform.selected]
        phase = [{'gene_id': gene, 'gamma': float(transform.gamma[j]), 'supported': bool(transform.supported[j]),
                  'unspliced_positive_cells': int(np.count_nonzero(uf[:, j])), 'training_variance_explained': float(r2[j]),
                  'training_zero_intercept_R2': float(mse_r2[j])} for j, gene in enumerate(genes)]
        save_csv(run.directory/'training_gene_phase_support.csv', phase)
        # Predeclared choice: top training S-variance genes, not the nicest-looking phase portraits.
        selected_portraits = np.arange(min(config['phase_portrait_genes'], len(genes)))
        with (run.directory/'phase_portraits_data.npz').open('xb') as stream:
            np.savez_compressed(stream, S=sf[:, selected_portraits], U=uf[:, selected_portraits],
                                gamma=transform.gamma[selected_portraits], gene_ids=genes[selected_portraits])
        time_rows, guide_rows = [], []
        for label in sorted(set(train.obs.condition)):
            ia = ((train.obs.condition == label) & (train.obs.day == 4)).to_numpy()
            ib = ((train.obs.condition == label) & (train.obs.day == 5)).to_numpy()
            change = original['z'][ib].mean(0)-original['z'][ia].mean(0)
            source_v = original['velocity'][ia].mean(0)
            time_rows.append({'condition': label, 'source_cells': int(ia.sum()), 'target_cells': int(ib.sum()),
                              'training_mean_velocity_next_change_cosine': float(cosine_rows(source_v[None], change[None])[0]),
                              'independent_future_anchor': False})
            guide_changes, names, counts = [], [], []
            for guide in sorted(set(train.obs.loc[train.obs.condition == label, 'guide_name'])):
                ga = ia & (train.obs.guide_name == guide).to_numpy()
                gb = ib & (train.obs.guide_name == guide).to_numpy()
                if min(ga.sum(), gb.sum()) >= config['minimum_cells_per_guide_day']:
                    guide_changes.append(original['z'][gb].mean(0)-original['z'][ga].mean(0))
                    names.append(guide); counts.append([int(ga.sum()), int(gb.sum())])
            if len(guide_changes) == 2:
                guide_rows.append({'condition': label, 'guide_a': names[0], 'guide_b': names[1],
                    'a_day4': counts[0][0], 'a_day5': counts[0][1], 'b_day4': counts[1][0], 'b_day5': counts[1][1],
                    'training_guide_time_change_cosine': float(cosine_rows(guide_changes[0][None], guide_changes[1][None])[0])})
        save_csv(run.directory/'training_time_alignment.csv', time_rows)
        save_csv(run.directory/'training_guide_concordance.csv', guide_rows,
                 fields=['condition', 'guide_a', 'guide_b', 'a_day4', 'a_day5', 'b_day4', 'b_day5', 'training_guide_time_change_cosine'])
        summary = {'status': 'TRAINING_QUALITY_DESCRIPTION_COMPLETE', 'train_cells': len(train),
                   'supported_velocity_genes': int(transform.supported.sum()),
                   'median_gene_zero_intercept_R2_supported': float(np.median(mse_r2[transform.supported])),
                   'median_condition_seed_thinning_direction_cosine': float(np.median([row['median_velocity_direction_cosine'] for row in stability_rows])),
                   'median_training_time_alignment_cosine': float(np.median([row['training_mean_velocity_next_change_cosine'] for row in time_rows])),
                   'guide_comparable_conditions': len(guide_rows),
                   'median_guide_time_change_cosine': float(np.median([row['training_guide_time_change_cosine'] for row in guide_rows])) if guide_rows else None,
                   'formal_QC_approved': False, 'confirmation_evaluated': False,
                   'limitations': ['Thinning measures estimator robustness, not correctness of biological velocity.',
                                  'Training day5 entered the original velocity fit; time alignment is not an independent anchor.',
                                  'Guide concordance is not editing validation or a complete doublet audit.']}
        save_json(run.directory/'summary.json', summary)
        (run.directory/'RESULTS.md').write_text('# Training-only quality audit\n\n'+json.dumps(summary, indent=2)
                                               +'\n\n'+json.dumps(technical, indent=2)+'\n')
    return summary
