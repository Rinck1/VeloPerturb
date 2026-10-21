"""Source-only local U controls for L3; frozen CellFlow packs stay untouched."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from w4_artifacts import Artifacts, sha256
from w4_shorttime import read_count_rows, read_metadata
from w4_streams import SourceTransform
from w15a_cellflow_adapter import ROOT, SourceCohort, array_hash, pooled_tokens
from w15a_cellflow_train import load_npz, time_from_label


FEATURES = ('cell_kin', 'cell_directU', 'cell_kin_Umean', 'cell_kin_Uzero')


def local_features(transform, cohort, reference, *, neighbors=64):
    """Hold S, geometry, gamma and the observed reference fixed under U edits.

    Umean replaces each CURRENT source cell's raw U with its cohort mean U;
    Uzero replaces it with zero. Neither intervention modifies the baseline.
    No target argument exists, and neighborhoods never cross source cohorts.
    """
    if reference.time_hours > cohort.time_hours:
        raise ValueError('Reference population comes from the future')
    if int(neighbors) != neighbors or neighbors < 1:
        raise ValueError('Source neighbor count must be a positive integer')
    z = transform.state(cohort.s)
    count = min(int(neighbors), len(z))
    indices = NearestNeighbors(n_neighbors=count).fit(z).kneighbors(z, return_distance=False)
    graph = sparse.csr_matrix((np.ones(indices.size),
        (np.repeat(np.arange(len(z)), count), indices.ravel())), shape=(len(z), len(z)))
    depth = np.asarray(graph @ np.asarray(cohort.s.sum(1)).ravel())[:, None]
    if (depth <= 0).any():
        raise ValueError('Source neighborhood has zero spliced depth')
    factor = transform.target_sum/depth
    s = (graph @ cohort.s[:, transform.genes]).toarray()*factor
    u = (graph @ cohort.u[:, transform.genes]).toarray()*factor
    reference_tokens = pooled_tokens(transform, reference)
    u_mean = np.asarray(cohort.u[:, transform.genes].mean(0))*count*factor
    kinetic = lambda value: transform.projected_rate(s, value-transform.gamma*s)-reference_tokens['kin']
    return dict(z=z, cell_ids=cohort.cell_ids.copy(),
        cell_kin=kinetic(u),
        cell_directU=np.log1p(u) @ transform.components.T-reference_tokens['directU'],
        cell_kin_Umean=kinetic(u_mean), cell_kin_Uzero=kinetic(np.zeros_like(u)),
        neighbor_count=count, neighbor_indices_sha256=array_hash(indices),
        source_hash=cohort.fingerprint(), reference_hash=reference.fingerprint(),
        reference_key=reference.key)


def fit_direct_normalization(packs, *, allowed_training_source_keys):
    if not packs or len({p['source_key'] for p in packs}) != len(packs):
        raise ValueError('Training local-U packs must be nonempty and unique')
    keys = tuple(p['source_key'] for p in packs)
    if not set(keys) <= set(allowed_training_source_keys):
        raise ValueError('Local-U normalization contains a forbidden source cohort')
    values = np.concatenate([p['cell_directU'] for p in packs])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite local-U training features')
    return values.mean(0), np.maximum(values.std(0), 1e-5), keys


def unique_records(manifest):
    result = []
    for role, name in (('training_source', 'training_groups'),
                       ('development_source', 'development_sources')):
        seen = {}
        for row in manifest[name]:
            key = row['source_key']
            identity = (row['path'], row['sha256'], row['replicate'], row['label'], row['time_hours'])
            if key in seen:
                if seen[key] != identity:
                    raise ValueError('Repeated source key points to different cohorts')
                continue
            seen[key] = identity
            result.append(dict(row, role=role))
    if len({r['source_key'] for r in result}) != len(result):
        raise ValueError('Training and development source cohorts overlap')
    return result


def prepare(configuration_path, parent_pack, output):
    configuration_path, parent_pack = Path(configuration_path), Path(parent_pack)
    config = json.loads(configuration_path.read_text())
    manifest_path = parent_pack/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source_data = ROOT/config['data_path']
    if sha256(source_data) != config['data_sha256']:
        raise ValueError('Raw source data changed')
    transform_path = ROOT/config['transform_path']
    if sha256(transform_path) != manifest['source_transform_sha256']:
        raise ValueError('Frozen source transform changed')
    if sha256(parent_pack/'token_normalizer.npz') != manifest['token_normalizer_sha256']:
        raise ValueError('Original training token normalization changed')
    transform = SourceTransform.load(transform_path)
    obs, genes = read_metadata(source_data)
    if not np.array_equal(np.asarray(genes, dtype=str), transform.gene_names):
        raise ValueError('Source gene order differs from the frozen transform')
    if not obs.index.is_unique:
        raise ValueError('Raw source cell IDs are not unique')
    lookup = {str(key): i for i, key in enumerate(obs.index)}
    records = unique_records(manifest)
    by_group = {(r['replicate'], r['label']): r for r in records}
    run = Artifacts(output, seed=0, inputs=[Path(__file__), configuration_path,
        manifest_path, transform_path, parent_pack/'token_normalizer.npz'])
    baselines = {}

    def read_cohort(row):
        if sha256(row['path']) != row['sha256']:
            raise ValueError('Parent source pack changed')
        # Read ONLY identity and source state/kinetics, never a parent target array.
        with np.load(row['path'], allow_pickle=False) as handle:
            ids = handle['cell_ids'].astype(str)
            expected_z, expected_kin = handle['z'].copy(), handle['cell_kin'].copy()
        if len(set(ids)) != len(ids) or not set(ids) <= set(lookup):
            raise ValueError('Invalid source identities')
        indices = np.array([lookup[identity] for identity in ids])
        selected = obs.iloc[indices]
        if not ((selected['rep'].astype(str) == row['replicate']).all()
                and (selected['timepoint'].astype(str) == row['label']).all()):
            raise ValueError('Parent identity is not from the declared source cohort')
        s, u = [read_count_rows(source_data, layer, indices) for layer in ('spliced', 'unspliced')]
        condition = row['source_key'].split('|')[1]
        cohort = SourceCohort(condition, time_from_label(row['label']), row['replicate'], s, u, ids)
        if cohort.key != row['source_key'] or cohort.fingerprint() != row['source_hash']:
            raise ValueError('Raw source identity/count fingerprint changed')
        return cohort, expected_z, expected_kin

    rows, normalization_packs = [], []
    for row in records:
        baseline_label = 'D3' if row['role'] == 'training_source' else '05h'
        baseline_key = (row['replicate'], baseline_label)
        if baseline_key not in baselines:
            baselines[baseline_key] = read_cohort(by_group[baseline_key])[0]
        cohort, expected_z, expected_kin = read_cohort(row)
        features = local_features(transform, cohort, baselines[baseline_key], neighbors=config['neighbors'])
        np.testing.assert_allclose(features['z'], expected_z, rtol=1e-6, atol=2e-5)
        np.testing.assert_allclose(features['cell_kin'], expected_kin, rtol=1e-6, atol=2e-5)
        features['source_key'] = row['source_key']
        if row['role'] == 'training_source':
            normalization_packs.append(dict(source_key=row['source_key'], cell_directU=features['cell_directU']))
        path = run.directory/f"{row['replicate']}_{row['label']}.npz"
        # Keep the original S/kin arrays bitwise unchanged; only controls are new.
        np.savez_compressed(path, z=expected_z, cell_ids=features['cell_ids'], cell_kin=expected_kin,
            **{key:features[key].astype('float32') for key in FEATURES if key != 'cell_kin'})
        rows.append(dict(source_key=row['source_key'], role=row['role'], path=str(path.resolve()),
            sha256=sha256(path), parent_path=row['path'], parent_sha256=row['sha256'],
            replicate=row['replicate'], label=row['label'], time_hours=row['time_hours'],
            reference_key=features['reference_key'], reference_hash=features['reference_hash'],
            source_hash=features['source_hash'], neighbor_count=features['neighbor_count'],
            neighbor_indices_sha256=features['neighbor_indices_sha256'], cells=len(features['cell_ids']),
            maximum_kinetic_reproduction_difference=float(np.max(np.abs(features['cell_kin']-expected_kin)))))
        run.json('status.json', dict(status='preparing_source_sidecar', completed=len(rows), total=len(records)))
    mean, scale, keys = fit_direct_normalization(normalization_packs,
        allowed_training_source_keys=[r['source_key'] for r in manifest['training_groups']])
    old = load_npz(parent_pack/'token_normalizer.npz')
    normalization = run.directory/'normalizer.npz'
    np.savez_compressed(normalization, mean_cell_kin=old['mean_cell_kin'], scale_cell_kin=old['scale_cell_kin'],
        mean_cell_directU=mean, scale_cell_directU=scale, training_source_keys=np.asarray(keys))
    result = dict(status='L3_SOURCE_KINETIC_SIDECAR_READY',
        parent_pack=str(parent_pack.resolve()), parent_pack_manifest_sha256=sha256(manifest_path),
        source_data_path=str(source_data.resolve()), source_data_sha256=config['data_sha256'],
        source_transform_sha256=sha256(transform_path), normalizer_sha256=sha256(normalization),
        training_groups=[r for r in rows if r['role'] == 'training_source'],
        development_sources=[r for r in rows if r['role'] == 'development_source'],
        neighbors=config['neighbors'], reference_policy='original_observed_reference_frozen',
        Umean_definition='replace_each_current_raw_U_row_with_its_source_cohort_mean',
        S_geometry_and_gamma_fixed=True, normalization_training_sources_only=True,
        target_arrays_read=False, r4_targets_read=False, original_pack_modified=False)
    run.json('manifest.json', result)
    run.json('status.json', dict(status='complete'))
    return result


def load_sidecar(directory, source_key):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['status'] != 'L3_SOURCE_KINETIC_SIDECAR_READY':
        raise ValueError('Source sidecar is not complete')
    if sha256(Path(manifest['parent_pack'])/'manifest.json') != manifest['parent_pack_manifest_sha256']:
        raise ValueError('Frozen parent pack changed')
    matches = [r for r in manifest['training_groups']+manifest['development_sources'] if r['source_key']==source_key]
    if len(matches) != 1 or sha256(matches[0]['path']) != matches[0]['sha256']:
        raise ValueError('Source sidecar identity/hash mismatch')
    return load_npz(matches[0]['path'])


def load_normalizer(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if sha256(directory/'normalizer.npz') != manifest['normalizer_sha256']:
        raise ValueError('Source sidecar normalization changed')
    return load_npz(directory/'normalizer.npz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', required=True)
    parser.add_argument('--parent-pack', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = prepare(args.configuration, args.parent_pack, args.output)
    print(json.dumps({k:result[k] for k in ('status', 'r4_targets_read', 'original_pack_modified')}, indent=2))
