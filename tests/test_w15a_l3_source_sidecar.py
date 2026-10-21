"""Source-only local direct-U and fixed-reference U perturbations."""
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_artifacts import sha256
from w15a_cellflow_adapter import (SourceCohort, TokenNormalizer,
    fit_source_transform, pooled_tokens, source_tokens)
import w15a_l3_source_sidecar as sidecar


@pytest.fixture
def toy():
    rng = np.random.default_rng(7)
    groups = []
    for index, (rep, hour) in enumerate((('r1', 0), ('r1', 24), ('r4', 5), ('r4', 10))):
        groups.append(SourceCohort('neural_tube_differentiation', hour, rep,
            sparse.csr_matrix(rng.poisson(5, (12, 12))),
            sparse.csr_matrix(rng.poisson(2, (12, 12))),
            np.array([f'group{index}_cell{i}' for i in range(12)])))
    transform = fit_source_transform(groups[:1], [f'g{i}' for i in range(12)],
        allowed_source_ids=groups[0].cell_ids, n_hvg=8, dimensions=3, clusters=2)
    return groups, transform


def test_reconstructs_original_local_kinetics_exactly(toy):
    groups, transform = toy
    for group, reference in ((groups[1], groups[0]), (groups[2], groups[2])):
        actual = sidecar.local_features(transform, group, reference, neighbors=4)
        original = source_tokens(transform, group, reference, neighbors=4)
        np.testing.assert_array_equal(actual['z'], original['z'])
        np.testing.assert_array_equal(actual['cell_kin'], original['cell_kin'])
        assert actual['neighbor_count'] == 4


def test_local_direct_u_uses_same_neighbors_and_full_s_gene_depth(toy):
    groups, transform = toy
    group, reference = groups[1], groups[0]
    z = transform.state(group.s)
    indices = NearestNeighbors(n_neighbors=4).fit(z).kneighbors(z, return_distance=False)
    expected = []
    for neighbors in indices:
        factor = transform.target_sum/float(group.s[neighbors].sum())
        pooled_u = np.asarray(group.u[neighbors][:, transform.genes].sum(0)).ravel()*factor
        expected.append(np.log1p(pooled_u) @ transform.components.T)
    expected = np.asarray(expected)-pooled_tokens(transform, reference)['directU']
    np.testing.assert_allclose(sidecar.local_features(transform, group, reference, neighbors=4)['cell_directU'], expected)


def test_u_controls_match_literal_source_count_edits_and_preserve_s(toy):
    groups, transform = toy
    group, reference = groups[1], groups[0]
    before = (group.fingerprint(), reference.fingerprint(), transform.fingerprint())
    actual = sidecar.local_features(transform, group, reference, neighbors=4)
    mean_u = sparse.csr_matrix(np.broadcast_to(np.asarray(group.u.mean(0)), group.u.shape))
    for key, edited in (('cell_kin_Umean', mean_u), ('cell_kin_Uzero', sparse.csr_matrix(group.u.shape))):
        edited_group = SourceCohort(group.condition, group.time_hours, group.replicate,
            group.s, edited, group.cell_ids)
        expected = source_tokens(transform, edited_group, reference, neighbors=4)
        np.testing.assert_allclose(actual[key], expected['cell_kin'])
        np.testing.assert_array_equal(actual['z'], expected['z'])
    assert before == (group.fingerprint(), reference.fingerprint(), transform.fingerprint())


def test_future_reference_and_invalid_neighbor_count_refused(toy):
    groups, transform = toy
    with pytest.raises(ValueError, match='future'):
        sidecar.local_features(transform, groups[0], groups[1], neighbors=4)
    for invalid in (0, -1, 2.5):
        with pytest.raises(ValueError, match='neighbor'):
            sidecar.local_features(transform, groups[0], groups[0], neighbors=invalid)


def test_direct_u_normalization_only_fits_allowed_training_sources():
    packs = [dict(source_key='train0', cell_directU=np.array([[1., 5.], [3., 5.]])),
             dict(source_key='train1', cell_directU=np.array([[5., 5.]]))]
    mean, scale, keys = sidecar.fit_direct_normalization(packs, allowed_training_source_keys=['train0', 'train1'])
    np.testing.assert_allclose(mean, [3., 5.])
    assert scale[1] == 1e-5 and keys == ('train0', 'train1')
    with pytest.raises(ValueError, match='forbidden'):
        sidecar.fit_direct_normalization(packs, allowed_training_source_keys=['train0'])


def test_duplicate_source_windows_must_use_identical_source_pack():
    row = dict(source_key='r4|c|5h', path='source5.npz', sha256='abc', replicate='r4', label='05h', time_hours=5.)
    manifest = dict(training_groups=[], development_sources=[row, dict(row, stop_hours=15.)])
    assert len(sidecar.unique_records(manifest)) == 1
    manifest['development_sources'][1]['sha256'] = 'other'
    with pytest.raises(ValueError, match='different cohorts'):
        sidecar.unique_records(manifest)


def test_complete_source_only_sidecar_and_hash_checked_loaders(toy, tmp_path, monkeypatch):
    groups, transform = toy
    labels = ['D3', 'D4', '05h', '10h']
    s = sparse.vstack([g.s for g in groups]+[sparse.csr_matrix(np.full((2, 12), 1e6))], format='csr')
    u = sparse.vstack([g.u for g in groups]+[sparse.csr_matrix(np.full((2, 12), 1e6))], format='csr')
    ids = np.concatenate([g.cell_ids for g in groups]+[np.array(['forbidden_target0', 'forbidden_target1'])])
    obs = pd.DataFrame(index=ids,
        data=dict(rep=np.concatenate([np.repeat(g.replicate, len(g.cell_ids)) for g in groups]+[np.repeat('r4', 2)]),
            timepoint=np.concatenate([np.repeat(label, len(g.cell_ids)) for label,g in zip(labels, groups)]+[np.repeat('20h', 2)])))
    data = ad.AnnData(X=s, obs=obs, var=pd.DataFrame(index=transform.gene_names))
    data.layers['spliced'], data.layers['unspliced'] = s, u
    raw = tmp_path/'source_data.h5ad'; data.write_h5ad(raw)
    transform_path = tmp_path/'source_transform.npz'; transform.save(transform_path)
    parent = tmp_path/'parent'; parent.mkdir(); records=[]; train_packs=[]
    for index, (group, label) in enumerate(zip(groups, labels)):
        reference = groups[0] if index < 2 else groups[2]
        features = source_tokens(transform, group, reference, neighbors=4)
        path = parent/f'{group.replicate}_{label}.npz'
        np.savez_compressed(path, cell_ids=group.cell_ids, z=features['z'].astype('float32'),
            cell_kin=features['cell_kin'].astype('float32'), target=np.full_like(features['z'], 1e8))
        records.append(dict(source_key=group.key, path=str(path), sha256=sha256(path),
            source_hash=group.fingerprint(), replicate=group.replicate, label=label, time_hours=group.time_hours))
        if index < 2:train_packs.append(features)
    normalizer = TokenNormalizer.fit(train_packs, allowed_training_source_keys=[g.key for g in groups[:2]])
    np.savez_compressed(parent/'token_normalizer.npz',
        **{'mean_'+k:v for k,v in normalizer.mean.items()},
        **{'scale_'+k:v for k,v in normalizer.scale.items()})
    manifest = dict(training_groups=records[:2], development_sources=records[2:]+[dict(records[2], stop_hours=15.)],
        source_transform_sha256=sha256(transform_path), token_normalizer_sha256=sha256(parent/'token_normalizer.npz'))
    (parent/'manifest.json').write_text(json.dumps(manifest))
    config = tmp_path/'selected.json'; config.write_text(json.dumps(dict(data_path=str(raw), data_sha256=sha256(raw),
        transform_path=str(transform_path), neighbors=4)))
    original = sidecar.read_count_rows; read_rows=[]
    def source_only(path, layer, rows):
        assert set(rows) <= set(range(48)), 'Forbidden future target count read'
        read_rows.extend(rows); return original(path, layer, rows)
    monkeypatch.setattr(sidecar, 'read_count_rows', source_only)
    parent_hashes = {str(p):sha256(p) for p in parent.iterdir() if p.is_file()}
    output = tmp_path/'sidecar'; result = sidecar.prepare(config, parent, output)
    assert result['status'] == 'L3_SOURCE_KINETIC_SIDECAR_READY'
    assert len(result['training_groups']) == 2 and len(result['development_sources']) == 2
    assert result['target_arrays_read'] is False and read_rows
    assert parent_hashes == {str(p):sha256(p) for p in parent.iterdir() if p.is_file()}
    loaded = sidecar.load_sidecar(output, groups[3].key)
    assert set(loaded) == {'z', 'cell_ids', *sidecar.FEATURES}
    fitted = sidecar.load_normalizer(output)
    np.testing.assert_array_equal(fitted['mean_cell_kin'], normalizer.mean['cell_kin'])
    assert set(fitted['training_source_keys']) == {g.key for g in groups[:2]}
    selected = result['development_sources'][1]['path']
    with Path(selected).open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        sidecar.load_sidecar(output, groups[3].key)
