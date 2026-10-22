import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w16a_traxler_graph_cv as module


def test_three_folds_cover_63_KOs_once_and_never_include_reserved_test():
    original = dict(train=[f'KO{i}' for i in range(56)], val=[f'KO{i}' for i in range(56, 63)], test=['reserved'])
    folds = module.registered_splits(original, 3, 22)
    assert all(len(f['train']) == 42 and len(f['val']) == 21 for f in folds)
    assert sorted(k for f in folds for k in f['val']) == sorted(original['train']+original['val'])
    for f in folds:
        assert not set(f['train']) & set(f['val'])
        assert not (set(f['train']) | set(f['val'])) & set(f['test'])


def test_original_split_overlap_is_rejected():
    with pytest.raises(ValueError, match='overlap'):
        module.registered_splits(dict(train=['a', 'b'], val=['b'], test=[]), 2, 22)


def csr(group, value):
    value = sparse.csr_matrix(value)
    group.attrs['encoding-type'] = 'csr_matrix'; group.attrs['shape'] = value.shape
    for k in ('data', 'indices', 'indptr'): group.create_dataset(k, data=getattr(value, k))


def test_CSR_reader_preserves_requested_order_and_ignores_other_values(tmp_path):
    with h5py.File(tmp_path/'test.h5', 'w') as f:
        csr(f.create_group('m'), [[1., 0., 2.], [np.nan, 0., 0.], [0., 3., 4.]])
        actual = module.read_csr_rows(f['m'], [2, 0]).toarray()
    np.testing.assert_array_equal(actual, [[0., 3., 4.], [1., 0., 2.]])


def test_sparse_transform_matches_original_full_library_formula():
    from veloroute.latent import FrozenSplicingTransform
    from veloroute.contracts import FitScope
    rng = np.random.default_rng(12)
    s = rng.poisson(rng.uniform(1, 10, 14), size=(64, 14)).astype(float)
    u = rng.poisson(rng.uniform(1, 4, 14), size=(64, 14)).astype(float)
    ids = np.array([f'c{i}' for i in range(64)]); genes = np.array([f'g{i}' for i in range(14)])
    expected = FrozenSplicingTransform.fit(s, u, gene_ids=genes, cell_ids=ids, scope=FitScope(frozenset(ids)),
        n_genes=10, n_components=3, seed=0)
    actual = module.fit_transform(sparse.csr_matrix(s), sparse.csr_matrix(u), genes, ids,
        n_genes=10, dimensions=3, seed=0)
    np.testing.assert_array_equal(expected.selected, actual.selected)
    for name in ('components', 'gamma', 'mean', 'u_coef'):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=1e-9)
    result = module.apply_transform(actual, sparse.csr_matrix(s), sparse.csr_matrix(u))
    reference = expected.transform(s, u, gene_ids=genes)
    for k in reference: np.testing.assert_allclose(result[k], reference[k], atol=1e-6)
    assert set(module.apply_transform(actual, sparse.csr_matrix(s))) == {'z'}


def test_permutation_stays_in_source_library_and_keeps_singletons():
    ids = np.array(['B1:a', 'B1:b', 'B1:c', 'B2:a', 'B2:b', 'B3:a'])
    order = module.stratified_order(ids, 0, 'KO')
    np.testing.assert_array_equal(np.sort(order), np.arange(6))
    libraries = [k.split(':')[0] for k in ids]
    assert [libraries[i] for i in order] == libraries
    assert np.all(order[:5] != np.arange(5)) and order[5] == 5
    np.testing.assert_array_equal(order, module.stratified_order(ids, 0, 'KO'))


def test_seven_graphs_reuse_original_real_S_U_edges():
    import w15o_velocity_neighborhood as parent
    rng = np.random.default_rng(3)
    group = dict(label='KO', z=rng.normal(size=(12, 5)), velocity_program=rng.normal(size=(12, 3)),
        U_program=rng.normal(size=(12, 3)), shuffled_velocity_program=rng.normal(size=(12, 3)),
        cell_ids=np.array([f'B{i%2}:{i}' for i in range(12)]))
    plan = dict(state_candidates=8, graph_neighbors=4, shuffle_seeds=[0, 1, 2])
    expected = parent.source_graphs(group, 8, 4); actual, movement = module.graph_grid(group, plan)
    assert set(actual) == set(module.ARMS)
    for key in module.ARMS[:4]: np.testing.assert_array_equal(actual[key], expected[key])
    assert all(v['moved_fraction'] == 1. for v in movement)


def test_exact_mixture_energy_and_centering_have_correct_translation_behavior():
    rng = np.random.default_rng(5); x = rng.normal(size=(10, 2, 3))
    q = np.full((10, 2), .5); y = x.reshape(-1, 3)+4.
    result = module.exact_metrics(x, q, y, np.eye(3))
    assert result['energy'] > 0 and result['centered_energy'] < 1e-10
    assert result['mean_mse'] == pytest.approx(16.)
    assert result['variance_ratio'] == pytest.approx(1.)


def test_KO_summary_averages_shuffle_errors_before_relative_gain():
    rows = []
    for i in range(12):
        for seed in range(3):
            for arm in module.ARMS:
                error = .9 if arm == 'velocity_graph' else {'shuffle0':1., 'shuffle1':1.1, 'shuffle2':1.2}.get(arm, 1.)
                rows.append(dict(KO=f'KO{i}', seed=seed, fold=i%3, arm=arm, energy=error))
    plan = dict(model_seeds=[0, 1, 2], shuffle_seeds=[0, 1, 2], split_seed=22,
        bootstrap_repetitions=100, confidence=.95)
    actual = module.summarize(rows, plan)
    r = next(r for r in actual if r['control'] == 'shuffle_mean')
    assert r['energy_gain'] == pytest.approx(1-.9/1.1)
    assert r['ci_low'] == pytest.approx(r['ci_high']) and r['KOs'] == 12


def test_fold_preparation_does_not_load_validation_or_reserved_target_values(tmp_path):
    rng = np.random.default_rng(4); metadata = []
    labels = ['a', 'b', 'c', 'heldout', 'reserved']; h5dir = tmp_path/'raw'; h5dir.mkdir()
    for lib, hour in [('B1', 6), ('C1', 24)]:
        ids = [f'{lib}:{label}_{i}' for label in labels for i in range(12)]
        s = rng.poisson(4, size=(len(ids), 12)).astype(float)
        u = rng.poisson(2, size=(len(ids), 12)).astype(float)
        if hour == 24: s[36:] = np.nan; u[:] = np.nan
        with h5py.File(h5dir/(lib+'.h5ad'), 'w') as f:
            for axis, values in [('obs', ids), ('var', [f'g{i}' for i in range(12)])]:
                g = f.create_group(axis); g.attrs['_index'] = '_index'
                g.create_dataset('_index', data=np.asarray(values, dtype=h5py.string_dtype()))
            g = f.create_group('layers'); csr(g.create_group('spliced'), s); csr(g.create_group('unspliced'), u)
        metadata.extend(dict(KO=label, library=lib, barcode=f'{label}_{i}', time=hour) for label in labels for i in range(12))
    base = tmp_path/'base.json'; base.write_text(json.dumps({}))
    plan = dict(h5dir=str(h5dir), source_hours=6, target_hours=24, n_genes=10, n_components=3,
        transform_seed=0, base_configuration=str(base), conditions='public_ESM.npz', model_seeds=[0, 1, 2], steps=5000)
    split = dict(train=['a', 'b', 'c'], val=['heldout'], test=['reserved'])
    output = tmp_path/'fold'; module.build_fold(plan, split, output, pd.DataFrame(metadata))
    assert not (output/'validation_target.npz').exists() and not (output/'test_source.npz').exists()
    assert json.loads((output/'ready.json').read_text())['heldout_target_loaded'] is False
    with np.load(output/'train_target.npz', allow_pickle=False) as f:
        assert np.isfinite(f['z']).all()
        assert set(f['conditions']) == set(split['train'])


def evaluation_fixture(tmp_path, monkeypatch):
    from veloroute.latent import FrozenSplicingTransform
    from veloroute.contracts import FitScope
    root = tmp_path/'study'; (root/'prepare').mkdir(parents=True)
    (root/'prepare'/'summary.json').write_text(json.dumps(dict(status='complete')))
    rng = np.random.default_rng(6); s = rng.poisson(5, size=(20, 4)).astype(float)
    u = rng.poisson(2, size=(20, 4)).astype(float)
    genes = np.array([f'g{i}' for i in range(4)]); ids = np.array([f'fit{i}' for i in range(20)])
    tf = FrozenSplicingTransform.fit(s, u, gene_ids=genes, cell_ids=ids, scope=FitScope(frozenset(ids)),
        n_genes=4, n_components=2, seed=0)
    metadata = []
    for fold in range(3):
        pack = root/'pack'/f'fold{fold}'; pack.mkdir(parents=True); tf.save(pack/'transform.npz')
        (pack/'split.json').write_text(json.dumps(dict(train=['other'], val=[f'KO{fold}'], test=['reserved'])))
        directory = root/'training'/f'fold{fold}'; directory.mkdir(parents=True)
        (directory/'status.json').write_text(json.dumps(dict(status='complete')))
        np.savez(directory/'training_programs.npz', axes=np.eye(2))
        records = []
        for arm in module.ARMS:
            model = directory/(arm+'.msgpack'); model.write_bytes(arm.encode())
            path = directory/(arm+'.npz')
            np.savez(path, candidates=rng.normal(size=(2, 8, 2)), q=np.full((2, 8), 1/8),
                full_source_q=np.full((3, 8), 1/8), source_ids=np.array(['source0', 'source1']))
            records.append(dict(fold=fold, KO=f'KO{fold}', seed=0, arm=arm, path=str(path),
                sha256=module.sha256(path), model=str(model), model_sha256=module.sha256(model), heldout_future_used=False))
        (directory/'frozen_grid.json').write_text(json.dumps(dict(predictions=records)))
        metadata.extend(dict(KO=f'KO{fold}', library='C1', barcode=f'{fold}_{i}', time=24) for i in range(3))
    manifest = tmp_path/'cells.csv'; pd.DataFrame(metadata).to_csv(manifest, index=False)
    plan = dict(root=str(root), folds=3, model_seeds=[0], shuffle_seeds=[0, 1, 2], planned_fits=21,
        primary_controls=['no_graph', 'U_graph', 'shuffle_mean'], split_seed=22, bootstrap_repetitions=30,
        confidence=.95, cell_manifest=str(manifest), target_hours=24, h5dir='unused', shuffle_scope='synthetic')
    configuration = tmp_path/'config.json'; configuration.write_text(json.dumps(plan))
    monkeypatch.setattr(module, 'code_inputs', lambda *args:[])
    def load(h5dir, requested, layers):
        assert (root/'evaluation'/'whole_grid_frozen.json').exists(), 'Premature target read'
        assert layers == ('spliced',)
        return np.asarray(requested), dict(spliced=sparse.csr_matrix(s[:3])), genes
    monkeypatch.setattr(module, 'load_counts', load)
    return configuration, root


def test_full_evaluator_freezes_all_grids_before_reading_targets(tmp_path, monkeypatch):
    configuration, root = evaluation_fixture(tmp_path, monkeypatch)
    result = module.evaluate(configuration)
    assert result['fits'] == 21 and result['KOs'] == 3
    assert result['predictions'] == 21
    assert (root/'evaluation'/'metrics.csv').exists()
    assert json.loads((root/'evaluation'/'status.json').read_text())['status'] == 'complete'


def test_changed_prediction_aborts_evaluation_before_any_target_read(tmp_path, monkeypatch):
    configuration, root = evaluation_fixture(tmp_path, monkeypatch)
    (root/'training'/'fold0'/'velocity_graph.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Unfrozen prediction'):
        module.evaluate(configuration)
    assert not (root/'evaluation').exists()
