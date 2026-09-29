import numpy as np
import pytest

from veloroute.contracts import FitScope
from veloroute.latent import FrozenSplicingTransform, load_pack, normalized_counts, save_pack


def fitted():
    rng = np.random.default_rng(1)
    s, u = rng.poisson(10, (40, 8)), rng.poisson(3, (40, 8))
    ids = tuple(f'train:{i}' for i in range(40))
    transform = FrozenSplicingTransform.fit(s, u, gene_ids=tuple(f'G{i}' for i in range(8)), cell_ids=ids,
                scope=FitScope(frozenset(ids)), n_genes=8, n_components=3, minimum_u_cells=3)
    return transform, s, u


def test_transform_static_independent_of_u_but_velocity_uses_u():
    t, s, u = fitted()
    a, b = t.transform(s, u, gene_ids=t.gene_ids), t.transform(s, u[::-1], gene_ids=t.gene_ids)
    np.testing.assert_array_equal(a['z'], b['z'])
    assert not np.allclose(a['velocity'], b['velocity'])
    np.testing.assert_array_equal(a['u_predicted'], b['u_predicted'])


def test_transform_rejects_heldout_fit_and_bad_gene_order():
    t, s, u = fitted()
    with pytest.raises(ValueError, match='Leakage'):
        FrozenSplicingTransform.fit(s, u, gene_ids=t.gene_ids, cell_ids=[f'heldout:{i}' for i in range(40)],
                                   scope=FitScope(frozenset({'train:1'})))
    with pytest.raises(ValueError, match='gene order'):
        t.transform(s, u, gene_ids=t.gene_ids[::-1])


def test_transform_save_roundtrip(tmp_path):
    t, s, u = fitted()
    t.save(tmp_path/'transform.npz')
    restored = FrozenSplicingTransform.load(tmp_path/'transform.npz')
    for key, value in t.transform(s, u, gene_ids=t.gene_ids).items():
        np.testing.assert_array_equal(value, restored.transform(s, u, gene_ids=t.gene_ids)[key])


def test_zero_spliced_is_not_silently_normalized():
    with pytest.raises(ValueError, match='Zero spliced'):
        normalized_counts([[0, 0]], [[1, 2]])


def test_source_pack_cannot_include_future(tmp_path):
    t, s, u = fitted()
    arrays = t.transform(s, u, gene_ids=t.gene_ids)
    arrays.update(cell_ids=np.array([f'c{i}' for i in range(40)]), conditions=np.repeat('TF', 40), depth=s.sum(1))
    save_pack(tmp_path/'source.npz', arrays, {'side': 'source'})
    load_pack(tmp_path/'source.npz', expected_side='source')
    arrays['future_z'] = arrays['z']
    save_pack(tmp_path/'bad.npz', arrays, {'side': 'source'})
    with pytest.raises(ValueError, match='leakage'):
        load_pack(tmp_path/'bad.npz', expected_side='source')
