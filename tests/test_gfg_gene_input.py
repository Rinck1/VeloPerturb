import json

import numpy as np
import pytest

from veloroute.artifacts import sha256
from veloroute.contracts import FitScope
from veloroute.gfg_inputs import validate_gene_source
from veloroute.latent import FrozenSplicingTransform, normalized_counts
from veloroute.full_experiments import train_full_model
from veloroute.full_training import FullTrainConfig


@pytest.mark.parametrize('corruption', ['none', 'gene_order', 'future_field', 'wrong_day', 'wrong_S'])
def test_GFG_gene_contract_rejects_misalignment_and_future_fields(tmp_path, corruption):
    rng = np.random.default_rng(19)
    s, u = rng.poisson(7, (16, 8))+1, rng.poisson(3, (16, 8))+1
    ids, genes = [f'cell{i}' for i in range(16)], [f'gene{i}' for i in range(8)]
    transform = FrozenSplicingTransform.fit(s, u, gene_ids=genes, cell_ids=ids,
        scope=FitScope(frozenset(ids)), n_genes=6, n_components=3)
    transform.save(tmp_path/'transform.npz')
    source = transform.transform(s, u, gene_ids=genes)
    sn, un = normalized_counts(s, u)
    metadata = dict(transform_hash=sha256(tmp_path/'transform.npz'))
    gm = dict(side='source_gene_US', day=4,
        normalization='source_S_library_10000_unlogged_US_no_neighbor_smoothing')
    arrays = dict(gene_us=np.concatenate((un[:, transform.selected], sn[:, transform.selected]), 1),
        gene_ids=np.asarray(genes)[transform.selected], cell_ids=np.asarray(ids))
    if corruption == 'gene_order':
        arrays['gene_ids'] = arrays['gene_ids'][::-1]
    elif corruption == 'future_field':
        arrays['future_z'] = source['z']
    elif corruption == 'wrong_day':
        gm['day'] = 5
    elif corruption == 'wrong_S':
        arrays['gene_us'][:, 6:] += 10
    arrays['metadata_json'] = np.array(json.dumps(gm))
    path = tmp_path/'gene_source.npz'
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as data:
        if corruption == 'none':
            validate_gene_source(data, source, metadata, gm, tmp_path/'train_source.npz')
        else:
            with pytest.raises(ValueError):
                validate_gene_source(data, source, metadata, gm, tmp_path/'train_source.npz')


def test_old_PCA_entry_cannot_silently_claim_GFG_training(tmp_path):
    with pytest.raises(ValueError, match='gene-level'):
        train_full_model('unopened_source', 'unopened_target', 'unopened_conditions',
            {'model': {'dynamics_backend': 'gfg'}}, tmp_path/'forbidden', exploratory_real=True)
    assert not (tmp_path/'forbidden').exists()


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_GFG_native_weight_must_be_finite_and_nonnegative(weight):
    with pytest.raises(ValueError, match='gfg_native_weight'):
        FullTrainConfig(gfg_native_weight=weight)
