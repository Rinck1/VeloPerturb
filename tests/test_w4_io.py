import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_shorttime import (condition_embeddings, read_count_rows, read_metadata,
                          validate_development_times)
from w4_arms import fit_ridge_training_cv


def test_h5ad_selected_rows_match_original_without_loading_unselected_rows(tmp_path, monkeypatch):
    values = sparse.csr_matrix(np.arange(80).reshape(10,8))
    path = tmp_path/'counts.h5ad'
    a = ad.AnnData(values, obs=pd.DataFrame(index=[f'cell{i}' for i in range(10)]))
    a.layers['spliced'] = values
    a.write_h5ad(path)
    requested = np.array([0,1,5,9])
    reads = []
    original = h5py.Dataset.__getitem__
    def record(dataset, key):
        if dataset.name == '/layers/spliced/data':
            reads.append(key)
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', record)
    np.testing.assert_array_equal(read_count_rows(path, 'spliced', requested).toarray(), values[requested].toarray())
    pointers = values.indptr
    allowed = set(j for i in requested for j in range(pointers[i], pointers[i+1]))
    assert reads
    assert all(set(range(key.start,key.stop)) <= allowed for key in reads)
    obs, genes = read_metadata(path)
    assert obs.index.tolist() == a.obs_names.tolist() and len(genes) == 8


def test_training_targets_change_response_fit_but_not_source_geometry():
    rng = np.random.default_rng(22)
    x, y = rng.normal(size=(12,5)), rng.normal(size=(12,3))
    scaler1, fit1, _ = fit_ridge_training_cv(x, y)
    scaler2, fit2, _ = fit_ridge_training_cv(x, y+10)
    np.testing.assert_array_equal(scaler1.mean_, scaler2.mean_)
    np.testing.assert_allclose(fit2.predict(scaler2.transform(x))-fit1.predict(scaler1.transform(x)), 10, atol=1e-10)


def test_ridge_equal_scores_choose_stronger_regularization():
    x, y = np.arange(30).reshape(6,5), np.ones((6,2))
    _, model, _ = fit_ridge_training_cv(x, y)
    assert model.alpha == 1000


def test_control_embedding_is_empty_perturbation_and_missing_tf_is_error(tmp_path):
    path = tmp_path/'esm.npz'
    np.savez(path, conditions=np.array(['TF_A']), embeddings=np.ones((1,5)),
        metadata_json='{"frozen":true,"source":"test"}')
    result = condition_embeddings(path, ['CTRL','AAVS1','TF_A'])
    np.testing.assert_array_equal(result[:2], np.zeros((2,5)))
    np.testing.assert_array_equal(result[2], np.ones(5))
    with pytest.raises(ValueError, match='Missing TF'):
        condition_embeddings(path, ['TF_unknown'])


def test_replication_cannot_enter_development_runner():
    validate_development_times(pd.DataFrame({'day':[4]}), pd.DataFrame({'day':[5]}))
    with pytest.raises(ValueError, match='frozen protocol'):
        validate_development_times(pd.DataFrame({'day':[2]}), pd.DataFrame({'day':[3]}))
