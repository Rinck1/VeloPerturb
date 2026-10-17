import io
from pathlib import Path
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w5_scifate2_technical import bounded_copy,inspect_schema
from w5_scifate2_technical import inspect_existing


def test_download_copy_enforces_byte_and_time_bounds():
    target = io.BytesIO()
    assert bounded_copy(io.BytesIO(b'abc'),target,limit=3,deadline=time.monotonic()+10) == 3
    assert target.getvalue() == b'abc'
    with pytest.raises(ValueError):bounded_copy(io.BytesIO(b'abcd'),io.BytesIO(),limit=3,deadline=time.monotonic()+10)
    with pytest.raises(TimeoutError):bounded_copy(io.BytesIO(b'a'),io.BytesIO(),limit=3,deadline=time.monotonic()-1)


def test_technical_schema_requires_actual_SU_and_only_groups_metadata(tmp_path):
    data = ad.AnnData(X=sparse.csr_matrix(np.eye(3)),
        obs=pd.DataFrame(dict(rep=['r1','r1','r4'],timepoint=['D3','D4','05h']),index=['a','b','c']))
    data.layers['spliced'] = data.X.copy();data.layers['unspliced'] = data.X.copy()
    path = tmp_path/'paired.h5ad';data.write_h5ad(path)
    result,cohorts = inspect_schema(path)
    assert result['cells'] == 3 and result['gene_ids_unique']
    assert not result['expression_count_values_checked'] and not result['biological_effects_read']
    assert cohorts['cells'].sum() == 3
    del data.layers['unspliced'];data.layers['new'] = data.X.copy()
    path2 = tmp_path/'new_not_U.h5ad';data.write_h5ad(path2)
    with pytest.raises(ValueError):inspect_schema(path2)


def test_existing_inspection_does_not_accept_a_different_modality_contract(tmp_path):
    import json
    path = tmp_path/'contract.json'
    path.write_text(json.dumps(dict(prediction_or_response_analysis=True,kind='scifate2_public_SU_technical_only')))
    with pytest.raises(ValueError):inspect_existing(path,tmp_path/'unused')
    assert not (tmp_path/'unused').exists()
