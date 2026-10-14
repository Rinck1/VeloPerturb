import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/phase0'))
from w3_6c_deltaU_identifiability import condition_roles, csr_selected_rows, log_bulk


def test_csr_loader_reads_only_selected_data_spans():
    raw = sparse.csr_matrix(np.array([[1, 0, 2], [999, 999, 999], [3, 4, 0], [0, 5, 6], [999, 999, 999]]))
    reads = []
    class Dataset:
        def __init__(self, name, array): self.name, self.array = name, array
        def __getitem__(self, item):
            reads.append((self.name, item)); return self.array[item]
    class Group(dict):
        attrs = {'encoding-type': 'csr_matrix', 'shape': raw.shape}
    group = Group(data=Dataset('data', raw.data), indices=Dataset('indices', raw.indices), indptr=Dataset('indptr', raw.indptr))
    selected = csr_selected_rows(group, [0, 2, 3])
    np.testing.assert_array_equal(selected.toarray(), raw[[0, 2, 3]].toarray())
    touched = set()
    for name, item in reads:
        if name == 'data': touched.update(range(item.start, item.stop))
    forbidden = set(range(raw.indptr[1], raw.indptr[2])) | set(range(raw.indptr[4], raw.indptr[5]))
    assert not touched & forbidden


def test_csr_loader_refuses_unsorted_duplicate_or_invalid_rows(tmp_path):
    with h5py.File(tmp_path / 'test.h5', 'w') as f:
        group = f.create_group('matrix')
        group.attrs['encoding-type'] = 'csr_matrix'; group.attrs['shape'] = (5, 3)
        for rows in ([2, 0], [1, 1], [-1], [5], []):
            with pytest.raises(ValueError, match='Rows'):
                csr_selected_rows(group, rows)


def test_conflicting_reservation_roles_fail_closed(tmp_path):
    path = tmp_path / 'roles.csv'
    path.write_text('condition,role\nA,train\nA,confirmation\n')
    with pytest.raises(ValueError, match='conflicting'):
        condition_roles(path)


def test_bulk_normalization_uses_all_genes_before_panel_selection():
    counts = sparse.csr_matrix([[10, 10, 980], [10, 10, 980]])
    result = log_bulk(counts, np.array([True, True]))
    np.testing.assert_allclose(result[:2], np.log1p([100, 100]))
