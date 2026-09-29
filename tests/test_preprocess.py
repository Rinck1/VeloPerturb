import json

import numpy as np
import pytest
from scipy import io, sparse

from veloroute.preprocess import call_guides, check_layout, collapse_gene_versions, load_usa


def test_usa_order_and_ambiguous_separation(tmp_path):
    (tmp_path/'quant.json').write_text(json.dumps({'usa_mode': True, 'num_genes': 6}))
    (tmp_path/'quants_mat_rows.txt').write_text('AA\nBB\n')
    (tmp_path/'quants_mat_cols.txt').write_text('ENSG1.1\nENSG2.1\n')
    io.mmwrite(tmp_path/'quants_mat.mtx', sparse.coo_matrix([[1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5]]))
    bc, genes, (s, u, a), _ = load_usa(tmp_path)
    assert bc == ['AA', 'BB']
    np.testing.assert_equal(s.toarray(), [[1, 2], [2, 1]])
    np.testing.assert_equal(u.toarray(), [[3, 4], [4, 3]])
    np.testing.assert_equal(a.toarray(), [[5, 6], [6, 5]])
    # Current alevin-fry suffixes its U and A column labels.
    (tmp_path/'quants_mat_cols.txt').write_text('ENSG1.1\nENSG2.1\nENSG1.1-U\nENSG2.1-U\nENSG1.1-A\nENSG2.1-A\n')
    _, _, (s2, u2, a2), _ = load_usa(tmp_path)
    for left, right in ((s, s2), (u, u2), (a, a2)):
        np.testing.assert_equal(left.toarray(), right.toarray())


def test_no_fabricated_usa(tmp_path):
    (tmp_path/'quant.json').write_text(json.dumps({'usa_mode': False}))
    with pytest.raises(ValueError, match='not USA'):
        load_usa(tmp_path)


def test_gene_version_sum_preserves_par_y():
    genes, (matrix,) = collapse_gene_versions(['ENSG1.1', 'ENSG1.2', 'ENSG1.1_PAR_Y'],
                                             [sparse.csr_matrix([[1, 2, 4]])])
    assert genes == ['ENSG1', 'ENSG1_PAR_Y']
    np.testing.assert_equal(matrix.toarray(), [[3, 4]])


def test_guide_caller_rejects_ties_and_multiguide_without_forcing_labels():
    manifest = [{'feature_id': k, 'guide_name': k, 'target_from_published_name': k, 'control_class': 'TF'}
                for k in ['A', 'B']]
    cfg = dict(minimum_top_umi=3, minimum_top_fraction=.8, minimum_top_runner_ratio=5,
               formal_validation_approved=False)
    result = call_guides([[20, 1], [8, 8], [0, 0], [4, 1]], ['A', 'B'], manifest, cfg)
    assert list(result.condition) == ['A', 'unassigned', 'unassigned', 'unassigned']
    assert not result.guide_labels_validated.any()
    with pytest.raises(ValueError, match='raw'):
        call_guides([[.5, 1]], ['A', 'B'], manifest, cfg)


def test_layout_geometry_requires_observed_lengths_and_offset():
    cfg = dict(read_length=91, minimum_barcode_match=.5, minimum_barcode_offset_ratio=20)
    audit = dict(records_examined=100, r1_length_histogram={26: 100}, r2_length_histogram={91: 100},
                 published_barcode_match_fraction_by_zero_based_offset={'0': .9, '1': .001})
    check_layout(audit, cfg)
    audit['r1_length_histogram'] = {8: 100}
    with pytest.raises(ValueError, match='read length'):
        check_layout(audit, cfg)
