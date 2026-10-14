import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/phase0'))
from w3_6b_author_mapping_audit import mapping_reasons, official_cohort


def test_mapping_audit_distinguishes_absence_ambiguity_and_conflict():
    rows = [dict(cell_barcode=str(i), author_clone='A') for i in range(4)]
    local = {'1': dict(n_barcodes='2', clone_barcode='A'),
             '2': dict(n_barcodes='1', clone_barcode='B'),
             '3': dict(n_barcodes='1', clone_barcode='A')}
    assert mapping_reasons(rows, local) == dict(unassigned=1, multiple_barcodes=1,
                                               single_clone_mismatch=1, single_clone_exact=1)


def test_official_join_uses_clone_calls_without_local_label_selection():
    src = [dict(author_clone='A', quant_barcode_match=True)] * 3
    dst = [dict(author_clone='A', quant_barcode_match='True', valid_author_group='True', bc_group='group1')] * 3
    assert official_cohort(src, dst) == [dict(clone='A', group='group1', source_cells=3, target_cells=3)]
    with pytest.raises(ValueError, match='conflict'):
        official_cohort(src, dst + [dict(dst[0], bc_group='group2')])
