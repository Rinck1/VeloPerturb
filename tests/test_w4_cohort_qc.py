import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_cohort_qc import eligible_cohorts


def test_common_cohort_uses_technical_qc_only_and_keeps_controls_out():
    counts = {day: dict(A=50,B=51,C=50,CTRL=500,AAVS1=100,unassigned=1000) for day in (2,3,4,5)}
    counts[3]['B'] = 49
    counts[5]['C'] = 49
    result = eligible_cohorts(counts)
    assert result['H1_TFs'] == ['A','C']
    assert result['H2_TFs'] == ['A']
    assert result['interval_TFs']['2_to_3'] == ['A','C']
    assert result['interval_TFs']['2_to_5'] == ['A','B']
    with pytest.raises(ValueError, match='50-cell'):
        eligible_cohorts(counts, minimum_cells=49)
