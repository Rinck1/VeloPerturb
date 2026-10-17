import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_wait_calibration import validate_qc


def ready_qc():
    return dict(status='TECHNICAL_COHORT_QC_COMPLETE', biological_effects_read=False,
        minimum_TF_cells_per_day=50, H1_TFs=['A','B','C','D','E'])


def test_calibration_waiter_accepts_only_completed_fixed_technical_cohort():
    assert validate_qc(ready_qc()) == 5
    for change in (dict(status='running'), dict(biological_effects_read=True),
                   dict(minimum_TF_cells_per_day=49), dict(H1_TFs=['A','B','C','D']),
                   dict(H1_TFs=['A','B','C','CTRL','E']), dict(H1_TFs=['A','B','B','D','E'])):
        with pytest.raises(ValueError, match='technical-only QC'):
            validate_qc(dict(ready_qc(), **change))
