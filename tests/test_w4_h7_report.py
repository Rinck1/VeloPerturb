import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_artifacts import sha256
from w4_arms import CORE_ARMS
from w4_h7_report import report


def test_auxiliary_report_uses_frozen_errors_and_preserves_H1(tmp_path):
    inputs = tmp_path/'evaluation'; inputs.mkdir()
    (inputs/'summary.json').write_text(json.dumps(dict(status='H1_REPLICATION_EVALUATED',
        kind='formal_replication', H1_TFs=['A','B','C','D','E'], intervals=[[2,3],[3,4]], H1_supported=False)))
    errors = {arm:np.full((5,2), 10.) for arm in CORE_ARMS}
    errors['direct_U_pool'][:] = 8.
    errors['vel_pool'][:] = 9.
    np.savez(inputs/'TF_interval_errors.npz', TFs=np.array(['A','B','C','D','E']),
        intervals=np.array([[2,3],[3,4]]), **errors)
    before = sha256(inputs/'summary.json')
    result = report(inputs, tmp_path/'report')
    assert not result['H1_supported'] and not result['H1_success_rule_changed']
    np.testing.assert_allclose(result['comparisons']['direct_U_vs_static']['gain'], .2)
    np.testing.assert_allclose(result['comparisons']['velocity_vs_direct_U']['gain'], -.1)
    assert result['comparisons']['direct_U_vs_static']['supported']
    assert sha256(inputs/'summary.json') == before
