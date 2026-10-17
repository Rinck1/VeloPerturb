from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w12_cohort_calibration import calibrate_rows


def test_source_only_kl_projection_matches_cohort_mass_and_keeps_row_probabilities():
    q=np.array([[.9,.08,.02],[.8,.1,.1],[.2,.2,.6],[.1,.2,.7]])
    mass=np.array([.1,.6,.3]);before=q.copy()
    result,_=calibrate_rows(q,mass)
    np.testing.assert_allclose(result.sum(1),1,atol=1e-12)
    np.testing.assert_allclose(result.mean(0),mass,atol=1e-8)
    np.testing.assert_array_equal(q,before)
    assert (result>0).all()


def test_projection_is_identity_when_source_masses_already_match():
    q=np.array([[.1,.9],[.8,.2],[.4,.6]])
    result,_=calibrate_rows(q,q.mean(0))
    np.testing.assert_allclose(result,q,atol=1e-12,rtol=1e-12)
