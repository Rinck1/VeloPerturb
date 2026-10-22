from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15q_velocity_countsplit as module


def test_split_conserves_selected_and_unselected_counts_and_is_deterministic():
    s=np.array([[100,400],[500,100]]);u=s//2;depth=np.array([1000,1500])
    a,b=module.split_counts(s,u,depth,0);c,d=module.split_counts(s,u,depth,0)
    for actual,original in zip((a[0]+b[0],a[1]+b[1],a[2]+b[2]),(s,u,depth)):
        np.testing.assert_array_equal(actual,original)
    for first,second in zip(a,c):np.testing.assert_array_equal(first,second)
    assert np.all(a[2]>=a[0].sum(1)) and np.all(b[2]>=b[0].sum(1))


def test_fractional_negative_or_nonfinite_counts_rejected_not_rounded():
    for values in ([[.5]],[[np.nan]],[[-1.]]):
        with pytest.raises(ValueError,match='integer counts'):module.integer_counts(values)


def test_normalization_uses_total_S_depth_and_cosine_detects_reverse():
    s=np.array([[10.,20.]]);u=np.array([[4.,1.]])
    sn,un=module.normalized(s,u,np.array([100.]),10000.)
    np.testing.assert_array_equal(sn,s*100);np.testing.assert_array_equal(un,u*100)
    np.testing.assert_allclose(module.cosine(s,s),1.)
    np.testing.assert_allclose(module.cosine(s,-s),-1.)
    with pytest.raises(ValueError,match='undefined'):module.cosine(s,s*0)
