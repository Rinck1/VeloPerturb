import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w14_loro import energy
from w14_supplementary_metrics import bootstrap_energy,source_kernels


def test_cell_kernels_equal_full_mixture_energy_and_reweighted_samples():
    rng=np.random.default_rng(2);x=rng.normal(size=(5,3,2));q=rng.uniform(size=(5,3));q/=q.sum(1,keepdims=True)
    y=rng.normal(size=(7,2));a,b,d=source_kernels(x,q,y,block=2)
    aw=np.array([[.2]*5,[.4,.2,.2,.2,0.]])
    bw=np.array([[1/7]*7,[2/7,1/7,1/7,1/7,1/7,1/7,0.]])
    actual=bootstrap_energy(a,b,d,aw,bw,2)
    assert actual[0]==pytest.approx(energy(x.reshape(-1,2),q.ravel()/5,y),abs=1e-12)
    from scipy.spatial.distance import cdist
    w=(q*aw[1,:,None]).ravel();xx=x.reshape(-1,2)
    expected=(2*w@cdist(xx,y)@bw[1]-w@cdist(xx,xx)@w-bw[1]@cdist(y,y)@bw[1])/np.sqrt(2)
    assert actual[1]==pytest.approx(expected,abs=1e-12)
