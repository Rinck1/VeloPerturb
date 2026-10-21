import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15g_response_program_cv as module


def test_inner_response_axes_do_not_read_heldout_future():
    rng=np.random.default_rng(15);n=12
    cohort={k:rng.normal(size=(n,4)) for k in ('state','U','velocity','embedding')}
    cohort.update(labels=np.array([f'KO{i}' for i in range(n)]),cells=np.full(n,40))
    target=rng.normal(size=(n,4));tr=np.arange(9);te=np.arange(9,12)
    c=dict(programs=2,protein_dimensions=2,ridge_alphas=[10],shuffle_seeds=[0,1,2])
    a=module.partition(cohort,target,tr,te,c)
    changed=target.copy();changed[te]=1e10
    b=module.partition(cohort,changed,tr,te,c)
    np.testing.assert_array_equal(a['axes'],b['axes'])
    pa,_=module.probe.fit_predictions(a,c);pb,_=module.probe.fit_predictions(b,c)
    for arm in pa['10']:np.testing.assert_array_equal(pa['10'][arm],pb['10'][arm])
