"""Tiny CPU checks for genuine L1 six-window comparison, no new biological data."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.distance import cdist

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w15_l1_common_alignment import (KEYS,VARIANTS,WINDOWS,join,joined_decay_rows,
                                    original_mixture,weighted_energy)


def test_original_three_seed_weights_are_not_uniform_over_experts():
    x=np.arange(12,dtype=float).reshape(6,2,1)
    q=np.array([[.9,.1],[.2,.8],[.4,.6],[.7,.3],[1.,0.],[.25,.75]])
    flat,w=original_mixture(x,q,2)
    np.testing.assert_array_equal(flat,x.reshape(-1,1))
    np.testing.assert_array_equal(w,q.ravel()/6)
    np.testing.assert_allclose(w.reshape(3,2,2).sum((1,2)),[1/3]*3)
    y=np.array([[2.],[5.],[10.]])
    expected=2*w@cdist(flat,y).mean(1)-w@cdist(flat,flat)@w-cdist(y,y).mean()
    assert weighted_energy(flat,w,y,block=3)==pytest.approx(expected,abs=1e-12)


def synthetic_results():
    reference=dict(study_manifest='frozen',source_samples='same')
    caches={'10':dict(path='old10.npz',sha256='a'),'15':dict(path='old15.npz',sha256='b'),
            '20':dict(path='old20.npz',sha256='c')}
    l1=dict(seeds=[0,1,2],reference_hashes=reference,target_caches=caches,
        windows_per_arm=6,linear_clock_not_available=True,metrics=[])
    for variant in VARIANTS:
        for a,b in WINDOWS:
            l1['metrics'].append(dict(variant=variant,key=f'{a:02d}_{b:02d}_decay',energy=2.,
                source_cells_per_seed=2,target_cells=3))
    common=dict(arms=['cf_static','cf_kin'],expected_seeds=[0,1,2],reference_hashes=reference,
                target_caches=caches,metrics=[])
    for arm in common['arms']:
        for key,(a,b,clock) in KEYS.items():
            common['metrics'].append(dict(arm=arm,key=key,kernel=clock,energy=1.,
                source_cells_per_seed=2,target_cells=3))
    return l1,common


def test_join_keeps_only_six_genuine_decay_windows():
    l1,common=synthetic_results()
    rows=joined_decay_rows(l1,common,['cf_static','cf_kin'])
    assert len(rows)==12
    assert {r['kernel'] for r in rows}=={'decay'}
    assert all(r['common_gain_vs_L1_real']==.5 for r in rows)
    assert len(l1['metrics'])==18
    assert not any('linear' in r['key'] for r in l1['metrics'])


def test_missing_common_window_or_nonmatching_target_cache_rejected():
    l1,common=synthetic_results();partial=copy.deepcopy(common);partial['metrics'].pop()
    with pytest.raises(ValueError,match='twelve'):
        joined_decay_rows(l1,partial,common['arms'])
    changed=copy.deepcopy(common);changed['target_caches']['10']['sha256']='new_target'
    with pytest.raises(ValueError,match='exact same'):
        joined_decay_rows(l1,changed,common['arms'])
    with pytest.raises(ValueError,match='explicitly requested'):
        joined_decay_rows(l1,common,['cf_static'])


def test_running_L2_never_opens_partial_summary(tmp_path):
    left,right=tmp_path/'L1',tmp_path/'L2';left.mkdir();right.mkdir()
    (left/'status.json').write_text(json.dumps(dict(status='complete')))
    (right/'status.json').write_text(json.dumps(dict(status='scoring')))
    # Neither summary exists: the status gate must reject before trying to read one.
    with pytest.raises(ValueError,match='partial results are not opened'):
        join(left,right,tmp_path/'comparison',expected_arms=['cf_static'])
    assert not (tmp_path/'comparison').exists()
