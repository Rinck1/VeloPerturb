import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15p_velocity_graph_controls as module
from test_w15n_velocity_auxiliary import fixture


def test_control_graphs_deterministic_local_bijection_and_source_only():
    g,c,_=fixture();g['velocity']=g['velocity_program'];g['U']=g['U_program']
    norm=dict(velocity_reference=np.zeros(2),U_reference=np.zeros(2),axes=np.eye(2))
    class Forbidden:
        def __array__(self,*a,**kw):raise AssertionError('Future target read')
    g['target']=Forbidden()
    plan=dict(global_shuffle_seeds=[1,2],local_shuffle_seed=0)
    parent=dict(state_candidates=5,graph_neighbors=2)
    a,mapping=module.make_controls([g],norm,plan,parent);b,second=module.make_controls([g],norm,plan,parent)
    assert set(a['A'])==set(module.ARMS) and mapping==second
    order=np.array(mapping['A']['permutation'])
    np.testing.assert_array_equal(np.sort(order),np.arange(7))
    assert not (order==np.arange(7)).any()
    for arm in module.ARMS:
        np.testing.assert_array_equal(a['A'][arm],b['A'][arm])
        assert a['A'][arm].shape==(7,2)
        assert not (a['A'][arm]==np.arange(7)[:,None]).any()


def test_average_null_errors_before_forming_gain():
    c=dict(seeds=[0]);split=dict(val=['A']);arms=module.graph.ARMS+module.ARMS
    values=dict.fromkeys(arms,2.);values.update(velocity_graph=1.,shuffled_velocity_graph=1.,global_shuffle_1=2.,global_shuffle_2=3.)
    rows=[dict(KO='A',seed=0,arm=arm,energy=values[arm]) for arm in arms]
    summaries,comparisons=module.paired_summary(rows,c,split)
    result=next(r for r in summaries if r['control']=='mean_three_global_shuffles')
    assert result['energy_gain']==.5
    assert result['all_seeds_positive'] and result['winning_KOs']==1
    assert len(comparisons)==8


def test_parent_grid_alone_rejected_before_target_access(tmp_path,monkeypatch):
    import w15h_traxler_program_router as base
    monkeypatch.setattr(base,'load_pack',lambda *a,**k:pytest.fail('Premature target read'))
    with pytest.raises(ValueError,match='complete parent and new null grid'):
        module.evaluate(tmp_path,dict(seeds=[0]),dict(val=['A']),{},[])
