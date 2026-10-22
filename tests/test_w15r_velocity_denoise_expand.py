from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15r_velocity_denoise_expand as module
from test_w15n_velocity_auxiliary import fixture


def test_operator_stochastic_self_preserving_and_never_crosses_KO():
    z=np.array([[0.],[1.],[2.],[3.],[4.],[5.]])
    labels=np.array(['A']*3+['B']*3);T=module.source_operator(z,labels,5,.5).toarray()
    np.testing.assert_allclose(T.sum(1),1.)
    np.testing.assert_allclose(T.diagonal(),.5)
    assert not T[:3,3:].any() and not T[3:,:3].any()
    np.testing.assert_allclose(T@np.ones(6),np.ones(6))
    with pytest.raises(ValueError,match='singleton'):module.source_operator(z[:1],labels[:1],5,.5)


def test_count_moment_teacher_is_not_averaged_unit_velocity_and_keeps_shared_input():
    g,c,_=fixture();rng=np.random.default_rng(5)
    s=rng.uniform(1,5,(7,3));u=rng.uniform(0,5,(7,3))
    transform=SimpleNamespace(mean=np.zeros(3),components=np.eye(3),u_mean=np.zeros(3),
        gamma=np.array([.2,.5,.8]),supported=np.ones(3,dtype=bool),velocity_scale=1.)
    norm=dict(axes=np.eye(3)[:2],velocity_reference=np.zeros(3),U_reference=np.zeros(3))
    source=dict(cell_ids=g['cell_ids'],z=g['z'],conditions=np.array(['A']*7))
    T=module.source_operator(g['z'],source['conditions'],3,.5)
    groups,updated=module.teacher_groups([g],source,s,u,T,transform,norm)
    expected=module.counts.latent(T@s,T@u,transform,norm)
    np.testing.assert_allclose(groups[0]['velocity'],expected['velocity_arrow50'],rtol=1e-6)
    np.testing.assert_array_equal(groups[0]['z'],g['z'])
    np.testing.assert_array_equal(groups[0]['U_normalized'],g['U_normalized'])
    assert not np.allclose(expected['velocity_arrow50'],T@module.counts.latent(s,u,transform,norm)['velocity_arrow50'])
    np.testing.assert_allclose(updated['velocity_reference'],expected['velocity_arrow50'].mean(0))


def test_six_graph_controls_source_only_and_deterministic():
    g,c,_=fixture();g['velocity']=g['velocity_program'];g['U']=g['U_program']
    class Forbidden:
        def __array__(self,*a,**kw):raise AssertionError('Future target read')
    g['target']=Forbidden();norm=dict(axes=np.eye(2),velocity_reference=np.zeros(2),U_reference=np.zeros(2))
    plan=dict(global_shuffle_seeds=[1,2],local_shuffle_seed=0);gp=dict(state_candidates=5,graph_neighbors=2)
    first,_,_=module.build_graphs([g],norm,plan,gp);second,_,_=module.build_graphs([g],norm,plan,gp)
    assert set(first['A'])==set(module.ARMS)
    for arm in module.ARMS:
        np.testing.assert_array_equal(first['A'][arm],second['A'][arm])
        assert not (first['A'][arm]==np.arange(7)[:,None]).any()


def test_scoring_rejects_partial_grid_before_any_future_read(tmp_path,monkeypatch):
    import w15h_traxler_program_router as base
    monkeypatch.setattr(base,'load_pack',lambda *a,**k:pytest.fail('Premature future target'))
    with pytest.raises(ValueError,match='Entire expansion grid'):
        module.evaluate(tmp_path,{},dict(seeds=[0]),dict(val=['A']),[])


def test_factorial_comparisons_use_same_case_U_nulls_and_external_U_reference():
    c=dict(seeds=[0]);split=dict(val=['A']);rows=[]
    for case in ('raw_w01',)+module.CASE_NAMES:
        for arm in module.ARMS+('no_graph',):
            energy=1. if arm=='velocity_graph' else 2.
            if arm=='global_shuffle_2':energy=5.
            rows.append(dict(case=case,arm=arm,KO='A',seed=0,energy=energy))
    rows.extend(dict(case='input_reference',arm=arm,KO='A',seed=0,energy=4.) for arm in module.REFERENCE_ARMS)
    summaries,_=module.comparison_summary(rows,c,split)
    get=lambda case,key:next(r for r in summaries if r['case']==case and r['control']==key)
    assert get('smooth_w01','direct_U_input')['energy_gain']==.75
    assert get('smooth_w01','same_case_mean_three_shuffles')['energy_gain']==pytest.approx(2/3)
    assert get('smooth_w01','same_case_U_graph')['energy_gain']==.5
    assert get('smooth_w01','original_velocity_graph')['energy_gain']==0.


def test_strength_cases_fixed_without_changing_model_budget():
    import json
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'configs/w15r_velocity_denoise_expand_20261022_v1.json').read_text())
    assert tuple(c['name'] for c in plan['cases'])==module.CASE_NAMES
    assert plan['primary_candidate']=='smooth_w01'
    assert [c['gpu'] for c in plan['cases']]==[0,2,3]
    assert plan['steps']==5000 and len(plan['model_seeds'])*len(module.ARMS)*len(module.CASE_NAMES)==54
