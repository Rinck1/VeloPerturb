import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15k_traxler_l2_cell as module


def fixture():
    rng=np.random.default_rng(13)
    def group(label,training=False):
        z=rng.normal(0,.1,(7,3)).astype('float32')
        g=dict(label=label,z=z,U_normalized=rng.normal(size=(7,3)).astype('float32'),
            velocity_program=rng.normal(size=(7,2)).astype('float32'),
            shuffled_velocity_program=rng.normal(size=(7,2)).astype('float32'),
            U_program=rng.normal(size=(7,2)).astype('float32'),embedding=np.ones(5,dtype='float32'),
            cell_ids=np.array([label+str(i) for i in range(7)]),time_hours=6,
            score_indices=np.array([0,2,6]))
        if training:g['target']=z+.1
        return g
    c=dict(width=8,depth=1,condition_dim=4,time_freqs=8,time_width=8,time_depth=1,pooling='mean',
        learning_rate=.001,ot_epsilon=.1,ot_max_iterations=20,steps=2,batch_size=4,log_interval=1,
        prediction_batch=4,source_hours=6,target_hours=24,programs=2,seeds=[0])
    return c,[group('A',True),group('B',True)],group('Heldout')


def test_four_arms_have_identical_source_information_slots():
    c,training,_=fixture();g=training[0]
    inputs=[module.l2_condition(g,a,c) for a in module.L2_ARMS]
    assert all(set(x)=={'perturbation','bio_time','bio_delta','source_state','source_rna'} for x in inputs)
    for x in inputs:
        np.testing.assert_array_equal(x['source_state'][:,0],g['z'])
        np.testing.assert_array_equal(x['source_rna'][:,0,:3],g['U_normalized'])
        assert x['source_rna'].shape==(7,1,5)
    assert not inputs[0]['source_rna'][:,:,3:].any()
    np.testing.assert_array_equal(inputs[1]['source_rna'][:,0,3:],g['velocity_program'])
    np.testing.assert_array_equal(inputs[2]['source_rna'][:,0,3:],g['shuffled_velocity_program'])
    np.testing.assert_array_equal(inputs[3]['source_rna'][:,0,3:],g['U_program'])


def test_parameter_matching_is_shape_only_and_matches_actual_native_count():
    import jax
    c,training,_=fixture();condition=module.l2_condition(training[0],module.L2_ARMS[0],c)
    larger=dict(c,width=16)
    target=module.parameter_count(module.native_field(larger,3),condition,3,c['condition_dim'])
    chosen,report=module.match_width(c,condition,target,[8,16],.001)
    assert chosen['width']==16
    assert report['selected']['relative_gap']==0.
    assert report['selection_uses_outcomes'] is False
    solver=module.create_solver(larger,{k:v[:1] for k,v in condition.items()},3,0)
    assert target==sum(x.size for x in jax.tree_util.tree_leaves(solver.vf_state.params))


def test_actual_official_ot_training_checkpoint_restore_and_particle_ODE():
    import jax
    from flax import serialization
    from w15a_cellflow_train import fit_groups
    c,training,heldout=fixture()
    groups=[dict(z=g['z'],target=g['target'],condition=module.l2_condition(g,module.L2_ARMS[1],c)) for g in training]
    solver=module.create_solver(c,{k:v[:1] for k,v in groups[0]['condition'].items()},3,0)
    before=serialization.to_bytes(solver.vf_state.params)
    history=fit_groups(solver,groups,c,0)
    assert history[-1]['step']==2 and np.isfinite(history[-1]['loss'])
    assert before!=serialization.to_bytes(solver.vf_state.params)
    saved=serialization.to_bytes(solver.vf_state_inference)
    solver.vf_state_inference=serialization.from_bytes(solver.vf_state_inference,saved)
    p=module.predict_l2(solver,heldout,module.l2_condition(heldout,module.L2_ARMS[1],c),c)
    assert p['candidates'].shape==(3,1,3)
    assert np.isfinite(p['candidates']).all()
    np.testing.assert_array_equal(p['source_ids'],heldout['cell_ids'][[0,2,6]])
    np.testing.assert_array_equal(p['q'],np.ones((3,1)))


def test_new_source_conditions_follow_duplicated_OT_source_indices():
    from w15a_cellflow_adapter import align_conditions_to_source
    c,training,_=fixture();condition=module.l2_condition(training[0],module.L2_ARMS[1],c)
    order=np.array([5,1,5,0])
    aligned=align_conditions_to_source(condition,order,source_size=7)
    for name in condition:np.testing.assert_array_equal(aligned[name],condition[name][order])


def test_incomplete_l2_grid_never_opens_validation_targets(tmp_path,monkeypatch):
    import w15h_traxler_program_router as base
    monkeypatch.setattr(base,'load_pack',lambda *a,**kw:pytest.fail('Incomplete grid cannot open future'))
    with pytest.raises(ValueError,match='Freeze every'):
        module.evaluate_l2(tmp_path,dict(seeds=[0]),[],dict(val=['A']),{},tmp_path)


def test_mismatched_lane_or_gpu_is_rejected_before_launch(tmp_path):
    with pytest.raises(ValueError,match='assigns L2'):
        module.launch('unused',tmp_path/'out','l2',7,tmp_path/'job')
