import copy
import functools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15n_velocity_auxiliary as module


def fixture():
    import w15h_traxler_program_router as base
    rng=np.random.default_rng(3)
    g=dict(label='A',z=rng.normal(0,.1,(7,3)).astype('float32'),U_normalized=rng.normal(size=(7,3)).astype('float32'),
        embedding=np.ones(5,dtype='float32'),time_hours=6,cell_ids=np.array([f'a{i}' for i in range(7)]),score_indices=np.array([0,2,6]))
    g['target']=g['z']+.2
    g['velocity_program']=rng.normal(size=(7,2)).astype('float32');g['shuffled_velocity_program']=g['velocity_program'][np.roll(np.arange(7),1)]
    g['U_program']=rng.normal(size=(7,2)).astype('float32')
    c=dict(width=8,depth=1,condition_dim=4,time_freqs=8,time_width=8,time_depth=1,pooling='mean',
        learning_rate=.001,ot_epsilon=.1,ot_max_iterations=20,steps=2,batch_size=4,log_interval=1,
        prediction_batch=4,source_hours=6,target_hours=24,programs=2,seeds=[0])
    return g,c,base


def test_same_forward_input_does_not_read_velocity_or_auxiliary_targets():
    g,c,_=fixture()
    class Forbidden:
        def __array__(self,*args,**kwargs):raise AssertionError('Velocity was read')
    for key in ('velocity','velocity_program','shuffled_velocity_program','U_program'):g[key]=Forbidden()
    g.pop('target');source=module.deployment_source(g,c['programs'])
    assert set(source)=={'label','z','embedding','cell_ids','time_hours','score_indices','kinetic_input'}
    assert source['kinetic_input'].shape==(7,5)
    assert not source['kinetic_input'][:,3:].any()
    np.testing.assert_array_equal(source['kinetic_input'][:,:3],g['U_normalized'])
    np.testing.assert_array_equal(module.supervision(g,'no_auxiliary',2),np.zeros((7,2)))


def test_only_auxiliary_targets_change_between_four_arms():
    g,c,_=fixture()
    for arm,key in [('velocity_auxiliary','velocity_program'),('shuffled_velocity_auxiliary','shuffled_velocity_program'),('U_program_auxiliary','U_program')]:
        np.testing.assert_array_equal(module.supervision(g,arm,2),g[key])
    with pytest.raises(ValueError,match='future'):module.deployment_source(g,2)


def test_original_initialization_and_predictions_survive_auxiliary_head_removal():
    import jax
    import jax.numpy as jnp
    from flax import serialization
    g,c,base=fixture();deployed=base.create_model(c,3,0,g['embedding']);model=module.auxiliary_model(deployed,2)
    args=(jnp.asarray(g['z'][:1]),jnp.asarray(module.shared_input(g,2)[:1]),jnp.asarray(g['z'][:1]),.5,base.condition(g,1,c))
    plain=deployed.init(jax.random.PRNGKey(0),*args)['params'];extended=model.init(jax.random.PRNGKey(0),*args)['params']
    clean=module.strip_auxiliary(extended)
    assert serialization.to_bytes(plain)==serialization.to_bytes(clean)
    assert 'auxiliary' in extended['router'] and 'auxiliary' not in clean['router']
    assert sum(x.size for x in jax.tree_util.tree_leaves(extended))-sum(x.size for x in jax.tree_util.tree_leaves(plain))==65*2
    q1=model.apply(dict(params=extended),args[0],args[1],method=model.route)
    q2=deployed.apply(dict(params=clean),args[0],args[1],method=deployed.route)
    np.testing.assert_array_equal(q1,q2)


def test_auxiliary_gradient_updates_shared_representation_not_experts_or_logits():
    import jax
    import jax.numpy as jnp
    g,c,base=fixture();deployed=base.create_model(c,3,0,g['embedding']);model=module.auxiliary_model(deployed,2)
    x=jnp.asarray(g['z'][:4]);u=jnp.asarray(module.shared_input(g,2)[:4]);condition=base.condition(g,4,c)
    params=model.init(jax.random.PRNGKey(0),x,u,x,.5,condition)['params']
    target=jnp.asarray(g['velocity_program'][:4])
    def loss(p):return jnp.mean((model.apply(dict(params=p),x,u,method=model.auxiliary_prediction)-target)**2)
    gradient=jax.grad(loss)(params)
    norm=lambda tree:sum(float(jnp.square(v).sum()) for v in jax.tree_util.tree_leaves(tree))
    assert norm(gradient['router']['hidden'])>0
    assert norm(gradient['kinetic_encoder'])>0
    assert norm(gradient['router']['auxiliary'])>0
    assert norm(gradient['router']['logits'])==0
    assert norm(gradient['backbone'])==0
    assert all(norm(value)==0 for key,value in gradient.items() if key.startswith('expert_'))


def test_zero_weight_matches_original_fm_loss_and_has_zero_auxiliary_head_gradient():
    import jax
    import jax.numpy as jnp
    from w15a_cellflow_moe import paired_flow_matching_loss,PairedBatch
    g,c,base=fixture();plain=base.create_model(c,3,0,g['embedding']);model=module.auxiliary_model(plain,2)
    x=jnp.asarray(g['z'][:4]);u=jnp.asarray(module.shared_input(g,2)[:4]);target=jnp.asarray(g['target'][:4]);cond=base.condition(g,4,c)
    params=model.init(jax.random.PRNGKey(0),x,u,x,.5,cond)['params'];tau=jnp.full((4,1),.5)
    def objective(p):return module.loss_on_paired(model,p,x,u,target,jnp.zeros((4,2)),tau,cond,0.)
    (loss,metrics),gradient=jax.value_and_grad(objective,has_aux=True)(params)
    original,_=paired_flow_matching_loss(plain,dict(params=module.strip_auxiliary(params)),PairedBatch(x,u,target,cond),tau)
    np.testing.assert_allclose(loss,original,rtol=1e-6)
    assert metrics['auxiliary_contribution']==0
    assert all(np.all(np.asarray(v)==0) for v in jax.tree_util.tree_leaves(gradient['router']['auxiliary']))


def test_actual_auxiliary_ot_update_head_discard_and_independent_expert_prediction():
    import jax
    g,c,base=fixture()
    from cellflow.utils import match_linear
    from flax import serialization
    plain=base.create_model(c,3,0,g['embedding']);model=module.auxiliary_model(plain,2)
    matcher=functools.partial(match_linear,epsilon=.1,scale_cost='mean',max_iterations=20)
    step=module.make_step(model,matcher)
    params,initial,history=module.fit(model,step,[g],c,'velocity_auxiliary',0,1.,lambda r:None)
    assert history[-1]['step']==2 and history[0]['auxiliary_head_gradient_norm']>0
    assert history[0]['router_gradient_norm']>0 and np.isfinite(history[-1]['loss'])
    clean=module.strip_auxiliary(params);template=module.strip_auxiliary(initial)
    restored=serialization.from_bytes(template,serialization.to_bytes(clean))
    source=dict(g);source.pop('target')
    prediction=base.predict(plain,base.make_candidate_integrator(plain),restored,module.deployment_source(source,2),c)
    assert prediction['candidates'].shape==(3,8,3)
    np.testing.assert_allclose(prediction['q'].sum(1),1.,atol=1e-6)


def test_incomplete_grid_rejected_before_opening_validation_targets(tmp_path,monkeypatch):
    _,c,base=fixture()
    monkeypatch.setattr(base,'load_pack',lambda *a,**k:pytest.fail('Premature target read'))
    with pytest.raises(ValueError,match='entire auxiliary grid'):module.evaluate(tmp_path,c,dict(val=['A']),{},[])


def test_auxiliary_targets_follow_repeated_OT_source_indices():
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState
    g,c,base=fixture();plain=base.create_model(c,3,0,g['embedding']);model=module.auxiliary_model(plain,2)
    x=jnp.asarray(g['z'][:4]);u=jnp.asarray(module.shared_input(g,2)[:4]);y=jnp.asarray(g['target'][:4])
    targets=jnp.asarray(g['velocity_program'][:4]);cond=base.condition(g,4,c)
    params=model.init(jax.random.PRNGKey(0),x,u,x,.5,cond)['params']
    state=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adam(.001))
    coupling=jnp.zeros((4,4)).at[2,:].set(.25)
    _,metrics=module.make_step(model,lambda s,t:coupling)(state,jax.random.PRNGKey(9),x,y,u,targets,cond,jnp.asarray(1.))
    np.testing.assert_array_equal(metrics['source_indices'],np.full(4,2))
    predicted=model.apply(dict(params=params),x,u,method=model.auxiliary_prediction)
    expected=jnp.mean((predicted[metrics['source_indices']]-targets[metrics['source_indices']])**2)
    np.testing.assert_allclose(metrics['auxiliary_mse'],expected,rtol=1e-6)
