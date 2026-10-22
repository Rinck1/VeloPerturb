import copy
import functools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15o_velocity_neighborhood as module
from test_w15n_velocity_auxiliary import fixture


def test_graphs_source_only_local_distinct_and_no_self_edges():
    g,_,_=fixture()
    class Forbidden:
        def __array__(self,*a,**kw):raise AssertionError('Future target read')
    g['target']=Forbidden()
    graphs=module.source_graphs(g,5,2)
    assert set(graphs)==set(module.ARMS)
    for arm,edge in graphs.items():
        assert edge.shape==(7,2)
        assert not (edge==np.arange(7)[:,None]).any()
        assert all(len(set(row))==2 for row in edge)
    np.testing.assert_array_equal(graphs['no_graph'],graphs['state_graph'])


def test_projective_graph_invariant_to_individual_velocity_reversal():
    g,_,_=fixture();original=module.source_graphs(g,6,3)
    changed=copy.deepcopy(g);changed['velocity_program'][::2]*=-1
    reverse=module.source_graphs(changed,6,3)
    np.testing.assert_array_equal(original['velocity_graph'],reverse['velocity_graph'])
    reversed_again=module.source_graphs(g,6,3)
    np.testing.assert_array_equal(original['velocity_graph'],reversed_again['velocity_graph'])


def test_graph_sizes_adapt_to_small_cohort_and_reject_singleton():
    g,_,_=fixture();graphs=module.source_graphs(g,20,10)
    assert graphs['velocity_graph'].shape==(7,6)
    with pytest.raises(ValueError):module.source_graphs(dict(g,z=g['z'][:1]),20,4)


def test_js_is_symmetric_zero_for_equal_and_one_for_disjoint():
    import jax.numpy as jnp
    p=jnp.asarray([[1.,0.],[.1,.9]]);q=jnp.asarray([[0.,1.],[.8,.2]])
    np.testing.assert_allclose(module.normalized_js(p,p),0.,atol=1e-7)
    np.testing.assert_allclose(module.normalized_js(p,q),module.normalized_js(q,p),atol=1e-7)
    np.testing.assert_allclose(module.normalized_js(p[:1],q[:1]),1.,atol=1e-7)


def test_graph_gradient_only_updates_router_and_U_encoder():
    import jax
    import jax.numpy as jnp
    g,c,base=fixture();model=base.create_model(c,3,0,g['embedding'])
    x=jnp.asarray(g['z'][:4]);u=jnp.asarray(module.shared_input(g,2)[:4])
    params=model.init(jax.random.PRNGKey(0),x,u,x,.5,base.condition(g,4,c))['params']
    grad=jax.grad(lambda p:module.graph_loss(model,p,x,u,x[::-1],u[::-1]))(params)
    norm=lambda t:sum(float(jnp.square(v).sum()) for v in jax.tree_util.tree_leaves(t))
    assert norm(grad['router'])>0 and norm(grad['kinetic_encoder'])>0
    assert norm(grad['backbone'])==0
    assert all(norm(value)==0 for key,value in grad.items() if key.startswith('expert_'))


def test_actual_graph_update_and_source_only_ODE_prediction():
    g,c,base=fixture()
    from cellflow.utils import match_linear
    model=base.create_model(c,3,0,g['embedding'])
    step=module.make_step(model,functools.partial(match_linear,epsilon=.1,scale_cost='mean',max_iterations=20))
    graphs={g['label']:module.source_graphs(g,5,2)}
    params,initial,history=module.fit(model,step,[g],graphs,c,dict(graph_weight_normalized=.1),
        'velocity_graph',0,lambda row:None)
    assert history[-1]['step']==2 and history[0]['graph_contribution']>0
    assert np.isfinite(history[-1]['loss'])
    source=dict(g);source.pop('target')
    prediction=base.predict(model,base.make_candidate_integrator(model),params,module.deployment_source(source,2),c)
    assert prediction['candidates'].shape==(3,8,3)
    np.testing.assert_allclose(prediction['q'].sum(1),1.,atol=1e-6)


def test_no_graph_matches_original_initialization_sampling_and_updates():
    from flax import serialization
    g,c,base=fixture();model=base.create_model(c,3,0,g['embedding'])
    from cellflow.utils import match_linear
    matcher=functools.partial(match_linear,epsilon=.1,scale_cost='mean',max_iterations=20)
    graphstep=module.make_step(model,matcher)
    graphs={g['label']:module.source_graphs(g,5,2)}
    graph,init,_=module.fit(model,graphstep,[g],graphs,c,dict(graph_weight_normalized=.1),'no_graph',0,lambda r:None)
    clean=dict(g,kinetic_input=module.shared_input(g,2))
    original,initial,_=base.train(model,base.make_step(model,c),[clean],c,0,lambda r:None)
    assert serialization.to_bytes(init)==serialization.to_bytes(initial)
    import jax
    for a,b in zip(jax.tree_util.tree_leaves(graph),jax.tree_util.tree_leaves(original)):
        np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)


def test_neighbors_follow_repeated_OT_source_indices():
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState
    g,c,base=fixture();model=base.create_model(c,3,0,g['embedding'])
    x=jnp.asarray(g['z'][:4]);u=jnp.asarray(module.shared_input(g,2)[:4]);ns=x[::-1];nu=u[::-1]
    params=model.init(jax.random.PRNGKey(0),x,u,x,.5,base.condition(g,4,c))['params']
    state=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adam(.001))
    coupling=jnp.zeros((4,4)).at[2,:].set(.25)
    _,metrics=module.make_step(model,lambda s,t:coupling)(state,jax.random.PRNGKey(9),x,x+.2,u,ns,nu,
        base.condition(g,4,c),jnp.asarray(.1))
    np.testing.assert_array_equal(metrics['source_indices'],np.full(4,2))
    ix=metrics['source_indices']
    expected=module.graph_loss(model,params,x[ix],u[ix],ns[ix],nu[ix])
    np.testing.assert_allclose(metrics['graph_js'],expected,rtol=1e-5)


def test_incomplete_grid_fails_before_target_read(tmp_path,monkeypatch):
    _,c,base=fixture()
    monkeypatch.setattr(base,'load_pack',lambda *a,**k:pytest.fail('Premature target read'))
    with pytest.raises(ValueError,match='entire neighborhood grid'):
        module.evaluate(tmp_path,c,dict(val=['A']),{},[])
