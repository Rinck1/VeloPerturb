"""Direction prior preserves OT alignment and uses absolute PCA geometry."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15e_direction_pilot as module


def test_cosine_prefers_forward_expert_and_ignores_speed_magnitude():
    q=np.array([[.5,.5]],dtype='float32')
    v=np.array([[1.,0.]],dtype='float32')
    experts=np.array([[[2.,0.],[-2.,0.]]],dtype='float32')
    weights,scores=module.directional_weights(q,v,experts,.5)
    np.testing.assert_allclose(scores,[[1.,-1.]])
    assert weights[0,0]>weights[0,1]
    np.testing.assert_allclose(weights,module.directional_weights(q,100*v,3*experts,.5)[0])
    np.testing.assert_array_equal(module.directional_weights(q,v,experts,0.)[0],q)
    np.testing.assert_allclose(module.directional_weights(q,0*v,experts,.5)[0],q)


def test_direction_shuffle_preserves_each_norm_and_direction_set():
    v=np.random.default_rng(9).normal(size=(50,3)).astype('float32')
    shuffled=module.shuffle_directions(v,'group')
    np.testing.assert_allclose(np.linalg.norm(v,axis=1),np.linalg.norm(shuffled,axis=1),rtol=1e-6)
    u=v/np.linalg.norm(v,axis=1,keepdims=True)
    us=shuffled/np.linalg.norm(shuffled,axis=1,keepdims=True)
    np.testing.assert_allclose(np.sort(u,axis=0),np.sort(us,axis=0),atol=1e-6)
    assert not np.allclose(v,shuffled)


def test_cosine_has_no_expert_rotation_gradient():
    import jax
    import jax.numpy as jnp
    q=jnp.array([[.5,.5]])
    v=jnp.array([[1.,0.]])
    experts=jnp.array([[[1.,1.],[-1.,1.]]])
    gradient=jax.grad(lambda e:module.directional_weights(q,v,e,.5)[0][0,0])(experts)
    np.testing.assert_array_equal(gradient,np.zeros_like(experts))


def test_actual_official_ot_update_and_directional_prediction():
    import jax
    import jax.numpy as jnp
    from flax.training.train_state import TrainState
    import optax
    rng=np.random.default_rng(12)
    c=dict(width=8,depth=1,condition_dim=4,time_freqs=8,time_width=8,time_depth=1,
        pooling='mean',learning_rate=.001,ot_epsilon=.1,ot_max_iterations=20)
    z=rng.normal(0,.1,size=(4,3)).astype('float32')
    v=rng.normal(size=z.shape).astype('float32')
    embedding=np.array([.2,.4],dtype='float32')
    condition=module.source_state_conditions(4,0.,embedding)
    model=module.create_model(c,3,0,embedding)
    params=model.init(jax.random.PRNGKey(0),jnp.asarray(z),jnp.asarray(v),jnp.asarray(z),.5,condition)['params']
    state=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adam(.001))
    updated,metrics=module.make_train_step(model,c)(state,jax.random.PRNGKey(2),
        jnp.asarray(z),jnp.asarray(z+.1),jnp.asarray(v),jnp.asarray(v),condition,jnp.asarray(.5))
    assert np.isfinite(float(metrics['loss']))
    assert float(metrics['routing_change'])>0
    route=module.make_route(model)
    q,_=route(updated.params,jnp.asarray(z),jnp.asarray(v),jnp.asarray(v),condition,jnp.asarray(.5))
    np.testing.assert_allclose(np.asarray(q).sum(1),1.,atol=1e-6)
    # Every expert integrates separately; only the source-side q is modified.
    candidates=module.make_candidate_integrator(model)(updated.params,jnp.asarray(z),condition,1.)
    assert np.shape(candidates)==(4,8,3)
    assert np.isfinite(candidates).all()
