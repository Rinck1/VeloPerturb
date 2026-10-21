"""W15-A source-only conditions and official CellFlow integration contracts."""
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w15a_cellflow_adapter import (ARMS, TokenNormalizer, SourceCohort,
    add_cellflow_paths, align_conditions_to_source, biological_fourier,
    condition_inputs, effective_hours, fit_source_transform, plugin_network_kwargs,
    pooled_tokens, predict_cellflow_tau, select_source_cohort,
    shuffle_group_tokens, source_tokens, time_schedule)


@pytest.fixture
def toy():
    rng = np.random.default_rng(12)
    groups = [SourceCohort(f'c{i}', 24., 'train',
        sparse.csr_matrix(rng.poisson(5, (24, 12))), sparse.csr_matrix(rng.poisson(2, (24, 12))),
        np.array([f'c{i}_{j}' for j in range(24)])) for i in range(3)]
    allowed = set(np.concatenate([g.cell_ids for g in groups]))
    transform = fit_source_transform(groups, [f'g{i}' for i in range(12)],
        allowed_source_ids=allowed, n_hvg=8, dimensions=3, clusters=2)
    packs = [source_tokens(transform, g, neighbors=4) for g in groups]
    norm = TokenNormalizer.fit(packs, allowed_training_source_keys=[g.key for g in groups])
    return groups, transform, packs, norm


def test_target_changes_do_not_change_any_source_statistics(toy):
    groups, transform, packs, norm = toy
    cohort = groups[0]
    target_s = np.ones((5, 12)); target_u = np.ones((5, 12))
    original = (transform.fingerprint(), norm.fingerprint(), cohort.fingerprint())
    target_s[:] = 1e9; target_u[:] = 1e8
    new_pack = source_tokens(transform, cohort, neighbors=4)
    assert original == (transform.fingerprint(), norm.fingerprint(), cohort.fingerprint())
    for key in ('kin', 'Ssum', 'directU'):
        np.testing.assert_array_equal(packs[0]['population'][key], new_pack['population'][key])
    assert 'target' not in inspect.signature(source_tokens).parameters


def test_selection_uses_exact_source_rows_and_refuses_targets(toy):
    groups, *_ = toy
    g = groups[0]; rows = np.array([0, 3, 7])
    selected = select_source_cohort(g.s, g.u, g.cell_ids, rows, allowed_source_ids=g.cell_ids,
        condition='c', time_hours=1, replicate='r')
    np.testing.assert_array_equal(selected.cell_ids, g.cell_ids[rows])
    np.testing.assert_array_equal(selected.s.toarray(), g.s[rows].toarray())
    with pytest.raises(ValueError, match='forbidden'):
        select_source_cohort(g.s, g.u, g.cell_ids, rows, allowed_source_ids=g.cell_ids[4:],
            condition='c', time_hours=1, replicate='r')


def test_pooled_formula_uses_full_gene_depth_and_no_pca_center_in_derivative(toy):
    groups, transform, *_ = toy
    g = groups[0]; total = g.s.sum(); factor = transform.target_sum/total
    s = np.asarray(g.s[:, transform.genes].sum(0)).ravel()*factor
    u = np.asarray(g.u[:, transform.genes].sum(0)).ravel()*factor
    expected = ((u-transform.gamma*s)/(1+s)) @ transform.components.T
    np.testing.assert_allclose(pooled_tokens(transform, g)['kin'], expected)


def test_earliest_population_is_zero_but_not_artificially_zero_local(toy):
    groups, transform, *_ = toy
    result = source_tokens(transform, groups[0], groups[0], neighbors=4)
    assert all(np.count_nonzero(v) == 0 for v in result['population'].values())
    assert np.count_nonzero(result['cell_kin']) > 0
    future = SourceCohort('c', 48., 'r', groups[0].s, groups[0].u, groups[0].cell_ids)
    with pytest.raises(ValueError, match='future'):
        source_tokens(transform, groups[0], future)


def test_normalizer_fit_requires_training_source_keys(toy):
    groups, _, packs, _ = toy
    with pytest.raises(ValueError, match='forbidden'):
        TokenNormalizer.fit(packs, allowed_training_source_keys=[groups[0].key])


def test_two_times_are_independent_inputs(toy):
    _, _, packs, norm = toy
    kwargs = dict(time_hours=5., condition_delta_hours=24.)
    a = condition_inputs(packs[0], norm, 'cf_kin', **kwargs)
    b = condition_inputs(packs[0], norm, 'cf_kin', **dict(kwargs, time_hours=10.))
    assert not np.array_equal(a['bio_time'], b['bio_time'])
    for key in a:
        if key != 'bio_time':
            np.testing.assert_array_equal(a[key], b[key])
    assert 'tau' not in inspect.signature(condition_inputs).parameters
    np.testing.assert_allclose(biological_fourier(0), np.r_[np.zeros(4), np.ones(4)])


def test_clock_matches_hand_computation_and_seen_span():
    expected = (1-np.exp(-5/24))/(1-np.exp(-1))
    assert time_schedule(5)['tau_end'] == pytest.approx(expected)
    assert time_schedule(5)['condition_delta_hours'] == 24
    assert time_schedule(24)['tau_end'] == 1
    assert time_schedule(6, (6, 24))['tau_end'] == 1
    assert time_schedule(5, kernel='linear')['tau_end'] == 5/24
    assert effective_hours(24) == pytest.approx(24)
    with pytest.raises(ValueError):
        time_schedule(-1)


def test_arm_shapes_and_static_constants_match(toy):
    groups, _, packs, norm = toy
    shuffled, _ = shuffle_group_tokens(packs, groups, norm, seed=0)
    conditions = {arm:condition_inputs(packs[0], norm, arm, time_hours=5, condition_delta_hours=24,
        shuffled_population_token=shuffled[0]) for arm in ARMS}
    shapes = [{k:v.shape for k,v in c.items()} for c in conditions.values()]
    assert all(s == shapes[0] for s in shapes)
    assert not np.any(conditions['cf_static']['population_token'])
    assert not np.any(conditions['cf_static']['cell_token'])
    assert np.any(conditions['cf_kin_cell']['cell_token'])
    assert not np.any(conditions['cf_kin_cell']['population_token'])


def test_group_shuffle_preserves_token_multiset_and_refuses_single_condition(toy):
    groups, _, packs, norm = toy
    shuffled, mapping = shuffle_group_tokens(packs, groups, norm, seed=0)
    real = np.stack([norm.apply('kin', p['population']['kin']) for p in packs])
    np.testing.assert_array_equal(shuffled, real[mapping])
    assert all(groups[i].condition != groups[j].condition for i,j in enumerate(mapping))
    assert set(mapping) == set(range(len(groups)))
    with pytest.raises(ValueError, match='three'):
        shuffle_group_tokens(packs[:1], groups[:1], norm, seed=0)


def test_local_labels_follow_ot_source_resampling_with_duplicates():
    conditions = dict(cell_token=np.arange(12).reshape(4,1,3), shared=np.ones((1,1,2)))
    indices = np.array([3, 1, 1, 0])
    reordered = align_conditions_to_source(conditions, indices, source_size=4)
    np.testing.assert_array_equal(reordered['cell_token'], conditions['cell_token'][indices])
    np.testing.assert_array_equal(reordered['shared'], conditions['shared'])


def test_author_time_origin_is_relative_to_day_three():
    from w15a_cellflow_train import time_from_label
    assert time_from_label('D3') == 0
    assert time_from_label('D4') == 24
    assert time_from_label('D8') == 120
    assert time_from_label('05h') == 5
    assert time_from_label('20h') == 20


def test_scoring_distinguishes_shape_and_mode_proportions():
    from w15a_cellflow_evaluate import score
    y=np.array([[-2.],[-2.],[2.],[2.]])
    centers=np.array([[-2.],[2.]])
    same=score(y,y,centers)
    assert same['energy']==0 and same['mode_proportion_TV']==0
    collapsed=score(np.zeros_like(y),y,centers)
    assert collapsed['energy']>0 and collapsed['mode_proportion_TV']==.5


def test_native_network_params_equal_and_tau_wrapper_matches_full_solve(toy):
    add_cellflow_paths()
    import jax
    import jax.numpy as jnp
    import optax
    from cellflow.networks import ConditionalVelocityField
    from cellflow.solvers import OTFlowMatching
    from cellflow._compat import ConstantNoiseFlow
    groups, _, packs, norm = toy
    shuffled, _ = shuffle_group_tokens(packs, groups, norm, seed=0)
    counts=[]; states=[]; first=None
    vf = ConditionalVelocityField(output_dim=3, max_combination_length=1,
        condition_embedding_dim=4, pooling='mean', regularization=0.,
        time_freqs=8, time_encoder_dims=(8,), hidden_dims=(8,), decoder_dims=(8,),
        **plugin_network_kwargs(dimension=3, condition_dim=4))
    for arm in ARMS:
        inputs=condition_inputs(packs[0], norm, arm, time_hours=5, condition_delta_hours=24,
            shuffled_population_token=shuffled[0])
        state=vf.create_train_state(jax.random.PRNGKey(3), optax.adam(1e-3),
            input_dim=3, conditions={k:jnp.asarray(v[:1]) for k,v in inputs.items()})
        states.append(state)
        counts.append(sum(p.size for p in jax.tree.leaves(state.params)))
        if first is None:first=inputs
    assert len(set(counts)) == 1
    for a,b in zip(jax.tree.leaves(states[0].params), jax.tree.leaves(states[1].params)):
        np.testing.assert_array_equal(a,b)
    solver=OTFlowMatching(vf, ConstantNoiseFlow(0.), match_fn=lambda x,y:jnp.eye(len(x))/len(x),
        rng=jax.random.PRNGKey(3), optimizer=optax.adam(1e-3),
        conditions={k:jnp.asarray(v[:1]) for k,v in first.items()})
    x=packs[0]['z'][:3].astype('float32'); cond={k:jnp.asarray(v[:1]) for k,v in first.items()}
    native=np.asarray(solver.predict(x, condition=cond))
    wrapped=predict_cellflow_tau(solver, x, cond, tau_end=1.)
    np.testing.assert_array_equal(native,wrapped)
    np.testing.assert_array_equal(predict_cellflow_tau(solver,x,cond,tau_end=0.),x)
    particle={k:jnp.repeat(v, len(x),axis=0) for k,v in cond.items()}
    np.testing.assert_array_equal(predict_cellflow_tau(solver,x,particle,tau_end=1.,per_particle=True),native)


def test_source_label_wrapper_keeps_official_shared_condition_update_exact():
    add_cellflow_paths()
    import jax
    import jax.numpy as jnp
    from w15a_cellflow_train import create_solver,source_labeled_step
    config=dict(width=8,depth=1,condition_dim=4,time_freqs=8,time_width=8,time_depth=1,
                learning_rate=.001,ot_epsilon=.1,ot_max_iterations=30)
    cond=dict(perturbation=jnp.ones((1,1,1)),bio_time=jnp.ones((1,1,8)),bio_delta=jnp.ones((1,1,8)),
              population_token=jnp.zeros((1,1,2)),cell_token=jnp.zeros((1,1,2)))
    left=create_solver(config,cond,2,4);right=create_solver(config,cond,2,4)
    batch=dict(src_cell_data=jnp.arange(16,dtype='float32').reshape(8,2)/8,
               tgt_cell_data=jnp.arange(16,dtype='float32').reshape(8,2)/8+.4,condition=cond)
    key=jax.random.PRNGKey(4)
    native=left.step_fn(key,batch);wrapped=source_labeled_step(right,key,batch)
    np.testing.assert_array_equal(native,wrapped)
    for a,b in zip(jax.tree.leaves(left.vf_state.params),jax.tree.leaves(right.vf_state.params)):
        np.testing.assert_array_equal(a,b)


def test_complete_toy_train_freeze_predict_and_score_pipeline(toy,tmp_path):
    import anndata as ad
    import pandas as pd
    from w4_artifacts import sha256
    from w15a_cellflow_train import train
    from w15a_cellflow_predict import predict
    from w15a_cellflow_evaluate import evaluate
    _,transform,packs,norm=toy
    root=tmp_path/'pack';root.mkdir()
    transform.save(root/'transform.npz')
    config=dict(width=8,depth=1,condition_dim=4,time_freqs=8,time_width=8,time_depth=1,
        learning_rate=.001,ot_epsilon=.1,ot_max_iterations=30,batch_size=8,steps=2,log_interval=1,
        evaluation_seed=0,evaluation_source_cells=8,evaluation_target_cells=8,
        selection_intervals_hours=[[5,10]],transform_path=str(root/'transform.npz'))
    p=packs[0]
    values=dict(z=p['z'].astype('float32'),cell_kin=p['cell_kin'].astype('float32'),cell_ids=p['source_ids'],
        **{'population_'+k:v for k,v in p['population'].items()})
    np.savez_compressed(root/'train.npz',**values,target=(p['z']+.1).astype('float32'))
    np.savez_compressed(root/'source.npz',**values)
    np.savez_compressed(root/'token_normalizer.npz',
        **{'mean_'+k:v for k,v in norm.mean.items()},**{'scale_'+k:v for k,v in norm.scale.items()},
        training_source_keys=norm.training_source_keys)
    np.save(root/'training_target_mode_centers.npy',transform.centers)
    s=sparse.csr_matrix(np.full((12,12),500.));u=sparse.csr_matrix(np.full((12,12),100.))
    data=tmp_path/'toy_targets.h5ad'
    ad.AnnData(X=s,obs=pd.DataFrame(dict(rep=['r4']*12,timepoint=['10h']*12),
        index=[f'target{i}' for i in range(12)]),layers=dict(spliced=s,unspliced=u)).write_h5ad(data)
    config['data_path']=str(data)
    train_row=dict(path=str(root/'train.npz'),sha256=sha256(root/'train.npz'),source_key=p['source_key'],time_hours=0.)
    source_row=dict(path=str(root/'source.npz'),sha256=sha256(root/'source.npz'),source_key=p['source_key'],
        time_hours=5.,stop_hours=10.,delta_hours=5.,population_reference_is_self=True)
    (root/'manifest.json').write_text(json.dumps(dict(config=config,training_groups=[train_row],development_sources=[source_row])))
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config))
    fitted=train(root,cfg,tmp_path/'fit',arm='cf_static',seed=0)
    assert fitted['status']=='CELLFLOW_OFFICIAL_TRAINED' and not fitted['r4_targets_read']
    frozen=predict(tmp_path/'fit',tmp_path/'prediction')
    assert not frozen['target_read'] and len(frozen['windows'])==2
    scored=evaluate([tmp_path/'prediction'],tmp_path/'evaluation')
    assert np.isfinite(scored['selection_energy']) and scored['development_only']
    with (tmp_path/'prediction/predictions.npz').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='frozen'):
        evaluate([tmp_path/'prediction'],tmp_path/'tampered_eval')
