import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from veloroute.artifacts import save_json
from veloroute.coupling import build_responsibilities
from veloroute.full_experiments import condition_token_sets, predict_full_model, train_full_model
from veloroute.full_model import FullConfig
from veloroute.full_smoke import run_full_smoke
from veloroute.full_training import FullTrainConfig
from veloroute.latent import load_pack, save_pack
from veloroute.model import ModelConfig, VeloRoute


def test_combination_map_is_explicit_and_order_invariant(tmp_path):
    path = tmp_path/'tokens.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, conditions=['A', 'B'], embeddings=np.eye(2),
            metadata_json=np.array(json.dumps({'source': 'synthetic', 'frozen': True})))
    with pytest.raises(ValueError, match='Missing'):
        condition_token_sets(path, ['A+B'])
    save_json(tmp_path/'map.json', {'A+B': ['B', 'A']})
    tokens, mask, _ = condition_token_sets(path, ['A+B', 'A'], tmp_path/'map.json')
    np.testing.assert_equal(tokens[0], np.eye(2))
    assert mask.tolist() == [[True, True], [True, False]]
    save_json(tmp_path/'invalid.json', {'A+B': ['A', 'A']})
    with pytest.raises(ValueError, match='distinct'):
        condition_token_sets(path, ['A+B'], tmp_path/'invalid.json')


def test_pair_dynamic_term_changes_k1_coupling():
    model = VeloRoute(ModelConfig(state_dim=2, velocity_dim=2, condition_dim=2, hidden_dim=8,
        router_hidden_dim=8, time_hidden_dim=8, residual_blocks=1, expert_rank=2, n_experts=1, top_k=1))
    with torch.no_grad():
        for parameter in model.field.parameters():
            parameter.zero_()
    z = torch.tensor([[0., 0.], [0., 0.]])
    target = torch.tensor([[-1., 0.], [1., 0.]])
    v = torch.tensor([[-1., 0.], [1., 0.]])
    kwargs = dict(source_conditions=['A', 'A'], target_conditions=['A', 'A'])
    a = build_responsibilities(model, z, target, v, torch.zeros(2, 2), **kwargs)
    b = build_responsibilities(model, z, target, v, torch.zeros(2, 2), lambda_pair_dynamic=.5, **kwargs)
    assert not torch.allclose(a['gamma'], b['gamma'])
    assert b['gamma'][0, 0, 0] > b['gamma'][0, 1, 0]


def test_full_file_training_prediction_evaluation_and_scope(tmp_path):
    config = {'model': asdict(FullConfig(state_dim=3, esm_dim=5, representation_dim=4, condition_dim=8,
        encoder_hidden=16, hidden_dim=16, router_hidden_dim=16, time_hidden_dim=8, residual_blocks=1,
        expert_rank=4, max_experts=3, attention_heads=2, attention_head_dim=4, reference_neighbors=3)),
        'training': asdict(FullTrainConfig()), 'device': 'cpu'}
    result = run_full_smoke(config, tmp_path/'full', seed=9)
    assert result['status'] == 'FULL_ENGINEERING_PASS' and result['teacher_updates'] == 9
    root = tmp_path/'full'
    source, sm = load_pack(root/'fold/train_source.npz', expected_side='source')
    target, tm = load_pack(root/'fold/train_target.npz', expected_side='target')
    sm['kind'] = tm['kind'] = 'engineering'
    save_pack(root/'real_source.npz', source, sm)
    save_pack(root/'real_target.npz', target, tm)
    with pytest.raises(ValueError, match='cannot authorize real'):
        train_full_model(root/'real_source.npz', root/'real_target.npz', root/'toy_protein_embeddings.npz', config,
                         root/'forbidden', combination_path=root/'combinations.json', synthetic_engineering=True)
    with pytest.raises(ValueError, match='G1-GO'):
        train_full_model(root/'real_source.npz', root/'real_target.npz', root/'toy_protein_embeddings.npz', config,
                         root/'forbidden2', combination_path=root/'combinations.json')
    with pytest.raises(ValueError, match='Confirmation'):
        predict_full_model(root/'train/model.pt', root/'fold/confirmation_source.npz', root/'toy_protein_embeddings.npz',
                           root/'forbidden3', combination_path=root/'combinations.json')
