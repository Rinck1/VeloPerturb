import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15z_short_variance_flux as module
from veloroute.latent import save_pack


def packs(tmp_path, *, source_hours=0, labels=None):
    labels = np.repeat(['KO0', 'KO1'], 4) if labels is None else np.asarray(labels)
    z = np.arange(16, dtype=float).reshape(8, 2)/4
    ids = np.array([f'B{1+i%4//2}:{i}' for i in range(8)])
    source = dict(z=z, velocity=.5*z, u_features=z+1, u_predicted=z,
                  cell_ids=ids, conditions=labels, depth=np.full(8, 50.))
    target = dict(z=2*z, cell_ids=np.array([f'target:{i}' for i in range(8)]), conditions=labels)
    save_pack(tmp_path/'train_source.npz', source, dict(side='source', day=source_hours))
    save_pack(tmp_path/'train_target.npz', target, dict(side='target', day=6))
    (tmp_path/'split.json').write_text(json.dumps(dict(train=['KO0', 'KO1'], validation=['heldout'], test=['sealed'])))
    np.savez(tmp_path/'conditions.npz', conditions=np.array(['KO0', 'KO1']), embeddings=np.eye(2))
    return dict(fold=str(tmp_path), split=str(tmp_path/'split.json'), conditions=str(tmp_path/'conditions.npz'),
                source_hours=0, target_hours=6, shuffle_seeds=[0, 1, 2])


def test_short_loader_only_needs_training_packs_and_computes_known_flux(tmp_path):
    plan = packs(tmp_path)
    data, movement = module.load_training_data(plan)
    np.testing.assert_allclose(data['y'], np.log(4.))
    np.testing.assert_allclose(data['flux']['velocity_positive'], .5*data['variance'])
    assert data['local'].shape == (2, 2, 6)
    assert all(r['moved_fraction'] == 1. for r in movement)
    assert not (tmp_path/'validation_target.npz').exists()
    assert not (tmp_path/'test_target.npz').exists()


def test_short_loader_rejects_wrong_time_window(tmp_path):
    plan = packs(tmp_path, source_hours=1)
    with pytest.raises(ValueError, match='Wrong short-window'):
        module.load_training_data(plan)


def test_short_loader_rejects_unregistered_KOs(tmp_path):
    plan = packs(tmp_path, labels=np.repeat(['KO0', 'heldout'], 4))
    with pytest.raises(ValueError, match='Only registered training'):
        module.load_training_data(plan)
