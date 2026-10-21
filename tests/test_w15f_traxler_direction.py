import sys
from pathlib import Path
import numpy as np
import pytest
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15f_traxler_direction as module


def test_six_hour_condition_and_unchanged_source_directions():
    g=dict(z=np.ones((4,3)),embedding=np.ones(7),time_hours=0.,label='Stat1',
        cell_kin=np.arange(12).reshape(4,3),cell_directU=np.arange(12).reshape(4,3)+1,
        absolute_velocity=np.arange(12).reshape(4,3).astype('float32'))
    norm={k:dict(mean=np.zeros(3),scale=np.ones(3)) for k in ('U','kin')}
    conditions=module.condition(g,4,6)
    expected=module.source_state_conditions(4,0,np.ones(7),delta_hours=6)
    np.testing.assert_array_equal(conditions['bio_delta'],expected['bio_delta'])
    assert not np.allclose(conditions['bio_delta'],module.source_state_conditions(4,0,np.ones(7))['bio_delta'])
    assert not module.apply_arm(g,norm,'static')['kinetic_input'].any()
    np.testing.assert_array_equal(module.apply_arm(g,norm,'direction')['absolute_velocity'],g['absolute_velocity'])
    np.testing.assert_array_equal(module.apply_arm(g,norm,'directU')['kinetic_input'],g['cell_directU'])
    with pytest.raises(ValueError,match='Unknown'):
        module.apply_arm(g,norm,'unknown')


def test_prepare_reads_training_targets_and_validation_source_only(tmp_path,monkeypatch):
    split=tmp_path/'split.json'
    split.write_text(json.dumps(dict(train=['A','B'],val=['C'],test=['D'])))
    conditions=tmp_path/'conditions.npz'
    np.savez_compressed(conditions,conditions=np.array(['A','B','C','D']),embeddings=np.eye(4),
        metadata_json=np.array(json.dumps(dict(frozen=True,no_cell_expression_used=True))))
    def source(labels,offset=0.):
        n=len(labels)
        return dict(z=np.ones((n,3)),velocity=np.arange(n*3).reshape(n,3)+offset,
            u_features=np.ones((n,3))+offset,conditions=np.array(labels),
            cell_ids=np.array([f'{label}_{i}' for i,label in enumerate(labels)]))
    inputs={'train_source.npz':source(['A','A','B','B']),
            'train_target.npz':dict(z=np.ones((4,3)),conditions=np.array(['A','A','B','B'])),
            'validation_source.npz':source(['C','C'],1e6)}
    calls=[]
    def reader(path,expected_side):
        calls.append(path.name)
        if path.name not in inputs:pytest.fail('Validation/test future opened during preparation')
        return inputs[path.name],{}
    monkeypatch.setattr(module,'load_pack',reader)
    groups,validation,norm,s=module.prepare_groups(dict(fold=str(tmp_path),split=str(split),
        conditions=str(conditions),evaluation_cells=2))
    assert calls==['train_source.npz','train_target.npz','validation_source.npz']
    assert [g['label'] for g in groups]==['A','B']
    assert 'target' not in validation[0]
    np.testing.assert_array_equal(norm['kin']['mean'],inputs['train_source.npz']['velocity'].mean(0))
