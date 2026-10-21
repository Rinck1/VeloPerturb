"""Fixed18 grid and no-score-before216-window orchestration tests."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_artifacts import sha256
import w15a_cellflow_registered_grid as registered
import w15a_cellflow_common_evaluation as common
from test_w15a_common_evaluation import reference,forecasts


def test_only_six_registered_arms_three_training_seeds_single_permutation():
    jobs=registered.jobs()
    assert len(jobs)==18 and len({(j['arm'],j['seed']) for j in jobs})==18
    assert sum(j['reuse_static_seed0'] for j in jobs)==1
    assert {j['seed'] for j in jobs}=={0,1,2}
    shuffles=[j for j in jobs if j['local_shuffle_seed'] is not None]
    assert len(shuffles)==3 and {j['local_shuffle_seed'] for j in shuffles}=={0}
    assert 'cf_kin_shuf' not in {j['arm'] for j in jobs}


@pytest.fixture
def frozen(tmp_path,monkeypatch):
    cfg=dict(width=128,depth=2,steps=5000,learning_rate=.0003)
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    pack=tmp_path/'pack';pack.mkdir();(pack/'manifest.json').write_text('{}')
    np.save(pack/'training_target_mode_centers.npy',np.array([[0.,0.,0.],[1.,1.,1.]]))
    candidate=tmp_path/'old_static0';training=candidate/'train';training.mkdir(parents=True)
    (training/'model.msgpack').write_bytes(b'fixed_static_seed0')
    summary=dict(arm='cf_static',seed=0,config=cfg,model_sha256=sha256(training/'model.msgpack'),history=[dict(step=5000)])
    (training/'summary.json').write_text(json.dumps(summary))
    record=dict(freeze_commit='toy_commit',cf_static_frozen=True,configuration_sha256=sha256(path),
        source_pack_manifest_sha256=sha256(pack/'manifest.json'),source_hashes={},winner=dict(directory=str(candidate)))
    freeze=tmp_path/'freeze.json';freeze.write_text(json.dumps(record))
    monkeypatch.setattr(registered,'commit_bytes',lambda *args:path.read_bytes())
    return path,freeze,pack,training,cfg,record


def test_static0_reuse_requires_equal_configuration_and_model_bytes(frozen):
    cfg,record_path,pack,training,configuration,record=frozen
    assert registered.reuse_static_training(record,configuration)[0]==training
    with pytest.raises(ValueError,match='recipe'):
        registered.reuse_static_training(record,dict(configuration,steps=4999))
    (training/'model.msgpack').write_bytes(b'mutated')
    with pytest.raises(ValueError,match='checkpoint'):
        registered.reuse_static_training(record,configuration)


def test_freeze_commit_and_pack_bytes_are_enforced(frozen,monkeypatch):
    cfg,record_path,pack,*_=frozen
    registered.validate_freeze(cfg,record_path,pack)
    monkeypatch.setattr(registered,'commit_bytes',lambda *args:b'not_the_selected_config')
    with pytest.raises(ValueError,match='commit bytes'):
        registered.validate_freeze(cfg,record_path,pack)


def test_all18_models_and216_predictions_frozen_before_scoring(reference,frozen,tmp_path,monkeypatch):
    cfg,record_path,pack,*_=frozen;output=tmp_path/'grid';training_calls=[];prediction_calls=[]
    def train(pack,configuration_path,directory,*,arm,seed):
        directory=Path(directory);directory.mkdir(parents=True)
        (directory/'model.msgpack').write_bytes(f'{arm}_{seed}'.encode())
        model_sha=sha256(directory/'model.msgpack')
        (directory/'summary.json').write_text(json.dumps(dict(arm=arm,seed=seed,model_sha256=model_sha)))
        training_calls.append((arm,seed));return dict(model_sha256=model_sha)
    def predict(training,directory,reference_root):
        summary=json.loads((Path(training)/'summary.json').read_text())
        values=forecasts(reference)
        for value in values.values():value['x']=value['x']+.1
        result=common.save_common_predictions(directory,values,reference_root=reference,
            arm=summary['arm'],seed=summary['seed'])
        prediction_calls.append((summary['arm'],summary['seed']))
        return result
    real_evaluate=common.evaluate_common
    def evaluate(paths,directory,**kwargs):
        assert len(prediction_calls)==18
        index=json.loads((output/'whole_grid_frozen.json').read_text())
        assert index['model_count']==18 and index['window_count']==216
        assert len(paths)==18
        return real_evaluate(paths,directory,**kwargs)
    monkeypatch.setattr(registered,'train',train)
    monkeypatch.setattr(registered,'predict_common',predict)
    monkeypatch.setattr(registered,'evaluate_common',evaluate)
    result=registered.run_grid(cfg,record_path,pack,reference,output,approved=True)
    assert len(training_calls)==17 and ('cf_static',0) not in training_calls
    assert len(prediction_calls)==18
    assert result['models']==18 and result['all_216_source_only_windows_frozen_before_any_scoring']
    assert result['population_CF_H3']=='NOT_TESTED_SINGLE_CONDITION'
    assert result['local_shuffle_permutation_seeds']==[0]


def test_unapproved_grid_never_reads_configuration():
    with pytest.raises(ValueError,match='approval'):
        registered.run_grid('missing_config','missing_freeze','missing_pack','missing_reference','missing_output')
