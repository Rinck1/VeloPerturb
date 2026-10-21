"""CPU fixtures for fixed RENGE queue, identity freeze, and target-read ordering."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'/'phase0'))
import w15c_renge_program as queue


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    pack = tmp_path/'pack'
    config = tmp_path/'config.json'
    code = tmp_path/'implementation.py'
    code.write_text('frozen implementation')
    write_json(config, dict(steps=5000, batch_size=128, learning_rate=.0003))
    folds = []
    for tf in range(16):
        for day in (2,3):
            name = f'day{day}_to_day{day+1}__TF{tf:02d}'
            path = pack/name/'manifest.json'
            write_json(path, dict(name=name, TF=f'TF{tf:02d}', source_day=day,target_day=day+1))
            folds.append(dict(name=name,path=str(path),sha256=queue.sha256(path)))
    write_json(pack/'study_manifest.json',dict(folds=folds))
    write_json(pack/'status.json',dict(status='ready'))
    monkeypatch.setattr(queue, 'code_inputs', lambda configuration,pack:[Path(configuration),Path(pack)/'study_manifest.json',code])
    path = tmp_path/'transfer_lock.json'
    lock = queue.freeze_lock(config,pack,path)
    return dict(path=path,lock=lock,pack=pack,config=config,code=code)


def create_fit(root,job,lock, *, receipt=True):
    directory = queue.job_directory(root,job)
    directory.mkdir(parents=True)
    (directory/'model.msgpack').write_bytes(f'{job}'.encode())
    n,k = 2,1 if job['lane']=='L2' else 8
    np.savez_compressed(directory/'predictions.npz',candidates=np.zeros((n,k,50)),
        q=np.full((n,k),1/k),source_ids=np.array(['source0','source1']))
    write_json(directory/'prediction_manifest.json',dict(fold=job['fold'],variant=job['variant'],seed=job['seed'],
        path=str(directory/'predictions.npz'),sha256=queue.sha256(directory/'predictions.npz'),
        model_sha256=queue.sha256(directory/'model.msgpack'),heldout_future_used=False))
    write_json(directory/'status.json',dict(status='complete'))
    if receipt:
        write_json(directory/'queue_receipt.json',dict(job=job,lock_sha256=lock['lock_sha256']))
    return directory


def test_fixed_1248_fit_identity_set_and_gpu_bounds(frozen):
    lock = frozen['lock']
    assert len(lock['jobs']['L2'])==672
    assert len(lock['jobs']['L3'])==576
    assert len({(j['lane'],j['fold'],j['variant'],j['seed']) for lane in queue.LANES for j in lock['jobs'][lane]})==1248
    assert lock['physical_gpus']=={'L2':2,'L3':3}
    assert lock['mean_anchor'] is False
    assert queue.verify_lock(frozen['path'])['lock_sha256']==lock['lock_sha256']


def test_fold_design_rejects_missing_interval_and_changed_manifest(frozen):
    folds = frozen['lock']['folds']
    with pytest.raises(ValueError,match='exactly'):
        queue.verify_fold_design(folds[:-1])
    manifest = Path(folds[0]['path'])
    content = json.loads(manifest.read_text())
    content['target_day']=4
    write_json(manifest,content)
    with pytest.raises(ValueError,match='changed'):
        queue.verify_lock(frozen['path'])


def test_code_or_config_change_refuses_resume(frozen):
    frozen['code'].write_text('changed implementation')
    with pytest.raises(ValueError,match='Frozen transfer code'):
        queue.verify_lock(frozen['path'])


@pytest.mark.parametrize('lane', ['L2','L3'])
def test_every_complete_fit_frozen_before_any_target_access(frozen,tmp_path,monkeypatch,lane):
    root=tmp_path/lane
    jobs=frozen['lock']['jobs'][lane]
    monkeypatch.setattr(queue,'_source_ids',lambda *args:np.array(['source0','source1']))
    for job in jobs:
        create_fit(root,job,frozen['lock'])
    target_reads=[]
    def target(fold):
        bound=json.loads((root/'whole_grid_frozen.json').read_text())
        assert bound['total']==len(jobs)
        assert len(bound['predictions'])==len(jobs)
        assert (root/'evaluation'/'frozen_grid.json').is_file()
        target_reads.append(fold['name'])
        fm=json.loads(Path(fold['path']).read_text())
        return np.ones((2,50)),{key:fm[key] for key in ('TF','source_day','target_day')}
    monkeypatch.setattr(queue,'load_target',target)
    import w14_loro
    monkeypatch.setattr(w14_loro,'energy',lambda x,w,y,device:float(w.sum()))
    result=queue.evaluate(frozen['path'],root,lane,root/'evaluation',device='cpu')
    assert result['models']==len(jobs)
    assert len(target_reads)==32
    assert all(abs(value)<1e-10 for value in result['gain_by_control'].values())


def test_incomplete_lane_never_reads_targets(frozen,tmp_path,monkeypatch):
    root=tmp_path/'L2'
    monkeypatch.setattr(queue,'_source_ids',lambda *args:np.array(['source0','source1']))
    monkeypatch.setattr(queue,'load_target',lambda *a:pytest.fail('Targets read from incomplete lane'))
    for job in frozen['lock']['jobs']['L2'][:-1]:
        create_fit(root,job,frozen['lock'])
    with pytest.raises(ValueError,match='Incomplete fit'):
        queue.evaluate(frozen['path'],root,'L2',root/'evaluation',device='cpu')
    assert not (root/'whole_grid_frozen.json').exists()


def test_partial_fit_preserved_without_adapter_retry(frozen,tmp_path,monkeypatch):
    root=tmp_path/'L2'
    job=frozen['lock']['jobs']['L2'][0]
    directory=queue.job_directory(root,job)
    directory.mkdir(parents=True)
    checkpoint=directory/'checkpoint.msgpack'
    checkpoint.write_bytes(b'preserve me')
    monkeypatch.setattr(queue,'_source_ids',lambda *args:np.array(['source0','source1']))
    monkeypatch.setattr(queue.importlib,'import_module',lambda *a:SimpleNamespace(train_predict=lambda *a,**k:pytest.fail('Incomplete fit retried')))
    with pytest.raises(ValueError,match='Incomplete fit'):
        queue.worker(frozen['path'],root,'L2',require_gpu=False)
    assert checkpoint.read_bytes()==b'preserve me'
    assert json.loads((root/'status.json').read_text())['automatic_retry'] is False


def test_hash_source_ids_and_checkpoint_tampering_refused(frozen,tmp_path):
    job=frozen['lock']['jobs']['L3'][0]
    directory=create_fit(tmp_path/'L3',job,frozen['lock'])
    ids=np.array(['source0','source1'])
    queue.verify_fit(directory,job,source_ids=ids,lock_sha256=frozen['lock']['lock_sha256'])
    with pytest.raises(ValueError,match='source IDs'):
        queue.verify_fit(directory,job,source_ids=ids[::-1])
    (directory/'model.msgpack').write_bytes(b'altered weights')
    with pytest.raises(ValueError,match='artifact changed'):
        queue.verify_fit(directory,job,source_ids=ids)


def test_launcher_only_uses_approved_two_physical_gpus(tmp_path,monkeypatch):
    commands=[]
    def launch(command,**kwargs):
        commands.append((command,kwargs))
        return json.dumps(dict(status='LAUNCHED'))
    monkeypatch.setattr(queue.subprocess,'check_output',launch)
    queue.launch_lane('L2',['worker'],tmp_path/'L2')
    queue.launch_lane('L3',['worker'],tmp_path/'L3')
    assert [args[args.index('--gpu')+1] for args,_ in commands]==['2','3']
    assert all(kwargs['env']['JAX_PLATFORMS']=='cuda' for _,kwargs in commands)
    assert all('/cuda_deps/nvidia/cudnn/lib' in kwargs['env']['LD_LIBRARY_PATH'] for _,kwargs in commands)
    assert all(kwargs['env']['CUDA_CACHE_PATH'].startswith('/data/') for _,kwargs in commands)
    with pytest.raises(ValueError,match='Only GPU2'):
        queue.launch_lane('GPU0',[],tmp_path/'no')


def test_independent_fit_caches_are_released(monkeypatch):
    calls=[]
    monkeypatch.setitem(queue.sys.modules,'jax',SimpleNamespace(clear_caches=lambda:calls.append('jax')))
    monkeypatch.setattr(queue.gc,'collect',lambda:calls.append('gc'))
    queue.release_fit_caches()
    assert calls==['jax','gc']


def test_approved_controller_launches_both_lanes_without_waiting_for_first(frozen,tmp_path,monkeypatch):
    launches=[]
    def launch(lane,args,directory):
        launches.append((lane,args))
        write_json(Path(directory)/'status.json',dict(status='complete'))
        return dict(status='LAUNCHED',physical_gpu=queue.LANES[lane]['gpu'])
    monkeypatch.setattr(queue,'launch_lane',launch)
    result=queue.program(frozen['pack'],frozen['config'],tmp_path/'program',tmp_path/'jobs',
                         synthetic_gate='full-synthetic',approved=True,poll_seconds=.01)
    assert [lane for lane,_ in launches]==['L2','L3']
    assert result['total_models']==1248
    assert result['status']=='RENGE_L2_L3_TRANSFER_COMPLETE'


def test_controller_requires_explicit_approval_and_bounded_deadline():
    with pytest.raises(ValueError,match='Explicit approval'):
        queue.program('pack','config','out','jobs',synthetic_gate='synthetic')
    with pytest.raises(ValueError,match='bounded'):
        queue.program('pack','config','out','jobs',synthetic_gate='synthetic',approved=True,deadline_hours=49)
