"""Recovery scheduling and preservation only; no GPU or real model fitting."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w15_renge_partial_resume import (complete_job,job_directory,jobs_hash,
                                    preserve_partial,recovery_selection,
                                    require_original_worker0_while_pending)
import w15_renge_partial_resume as recovery
from w4_artifacts import sha256


def job(worker=1,seed=0):
    return dict(fold='day2_to_day3__ETV4',worker=worker,seed=seed,arm='real',shuffle_seed=0,name='real')


def completed(root,item):
    d=job_directory(root,item);d.mkdir(parents=True)
    (d/'model.pt').write_bytes(b'frozen model bytes')
    (d/'predictions.npz').write_bytes(b'frozen prediction bytes')
    (d/'status.json').write_text(json.dumps(dict(status='complete')))
    (d/'prediction_manifest.json').write_text(json.dumps(dict(job=item,heldout_future_used=False,
        path=str((d/'predictions.npz').resolve()),sha256=sha256(d/'predictions.npz'),model_sha256=sha256(d/'model.pt'))))
    return d


def test_only_failed_workers_selected_original_grid_not_mutated(tmp_path):
    items=[job(0),job(1),job(2)]
    manifest=dict(jobs=items);before=jobs_hash(items)
    result=recovery_selection(tmp_path,manifest,1)
    assert result['pending']==[items[1]]
    assert result['original_jobs_sha256']==before==jobs_hash(manifest['jobs'])
    assert result['partial']==[]
    with pytest.raises(ValueError,match='worker0'):recovery_selection(tmp_path,manifest,0)


def test_hash_valid_complete_is_skipped_without_touching_bytes(tmp_path):
    item=job();d=completed(tmp_path,item)
    old={p.name:p.read_bytes() for p in d.iterdir() if p.is_file()}
    result=recovery_selection(tmp_path,dict(jobs=[item,job(1,1)]),1)
    assert result['complete']==[item]
    assert result['pending']==[job(1,1)]
    assert {p.name:p.read_bytes() for p in d.iterdir() if p.is_file()}==old


def test_bad_complete_refused_not_retrained_or_archived(tmp_path):
    item=job();d=completed(tmp_path,item)
    (d/'model.pt').write_bytes(b'tampered model')
    with pytest.raises(ValueError,match='Model hash mismatch'):
        recovery_selection(tmp_path,dict(jobs=[item]),1)
    assert d.exists() and (d/'prediction_manifest.json').exists()


def test_partial_is_preserved_byte_exact_in_attempts(tmp_path):
    root=tmp_path/'original';item=job();d=job_directory(root,item);d.mkdir(parents=True)
    (d/'status.json').write_text(json.dumps(dict(status='training',step=9000)))
    (d/'status.json.tmp').write_bytes(b'')
    code=d/'code_snapshot';code.mkdir();(code/'frozen.py').write_text('original scientific code\n')
    result=recovery_selection(root,dict(jobs=[item]),1)
    assert result['partial']==[item] and result['pending']==[item]
    record=preserve_partial(root,item,tmp_path/'attempts')
    target=Path(record['preserved'])
    assert not d.exists() and target.exists()
    assert (target/'status.json.tmp').read_bytes()==b''
    assert (target/'code_snapshot/frozen.py').read_text()=='original scientific code\n'
    assert record['files_sha256']['status.json']==sha256(target/'status.json')
    assert not (target/'model.pt').exists()
    d.mkdir(parents=True);(d/'model.pt').write_bytes(b'checkpoint retained')
    with pytest.raises(ValueError,match='checkpoint'):
        recovery_selection(root,dict(jobs=[item]),1)


def test_original_worker0_exit_is_detected_but_never_restarted(tmp_path,monkeypatch):
    item=job(0);manifest=dict(jobs=[item])
    monkeypatch.setattr(recovery,'live_original_worker',lambda root,worker:None)
    with pytest.raises(RuntimeError,match='not automatically restarted'):
        require_original_worker0_while_pending(tmp_path,manifest)
    completed(tmp_path,item)
    require_original_worker0_while_pending(tmp_path,manifest)
