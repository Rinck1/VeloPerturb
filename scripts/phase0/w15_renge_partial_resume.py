"""Resume original RENGE job identities without modifying frozen scientific code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

from w4_artifacts import Artifacts,sha256
from w7_router_scaling import atomic_json
import w14b_renge_router as original

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/data/yuchang/veloroute_renge_recovery_20261021')


def jobs_hash(jobs):
    return hashlib.sha256(json.dumps(jobs,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def job_directory(root,job):
    return Path(root)/'training'/job['fold']/f'seed{job["seed"]}'/job['name']


def complete_job(root,job):
    """An absent manifest is pending; a present but invalid result is never reused."""
    directory=job_directory(root,job)
    manifest=directory/'prediction_manifest.json'
    if not manifest.exists():
        state=directory/'status.json'
        if state.exists():
            try:value=json.loads(state.read_text())
            except (ValueError,OSError):value={}
            if value.get('status')=='complete':
                raise ValueError(f'Complete status without frozen prediction manifest: {directory}')
        return False
    try:
        record=json.loads(manifest.read_text())
        if record['job']!=job or record['heldout_future_used']:
            raise ValueError('Prediction identity/source-only contract changed')
        prediction=directory/'predictions.npz'
        if Path(record['path']).resolve()!=prediction.resolve():
            raise ValueError('Prediction points outside original task directory')
        if sha256(prediction)!=record['sha256']:
            raise ValueError('Prediction hash mismatch')
        if sha256(directory/'model.pt')!=record['model_sha256']:
            raise ValueError('Model hash mismatch')
    except Exception as error:
        raise ValueError(f'Invalid completed artifact retained, not overwritten: {directory}: {error}') from error
    return True


def recovery_selection(root,manifest,worker):
    if worker not in (1,2):
        raise ValueError('Only failed original workers1/2 may be replayed; worker0 is untouched')
    complete,pending,partial=[],[],[]
    for job in [j for j in manifest['jobs'] if j['worker']==worker]:
        if complete_job(root,job):
            complete.append(job);continue
        pending.append(job)
        directory=job_directory(root,job)
        if directory.exists():
            if (directory/'model.pt').exists():
                raise ValueError(f'Partial has a checkpoint; explicit checkpoint recovery required: {directory}')
            partial.append(job)
    return dict(complete=complete,pending=pending,partial=partial,
        original_jobs_sha256=jobs_hash(manifest['jobs']),pending_jobs_sha256=jobs_hash(pending))


def tree_hashes(directory):
    return {str(p.relative_to(directory)):sha256(p) for p in sorted(Path(directory).rglob('*')) if p.is_file()}


def preserve_partial(root,job,attempts):
    directory=job_directory(root,job)
    if not directory.is_dir() or (directory/'prediction_manifest.json').exists() or (directory/'model.pt').exists():
        raise ValueError('Only original unfinished tasks without a model may be archived')
    destination=Path(attempts)/f'worker{job["worker"]}'/job['fold']/f'seed{job["seed"]}'/job['name']
    if destination.exists():
        raise ValueError('A prior attempt destination is retained, never overwritten')
    before=tree_hashes(directory)
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(directory),str(destination))
    after=tree_hashes(destination)
    if before!=after:
        raise ArithmeticError('Preserved partial tree changed during archival')
    return dict(original=str(directory.resolve()),preserved=str(destination.resolve()),
        files_sha256=after,bytes=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),
        original_job=job,restarted_from_seed=True,old_checkpoint_modified=False)


def live_original_worker(root,worker):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:args=(proc/'cmdline').read_bytes().split(b'\0');args=[a.decode() for a in args if a]
        except (OSError,UnicodeError):continue
        if not any(Path(a).name=='w14b_renge_router.py' for a in args):continue
        try:
            if args[args.index('--worker')+1]!=str(worker):continue
            if Path(args[args.index('--root')+1]).resolve()!=Path(root).resolve():continue
        except (ValueError,IndexError):continue
        return int(proc.name)
    return None


def require_original_worker0_while_pending(root,manifest):
    if live_original_worker(root,0) is not None:
        return
    # Re-read completed manifests after the process check to avoid a race with
    # worker0 writing its final prediction and exiting during the grid scan.
    if any(not complete_job(root,j) for j in manifest['jobs'] if j['worker']==0):
        raise RuntimeError('Original worker0 exited with pending jobs; retained, not automatically restarted')


def audit(root,output):
    root=Path(root).resolve();manifest=original.verify(root)
    run=Artifacts(output,seed=manifest['config']['seeds'],inputs=[Path(__file__),root/'study_manifest.json'])
    rows=[]
    for index in (0,1,2):
        complete,pending,partial=[],[],[]
        for job in [j for j in manifest['jobs'] if j['worker']==index]:
            if complete_job(root,job):complete.append(job)
            else:
                pending.append(job);directory=job_directory(root,job)
                if directory.exists():
                    state=directory/'status.json'
                    try:value=json.loads(state.read_text())
                    except (OSError,ValueError):value={}
                    partial.append(dict(job=job,path=str(directory.resolve()),last_step=value.get('step'),
                        model_exists=(directory/'model.pt').exists(),live_original_pid=live_original_worker(root,index)))
        rows.append(dict(worker=index,valid_complete=len(complete),pending=len(pending),partial=partial,
                         live_original_pid=live_original_worker(root,index)))
    result=dict(status='ORIGINAL_RENGE_PARTIAL_GRID_AUDITED',original_jobs=len(manifest['jobs']),
        original_jobs_sha256=jobs_hash(manifest['jobs']),workers=rows,
        files_modified=False,workers_started=False)
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    print(json.dumps(result,indent=2));return result


def worker(root,index,output,attempts):
    root,attempts=Path(root).resolve(),Path(attempts).resolve()
    if not attempts.is_relative_to(DATA/'attempts'):
        raise ValueError('Failed attempts must stay in the explicit task data directory')
    manifest=original.verify(root)
    if len(manifest['jobs'])!=576:raise ValueError('Original fixed576 job grid required')
    if live_original_worker(root,index) is not None:
        raise RuntimeError('Original target worker is still alive; do not move its files')
    selection=recovery_selection(root,manifest,index)
    study_hash=sha256(root/'study_manifest.json')
    run=Artifacts(output,seed=manifest['config']['seeds'],inputs=[Path(__file__),root/'study_manifest.json'])
    run.json('selection.json',dict(worker=index,**selection,study_sha256=study_hash,
        frozen_science_unchanged=True,worker0_untouched=True))
    moved=[]
    for job in selection['partial']:
        moved.append(preserve_partial(root,job,attempts))
        run.json('preserved_attempts.json',dict(attempts=moved))
    subset=dict(manifest,jobs=selection['pending'])
    run.json('status.json',dict(status='running_original_pending_jobs',worker=index,
        already_complete=len(selection['complete']),pending=len(selection['pending'])))
    # Only scheduling is filtered in memory. fit(), source_predictions(), seeds,
    # training budgets, original job dictionaries and all frozen files are unchanged.
    with patch.object(original,'verify',lambda value:subset if Path(value).resolve()==root
                      else (_ for _ in ()).throw(ValueError('Unexpected original root'))):
        original.train_worker(root,index)
    original.verify(root)
    if sha256(root/'study_manifest.json')!=study_hash:raise ValueError('Original manifest changed')
    if jobs_hash(manifest['jobs'])!=selection['original_jobs_sha256']:raise ValueError('Original job list changed')
    jobs=[j for j in manifest['jobs'] if j['worker']==index]
    if not all(complete_job(root,j) for j in jobs):raise ValueError('Worker did not complete original subset')
    run.json('summary.json',dict(status='ORIGINAL_RENGE_WORKER_RECOVERED',worker=index,
        originally_complete=len(selection['complete']),replayed=len(selection['pending']),
        total_valid=len(jobs),original_jobs_sha256=selection['original_jobs_sha256']))
    run.json('status.json',dict(status='complete',worker=index,valid=len(jobs)))


def controller(root,output,worker1_job,worker2_job,evaluation_job,evaluation_output,*,publish=False):
    root=Path(root).resolve();manifest=original.verify(root)
    if len(manifest['jobs'])!=576:raise ValueError('Original fixed576 job grid required')
    evaluation_output=Path(evaluation_output).resolve()
    if not evaluation_output.is_relative_to(DATA/'evaluations'):
        raise ValueError('Evaluation artifacts must stay in the explicit task data directory')
    run=Artifacts(output,seed=manifest['config']['seeds'],inputs=[Path(__file__),root/'study_manifest.json'])
    queue=root/'queue_status.json'
    if queue.exists():run.json('previous_queue.json',dict(previous=json.loads(queue.read_text())))
    original_sha=sha256(root/'study_manifest.json');deadline=time.monotonic()+7*86400
    atomic_json(queue,dict(status='partial_recovery_training',controller=str(run.directory.resolve()),
        worker0='original_worker0_continues_untouched',worker1_job=str(worker1_job),worker2_job=str(worker2_job)))
    last=-1
    while time.monotonic()<deadline:
        original.verify(root)
        if sha256(root/'study_manifest.json')!=original_sha:raise ValueError('Original grid changed')
        valid=sum(complete_job(root,j) for j in manifest['jobs'])
        if valid==576:break
        require_original_worker0_while_pending(root,manifest)
        for path in (Path(worker1_job)/'status.json',Path(worker2_job)/'status.json'):
            if path.exists() and json.loads(path.read_text()).get('status')=='failed':
                raise RuntimeError(f'Recovered scientific worker failed: {path}')
        if valid!=last:
            run.json('status.json',dict(status='waiting_for_all_original_predictions',valid=valid,total=576));last=valid
        time.sleep(20)
    else:raise TimeoutError('Original RENGE grid recovery timed out')
    run.json('frozen_original_grid.json',dict(jobs=manifest['jobs'],valid_predictions=576,
        original_jobs_sha256=jobs_hash(manifest['jobs']),study_sha256=original_sha))
    subprocess.run([sys.executable,str(ROOT/'scripts/phase0/launch_gpu_job.py'),'--gpu','1',
        '--script','scripts/phase0/w14b_renge_router.py','--output',str(evaluation_job),'--',
        'evaluate','--root',str(root),'--output',str(evaluation_output)],cwd=ROOT,check=True)
    while time.monotonic()<deadline:
        state=Path(evaluation_job)/'status.json'
        if state.exists():
            value=json.loads(state.read_text())
            if value['status']=='failed':raise RuntimeError('Original RENGE evaluation failed')
            if value['status']=='complete':break
        time.sleep(20)
    else:raise TimeoutError('Original RENGE evaluation timed out')
    result=json.loads((evaluation_output/'summary.json').read_text())
    if result['status']!='RENGE_EXPLORATORY_ROUTER_TRANSFER_COMPLETE':raise ValueError('Evaluation not complete')
    records=run.directory/'evaluation_records';records.mkdir()
    names=('RESULTS.md','summary.json','metrics.csv','gains.csv','frozen_grid.json')
    for name in names:shutil.copyfile(evaluation_output/name,records/name)
    publication=None
    if publish:
        try:
            from w14_publication_recovery import publish_records
            doc=ROOT/'docs/current/w14b_renge_results.md'
            text=(records/'RESULTS.md').read_text()
            if doc.exists() and doc.read_text()!=text:raise ValueError('Prior RENGE report retained')
            if not doc.exists():doc.write_text(text)
            publication=publish_records(ROOT,[doc]+[records/n for n in names],
                'W14-B3: complete original576-job RENGE grid after bounded partial recovery')
        except Exception as error:publication=dict(status='publication_pending',error=repr(error),computation_complete=True)
    run.json('summary.json',dict(status='RENGE_PARTIAL_RECOVERY_PROGRAM_COMPLETE',result=result,publication=publication))
    run.json('status.json',dict(status='complete',computation_complete=True,publication=publication))
    atomic_json(queue,dict(status='complete',controller=str(run.directory.resolve()),evaluation=str(evaluation_output),publication=publication))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','worker','controller'])
    p.add_argument('--root',default='outputs/w14b_renge_router_v1');p.add_argument('--output',required=True)
    p.add_argument('--worker',type=int,choices=[1,2]);p.add_argument('--attempts-root')
    p.add_argument('--worker1-job');p.add_argument('--worker2-job');p.add_argument('--evaluation-job');p.add_argument('--evaluation-output')
    p.add_argument('--publish',action='store_true');a=p.parse_args()
    if a.action=='audit':audit(a.root,a.output)
    elif a.action=='worker':worker(a.root,a.worker,a.output,a.attempts_root)
    else:controller(a.root,a.output,a.worker1_job,a.worker2_job,a.evaluation_job,a.evaluation_output,publish=a.publish)
