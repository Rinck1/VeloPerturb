"""One sequential queue per allowed GPU, independent of external baselines."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from veloroute.artifacts import Run,load_config,save_csv,save_json,utc_now
from veloroute.gpu_policy import enforce_gpu_policy
from veloroute.kang_schedule import worker_stages


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,choices=range(4),required=True)
    args=parser.parse_args()
    visible=enforce_gpu_policy('cuda')
    if visible!=(args.worker,):
        raise ValueError('Queue worker must own exactly its corresponding physical GPU')
    workspace=Path(__file__).resolve().parents[2]
    protocol=workspace/'configs/veloroute_kang_20260914.yaml'
    extension=workspace/'configs/veloroute_kang_extended_20260914.yaml'
    c=load_config(protocol);root=Path(c['root'])
    stages=worker_stages(args.worker)
    queue_root=root/'schedules/mainline_first_v1'
    queue_root.mkdir(parents=True,exist_ok=True)
    # Hold this lock across child processes; a second queue cannot share this GPU.
    with (queue_root/f'gpu{args.worker}.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('A Kang mainline queue already owns this GPU') from error
        with Run(queue_root/f'worker{args.worker}',stage='Kang_mainline_first_queue',kind='engineering',
            config=dict(worker=args.worker,physical_gpu=args.worker,donors=c['donors'][args.worker::4],
                stages=[name for name,_,_ in stages],external_baselines_not_dependencies=True,
                order_only_no_model_or_split_changes=True),
            inputs=[protocol,extension,Path(__file__),*[workspace/script for _,script,_ in stages]]) as run:
            save_csv(run.directory/'schedule.csv',[
                dict(order=i,stage=name,script=script,arguments=json.dumps(arguments))
                for i,(name,script,arguments) in enumerate(stages)])
            (run.directory/'RESULTS.md').write_text('# Kang mainline-first execution\n\n'
                'Execution record, not model performance. Full joint GFG/router/experts runs first; '
                'router controls and full ablations follow on the same GPU. No external baseline dependency. '
                'Real training requires validated S/U folds. See status.json and child training provenance.\n')
            for name,script,arguments in stages:
                command=[sys.executable,str(workspace/script),*arguments]
                state=dict(status='stage_launched',stage=name,worker=args.worker,command=command,updated_utc=utc_now())
                save_json(run.directory/'status.json',state);print(json.dumps(state),flush=True)
                result=subprocess.run(command,cwd=workspace)
                if result.returncode:
                    save_json(run.directory/'status.json',dict(status='failed',stage=name,exit_code=result.returncode,updated_utc=utc_now()))
                    raise RuntimeError(f'Kang {name} failed ({result.returncode}); preserve outputs and inspect before resuming')
            save_json(run.directory/'status.json',dict(status='complete',worker=args.worker,updated_utc=utc_now()))


if __name__=='__main__':
    main()
