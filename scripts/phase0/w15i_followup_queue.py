"""CPU-only dispatcher: wait for an unoccupied GPU0, then launch follow-ups.

This module never imports JAX/PyTorch and never terminates another process.
The existing detached job runner owns the child worker and persistent logs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from w4_artifacts import Artifacts

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def gpu0_status():
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    selected = [r for r in csv.reader(devices.splitlines(), skipinitialspace=True) if int(r[0]) == 0]
    if len(selected) != 1: raise RuntimeError('Cannot resolve approved physical GPU0')
    row = selected[0]; uuid = row[1].strip(); memory = int(row[2])
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    pids = [int(p[1]) for p in csv.reader(processes.splitlines(), skipinitialspace=True)
            if len(p) == 2 and p[0].strip() == uuid]
    return dict(physical_gpu=0, uuid=uuid, memory_mib=memory, active_pids=pids,
                free=not pids and memory <= 256)


def worker_environment(runtime):
    runtime = Path(runtime)
    for name in ('tmp', 'jax_cache', 'xdg_cache', 'cuda_cache'):
        (runtime/name).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        LD_LIBRARY_PATH='/data/yuchang/veloroute_w15a/runtime/cuda_deps/nvidia/cudnn/lib',
        TMPDIR=str(runtime/'tmp'), JAX_COMPILATION_CACHE_DIR=str(runtime/'jax_cache'),
        XDG_CACHE_HOME=str(runtime/'xdg_cache'), CUDA_CACHE_PATH=str(runtime/'cuda_cache'),
        PYTHONPATH=str(ROOT/'src'))
    return env


def program(configuration, output, worker_output, worker_job, runtime):
    c = json.loads(Path(configuration).read_text())
    inputs = [Path(configuration), Path(__file__), HERE/'w15i_program_router_followups.py',
              Path(c['base_configuration'])]
    run = Artifacts(output, seed=[0, 1, 2], inputs=inputs)
    deadline = time.monotonic()+c['gpu_wait_hours']*3600
    try:
        run.json('queue_manifest.json', dict(configuration=c, worker_output=str(worker_output),
            worker_job=str(worker_job), new_fits=21, physical_gpu=0, reserved_GPUs=[6, 7],
            other_processes_will_not_be_terminated=True))
        while True:
            parent_path = Path(c['predecessor_job_status'])
            parent = json.loads(parent_path.read_text()) if parent_path.exists() else {}
            if parent.get('status') == 'failed': raise RuntimeError('Predecessor failed')
            state = gpu0_status()
            if parent.get('status') == 'complete' and state['free']: break
            run.json('status.json', dict(status='waiting_for_gpu0', gpu=state,
                predecessor_status=parent.get('status'), planned_new_fits=21))
            if time.monotonic() > deadline: raise TimeoutError('GPU0 wait deadline exceeded')
            time.sleep(30)
        if Path(worker_output).exists() or (ROOT/worker_job).exists():
            raise FileExistsError('New worker output/log directory must not overwrite previous work')
        command = [sys.executable, str(HERE/'launch_gpu_job.py'), '--gpu', '0',
            '--script', str(HERE/'w15i_program_router_followups.py'), '--output', str(worker_job),
            '--', '--configuration', str(Path(configuration).resolve()), '--output', str(worker_output)]
        launched = json.loads(subprocess.check_output(command, cwd=ROOT,
            env=worker_environment(runtime), text=True, timeout=60))
        run.json('launch_receipt.json', launched)
        run.json('status.json', dict(status='worker_running', child=launched, new_fits=21))
        print(json.dumps(dict(launched, status='followup_worker_launched')), flush=True)
        while True:
            path = ROOT/worker_job/'status.json'
            state = json.loads(path.read_text()) if path.exists() else {}
            if state.get('status') == 'failed': raise RuntimeError('Follow-up worker failed; inspect child log')
            if state.get('status') == 'complete':
                summary = json.loads((Path(worker_output)/'summary.json').read_text())
                run.json('summary.json', dict(status='FOLLOWUP_DISPATCH_COMPLETE', child=launched,
                    worker_summary=str(Path(worker_output)/'summary.json'), new_fits=summary['new_fits']))
                run.json('status.json', dict(status='complete', child=launched))
                return
            time.sleep(30)
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc))); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('configuration', 'output', 'worker-output', 'worker-job', 'runtime'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    program(args.configuration, args.output, Path(args.worker_output), Path(args.worker_job), args.runtime)
