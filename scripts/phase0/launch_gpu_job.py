"""Detached, logged, bounded project jobs that survive a chat/terminal interruption."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from veloroute.artifacts import save_json
from veloroute.gpu_policy import ALLOWED_PHYSICAL_GPUS, enforce_gpu_policy, gpu_uuids

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--worker')
parser.add_argument('--gpu', type=int, choices=ALLOWED_PHYSICAL_GPUS)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--script')
parser.add_argument('--output')
parser.add_argument('arguments', nargs=argparse.REMAINDER)
args = parser.parse_args()
if args.worker:
    directory = Path(args.worker)
    job = json.loads((directory/'job.json').read_text())
    os.environ['CUDA_VISIBLE_DEVICES'] = job['gpu_uuid'] or ''
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '4'
    enforce_gpu_policy('cpu' if job['physical_gpu'] is None else 'cuda')
    save_json(directory/'status.json', dict(status='running', pid=os.getpid(), physical_gpu=job['physical_gpu']))
    result = subprocess.run(job['command'], cwd=ROOT, env=os.environ.copy())
    save_json(directory/'status.json', dict(status='complete' if result.returncode == 0 else 'failed',
        exit_code=result.returncode, physical_gpu=job['physical_gpu'], finished_utc=datetime.now(timezone.utc).isoformat()))
    sys.exit(result.returncode)
else:
    if (args.gpu is None) == (not args.cpu) or not args.script or not args.output:
        parser.error('Exactly one of --gpu or --cpu, plus --script and --output, is required')
    script = (ROOT/args.script).resolve()
    if not script.is_relative_to(ROOT/'scripts/phase0') or not script.is_file():
        raise ValueError('Only explicit project phase0 scripts may be launched')
    directory = (ROOT/args.output).resolve()
    if not directory.is_relative_to(ROOT/'outputs'):
        raise ValueError('Job records must remain under project outputs')
    directory.mkdir(parents=True, exist_ok=False)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    job = dict(physical_gpu=args.gpu, gpu_uuid=None if args.cpu else gpu_uuids()[args.gpu],
        command=[sys.executable, str(script), *arguments], cwd=str(ROOT),
        allowed_gpus=list(ALLOWED_PHYSICAL_GPUS), reserved_gpus=[4, 5, 6, 7],
        started_utc=datetime.now(timezone.utc).isoformat())
    save_json(directory/'job.json', job)
    with (directory/'job.log').open('xb') as stream:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(directory)],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    save_json(directory/'launcher.json', dict(pid=process.pid, detached=True, physical_gpu=args.gpu))
    print(json.dumps(dict(status='LAUNCHED', pid=process.pid, physical_gpu=args.gpu, job_directory=str(directory))))
