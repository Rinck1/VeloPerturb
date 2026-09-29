"""Kang metadata/index/extraction/USA/fold stages, with explicit readiness checks."""
import argparse
import json
import os
from pathlib import Path
import time

from veloroute.artifacts import load_config, save_json
from veloroute.kang_data import extract_bam, freeze_metadata, import_counts, prepare_kang_fold
from veloroute.raw import run_command

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['freeze','index','sample','folds'])
parser.add_argument('--config', default='configs/veloroute_kang_20260914.yaml')
parser.add_argument('--condition', choices=['ctrl','stim'])
parser.add_argument('--wait-hours', type=float, default=48)
args = parser.parse_args()
c = load_config(args.config)
root = Path(c['root'])
prep = c['preprocessing']
os.environ['TMPDIR'] = str(root/'tmp')
env = dict(ALEVIN_FRY_HOME=str(Path('tools/af-home').resolve()), TMPDIR=str(root/'tmp'))
tool = str(Path('tools/rna-quant/bin/simpleaf').resolve())
index = root/'reference/splici_r98_v2'


def completed(path):
    path = Path(path)/'provenance.json'
    if not path.exists(): return False
    state = json.loads(path.read_text())['status']
    if state == 'failed': raise RuntimeError(f'Dependency failed; inspect without overwriting: {path}')
    return state == 'complete'


def wait_for(paths):
    deadline = time.monotonic()+args.wait_hours*3600
    while not all(completed(p) for p in paths):
        if time.monotonic() >= deadline: raise TimeoutError('Data dependency wait budget exhausted')
        print(json.dumps(dict(status='waiting', dependencies=[str(p) for p in paths])), flush=True)
        time.sleep(30)


if args.stage == 'freeze':
    print(freeze_metadata(c, root/'protocol_freeze'), flush=True)
elif args.stage == 'index':
    run_command([tool,'index','--fasta',str(Path(prep['reference_fasta']).resolve()),'--gtf',
        str(Path(prep['reference_gtf']).resolve()),'--rlen','98','--ref-type','spliced+intronic',
        '--threads','8','--output',str(index),'--work-dir',str(index/'work'),'--ram-limit-gib','64'],
        root/'commands/build_index_v2', timeout_seconds=14400, environment=env, working_directory=str(root),
        inputs=[prep['reference_fasta'],prep['reference_gtf'],args.config])
elif args.stage == 'sample':
    if not args.condition: parser.error('sample needs --condition')
    raw = root/'raw'/args.condition
    wait_for([raw/'validation'])
    reads = root/'fastq'/args.condition
    if not completed(reads): extract_bam(c, args.condition, reads)
    wait_for([root/'commands/build_index_v2'])
    out = root/'quant'/args.condition
    command = root/'commands'/f'quant_{args.condition}'
    if not completed(command):
        run_command([tool,'quant','--index',str(index/'index'),'--chemistry','1{b[14]u[10]}2{r:}',
            '--expected-ori',prep['expected_orientation'],'--reads1',str(reads/'barcode.fastq'),
            '--reads2',str(reads/'cdna.fastq'),'--explicit-pl',str(reads/'barcodes.txt'),
            '--resolution','cr-like','--threads',str(c['resources']['quant_threads_per_library']),
            '--output',str(out)], command, timeout_seconds=43200, environment=env,
            inputs=[reads/'provenance.json',args.config])
    imported = root/'processed'/args.condition
    if not completed(imported): print(import_counts(c, args.condition, out, imported),flush=True)
elif args.stage == 'folds':
    wait_for([root/'processed/ctrl',root/'processed/stim'])
    for donor in c['donors']:
        out = root/'folds'/donor
        if not completed(out): print(prepare_kang_fold(c, donor, out), flush=True)
    save_json(root/'folds/ready.json', dict(status='ALL_KANG_FOLDS_READY', donors=c['donors']))
