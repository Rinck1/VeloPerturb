"""Fixed RENGE L2/L3 transfer: two bounded GPU lanes, then frozen scoring.

No target scoring occurs until the entire lane's predictions have been verified.
Completed fits are reusable only with unchanged code, configuration, identities,
source IDs, checkpoint hashes, and prediction hashes. Incomplete attempts are
never overwritten or silently restarted.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from w4_artifacts import Artifacts, sha256


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNTIME = Path('/data/yuchang/veloroute_w15a/runtime')
LANES = {
    'L2': dict(gpu=2, module='w15c_renge_l2', real='cf_kin', variants=(
        'cf_static', 'cf_Ssum', 'cf_directU', 'cf_kin',
        'cf_kin_shuf_perm0', 'cf_kin_shuf_perm1', 'cf_kin_shuf_perm2')),
    'L3': dict(gpu=3, module='w15c_renge_l3', real='moe_kin', variants=(
        'moe_static', 'moe_kin', 'moe_directU',
        'moe_kin_shuf_perm0', 'moe_kin_shuf_perm1', 'moe_kin_shuf_perm2')),
}
SEEDS = (0, 1, 2)


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def registered_jobs(folds, lane):
    if lane not in LANES:
        raise ValueError('Only registered L2/L3 lanes are supported')
    names = [fold['name'] if isinstance(fold, dict) else str(fold) for fold in folds]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate fold identities')
    return [dict(lane=lane, fold=name, variant=variant, seed=seed)
            for name in names for variant in LANES[lane]['variants'] for seed in SEEDS]


def verify_fold_design(folds):
    if len(folds) != 32:
        raise ValueError('RENGE transfer requires exactly the original 32 folds')
    by_tf = {}
    for row in folds:
        manifest = read_json(row['path'])
        if sha256(row['path']) != row['sha256'] or manifest['name'] != row['name']:
            raise ValueError('Fold manifest changed or identity mismatch')
        by_tf.setdefault(manifest['TF'], []).append((manifest['source_day'], manifest['target_day']))
    if len(by_tf) != 16 or any(sorted(pairs) != [(2, 3), (3, 4)] for pairs in by_tf.values()):
        raise ValueError('Expected sixteen TFs with day2→3 and day3→4, without duplicates')


def code_inputs(configuration, pack):
    """Bind local adapters and all inherited mathematical implementation files."""
    names = ['w15c_renge_program.py', 'w15c_renge_data.py', 'w15c_renge_l2.py', 'w15c_renge_l3.py',
             'w15a_cellflow_adapter.py', 'w15a_cellflow_train.py', 'w15a_cellflow_moe.py',
             'w15a_cellflow_moe_official.py', 'w15a_cellflow_moe_real_train.py',
             'w15a_cellflow_moe_real_predict.py', 'w14_loro.py', 'w14_data.py',
             'w4_streams.py', 'w4_shorttime.py', 'w4_artifacts.py', 'launch_gpu_job.py']
    return [Path(configuration).resolve(), Path(pack).resolve()/'study_manifest.json'] + [HERE/name for name in names]


def freeze_lock(configuration, pack, output):
    pack = Path(pack).resolve()
    manifest = read_json(pack/'study_manifest.json')
    verify_fold_design(manifest['folds'])
    config = read_json(configuration)
    if (config.get('steps'), config.get('batch_size'), config.get('learning_rate')) != (5000, 128, .0003):
        raise ValueError('Registered transfer inherits 5000 steps, batch128, lr3e-4 without HPO')
    inputs = code_inputs(configuration, pack)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise ValueError(f'Transfer implementation must be complete before launch: {missing}')
    lock = dict(configuration=str(Path(configuration).resolve()), pack=str(pack),
        config_sha256=sha256(configuration), pack_manifest_sha256=sha256(pack/'study_manifest.json'),
        frozen_sha256={str(path):sha256(path) for path in inputs}, folds=manifest['folds'],
        jobs={lane:registered_jobs(manifest['folds'], lane) for lane in LANES},
        models={'L2':672, 'L3':576}, total_models=1248,
        physical_gpus={'L2':2, 'L3':3}, seeds=list(SEEDS),
        no_HPO=True, mean_anchor=False, target_scores_opened=False)
    lock['lock_sha256'] = digest_object(lock)
    atomic_json(output, lock)
    return lock


def verify_lock(path):
    lock = read_json(path)
    expected = lock['lock_sha256']
    if digest_object({key:value for key,value in lock.items() if key != 'lock_sha256'}) != expected:
        raise ValueError('Transfer lock changed')
    for source, expected in lock['frozen_sha256'].items():
        if sha256(source) != expected:
            raise ValueError(f'Frozen transfer code/configuration changed: {source}')
    verify_fold_design(lock['folds'])
    for lane in LANES:
        if lock['jobs'][lane] != registered_jobs(lock['folds'], lane):
            raise ValueError('Registered lane identities changed')
    return lock


def job_directory(lane_root, job):
    return Path(lane_root)/'training'/job['fold']/job['variant']/f"seed{job['seed']}"


def release_fit_caches():
    """Independent model closures must not retain hundreds of compiled graphs."""
    jax = sys.modules.get('jax')
    if jax is not None:
        jax.clear_caches()
    gc.collect()


def _source_ids(pack, fold):
    from w15c_renge_data import load_fold
    _, _, source, _ = load_fold(pack, fold)
    ids = np.asarray(source['cell_ids'])
    indices = np.asarray(source['score_indices'], dtype=int)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(ids)):
        raise ValueError('Source scoring indices are invalid')
    return ids[indices]


def verify_fit(directory, job, *, source_ids, lock_sha256=None):
    directory = Path(directory)
    required = [directory/name for name in ('status.json', 'model.msgpack', 'predictions.npz', 'prediction_manifest.json')]
    if any(not path.is_file() for path in required):
        raise ValueError(f'Incomplete fit is preserved; explicit recovery is required: {directory}')
    status = read_json(directory/'status.json')
    manifest = read_json(directory/'prediction_manifest.json')
    if status.get('status') != 'complete':
        raise ValueError(f'Incomplete fit is preserved; explicit recovery is required: {directory}')
    if any(manifest.get(key) != job[key] for key in ('fold', 'variant', 'seed')):
        raise ValueError('Fit prediction identity mismatch')
    if manifest.get('heldout_future_used') is not False:
        raise ValueError('Held-out target input is forbidden')
    if Path(manifest['path']).resolve() != (directory/'predictions.npz').resolve():
        raise ValueError('Prediction path escapes its registered fit')
    if sha256(directory/'predictions.npz') != manifest['sha256'] or sha256(directory/'model.msgpack') != manifest['model_sha256']:
        raise ValueError('Prediction or model artifact changed')
    if lock_sha256 is not None:
        receipt = read_json(directory/'queue_receipt.json')
        if receipt != dict(job=job, lock_sha256=lock_sha256):
            raise ValueError('Completed fit belongs to another frozen transfer')
    with np.load(directory/'predictions.npz', allow_pickle=False) as data:
        x, q, ids = [np.asarray(data[key]) for key in ('candidates', 'q', 'source_ids')]
    expected_k = 1 if job['lane'] == 'L2' else 8
    if x.shape != (len(source_ids), expected_k, 50) or q.shape != x.shape[:2] or not len(q):
        raise ValueError('Prediction candidate/weight shapes differ from the registered model')
    if not np.array_equal(ids, source_ids):
        raise ValueError('Frozen prediction source IDs differ from the original L1 cohort')
    if not np.isfinite(x).all() or not np.isfinite(q).all() or np.any(q < 0) or not np.allclose(q.sum(1), 1, atol=1e-6):
        raise ValueError('Predictions contain invalid candidate geometry or mixture weights')
    return dict(**job, path=str((directory/'predictions.npz').resolve()), sha256=manifest['sha256'],
                model_sha256=manifest['model_sha256'], manifest_sha256=sha256(directory/'prediction_manifest.json'),
                directory=str(directory.resolve()))


def freeze_lane(lock_path, lane_root, lane):
    """Only source-side features are read until ALL expected models are validated."""
    lock = verify_lock(lock_path)
    ids = {row['name']:_source_ids(lock['pack'], row['name']) for row in lock['folds']}
    records = [verify_fit(job_directory(lane_root, job), job, source_ids=ids[job['fold']],
                          lock_sha256=lock['lock_sha256']) for job in lock['jobs'][lane]]
    result = dict(lane=lane, predictions=records, total=len(records), expected=lock['models'][lane],
                  lock_sha256=lock['lock_sha256'], target_scores_opened=False)
    atomic_json(Path(lane_root)/'whole_grid_frozen.json', result)
    return lock, records


def load_target(fold_record):
    """Exactly the original RENGE L1 QC, transform, and RNG0 <=512 selection."""
    from w4_shorttime import read_metadata, read_count_rows, qc_mask
    from w4_streams import SourceTransform
    fold = read_json(fold_record['path'])
    original = fold.get('original_manifest', fold)
    if isinstance(original, str):
        original = read_json(original)
    if sha256(original['transform_path']) != original['transform_sha256']:
        raise ValueError('Original L1 transform changed')
    obs, _ = read_metadata(original['target_path'])
    selected = np.flatnonzero(qc_mask(obs) & (obs['condition'].astype(str).to_numpy() == original['TF']))
    transform = SourceTransform.load(original['transform_path'])
    target = transform.state(read_count_rows(original['target_path'], 'spliced', selected))
    chosen = np.random.default_rng(0).choice(len(target), min(len(target), 512), replace=False)
    return target[chosen], dict(TF=original['TF'], source_day=original['source_day'], target_day=original['target_day'])


def _csv(path, rows):
    if not rows:
        raise ValueError('Refusing empty evaluation table')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(lock_path, lane_root, lane, output, *, device='cuda'):
    lock, records = freeze_lane(lock_path, lane_root, lane)
    if device != 'cpu':
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy(device)
    from w14_loro import energy
    run = Artifacts(output, seed=list(SEEDS), inputs=[Path(lock_path), Path(lane_root)/'whole_grid_frozen.json'])
    run.json('frozen_grid.json', dict(predictions=records, target_scores_opened_before_freeze=False))
    rows = []
    for fold in lock['folds']:
        target, metadata = load_target(fold)
        for variant in LANES[lane]['variants']:
            selected = [row for row in records if row['fold'] == fold['name'] and row['variant'] == variant]
            selected.sort(key=lambda row:row['seed'])
            if [row['seed'] for row in selected] != list(SEEDS):
                raise ValueError('Registered seed mixture is incomplete')
            packs = []
            for record in selected:
                with np.load(record['path'], allow_pickle=False) as data:
                    packs.append({key:np.asarray(data[key]) for key in ('candidates', 'q', 'source_ids')})
            for p in packs[1:]:
                np.testing.assert_array_equal(packs[0]['source_ids'], p['source_ids'])
            x = np.concatenate([p['candidates'].reshape(-1,50) for p in packs])
            w = np.concatenate([p['q'].ravel()/len(p['q'])/len(packs) for p in packs])
            value = energy(x, w, target, device)
            rows.append(dict(lane=lane, fold=fold['name'], **metadata, arm=variant, energy=value,
                             source_cells=len(packs[0]['q']), target_cells=len(target), seeds=len(packs)))
        run.json('status.json', dict(status='evaluating', completed_folds=len(rows)//len(LANES[lane]['variants']),
                                    total_folds=32))
    gains = []
    real = LANES[lane]['real']
    for tf in sorted({row['TF'] for row in rows}):
        for control in LANES[lane]['variants']:
            if control == real:
                continue
            values = []
            for day in (2,3):
                by = {row['arm']:row['energy'] for row in rows if row['TF'] == tf and row['source_day'] == day}
                values.append(1-by[real]/by[control])
            gains.append(dict(TF=tf, control=control, mean_gain=float(np.mean(values))))
    result = dict(status='RENGE_LANE_TRANSFER_COMPLETE', lane=lane, models=len(records), folds=len(lock['folds']),
        real_arm=real, gain_by_control={control:float(np.mean([row['mean_gain'] for row in gains if row['control']==control]))
            for control in LANES[lane]['variants'] if control != real},
        averaging='equal_TF_after_equal_two_intervals_and_equal_seed_predictive_mixture',
        whole_lane_predictions_frozen_before_target_scoring=True, fixed_recipe_no_HPO=True,
        mean_anchor=False, scope='RENGE_early_response_day2to3_day3to4_leave_one_TF_out')
    _csv(run.directory/'metrics.csv', rows)
    _csv(run.directory/'gains.csv', gains)
    run.json('summary.json', result)
    (run.directory/'RESULTS.md').write_text('# RENGE ' + lane + ' transfer\n\n' + json.dumps(result, indent=2) + '\n')
    run.json('status.json', dict(status='complete'))
    return result


def worker(lock_path, lane_root, lane, *, synthetic_gate=None, deadline_hours=48., require_gpu=True):
    lock = verify_lock(lock_path)
    if not 0 < deadline_hours <= 48:
        raise ValueError('Worker deadline must be within 48 hours')
    if require_gpu:
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy('cuda')
    adapter = importlib.import_module(LANES[lane]['module'])
    if lane == 'L3' and synthetic_gate is None:
        raise ValueError('The completed L3 synthetic engineering gate is required')
    lane_root = Path(lane_root)
    lane_root.mkdir(parents=True, exist_ok=True)
    bound = lane_root/'lane_lock.json'
    receipt = dict(lane=lane, lock_sha256=lock['lock_sha256'])
    if bound.exists() and read_json(bound) != receipt:
        raise ValueError('Existing lane belongs to another frozen configuration')
    atomic_json(bound, receipt)
    deadline = time.monotonic() + deadline_hours*3600
    ids = {}
    try:
        for index, job in enumerate(lock['jobs'][lane]):
            if time.monotonic() > deadline:
                raise TimeoutError('Bounded RENGE training deadline exceeded')
            verify_lock(lock_path)
            directory = job_directory(lane_root, job)
            if job['fold'] not in ids:
                ids[job['fold']] = _source_ids(lock['pack'], job['fold'])
            atomic_json(lane_root/'status.json', dict(status='training', completed=index, total=lock['models'][lane], job=job))
            if directory.exists():
                verify_fit(directory, job, source_ids=ids[job['fold']], lock_sha256=lock['lock_sha256'])
                continue
            kwargs = dict(variant=job['variant'], seed=job['seed'])
            if lane == 'L3':
                kwargs['synthetic_gate'] = synthetic_gate
            adapter.train_predict(lock['pack'], job['fold'], lock['configuration'], directory, **kwargs)
            # The queue binds a successfully fitted adapter to its immutable launch.
            verify_fit(directory, job, source_ids=ids[job['fold']])
            atomic_json(directory/'queue_receipt.json', dict(job=job, lock_sha256=lock['lock_sha256']))
            release_fit_caches()
        freeze_lane(lock_path, lane_root, lane)
        atomic_json(lane_root/'status.json', dict(status='all_predictions_frozen', models=lock['models'][lane]))
        result = evaluate(lock_path, lane_root, lane, lane_root/'evaluation', device='cuda' if require_gpu else 'cpu')
        atomic_json(lane_root/'summary.json', result)
        atomic_json(lane_root/'status.json', dict(status='complete', models=lock['models'][lane]))
        return result
    except Exception as error:
        atomic_json(lane_root/'status.json', dict(status='failed', error=repr(error), automatic_retry=False))
        raise


def launch_lane(lane, arguments, job_directory):
    if lane not in LANES:
        raise ValueError('Only GPU2/L2 and GPU3/L3 may be launched')
    env = os.environ.copy()
    env.update(JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        TMPDIR=str(RUNTIME/'tmp'), JAX_COMPILATION_CACHE_DIR=str(RUNTIME/('jax_renge_'+lane+'_cache')),
        XDG_CACHE_HOME=str(RUNTIME/('cache_renge_'+lane)),
        CUDA_CACHE_PATH=str(RUNTIME/('cuda_renge_'+lane+'_cache')),
        LD_LIBRARY_PATH=str(RUNTIME/'cuda_deps/nvidia/cudnn/lib'), PYTHONPATH=str(ROOT/'src'))
    command = [sys.executable, str(HERE/'launch_gpu_job.py'), '--gpu', str(LANES[lane]['gpu']),
               '--script', str(Path(__file__)), '--output', str(job_directory), '--', *arguments]
    return json.loads(subprocess.check_output(command, cwd=ROOT, env=env, text=True))


def _stop_owned_launcher(job_directory):
    """Only terminate the exact detached process group launched by this queue."""
    launcher = read_json(Path(job_directory)/'launcher.json')
    pid = int(launcher['pid'])
    command = Path(f'/proc/{pid}/cmdline')
    if not command.exists():
        return
    args = command.read_bytes().replace(b'\0',b' ').decode()
    if 'launch_gpu_job.py' not in args or str(Path(job_directory).resolve()) not in args:
        raise RuntimeError('Refusing to terminate an unverified queue process')
    os.killpg(pid, signal.SIGTERM)


def program(pack, configuration, output, jobs_output, *, synthetic_gate, approved=False,
            deadline_hours=48., poll_seconds=30.):
    if not approved:
        raise ValueError('Explicit approval is required for 1248 fixed training fits')
    if not 0 < deadline_hours <= 48 or not 0 < poll_seconds <= 60:
        raise ValueError('Controller must be bounded to 48h with polls no longer than 60s')
    output, jobs_output, pack = map(Path, (output, jobs_output, pack))
    run = Artifacts(output, seed=list(SEEDS), inputs=[Path(configuration), Path(__file__)])
    deadline = time.monotonic() + deadline_hours*3600
    active = []
    try:
        while not (pack/'status.json').is_file() or read_json(pack/'status.json').get('status') != 'ready':
            if (pack/'status.json').is_file() and read_json(pack/'status.json').get('status') == 'failed':
                raise RuntimeError('Source-only pack preparation failed; no automatic retry')
            if time.monotonic() > deadline:
                raise TimeoutError('RENGE source pack preparation exceeded controller deadline')
            run.json('status.json', dict(status='waiting_for_source_pack'))
            time.sleep(poll_seconds)
        lock_path = run.directory/'transfer_lock.json'
        lock = freeze_lock(configuration, pack, lock_path)
        run.json('study_manifest.json', dict(total_models=1248, models=lock['models'], physical_gpus=lock['physical_gpus'],
            lock_sha256=lock['lock_sha256'], pack=str(pack.resolve()), no_HPO=True, target_scores_opened=False))
        for lane in LANES:
            job_directory = jobs_output/(lane.lower()+'_gpu'+str(LANES[lane]['gpu']))
            arguments = ['worker', '--lock', str(lock_path.resolve()), '--lane', lane,
                         '--output', str((output/lane).resolve()), '--synthetic-gate', str(synthetic_gate),
                         '--deadline-hours', str(max((deadline-time.monotonic())/3600, .001))]
            launched = launch_lane(lane, arguments, job_directory)
            active.append(dict(lane=lane, directory=str(job_directory.resolve()), launched=launched))
        run.json('launched_lanes.json', dict(lanes=active))
        while True:
            states = {job['lane']:read_json(Path(job['directory'])/'status.json')
                      if (Path(job['directory'])/'status.json').exists() else {'status':'starting'} for job in active}
            run.json('status.json', dict(status='waiting_for_lanes', lanes=states))
            if all(state['status'] in ('complete','failed') for state in states.values()):
                break
            if time.monotonic() > deadline:
                for job in active:
                    if states[job['lane']]['status'] not in ('complete','failed'):
                        _stop_owned_launcher(job['directory'])
                raise TimeoutError('RENGE controller 48h deadline exceeded; only owned GPU jobs terminated')
            time.sleep(poll_seconds)
        failed = [lane for lane,state in states.items() if state['status']=='failed']
        if failed:
            raise RuntimeError(f'Completed other lanes; failed lanes are preserved without retry: {failed}')
        result = dict(status='RENGE_L2_L3_TRANSFER_COMPLETE', total_models=1248,
            evaluations={lane:str((output/lane/'evaluation'/'summary.json').resolve()) for lane in LANES},
            physical_gpus={'L2':2,'L3':3}, automatic_retuning=False)
        run.json('summary.json', result)
        run.json('status.json', dict(status='complete'))
        return result
    except Exception as error:
        run.json('status.json', dict(status='failed', error=repr(error), automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('program','worker','evaluate'))
    for name in ('pack','configuration','jobs-output','synthetic-gate','lock'):
        parser.add_argument('--'+name)
    parser.add_argument('--lane', choices=tuple(LANES))
    parser.add_argument('--output', required=True)
    parser.add_argument('--deadline-hours', type=float, default=48.)
    parser.add_argument('--approved', action='store_true')
    args = parser.parse_args()
    if args.action == 'program':
        result = program(args.pack,args.configuration,args.output,args.jobs_output,
            synthetic_gate=args.synthetic_gate,approved=args.approved,deadline_hours=args.deadline_hours)
    elif args.action == 'worker':
        result = worker(args.lock,args.output,args.lane,synthetic_gate=args.synthetic_gate,deadline_hours=args.deadline_hours)
    else:
        result = evaluate(args.lock,Path(args.output).parent,args.lane,args.output)
    print(json.dumps(result, indent=2))
