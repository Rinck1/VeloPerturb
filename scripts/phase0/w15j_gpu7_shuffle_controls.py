"""Use newly authorized GPU7 for two additional fixed shuffle controls.

Reuse the frozen parent real/S+U/U-program models; train only permutation1/2
controls with seeds0/1/2. The shared launcher file is deliberately unmodified:
it is hash-bound by the running RENGE jobs. New job metadata reflects GPU7's
explicit authorization and GPU6's continuing reservation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from w4_artifacts import Artifacts, sha256
from veloroute.artifacts import save_json
from veloroute.gpu_policy import (ALLOWED_PHYSICAL_GPUS, MAX_VISIBLE_GPUS,
    RESERVED_PHYSICAL_GPUS, enforce_gpu_policy, gpu_uuids)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def launch(configuration, output, job_directory):
    from w15i_followup_queue import worker_environment
    directory = (ROOT/job_directory).resolve()
    if not directory.is_relative_to(ROOT/'outputs'):
        raise ValueError('Job records must remain under project outputs')
    uuid = gpu_uuids()[7]
    memory = subprocess.check_output(['nvidia-smi', '--id='+uuid, '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    if int(memory.strip()) > 256 or any(row.split(',')[0].strip() == uuid for row in processes.splitlines()):
        raise RuntimeError('GPU7 is no longer free; do not collide with another project')
    if Path(output).exists(): raise FileExistsError('New output directory required')
    directory.mkdir(parents=True, exist_ok=False)
    env = worker_environment(Path(output).parent/'runtime_gpu7')
    job = dict(physical_gpu=7, gpu_uuid=uuid,
        command=[sys.executable, str(Path(__file__).resolve()), '--configuration',
                 str(Path(configuration).resolve()), '--output', str(Path(output).resolve())],
        cwd=str(ROOT), allowed_gpus=list(ALLOWED_PHYSICAL_GPUS),
        reserved_gpus=list(RESERVED_PHYSICAL_GPUS), max_visible_gpus=MAX_VISIBLE_GPUS,
        authorization='user explicitly authorized physical GPU7 on 2026-10-21',
        runtime_environment={k: env[k] for k in ('JAX_PLATFORMS', 'LD_LIBRARY_PATH', 'TMPDIR',
            'JAX_COMPILATION_CACHE_DIR', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH')})
    save_json(directory/'job.json', job)
    with (directory/'job.log').open('xb') as stream:
        process = subprocess.Popen([sys.executable, str(HERE/'launch_gpu_job.py'), '--worker', str(directory)],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    save_json(directory/'launcher.json', dict(pid=process.pid, physical_gpu=7, detached=True))
    return dict(status='LAUNCHED', pid=process.pid, physical_gpu=7, job_directory=str(directory))


def program(configuration, output):
    indices = enforce_gpu_policy('cuda')
    if indices != (7,): raise ValueError('This independent lane is explicitly assigned to physical GPU7')
    import numpy as np
    import jax
    from flax import serialization
    import w15h_traxler_program_router as base
    from w7_router_scaling import write_csv

    plan = json.loads(Path(configuration).read_text())
    config = json.loads(Path(plan['base_configuration']).read_text())
    parent = Path(plan['parent'])
    parent_grid = json.loads((parent/'frozen_grid.json').read_text())['predictions']
    shared_arm = 'S_U_shuffled_velocity_program'
    fold = Path(config['fold'])
    inputs = [Path(configuration), Path(plan['base_configuration']), Path(__file__),
        HERE/'w15h_traxler_program_router.py', HERE/'w15g_response_program_probe.py',
        HERE/'w15c_renge_l3.py', HERE/'w15a_cellflow_moe.py', HERE/'w15a_cellflow_moe_official.py',
        HERE/'w15a_cellflow_moe_real_predict.py', HERE/'w15a_cellflow_train.py',
        HERE/'w15a_cellflow_adapter.py', ROOT/'src/veloroute/gpu_policy.py',
        parent/'frozen_grid.json', parent/'training_programs.npz', Path(config['split']),
        Path(config['conditions']), fold/'train_source.npz', fold/'train_target.npz',
        fold/'validation_source.npz', fold/'transform.npz']
    run = Artifacts(output, seed=config['seeds'], inputs=inputs)
    deadline = time.monotonic()+plan['maximum_hours']*3600
    try:
        run.json('study_manifest.json', dict(config=plan, new_fits=6, physical_gpu=7,
            reused_parent_models=9, test_targets_read=False, original_parent_untouched=True))
        completed = 0; permutation_results = []
        for permutation in plan['additional_shuffle_seeds']:
            c = dict(config, shuffle_seed=permutation)
            training, heldout, norm, split, mapping = base.prepare(c)
            saved = base.read_npz(parent/'training_programs.npz')
            for key in norm: np.testing.assert_array_equal(norm[key], saved[key])
            directory = run.directory/f'permutation{permutation}'
            part = Artifacts(directory, seed=c['seeds'], inputs=[Path(configuration), Path(__file__)])
            part.json('study_manifest.json', dict(config=c, shuffle_mapping=mapping, new_fits=3,
                reused_parent_models=9, test_targets_read=False))
            frozen = [dict(row) for row in parent_grid if row['arm'] != shared_arm]
            model = base.create_model(c, training[0]['z'].shape[1], 0, training[0]['embedding'])
            step = base.make_step(model, c); integrate = base.make_candidate_integrator(model)
            for seed in c['seeds']:
                if time.monotonic() > deadline: raise TimeoutError('GPU7 shuffle-control deadline exceeded')
                fit = Artifacts(directory/f'seed{seed}', seed=seed, inputs=[Path(configuration), Path(__file__)])
                run.json('status.json', dict(status='training', completed_models=completed, total_models=6,
                    permutation=permutation, seed=seed, physical_gpu=7))
                params, initial, history = base.train(model, step,
                    [base.apply_arm(g, shared_arm, c) for g in training], c, seed,
                    lambda row: fit.json('status.json', dict(status='training', **row)))
                initial_hash = hashlib.sha256(serialization.to_bytes(initial)).hexdigest()
                reference = json.loads((parent/'fits'/f'seed{seed}'/'S_U_router'/'summary.json').read_text())
                if initial_hash != reference['initial_parameters_sha256']:
                    raise ValueError('Extra shuffle control changed model initialization')
                checkpoint = fit.directory/'model.msgpack'
                checkpoint.write_bytes(serialization.to_bytes(params))
                params = serialization.from_bytes(initial, checkpoint.read_bytes())
                model_hash = sha256(checkpoint)
                fit.json('training_history.json', dict(history=history))
                for source in heldout:
                    p = base.predict(model, integrate, params, base.apply_arm(source, shared_arm, c), c)
                    path = fit.directory/(source['label']+'.npz'); np.savez_compressed(path, **p)
                    frozen.append(dict(KO=source['label'], seed=seed, arm=shared_arm,
                        path=str(path.resolve()), sha256=sha256(path), model=str(checkpoint.resolve()),
                        model_sha256=model_hash, heldout_future_used=False))
                fit.json('summary.json', dict(status='complete', permutation=permutation, seed=seed,
                    parameters=sum(x.size for x in jax.tree_util.tree_leaves(params)), steps=c['steps'],
                    initial_parameters_sha256=initial_hash, physical_gpu=7,
                    backend=[str(device) for device in jax.devices()]))
                fit.json('status.json', dict(status='complete'))
                completed += 1
                print(json.dumps(dict(status='extra_shuffle_complete', permutation=permutation,
                    seed=seed, completed_models=completed)), flush=True)
            part.json('frozen_grid.json', dict(predictions=frozen, test_targets_read=False))
            result = base.evaluate(directory, c, frozen, split, norm)
            result.update(new_fits=3, reused_parent_models=9, shuffle_seed=permutation, physical_gpu=7)
            part.json('summary.json', result); part.json('status.json', dict(status='complete'))
            (directory/'RESULTS.md').write_text('# Additional shuffle control\n\n'+json.dumps(result, indent=2)+'\n')
            permutation_results.append(result)
        # All three fixed permutations contribute equally; no best-null selection.
        original_rows = list(csv.DictReader((parent/'metrics.csv').open()))
        all_shuffle = [dict(permutation=0, **r) for r in original_rows if r['arm'] == shared_arm]
        for permutation in plan['additional_shuffle_seeds']:
            with (run.directory/f'permutation{permutation}'/'metrics.csv').open() as stream:
                all_shuffle.extend(dict(permutation=permutation, **r) for r in csv.DictReader(stream)
                                   if r['arm'] == shared_arm)
        write_csv(run.directory/'all_shuffle_metrics.csv', all_shuffle)
        paired = []
        for r in original_rows:
            if r['arm'] != 'S_U_velocity_program': continue
            controls = [float(s['energy']) for s in all_shuffle if (s['KO'], str(s['seed'])) == (r['KO'], str(r['seed']))]
            if len(controls) != 3: raise ValueError('Three fixed permutation controls required')
            value = float(np.mean(controls))
            paired.append(dict(KO=r['KO'], seed=int(r['seed']), real_energy=float(r['energy']),
                three_shuffle_mean_energy=value, real_gain_vs_three_shuffle=1-float(r['energy'])/value))
        write_csv(run.directory/'three_shuffle_comparisons.csv', paired)
        result = dict(status='GPU7_ADDITIONAL_SHUFFLE_CONTROLS_COMPLETE', new_fits=6,
            physical_gpu=7, reserved_gpus=[6], test_targets_read=False, automatic_retuning=False,
            permutation_seeds=[0, *plan['additional_shuffle_seeds']],
            real_gain_vs_three_shuffle=float(np.mean([r['real_gain_vs_three_shuffle'] for r in paired])),
            results=permutation_results)
        run.json('summary.json', result)
        (run.directory/'RESULTS.md').write_text('# GPU7 fixed shuffle replications\n\n'+json.dumps(result, indent=2)+'\n')
        run.json('status.json', dict(status='complete', completed_models=6, total_models=6))
        return result
    except Exception as exc:
        run.json('status.json', dict(status='failed', error=repr(exc))); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--launch-job')
    args = parser.parse_args()
    result = (launch(args.configuration, args.output, args.launch_job) if args.launch_job else
              program(args.configuration, args.output))
    print(json.dumps(result, indent=2))
