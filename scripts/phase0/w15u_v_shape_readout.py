"""Post-hoc shape diagnostics on every frozen U/V prediction; no fitting."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts, sha256

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path('/data/yuchang/veloroute_w15t_conditioned_router_20261022/four_arm_v1')
U_ROOT = Path('/data/yuchang/veloroute_w15u_velocity_contrast_20261022/fixed_contrast_v1')
V_ROOT = Path('/data/yuchang/veloroute_w15v_target_estimator_20261022/fixed_estimator_v1')


def energy_pair(x, w, y):
    from scipy.spatial.distance import cdist
    x, w, y = np.asarray(x, float), np.asarray(w, float), np.asarray(y, float)
    w = w/w.sum()
    self_term = w@cdist(x, x)@w+cdist(y, y).mean()
    raw = (2*w@cdist(x, y).mean(1)-self_term)/np.sqrt(x.shape[1])
    centered = (2*w@cdist(x-w@x, y-y.mean(0)).mean(1)-self_term)/np.sqrt(x.shape[1])
    if min(raw, centered) < -1e-7: raise ArithmeticError('Negative Energy beyond rounding tolerance')
    return max(0., float(raw)), max(0., float(centered))


def mixture_variance(candidates, q):
    x, q = np.asarray(candidates, float), np.asarray(q, float)
    mean_by_source = (q[:, :, None]*x).sum(1)
    within = float(np.mean(np.sum(q*np.sum((x-mean_by_source[:, None])**2, axis=-1), axis=-1)))
    between = float(np.mean(np.sum((mean_by_source-mean_by_source.mean(0))**2, axis=-1)))
    return dict(within_source_variance=within, between_source_variance=between,
                within_source_fraction=within/max(within+between, 1e-12))


def records():
    result, expected_energy = {}, {}
    for prefix, root in [('contrast', U_ROOT), ('estimated', V_ROOT)]:
        if json.loads((root/'status.json').read_text())['status'] != 'complete':
            raise ValueError('Both complete fixed grids required')
        archived = {(int(r['seed']), r['KO'], r['arm']): float(r['energy'])
                    for r in csv.DictReader((root/'metrics.csv').open())}
        for r in json.loads((root/'whole_grid_frozen.json').read_text())['predictions']:
            name = r['arm'] if r['arm'] in ('no_graph', 'positive_velocity_graph') else prefix+'_'+r['arm']
            key = (r['seed'], r['KO'], name)
            value = dict(r, name=name)
            if key in result and result[key]['sha256'] != value['sha256']:
                raise ValueError('Shared reference mismatch')
            if sha256(r['path']) != r['sha256']: raise ValueError('Frozen prediction changed')
            result[key] = value; expected_energy[key] = archived[(r['seed'], r['KO'], r['arm'])]
    return list(result.values()), expected_energy


def program(output):
    from veloroute.latent import load_pack
    from w7_router_scaling import write_csv
    from threadpoolctl import threadpool_limits
    frozen, archived = records()
    c = json.loads((ROOT/'configs/w15h_traxler_program_router_20261021_v1.json').read_text())
    fold = Path(c['fold'])
    inputs = [Path(__file__), ROOT/'PLANS/w15u_v_shape_readout_20261022.md',
              ROOT/'configs/w15h_traxler_program_router_20261021_v1.json', fold/'validation_target.npz',
              PARENT/'training_programs.npz']
    run = Artifacts(output, seed=c['seeds'], inputs=inputs)
    run.json('frozen_inputs.json', dict(predictions=frozen, new_training=False, test_targets_read=False,
        manifests={str(r/'whole_grid_frozen.json'): sha256(r/'whole_grid_frozen.json') for r in (U_ROOT, V_ROOT)}))
    targets, _ = load_pack(fold/'validation_target.npz', expected_side='target')
    with np.load(PARENT/'training_programs.npz', allow_pickle=False) as f: axes = f['axes']
    rows, max_difference = [], 0.
    with threadpool_limits(limits=1):
        for ko in sorted(set(targets['conditions'])):
            y = targets['z'][targets['conditions'] == ko]
            y = y[np.random.default_rng(c['evaluation_seed']).choice(len(y), min(len(y), c['evaluation_cells']), replace=False)]
            for r in [r for r in frozen if r['KO'] == ko]:
                with np.load(r['path'], allow_pickle=False) as p: x, q = p['candidates'], p['q']
                weights = q.ravel()/len(q)
                raw, shape = energy_pair(x.reshape(-1, x.shape[-1]), weights, y)
                raw5, shape5 = energy_pair((x@axes.T).reshape(-1, len(axes)), weights, y@axes.T)
                difference = abs(raw-archived[(r['seed'], ko, r['name'])])
                if difference > 1e-7: raise ValueError('Original Energy was not reproduced')
                max_difference = max(max_difference, difference)
                rows.append(dict(KO=ko, seed=r['seed'], arm=r['name'], energy_full=raw, energy_program=raw5,
                                 shape_full=shape, shape_program=shape5, **mixture_variance(x, q)))
    comparisons = []
    by = {(r['seed'], r['KO'], r['arm']): r for r in rows}
    for real, controls in [('contrast_signed_velocity', ('no_graph', 'contrast_shuffled_velocity', 'contrast_signed_U', 'contrast_unsigned_velocity')),
                           ('estimated_estimated_velocity', ('no_graph', 'estimated_shuffled_estimated_velocity', 'estimated_estimated_U'))]:
        for control in controls:
            for metric in ('energy_full', 'energy_program', 'shape_full', 'shape_program'):
                values = [(r, 1-r[metric]/by[(r['seed'], r['KO'], control)][metric]) for r in rows if r['arm'] == real]
                comparisons.append(dict(real=real, control=control, metric=metric,
                    gain=float(np.mean([v for r, v in values])),
                    by_seed={str(s):float(np.mean([v for r, v in values if r['seed'] == s])) for s in c['seeds']}))
    means = {a:{k:float(np.mean([r[k] for r in rows if r['arm'] == a]))
        for k in ('energy_full', 'energy_program', 'shape_full', 'shape_program', 'within_source_fraction')}
        for a in sorted({r['arm'] for r in rows})}
    result = dict(status='FROZEN_U_V_SHAPE_READOUT_COMPLETE', predictions=len(frozen), independent_models=len(frozen)//7,
        comparisons=comparisons, means=means, original_energy_max_abs_difference=max_difference,
        primary_rules_changed=False, posthoc_diagnostic=True, centering_uses_target_mean_for_evaluation_only=True,
        new_training=False, test_targets_read=False)
    write_csv(run.directory/'metrics.csv', rows); write_csv(run.directory/'comparisons.csv', comparisons)
    run.json('summary.json', result)
    (run.directory/'RESULTS.md').write_text('# Frozen U/V shape diagnostics\n\n'+json.dumps(result, indent=2)+'\n')
    run.json('status.json', dict(status='complete'))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); args = p.parse_args()
    print(json.dumps(program(args.output), indent=2), flush=True)
