"""Execution order only: unchanged Kang experiments, mainline model first."""
from pathlib import Path


MAINLINE = 'full_GFG_EM_gate'


def worker_stages(worker):
    if worker not in range(4):
        raise ValueError('Kang workers are restricted to GPUs 0-3')
    return [
        ('mainline', 'scripts/phase0/run_kang_full_grid.py', ['--worker', str(worker), '--phase', 'mainline']),
        ('router_controls', 'scripts/phase0/run_kang_experiment_grid.py', ['--worker', str(worker)]),
        ('full_ablations', 'scripts/phase0/run_kang_full_grid.py', ['--worker', str(worker), '--phase', 'followups']),
    ]


def full_variants(extended, phase):
    if phase not in ('all', 'mainline', 'followups'):
        raise ValueError('Unknown full-grid phase')
    variants = extended['full_variants']
    names = [v['name'] for v in variants]
    if len(names) != len(set(names)) or names.count(MAINLINE) != 1:
        raise ValueError('Expected exactly one registered mainline variant and unique arm names')
    if phase == 'mainline':
        return [v for v in variants if v['name'] == MAINLINE]
    if phase == 'followups':
        return [v for v in variants if v['name'] != MAINLINE]
    return variants


def full_dependencies(root, worker, phase):
    worker_stages(worker)  # validate resource identity
    root = Path(root)
    if phase == 'mainline':
        return [root/'folds/ready.json']
    if phase == 'followups':
        return [root/f'mainline_worker{worker}_complete.json', root/f'primary_worker{worker}_complete.json']
    if phase == 'all':
        return [root/f'primary_worker{worker}_complete.json']
    raise ValueError('Unknown full-grid phase')
