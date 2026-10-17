from pathlib import Path
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w6_local_router import energy_from_weights, fit_local
from w7_router_scaling import blocked_energy, build_design, worker, aggregate, sha256


def test_blocked_energy_is_original_weighted_statistic():
    generator = torch.Generator().manual_seed(9)
    x = torch.randn((19, 4), dtype=torch.float64, generator=generator)
    y = torch.randn((11, 4), dtype=torch.float64, generator=generator)
    w = torch.rand(19, dtype=torch.float64, generator=generator)
    for block in (1, 4, 100):
        torch.testing.assert_close(blocked_energy(x, w, y, block), energy_from_weights(x, w, y), atol=1e-12, rtol=1e-12)


def test_scaling_grid_is_paired_balanced_and_one_variable_at_a_time():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/w7_scifate2_router_scaling_v1.json').read_text())
    base = json.loads((root/'configs/w6_local_router_scifate2_v1.json').read_text())
    jobs, cases = build_design(config, base)
    assert len(jobs) == 51 and len(cases) == 36
    assert len({j['setting'] for j in jobs}) == 9
    for job in jobs:
        assert job['hidden'] == 128 or job['experts'] == 4
        assert job['config']['expert_rank'] == 32 and job['config']['router_hidden'] == 64
        assert job['worker'] in range(4)
        if job['experts'] > 1:
            other = [j for j in jobs if (j['setting'],j['seed']) == (job['setting'],job['seed'])]
            assert {j['arm'] for j in other} == {'static_router', 'velocity_router'}
    loads = [sum(j['cost'] for j in jobs if j['worker'] == i) for i in range(4)]
    assert max(loads)/min(loads) < 1.2


def test_checkpoints_do_not_change_training_trajectory():
    rng = np.random.default_rng(9)
    groups = [dict(z=rng.normal(size=(12,3)), velocity=rng.normal(size=(12,3)),
                   target=rng.normal(size=(14,3)), global_rate=np.zeros(3))]
    norm = dict(z_mean=np.zeros(3), z_scale=np.ones(3), v_mean=np.zeros(3), v_scale=np.ones(3))
    config = dict(experts=2, hidden=8, expert_rank=4, router_hidden=8, calibration_hours=24,
        maximum_state_correction=2., learning_rate=.0003, weight_decay=.0001,
        steps=4, source_batch=4, target_batch=4, residual_regularization=.001, routing_regularization=.002,
        checkpoint_steps=[1, 4])
    snapshots = []
    def checkpoint(step, model):
        snapshots.append(step)
        with torch.no_grad(): model(torch.tensor(groups[0]['z'], dtype=torch.float32),
            torch.tensor(groups[0]['velocity'], dtype=torch.float32), torch.zeros(3), 5.)
    a, _ = fit_local(groups, norm, config, arm='velocity_router', seed=0, device='cpu', checkpoint=checkpoint)
    b, _ = fit_local(groups, norm, config, arm='velocity_router', seed=0, device='cpu')
    assert snapshots == [1, 4]
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)


def test_tiny_scaling_worker_and_report(tmp_path):
    rng = np.random.default_rng(2)
    root = tmp_path/'study'; pack = tmp_path/'pack'
    root.mkdir(); pack.mkdir()
    z, v, y = [rng.normal(size=shape).astype('float32') for shape in ((12,3),(12,3),(14,3))]
    norm = dict(z_mean=np.zeros(3), z_scale=np.ones(3), v_mean=np.zeros(3), v_scale=np.ones(3))
    np.savez(pack/'normalization.npz', **norm)
    train = pack/'train.npz'
    np.savez(train, z=z, velocity=v, target=y, global_rates=np.zeros((1,3)))
    anchor = z.astype(float).mean(0)+.1
    source = pack/'source.npz'
    np.savez(source, z=z, velocity=v, duration=5., anchors=anchor[None], source_mean=z.astype(float).mean(0))
    pm = dict(training_groups=[dict(path=str(train), sha256=sha256(train))],
        prediction_groups=[dict(path=str(source), sha256=sha256(source), source_hour=5, target_hour=10)])
    (pack/'manifest.json').write_text(json.dumps(pm))
    base = dict(experts=2, hidden=8, expert_rank=4, router_hidden=8, calibration_hours=24,
        maximum_state_correction=2., learning_rate=.0003, weight_decay=.0001,
        source_batch=4, target_batch=4, residual_regularization=.001, routing_regularization=.002)
    config = dict(expert_counts=[1,2], backbone_widths=[8], baseline_experts=2, baseline_width=8,
        steps=2, checkpoint_steps=[1,2], long_steps=3, long_checkpoint_steps=[1,2,3],
        arms=['static_router','velocity_router'], seeds=[0], workers=1, energy_block_size=4,
        saturation_consecutive_doublings=2, saturation_relative_improvement=.002)
    jobs, cases = build_design(config, base)
    cache = root/'evaluation_5_to_10.npz'
    np.savez(cache, target=y[:7].astype(float), target_mean=y.astype(float).mean(0),
        target_cov=np.cov(y.astype(float), rowvar=False, bias=True), anchors=anchor[None])
    manifest = dict(config=config, pack=str(pack), pack_manifest_sha256=sha256(pack/'manifest.json'),
        jobs=jobs, cases=cases, evaluation_caches=[dict(path=str(cache), sha256=sha256(cache), source_hour=5, target_hour=10)])
    (root/'study_manifest.json').write_text(json.dumps(manifest))
    (root/'source_samples.json').write_text(json.dumps({'5_to_10':list(range(6))}))
    worker(root, 0, 'cpu'); aggregate(root, False)
    result = json.loads((root/'report'/'summary.json').read_text())
    assert result['training_jobs'] == 3 and result['cases'] == 8
    assert result['mean_anchor_max_error'] < 1e-6
    assert len(result['velocity_comparisons']) == 3
