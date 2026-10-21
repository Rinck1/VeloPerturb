"""Isolated post-completion reporting; no training or premature score access."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_artifacts import sha256
import w15a_cellflow_common_evaluation as common
import w15a_l2_reporter as reporter
from test_w15a_common_evaluation import reference,forecasts


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value))


def rows():
    return [dict(arm=arm,key=key,source_hours=start,target_hours=stop,kernel=clock,seeds=3,
                 energy=float(1+index),sampled_prediction_mean_mse=.01,mode_proportion_TV=.1)
        for arm in reporter.ARMS for index,(key,(start,stop,clock)) in enumerate(common.KEYS.items())]


@pytest.fixture
def complete_grid(reference,tmp_path):
    root=tmp_path/'grid';root.mkdir();pack=tmp_path/'pack';pack.mkdir()
    save(pack/'manifest.json',{})
    np.savez_compressed(pack/'token_normalizer.npz',scale=np.ones(3))
    np.save(pack/'training_target_mode_centers.npy',np.array([[0.,0.,0.],[1.,1.,1.]]))
    completed=[];paths=[]
    offsets=dict(cf_static=.30,cf_kin=.15,cf_Ssum=.22,cf_directU=.12,cf_kin_cell=.16,cf_kin_cell_shuf=.20)
    for arm in reporter.ARMS:
        for seed in reporter.SEEDS:
            train=root/arm/f'seed{seed}'/'train';train.mkdir(parents=True)
            (train/'model.msgpack').write_bytes(f'{arm}_{seed}'.encode())
            model_sha=sha256(train/'model.msgpack')
            save(train/'summary.json',dict(arm=arm,seed=seed,model_sha256=model_sha))
            predicted=root/arm/f'seed{seed}'/'common_predict';values=forecasts(reference)
            for item in values.values():item['x']=item['x']+offsets[arm]+seed*.001
            common.save_common_predictions(predicted,values,reference_root=reference,arm=arm,seed=seed,
                extra=dict(model_sha256=model_sha))
            completed.append(dict(arm=arm,seed=seed,training=str(train),prediction=str(predicted),model_sha256=model_sha))
            paths.append(predicted)
    save(root/'study_manifest.json',dict(models=18,prediction_windows=216,arms=list(reporter.ARMS),
        seeds=list(reporter.SEEDS),reference=str(reference),pack=str(pack),freeze_commit='toy_freeze',data_hashes={}))
    save(root/'completed_models.json',dict(completed=completed))
    _,frozen=common.freeze_common_grid(paths,reference_root=reference,expected_arms=reporter.ARMS)
    save(root/'whole_grid_frozen.json',dict(model_count=18,window_count=216,models=frozen))
    common.evaluate_common(paths,root/'evaluation',reference_root=reference,expected_arms=reporter.ARMS,
        mode_centers=pack/'training_target_mode_centers.npy')
    save(root/'summary.json',dict(status='REGISTERED_L2_DEVELOPMENT_GRID_COMPLETE'))
    save(root/'status.json',dict(status='complete'))
    return root


def test_full_window_horizon_clock_tables_use_ratio_of_mean_errors():
    metrics=rows()
    for r in metrics:
        if r['arm']=='cf_kin':r['energy']*=.8
    windows,horizons,sensitivity=reporter.comparison_tables(metrics)
    assert (len(windows),len(horizons),len(sensitivity))==(48,24,12)
    for r in horizons:
        assert r['windows']=={5:3,10:2,15:1}[r['horizon_hours']]
        assert not r['biological_replicates_inferred']
        if r['comparison']=='group_kin_vs_static':assert r['relative_energy_gain']==pytest.approx(.2)
    # Average per-window percentages would be .30; the ratio of mean errors is .9/21.
    for r in metrics:
        if r['key']=='05_10_decay' and r['arm']=='cf_kin':r['energy']=.1
        if r['key']=='05_10_decay' and r['arm']=='cf_static':r['energy']=1.
        if r['key'] in ('10_15_decay','15_20_decay'):
            r['energy']=10.
    _,horizons,_=reporter.comparison_tables(metrics)
    row=next(r for r in horizons if r['comparison']=='group_kin_vs_static' and r['horizon_hours']==5 and r['kernel']=='decay')
    assert row['relative_energy_gain']==pytest.approx(.9/21.)
    assert row['self_reference_windows']==1
    assert sum(r['population_reference_self'] for r in windows)==24


def test_incomplete_or_duplicate_metric_grid_is_rejected():
    metrics=rows()
    with pytest.raises(ValueError,match='complete'):reporter.comparison_tables(metrics[:-1])
    with pytest.raises(ValueError,match='Duplicate'):reporter.comparison_tables(metrics+[metrics[0]])


def test_partial_grid_never_reads_targets_or_scores(tmp_path,monkeypatch):
    root=tmp_path/'partial';root.mkdir();save(root/'status.json',dict(status='training'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('No array may open before completed grid'))
    monkeypatch.setattr(reporter,'join_l1',lambda *a,**k:pytest.fail('No early L1 join'))
    assert not reporter.require_ready(root)
    with pytest.raises(ValueError,match='no early'):reporter.analyze(root,tmp_path/'report',l1_directory='unused')
    assert not (tmp_path/'report').exists()


def test_complete_grid_reports_all216_seed_rows_and_no_new_intervention(complete_grid,tmp_path):
    original=(complete_grid/'status.json').read_bytes();output=tmp_path/'report';doc=tmp_path/'current.md'
    result=reporter.analyze(complete_grid,output,report_path=doc,l1_directory=None)
    assert result['science_status']=='complete' and result['models']==18 and result['prediction_windows']==216
    assert result['population_CF_H3']=='NOT_TESTED_SINGLE_CONDITION'
    assert result['local_permutation_seeds']==[0] and not result['frozen_weight_intervention_executed']
    for name,n in [('window_metrics.csv',72),('window_comparisons.csv',48),('horizon_summary.csv',24),
                   ('clock_sensitivity.csv',12),('per_seed_metrics.csv',216)]:
        with (output/name).open() as f:assert len(list(csv.DictReader(f)))==n
    assert json.loads((output/'prepared_intervention.json').read_text())['status']=='PREPARED_NOT_RUN'
    provenance=json.loads((output/'provenance.json').read_text())
    assert len([r for r in provenance['artifacts'] if r['path'].endswith('model.msgpack')])==18
    assert all(sha256(r['path'])==r['sha256'] for r in provenance['artifacts'])
    assert (complete_grid/'status.json').read_bytes()==original
    assert doc.read_text()==(output/'RESULTS.md').read_text()
    assert '动力学变换的额外价值尚未建立' in doc.read_text()
    with (output/'window_metrics.csv').open() as f:
        assert sum(r['population_reference_self']=='True' for r in csv.DictReader(f))==36
    assert 'origin10/15包含相对05h的实际变化' in doc.read_text()


def test_publication_failure_does_not_fail_completed_science(complete_grid,tmp_path):
    def broken(*args):raise RuntimeError('simulated push rejection')
    result=reporter.analyze(complete_grid,tmp_path/'report',publish=True,publisher=broken,l1_directory=None)
    assert result['science_status']=='complete' and result['publication']['status']=='failed'
    assert json.loads((tmp_path/'report/status.json').read_text())['status']=='complete'
    assert reporter.require_ready(complete_grid)


def test_completed_checkpoint_tamper_is_detected(complete_grid,tmp_path):
    path=complete_grid/'cf_static/seed0/train/model.msgpack';path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):
        reporter.analyze(complete_grid,tmp_path/'report',l1_directory=None)


def test_prepared_mask_changes_only_standardized_population_token_without_mutation():
    conditions=dict(population_token=np.ones((5,1,3)),cell_token=np.arange(15).reshape(5,1,3),
        bio_time=np.ones((5,2)),bio_delta=np.ones((5,2))*2,perturbation=np.ones((5,1,2)))
    masked=reporter.population_token_mask(conditions)
    assert np.all(masked['population_token']==0) and np.all(conditions['population_token']==1)
    for k in conditions:
        if k!='population_token':np.testing.assert_array_equal(masked[k],conditions[k])
    conditions['population_token'][0]=2
    with pytest.raises(ValueError,match='constant'):reporter.population_token_mask(conditions)


def test_l1_join_occurs_only_after_complete_grid_and_remains_decay_only(complete_grid,tmp_path,monkeypatch):
    l1=tmp_path/'l1';l1.mkdir();save(l1/'summary.json',dict(status='L1_W13_SHARED_SCORER_REPRODUCED'))
    def joined(l1_directory,common_directory,output,*,expected_arms):
        assert reporter.require_ready(complete_grid) and tuple(expected_arms)==reporter.ARMS
        rows=[dict(common_arm=arm,horizon_hours=horizon,L1_real_energy=.15,L1_static_energy=.2,
                   common_energy=.3) for arm in expected_arms for horizon in (5,10,15)]
        result=dict(status='L1_COMMON_GENUINE_DECAY_COMPARISON_COMPLETE',horizon_summary=rows,
            L1_linear_fabricated=False)
        save(Path(output)/'summary.json',result);return result
    monkeypatch.setattr(reporter,'join_l1',joined)
    result=reporter.analyze(complete_grid,tmp_path/'report',l1_directory=l1)
    assert not result['L1_linear_fabricated'] and not result['L1_comparison']['L1_linear_fabricated']
    text=(tmp_path/'report/RESULTS.md').read_text()
    assert 'L1保留解析均值锚' in text and '没有构造或推测L1 linear' in text


def test_watcher_stops_on_failed_training_without_restart(tmp_path):
    status=tmp_path/'job_status.json';save(status,dict(status='failed'))
    with pytest.raises(RuntimeError,match='will not restart'):
        reporter.watch(tmp_path/'missing',tmp_path/'report',training_job_status=status,poll_seconds=.001)
