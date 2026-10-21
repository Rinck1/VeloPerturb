"""One bounded cf_static development candidate: train -> freeze -> score.

This runner does not tune plugin arms, touch Traxler, or launch another GPU.
Root-agent approval is explicit; CF-1 evidence is required before training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from w4_artifacts import Artifacts,sha256


def run(config_path,pack,gate_path,test_xml,output,*,candidate,seed=0,approved=False):
    if not approved:
        raise ValueError('Root-agent approval required after reviewing complete CF-1 results')
    gate_path=Path(gate_path);test_xml=Path(test_xml)
    gate=json.loads(gate_path.read_text())
    if gate.get('status')!='CF1_ENGINEERING_GATE_EVALUATED' or not gate.get('engineering_gate_passed'):
        raise ValueError('Complete passing CF-1 engineering gate is required')
    test_root=ET.parse(test_xml).getroot();suites=list(test_root.iter('testsuite'))
    if not suites or sum(int(s.get('tests','0')) for s in suites)<15 or any(
        int(s.get('failures','0'))+int(s.get('errors','0')) for s in suites):
        raise ValueError('All adapter and full pipeline tests must pass')
    config=json.loads(Path(config_path).read_text())
    matches=[c for c in config['initial_static_configurations'] if c['name']==candidate]
    if len(matches)!=1:raise ValueError('Unknown registered static candidate')
    config.update(matches[0])
    run=Artifacts(output,seed=seed,inputs=[Path(config_path),gate_path,test_xml,Path(__file__)])
    actual=run.directory/'static_config.json';actual.write_text(json.dumps(config,indent=2)+'\n')
    run.json('status.json',dict(status='training_cf_static',candidate=candidate))
    from w15a_cellflow_train import train
    from w15a_cellflow_predict import predict
    from w15a_cellflow_evaluate import evaluate
    fitted=train(pack,actual,run.directory/'train',arm='cf_static',seed=seed)
    run.json('status.json',dict(status='freezing_source_only_predictions',candidate=candidate))
    predict(run.directory/'train',run.directory/'predict')
    run.json('status.json',dict(status='scoring_development',candidate=candidate))
    result=evaluate([run.directory/'predict'],run.directory/'evaluation')
    summary=dict(status='CF2_SINGLE_STATIC_CANDIDATE_COMPLETE',candidate=candidate,seed=seed,
        parameters=fitted['parameters'],selection_energy=result['selection_energy'],
        model_sha256=fitted['model_sha256'],configuration_sha256=sha256(actual),
        development_only=True,formal_three_seed_comparison=False,Traxler_read=False,
        remaining_static_candidates='not_automatically_launched_by_this_one_candidate_runner')
    run.json('summary.json',summary);run.json('status.json',dict(status='complete'))
    return summary


def grid(config_path,pack,gate_path,test_xml,output,*,seed=0,approved=False):
    """Four approved static candidates sequentially on the caller's single GPU."""
    if not approved:raise ValueError('Root-agent approval required')
    config=json.loads(Path(config_path).read_text());candidates=config['initial_static_configurations']
    if len(candidates)>config['maximum_static_configurations'] or len(candidates)!=4:
        raise ValueError('This queue is limited to the four approved initial static candidates')
    runlog=Artifacts(output,seed=seed,inputs=[Path(config_path),Path(gate_path),Path(test_xml),Path(__file__)])
    results=[]
    for specification in candidates:
        name=specification['name'];directory=runlog.directory/f'{name}_seed{seed}'
        runlog.json('status.json',dict(status='running_static_candidate',candidate=name,
            completed=len(results),total=len(candidates),physical_gpu_policy='single_inherited_visible_device'))
        result=run(config_path,pack,gate_path,test_xml,directory,candidate=name,seed=seed,approved=True)
        results.append(dict(candidate=name,selection_energy=result['selection_energy'],parameters=result['parameters'],
            directory=str(directory.resolve()),configuration_sha256=result['configuration_sha256']))
        runlog.json('candidate_results.json',dict(completed=results))
    # Deterministic ties prefer the earlier registered candidate, not a plugin score.
    winner=min(results,key=lambda row:row['selection_energy']);selected=Path(winner['directory'])/'static_config.json'
    frozen_directory=runlog.directory/'selection';frozen_directory.mkdir()
    frozen_configuration=frozen_directory/'frozen_l2_configuration.json'
    frozen_configuration.write_bytes(selected.read_bytes())
    selected_config=json.loads(selected.read_text())
    evidence=dict(status='L2_STATIC_CONFIGURATION_SELECTED_AWAITING_COMMIT',
        winner=winner,candidates=results,selection_arm='cf_static',selection_seed=seed,
        selection_metric='equal_weight_05to10_10to15_15to20_decay_energy',
        all_plugin_arms_must_inherit_this_configuration=True,plugin_results_used=False,
        configuration_path=str(frozen_configuration.resolve()),configuration_sha256=sha256(frozen_configuration),
        source_pack_manifest_sha256=sha256(Path(pack)/'manifest.json'),
        backbone={key:selected_config[key] for key in ('width','depth','condition_dim','time_width','time_depth','time_freqs','pooling')},
        optimizer={key:selected_config[key] for key in ('learning_rate','steps','batch_size')},
        solver=dict(name='official_OTFlowMatching',probability_path='ConstantNoiseFlow_sigma0',
            matching='official_balanced_Sinkhorn_mean_scale_cost',epsilon=selected_config['ot_epsilon'],
            max_iterations=selected_config['ot_max_iterations'],prediction='Tsit5_PID_rtol_atol_1e-5',
            primary_time='decay_h(delta)/h(24)',sensitivity='linear_delta/24'),
        source_hashes={name:sha256(Path(__file__).with_name(name)) for name in (
            'w15a_cellflow_adapter.py','w15a_cellflow_train.py','w15a_cellflow_predict.py','w15a_cellflow_evaluate.py')},
        runtime_audit=str((Path(__file__).resolve().parents[2]/'outputs/w15a_cellflow_audit_v1/summary.json')),
        freeze_commit=None,CF3_status='root_agent_must_commit_configuration_and_bind_official_hidden_adapter_before_L3',
        Traxler_read=False,development_only=True)
    (frozen_directory/'freeze_record.json').write_text(json.dumps(evidence,indent=2)+'\n')
    runlog.json('summary.json',evidence);runlog.json('status.json',dict(status='complete'))
    return evidence


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--pack',required=True)
    p.add_argument('--gate',required=True);p.add_argument('--tests',required=True);p.add_argument('--output',required=True)
    p.add_argument('--candidate');p.add_argument('--all-static',action='store_true');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--approved',action='store_true');args=p.parse_args()
    if args.all_static:
        result=grid(args.config,args.pack,args.gate,args.tests,args.output,seed=args.seed,approved=args.approved)
    else:
        if args.candidate is None:p.error('--candidate or --all-static is required')
        result=run(args.config,args.pack,args.gate,args.tests,args.output,candidate=args.candidate,
            seed=args.seed,approved=args.approved)
    print(json.dumps(result,indent=2))
