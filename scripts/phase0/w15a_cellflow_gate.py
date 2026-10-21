"""Evaluate the separately recorded CF-1 engineering gate without selecting runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts


def evaluate_gate(synthetic_directory,config_path,output):
    directory=Path(synthetic_directory);config=json.loads(Path(config_path).read_text())
    result=json.loads((directory/'summary.json').read_text())
    if result['status']!='CELLFLOW_SYNTHETIC_COMPLETE':raise ValueError('Synthetic grid incomplete')
    with (directory/'all_repetitions.csv').open() as stream:rows=list(csv.DictReader(stream))
    run=Artifacts(output,seed=config['bootstrap_seed'],inputs=[Path(config_path),
        directory/'summary.json',directory/'all_repetitions.csv',Path(__file__)])
    comparisons=[];rng=np.random.default_rng(config['bootstrap_seed'])
    for scenario in ('positive','null'):
        for control in config['controls']:
            gains=[];differences=[]
            for repetition in range(config['repetitions_per_scenario']):
                match={r['arm']:float(r['mean_energy']) for r in rows
                    if r['scenario']==scenario and int(r['repetition'])==repetition}
                if set(match)!={config['real_arm'],*config['controls']}:raise ValueError('Incomplete repetition')
                difference=match[control]-match[config['real_arm']]
                differences.append(difference);gains.append(difference/max(match[control],1e-12))
            values=np.array(gains)
            sampled=values[rng.integers(len(values),size=(config['bootstrap_repetitions'],len(values)))].mean(1)
            alpha=(1-config['confidence'])/2;low,high=np.quantile(sampled,[alpha,1-alpha])
            comparisons.append(dict(scenario=scenario,control=control,relative_gains=gains,
                absolute_energy_differences=differences,mean=float(values.mean()),median=float(np.median(values)),
                mean_gain_ci_low=float(low),mean_gain_ci_high=float(high),repetitions=len(values)))
    positive=all(r['median']>0 for r in comparisons if r['scenario']=='positive')
    suspicious_null=all(r['mean_gain_ci_low']>0 for r in comparisons if r['scenario']=='null')
    result=dict(status='CF1_ENGINEERING_GATE_EVALUATED',positive_construction_recognized=positive,
        suspicious_systematic_null_gain=suspicious_null,engineering_gate_passed=positive and not suspicious_null,
        comparisons=comparisons,biological_success=False,false_positive_rate_calibrated=False,
        CF2_start='await_root_agent_review')
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--synthetic',required=True);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    print(json.dumps(evaluate_gate(args.synthetic,args.config,args.output),indent=2))
