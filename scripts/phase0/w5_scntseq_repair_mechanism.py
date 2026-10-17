"""Exploratory decomposition of v2 gains; no change to the fixed 67-arm grid.

Compare the selected candidate with matched nonlinear S-only, U-derivative,
and source U-total-only controls. All readouts use the same calibration and
ridge/time kernel. Never interpret a within-population U shuffle as a null
for pooled features (their sum would be unchanged).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts, sha256
from w4_shorttime import read_metadata
from w4_streams import SourceTransform
from w5_scntseq_minutes import INTERVALS, load_population
from w5_scntseq_repair import (effective_minutes, fit_readout, readout_rate, setup, sufficient_features)


def components(source, baseline, gamma, rotation):
    s = (-gamma*(source['pool_s']-baseline['pool_s'])) @ rotation.T
    u = (source['pool_u']-baseline['pool_u']) @ rotation.T
    if baseline['full_U_to_S'] <= 0:
        raise ValueError('Positive baseline total U/S required')
    replacement = baseline['normalized_pool_u']*(source['full_U_to_S']/baseline['full_U_to_S'])
    control = ((replacement/(1+source['normalized_pool_s'])-gamma*source['pool_s'])
        -(baseline['pool_u']-gamma*baseline['pool_s'])) @ rotation.T
    return dict(S_kinetic=s,U_derivative=u,U_total_only=control)


def audit(predictions,output):
    directory = Path(predictions)
    manifest = json.loads((directory/'prediction_manifest.json').read_text())
    config_path = Path(manifest['config_path'])
    config,original_config,data,transform_path = setup(config_path)
    if sha256(directory/'predictions.npz') != manifest['prediction_sha256']:
        raise ValueError('Fixed candidate changed')
    run = Artifacts(output,seed=config['seed'],inputs=[Path(__file__),
        Path(__file__).with_name('w5_scntseq_repair.py'),config_path,data,transform_path,
        directory/'prediction_manifest.json',directory/'gamma_estimates.npz',directory/'predictions.npz'])
    transform = SourceTransform.load(transform_path)
    with np.load(directory/'gamma_estimates.npz',allow_pickle=False) as saved:
        gamma = saved['pooled_shrink']
    obs,genes = read_metadata(data);np.testing.assert_array_equal(genes,transform.gene_names)
    populations = {}
    def load(minute):
        layers,_ = load_population(data,obs,minute,original_config,source=True)
        values = sufficient_features(transform,layers)
        values['full_U_to_S'] = float(layers['unspliced'].sum()/layers['spliced'].sum())
        return values
    for minute in (0,15,30):populations[minute] = load(minute)
    def features(minute):
        offset = populations[minute]['state']-populations[0]['state']
        return {key:[offset,value] for key,value in components(populations[minute],populations[0],gamma,transform.components).items()}
    rate = (populations[30]['state']-populations[15]['state'])/15
    fitted = {key:fit_readout(value,rate,[False,key == 'U_total_only'],ridge=config['readout_ridge'])
              for key,value in features(15).items()}
    predictions = []
    for start,stop in INTERVALS:
        if start == 60:populations[60] = load(60)
        duration = effective_minutes(stop-start,15,'decay')
        prediction = {key:populations[start]['state']+duration*readout_rate(value,fitted[key])
                      for key,value in features(start).items()}
        np.savez_compressed(run.directory/f'controls_{start}_to_{stop}.npz',**prediction)
        predictions.append(prediction)
    with np.load(directory/'predictions.npz',allow_pickle=False) as saved:
        index = saved['arms'].tolist().index(config['primary_candidate'])
        candidate = saved['predictions'][:,index].copy()
    last,_ = load_population(data,obs,120,original_config,source=False)
    targets = [populations[60]['state'],transform.state(last['spliced']).mean(0)]
    rows = []
    names = ['candidate','S_kinetic','U_derivative','U_total_only']
    for i,(start,stop) in enumerate(INTERVALS):
        base_error = float(np.square(populations[start]['state']-targets[i]).sum())
        values = dict(candidate=candidate[i],**predictions[i])
        for key,prediction in values.items():
            error = float(np.square(prediction-targets[i]).sum())
            rows.append(dict(source_minute=start,target_minute=stop,arm=key,
                mse=error/len(targets[i]),gain_over_identity=(base_error-error)/base_error))
    errors = {name:[r['mse'] for r in rows if r['arm'] == name] for name in names}
    result = dict(status='EXPLORATORY_REPAIR_MECHANISM_COMPLETE',exploratory=True,
        original_grid_changed=False,independent_confirmation=False,matched_parameters=2,
        candidate_beats_control_mean={name:bool(np.mean(errors['candidate']) < np.mean(errors[name])) for name in names[1:]},
        candidate_beats_control_both={name:bool((np.array(errors['candidate']) < np.array(errors[name])).all()) for name in names[1:]},
        mse_by_interval=errors,calibration_interval=[15,30],readouts=fitted,
        U_total_control_uses_current_total_U_but_not_current_U_gene_composition=True)
    with (run.directory/'metrics.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    run.json('summary.json',result)
    (run.directory/'RESULTS.md').write_text('# Exploratory v2 mechanism controls\n\n'
        'The nonlinear S-only arm uses no current U. The U-total control uses current U depth '
        'but baseline gene composition. No v2 primary score is changed.\n\n'+json.dumps(result,indent=2)+'\n')
    run.json('status.json',dict(status='complete'))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions',default='outputs/w5_scntseq_repair_predictions_v2')
    parser.add_argument('--output',required=True)
    args = parser.parse_args();audit(args.predictions,args.output)
