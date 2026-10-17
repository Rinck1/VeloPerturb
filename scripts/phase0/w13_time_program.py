"""Bounded unattended execution of requested steps 1/2/4, explicitly excluding 3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from w4_artifacts import Artifacts,sha256
from w7_router_scaling import atomic_json,training_path,write_csv
from w10_recipe_bridge import wait_for_workers,aggregate
from w13_time_training import prepare


ROOT = Path(__file__).resolve().parents[2]


def launch(script,gpu,output,arguments):
    subprocess.run([sys.executable,str(ROOT/'scripts/phase0/launch_gpu_job.py'),'--gpu',str(gpu),
        '--script',script,'--output',str(output),'--',*arguments],cwd=ROOT,check=True)


def wait_job(directory):
    deadline = time.monotonic()+6*3600
    while time.monotonic()<deadline:
        path = Path(directory)/'status.json'
        if path.exists():
            status = json.loads(path.read_text())
            if status['status']=='complete':return
            if status['status']=='failed':raise RuntimeError(f'Job failed: {directory}')
        time.sleep(10)
    raise TimeoutError(f'Job exceeded six hours: {directory}')


def endpoint_spec(parent,name):
    parent = Path(parent);manifest = json.loads((parent/'study_manifest.json').read_text())
    records = []
    for job in manifest['jobs']:
        path = training_path(parent,job)/'model_step36000.pt'
        records.append(dict(arm=job['arm'],seed=job['seed'],path=str(path.resolve()),sha256=sha256(path)))
    return dict(name=name,kind='endpoint',checkpoints=records)


def run_program(continuation,output,config_path,audit_xml,job_root):
    continuation,output,job_root = Path(continuation),Path(output),Path(job_root)
    import xml.etree.ElementTree as ET
    suites = ET.parse(audit_xml).getroot()
    selected = [suites] if suites.tag=='testsuite' else list(suites)
    if not selected or not sum(int(s.get('tests',0)) for s in selected) or any(
        int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in selected):
        raise ValueError('Passing audit required')
    run = Artifacts(output,seed=[0,1,2],inputs=[Path(config_path),Path(audit_xml),Path(__file__),
        ROOT/'PLANS/w13_time_dynamics.md'])
    def status(stage,**values):
        run.json('status.json',dict(status=stage,**values));print(json.dumps(dict(status=stage,**values)),flush=True)
    try:
        status('waiting_for_direct_continuation')
        wait_for_workers(continuation);continued = aggregate(continuation)
        time_root = output/'time_training';manifest = prepare(continuation,time_root,config_path)
        status('training_only_time_smoke')
        smoke_job = job_root/'time_smoke_gpu1'
        launch('scripts/phase0/w13_time_training.py',1,smoke_job,
            ['--root',str(time_root),'--benchmark-output',str(output/'training_only_smoke')])
        wait_job(smoke_job)
        # Workers 1 and 2 can train while GPU1 freezes/scores the endpoint horizons.
        for index,gpu in ((1,2),(2,3)):
            launch('scripts/phase0/w13_time_training.py',gpu,job_root/f'time_gpu{gpu}',
                ['--root',str(time_root),'--worker',str(index)])
        endpoints = [endpoint_spec(ROOT/'outputs/w10_scifate2_recipe_bridge_v1_continuation','b0_endpoint36k'),
                     endpoint_spec(continuation,'b1_endpoint36k')]
        first_spec = output/'endpoint_horizon_spec.json'
        atomic_json(first_spec,dict(reference_root=str(continuation.resolve()),models=endpoints))
        first_job = job_root/'endpoint_horizons_gpu1'
        status('endpoint_horizons_and_time_training')
        launch('scripts/phase0/w13_frozen_horizons.py',1,first_job,
            ['--spec',str(first_spec),'--output',str(output/'endpoint_horizons')])
        wait_job(first_job)
        launch('scripts/phase0/w13_time_training.py',1,job_root/'time_gpu1',
            ['--root',str(time_root),'--worker','0'])
        status('training_time_grid',jobs=len(manifest['jobs']))
        for gpu in (1,2,3):wait_job(job_root/f'time_gpu{gpu}')
        recipes = [endpoints[1]]
        for kind in manifest['config']['dynamics']:
            records = []
            for job in [j for j in manifest['jobs'] if j['dynamics']==kind]:
                path = training_path(time_root,job)/f'model_step{job["config"]["steps"]}.pt'
                records.append(dict(arm=job['arm'],seed=job['seed'],path=str(path.resolve()),sha256=sha256(path)))
            recipes.append(dict(name=f'{kind}_40k',kind=kind,checkpoints=records))
        final_spec = output/'time_horizon_spec.json'
        atomic_json(final_spec,dict(reference_root=str(continuation.resolve()),models=recipes))
        final_job = job_root/'time_horizons_gpu1';status('frozen_time_horizon_evaluation')
        launch('scripts/phase0/w13_frozen_horizons.py',1,final_job,
            ['--spec',str(final_spec),'--output',str(output/'time_horizons')])
        wait_job(final_job)
        first = json.loads((output/'endpoint_horizons/summary.json').read_text())
        final = json.loads((output/'time_horizons/summary.json').read_text())
        rows = [r for r in first['horizon_summary'] if r['model']=='b0_endpoint36k']+final['horizon_summary']
        integration = []
        for horizon in (5,10,15):
            by_name = {r['model']:r for r in rows if r['horizon_hours']==horizon}
            ode = by_name['ode_40k']['real_energy'];endpoint = by_name['endpoint_time_40k']['real_energy']
            integration.append(dict(horizon_hours=horizon,ode_energy=ode,matched_endpoint_energy=endpoint,
                gain_vs_matched_endpoint=1-ode/endpoint,
                gain_vs_original_endpoint=1-ode/by_name['b1_endpoint36k']['real_energy']))
        write_csv(output/'horizon_summary.csv',rows);write_csv(output/'integration_comparisons.csv',integration)
        result = dict(status='REQUESTED_STEPS_1_2_4_COMPLETE',continuation_comparisons=continued['comparisons'],
            horizon_summary=rows,integration_comparisons=integration,excluded_step_3=True,
            ode_parameters=manifest['jobs'][0]['expected_parameters'],
            scope='same_series_development',original_gfg_used=False,
            global_mean_frozen=True,individual_fates=False,
            interpretation='ODE has extra compute per update; parameters, initialization and extra update count matched')
        run.json('summary.json',result)
        report = '# W13: continuation, horizons, and genuine ODE\n\n'
        report += '| Model | Hours | Energy | vs static | vs shuffled |\n|---|---:|---:|---:|---:|\n'
        report += '\n'.join(f'| {r["model"]} | {r["horizon_hours"]} | {r["real_energy"]:.6f} | '
            f'{100*r["gain_vs_static"]:+.3f}% | {100*r["gain_vs_shuffled"]:+.3f}% |' for r in rows)
        report += '\n\n## Integration effect\n\n'
        report += '\n'.join(f'- {r["horizon_hours"]}h: {100*r["gain_vs_matched_endpoint"]:+.3f}% vs matched endpoint time model.' for r in integration)
        report += '\n\nSame-series development, not independent replication or genetic perturbation generalization. '
        report += 'Static/shuffled labels describe local routing; the analytical mean is shared. '
        report += 'Step 3 was not run. No original GFG/JVP was introduced.\n'
        (output/'RESULTS.md').write_text(report)
        status('complete',report=str(output/'RESULTS.md'))
    except Exception as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--continuation',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--job-root',required=True)
    parser.add_argument('--config',default='configs/w13_scifate2_time_v1.json');parser.add_argument('--audit-xml',required=True)
    args = parser.parse_args();run_program(args.continuation,args.output,args.config,args.audit_xml,args.job_root)
