"""CF-0 environment/API evidence and CF-1 official-model synthetic checks."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import time
import urllib.request

import numpy as np
from scipy.spatial.distance import cdist

from w4_artifacts import Artifacts,sha256
from w15a_cellflow_adapter import (CELLFLOW_ROOT,ROOT,add_cellflow_paths,biological_fourier,
    predict_cellflow_tau)


def audit(output):
    add_cellflow_paths()
    import cellflow
    import jax
    import jax.numpy as jnp
    import diffrax
    run=Artifacts(output,seed=0,inputs=[Path(__file__),Path(__file__).with_name('w15a_cellflow_adapter.py')])
    versions={}
    for name in ('jax','jaxlib','jax-cuda12-plugin','jax-cuda12-pjrt','flax','ott-jax','diffrax','optimistix','equinox','optax'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    evidence={}
    urls=dict(geo='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7548960',
        author_notebook='https://raw.githubusercontent.com/rorymaizels/Maizels2023aa/main/analysis/A0.0_making_adata_sets.ipynb')
    for label,url in urls.items():
        request=urllib.request.Request(url,headers={'User-Agent':'VeloRoute technical audit'})
        with urllib.request.urlopen(request,timeout=90) as response:payload=response.read()
        path=run.directory/(label+('.ipynb' if label=='author_notebook' else '.html'))
        path.write_bytes(payload);evidence[label]=dict(url=url,path=str(path),sha256=sha256(path))
    metadata=json.loads((CELLFLOW_ROOT.parent/'CellFlow_acquisition_branch/summary.json').read_text())
    source_files={str(p.relative_to(CELLFLOW_ROOT)):sha256(p) for p in sorted((CELLFLOW_ROOT/'src/cellflow').rglob('*.py'))}
    check=float(jnp.ones((4,4)).sum().block_until_ready())
    result=dict(status='CELLFLOW_API_AUDIT_COMPLETE',cellflow_source=str(Path(cellflow.__file__).resolve()),
        source_archive=metadata,source_python_sha256=source_files,versions=versions,
        devices=[str(d) for d in jax.devices()],gpu_operation_sum=check,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        xla_python_client_preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
        ld_library_path=os.environ.get('LD_LIBRARY_PATH'),global_environment_modified=False,
        api=dict(arbitrary_source='native_source_control_boolean_semantics_need_adapter',
            nonpooled_covariates='native_ConditionEncoder_covariates_not_pooled',
            group_plugin_MLP='native_per_input_layers_before_pool',
            per_cell_conditions='field_native_but_OT_resampling_and_predict_vmap_need_wrapper',
            partial_tau='wrapper_native_predict_hardcodes_t1_1',
            default_solver='diffrax.Tsit5_PID_rtol1e-5_atol1e-5_adaptive_not_fixed_steps'),
        biological_time=dict(origin='D3_RA_SAG_start_equals_0h',
            daily='24*(day_number-3)',hourly='author_nominal_05_10_15_20h',
            distinction='GEO_decimal_days_are_rounded_not_used_as_4.8h_intervals',
            origin_is_implementation_inference_from_combined_author_metadata=True,
            frequencies=[1,2,4,8],evidence=evidence),
        known_mathematics=dict(L3_weighted_energy='exact_energy_of_mixture_measure',
            fixed_one_expert_per_source_Vstatistic='NOT_exact_same_expectation_due_within_source_diagonal',
            expected_sampled_minus_weighted='sum_i E_k,l distance(candidate_i,k,candidate_i,l) / N^2'),
        biological_training=False,Traxler_effects_read=False)
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    return result


def energy(x,y):
    return float(2*cdist(x,y).mean()-cdist(x,x).mean()-cdist(y,y).mean())


def synthetic(output,repetitions=20,steps=400):
    add_cellflow_paths()
    import jax
    import jax.numpy as jnp
    from w15a_cellflow_train import create_solver,fit_groups
    run=Artifacts(output,seed=list(range(repetitions)),inputs=[Path(__file__),
        Path(__file__).with_name('w15a_cellflow_adapter.py'),Path(__file__).with_name('w15a_cellflow_train.py')])
    config=dict(width=32,depth=2,condition_dim=8,time_freqs=8,time_width=16,time_depth=1,
        learning_rate=2e-3,ot_epsilon=.1,ot_max_iterations=100,batch_size=32,steps=steps,log_interval=steps)
    def conditions(hidden,arm,n=1):
        return dict(perturbation=np.ones((n,1,1),dtype='float32'),
            bio_time=np.broadcast_to(biological_fourier(0),(n,1,8)).astype('float32'),
            bio_delta=np.broadcast_to(biological_fourier(24),(n,1,8)).astype('float32'),
            population_token=np.full((n,1,1),hidden if arm=='cf_kin' else 0.,dtype='float32'),
            cell_token=np.zeros((n,1,1),dtype='float32'))
    solver=create_solver(config,conditions(0.,'cf_static'),1,0)
    # One cached native-field RK4 predictor for all 120 fits. No private source modifications.
    def prediction(params,x,cond):
        def velocity(state,t):
            return solver.vf.apply({'params':params},jnp.full((len(state),1),t),state,cond,
                encoder_noise=jnp.zeros((len(state),8)),train=False)[0]
        def advance(index,state):
            t=index/32.;h=1/32.;a=velocity(state,t);b=velocity(state+h*a/2,t+h/2)
            c=velocity(state+h*b/2,t+h/2);d=velocity(state+h*c,t+h)
            return state+h*(a+2*b+2*c+d)/6
        return jax.lax.fori_loop(0,32,advance,x)
    predict=jax.jit(prediction)
    rows=[];started=time.monotonic()
    for scenario in ('positive','null'):
        for repetition in range(repetitions):
            rng=np.random.default_rng(1000+repetition)
            train=[];test=[]
            for index in range(24):
                hidden=float(rng.choice([-1.,1.]))
                p=(.85 if hidden>0 else .15) if scenario=='positive' else .5
                x=rng.normal(0,.3,size=(128,1)).astype('float32')
                # Same mode widths and separation; only component mass changes.
                y=(np.where(rng.random((128,1))<p,2.,-2.)+rng.normal(0,.15,size=(128,1))).astype('float32')
                record=dict(z=x,target=y,hidden=hidden)
                (train if index<16 else test).append(record)
            for arm in ('cf_static','cf_kin','cf_Ssum'):
                solver.vf_state=solver.vf.create_train_state(jax.random.PRNGKey(repetition),
                    solver.vf_state.tx,input_dim=1,conditions=conditions(0.,arm))
                solver.vf_state_inference=solver.vf_state
                groups=[dict(z=g['z'],target=g['target'],condition=conditions(g['hidden'],arm)) for g in train]
                fit_groups(solver,groups,config,repetition)
                x=np.concatenate([g['z'] for g in test]); cond={k:np.concatenate([
                    conditions(g['hidden'],arm,len(g['z']))[k] for g in test]) for k in conditions(0.,arm)}
                pred=np.asarray(predict(solver.vf_state_inference.params,jnp.asarray(x),
                    {k:jnp.asarray(v) for k,v in cond.items()}))
                distances=[energy(pred[i*128:(i+1)*128],g['target']) for i,g in enumerate(test)]
                row=dict(scenario=scenario,repetition=repetition,arm=arm,mean_energy=float(np.mean(distances)),
                    runtime_seconds=time.monotonic()-started)
                rows.append(row)
                run.json('status.json',dict(status='synthetic_training',completed_fits=len(rows),total_fits=6*repetitions,
                    scenario=scenario,repetition=repetition,arm=arm))
                print(json.dumps(row),flush=True)
    gains={}
    for scenario in ('positive','null'):
        differences=[]
        for repetition in range(repetitions):
            d={r['arm']:r['mean_energy'] for r in rows if r['scenario']==scenario and r['repetition']==repetition}
            differences.append((d['cf_static']-d['cf_kin'])/max(d['cf_static'],1e-12))
        gains[scenario]=dict(relative_gains=differences,mean=float(np.mean(differences)),
            median=float(np.median(differences)),positive_repetitions=int(np.sum(np.array(differences)>0)))
    with (run.directory/'all_repetitions.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='CELLFLOW_SYNTHETIC_COMPLETE',repetitions_per_scenario=repetitions,steps=steps,
        fits=len(rows),gains=gains,config=config,source_summary_control='theoretical_identical_S_population_summary_zero',
        synthetic_only=True,biological_positive=False,
        interpretation='All repetitions retained; no single-run threshold is a statistical discovery rule')
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['audit','synthetic'])
    parser.add_argument('--output',required=True);parser.add_argument('--repetitions',type=int,default=20)
    parser.add_argument('--steps',type=int,default=400);args=parser.parse_args()
    result=audit(args.output) if args.action=='audit' else synthetic(args.output,args.repetitions,args.steps)
    print(json.dumps(result,indent=2))
