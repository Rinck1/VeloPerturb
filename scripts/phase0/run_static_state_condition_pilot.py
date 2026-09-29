"""Minimal static state x condition router pilot on the RENGE development fold.

This is the lowest-risk alternative to a velocity fate router: same experts,
same unpaired training protocol and budget, but the deployed router receives
only z0 and the perturbation condition embedding. No U/S or GFG call is made.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

from veloroute.artifacts import load_config, save_json, sha256
from veloroute.gfg_experiments import train_gfg
from veloroute.gpu_policy import enforce_gpu_policy
from veloroute.latent import load_pack

from run_gain_mainline import build_config, predict_arm

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=0); p.add_argument('--max-steps',type=int,default=400); args=p.parse_args()
    enforce_gpu_policy(args.device)
    fold=Path('outputs/veloroute_real_pipeline_20260912_v2/fold'); gene_dir=Path('outputs/veloroute_gfg_inputs_20260914')
    conditions='data/renge/conditions/esm2_3b_v1/conditions.npz'; output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    config=build_config(fold,args.seed,args.device,coupling='ot',conditions_path=conditions)
    config['pilot']['stage_a_steps']=1; config['pilot']['stage_b_steps']=max(20,args.max_steps//4); config['pilot']['stage_c_steps']=args.max_steps
    config['training']['coupling_mode']='ot'; config['model']['use_dynamics']=False; config['model']['joint_dynamics']=False; config['model']['static_uses_trained_field']=True
    save_json(output/'config.json',config)
    train=output/'train'; train_gfg(config,gene_dir,train,arm='static',seed=args.seed,pilot=True)
    pred=predict_arm(config,fold,train/'model.pt','static',args.seed,output/'predict',gene_dir=gene_dir)
    target,tm=load_pack(fold/'validation_target.npz',expected_side='target')
    import numpy as np
    from veloroute.metrics import energy_distance
    with np.load(pred,allow_pickle=False) as d:
        rows=[]
        for c in sorted(set(d['conditions'])):
            x=d['z'][d['conditions']==c]; y=target['z'][target['conditions']==c]
            rows.append({'condition':c,'energy_distance':float(energy_distance(x,y)),'variance_ratio':float(x.var(0).sum()/max(y.var(0).sum(),1e-12)),'source_cells':len(x),'target_cells':len(y)})
    save_json(output/'metrics.json',rows)
    result={'status':'STATIC_STATE_CONDITION_ROUTER_PILOT_COMPLETE','router_inputs':['z0','condition_embedding'],'velocity_read':False,'target_used_only_after_prediction_freeze':True,'steps':args.max_steps,'mean_energy_distance':float(np.mean([r['energy_distance'] for r in rows])),'mean_variance_ratio':float(np.mean([r['variance_ratio'] for r in rows])),'prediction_sha256':sha256(pred),'rows':rows,'interpretation':'exploratory pilot; not a confirmatory comparison'}
    save_json(output/'summary.json',result); (output/'RESULTS.md').write_text('# Static state x condition router pilot\n\n'+json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
