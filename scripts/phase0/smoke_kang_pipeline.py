"""Actual-GFG engineering smoke on explicitly SYNTHETIC Kang-shaped counts."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from veloroute.artifacts import load_config, save_json
from veloroute.gfg_experiments import train_gfg
from veloroute.kang_data import prepare_kang_fold
from veloroute.kang_experiments import (ROUTER_ARMS,base_config,prepare_geometry,train_router,predict_router,predict_simple)

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
c=load_config('configs/veloroute_kang_20260914.yaml')
c['root']=str(root);c['synthetic_smoke']=True
c['representation'].update(selected_genes=6,pca_components=3)
c['cell_types']['minimum_cells_per_donor_side']=2
c['training'].update(geometry_stage_a_steps=2,geometry_stage_b_steps=2,geometry_stage_c_steps=4,
    router_steps=4,source_batch=4,target_batch=8,model_width=16,residual_blocks=1)
rng=np.random.default_rng(3)
for side in ('ctrl','stim'):
    obs=pd.DataFrame(dict(donor=np.repeat(c['donors'],6),condition=side,cell_type='CD14+ Monocytes',technical_qc_pass=True),
        index=[side+':'+d+'_'+str(i) for d in c['donors'] for i in range(6)])
    s=sparse.csr_matrix(rng.poisson(10,(len(obs),8))+1,dtype='float32')
    u=sparse.csr_matrix(rng.poisson(5,(len(obs),8))+1,dtype='float32')
    data=ad.AnnData(s.copy(),obs=obs,var=pd.DataFrame(index=[f'ENSG{i}' for i in range(8)]))
    data.layers.update(spliced=s,unspliced=u)
    path=root/'processed'/side;path.mkdir(parents=True)
    data.write_h5ad(path/f'{side}.h5ad')
donor='101';fold=root/'folds'/donor
prepare_kang_fold(c,donor,fold)
base=base_config(c,donor,0)
train_gfg(base,fold,root/'geometry_train',arm='static',seed=0,pilot=False)
prepare_geometry(c,donor,0,root/'geometry_train/model.pt',root/'geometry')
rows=[]
for arm in ROUTER_ARMS:
    train=root/arm/'train';pred=root/arm/'predict'
    result=train_router(c,donor,0,arm,root/'geometry',train)
    predict_router(c,donor,0,arm,root/'geometry',train/'model.pt',pred)
    rows.append(result)
for arm in c['experiments']['simple_baselines']:
    predict_simple(c,donor,0,arm,root/arm/'predict')
with np.load(root/'gfg_joint_router/predict/predictions.npz') as real:
    assert real['q'].shape==(6,2) and np.allclose(real['q'].sum(1),1)
    for arm in ROUTER_ARMS:
        with np.load(root/arm/'predict/predictions.npz') as other:
            np.testing.assert_array_equal(real['candidates'],other['candidates'])
assert next(r['maximum_GFG_task_gradient'] for r in rows if r['arm']=='gfg_joint_router')>0
save_json(root/'summary.json',dict(status='KANG_SYNTHETIC_ENGINEERING_SMOKE_PASSED',actual_GFG=True,
    common_geometry_verified=True,source_only_predictions=True,rows=rows,real_Kang_results=False))
(root/'RESULTS.md').write_text('# Kang pipeline engineering smoke\n\nSynthetic counts only. Actual GFG JVP/router gradient and all five routing controls completed. No biological success claim.\n')
print(json.dumps(dict(status='KANG_SYNTHETIC_ENGINEERING_SMOKE_PASSED',root=str(root))),flush=True)
