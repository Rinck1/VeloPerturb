"""Exercise full EM/gate prediction on synthetic fixtures; do not retrain CellOT."""
import json
from pathlib import Path

from veloroute.artifacts import load_config,save_json
from veloroute.gfg_experiments import train_gfg
from veloroute.kang_full import predict_full,variant_config

c=load_config('configs/veloroute_kang_20260914.yaml');actual_root=Path(c['root'])
c['root']=str(actual_root/'smoke_v1');c['synthetic_smoke']=True
c['representation'].update(selected_genes=6,pca_components=3)
c['training'].update(full_stage_a_steps=2,full_stage_b_steps=2,full_stage_c_steps=4,source_batch=4)
extended=load_config('configs/veloroute_kang_extended_20260914.yaml')
root=actual_root/'smoke_extended_v2';root.mkdir(exist_ok=False)
rows=[]
for v in (extended['full_variants'][0],extended['full_variants'][2]):
    variant={**v,'experts':2}
    base=variant_config(c,'101',0,variant)
    base['model'].update(hidden_dim=16,residual_blocks=1)
    base['training'].update(e_interval=1,teacher_lag=1,activation_interval=1,reference_interval=2,usage_interval=4)
    out=root/variant['name']
    rows.append(train_gfg(base,Path(c['root'])/'folds/101',out/'train',arm=variant['arm'],seed=0,pilot=False))
    predict_full(c,'101',0,variant,out/'train/model.pt',out/'predict')
save_json(root/'summary.json',dict(status='KANG_EXTENDED_SYNTHETIC_SMOKE_PASSED',full_variants=[r['arm'] for r in rows],
    cellot='not_run_reuse_existing_only',real_Kang=False))
(root/'RESULTS.md').write_text('# Extended engineering smoke\n\nSynthetic counts only. Full EM/gate, static matched prediction, and source-only corruption exercised. CellOT is excluded: reuse existing reproduction only.\n')
print(json.dumps(dict(status='KANG_EXTENDED_SYNTHETIC_SMOKE_PASSED')),flush=True)
