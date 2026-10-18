"""Publish the technical acquisition report after all bounded download jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from w4_artifacts import Artifacts,sha256


ROOT=Path(__file__).resolve().parents[2]
STAGES={
    'counts':('traxler_acquisition_v2','w14b_traxler_acquisition_v2'),
    'raw':('traxler_raw','w14b_traxler_raw_v1'),
    'layout':('traxler_layout','w14b_traxler_layout_v1'),
    'reference':('traxler_reference','w14b_traxler_reference_v1'),
    'metadata':('metadata_audits','w14b_metadata_audits_v1'),
    'zscape_counts':('zscape_counts_v2','w14b_zscape_counts_v2')}


def run(output):
    run=Artifacts(output,seed=0,inputs=[Path(__file__)])
    deadline=time.monotonic()+7*86400
    while time.monotonic()<deadline:
        states={}
        for key,(job,artifact) in STAGES.items():
            p=ROOT/'outputs/w14_jobs_20261018'/job/'status.json'
            states[key]=json.loads(p.read_text()).get('status') if p.exists() else 'not_started'
        if 'failed' in states.values():
            run.json('status.json',dict(status='failed',stages=states,reason='Acquisition failed; raw/partial files retained'))
            raise RuntimeError(f'Technical acquisition failed: {states}')
        if all(v=='complete' for v in states.values()):break
        run.json('status.json',dict(status='waiting_for_acquisition',stages=states));time.sleep(20)
    else:raise TimeoutError('Acquisition timeout')
    records={}
    for key,(_,artifact) in STAGES.items():
        p=ROOT/'outputs'/artifact/'summary.json'
        records[key]=dict(path=str(p),sha256=sha256(p),summary=json.loads(p.read_text()))
    run.json('source_records.json',records)
    result=dict(status='W14B_TECHNICAL_ACQUISITION_COMPLETE',SRA_bytes=records['raw']['summary']['total_bytes'],
        SRA_runs=records['raw']['summary']['runs'],S_U_quantification_complete=False,
        biological_effects_read=False,models_trained_on_new_data=False,
        next='Confirm observed barcode layout, build mouse USA index and quantify before W15 registration')
    run.json('summary.json',result)
    text='# W14-B：新数据技术准备\n\n'+json.dumps(result,indent=2)+'\n\n'
    text+='SRA、发布矩阵和参考已完整获取；S/U定量未完成，不把原始数据下载当作新数据阳性。\n'
    (run.directory/'RESULTS.md').write_text(text)
    doc=ROOT/'docs/current/w14b_data_results.md'
    if doc.exists():raise ValueError('Prior technical report retained')
    doc.write_text(text+'\n来源与哈希：`outputs/w14b_data_readiness_v1/source_records.json`。\n')
    files=[str(doc.relative_to(ROOT))]+[str((run.directory/n).relative_to(ROOT)) for n in
        ('RESULTS.md','summary.json','source_records.json')]
    subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--only','-m','W14-B1/B2/B4: archive completed technical acquisition','--',*files],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','master'],cwd=ROOT,check=True)
    run.json('status.json',dict(status='complete',published=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
