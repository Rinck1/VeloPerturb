"""Acquire the mouse GRCm38/Ensembl93 reference used by the Traxler authors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from w4_artifacts import Artifacts
from w14b_data_download import acquire


def run(output):
    root=Path('/data/yuchang/veloroute_w4/traxler2025/reference_ensembl93')
    root.mkdir(parents=True,exist_ok=True)
    urls=[
        'https://ftp.ensembl.org/pub/release-93/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz',
        'https://ftp.ensembl.org/pub/release-93/gtf/mus_musculus/Mus_musculus.GRCm38.93.gtf.gz']
    run=Artifacts(output,seed=0,inputs=[Path(__file__),Path(__file__).with_name('w14b_data_download.py')])
    records=[]
    for url in urls:
        records.append(acquire(url,root/Path(url).name,2*1024**3))
        run.json('status.json',dict(status='downloading_reference',completed=len(records)))
    result=dict(status='MOUSE_REFERENCE_DOWNLOADED',assembly='GRCm38/mm10',annotation='Ensembl93',
        files=records,index_built=False,reason='Wait for read-layout audit before choosing index rlen and chemistry')
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    (run.directory/'RESULTS.md').write_text('# Traxler mouse reference\n\n'+json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
