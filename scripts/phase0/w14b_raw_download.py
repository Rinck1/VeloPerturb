"""User-approved, resumable NCBI SRA acquisition retaining technical reads."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time
import urllib.request

from w4_artifacts import Artifacts,sha256
from w14b_data_download import acquire


def run(metadata,inventory,output,data_root,max_bytes):
    data_root=Path(data_root).resolve()
    if data_root!=Path('/data/yuchang/veloroute_w4/traxler2025/sra'):
        raise ValueError('Use the explicit Traxler SRA acquisition directory')
    samples=json.loads(Path(metadata).read_text())['samples'];by_sample={s['accession']:s for s in samples}
    rows=json.loads(Path(inventory).read_text())['runs']
    selected=[r for r in rows if by_sample[r['library_name']]['title'][0].endswith('_GEX')]
    if len(selected)!=18:raise ValueError('Unexpected GEX run count; inspect library metadata')
    run=Artifacts(output,seed=0,inputs=[Path(metadata),Path(inventory),Path(__file__),Path(__file__).with_name('w14b_data_download.py')])
    def head(row):
        acc=row['run_accession'];url=f'https://sra-pub-run-odp.s3.amazonaws.com/sra/{acc}/{acc}'
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as r:
            size=int(r.headers['Content-Length']);etag=r.headers.get('ETag')
        return dict(accession=acc,sample=row['library_name'],title=by_sample[row['library_name']]['title'][0],
            url=url,bytes=size,etag=etag)
    with ThreadPoolExecutor(max_workers=4) as pool:objects=list(pool.map(head,selected))
    total=sum(o['bytes'] for o in objects)
    if total>max_bytes:raise ValueError(f'SRA total {total} exceeds user-approved {max_bytes}; request new approval')
    data_root.mkdir(parents=True,exist_ok=True)
    existing=sum((data_root/(o['accession']+'.sra')).stat().st_size for o in objects if (data_root/(o['accession']+'.sra')).exists())
    if shutil.disk_usage(data_root).free<total-existing+10*1024**3:raise OSError('Insufficient disk for all SRA objects')
    run.json('download_manifest.json',dict(objects=objects,total_bytes=total,user_approved_max_bytes=max_bytes,
        source='NCBI_public_SRA_AWS',technical_reads_retained=True,ena_fastq_not_used=True))
    def fetch(obj):
        path=data_root/(obj['accession']+'.sra')
        for attempt in range(4):
            try:
                result=acquire(obj['url'],path,obj['bytes'])
                if result['bytes']!=obj['bytes']:raise ValueError('SRA byte count differs from pinned HEAD')
                run.json(obj['accession']+'.json',dict(status='complete',record=result,attempt=attempt+1))
                return result
            except Exception:
                if attempt==3:raise
                time.sleep(5*(attempt+1))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(fetch,o) for o in objects]
        while not all(f.done() for f in futures):
            if any(f.done() and f.exception() for f in futures):
                for f in futures:
                    if f.done() and f.exception():raise f.exception()
            completed=sum(f.done() for f in futures)
            downloaded=sum(p.stat().st_size for p in data_root.iterdir() if p.name.endswith(('.sra','.partial')))
            run.json('status.json',dict(status='downloading',completed_runs=completed,total_runs=len(objects),
                downloaded_bytes=downloaded,total_bytes=total))
            time.sleep(15)
        results=[f.result() for f in futures]
    result=dict(status='TRAXLER_GEX_SRA_DOWNLOADED',runs=len(results),total_bytes=total,objects=results,
        technical_reads_retained=True,biological_effects_read=False,
        next='Validate SRA read layout, then extract using fasterq-dump --include-technical; no W15 model before protocol')
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    (run.directory/'RESULTS.md').write_text('# Traxler raw GEX acquisition\n\n'+json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],runs=len(results),total_bytes=total)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata',default='outputs/w14b_traxler_acquisition_v1/sample_metadata.json')
    p.add_argument('--inventory',default='outputs/w14b_traxler_acquisition_v1/run_inventory.json')
    p.add_argument('--output',required=True);p.add_argument('--data-root',default='/data/yuchang/veloroute_w4/traxler2025/sra')
    p.add_argument('--max-bytes',type=int,default=180000000000);a=p.parse_args()
    run(a.metadata,a.inventory,a.output,a.data_root,a.max_bytes)
