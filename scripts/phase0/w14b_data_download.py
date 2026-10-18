"""Bounded technical-only acquisition of Traxler KO150 metadata/count files."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import time
import urllib.parse
import urllib.request

from w4_artifacts import Artifacts, sha256


ROOT = Path(__file__).resolve().parents[2]
BASE = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE263nnn/GSE263761/'


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'VeloRoute-technical-audit/1.0'}),timeout=90)


def acquire(url, destination, maximum):
    """Resume a known partial only when the server honors its byte range."""
    destination = Path(destination)
    if destination.exists():
        return dict(path=str(destination),bytes=destination.stat().st_size,sha256=sha256(destination),url=url,reused=True)
    partial = destination.with_suffix(destination.suffix+'.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {'User-Agent':'VeloRoute-technical-audit/1.0'}
    if offset:headers['Range']=f'bytes={offset}-'
    response = urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120)
    if offset and response.status != 206:
        raise ValueError('Server ignored range; retained partial without overwriting')
    length = int(response.headers.get('Content-Length',0))
    if length+offset > maximum:raise ValueError('Declared file exceeds acquisition limit')
    size = offset
    with response,partial.open('ab' if offset else 'xb') as stream:
        while block:=response.read(2**20):
            size += len(block)
            if size > maximum:raise ValueError('Stream exceeds acquisition limit')
            stream.write(block)
    if length and size-offset != length:raise IOError('Incomplete HTTP body retained as partial')
    partial.replace(destination)
    return dict(path=str(destination),bytes=size,sha256=sha256(destination),url=url,reused=False)


def parse_soft(text):
    samples=[];current=None
    for line in text.splitlines():
        if line.startswith('^SAMPLE = '):
            current=dict(accession=line.split(' = ',1)[1],characteristics=[],supplementary=[],relations=[])
            samples.append(current)
        elif current is not None and line.startswith('!Sample_') and ' = ' in line:
            key,value=line.split(' = ',1);key=key.removeprefix('!Sample_')
            if key.startswith('characteristics'):current['characteristics'].append(value)
            elif key.startswith('supplementary_file'):current['supplementary'].append(value)
            elif key=='relation':current['relations'].append(value)
            elif key in ('title','source_name_ch1','library_strategy','library_source','library_selection','data_processing'):
                current.setdefault(key,[]);current[key].append(value)
    return samples


def run(output,data_root,download_counts):
    data_root=Path(data_root).resolve()
    if data_root != Path('/data/yuchang/veloroute_w4/traxler2025'):
        raise ValueError('Use the explicit project acquisition directory')
    data_root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(data_root).free < 5*1024**3:raise OSError('Insufficient acquisition disk space')
    run=Artifacts(output,seed=0,inputs=[Path(__file__),ROOT/'PLANS/w14b_perturbation_track.md'])
    records=[]
    soft=acquire(BASE+'soft/GSE263761_family.soft.gz',data_root/'GSE263761_family.soft.gz',64*1024**2)
    records.append(soft)
    soft_text=gzip.decompress(Path(soft['path']).read_bytes()).decode()
    samples=parse_soft(soft_text)
    run.json('sample_metadata.json',dict(samples=samples,expression_read=False))
    fields='run_accession,experiment_accession,sample_accession,study_accession,library_name,library_strategy,library_source,library_layout,fastq_ftp,fastq_bytes,fastq_md5,submitted_ftp,submitted_bytes,submitted_md5'
    url='https://www.ebi.ac.uk/ena/portal/api/filereport?'+urllib.parse.urlencode(dict(
        accession='PRJNA1099196',result='read_run',fields=fields,format='tsv',download='false'))
    ena=acquire(url,data_root/'PRJNA1099196_runs.tsv',8*1024**2);records.append(ena)
    rows=list(csv.DictReader(Path(ena['path']).open(),delimiter='\t'))
    if not rows or 'run_accession' not in rows[0]:raise ValueError('Invalid ENA run table')
    def total(key):return sum(int(v) for r in rows for v in r.get(key,'').split(';') if v)
    total_fastq=total('fastq_bytes');total_submitted=total('submitted_bytes')
    by_sample={s['accession']:s for s in samples}
    gex=[r for r in rows if by_sample[r['library_name']]['title'][0].endswith('_GEX')]
    gex_bytes=sum(int(v) for r in gex for v in r['fastq_bytes'].split(';') if v)
    run.json('run_inventory.json',dict(correct_bioproject='PRJNA1099196',original_plan_bioproject='PRJNA1099184',
        runs=rows,total_all_library_fastq_bytes=total_fastq,total_submitted_bytes=total_submitted,
        gex_runs=len(gex),gex_fastq_bytes=gex_bytes,
        raw_download_authorized_below_bytes=50*10**9,raw_download_started=False,
        reason='Classify GEX/HTO/GDO and validate read layout before selecting raw runs'))
    run.json('status.json',dict(status='metadata_complete_downloading_processed_counts',samples=len(samples),runs=len(rows)))
    technical=[]
    if download_counts:
        # This series publishes individual H5s, not a GSE*_RAW.tar archive.
        urls=re.findall(r'^!Series_supplementary_file = (.+)$',soft_text,flags=re.M)
        total_counts=0
        for url in urls:
            url=url.replace('ftp://ftp.ncbi.nlm.nih.gov/','https://ftp.ncbi.nlm.nih.gov/')
            name=Path(urllib.parse.urlparse(url).path).name
            record=acquire(url,data_root/name,3*1024**3-total_counts)
            total_counts+=record['bytes'];records.append(record)
            if name.endswith('.h5'):
                import h5py
                with h5py.File(record['path'],'r') as h:
                    technical.append(dict(file=name,matrix_shape=h['matrix/shape'][()].tolist(),
                        feature_types=sorted(set(v.decode() for v in h['matrix/features/feature_type'][()]))))
            run.json('download_progress.json',dict(completed=len(records),bytes=sum(r['bytes'] for r in records),latest=name))
        run.json('matrix_technical_headers.json',dict(headers=technical,expression_values_read=False))
    result=dict(status='TRAXLER_METADATA_AND_COUNTS_ACQUIRED' if download_counts else 'TRAXLER_METADATA_ACQUIRED',
        correct_bioproject='PRJNA1099196',samples=len(samples),runs=len(rows),downloads=records,
        total_all_library_fastq_bytes=total_fastq,gex_runs=len(gex),gex_fastq_bytes=gex_bytes,raw_download_started=False,
        biological_effects_read=False,prediction_models_trained=False,
        next='Map libraries and hashing labels; inspect count-file headers only; freeze W15 before biological analysis')
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    (run.directory/'RESULTS.md').write_text('# Traxler technical acquisition\n\n'+json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--data-root',default='/data/yuchang/veloroute_w4/traxler2025')
    p.add_argument('--download-counts',action='store_true');a=p.parse_args()
    run(a.output,a.data_root,a.download_counts)
