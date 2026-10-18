"""Metadata-only MeRLin sampling and ZSCAPE raw-download inventory."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import re
import urllib.parse

import openpyxl

from w4_artifacts import Artifacts,sha256
from w7_router_scaling import write_csv
from w14b_data_download import acquire


def run(output):
    workbook=Path('/data/yuchang/veloroute_ucheck_20260915/merlin_suppl.xlsx')
    run=Artifacts(output,seed=0,inputs=[Path(__file__),workbook])
    wb=openpyxl.load_workbook(workbook,read_only=True,data_only=True)
    rows=[]
    for row in wb['Supplementary Table 4'].iter_rows(min_row=25,max_row=29,values_only=True):
        label,counts,_,clones=row[:4];known,total=[int(v) for v in re.findall(r'\d+',counts)]
        rows.append(dict(sample=label,after_filter_cells=total,barcode_known_cells=known,unique_barcodes=int(clones),
            source='original_local_supplement_Table4_not_new_cell_QC'))
    write_csv(run.directory/'merlin_published_counts.csv',rows)
    sampling=dict(in_vivo_scRNA_days=[0,21,91],in_vivo_scRNA_intervals_hours=[504,1680],
        endpoint_day_91_source='2026_published_methods',bulk_days=[0,21,57,91],
        in_vitro_treatment_days=7,in_vitro_hours=168,
        hour_scale_intervals_found=False,day57_not_available_as_matched_scRNA_timepoint=True,
        original_table_sha256=sha256(workbook),counts=rows,
        sources=['https://link.springer.com/article/10.1186/s12943-026-02622-9'],
        decision='TIME_SCALE_NOT_SUITABLE_FOR_CURRENT_HOUR_FORECAST_NO_NEW_MERLIN_TRAINING',
        scope='Published main MeRLin experiment; no claim that every conceivable MeRLin-derived study lacks shorter times')
    run.json('merlin_sampling.json',sampling)
    data=Path('/data/yuchang/veloroute_w4/zscape_metadata');data.mkdir(parents=True,exist_ok=True)
    soft=acquire('https://ftp.ncbi.nlm.nih.gov/geo/series/GSE202nnn/GSE202639/soft/GSE202639_family.soft.gz',
        data/'GSE202639_family.soft.gz',8*1024**2)
    text=gzip.decompress(Path(soft['path']).read_bytes()).decode()
    supplements=re.findall(r'^!Series_supplementary_file = (.+)$',text,re.M)
    fields='run_accession,experiment_accession,sample_accession,library_strategy,library_layout,fastq_ftp,fastq_bytes'
    url='https://www.ebi.ac.uk/ena/portal/api/filereport?'+urllib.parse.urlencode(dict(accession='PRJNA836866',
        result='read_run',fields=fields,format='tsv',download='false'))
    record=acquire(url,data/'PRJNA836866_runs.tsv',8*1024**2)
    runs=list(csv.DictReader(Path(record['path']).open(),delimiter='\t'))
    total=sum(int(v) for r in runs for v in r['fastq_bytes'].split(';') if v)
    zscape=dict(GEO='GSE202639',BioProject='PRJNA836866',runs=runs,fastq_bytes_all_runs=total,
        supplementary=supplements,downloads=[soft,record],raw_reads_downloaded=False,expression_matrix_read=False,
        published_perturbation_hpf=[18,24,36,48,72],minimum_nominal_interval_hours=6,
        S_U_layers_not_established=True,
        source='https://www.nature.com/articles/s41586-023-06720-2',
        technical_caution='sci-RNA-seq3 combinatorial barcodes require author demultiplexing, not a drop-in 10x chemistry')
    run.json('zscape_inventory.json',zscape)
    result=dict(status='W14B_SAMPLING_AND_INVENTORY_COMPLETE',merlin_hour_scale=False,
        zscape_GEO=zscape['GEO'],zscape_fastq_bytes=total,zscape_runs=len(runs),
        zscape_cell_count_table='PENDING_AUTHOR_METADATA_ONLY_NOT_DOWNLOADED',expression_matrices_read=False)
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    (run.directory/'RESULTS.md').write_text('# W14-B metadata audits\n\n'+json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
