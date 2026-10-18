"""Count time x perturbation cells/embryos from author metadata, not expression."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import time

from w4_artifacts import Artifacts
from w7_router_scaling import write_csv
from w14b_data_download import acquire


def run(output):
    data=Path('/data/yuchang/veloroute_w4/zscape_metadata');data.mkdir(parents=True,exist_ok=True)
    run=Artifacts(output,seed=0,inputs=[Path(__file__)])
    url='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE202nnn/GSE202639/suppl/GSE202639_zperturb_full_cell_metadata.csv.gz'
    for attempt in range(4):
        try:
            record=acquire(url,data/Path(url).name,512*1024**2);break
        except OSError:
            if attempt==3:raise
            time.sleep(5*(attempt+1))
    counts=Counter();embryos=defaultdict(set)
    with gzip.open(record['path'],'rt') as stream:
        rows=csv.DictReader(stream)
        if not {'timepoint','gene_target','embryo'}<=set(rows.fieldnames):raise ValueError('Author metadata columns changed')
        for row in rows:
            key=(row['timepoint'],row['gene_target']);counts[key]+=1;embryos[key].add(row['embryo'])
    result_rows=[dict(timepoint=t,gene_target=g,cells=n,embryos=len(embryos[(t,g)])) for (t,g),n in sorted(counts.items())]
    write_csv(run.directory/'time_by_target_cells.csv',result_rows)
    result=dict(status='ZSCAPE_TECHNICAL_COHORT_COUNTS_COMPLETE',metadata=record,total_cells=sum(counts.values()),
        groups=len(counts),expression_matrices_downloaded=False,raw_reads_downloaded=False,
        biological_effects_computed=False,columns_used=['timepoint','gene_target','embryo'])
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    (run.directory/'RESULTS.md').write_text('# ZSCAPE metadata-only cohort counts\n\n'+json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
