"""Audit BAM->USA chemistry on a bounded prefix; never publish it as full counts."""
import json
from pathlib import Path
import sys
import itertools

from veloroute.artifacts import Run,load_config,save_json
from veloroute.kang_data import published_metadata,tagged_fastq
from veloroute.preprocess import load_usa
from veloroute.raw import run_command

c=load_config('configs/veloroute_kang_20260914.yaml');root=Path(c['root'])
sys.path.insert(0,str(root/'python_deps'))
import pysam
prefix=root/'raw/ctrl/probe/prefix.partial.bam'
meta=published_metadata(c['published_root'])
permit=set(b.split('-')[0] for b in meta[(meta.condition=='ctrl')&(meta.multiplets=='singlet')].barcode)
with Run(root/'quant_smoke_prefix_v1',stage='Kang_prefix_USA_chemistry',kind='engineering',
    config=dict(prefix_records=10000,orientation=c['preprocessing']['expected_orientation'],full_counts=False),inputs=[prefix]) as run:
    retained=0
    with (run.directory/'permit.txt').open('x') as f:f.write('\n'.join(sorted(permit))+'\n')
    with pysam.AlignmentFile(str(prefix),'rb',ignore_truncation=True) as bam, \
         (run.directory/'barcode.fastq').open('x') as r1,(run.directory/'cdna.fastq').open('x') as r2:
        for read in itertools.islice(bam,10000):
            values,_=tagged_fastq(read,permit)
            if values: r1.write(values[0]);r2.write(values[1]);retained+=1
    tool=str(Path('tools/rna-quant/bin/simpleaf').resolve())
    index=root/'reference/splici_r98_v2/index';quant=run.directory/'quant'
    run_command([tool,'quant','--index',str(index),'--chemistry','1{b[14]u[10]}2{r:}',
        '--expected-ori',c['preprocessing']['expected_orientation'],'--reads1',str(run.directory/'barcode.fastq'),
        '--reads2',str(run.directory/'cdna.fastq'),'--explicit-pl',str(run.directory/'permit.txt'),
        '--resolution','cr-like','--threads','4','--output',str(quant)],run.directory/'command',
        timeout_seconds=1800,environment=dict(ALEVIN_FRY_HOME=str(Path('tools/af-home').resolve()),TMPDIR=str(root/'tmp')),
        working_directory=str(root))
    barcodes,genes,matrices,_=load_usa(quant/'af_quant')
    result=dict(status='KANG_PREFIX_USA_SMOKE_PASSED',scanned=10000,retained=retained,
                barcodes=len(barcodes),genes=len(genes),USA_totals=[float(m.sum()) for m in matrices],
                whole_library=False,representative_QC=False,real_velocity_increment=False)
    save_json(run.directory/'summary.json',result)
    (run.directory/'RESULTS.md').write_text('# Bounded Kang chemistry smoke\n\n'+json.dumps(result,indent=2)
        +'\n\nCoordinate-sorted prefix; not representative of the whole library and not eligible for biological model training.\n')
    print(json.dumps(result),flush=True)
