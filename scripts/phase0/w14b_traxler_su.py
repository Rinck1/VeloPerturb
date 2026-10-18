"""Technical-only, bounded SRA -> joint-library USA counts; never fit W15 models."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from veloroute.preprocess import collapse_gene_versions,load_usa
from w4_artifacts import Artifacts,sha256
from w7_router_scaling import write_csv


ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/data/yuchang/veloroute_w4/traxler2025')


def execute(run,name,argv,timeout=86400):
    argv=[str(v) for v in argv]
    run.json(name+'_command.json',dict(command=argv))
    env=os.environ.copy();env['ALEVIN_FRY_HOME']=str(ROOT/'tools/af-home')
    env['NCBI_SETTINGS']=str(ROOT/'tools/ncbi-config/user-settings.mkfg')
    with (run.directory/(name+'.log')).open('x') as stream:
        subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,check=True,timeout=timeout)


def layout(sra):
    command=[str(ROOT/'tools/sratoolkit.3.4.1-ubuntu64/bin/vdb-dump'),str(sra),'-T','SEQUENCE',
        '-R','1-10','-C','READ_START,READ_LEN','-f','json']
    rows=json.loads(subprocess.check_output(command,text=True,timeout=180))
    if any(r['READ_LEN']!=[26,92,10,10] or r['READ_START'][:2]!=[0,26] for r in rows):
        raise ValueError('Unexpected per-run read layout; retain data and inspect before extraction')
    return rows


def barcode_audit(sra,published):
    with h5py.File(published,'r') as h:valid={b.decode().split('-')[0] for b in h['matrix/barcodes'][()]}
    command=[str(ROOT/'tools/sratoolkit.3.4.1-ubuntu64/bin/vdb-dump'),str(sra),'-T','SEQUENCE',
        '-R','1-10000','-C','READ','-f','json']
    rows=json.loads(subprocess.check_output(command,text=True,timeout=180))
    rates={offset:sum(r['READ'][offset:offset+16] in valid for r in rows)/len(rows) for offset in (0,1,2)}
    if rates[0]<.5 or rates[0]<20*max(rates[1],rates[2],1/len(rows)):
        raise ValueError('Declared barcode cycle layout not supported by published raw whitelist')
    return dict(rows=len(rows),exact_whitelist_fraction_by_offset=rates,approved_offset=0,barcode_bases=16,UMI_bases=10)


def extract(run,obj):
    acc=obj['accession'];sra=DATA/'sra'/(acc+'.sra');fastq=DATA/'fastq'/acc;scratch=DATA/'scratch'/acc
    audit=layout(sra);run.json('layout_'+acc+'.json',dict(run=acc,rows=audit))
    outputs=[fastq/(acc+f'_{i}.fastq.gz') for i in (1,2,3,4)]
    if all(p.exists() for p in outputs):return outputs[:2]
    if fastq.exists() and any(fastq.iterdir()):raise ValueError('Partial extraction retained; use an explicit recovery run')
    if shutil.disk_usage(DATA).free<160*1024**3:raise OSError('Less than160GiB free for one bounded extraction')
    fastq.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    execute(run,'extract_'+acc,[ROOT/'tools/sratoolkit.3.4.1-ubuntu64/bin/fasterq-dump',sra,
        '--split-files','--include-technical','--threads','8','--outdir',fastq,'--temp',scratch])
    for i in (1,2,3,4):
        # Compression replaces only derived plaintext FASTQs. Original SRA is retained.
        execute(run,f'compress_{acc}_{i}',[ROOT/'tools/rna-quant/bin/pigz','-p','8',fastq/(acc+f'_{i}.fastq')])
    execute(run,'test_gzip_'+acc,[ROOT/'tools/rna-quant/bin/pigz','-t',*outputs])
    return outputs[:2]


def import_counts(run,library,quant,published,processed):
    cells,genes,layers,meta=load_usa(quant/'af_quant');genes,layers=collapse_gene_versions(genes,layers)
    s,u,amb=layers
    with h5py.File(published,'r') as h:
        m=h['matrix'];shape=tuple(m['shape'][()]);author_bcs=[b.decode().split('-')[0] for b in m['barcodes'][()]]
        ids=[b.decode() for b in m['features/id'][()]];types=np.array([b.decode() for b in m['features/feature_type'][()]])
        matrix=sparse.csc_matrix((m['data'][()],m['indices'][()],m['indptr'][()]),shape=shape)
    barcode_map={bc:i for i,bc in enumerate(author_bcs)};canonical=[bc.split('-')[0] for bc in cells]
    if any(bc not in barcode_map for bc in canonical):raise ValueError('Quant barcode outside author permit list')
    col=np.array([barcode_map[bc] for bc in canonical])
    guide_idx=np.flatnonzero(types=='CRISPR Guide Capture');hto_idx=np.flatnonzero(types=='Custom')
    guide=matrix[guide_idx][:,col].toarray();nonzero=(guide>0).sum(0);top=guide.argmax(0)
    guide_ids=np.array([ids[i] for i in guide_idx]);calls=np.where(nonzero==1,guide_ids[top],np.where(nonzero==0,'Negative','Multiplet'))
    target=np.array([c.rsplit('_',1)[0] if c not in ('Negative','Multiplet') else c for c in calls])
    obs=pd.DataFrame(dict(barcode=canonical,library=library,time_hours={'A':0,'B':6,'C':24}[library[0]],
        guide_call=calls,provisional_gene_target=target,n_nonzero_guides=nonzero,
        spliced_umi=np.asarray(s.sum(1)).ravel(),unspliced_umi=np.asarray(u.sum(1)).ravel(),
        ambiguous_umi=np.asarray(amb.sum(1)).ravel(),biological_replicate='UNRESOLVED_USE_HTO'),
        index=[library+':'+bc for bc in canonical])
    a=ad.AnnData(X=s+amb,obs=obs,var=pd.DataFrame(index=genes),layers=dict(spliced=s,unspliced=u,ambiguous=amb))
    a.obsm['published_hto_counts']=matrix[hto_idx][:,col].T.tocsr()
    a.uns['published_hto_ids']=[ids[i] for i in hto_idx]
    a.uns['semantics']='X=S+A; strict S/U/A layers; guide calls provisional, not validated editing or HTO demultiplexing'
    a.write_h5ad(processed)
    author_rna=np.asarray(matrix[types=='Gene Expression'][:,col].sum(0)).ravel()
    ours=np.asarray(s.sum(1)).ravel()
    correlation=float(np.corrcoef(author_rna,ours)[0,1]) if ours.std()>0 and author_rna.std()>0 else None
    record=dict(library=library,time_hours={'A':0,'B':6,'C':24}[library[0]],cells=len(cells),genes=len(genes),
        barcode_retained_fraction=len(cells)/len(author_bcs),author_total_vs_spliced_total_correlation=correlation,
        median_S=float(np.median(ours)),median_U=float(np.median(np.asarray(u.sum(1)).ravel())),
        path=str(processed),sha256=sha256(processed),guide_calls_provisional=True,HTO_demultiplexing_complete=False)
    counts=[dict(library=library,time_hours=record['time_hours'],target=str(g),cells=int(np.sum(target==g))) for g in sorted(set(target))]
    return record,counts


def run(output):
    run=Artifacts(output,seed=0,inputs=[Path(__file__),ROOT/'PLANS/w14b_perturbation_track.md'])
    deadline=time.monotonic()+7*86400
    for key in ('traxler_reference','traxler_layout'):
        while time.monotonic()<deadline:
            p=ROOT/'outputs/w14_jobs_20261018'/key/'status.json'
            state=json.loads(p.read_text()) if p.exists() else {}
            if state.get('status')=='failed':raise RuntimeError('Reference/layout acquisition failed')
            if state.get('status')=='complete':break
            run.json('status.json',dict(status='waiting_for_'+key));time.sleep(20)
        else:raise TimeoutError('Technical dependency timeout')
    manifest=json.loads((ROOT/'outputs/w14b_traxler_raw_v1/download_manifest.json').read_text())
    grouped=defaultdict(list)
    for obj in manifest['objects']:
        library=obj['title'].removeprefix('CROPseq_KO150_').removesuffix('_GEX');grouped[library].append(obj)
    if len(grouped)!=9 or any(len(v)!=2 for v in grouped.values()):raise ValueError('Nine joint-two-run GEX libraries required')
    first=json.loads((ROOT/'outputs/w14b_traxler_layout_v1/summary.json').read_text())['run']
    obj=next(o for o in manifest['objects'] if o['accession']==first)
    lib=obj['title'].removeprefix('CROPseq_KO150_').removesuffix('_GEX')
    raw_h5=DATA/f'GSE263761_CROPseq_KO150_{lib}_raw_feature_bc_matrix.h5'
    run.json('barcode_audit.json',barcode_audit(DATA/'sra'/(first+'.sra'),raw_h5))
    ref=DATA/'reference_ensembl93';index=ref/'splici_r92_v1'
    if index.exists():raise ValueError('Existing reference index retained; explicit audited reuse required')
    execute(run,'build_index',[ROOT/'tools/rna-quant/bin/simpleaf','index','--fasta',ref/'Mus_musculus.GRCm38.dna.primary_assembly.fa.gz',
        '--gtf',ref/'Mus_musculus.GRCm38.93.gtf.gz','--rlen','92','--ref-type','spliced+intronic','--threads','8',
        '--output',index,'--work-dir',index/'work','--ram-limit-gib','8'])
    processed_root=DATA/'processed_usa_v1';processed_root.mkdir(exist_ok=False)
    complete=[];qc=[]
    while len(complete)<9 and time.monotonic()<deadline:
        available=[lib for lib,objects in grouped.items() if lib not in complete and all(
            (DATA/'sra'/(o['accession']+'.sra')).is_file() and (DATA/'sra'/(o['accession']+'.sra')).stat().st_size==o['bytes'] for o in objects)]
        if not available:
            raw_state=json.loads((ROOT/'outputs/w14_jobs_20261018/traxler_raw/status.json').read_text())
            if raw_state.get('status')=='failed':raise RuntimeError('Raw acquisition failed')
            run.json('status.json',dict(status='waiting_for_joint_library_runs',completed=complete));time.sleep(20);continue
        lib=sorted(available)[0];objects=grouped[lib];reads=[extract(run,o) for o in objects]
        published=DATA/f'GSE263761_CROPseq_KO150_{lib}_filtered_feature_bc_matrix.h5'
        whitelist=run.directory/(lib+'_barcodes.txt')
        with h5py.File(published,'r') as h:
            whitelist.write_text('\n'.join(b.decode().split('-')[0] for b in h['matrix/barcodes'][()])+'\n')
        quant=DATA/'quant_usa_v1'/lib
        execute(run,'quant_'+lib,[ROOT/'tools/rna-quant/bin/simpleaf','quant','--index',index/'index',
            '--chemistry','1{b[16]u[10]x:}2{r:}','--expected-ori','rc',
            '--reads1',','.join(str(r[0]) for r in reads),'--reads2',','.join(str(r[1]) for r in reads),
            '--explicit-pl',whitelist,'--resolution','cr-like','--threads','8','--output',quant])
        record,counts=import_counts(run,lib,quant,published,processed_root/(lib+'.h5ad'))
        qc.append(record);write_csv(run.directory/(lib+'_technical_target_counts.csv'),counts)
        complete.append(lib);run.json('status.json',dict(status='quantifying',completed=complete,technical_qc=qc))
    if len(complete)!=9:raise TimeoutError('Traxler quantification timeout')
    result=dict(status='TRAXLER_S_U_A_COUNTS_COMPLETE',libraries=qc,guide_calls_provisional=True,
        HTO_demultiplexing_complete=False,new_dataset_biological_effects_read=False,prediction_models_trained=False,
        next='Complete author HTO/RNA QC and freeze W15 before any perturbation-response scoring')
    run.json('summary.json',result);write_csv(run.directory/'technical_qc.csv',qc)
    (run.directory/'RESULTS.md').write_text('# Traxler USA counts\n\n'+json.dumps(result,indent=2)+'\n')
    doc=ROOT/'docs/current/w14b_su_results.md'
    if doc.exists():raise ValueError('Prior USA report retained')
    doc.write_text((run.directory/'RESULTS.md').read_text())
    files=[str(doc.relative_to(ROOT))]+[str((run.directory/n).relative_to(ROOT)) for n in ('summary.json','RESULTS.md','technical_qc.csv')]
    subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--only','-m','W14-B1: record technical S/U/A quantification','--',*files],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','master'],cwd=ROOT,check=True)
    run.json('status.json',dict(status='complete'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
