"""Frozen mouse ESM2-3B KO embeddings; no single-cell outcome input."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from w4_artifacts import Artifacts, sha256


def select_mouse_proteins(fasta,targets):
    selected={}; header=None; sequence=[]
    def accept():
        if header is None:return
        fields=header.split()
        annotation={s.split(':',1)[0]:s.split(':',1)[1] for s in fields if ':' in s}
        symbol=annotation.get('gene_symbol')
        if symbol not in targets:return
        protein=''.join(sequence).rstrip('*')
        if not protein or set(protein)-set('ACDEFGHIKLMNPQRSTVWYBXZUO'):
            raise ValueError(f'Unexpected reference protein for {symbol}')
        record=dict(condition=symbol,protein_id=fields[0],gene_id=annotation['gene'],
            transcript_id=annotation['transcript'],length=len(protein),sequence=protein,
            sequence_sha256=hashlib.sha256(protein.encode()).hexdigest())
        previous=selected.get(symbol)
        if previous is None or (-record['length'],record['protein_id'])<(-previous['length'],previous['protein_id']):
            selected[symbol]=record
    with gzip.open(fasta,'rt') as stream:
        for line in stream:
            line=line.strip()
            if line.startswith('>'):
                accept();header=line[1:];sequence=[]
            else:sequence.append(line)
        accept()
    missing=set(targets)-set(selected)
    if missing:raise ValueError(f'Missing exact mouse protein symbols: {sorted(missing)}')
    return [selected[symbol] for symbol in sorted(targets)]


def build(fasta,checksums,weights,splits,output):
    import torch
    import esm
    from veloroute.gpu_policy import enforce_gpu_policy
    enforce_gpu_policy('cuda')
    expected=next(line.split()[:2] for line in Path(checksums).read_text().splitlines()
                  if line.split()[-1]==Path(fasta).name)
    actual=subprocess.check_output(['sum',str(fasta)],text=True).split()[:2]
    if expected!=actual:raise ValueError('Official Ensembl BSD checksum mismatch')
    targets=set()
    for split in splits:
        spec=json.loads(Path(split).read_text())
        targets.update(spec['train']+spec['val']+spec['test'])
    records=select_mouse_proteins(fasta,targets)
    if Path(weights).stat().st_size!=5678116398:raise ValueError('Expected complete ESM2-3B checkpoint')
    run=Artifacts(output,seed=0,inputs=[Path(__file__),Path(fasta),Path(checksums),Path(weights)]+list(map(Path,splits)))
    try:
        run.json('status.json',dict(status='loading_model',proteins=len(records)))
        torch.set_num_threads(4)
        with torch.serialization.safe_globals([argparse.Namespace]):
            data=torch.load(weights,map_location='cpu',weights_only=True)
        model,alphabet=esm.pretrained.load_model_and_alphabet_core('esm2_t36_3B_UR50D',data)
        del data
        model.eval().requires_grad_(False)
        model=model.half().to('cuda')
        converter=alphabet.get_batch_converter(); vectors=[]
        with torch.inference_mode():
            for record in records:
                sequence=record['sequence']; total=np.zeros(2560,dtype='float64')
                for begin in range(0,len(sequence),1022):
                    part=sequence[begin:begin+1022]
                    _,_,tokens=converter([(record['condition'],part)])
                    result=model(tokens.to('cuda'),repr_layers=[36],return_contacts=False)
                    total+=result['representations'][36][0,1:len(part)+1].float().sum(0).cpu().numpy()
                vector=(total/len(sequence)).astype('float32')
                if vector.shape!=(2560,) or not np.isfinite(vector).all():raise ValueError('Invalid ESM vector')
                vectors.append(vector)
                run.json('status.json',dict(status='embedding',completed=len(vectors),total=len(records),last=record['condition']))
                print(json.dumps(dict(completed=len(vectors),total=len(records),condition=record['condition'])),flush=True)
        metadata=dict(kind='public_mouse_protein_prior',frozen=True,model='esm2_t36_3B_UR50D',layer=36,
            dimension=2560,organism='Mus musculus',reference='Ensembl_release93_GRCm38',
            reference_url='https://ftp.ensembl.org/pub/release-93/fasta/mus_musculus/pep/Mus_musculus.GRCm38.pep.all.fa.gz',
            reference_sha256=sha256(fasta),weights_sha256=run.context['data_hashes'][str(Path(weights))],
            isoform_rule='longest_protein_tie_break_protein_id',pooling='mean_residues_excluding_special_tokens',
            long_sequence_rule='nonoverlap_1022_residue_windows_length_weighted_mean',
            no_cell_expression_used=True,heldout_outcomes_read=False)
        path=run.directory/'conditions.npz'
        np.savez_compressed(path,conditions=np.array([r['condition'] for r in records]),
            embeddings=np.stack(vectors),metadata_json=np.array(json.dumps(metadata,sort_keys=True)))
        from w7_router_scaling import write_csv
        write_csv(run.directory/'protein_manifest.csv',records)
        run.json('summary.json',dict(status='TRAXLER_FROZEN_MOUSE_ESM_CONDITIONS_COMPLETE',
            conditions=len(records),conditions_path=str(path.resolve()),conditions_sha256=sha256(path),**metadata))
        run.json('status.json',dict(status='complete',proteins=len(records)))
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arg in ('fasta','checksums','weights','output'):parser.add_argument('--'+arg,required=True)
    parser.add_argument('--splits',nargs='+',required=True)
    args=parser.parse_args();build(args.fasta,args.checksums,args.weights,args.splits,args.output)
