"""Frozen public-protein ESM2 conditions; no cell expression input is accepted."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .artifacts import Run, read_csv, save_csv, save_json, sha256


def select_proteins(fasta, targets):
    selected = {}
    header, sequence = None, []

    def accept(header, sequence):
        if header is None:
            return
        fields = header.split('|')
        if len(fields) < 8:
            raise ValueError('Expected GENCODE protein FASTA header with gene/protein IDs')
        symbol, protein = fields[6], ''.join(sequence).rstrip('*')
        if symbol not in targets:
            return
        if not protein or set(protein)-set('ACDEFGHIKLMNPQRSTVWYBXZUO'):
            raise ValueError(f'Unexpected protein sequence for {symbol}')
        record = {'condition': symbol, 'gene_id': fields[2], 'protein_id': fields[0],
                  'transcript_id': fields[1], 'sequence': protein, 'length': len(protein),
                  'sequence_sha256': hashlib.sha256(protein.encode()).hexdigest()}
        previous = selected.get(symbol)
        # Deterministic sequence-only rule, frozen before any perturbation outcome.
        if previous is None or (-len(protein), fields[0]) < (-previous['length'], previous['protein_id']):
            selected[symbol] = record

    opener = gzip.open if str(fasta).endswith('.gz') else open
    with opener(fasta, 'rt') as stream:
        for line in stream:
            line = line.strip()
            if line.startswith('>'):
                accept(header, sequence)
                header, sequence = line[1:], []
            else:
                sequence.append(line)
    accept(header, sequence)
    if set(targets)-set(selected):
        raise ValueError(f'No reference protein for {sorted(set(targets)-set(selected))}; no fabricated fallback')
    return [selected[symbol] for symbol in sorted(targets)]


def build_conditions(fasta, weights, guides, output, *, device='cuda:0', window=1022):
    import torch
    import esm
    if window < 1:
        raise ValueError('Invalid protein window')
    targets = sorted({r['target_from_published_name'] for r in read_csv(guides) if r['control_class'] == 'TF'})
    records = select_proteins(fasta, targets)
    if Path(weights).stat().st_size != 5678116398:
        raise ValueError('Official ESM2-3B checkpoint length differs; verify download/version before loading')
    checksum_lines = (Path(fasta).parent/'MD5SUMS').read_text().splitlines()
    expected = {line.split()[-1].lstrip('*'): line.split()[0] for line in checksum_lines}
    md5 = hashlib.md5()
    with Path(fasta).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            md5.update(block)
    if md5.hexdigest() != expected.get(Path(fasta).name):
        raise ValueError('GENCODE protein FASTA MD5 mismatch')
    config = {'model': 'esm2_t36_3B_UR50D', 'layer': 36, 'device': device,
              'isoform_rule': 'longest_GENCODE_v32_protein_tie_break_protein_id',
              'pooling': 'mean_residues_excluding_special_tokens', 'window': window,
              'long_sequence_rule': 'nonoverlap_windows_length_weighted_mean', 'uses_expression': False}
    with Run(output, stage='frozen_protein_conditions', kind='engineering', config=config, inputs=[fasta, weights, guides]) as run:
        torch.set_num_threads(4)
        # Official checkpoint contains configuration namespaces. Allow only that
        # known data type, never arbitrary pickle execution.
        with torch.serialization.safe_globals([argparse.Namespace]):
            model_data = torch.load(weights, map_location='cpu', weights_only=True)
        model, alphabet = esm.pretrained.load_model_and_alphabet_core('esm2_t36_3B_UR50D', model_data)
        del model_data
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise ValueError('Requested embedding GPU is unavailable')
            free, total = torch.cuda.mem_get_info(torch.device(device))
            if free < 20*(1 << 30):
                raise RuntimeError('Need at least 20 GiB free for bounded ESM2-3B inference')
            model = model.half()
        model = model.to(device)
        converter = alphabet.get_batch_converter()
        vectors = []
        with torch.inference_mode():
            for record in records:
                sequence, sums = record['sequence'], []
                for start in range(0, len(sequence), window):
                    subsequence = sequence[start:start+window]
                    _, _, tokens = converter([(record['condition'], subsequence)])
                    result = model(tokens.to(device), repr_layers=[36], return_contacts=False)
                    sums.append(result['representations'][36][0, 1:len(subsequence)+1].float().sum(0).cpu().numpy())
                vector = np.sum(sums, axis=0)/len(sequence)
                if vector.shape != (2560,) or not np.isfinite(vector).all():
                    raise ValueError('Invalid ESM2-3B embedding output')
                vectors.append(vector)
                save_json(run.directory/'progress.json', {'completed': len(vectors), 'total': len(records), 'last_condition': record['condition']})
        metadata = {'source': 'https://github.com/facebookresearch/esm', 'weights_url': 'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt',
                    'kind': 'public_protein_prior', 'frozen': True, 'config': config, 'weights_sha256': sha256(weights),
                    'fasta_sha256': sha256(fasta), 'no_cell_data_used': True, 'dimension': 2560}
        with (run.directory/'conditions.npz').open('xb') as stream:
            np.savez_compressed(stream, conditions=np.array(targets), embeddings=np.stack(vectors).astype(np.float32),
                                metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
        save_csv(run.directory/'protein_manifest.csv', records)
        save_json(run.directory/'summary.json', {**metadata, 'conditions': len(targets)})
        (run.directory/'RESULTS.md').write_text('# Frozen ESM2-3B conditions\n\n'+json.dumps(metadata, indent=2)
            +'\n\nPublic protein sequences only; held-out condition names do not disclose cell outcomes. '
             'No gene is silently replaced with a random or zero embedding. Contact prediction head is unused.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--guides', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wait-status', nargs='*', default=[])
    parser.add_argument('--timeout', type=int, default=7200)
    args = parser.parse_args()
    deadline = time.monotonic()+args.timeout
    for path in args.wait_status:
        while True:
            record = json.loads(Path(path).read_text()) if Path(path).is_file() else {}
            if record.get('status') == 'complete':
                break
            if record.get('status') in {'failed', 'interrupted_or_timeout'} or time.monotonic() > deadline:
                raise RuntimeError(f'Prerequisite failed/timed out: {path}')
            time.sleep(5)
    build_conditions(args.fasta, args.weights, args.guides, args.output, device=args.device)


if __name__ == '__main__':
    main()
