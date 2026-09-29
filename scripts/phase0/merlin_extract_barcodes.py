"""MeRLin clone barcode extraction from paired 10x fastq.

Rule (verified on SRR33960308): scan R2 for the constant flank
CAGATCTTAGCCACTTTTTAAAAGAAAAGGGGG; the clonal barcode is the 15/30 bases
immediately UPSTREAM in read order, with period-3 alphabet
{i%3==2: A/T, i%3==0: C/G, i%3==1: any}. CB from R1 first 16 bases.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

FLANK = b'CAGATCTTAGCCACTTTTTAAAAGAAAAGGGGG'
BATCH = 50316


def pattern_ok(b):
    for i, c in enumerate(b):
        ch = chr(c)
        if i % 3 == 2 and ch not in 'AT':
            return False
        if i % 3 == 0 and ch not in 'CG':
            return False
    return True


def scan_batch(payload):
    r1_lines, r2_lines = payload
    hits = []
    for i in range(1, len(r2_lines), 4):
        seq = r2_lines[i]
        p = seq.find(FLANK)
        if p < 30:
            continue
        b15 = seq[p-15:p]
        if not pattern_ok(b15):
            continue
        cb = r1_lines[i][:16].decode()
        hits.append((cb, b15.decode(), seq[p-30:p].decode()))
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r1', required=True)
    parser.add_argument('--r2', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--threads', type=int, default=48)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p1 = subprocess.Popen(['pigz', '-dc', args.r1], stdout=subprocess.PIPE, bufsize=1 << 22)
    p2 = subprocess.Popen(['pigz', '-dc', args.r2], stdout=subprocess.PIPE, bufsize=1 << 22)
    f1, f2 = p1.stdout, p2.stdout
    counts15 = defaultdict(Counter)
    counts30 = defaultdict(Counter)
    scanned = 0
    hits_total = 0
    with Pool(args.threads) as pool:
        while True:
            r1_lines, r2_lines = [], []
            for _ in range(BATCH):
                a = f1.readline()
                b = f2.readline()
                if not a or not b:
                    break
                r1_lines.extend([a.rstrip(b'\n')])
                r2_lines.extend([b.rstrip(b'\n')])
                for _ in range(3):
                    r1_lines.append(f1.readline().rstrip(b'\n'))
                    r2_lines.append(f2.readline().rstrip(b'\n'))
            if not r1_lines:
                break
            for cb, b15, b30 in pool.apply(scan_batch, ((r1_lines, r2_lines),)):
                counts15[cb][b15] += 1
                counts30[cb][b30] += 1
            scanned += len(r1_lines)//4
            hits_total = sum(sum(c.values()) for c in counts15.values())
            if scanned % (BATCH * 40) < BATCH:
                print(json.dumps(dict(scanned=scanned, hits=hits_total,
                                      cells=len(counts15), clones=len(set(b for c in counts15.values() for b in c)))), flush=True)
    p1.wait(); p2.wait()
    rows = []
    for cb, counter in counts15.items():
        top, n = counter.most_common(1)[0]
        total = sum(counter.values())
        rows.append(dict(cell_barcode=cb, clone_barcode=top, clone_reads=n, total_reads=total,
                         dominant_fraction=n/total, n_barcodes=len(counter)))
    assignment = [r for r in rows if r['dominant_fraction'] >= 0.5]
    import csv
    with (out/'cell_clone_assignments.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['cell_barcode'])
        writer.writeheader()
        writer.writerows(assignment)
    summary = dict(status='MERLIN_BARCODES_EXTRACTED', scanned_pairs=scanned, valid_hits=hits_total,
                   cells_with_barcode=len(counts15), cells_assigned=len(assignment),
                   unique_clones_15bp=len(set(r['clone_barcode'] for r in assignment)),
                   unique_clones_30bp=len(set(b for c in counts30.values() for b in c)),
                   r1=args.r1, r2=args.r2)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
