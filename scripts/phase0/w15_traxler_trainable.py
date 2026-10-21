"""Make Traxler KO150 trainable: manifest + frozen source-only KO split.

Uses ONLY identifiers, technical counts (S/U UMI) and author QC/guide/KO labels.
It does NOT compute differential expression, signatures, or any effect-based
selection. The split is frozen BEFORE any effect analysis.

Source identity = our own dominant-HTO rule (>=90% of HTO UMI), NOT the author's
GMM; biological_replicate remains UNRESOLVED (three independent freeze/thaw tubes
a/b/c, not three donors).

Outputs (outputs/w15_traxler_trainable_v1/):
  cells.csv.gz          one row per author-QC cell
  split_0to6.json       frozen train/val/test KO assignment for 0h->6h
  split_6to24.json      frozen train/val/test KO assignment for 6h->24h
  summary.json          coverage, U/S medians, split sizes, hashes
  RESULTS.md
"""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TIME_BY_LIB = {'A': 0, 'B': 6, 'C': 24}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demux', default='outputs/w15_traxler_demux_v2')
    ap.add_argument('--min-cells', type=int, default=30)
    ap.add_argument('--train-frac', type=float, default=0.8)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', default='outputs/w15_traxler_trainable_v1')
    args = ap.parse_args()
    demux = Path(args.demux); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    libs = sorted([f'{p}{i}' for p in 'ABC' for i in (1, 2, 3)])
    frames = []
    for lib in libs:
        p = demux / f'{lib}_technical_metadata.csv.gz'
        df = pd.read_csv(p)
        if df.columns[0].startswith('Unnamed'):
            df = df.drop(columns=df.columns[0])
        df['library'] = lib
        frames.append(df)
    allc = pd.concat(frames, ignore_index=True)
    print(f'[T] total cells={len(allc)}', flush=True)

    # author QC retained
    qc = allc[allc['author_post_QC_inclusion'] == True].copy()
    qc['KO'] = qc['author_KO_call'].astype(str)
    qc['time'] = qc['time_hours'].astype(int)
    qc['source'] = qc['source_identity_dominant_HTO'].astype(str)
    qc['S'] = qc['spliced_umi']
    qc['U'] = qc['unspliced_umi']
    keep = ['cell_id', 'barcode', 'library', 'time', 'KO', 'source', 'S', 'U',
            'guide_call', 'author_guide_call', 'author_RNA_UMI', 'author_detected_genes',
            'author_mito_percent', 'source_identity_semantics']
    qc = qc[[c for c in keep if c in qc.columns]]
    qc.to_csv(out / 'cells.csv.gz', index=False, compression='gzip')
    print(f'[T] author-QC cells={len(qc)}', flush=True)

    # coverage per (KO, time)
    cov = qc.groupby(['KO', 'time']).size().unstack(fill_value=0)
    for t in (0, 6, 24):
        if t not in cov.columns:
            cov[t] = 0

    def build_split(t0, t1, name):
        ok = cov[(cov[t0] >= args.min_cells) & (cov[t1] >= args.min_cells)].index
        ok = [k for k in ok if k.lower() not in ('nan', 'nontargeting', '')]
        kos = sorted(ok)
        # stratify by min coverage quartile, then shuffle within strata
        rng = np.random.default_rng(args.seed)
        strata = {}
        for k in kos:
            q = int(np.searchsorted(np.quantile(np.minimum(cov.loc[kos, t0].values, cov.loc[kos, t1].values), [0.25, 0.5, 0.75]), min(cov.loc[k, t0], cov.loc[k, t1])))
            strata.setdefault(q, []).append(k)
        train, val, test = [], [], []
        for q in sorted(strata):
            ks = list(strata[q]); rng.shuffle(ks)
            n = len(ks)
            n_tr = int(round(n * args.train_frac)); n_va = int(round(n * args.val_frac))
            train += ks[:n_tr]; val += ks[n_tr:n_tr + n_va]; test += ks[n_tr + n_va:]
        spec = dict(task=f'{t0}h_to_{t1}h', min_cells=args.min_cells, seed=args.seed,
                    fractions=dict(train=args.train_frac, val=args.val_frac),
                    n_kos=len(kos), n_train=len(train), n_val=len(val), n_test=len(test),
                    train=train, val=val, test=test,
                    selection='source-side coverage only; no differential expression or effect used',
                    source_semantics='dominant_HTO_90pct_not_author_GMM; biological_replicate UNRESOLVED')
        (out / f'split_{name}.json').write_text(json.dumps(spec, indent=2))
        print(f'[T] {name}: KOs={len(kos)} train/val/test={len(train)}/{len(val)}/{len(test)}', flush=True)
        return spec

    s06 = build_split(0, 6, '0to6')
    s624 = build_split(6, 24, '6to24')

    def depth(df):
        return dict(cells=int(len(df)), median_S=float(df['S'].median()), median_U=float(df['U'].median()),
                    median_U_frac=float((df['U'] / (df['S'] + df['U']).replace(0, np.nan)).median()),
                    pooled_U_frac=float(df['U'].sum() / (df['S'].sum() + df['U'].sum())))
    summary = dict(status='TRAXLER_TRAINABLE_MANIFEST_READY', seed=args.seed,
                   total_cells=int(len(allc)), author_QC_cells=int(len(qc)),
                   per_time={int(t): depth(qc[qc['time'] == t]) for t in sorted(qc['time'].unique())},
                   source_counts=qc['source'].value_counts().to_dict(),
                   splits=dict(s0=s06, s6=s624),
                   inputs={str(p): sha256(p) for p in sorted(demux.glob('*_technical_metadata.csv.gz'))},
                   notes=['no KO effect / DE / signature computed',
                          'split frozen before any effect analysis',
                          'source and biological replicate remain technical/unresolved'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'RESULTS.md').write_text('# Traxler trainable manifest\n\n' + json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'splits'}, indent=2)[:2000])


if __name__ == '__main__':
    main()
