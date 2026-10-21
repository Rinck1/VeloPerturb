"""Traxler technical annotation recovery only: no effects, embeddings or models.

Reads only A:L of S5 single_cell_annotations. In particular, never opens its
DEA/enrichment sheets and never reads the Mixscape columns in that same sheet.
Existing S/U matrices are opened backed/read-only and are not modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from w4_artifacts import Artifacts, sha256


DATA = Path('/data/yuchang/veloroute_w4/traxler2025')
S5_URL = ('https://medical-epigenomics.org/papers/traxler2025/results/'
          'main_supplement/S05_CROP-seq_large.xlsx')
NOTEBOOK_URL = ('https://raw.githubusercontent.com/epigen/macrophage-regulation/'
                'main/src/KO150_01_processing_unsupervised_analysis_all.R.ipynb')
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
ALLOWED = {
    'A': 'Cell barcode', 'B': 'Time point', 'C': 'Inferred knockout',
    'D': 'RNA counts', 'E': 'Genes detected', 'F': 'Antibody counts',
    'G': 'Antibodies detected', 'H': 'gRNA counts', 'I': 'gRNAs detected',
    'J': 'Batch', 'K': 'Mitochondrial count (fraction)', 'L': 'Inferred gRNA',
}


def fetch(url, destination):
    """Bounded technical supplement download; retain an existing original."""
    destination = Path(destination)
    if destination.exists():
        return dict(url=url, path=str(destination), bytes=destination.stat().st_size,
                    sha256=sha256(destination), reused=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as response:
        payload = response.read(32_000_001)
    if len(payload) > 32_000_000:
        raise ValueError('Technical supplement exceeds the bounded 32 MB acquisition')
    with destination.open('xb') as stream:
        stream.write(payload)
    return dict(url=url, path=str(destination), bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(), reused=False)


def shared_strings(archive, requested):
    """Resolve only references belonging to explicitly allowed technical cells."""
    result = {}
    if not requested:
        return result
    with archive.open('xl/sharedStrings.xml') as stream:
        index = -1
        for _, element in ET.iterparse(stream, events=('end',)):
            if element.tag == NS + 'si':
                index += 1
                if index in requested:
                    result[index] = ''.join(n.text or '' for n in element.iter(NS + 't'))
                element.clear()
    if set(result) != set(requested):
        raise ValueError('Unresolved technical-cell shared-string references')
    return result


def technical_annotations(path):
    """Hard-whitelist one worksheet and its A:L cells, ignoring all other data."""
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        rid = None
        for sheet in workbook.find(NS + 'sheets'):
            if sheet.attrib['name'] == 'single_cell_annotations':
                rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        relationships = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        targets = {r.attrib['Id']: r.attrib['Target'] for r in relationships}
        if rid not in targets:
            raise ValueError('Required technical annotation sheet is absent')
        target = targets[rid].lstrip('/')
        if not target.startswith('xl/'):
            target = 'xl/' + target
        raw_rows = []
        references = set()
        with archive.open(target) as stream:
            for _, element in ET.iterparse(stream, events=('end',)):
                if element.tag != NS + 'row':
                    continue
                row = {}
                for cell in element:
                    column = re.sub(r'[0-9]', '', cell.attrib['r'])
                    if column not in ALLOWED:
                        continue  # Never interpret Mixscape or other nontechnical values.
                    kind = cell.attrib.get('t', 'n')
                    value = cell.find(NS + 'v')
                    if kind == 'inlineStr':
                        text = ''.join(n.text or '' for n in cell.iter(NS + 't'))
                    else:
                        text = value.text if value is not None else None
                    if kind == 's' and text is not None:
                        references.add(int(text))
                    row[column] = (kind, text)
                raw_rows.append(row)
                element.clear()
        strings = shared_strings(archive, references)
    def decode(pair):
        kind, value = pair
        if value is None:
            return None
        return strings[int(value)] if kind == 's' else value
    header = {column: decode(pair) for column, pair in raw_rows[0].items()}
    if header != ALLOWED:
        raise ValueError('Author technical column contract changed: ' + repr(header))
    records = [{ALLOWED[column]: decode(row.get(column, ('n', None)))
                for column in ALLOWED} for row in raw_rows[1:]]
    frame = pd.DataFrame(records).dropna(subset=['Cell barcode'])
    for column in ('RNA counts', 'Genes detected', 'Antibody counts',
                   'Antibodies detected', 'gRNA counts', 'gRNAs detected',
                   'Mitochondrial count (fraction)'):
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    return frame


def canonical_key(cell, batch):
    match = re.fullmatch(r'([ABC][123])_([ACGT]{16})-1', str(cell))
    if match is None or match.group(1) != str(batch):
        raise ValueError('Author barcode/batch contract mismatch: ' + repr((cell, batch)))
    return match.group(1) + ':' + match.group(2)


def hto_dominance(counts, ids, minimum_fraction=.9):
    """Transparent high-dominance source tag, not a reconstructed author GMM.

    The accepted-cell set comes from the authors, not this threshold. The 0.9
    tag fraction is a fixed conservative technical identity annotation only.
    Labels that fail remain unresolved; never replace missing author QC with it.
    """
    counts = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != len(ids) or (counts < 0).any():
        raise ValueError('Invalid HTO count matrix')
    totals = counts.sum(1)
    top = counts.argmax(1)
    fraction = np.divide(counts[np.arange(len(counts)), top], totals,
                         out=np.zeros(len(counts), dtype=float), where=totals > 0)
    sources = []
    tags = []
    for index, dominance in zip(top, fraction):
        tag = str(ids[index])
        match = re.fullmatch(r'HTO-RAW264\.7-([abc])_(.+)', tag)
        if match is None:
            raise ValueError('Unknown author HTO feature identity: ' + tag)
        tags.append(tag)
        sources.append('RAW264.7-' + match.group(1) if dominance >= minimum_fraction else 'UNRESOLVED')
    return np.array(sources), np.array(tags), fraction, totals


def quantiles(values):
    array = np.asarray(values, dtype=float)
    return {str(q): float(np.quantile(array, q)) for q in (.05, .25, .5, .75, .95)}


def author_processing_evidence(data):
    """Retain source-code-only technical evidence, never notebook outputs."""
    urls = [(NOTEBOOK_URL,[3,6,7,13,15,17,20,26,40,44,51]),
            (NOTEBOOK_URL.replace('KO150_01_processing_unsupervised_analysis_all.R.ipynb',
                                   'Supplementary_Tables.py.ipynb'),[58])]
    records=[]
    for url, indices in urls:
        payload=urllib.request.urlopen(url,timeout=120).read()
        notebook=json.loads(payload)
        records.append(dict(url=url,sha256=hashlib.sha256(payload).hexdigest(),
            code_cells={str(i):''.join(notebook['cells'][i]['source']) for i in indices},
            notebook_outputs_read=False))
    soft=Path(data)/'GSE263761_family.soft.gz'
    with gzip.open(soft,'rt') as stream:
        extracts=[line.strip().split(' = ',1)[1] for line in stream
                  if line.startswith(('!Sample_treatment_protocol_ch1 = ',
                                      '!Sample_extract_protocol_ch1 = '))]
    return dict(notebooks=records,GEO_SOFT_path=str(soft),GEO_SOFT_sha256=sha256(soft),
        unique_GEO_technical_protocols=list(dict.fromkeys(extracts)),
        independent_sources='three RAW264.7 frozen vials thawed on different days, individually hashtagged',
        source_caveat='independent thaw/culture sources in one cell line, not three donors; chip IDs 1/2/3 are technical tubes',
        raw_QC_csv='QC_categories.csv per library: not present in GEO H5 or inspected author metadata directory',
        guide_threshold_csv='crispr_analysis/protospacer_umi_thresholds.csv per library: not publicly located',
        author_S5_HTO_omission='Supplementary_Tables.py.ipynb cell58 explicitly drops hto_demux and pass_QC',
        author_mito_units='Seurat PercentageFeatureSet: percent; S5 header says fraction but values are percentages')


def run(output, data=DATA):
    data = Path(data)
    artifacts = Artifacts(output, seed=0, inputs=[Path(__file__)])
    acquisition = fetch(S5_URL, data/'author_technical_metadata/S05_CROP-seq_large.xlsx')
    artifacts.json('acquisition.json', acquisition)
    artifacts.json('author_processing_evidence.json',author_processing_evidence(data))
    annotations = technical_annotations(acquisition['path'])
    annotations['cell_id'] = [canonical_key(c, b) for c, b in
                              zip(annotations['Cell barcode'], annotations['Batch'])]
    if annotations.cell_id.duplicated().any():
        raise ValueError('Duplicate author barcode keys')
    annotations = annotations.set_index('cell_id')
    annotations.to_csv(artifacts.directory/'author_technical_annotations.csv.gz')
    artifacts.json('author_annotation_contract.json', dict(
        worksheet='single_cell_annotations', columns_read=list(ALLOWED.values()),
        author_rows=len(annotations), batches=annotations['Batch'].value_counts().to_dict(),
        time_labels=annotations['Time point'].value_counts().to_dict(),
        excluded_columns='M:W (including every Mixscape field)',
        other_worksheets_read=False, cells_are_author_post_QC=True,
        exact_author_raw_QC_thresholds_available=False))
    records, library_qc, paths = [], [], []
    for library in [a + b for a in 'ABC' for b in '123']:
        path = data/'processed_usa_v1'/(library+'.h5ad')
        a = ad.read_h5ad(path, backed='r')
        try:
            obs = a.obs.copy()
            sources, tags, fraction, hto_total = hto_dominance(
                a.obsm['published_hto_counts'], a.uns['published_hto_ids'])
            obs['source_identity_dominant_HTO'] = sources
            obs['top_HTO_id'] = tags
            obs['top_HTO_fraction'] = fraction
            obs['HTO_total_UMI'] = hto_total
            obs['author_post_QC_inclusion'] = obs.index.isin(annotations.index)
            overlap = annotations.reindex(obs.index)
            expected_time={'A':'Untreated','B':'Listeria 6h','C':'Listeria 24h'}[library[0]]
            if set(overlap.loc[obs['author_post_QC_inclusion'],'Time point'])!={expected_time}:
                raise ValueError('Author time label versus library mismatch')
            for source, name in [('Inferred knockout','author_KO_call'),
                                 ('Inferred gRNA','author_guide_call'),
                                 ('RNA counts','author_RNA_UMI'),
                                 ('Genes detected','author_detected_genes'),
                                 ('Mitochondrial count (fraction)','author_mito_percent')]:
                obs[name] = overlap[source].to_numpy()
            obs['author_QC_plus_resolved_source'] = (obs['author_post_QC_inclusion'] &
                (obs['source_identity_dominant_HTO'] != 'UNRESOLVED'))
            obs['biological_replicate'] = 'UNRESOLVED_AUTHOR_HTO_CALLS_NOT_PROVIDED'
            obs['source_identity_semantics'] = 'dominant_HTO_90pct_not_exact_author_GMM'
            obs.to_csv(artifacts.directory/(library+'_technical_metadata.csv.gz'))
            records.append(obs)
            selected = obs[obs['author_post_QC_inclusion']]
            library_qc.append(dict(library=library, cells=len(obs), author_retained=len(selected),
                author_retained_resolved_source=int(selected['author_QC_plus_resolved_source'].sum()),
                provisional_nonzero_single_guide=int((obs.n_nonzero_guides == 1).sum()),
                author_retained_without_raw_nonzero_single_guide=int((selected.n_nonzero_guides != 1).sum()),
                HTO_ids=[str(x) for x in a.uns['published_hto_ids']],
                S_quantiles=quantiles(obs.spliced_umi), U_quantiles=quantiles(obs.unspliced_umi),
                U_fraction_S_U_quantiles=quantiles(obs.unspliced_umi/(obs.spliced_umi+obs.unspliced_umi).clip(lower=1)),
                author_RNA_quantiles=quantiles(selected.author_RNA_UMI),
                author_detected_gene_quantiles=quantiles(selected.author_detected_genes),
                author_mito_percent_quantiles=quantiles(selected.author_mito_percent)))
            paths.append(dict(path=str(path), sha256=sha256(path), original_modified=False))
        finally:
            a.file.close()
    combined = pd.concat(records)
    if not set(annotations.index) <= set(combined.index):
        raise ValueError('Author retained-cell set has barcodes missing from S/U matrices')
    selected = combined[combined.author_post_QC_inclusion].copy()
    selected.groupby(['time_hours','author_KO_call'], observed=True).size().rename('cells').reset_index().to_csv(
        artifacts.directory/'time_by_KO_author_QC_cells.csv',index=False)
    selected.groupby(['time_hours','source_identity_dominant_HTO','author_KO_call'], observed=True).size().rename('cells').reset_index().to_csv(
        artifacts.directory/'time_source_by_KO_technical_cells.csv',index=False)
    combined.groupby(['library','author_post_QC_inclusion','source_identity_dominant_HTO'],observed=True).size().rename('cells').reset_index().to_csv(
        artifacts.directory/'library_QC_source_counts.csv',index=False)
    available = {}
    for hour, group in selected.groupby('time_hours'):
        counts = group.author_KO_call.value_counts()
        ko_counts=counts.drop(index=['NonTargeting'],errors='ignore')
        available[str(int(hour))] = dict(cells=len(group), targets=int(len(counts)),
            targets_at_least_30_cells=int((counts >= 30).sum()),
            KO_targets_excluding_NonTargeting=int(len(ko_counts)),
            KO_targets_at_least_30_cells_excluding_NonTargeting=int((ko_counts>=30).sum()),
            NonTargeting_cells=int(counts.get('NonTargeting',0)),
            resolved_source_cells=int(group.author_QC_plus_resolved_source.sum()),
            source_counts=group.source_identity_dominant_HTO.value_counts().to_dict())
    result = dict(status='TRAXLER_AUTHOR_TECHNICAL_METADATA_RECOVERED',
        input_cells=len(combined), author_QC_retained=len(selected),
        available_by_time=available, libraries=library_qc, input_matrices=paths,
        author_HTO_calls_recovered=False, source_tags='a,b,c identity only from feature names',
        source_provenance='three independent frozen-vial thaws, not three donors; exact author cell-to-HTO calls absent',
        source_annotation='HTO top fraction >=0.90; otherwise unresolved; not exact author GMM',
        author_QC_retained_set_recovered=True, raw_QC_categories_available=False,
        exact_per_protospacer_thresholds_available=False,
        original_provisional_guide_calls_replaced_only_in_new_metadata=True,
        new_dataset_perturbation_effects_read=False, Mixscape_fields_read=False,
        trained_models=False, GPU_used=False,
        next='Obtain exact author HTO/QC_categories or validate a frozen technical demux before W15 source-held-out training')
    artifacts.json('summary.json', result)
    (artifacts.directory/'RESULTS.md').write_text('# Traxler technical metadata recovery\n\n'+
        json.dumps(result,indent=2)+'\n')
    artifacts.json('status.json', dict(status='complete'))
    print(json.dumps({k: result[k] for k in ('status','input_cells','author_QC_retained','available_by_time')},indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--data',default=str(DATA))
    args=parser.parse_args()
    run(args.output,args.data)
