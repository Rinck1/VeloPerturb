"""Strict gene-order and source-only contracts for the GFG-specific data path."""
from pathlib import Path

import numpy as np

from .artifacts import sha256
from .latent import FrozenSplicingTransform


def validate_gene_source(data, source, metadata, gene_metadata, source_path):
    if set(data.files) != {'gene_us', 'gene_ids', 'cell_ids', 'metadata_json'}:
        raise ValueError('Unknown GFG gene-source fields; source-only whitelist')
    kang = metadata.get('task') == 'Kang_ctrl_to_IFNB_donor_holdout'
    if kang:
        if (metadata.get('dataset') != 'Kang_GSE96583_batch2'
            or metadata.get('sample_condition') != 'ctrl'
            or gene_metadata.get('sample_condition') != 'ctrl'
            or gene_metadata.get('task') != metadata['task']
            or gene_metadata.get('dataset') != metadata['dataset']
            or not all(str(x).startswith('ctrl:') for x in source['cell_ids'])
            or gene_metadata.get('heldout_donor') != metadata.get('heldout_donor')):
            raise ValueError('Kang GFG accepts only aligned heldout-contract control S/U')
    elif gene_metadata.get('day') != 4:
        exploratory = (metadata.get('kind') == 'exploratory_gain_probe'
                       and gene_metadata.get('kind') == 'exploratory_gain_probe')
        if not exploratory:
            raise ValueError('GFG input must be day4 source_gene_US')
    if gene_metadata.get('side') != 'source_gene_US':
        raise ValueError('GFG input must have the source_gene_US side')
    if gene_metadata.get('normalization') != 'source_S_library_10000_unlogged_US_no_neighbor_smoothing':
        raise ValueError('GFG RNA input normalization mismatch')
    transform_path = Path(source_path).parent/'transform.npz'
    if sha256(transform_path) != metadata['transform_hash']:
        raise ValueError('GFG actual frozen transform hash mismatch')
    transform = FrozenSplicingTransform.load(transform_path)
    expected_genes = np.asarray(transform.gene_ids)[transform.selected]
    if not np.array_equal(data['gene_ids'], expected_genes):
        raise ValueError('GFG gene order differs from frozen PCA')
    values = data['gene_us']
    if values.shape != (len(source['z']), 2*len(expected_genes)) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('GFG gene-source counts have invalid dimensions or values')
    z = (np.log1p(values[:, len(expected_genes):])-transform.mean)@transform.components.T
    if not np.allclose(z, source['z'], rtol=1e-4, atol=1e-5):
        raise ValueError('GFG spliced counts do not reproduce the frozen source state')
