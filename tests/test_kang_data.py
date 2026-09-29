import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from veloroute.artifacts import load_config
from veloroute.gfg_experiments import read_gene_input
from veloroute.kang_data import prepare_kang_fold, published_metadata, sample_keys, tagged_fastq
from veloroute.latent import FrozenSplicingTransform, load_pack


def test_same_barcode_different_samples_is_not_a_collision(tmp_path):
    import gzip
    path = tmp_path/'GSE96583_batch2.total.tsne.df.tsv.gz'
    with gzip.open(path,'wt') as f:
        f.write('tsne1\ttsne2\tind\tstim\tcluster\tcell\tmultiplets\n')
        for condition in ('ctrl','stim'):
            f.write(f'AAAAAAAAAAAAAA-1\t0\t0\t101\t{condition}\t1\tCD14+ Monocytes\tsinglet\n')
    data = published_metadata(tmp_path)
    assert list(data.index) == ['ctrl:AAAAAAAAAAAAAA-1','stim:AAAAAAAAAAAAAA-1']


def write_kang_metadata(tmp_path, rows):
    import gzip
    path = tmp_path/'GSE96583_batch2.total.tsne.df.tsv.gz'
    with gzip.open(path,'wt') as f:
        f.write('tsne1\ttsne2\tind\tstim\tcluster\tcell\tmultiplets\n')
        for row in rows:
            f.write('\t'.join(str(x) for x in row)+'\n')


def test_sample_keys_keep_each_condition_published_suffix(tmp_path):
    write_kang_metadata(tmp_path, [
        ('AAAAAAAAAAAAAA-1',0,0,101,'ctrl',1,'CD14+ Monocytes','singlet'),
        ('AAAAAAAAAAAAAA-11',0,0,101,'stim',1,'CD14+ Monocytes','singlet')])
    metadata = published_metadata(tmp_path)
    assert sample_keys(metadata,'stim',['AAAAAAAAAAAAAA']) == ['stim:AAAAAAAAAAAAAA-11']
    assert sample_keys(metadata,'ctrl',['AAAAAAAAAAAAAA']) == ['ctrl:AAAAAAAAAAAAAA-1']


def test_sample_keys_reject_ambiguous_and_unknown_barcodes(tmp_path):
    write_kang_metadata(tmp_path, [
        ('BBBBBBBBBBBBBB-1',0,0,101,'ctrl',1,'CD14+ Monocytes','singlet'),
        ('AAAAAAAAAAAAAA-1',0,0,101,'stim',1,'CD14+ Monocytes','singlet'),
        ('AAAAAAAAAAAAAA-11',0,0,101,'stim',1,'CD14+ Monocytes','singlet')])
    metadata = published_metadata(tmp_path)
    with pytest.raises(ValueError,match='not unique within one condition'):
        sample_keys(metadata,'stim',['AAAAAAAAAAAAAA'])
    write_kang_metadata(tmp_path, [
        ('AAAAAAAAAAAAAA-1',0,0,101,'ctrl',1,'CD14+ Monocytes','singlet'),
        ('BBBBBBBBBBBBBB-1',0,0,101,'stim',1,'CD14+ Monocytes','singlet')])
    metadata = published_metadata(tmp_path)
    with pytest.raises(ValueError,match='align uniquely'):
        sample_keys(metadata,'stim',['CCCCCCCCCCCCCC'])


def test_bam_tags_reverse_sequence_and_intronic_records_are_preserved():
    pysam = pytest.importorskip('pysam')
    read = pysam.AlignedSegment()
    read.query_name = 'read1'
    read.query_sequence = 'ACGTA'
    read.query_qualities = pysam.qualitystring_to_array('ABCDE')
    read.flag = 16
    read.set_tag('CR','AAAAAAAAAAAAAA')
    read.set_tag('CB','AAAAAAAAAAAAAA-1')
    read.set_tag('UR','ACGTACGTAA')
    result, why = tagged_fastq(read, {'AAAAAAAAAAAAAA'})
    assert why == 'retained'  # no GX required, no alignment-based expression filter
    assert result[1].splitlines()[1] == 'TACGT'
    assert result[1].splitlines()[3] == 'EDCBA'
    assert result[0].splitlines()[1] == 'AAAAAAAAAAAAAAACGTACGTAA'
    read.flag = 256
    assert tagged_fastq(read, {'AAAAAAAAAAAAAA'})[1] == 'nonprimary'


def fixture_config(tmp_path):
    import anndata as ad
    c = load_config('configs/veloroute_kang_20260914.yaml')
    c['root'] = str(tmp_path/'data')
    c['representation'].update(selected_genes=6,pca_components=3)
    c['cell_types']['minimum_cells_per_donor_side'] = 2
    root = Path(c['root'])
    rng = np.random.default_rng(13)
    for condition in ('ctrl','stim'):
        cells = [condition+':'+d+'_'+str(i) for d in c['donors'] for i in range(6)]
        obs = pd.DataFrame(dict(donor=np.repeat(c['donors'],6), condition=condition,
            cell_type='CD14+ Monocytes',technical_qc_pass=True),index=cells)
        s = sparse.csr_matrix(rng.poisson(10,(len(cells),8))+1, dtype='float32')
        u = sparse.csr_matrix(rng.poisson(4,(len(cells),8)), dtype='float32')
        a = ad.AnnData(s.copy(),obs=obs,var=pd.DataFrame(index=[f'ENSG{i}' for i in range(8)]))
        a.layers.update(spliced=s,unspliced=u)
        out = root/'processed'/condition
        out.mkdir(parents=True)
        a.write_h5ad(out/f'{condition}.h5ad')
    return c


def test_kang_folds_source_contract_and_constant_perturbation_tokens(tmp_path):
    c = fixture_config(tmp_path)
    out = tmp_path/'fold'
    prepare_kang_fold(c,'101',out)
    us, source, meta = read_gene_input(out/'validation_gene_source.npz',out/'validation_source.npz')
    assert set(source['conditions']) == {'101|CD14+ Monocytes'}
    assert meta['task'] == 'Kang_ctrl_to_IFNB_donor_holdout'
    assert np.count_nonzero(source['velocity']) == 0  # online GFG, never fabricated
    with np.load(out/'conditions.npz') as data:
        assert np.all(data['embeddings'] == 1)
    fit = pd.read_csv(out/'fit_cell_ids.csv').cell_id.tolist()
    assert not set(fit) & set(source['cell_ids'])
    with np.load(out/'validation_gene_source.npz') as data:
        payload = {k:data[k] for k in data.files}
    gm = json.loads(str(payload['metadata_json']))
    gm['sample_condition'] = 'stim'
    payload['metadata_json'] = np.array(json.dumps(gm))
    np.savez_compressed(out/'bad.npz',**payload)
    with pytest.raises(ValueError,match='control'):
        read_gene_input(out/'bad.npz',out/'validation_source.npz')


def test_heldout_expression_cannot_change_fit_or_train_packs(tmp_path):
    import anndata as ad
    c = fixture_config(tmp_path)
    first, second = tmp_path/'first',tmp_path/'second'
    prepare_kang_fold(c,'101',first)
    for condition in ('ctrl','stim'):
        path = Path(c['root'])/'processed'/condition/f'{condition}.h5ad'
        data = ad.read_h5ad(path)
        mask = data.obs.donor.to_numpy() == '101'
        for layer in ('spliced','unspliced'):
            matrix = data.layers[layer].toarray()
            matrix[mask] = matrix[mask]*7+100
            data.layers[layer] = sparse.csr_matrix(matrix)
        data.write_h5ad(path)
    prepare_kang_fold(c,'101',second)
    a,b = [FrozenSplicingTransform.load(p/'transform.npz') for p in (first,second)]
    for name in ('selected','mean','components'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    for side in ('source','target'):
        x = load_pack(first/f'train_{side}.npz',expected_side=side)[0]
        y = load_pack(second/f'train_{side}.npz',expected_side=side)[0]
        for name in x: np.testing.assert_array_equal(x[name],y[name])
