import json
from pathlib import Path
import sys

import anndata as ad
import h5py
import numpy as np
import pytest
from scipy import io,sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w14b_traxler_su as pipeline


def test_layout_rejects_any_changed_technical_cycle(monkeypatch,tmp_path):
    good=[dict(READ_START=[0,26,118,128],READ_LEN=[26,92,10,10])]
    monkeypatch.setattr(pipeline.subprocess,'check_output',lambda *a,**k:json.dumps(good))
    assert pipeline.layout(tmp_path/'toy.sra')==good
    good[0]['READ_LEN'][0]=28
    with pytest.raises(ValueError,match='layout'):pipeline.layout(tmp_path/'toy.sra')


def test_usa_import_preserves_layers_and_marks_provisional_identity(tmp_path):
    q=tmp_path/'quant'/'af_quant';m=q/'alevin';m.mkdir(parents=True)
    (q/'quant.json').write_text(json.dumps(dict(usa_mode=True,num_genes=2)))
    (m/'quants_mat_rows.txt').write_text('CELL1\nCELL2\n')
    (m/'quants_mat_cols.txt').write_text('gene1\ngene2\n')
    matrix=sparse.csr_matrix(np.array([[3,0,2,1,1,0],[0,4,1,2,0,1]]))
    io.mmwrite(m/'quants_mat.mtx',matrix)
    author=sparse.csc_matrix(np.array([[4,0],[0,5],[3,0],[0,2],[4,1],[0,3]]))
    h5=tmp_path/'published.h5'
    with h5py.File(h5,'w') as h:
        group=h.create_group('matrix')
        for key,value in [('shape',author.shape),('data',author.data),('indices',author.indices),('indptr',author.indptr)]:
            group.create_dataset(key,data=value)
        group.create_dataset('barcodes',data=np.array(['CELL1-1','CELL2-1'],dtype='S'))
        f=group.create_group('features')
        f.create_dataset('id',data=np.array(['gene1','gene2','TF1_1','TF2_1','HTO1','HTO2'],dtype='S'))
        f.create_dataset('feature_type',data=np.array(['Gene Expression']*2+['CRISPR Guide Capture']*2+['Custom']*2,dtype='S'))
    path=tmp_path/'B2.h5ad';record,counts=pipeline.import_counts(None,'B2',tmp_path/'quant',h5,path)
    result=ad.read_h5ad(path)
    np.testing.assert_array_equal(result.layers['spliced'].toarray(),matrix[:,:2].toarray())
    np.testing.assert_array_equal(result.layers['unspliced'].toarray(),matrix[:,2:4].toarray())
    assert record['guide_calls_provisional'] and not record['HTO_demultiplexing_complete']
    assert set(result.obs['biological_replicate'])=={'UNRESOLVED_USE_HTO'}
    assert set(result.obs['provisional_gene_target'])=={'TF1','TF2'}
    assert result.obsm['published_hto_counts'].shape==(2,2)


def test_barcode_audit_uses_declared_offset_only(monkeypatch,tmp_path):
    bc=['AACCGGTTAACCGGTT','TTAACCGGAATTCCGG'];path=tmp_path/'published.h5'
    with h5py.File(path,'w') as h:h.create_dataset('matrix/barcodes',data=np.array([x+'-1' for x in bc],dtype='S'))
    reads=[dict(READ=x+'GTCAAGTCAA'+'A'*92) for x in bc]*50
    monkeypatch.setattr(pipeline.subprocess,'check_output',lambda *a,**k:json.dumps(reads))
    result=pipeline.barcode_audit(tmp_path/'toy.sra',path)
    assert result['approved_offset']==0 and result['exact_whitelist_fraction_by_offset'][0]==1
