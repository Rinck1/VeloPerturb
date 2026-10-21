"""Technical-only Traxler metadata contracts, no real expression inputs."""
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w15_traxler_demux import ALLOWED,canonical_key,hto_dominance,technical_annotations


def test_barcode_library_contract():
    assert canonical_key('A1_AAACCCAAGAAACACT-1','A1')=='A1:AAACCCAAGAAACACT'
    with pytest.raises(ValueError):canonical_key('A1_AAACCCAAGAAACACT-1','A2')
    with pytest.raises(ValueError):canonical_key('AAACCCAAGAAACACT','A1')


def test_Hashed_source_not_chip_and_ambiguous_remains_unresolved():
    ids=['HTO-RAW264.7-a_untreated','HTO-RAW264.7-b_untreated','HTO-RAW264.7-c_untreated']
    x=sparse.csr_matrix([[99,1,0],[1,99,0],[40,40,20],[0,0,0],[0,0,100]])
    source,top,dominance,total=hto_dominance(x,ids)
    assert source.tolist()==['RAW264.7-a','RAW264.7-b','UNRESOLVED','UNRESOLVED','RAW264.7-c']
    np.testing.assert_array_equal(total,[100,100,100,0,100])
    assert len(top)==len(dominance)==5


def test_only_technical_worksheet_columns_are_interpreted(tmp_path):
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    header=''.join(f'<c r="{col}1" t="inlineStr"><is><t>{name}</t></is></c>' for col,name in ALLOWED.items())
    # R2 is deliberately malformed as a shared-string reference. The parser must
    # never attempt to resolve it because R is a Mixscape field, outside A:L.
    vals={c:'1' for c in ALLOWED}
    vals.update(A='A1_AAACCCAAGAAACACT-1',B='untreated',C='NonTargeting',J='A1',L='CTRL-1')
    row=''.join(f'<c r="{col}2" t="inlineStr"><is><t>{val}</t></is></c>' for col,val in vals.items())
    path=tmp_path/'metadata.xlsx'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="single_cell_annotations" r:id="rId2"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships><Relationship Id="rId2" Target="worksheets/sheet2.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet2.xml',f'<worksheet xmlns="{ns}"><sheetData><row r="1">{header}</row><row r="2">{row}<c r="R2" t="s"><v>999999999</v></c></row></sheetData></worksheet>')
        z.writestr('xl/worksheets/sheet3.xml','NOT XML AND MUST NOT BE OPENED')
    frame=technical_annotations(path)
    assert frame.shape==(1,12)
    assert frame.iloc[0]['Inferred knockout']=='NonTargeting'
    assert not any('Mixscape' in c for c in frame.columns)
