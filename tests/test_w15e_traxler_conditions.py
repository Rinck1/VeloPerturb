import gzip
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w15e_traxler_protein_conditions import select_mouse_proteins


def test_mouse_exact_symbols_and_deterministic_longest_isoform(tmp_path):
    path=tmp_path/'mouse.fa.gz'
    with gzip.open(path,'wt') as stream:
        stream.write('>P2 pep gene:G1 transcript:T2 gene_symbol:Stat1\nACD\n'
                     '>P1 pep gene:G1 transcript:T1 gene_symbol:Stat1\nACD\n'
                     '>P3 pep gene:G1 transcript:T3 gene_symbol:Stat1\nAC\n'
                     '>P4 pep gene:G2 transcript:T4 gene_symbol:Rela\nACDE*\n')
    records=select_mouse_proteins(path,{'Stat1','Rela'})
    assert [r['condition'] for r in records]==['Rela','Stat1']
    assert records[1]['protein_id']=='P1'
    assert records[0]['sequence']=='ACDE'
    with pytest.raises(ValueError,match='Missing exact'):
        select_mouse_proteins(path,{'missing'})
