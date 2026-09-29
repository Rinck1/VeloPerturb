import pytest

from veloroute.protein_conditions import select_proteins


def test_longest_protein_is_deterministic_and_fails_missing_gene(tmp_path):
    path = tmp_path/'proteins.fa'
    path.write_text('>P2|T2|G1|x|y|gene-2|GENE|3|\nMAA\n'
                    '>P1|T1|G1|x|y|gene-1|GENE|3|\nMGG\n'
                    '>P0|T0|G1|x|y|gene-0|GENE|2|\nMA\n')
    selected = select_proteins(path, {'GENE'})
    assert selected[0]['protein_id'] == 'P1'
    assert selected[0]['sequence'] == 'MGG'
    with pytest.raises(ValueError, match='No reference protein'):
        select_proteins(path, {'MISSING'})
