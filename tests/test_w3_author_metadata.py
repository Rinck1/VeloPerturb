import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/phase0'))
from w3_6b_fetch_author_metadata import metadata_slot, unwrap


def test_metadata_slot_never_converts_expression_attributes():
    def obj(kind, value=None, tag=None, attributes=None):
        return SimpleNamespace(info=SimpleNamespace(type=kind), value=value, tag=tag,
                               attributes=attributes, referenced_object=None)
    end = obj('NIL')
    metadata = obj('DATAFRAME', 'expected_metadata')
    meta_node = obj('LIST', (metadata, end), obj('SYMBOL', 'meta.data'))
    attributes = obj('LIST', (obj('ASSAYS'), meta_node), obj('SYMBOL', 'assays'))
    parsed = SimpleNamespace(object=obj('S4', attributes=attributes), extra=SimpleNamespace(encoding='utf-8'))
    converted = []
    def convert(value, **kwargs):
        assert value.info.type != 'ASSAYS'
        converted.append(value.info.type)
        return value.value
    rdata = SimpleNamespace(parser=SimpleNamespace(RObjectType=SimpleNamespace(LIST='LIST')),
                            conversion=SimpleNamespace(convert=convert))
    assert metadata_slot(parsed, rdata) == 'expected_metadata'
    assert converted == ['SYMBOL', 'SYMBOL', 'DATAFRAME']


def test_cyclic_reference_fails_closed():
    obj = SimpleNamespace(referenced_object=None)
    obj.referenced_object = obj
    with pytest.raises(ValueError, match='Cyclic'):
        unwrap(obj)
