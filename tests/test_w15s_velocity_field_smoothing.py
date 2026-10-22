from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15s_velocity_field_smoothing as module
from test_w15n_velocity_auxiliary import fixture


def test_field_teacher_smooths_vectors_not_unit_directions_and_keeps_input():
    g,c,_=fixture();rng=np.random.default_rng(1)
    v=rng.normal(size=(7,3));u=rng.normal(size=(7,3))
    source=dict(cell_ids=g['cell_ids'],conditions=np.array(['A']*7),velocity=v,u_features=u)
    T=module.expansion.source_operator(g['z'],source['conditions'],3,.5)
    norm=dict(axes=np.eye(3)[:2],velocity_reference=np.zeros(3),U_reference=np.zeros(3))
    groups,reference=module.field_teacher([g],source,norm,T)
    np.testing.assert_allclose(groups[0]['velocity'],T@v,rtol=1e-6)
    np.testing.assert_allclose(groups[0]['U'],T@u,rtol=1e-6)
    np.testing.assert_array_equal(groups[0]['U_normalized'],g['U_normalized'])
    np.testing.assert_array_equal(groups[0]['z'],g['z'])
    np.testing.assert_allclose(reference['velocity_reference'],(T@v).mean(0))


def test_field_grid_rejected_before_target_read(tmp_path,monkeypatch):
    import w15h_traxler_program_router as base
    monkeypatch.setattr(base,'load_pack',lambda *a,**kw:pytest.fail('Premature future target'))
    with pytest.raises(ValueError,match='complete field and reference grid'):
        module.evaluate(tmp_path,{},dict(seeds=[0]),dict(val=['A']),[])


def test_field_configuration_fixed_small_and_uses_approved_fourth_card():
    import json
    p=Path(__file__).resolve().parents[1]/'configs/w15s_velocity_field_smoothing_20261022_v1.json'
    c=json.loads(p.read_text());case=c['cases'][0]
    assert case['name']==module.CASE and case['gpu']==4
    assert case['graph_weight_normalized']==.1 and c['steps']==5000
    assert len(module.expansion.ARMS)*len(c['model_seeds'])==18
