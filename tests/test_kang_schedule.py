from pathlib import Path

import pytest

from veloroute.artifacts import load_config
from veloroute.kang_schedule import MAINLINE,full_dependencies,full_variants,worker_stages


def test_full_mainline_precedes_controls_and_external_baselines_are_absent():
    for worker in range(4):
        stages=worker_stages(worker)
        assert [s[0] for s in stages]==['mainline','router_controls','full_ablations']
        assert stages[0][2][-2:]==['--phase','mainline']
        assert not any(b in str(stages).lower() for b in ('cellot','cellflow','scdfm'))
        assert all(s[2][:2]==['--worker',str(worker)] for s in stages)


@pytest.mark.parametrize('worker',[-1,4,7])
def test_reserved_or_unknown_gpu_is_rejected(worker):
    with pytest.raises(ValueError):worker_stages(worker)


def test_mainline_needs_real_data_not_completed_controls():
    root=Path('/data/example')
    assert full_dependencies(root,0,'mainline')==[root/'folds/ready.json']
    assert full_dependencies(root,0,'followups')==[root/'mainline_worker0_complete.json',root/'primary_worker0_complete.json']


def test_reordering_keeps_every_full_variant_exactly_once():
    config=load_config('configs/veloroute_kang_extended_20260914.yaml')
    main=full_variants(config,'mainline');rest=full_variants(config,'followups')
    assert [v['name'] for v in main]==[MAINLINE]
    assert main+rest==config['full_variants']
    assert len(main+rest)==7
    assert full_variants(config,'all')==config['full_variants']


def test_invalid_stage_or_duplicate_mainline_fails_closed():
    with pytest.raises(ValueError):full_dependencies('/data/example',0,'baseline')
    with pytest.raises(ValueError):full_variants({'full_variants':[{'name':MAINLINE}]*2},'mainline')
