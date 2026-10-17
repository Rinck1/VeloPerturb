import sys
import json
from pathlib import Path
import subprocess

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_make_lock import build_lock
from w4_artifacts import sha256
from w4_replication import verify_lock


def test_draft_protocol_is_rejected_before_any_QC_or_count_access(tmp_path):
    repo = tmp_path/'repo'
    (repo/'PLANS').mkdir(parents=True)
    protocol = repo/'PLANS/w4_shorttime_prereg.md'
    protocol.write_text('状态：**草案**\n')
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'add','PLANS/w4_shorttime_prereg.md'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=W4 fixture','-c','user.email=w4@example.invalid',
        'commit','-qm','draft fixture'],check=True)
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    with pytest.raises(ValueError, match='still DRAFT'):
        build_lock(commit, tmp_path/'missing_qc.json', tmp_path/'missing_counts.npz', repo=repo)
    with pytest.raises(ValueError, match='40-character'):
        build_lock('HEAD', tmp_path/'missing_qc.json', tmp_path/'missing_counts.npz', repo=repo)


def test_frozen_lock_matches_replication_validator_and_detects_data_changes(tmp_path):
    repo = tmp_path/'repo'
    (repo/'PLANS').mkdir(parents=True)
    protocol = repo/'PLANS/w4_shorttime_prereg.md'
    protocol.write_text('状态：**已冻结**\nFixture tests hash contracts only; no expression is decoded.\n')
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'add','PLANS/w4_shorttime_prereg.md'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=W4 fixture','-c','user.email=w4@example.invalid',
        'commit','-qm','frozen fixture'],check=True)
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    hashes = {}
    for day in (2,3,4):
        file = repo/f'day{day}.h5ad'
        file.write_bytes(f'hash-only fixture day{day}'.encode())
        hashes[str(file)] = sha256(file)
    embedding = repo/'embeddings.npz'
    embedding.write_bytes(b'hash-only frozen embeddings')
    qc = repo/'qc.json'
    qc.write_text(json.dumps(dict(status='TECHNICAL_COHORT_QC_COMPLETE', biological_effects_read=False,
        minimum_TF_cells_per_day=50, H1_TFs=['A','B','C','D','E'], data_hashes=hashes)))
    lock = build_lock(commit, qc, embedding, repo=repo)
    target = repo/'protocol_lock.json'
    target.write_text(json.dumps(lock))
    verified = verify_lock(target, repo=repo)
    assert verified['lock']['H1_TFs'] == ['A','B','C','D','E']
    assert verified['lock']['freeze_commit'] == commit
    embedding.write_bytes(b'changed after freeze')
    with pytest.raises(ValueError, match='Locked data changed: embeddings'):
        verify_lock(target, repo=repo)
