import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/phase0'))
from w14_publication_recovery import publish_records
from w14_recover_execution import scientific_completion


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def test_publish_explicit_ignored_records_preserves_unrelated_staged_changes(tmp_path):
    git(tmp_path, 'init', '-b', 'master')
    git(tmp_path, 'config', 'user.name', 'Test')
    git(tmp_path, 'config', 'user.email', 'test@example.invalid')
    (tmp_path / '.gitignore').write_text('outputs/\n')
    git(tmp_path, 'add', '.gitignore')
    git(tmp_path, 'commit', '-m', 'initial')
    (tmp_path / 'unrelated.txt').write_text('User work remains staged')
    git(tmp_path, 'add', 'unrelated.txt')
    out = tmp_path / 'outputs' / 'result'
    out.mkdir(parents=True)
    (out / 'summary.json').write_text('{"status":"complete"}')
    (out / 'model.pt').write_bytes(b'private weight')
    result = publish_records(tmp_path, [out / 'summary.json'], 'publish', push=False)
    assert result['status'] == 'committed'
    assert git(tmp_path, 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD') == 'outputs/result/summary.json'
    assert git(tmp_path, 'diff', '--cached', '--name-only') == 'unrelated.txt'
    assert 'model.pt' not in git(tmp_path, 'ls-files')


def test_publication_rejects_unrelated_paths(tmp_path):
    p = tmp_path / 'secret.json'
    p.write_text('{}')
    with pytest.raises(ValueError, match='explicit result'):
        publish_records(tmp_path, [p], 'no', push=False)


def test_scientific_completion_does_not_depend_on_git_status(tmp_path):
    (tmp_path / 'status.json').write_text(json.dumps(dict(status='complete')))
    (tmp_path / 'summary.json').write_text(json.dumps(dict(status='W14_GRID_EVALUATED', P1=True, P2=False)))
    for name in ('metrics.csv', 'comparisons.csv', 'frozen_grid.json', 'RESULTS.md'):
        (tmp_path / name).write_text('retained')
    assert scientific_completion(tmp_path)['P2'] is False
    (tmp_path / 'status.json').write_text(json.dumps(dict(status='failed')))
    with pytest.raises(ValueError, match='Complete scientific'):
        scientific_completion(tmp_path)
