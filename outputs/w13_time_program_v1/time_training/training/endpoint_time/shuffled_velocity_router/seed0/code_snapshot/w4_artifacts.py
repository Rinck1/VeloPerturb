"""Small W4 provenance helpers; every JSON carries the execution context."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class Artifacts:
    def __init__(self, output, *, seed, inputs):
        self.directory = Path(output)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.context = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            seed=seed, command=sys.argv, data_hashes={str(p): sha256(p) for p in inputs},
            python=sys.version, worktree_diff_sha256=hashlib.sha256(
                subprocess.check_output(['git', 'diff', '--binary'])).hexdigest())
        snapshot = self.directory/'code_snapshot'
        snapshot.mkdir()
        for path in inputs:
            path = Path(path)
            if path.suffix in ('.py', '.yaml', '.json', '.md'):
                destination = snapshot/path.name
                if destination.exists():
                    raise ValueError('Ambiguous snapshot file names')
                shutil.copyfile(path, destination)
        self.json('status.json', dict(status='running'))

    def json(self, name, value):
        path = self.directory/name
        payload = dict(self.context, runtime_seconds=time.monotonic()-self.started, **value)
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)

