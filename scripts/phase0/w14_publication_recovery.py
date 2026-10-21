"""Publish explicit small result files without confusing publication with computation."""
from __future__ import annotations

from pathlib import Path
import subprocess


def publish_records(root, paths, message, *, push=True):
    root = Path(root).resolve()
    selected = []
    for value in paths:
        p = Path(value)
        p = (root / p).resolve() if not p.is_absolute() else p.resolve()
        if not p.is_relative_to(root):
            raise ValueError('Publication outside repository refused')
        relative = p.relative_to(root)
        if not (relative.parts[0] == 'outputs' or relative.parts[:2] == ('docs', 'current')):
            raise ValueError('Only explicit result records may be published')
        if p.suffix not in ('.json', '.csv', '.md') or p.stat().st_size > 8 * 1024**2:
            raise ValueError('Only small text result records may be published')
        if '\0' in p.read_text():
            raise ValueError('Binary artifact refused')
        selected.append(str(relative))
    selected = sorted(set(selected))
    if not selected:
        raise ValueError('Explicit publication paths required')
    # outputs/ is intentionally ignored. Force ONLY these reviewed text files,
    # never the directory, arrays, model weights, or another staged change.
    subprocess.run(['git', 'add', '-f', '--', *selected], cwd=root, check=True)
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet', '--', *selected], cwd=root)
    if changed.returncode not in (0, 1):
        raise RuntimeError('Cannot inspect selected publication diff')
    if changed.returncode:
        subprocess.run(['git', 'commit', '--only', '-m', message, '--', *selected], cwd=root, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if push:
        subprocess.run(['git', 'push', 'origin', 'master'], cwd=root, check=True)
    return dict(status='published' if push else 'committed', commit=commit, files=selected)
