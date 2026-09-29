"""Immutable run directories and hash-bound records; never overwrite another run."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_config(path):
    with Path(path).open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("Config must be a mapping")
    return value


def save_json(path, value):
    """Atomic replacement is only for state files inside this run's own directory."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows and not fields:
        raise ValueError("Empty CSV requires a schema")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Run:
    def __init__(self, directory, *, stage, kind, config, inputs=(), seed=None):
        if kind not in {"engineering", "synthetic", "development", "confirmation"}:
            raise ValueError("Unknown run kind")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        package_root = Path(__file__).parent
        self.provenance = {
            "started_utc": utc_now(), "status": "running", "stage": stage,
            "kind": kind, "seed": seed, "command": sys.argv,
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES'),
            "cuda_device_order": os.environ.get('CUDA_DEVICE_ORDER'),
            "config_hash": object_hash(config),
            "inputs": [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in inputs],
            "code": {p.name: sha256(p) for p in sorted(package_root.glob("*.py"))},
            "packages": {},
        }
        for name in ("numpy", "scipy", "PyYAML", "torch", "veloroute", "anndata", "pandas", "h5py", "scikit-learn", "fair-esm"):
            try:
                self.provenance["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                self.provenance["packages"][name] = None
        (self.directory / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=True))
        snapshot = self.directory / 'code_snapshot'
        snapshot.mkdir()
        for source in sorted(package_root.glob('*.py')):
            (snapshot / source.name).write_bytes(source.read_bytes())
        save_json(self.directory / "provenance.json", self.provenance)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.provenance.update(status="failed" if exc else "complete", finished_utc=utc_now())
        if exc:
            self.provenance["error"] = f"{exc_type.__name__}: {exc}"
        self.provenance["outputs"] = {
            str(p.relative_to(self.directory)): sha256(p)
            for p in sorted(self.directory.rglob("*"))
            if p.is_file() and p != self.directory / "provenance.json"
        }
        save_json(self.directory / "provenance.json", self.provenance)
        return False
