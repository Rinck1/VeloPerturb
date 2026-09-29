"""Launcher-compatible wrapper for the minimal MeRLin Day0 shell pipeline."""
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[2]
script = ROOT / "scripts" / "phase0" / "acquire_merlin_day0_minimal.sh"
raise SystemExit(subprocess.run(["bash", str(script)], cwd=ROOT).returncode)
