"""Resumeable processing for only MeRLin Day0 runs 310 and 311.

Each run is independently barcode-extracted and quantified with the validated
10x 5' USA chemistry. Existing outputs are never overwritten.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/yuchang/veloroute_ucheck_20260915")
REPO = Path("/home/yuchang/wangjiaxuan")
TOOL = REPO / "tools/rna-quant/bin/simpleaf"
INDEX = Path("/data/yuchang/veloroute_kang_20260914/reference/splici_r98_v2/index")
OUTROOT = Path("/data/yuchang/veloroute_merlin_day0_minimal_20260923")
RUNS = ("SRR33960310", "SRR33960311")


def wait_for_fastq(run):
    r1, r2 = ROOT / "full" / f"{run}_1.fastq.gz", ROOT / "full" / f"{run}_2.fastq.gz"
    while not (r1.exists() and r2.exists()):
        time.sleep(60)
    return r1, r2


def whitelist(r1, path):
    counts = {}
    with gzip.open(r1, "rt") as handle:
        for n, line in enumerate(handle):
            if n % 4 == 1:
                bc = line.strip()[:16]
                if len(bc) == 16 and set(bc) <= set("ACGT"):
                    counts[bc] = counts.get(bc, 0) + 1
    values = sorted((n, bc) for bc, n in counts.items() if n >= 10)[::-1]
    path.write_text("\n".join(bc for _, bc in values) + "\n")
    return len(values)


def run_one(run):
    r1, r2 = wait_for_fastq(run)
    clone_out = ROOT / "clones" / run
    clone_summary = clone_out / "summary.json"
    if not clone_summary.exists():
        clone_out.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(REPO / ".venv-gpu/bin/python"), str(REPO / "scripts/phase0/merlin_extract_barcodes.py"),
                        "--r1", str(r1), "--r2", str(r2), "--output-dir", str(clone_out), "--threads", "48"], check=True)
    qout = ROOT / f"quant_merlin{run[-3:]}"
    qsummary = qout / "summary.json"
    if not qsummary.exists():
        qout.parent.mkdir(parents=True, exist_ok=True)
        bcs = OUTROOT / f"{run}_bcs.txt"
        nbc = whitelist(r1, bcs)
        cmd = [str(TOOL), "quant", "--index", str(INDEX), "--chemistry", "1{b[16]u[10]}2{r:}",
               "--expected-ori", "fw", "--reads1", str(r1), "--reads2", str(r2),
               "--explicit-pl", str(bcs), "--resolution", "cr-like", "--threads", "24", "--output", str(qout)]
        env = dict(**__import__("os").environ, ALEVIN_FRY_HOME=str(REPO / "tools/af-home"), TMPDIR=str(ROOT / "tmp"))
        (OUTROOT / f"quant_{run}.log").write_text(json.dumps({"run": run, "barcodes": nbc, "cmd": cmd}, indent=2) + "\n")
        with (OUTROOT / f"quant_{run}.log").open("a") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {"run": run, "clone_summary": str(clone_summary), "quant_summary": str(qsummary)}


def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "runs": list(RUNS)}
    (OUTROOT / "processing_status.json").write_text(json.dumps(result, indent=2))
    result["outputs"] = [run_one(run) for run in RUNS]
    result["status"] = "MERLIN_DAY0_PROCESSING_COMPLETE"
    (OUTROOT / "processing_summary.json").write_text(json.dumps(result, indent=2))
    (OUTROOT / "PROCESSING_RESULTS.md").write_text("# MeRLin Day0 minimal processing\n\n" + json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
