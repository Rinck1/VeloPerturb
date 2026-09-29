#!/usr/bin/env python3
"""Quantify the MeRLin in-vivo Day21 replicate SRR33960309.

This is a data-preparation job only.  It does not build a fold, fit a
transform, read Day21 expression as a source feature, or open confirmation.
The chemistry and whitelist rule match the validated SRR33960308 command.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
from pathlib import Path


def build_whitelist(r1: Path, output: Path, minimum_reads: int) -> int:
    counts: dict[str, int] = {}
    with gzip.open(r1, "rt") as stream:
        for line_number, line in enumerate(stream):
            if line_number % 4 != 1:
                continue
            barcode = line.strip()[:16]
            if len(barcode) == 16 and set(barcode) <= set("ACGT"):
                counts[barcode] = counts.get(barcode, 0) + 1
    barcodes = sorted((count, barcode) for barcode, count in counts.items() if count >= minimum_reads)[::-1]
    output.write_text("\n".join(barcode for _, barcode in barcodes) + "\n")
    return len(barcodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/yuchang/veloroute_ucheck_20260915"))
    parser.add_argument("--repo", type=Path, default=Path("/home/yuchang/wangjiaxuan"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--minimum-whitelist-reads", type=int, default=10)
    args = parser.parse_args()
    run = "SRR33960309"
    r1 = args.root / "full" / f"{run}_1.fastq.gz"
    r2 = args.root / "full" / f"{run}_2.fastq.gz"
    qout = args.root / "quant_merlin309"
    whitelist = args.root / "merlin309_bcs.txt"
    index = Path("/data/yuchang/veloroute_kang_20260914/reference/splici_r98_v2/index")
    tool = args.repo / "tools/rna-quant/bin/simpleaf"
    args.output.mkdir(parents=True, exist_ok=False)
    if not r1.is_file() or not r2.is_file():
        raise FileNotFoundError("SRR33960309 FASTQ pair is incomplete")
    if qout.exists():
        raise FileExistsError(f"Refusing to overwrite existing quantification directory: {qout}")
    if whitelist.exists():
        raise FileExistsError(f"Refusing to overwrite existing whitelist: {whitelist}")

    n_barcodes = build_whitelist(r1, whitelist, args.minimum_whitelist_reads)
    command = [str(tool), "quant", "--index", str(index), "--chemistry", "1{b[16]u[10]}2{r:}",
               "--expected-ori", "fw", "--reads1", str(r1), "--reads2", str(r2),
               "--explicit-pl", str(whitelist), "--resolution", "cr-like",
               "--threads", str(args.threads), "--output", str(qout)]
    environment = os.environ.copy()
    environment.update({"ALEVIN_FRY_HOME": str(args.repo / "tools/af-home"),
                        "TMPDIR": str(args.root / "tmp")})
    (args.output / "command.json").write_text(json.dumps({
        "run": run, "command": command, "whitelist_barcodes": n_barcodes,
        "minimum_whitelist_reads": args.minimum_whitelist_reads,
        "fastq_sizes": {"r1": r1.stat().st_size, "r2": r2.stat().st_size},
        "future_target_read": False, "confirmation_opened": False,
    }, indent=2) + "\n")
    args.root.joinpath("tmp").mkdir(parents=True, exist_ok=True)
    with (args.output / "simpleaf.log").open("w") as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    summary = {
        "status": "MERLIN_DAY21_309_QUANT_COMPLETE" if result.returncode == 0 else "MERLIN_DAY21_309_QUANT_FAILED",
        "run": run, "returncode": result.returncode, "quant_output": str(qout),
        "whitelist": str(whitelist), "whitelist_barcodes": n_barcodes,
        "chemistry": "1{b[16]u[10]}2{r:}", "expected_orientation": "fw",
        "future_target_read": False, "confirmation_opened": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "RESULTS.md").write_text("# MeRLin Day21 SRR33960309 quantification\n\n" + json.dumps(summary, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
