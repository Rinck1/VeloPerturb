"""Bounded raw tasks and FASTQ layout inspection; never infer chemistry from file size."""
from __future__ import annotations

import gzip
import itertools
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

from .artifacts import Run, save_csv, save_json, utc_now


def plan_raw(manifest, day, sra_bin, raw_root):
    selected = [r for r in manifest if int(r["day"]) == day]
    if len(selected) != 4 or Counter(r["library_type"] for r in selected) != {"mRNA": 2, "gRNA": 2}:
        raise ValueError("Need exactly two GEX and two guide runs for the selected timepoint")
    root, tools = Path(raw_root).resolve(), Path(sra_bin).resolve()
    tasks = []
    for row in sorted(selected, key=lambda r: (r["library_type"], r["run_accession"])):
        accession = row["run_accession"]
        if not re.fullmatch(r"SRR\d+", accession):
            raise ValueError("Invalid accession")
        tasks.append({"run": accession, "day": day, "library_type": row["library_type"],
                      "prefetch": [str(tools/"prefetch"), accession, "--max-size", "12G", "--verify", "yes",
                                   "--output-directory", str(root/"sra")],
                      "validate": [str(tools/"vdb-validate"), str(root/"sra"/accession)],
                      "extract": [str(tools/"fasterq-dump"), str(root/"sra"/accession), "--split-files",
                                  "--include-technical", "--threads", "8", "--outdir", str(root/"fastq"/accession),
                                  "--temp", str(root/"scratch"/accession)],
                      "usa_quantification": "pending_reference_and_chemistry_approval" if row["library_type"] == "mRNA"
                                            else "not_GEX_never_send_to_USA"})
    return tasks


def run_command(command, output, *, timeout_seconds=3600, inputs=(), environment=None, working_directory=None):
    if not command or not Path(command[0]).is_file() or timeout_seconds <= 0:
        raise ValueError("Need an existing executable and bounded timeout")
    config = {"argv": command, "timeout_seconds": timeout_seconds,
              "explicit_environment": environment or {}, "shell": False,
              "working_directory": str(working_directory) if working_directory else None}
    with Run(output, stage="raw_command", kind="engineering", config=config, inputs=inputs) as run:
        state = {"status": "starting", "started_utc": utc_now(), "argv": command}
        save_json(run.directory/"status.json", state)
        with (run.directory/"stdout.log").open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, **(environment or {})}, cwd=working_directory)
            state.update(pid=process.pid, status="running")
            save_json(run.directory/"status.json", state)
            try:
                result = process.wait(timeout=timeout_seconds)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                state.update(status="interrupted_or_timeout", finished_utc=utc_now())
                save_json(run.directory/"status.json", state)
                raise
        state.update(status="complete" if result == 0 else "failed", returncode=result, finished_utc=utc_now())
        save_json(run.directory/"status.json", state)
        (run.directory/"RESULTS.md").write_text(f"# Raw command\n\nStatus: {state['status']}; exit code: {result}.\n\nSee stdout.log and provenance.json. No S/U or model success is inferred from command completion.\n")
        if result:
            raise RuntimeError(f"Raw command failed with exit code {result}")
    return state


def _fastq_records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        while True:
            header = stream.readline()
            if not header:
                return
            sequence, plus, quality = stream.readline().rstrip(), stream.readline(), stream.readline().rstrip()
            if not header.startswith("@") or not plus.startswith("+") or not sequence or len(sequence) != len(quality):
                raise ValueError(f"Malformed/truncated FASTQ record in {path}")
            token = header.split()[0][1:]
            # SRA default uses the same spot identifier in both split reads; /1,/2 are also common.
            token = re.sub(r"/[12]$", "", token)
            yield token, sequence


def audit_fastq_pair(r1, r2, *, whitelist=(), max_records=100000):
    if max_records < 0:
        raise ValueError("max_records must be nonnegative; zero scans all records")
    reference = {barcode.split("-")[0] for barcode in whitelist}
    lengths = [Counter(), Counter()]
    leading_n, hits, n = 0, {0: 0, 1: 0}, 0
    pairs = itertools.zip_longest(_fastq_records(r1), _fastq_records(r2))
    if max_records:
        pairs = itertools.islice(pairs, max_records)
    for a, b in pairs:
        if a is None or b is None or a[0] != b[0]:
            raise ValueError("Split FASTQs have unequal lengths or mismatched spot identifiers")
        n += 1
        lengths[0][len(a[1])] += 1
        lengths[1][len(b[1])] += 1
        leading_n += a[1][0] == "N"
        for offset in hits:
            hits[offset] += a[1][offset:offset+16] in reference
    if not n:
        raise ValueError("Empty FASTQ pair")
    return {"records_examined": n, "integrity_scope": "full_files" if max_records == 0 else "bounded_prefix_only",
            "r1_length_histogram": dict(lengths[0]), "r2_length_histogram": dict(lengths[1]),
            "r1_leading_N_fraction": leading_n/n,
            "published_barcode_match_fraction_by_zero_based_offset": {str(k): v/n for k, v in hits.items()},
            "whitelist_kind": "published_observed_barcodes_not_full_10x_permit_list",
            "chemistry_approved": False, "umi_layout": "not_inferred_from_read_length_alone"}
