"""After the first GEX SRA arrives, inspect a few read-layout rows, not biology."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from w4_artifacts import Artifacts,sha256


ROOT=Path(__file__).resolve().parents[2]


def run(manifest,output):
    manifest=Path(manifest);m=json.loads(manifest.read_text())
    run=Artifacts(output,seed=0,inputs=[manifest,Path(__file__)])
    data=Path('/data/yuchang/veloroute_w4/traxler2025/sra');deadline=time.monotonic()+2*86400
    while time.monotonic()<deadline:
        complete=[(o,data/(o['accession']+'.sra')) for o in m['objects']
            if (data/(o['accession']+'.sra')).is_file() and (data/(o['accession']+'.sra')).stat().st_size==o['bytes']]
        if complete:break
        run.json('status.json',dict(status='waiting_for_first_complete_SRA'));time.sleep(30)
    else:raise TimeoutError('No complete SRA within two days')
    obj,path=complete[0]
    command=[str(ROOT/'tools/sratoolkit.3.4.1-ubuntu64/bin/vdb-dump'),str(path),'-T','SEQUENCE',
        '-R','1-10','-C','READ_START,READ_LEN,READ_TYPE','-f','json']
    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
    (run.directory/'read_layout.txt').write_text(result.stdout)
    (run.directory/'read_layout_stderr.txt').write_text(result.stderr)
    if result.returncode:raise RuntimeError('SRA read-layout inspection failed; raw retained')
    run.json('summary.json',dict(status='TRAXLER_SRA_LAYOUT_OBSERVED',run=obj['accession'],sra_sha256=sha256(path),
        command=command,rows=10,biological_expression_read=False,barcode_layout_approved=False,
        next='Review read-length/type table before configuring 10x5prime-v2 extraction; do not assume R1 cycles'))
    run.json('status.json',dict(status='complete'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='outputs/w14b_traxler_raw_v1/download_manifest.json')
    p.add_argument('--output',required=True);a=p.parse_args();run(a.manifest,a.output)
