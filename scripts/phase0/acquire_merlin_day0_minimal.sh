#!/usr/bin/env bash
# Minimal MeRLin acquisition: only the two in-vivo Day0 runs needed for the
# decisive Day0 -> Day21 clone/persister-program test. Existing range parts are
# never deleted; release an SRA only after vdb-validate passes.
set -euo pipefail
ROOT=/data/yuchang/veloroute_ucheck_20260915
FULL="$ROOT/full"
PARTS="$FULL/parts"
SRA="$FULL/sra"
TOOLKIT=/home/yuchang/wangjiaxuan/tools/sratoolkit.3.4.1-ubuntu64/bin
OUT=/data/yuchang/veloroute_merlin_day0_minimal_20260923
mkdir -p "$OUT" "$SRA" "$ROOT/tmp"
exec > >(tee -a "$OUT/acquisition.log") 2>&1

for run in SRR33960310 SRR33960311; do
  target="$SRA/$run.sra"
  if [[ -f "$FULL/${run}_1.fastq.gz" && -f "$FULL/${run}_2.fastq.gz" ]]; then
    echo "[$run] FASTQ already present; skip"
    continue
  fi
  if [[ ! -f "$target" ]]; then
    echo "[$run] assembling existing parts"
    tmp="$target.incomplete"
    rm -f "$tmp"
    shopt -s nullglob
    parts=("$PARTS/$run.sra.part"*)
    if (( ${#parts[@]} != 20 )); then
      echo "[$run] expected 20 parts, found ${#parts[@]}" >&2
      exit 2
    fi
    for part in $(printf '%s\n' "${parts[@]}" | sort -V); do cat "$part" >> "$tmp"; done
    mv "$tmp" "$target"
  fi
  echo "[$run] validating $(stat -c%s "$target") bytes"
  if ! "$TOOLKIT/vdb-validate" "$target" > "$OUT/vdb_$run.log" 2>&1; then
    echo "[$run] validation failed; preserving SRA and stopping" >&2
    exit 3
  fi
  echo "[$run] validation passed; converting"
  outdir="$ROOT/raw/$run"
  mkdir -p "$outdir"
  "$TOOLKIT/fasterq-dump" --split-files --include-technical --threads 8 --temp "$ROOT/tmp/$run" --outdir "$outdir" "$target" > "$OUT/fasterq_$run.log" 2>&1
  rm -rf "$ROOT/tmp/$run"
  for fq in "$outdir"/*.fastq; do pigz -p 32 "$fq"; done
  mv "$outdir"/*.fastq.gz "$FULL"/
  rm -rf "$outdir"
  echo "[$run] FASTQ complete"
done

python - <<'PY'
import json
from pathlib import Path
root=Path('/data/yuchang/veloroute_ucheck_20260915/full')
runs=['SRR33960310','SRR33960311']
files=[]
for r in runs:
    files += [str(p) for p in sorted(root.glob(r+'_*.fastq.gz'))]
result={'status':'MERLIN_DAY0_MINIMAL_ACQUISITION_COMPLETE' if all(files) else 'INCOMPLETE',
        'runs':runs,'fastq_files':files,'fastq_bytes':{p:Path(p).stat().st_size for p in files}}
Path('/data/yuchang/veloroute_merlin_day0_minimal_20260923/summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
PY
