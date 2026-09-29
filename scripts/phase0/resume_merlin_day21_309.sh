#!/usr/bin/env bash
# Resume the locally complete SRR33960309 parts without deleting any source.
set -euo pipefail
ROOT=/data/yuchang/veloroute_ucheck_20260915
RUN=SRR33960309
FULL="$ROOT/full"
PARTS="$FULL/parts"
SRA="$FULL/sra"
TOOLKIT=/home/yuchang/wangjiaxuan/tools/sratoolkit.3.4.1-ubuntu64/bin
OUT=/data/yuchang/veloroute_merlin_day21_309_resume_20260929
mkdir -p "$OUT" "$SRA" "$ROOT/tmp"
exec > >(tee -a "$OUT/run.log") 2>&1

target="$SRA/$RUN.sra"
if [[ ! -f "$target" ]]; then
  shopt -s nullglob
  parts=("$PARTS/$RUN.sra.part"*)
  if (( ${#parts[@]} != 20 )); then
    echo "expected 20 parts, found ${#parts[@]}" >&2
    exit 2
  fi
  tmp="$target.incomplete"
  rm -f "$tmp"
  for part in $(printf '%s\n' "${parts[@]}" | sort -V); do cat "$part" >> "$tmp"; done
  mv "$tmp" "$target"
fi
"$TOOLKIT/vdb-validate" "$target" > "$OUT/vdb_validate.log" 2>&1
outdir="$ROOT/raw/$RUN"
mkdir -p "$outdir"
"$TOOLKIT/fasterq-dump" --split-files --include-technical --threads 8 --temp "$ROOT/tmp/$RUN" --outdir "$outdir" "$target" > "$OUT/fasterq.log" 2>&1
rm -rf "$ROOT/tmp/$RUN"
for fq in "$outdir"/*.fastq; do pigz -p 32 "$fq"; done
mv "$outdir"/*.fastq.gz "$FULL"/
rmdir "$outdir" 2>/dev/null || true
python - <<'PY'
import json
from pathlib import Path
root=Path('/data/yuchang/veloroute_ucheck_20260915/full')
files=sorted(root.glob('SRR33960309_*.fastq.gz'))
result={'status':'MERLIN_DAY21_309_FASTQ_COMPLETE','run':'SRR33960309',
        'fastq_files':[str(p) for p in files],
        'fastq_bytes':{str(p):p.stat().st_size for p in files}}
Path('/data/yuchang/veloroute_merlin_day21_309_resume_20260929/summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
PY
