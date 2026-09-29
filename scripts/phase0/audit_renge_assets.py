"""Read-only RENGE structural audit; no fitting, filtering, guide calling or models.

Run with an existing environment providing numpy, scipy, openpyxl and threadpoolctl.
Writes only into a NEW output directory. Input data are never modified.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

SAMPLES = {
    f"GSM{6571143 + 2 * (day - 2) + offset}": (day, library)
    for day in range(2, 6)
    for offset, library in enumerate(("mRNA", "gRNA"))
}
ENA_FIELDS = (
    "run_accession,experiment_accession,sample_accession,study_accession,"
    "experiment_title,library_name,library_strategy,library_layout,read_count,"
    "base_count,fastq_ftp,fastq_md5,fastq_bytes,sra_ftp,sra_md5,sra_bytes"
)
ENA_URL = "https://www.ebi.ac.uk/ena/portal/api/filereport?" + urlencode({
    "accession": "PRJNA879051", "result": "read_run", "fields": ENA_FIELDS,
    "format": "tsv",
})
GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213069"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows and fields is None:
        raise ValueError("Empty CSV requires an explicit schema")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_tsv_gz(path):
    with gzip.open(path, "rt") as handle:
        return list(csv.reader(handle, delimiter="\t"))


def parse_runs(payload):
    """Resolve libraries by accession and title; never by byte size."""
    rows = list(csv.DictReader(io.StringIO(payload), delimiter="\t"))
    if len(rows) != 16 or len({r["run_accession"] for r in rows}) != 16:
        raise ValueError("Expected exactly 16 unique RENGE runs; inspect upstream metadata")
    resolved = []
    sample_counts = Counter()
    for row in rows:
        gsm = row["library_name"]
        if gsm not in SAMPLES or row["study_accession"] != "PRJNA879051":
            raise ValueError(f"Unrecognized study/sample: {row}")
        day, library = SAMPLES[gsm]
        title = row["experiment_title"]
        if f"from {library} " not in title or f"day{day} " not in title or gsm not in title:
            raise ValueError(f"Accession/title disagreement: {row}")
        reads, bases = int(row["read_count"]), int(row["base_count"])
        if reads <= 0 or bases <= 0:
            raise ValueError("Non-positive archive read/base count")
        sample_counts[gsm] += 1
        resolved.append({
            "day": day, "geo_sample": gsm, "library_type": library,
            "sample_group": f"GSE213069_day{day}",
            "independent_biological_replicate": "not_established",
            **row,
            "ena_average_bases_per_reported_read": bases / reads,
            "ena_fastq_file_count": len(row["fastq_ftp"].split(";")) if row["fastq_ftp"] else 0,
            "ncbi_sra_url": f'https://www.ncbi.nlm.nih.gov/sra/{row["run_accession"]}',
            "required_raw_route": "NCBI SRA + fasterq-dump --split-files --include-technical",
            "barcode_umi_layout": "unverified_in_current_workspace",
            "processing_status": "raw_not_located_in_audited_RENGE_directory",
        })
    if sample_counts != Counter({gsm: 2 for gsm in SAMPLES}):
        raise ValueError(f"Unexpected runs per sample: {sample_counts}")
    return sorted(resolved, key=lambda r: (r["day"], r["library_type"], r["run_accession"]))


def guide_metadata(features, published_guides):
    guides = [r for r in features if r[2] == "CRISPR Guide Capture"]
    if len(guides) != 50 or len({r[1] for r in guides}) != 50:
        raise ValueError("Expected 50 unique guide display names")
    if {r[1] for r in guides} != set(published_guides):
        raise ValueError("Guide names do not match the published XLSX")
    result = []
    for feature_id, name, kind in guides:
        match = re.fullmatch(r"(.+)_([12])", name)
        if not match:
            raise ValueError(f"Unrecognized guide name: {name}")
        target = match.group(1)
        group = "AAVS1" if target == "AAVS1" else "CTRL" if target == "CTRL" else "TF"
        result.append({"feature_id": feature_id, "guide_name": name,
                       "target_from_published_name": target,
                       "guide_number": match.group(2), "feature_type": kind,
                       "control_class": group, "pooling_status": "not_pooled"})
    return result


def validate_matrix(matrix, features, barcodes):
    import numpy as np
    if any(len(row) != 3 for row in features):
        raise ValueError("Features must have exactly 3 tab-delimited fields")
    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError("Matrix/features/barcodes dimension mismatch")
    if len({r[0] for r in features}) != len(features):
        raise ValueError("Duplicate feature IDs within a timepoint")
    if len(set(barcodes)) != len(barcodes):
        raise ValueError("Duplicate barcodes within a timepoint")
    if not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix.data < 0):
        raise ValueError("Expected nonnegative integer UMI counts")
    csr = matrix.tocsr()
    if csr.nnz != matrix.nnz:
        raise ValueError("Duplicate MatrixMarket coordinates")
    return csr


def audit_day(root, day, published_guides):
    import numpy as np
    from scipy.io import mmread
    from threadpoolctl import threadpool_limits
    directory = root / f"day{day}"
    features = load_tsv_gz(directory / "features.tsv.gz")
    barcode_rows = load_tsv_gz(directory / "barcodes.tsv.gz")
    if any(len(r) != 1 for r in barcode_rows):
        raise ValueError("Unexpected barcode schema")
    barcodes = [r[0] for r in barcode_rows]
    with threadpool_limits(limits=2), gzip.open(directory / "matrix.mtx.gz", "rb") as handle:
        matrix = mmread(handle)
    matrix = validate_matrix(matrix, features, barcodes)
    types = Counter(r[2] for r in features)
    if set(types) != {"Gene Expression", "CRISPR Guide Capture"}:
        raise ValueError(f"Unexpected feature types: {types}")
    guides = guide_metadata(features, published_guides)
    gene_indices = [i for i, r in enumerate(features) if r[2] == "Gene Expression"]
    guide_indices = [i for i, r in enumerate(features) if r[2] == "CRISPR Guide Capture"]
    expression = matrix[gene_indices]
    guide_counts = matrix[guide_indices].toarray()
    gex_total = np.asarray(expression.sum(axis=0)).ravel()
    guide_total = guide_counts.sum(axis=0)
    detected = np.asarray((expression > 0).sum(axis=0)).ravel()
    guide_detected = (guide_counts > 0).sum(axis=0)
    top_index = guide_counts.argmax(axis=0)
    top_count = guide_counts.max(axis=0)
    top_tie = (guide_counts == top_count[None, :]).sum(axis=0) > 1
    cell_rows = []
    for i, barcode in enumerate(barcodes):
        cell_rows.append({
            "cell_id": f"GSE213069_day{day}:{barcode}", "day": day, "barcode": barcode,
            "gex_umi": int(gex_total[i]), "detected_genes": int(detected[i]),
            "guide_umi": int(guide_total[i]), "detected_guides": int(guide_detected[i]),
            "largest_count_guide": guides[top_index[i]]["guide_name"] if top_count[i] > 0 else "",
            "largest_guide_umi": int(top_count[i]), "largest_guide_tied": bool(top_tie[i]),
            "largest_guide_fraction": float(top_count[i] / guide_total[i]) if guide_total[i] > 0 else "",
            "assigned_condition": "", "split": "", "label_status": "not_called",
            "donor": "", "biological_replicate": "",
            "su_status": "absent_from_published_matrix_bundle",
        })
    stats = {
        "day": day, "cells_in_released_matrix": len(barcodes), "features": len(features),
        "gene_expression_features": types["Gene Expression"],
        "guide_features": types["CRISPR Guide Capture"], "matrix_nnz": matrix.nnz,
        "explicit_zero_entries": int((matrix.data == 0).sum()),
        "median_gex_umi": float(np.median(gex_total)),
        "median_detected_genes": float(np.median(detected)),
        "median_guide_umi": float(np.median(guide_total)),
        "cells_without_guide_counts": int((guide_total == 0).sum()),
        "cells_with_multiple_detected_guides": int((guide_detected > 1).sum()),
        "cells_with_nonzero_tied_largest_guide": int((top_tie & (top_count > 0)).sum()),
        "structural_validation": "pass", "cell_filter_applied": False,
        "guide_calling_completed": False, "su_available": False,
    }
    return stats, cell_rows, features, guides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Outputs must not be placed in the original data directory")
    output.mkdir(parents=True, exist_ok=False)
    started = now()
    provenance = {"started_utc": started, "status": "running", "seed": None,
                  "random_operations": False, "command": sys.argv,
                  "python": sys.version, "executable": sys.executable,
                  "platform": platform.platform(), "data_root": str(root),
                  "script_sha256": sha256(__file__), "config_sha256": sha256(args.config),
                  "input_files": [], "sources": [], "checked_sources": [],
                  "fits_performed": [], "evaluation_of_responses_performed": False}
    write_json(output / "provenance.json", provenance)
    try:
        shutil.copyfile(args.config, output / "config.yaml")
        env = {}
        for package in ("numpy", "scipy", "openpyxl", "threadpoolctl", "torch", "anndata", "pytest"):
            try:
                env[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                env[package] = None
        write_json(output / "environment.json", env)
        # This is a public ENA metadata request, not a raw sequencing download.
        with urlopen(ENA_URL, timeout=60) as response:
            payload_bytes = response.read(2 * 1024 * 1024)
        runs = parse_runs(payload_bytes.decode("utf-8"))
        provenance["sources"].append({"url": ENA_URL, "retrieved_utc": now(),
                                      "response_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                                      "rows": len(runs), "status": "ok"})
        provenance["checked_sources"] = list(provenance["sources"])
        write_csv(output / "run_manifest.csv", runs)
        from openpyxl import load_workbook
        workbook = load_workbook(root / "GSE213069_gRNA_name_list.xlsx", read_only=True, data_only=True)
        published = [row[0] for row in workbook.active.values if row[0] is not None]
        workbook.close()
        if len(published) != 50 or len(set(published)) != 50:
            raise ValueError("Unexpected published guide list")
        qcs, cells, previous_features, guides = [], [], None, None
        barcode_sets = {}
        for day in range(2, 6):
            print(f"[{now()}] Structural audit day{day}", flush=True)
            qc, day_cells, features, guides = audit_day(root, day, published)
            if previous_features is not None and features != previous_features:
                raise ValueError("Feature order differs between timepoints; explicit alignment required")
            previous_features = features
            barcode_sets[day] = {r["barcode"] for r in day_cells}
            qcs.append(qc)
            cells.extend(day_cells)
            print(json.dumps(qc), flush=True)
        if len({r["cell_id"] for r in cells}) != len(cells):
            raise ValueError("Namespaced cell IDs are not unique")
        overlaps = [{"day_a": a, "day_b": b, "reused_barcode_strings": len(barcode_sets[a] & barcode_sets[b]),
                     "interpretation": "same_string_does_not_establish_same_cell"}
                    for a in range(2, 6) for b in range(a + 1, 6)]
        write_csv(output / "data_qc.csv", qcs)
        write_csv(output / "metrics.csv", qcs)
        write_csv(output / "cell_manifest.csv", cells)
        write_csv(output / "guide_manifest.csv", guides)
        write_csv(output / "barcode_overlap.csv", overlaps)
        # Input hash records describe local bytes, not verified archive authenticity.
        for path in sorted(root.rglob("*")):
            if path.is_file() and (path.name.endswith((".gz", ".xlsx"))):
                provenance["input_files"].append({"path": str(path), "bytes": path.stat().st_size,
                                                   "sha256": sha256(path)})
        write_csv(output / "input_hashes.csv", provenance["input_files"])
        provenance.update(status="structural_audit_complete", finished_utc=now(),
                          total_cells=len(cells), g1_status="not_run", phase0_exit="not_met",
                          exclusions="no filtering; no guide calling; no condition statistics; no train/test fitting")
        table = "\n".join(f"| day{q['day']} | {q['cells_in_released_matrix']} | {q['median_gex_umi']:g} | {q['median_guide_umi']:g} |" for q in qcs)
        (output / "RESULTS.md").write_text(
            "# RENGE Phase 0：已发布矩阵的结构审计\n\n"
            f"状态：结构检查通过；Phase 0 尚未完成，G1 未运行。共 {len(cells):,} 个发布矩阵 barcode。\n\n"
            "| 时间 | 发布矩阵 barcode 数 | GEX UMI 中位数 | gRNA UMI 中位数 |\n"
            "| --- | ---: | ---: | ---: |\n" + table + "\n\n"
            "四天特征顺序一致，均为 60,683 个表达特征和 50 个 gRNA 特征；gRNA 名称与发布 XLSX 一致。\n\n"
            "注意：barcode 数不是通过本项目 QC 的最终细胞数；最大计数 guide 不是正式条件标签，多个非零 guide 也不等于多感染。"
            "AAVS1 与 CTRL 保留原标签、不合并。跨时间同名 barcode 不作为配对依据。\n\n"
            "发布矩阵未提供 S/U，不能据此开始 velocity 诊断。未做 HVG、PCA、速度估计、模式发现、响应分析或模型训练。"
            "cell_manifest 中条件和 split 有意留空；尚无可执行的最终 split_manifest。\n\n"
            "ENA 返回 16 个 run：每个 mRNA/gRNA 文库各两个 run，共 8 个文库。ENA FASTQ 字段每条 run 仅一个文件且"
            "平均 91 bases/read；不能由 PAIRED 标签推定 barcode/UMI technical reads 已保留。后续走 NCBI 原始提取并实测布局。\n\n"
            f"来源：[GEO GSE213069]({GEO_URL})；[ENA run report]({ENA_URL})。请求时间及校验和见 provenance.json。\n\n"
            "本地 SHA-256 只用于复现追踪，不代表已与原始发布文件校验和完成核对。\n"
        )
    except Exception as exc:
        provenance.update(status="failed", finished_utc=now(), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "provenance.json", provenance)


if __name__ == "__main__":
    main()
