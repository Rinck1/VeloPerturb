# W4-1a RENGE technical data audit

Date: 2026-10-17. Scope: raw metadata, chemistry, count reconstruction, and technical QC only. No day2/day3 perturbation effects, embeddings, cross-condition expression comparisons, velocity fits, or forecast scores are inspected.

## Verified sample and run identities

ENA's live PRJNA879051 run report agrees with the repository's 2026-09-12 accession manifest on all 16 runs. The current response and its SHA256 are retained in `outputs/w4_renge_inventory_v1/`.

| Day | GEX sample / runs | Guide sample / runs | Initial local status |
|---|---|---|---|
| 2 | GSM6571143 / SRR21518315, SRR21518316 | GSM6571144 / SRR21518313, SRR21518314 | Published matrix present; raw acquisition needed |
| 3 | GSM6571145 / SRR21518311, SRR21518312 | GSM6571146 / SRR21518309, SRR21518310 | All four SRA files and technical-inclusive FASTQs cached since 2026-09-14 |
| 4 | GSM6571147 / SRR21518307, SRR21518308 | GSM6571148 / SRR21518305, SRR21518306 | Both GEX runs jointly quantified in the existing release |
| 5 | GSM6571149 / SRR21518303, SRR21518304 | GSM6571150 / SRR21518301, SRR21518302 | Both GEX runs jointly quantified in the existing release |

The small gRNA files are distinct guide libraries. They must not be included in the GEX USA quantification.

Day2 SRA HEAD sizes are 103,620,372; 100,197,194; 3,517,096,842; and 3,268,032,800 bytes for runs 313 through 316 respectively. New download total: **6,988,947,208 bytes (6.99 GB)**, below the authorized 50 GB dataset limit. Extraction and retained uncompressed FASTQs consume additional local disk, not network download volume. At inventory time `/data` had 765.8 GB free; the system filesystem had only 21.2 GB free. All new raw and quantification artifacts therefore go to `/data/yuchang/veloroute_w4/renge/`.

## Reused chemistry and guide/QC rules

Historical day4/day5 `simpleaf_quant_log.json` confirms the exact GEX invocation:

- GENCODE v32 primary-assembly GRCh38 splici r91 index, hash-bound against the old index manifest.
- SRA split FASTQ `_2` is the 26-base barcode/UMI read; `_3` is the 91-base cDNA read. `_1` is a separate index read.
- Geometry `1{b[16]u[10]x:}2{r:}`, expected orientation `rc`, `cr-like` resolution.
- The two GEX runs for a given day enter one mapping/UMI-resolution call. Run-level count matrices are not added.
- Day3's previous one-million-pair checks showed barcode offset-0 matches of 83.25–85.07%, versus 0.0323–0.0407% at offset 1. The new processor repeats this bounded audit and records input FASTQ SHA256 hashes.
- S/U/A remain separate; H5AD `X` is S+A. Downstream state/velocity work must explicitly select the `spliced` layer for strict S.
- The historical guide caller consumes the author's published `CRISPR Guide Capture` feature-UMI matrix, aligned by barcode. It does not infer a fresh guide feature matrix from the gRNA FASTQs. It calls a guide only when top UMI >=3, top fraction >=0.8, top/runner >=5, and top>runner.
- Technical thresholds: S+A expression UMI >=500, detected genes >=200, U UMI >=20, mitochondrial fraction <=0.2. These are computational screening labels; `formal_ready` remains false.

The existing released files are `data/renge/processed_release_v1/day4/day4.h5ad` and `.../day5/day5.h5ad`. Observations contain `condition`, `guide_call_pass`, `technical_qc_pass`, `role`, `day`, `sample`, and `barcode`; conditions include TF names, CTRL, AAVS1, and unassigned. New W4 technical QC reports only per-day U/S summaries and per-condition cell counts, not gene-expression differences.

## Same-input reproduction

The task draft requested processing one day4 run and comparing it with the release. The release used **both** day4 GEX runs jointly; a single-run matrix is a different depth and molecule set. The implemented reproduction therefore uses both original GEX runs and the same reference, geometry, observed barcode cohort, resolution, import, and QC rules. Only CPU thread count changes from 16 to at most 8.

Acceptance checks align the full barcode and gene universes, compare cell-total correlations for S/U/A against >=0.999, and also report full-matrix disagreement counts and maximum total-count discrepancies. A failed reproduction blocks day2/day3 quantification in this runner.

The reproduction job `outputs/w4_renge_day4_reproduction_job_v1/` **completed successfully at 2026-10-17 10:54:53 CST**. Its data outputs are `/data/yuchang/veloroute_w4/renge/day4_reproduction_v1/`; the summary recorded 345.6 seconds before final artifact hashing.

| Layer | Cell-total correlation | Maximum cell-total difference | Full-matrix differing entries |
|---|---:|---:|---:|
| S | 1.0 | 0 | 0 |
| U | 1.0 | 0 | 0 |
| A | 1.0 | 0 | 0 |

Barcode and gene universes matched exactly: 7,582 cells × 60,668 genes. This establishes count reconstruction consistency, not velocity validity or short-horizon predictive gain. The newly reproduced H5AD SHA256 is `e7dbbb130b6802030725432c3cae8025a12143db3f673317f5ddf22a6f4a5237`.

The small, hash-verified evidence bundle is retained separately from large matrices at [`outputs/w4_renge_day4_reproduction_evidence_v1/`](../outputs/w4_renge_day4_reproduction_evidence_v1/). Its export manifest retains the original `/data` paths and SHA256 hashes; the copied original provenance continues to describe files under that original root.

## Execution entry points

`scripts/phase0/w4_renge_data.py` exposes `audit`, `acquire`, `process`, and the serial `day23` pipeline. It uses `configs/w4_renge_day23_runs.csv` and `configs/w4_renge_day23_preprocess.yaml`; the latter inherits the historical preprocessing config and checks that chemistry values are unchanged. All output directories are exclusive, so old data and interrupted attempts are preserved.

The serial day2/day3 job is `outputs/w4_renge_day23_job_v1/`. It waits for a passed day4 same-input reproduction, quantifies cached day3, acquires and validates day2, then quantifies day2. Quantification and extraction never run concurrently in this job (at most eight CPU threads; zero GPUs). Its records are `/data/yuchang/veloroute_w4/renge/day23_pipeline_v1/`; final count files are `/data/yuchang/veloroute_w4/renge/processed_release_v2/day2/day2.h5ad` and `day3/day3.h5ad`.

## Day3 count reconstruction completed

At 2026-10-17 11:00 CST, day3 count reconstruction and technical QC are complete. All mapping/permit-list/collation/USA commands returned exit 0; the serial pipeline has moved to day2 raw acquisition. Day2 remains incomplete at this snapshot.

| Technical item | Day3 value |
|---|---:|
| Published-cohort cells reconstructed | 6,091 |
| ENSG genes | 60,668 |
| Conservative guide-assigned cells | 4,200 (68.95%) |
| Technical-QC passing cells | 5,726 |
| Passing both guide and technical QC | 4,153 |
| Median S UMI / cell | 12,163 |
| Median U UMI / cell | 921 |
| Median U/(S+U) | 0.068915 |

The completed file is `/data/yuchang/veloroute_w4/renge/processed_release_v2/day3/day3.h5ad`, SHA256 `3bf623c7dc09cbb03ec76c5cdb42a379b45cb28f9f9b83b8ca26507cf5b718a9`. The small summary/config/provenance/condition-cell-count evidence bundle is [`outputs/w4_renge_day3_technical_evidence_v1/`](../outputs/w4_renge_day3_technical_evidence_v1/). Only technical counts have been computed; short-time prediction, U increment, and any effect comparisons remain unrun and sealed pending protocol freeze.

## Day2 acquisition recovery

The initial day23 queue stopped after downloading and validating `SRR21518313`, before extracting any reads. The launcher had called `Path.resolve()` on `fasterq-dump`, resolving this SRA Toolkit symlink to the multicall binary `sratools.3.4.1`. Dispatch depends on the invocation basename, so the process immediately reported `unrecognized tool .../sratools.3.4.1` (exit 1). This was a command-construction error, not a chemistry or input-file failure; the empty FASTQ/scratch directories and downloaded SRA were preserved.

The command now preserves the `fasterq-dump` symlink name. `fasterq-dump --version` returned 3.4.1. Recovery reuses only existing **empty** extraction directories; any nonempty partial output is rejected and retained for explicit recovery. The day2-only queue `outputs/w4_renge_day2_job_v2/` runs in `/data/yuchang/veloroute_w4/renge/day2_pipeline_v2/`, leaving completed day3/day4 results intact. Its first extraction succeeded at 11:01:56 CST: **2,529,832 spots and 7,589,496 reads written**, with the expected index/barcode/cDNA FASTQs. The queue then continued compression and subsequent run acquisition. Tests in `tests/test_w4_renge_data.py` exercise symlink-name preservation, empty-directory reuse, and nonempty partial preservation.

The four command/recovery regression tests passed. Both day2 guide runs then completed download, extraction, compression, and gzip integrity checks; the queue proceeded to the two larger GEX SRA files.

A separate CPU queue, `outputs/w4_renge_cohort_wait_job_v1/`, waits for the day2 stage's **completed** provenance. It verifies the final H5AD hash, exports the small day2 evidence bundle to `outputs/w4_renge_day2_technical_evidence_v2/`, then invokes `w4_cohort_qc.py --output outputs/w4_cohort_qc_v1`. This uses only metadata and the fixed 50-cell eligibility threshold. It does not freeze the protocol or authorize/read response effects. If day2 fails, the waiter stops explicitly instead of using partial counts.

Public sources: [GEO GSE213069](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213069), [ENA PRJNA879051](https://www.ebi.ac.uk/ena/browser/view/PRJNA879051), [NCBI SRA public cloud data](https://www.ncbi.nlm.nih.gov/sra/docs/sra-cloud/). GEO's browser page returned a browser challenge in this session; live ENA metadata and NCBI-hosted SRA object headers were accessible.
