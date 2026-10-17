# W4-1c sci-fate technical QC

Status: `TOTAL_NEW_READY_US_MISSING`. No response effects or prediction analysis performed; preregistration is DRAFT/SEALED.

- Public input: GSE131351 / GSM3770930, six files, 229715904 bytes.
- Aligned release: `/data/yuchang/veloroute_w4/scifate/processed/scifate_total_new_only.h5ad`.
- Shape: 7404 cells × 43167 genes; total/new barcode and gene overlap 100%; exact matching order.
- Negative count entries: 0. Entries where new exceeds total: 0.
- Available gene layers: `total`, `new`; missing: `spliced`, `unspliced`.
- Timepoint cell counts (0,2,4,6,8,10 h): 1151,1189,1045,1404,1169,1446. Raw GEO cohort retained; published 6680-cell filtering not assumed.
- Complete raw FASTQ manifest: 190 runs, 116281104216 bytes; requires >50 GB download authorization, not downloaded.
- Runtime: 114.4959 seconds, no GPU; full provenance is in `summary.json` and `downloads.json`.

`technical_qc_by_time.csv` reports cell-level annotation totals, not gene-wise S/U. Newly labeled counts are not unspliced counts or direct net velocity. The current release does not yet permit the intronic-versus-labeling comparison.
