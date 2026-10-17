# scNT-seq exact S/U–labeling technical pairing

Completed **13,476 cells × 37,007 exact common gene names**.

- 100% of official S/U barcodes exactly match unfiltered GEO original IDs.
- Overall GEO cell coverage: 53.904%; per-time coverage in `technical_qc_by_time.csv`.
- Time conflicts: 0. Duplicate cell/gene keys: 0. No fuzzy matching or imputation.
- Shared layers: raw GEO `new/old/total`, official `spliced/unspliced`.
- Every retained count layer is nonnegative, finite, and integer-valued.
- GEO total equals new + old; it does **not** equal independently quantified S + U:
  10,314,416 common matrix entries differ. Measurement models must remain distinct.
- 654 GEO-only and 7014 S/U-only genes are excluded from the common matrix and listed separately.
- Historical 3060-cell labeling values/activity-gene selection were not used.
- All available S/U cells retained. The author's construction notebook names its inputs
  `0411_Ex_...`, providing Ex-cohort provenance; the original Ex/QC assignment is still
  incompletely documented. See `upstream_selection_audit.json`, obtained after this h5ad
  was frozen. No additional author response/cell-type/detected-gene filtering was applied.

Artifact: `/data/yuchang/veloroute_w4/scntseq/processed/scntseq_geo_labeling_dynamo_su_exact.h5ad`.
SHA256: `208cc0b266d5a6d5bfbb18834f9f153402a7a3c93aaa9c9e4c27e29c96cefbd1`.
Runtime: 19.31 s; numerical threads capped at 4; no GPU used.

The first alignment attempt stopped on literal N in an original barcode. v2 permits A/C/G/T/N
while still requiring full exact identifier equality and independent time-field agreement.
No ID was repaired, no missing cell was forced into the cohort, and the aborted v1 is preserved.

This is a technical-modality result. External scientific protocol remains sealed; no response,
direction, velocity, multimodality, or forecast metric was computed.
