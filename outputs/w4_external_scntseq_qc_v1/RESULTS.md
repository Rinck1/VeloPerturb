# scNT-seq GEO raw-labeling technical QC

Completed: 10 public GEO processed files, 49,779,872 bytes. No raw reads downloaded.

- Cells × genes: 25,000 × 37,661.
- Times: 0/15/30/60/120 min; cell counts 3000/5000/5000/7000/5000.
- Exact new/old barcode and gene sets/order match in every sample.
- C = new; T = old; total = C + T. No S/U inferred from these matrices.
- All original counts retained; no response, cell-type, or HVG selection.
- All matrices contain finite nonnegative integers; zero total-count cells: 0.
- Runtime: 279.25 s; numerical threads capped at 4; no GPU used.

Artifact: `/data/yuchang/veloroute_w4/scntseq/processed/scntseq_neuron_new_old_total_only.h5ad`.
SHA256: `f033ccfd61d767c6bf40ca613adb4174bd1c2f1af8f8c048a4d2c05b2646020a`.
Download URLs and input hashes are in `downloads.json`; cell-library QC is in `technical_qc_by_time.csv`.

The later exact S/U pairing is separately recorded in `outputs/w4_external_scntseq_pairing_v2/`.
No biological response, velocity, trajectory, or prediction analysis performed.
