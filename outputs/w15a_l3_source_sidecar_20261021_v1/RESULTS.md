# L3 local source U controls

Completed 15 training source cohorts (44,222 cells) and 3 development source cohorts (13,636 cells).
All 18 cohorts preserve the original state, local kinetic features and cell IDs bitwise.
Reconstruction before restoring the original float32 features differs by at most 9.52e-7.

Local direct-U and Umean/Uzero interventions share the original 64-neighbor S geometry and full-gene S library factors.
Only training sources fit the new direct-U normalizer. Observed historical references remain fixed during interventions.
No parent data pack was modified and no development target counts were read. Seven unit tests passed.

Large feature arrays: `/data/yuchang/veloroute_w15a/l3_source_sidecar_v1/`.
Source manifest SHA256: `1733feefc60e5bc8976cc06d0a1eb9348c9b0c2a5b4fb9723864a31ba63802ac`.
See `summary.json`, `technical_audit.csv` and `source_manifest.json` for counts and provenance.
