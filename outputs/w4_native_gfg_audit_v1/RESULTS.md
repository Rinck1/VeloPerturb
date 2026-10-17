# W4-0: native GFG source-U sensitivity audit

Completed native target-domain fits were re-audited on CPU; no model was retrained.

Native columns show the range of per-seed cell-median cos/norm ratio and global relative L2. The analytical reference is identical across training seeds.

| Dataset | U intervention | Native cosine | Native norm ratio | Native relative L2 | Analytic cosine | Analytic norm ratio | Analytic relative L2 |
|---|---|---:|---:|---:|---:|---:|---:|
| renge | V0_real | 1.000000–1.000000 | 1.000000–1.000000 | 0.000000–0.000000 | 1.000000 | 1.000000 | 0.000000 |
| renge | V3_local_U | 0.999188–0.999799 | 0.999886–1.001155 | 0.022994–0.043920 | 0.761603 | 0.994600 | 0.660915 |
| renge | V4_global_U | 0.998962–0.999775 | 1.000060–1.002769 | 0.024113–0.049221 | 0.655805 | 1.031776 | 0.812450 |
| renge | V7_zero_U | 0.998133–0.999669 | 0.974381–1.098584 | 0.037979–0.118793 | -0.194020 | 1.832599 | 2.053947 |
| merlin | V0_real | 1.000000–1.000000 | 1.000000–1.000000 | 0.000000–0.000000 | 1.000000 | 1.000000 | 0.000000 |
| merlin | V3_local_U | 0.997909–0.998994 | 0.983326–1.004276 | 0.067209–0.099137 | 0.440972 | 1.074391 | 1.148713 |
| merlin | V4_global_U | 0.997089–0.997958 | 0.976226–1.032191 | 0.077464–0.229683 | 0.181480 | 1.528942 | 1.670551 |
| merlin | V7_zero_U | 0.994165–0.998305 | 0.951782–1.068799 | 0.091685–0.129570 | -0.073450 | 0.568385 | 1.146598 |

## Interpretation

Under these target-domain native training settings, the output direction changes little under U interventions. Norm and L2 changes are not zero. This is an estimator-sensitivity result, not evidence that U has no biological information or that downstream prediction cannot improve.

The analytical reference is Mu − gamma·Ms with gamma fitted by full-source moment OLS and frozen across interventions; it is not W4's raw-count two-tail estimator. S moments, graph, and recipient normalizers are fixed. All valid-direction fractions are reported in metrics.csv. These are training-source audits, not held-out prediction results.

## Provenance

Git commit: `58f1954bd3cb54f83e1979a8d896efb28da8d0c3`; seeds: `[0, 1, 2]`; runtime: 18.899 seconds.

Exact commands, all source paths and SHA256 hashes are in summary.json/provenance.json; per-seed values and original summary/velocity paths are in metrics.csv.
