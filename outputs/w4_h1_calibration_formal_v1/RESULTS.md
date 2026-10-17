# W4 complete-H1 synthetic design calibration

Mode: **formal**; 16 TFs; 200 repetitions per scenario; 10000 TF bootstrap resamples per comparison.

This runs the two-interval outer-LOTO/inner-CV H1 decision design. A smoke run is not a power/FPR pass. Even formal runs estimate rates only under the declared simulation geometry/noise, not empirical RENGE power.

| Scenario | Latent rho | Events / repetitions | Rate | 95% exact binomial interval |
|---|---:|---:|---:|---:|
| detection | 0 | 200/200 | 1.0000 | [0.9817, 1.0000] |
| detection | 0.5 | 200/200 | 1.0000 | [0.9817, 1.0000] |
| detection | 0.9 | 200/200 | 1.0000 | [0.9817, 1.0000] |
| false_positive | 0 | 2/200 | 0.0100 | [0.0012, 0.0357] |
| false_positive | 0.5 | 1/200 | 0.0050 | [0.0001, 0.0275] |
| false_positive | 0.9 | 2/200 | 0.0100 | [0.0012, 0.0357] |

Four algorithmic seeds are averaged as predictions, not treated as biological repeats. Intervals are averaged within TF; both primary comparisons must pass CI, Wilcoxon and positive-interval checks. The middle snapshot is shared by adjacent intervals, but cells at distinct times are independently sampled.

Fixed source k-means centers are unused by the primary arms. Only identical per-arm feature matrices reuse a ridge fit; tied U-swap choices are generated separately for every seed.

All simulation seeds, fold audits, per-repetition comparisons and provenance hashes are stored alongside this report. No biological replication data or GPU were used.
