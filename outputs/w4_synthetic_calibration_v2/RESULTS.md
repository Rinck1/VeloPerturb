# W4-3 synthetic implementation calibration

Final functional fixture: 200 positive, 200 null, and 200 observed-S-determined
U stress replicates. Each replicate has 40 training populations and 24 independent
test populations; future cells are sampled independently from source cells.

| Scenario | Full two-comparison detection count | Rate |
|---|---:|---:|
| U innovation leads future S | 200 / 200 | 100% |
| U has no future predictive increment | 1 / 200 | 0.5% |
| U is a deterministic function of observed S | 1 / 200 | 0.5% |

These are implementation checks on a fixed train/test fixture and one transition,
not the power or false-positive rate of the two-interval RENGE LOTO protocol.
The latter requires the final QC cohort and a separate design calibration.

The v1 fixture shared source-cell noise with future cells and defined its stress
innovation from latent population S; v2 replaces these with independent snapshots
and strictly observed-S-determined U. v1 is retained but superseded.

All per-replicate values are in `replicates.csv`; `summary.json` includes source
hashes, baseline Git commit, command, seed, and elapsed time. `code_snapshot/`
preserves the exact four implementation files executed.
