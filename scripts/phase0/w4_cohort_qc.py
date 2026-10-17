"""Create W4 eligibility lists from technical metadata, with effects sealed."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from w4_artifacts import Artifacts
from w4_shorttime import CONTROL_LABELS, qc_mask, read_metadata


def eligible_cohorts(counts_by_day, minimum_cells=50):
    if set(counts_by_day) != {2,3,4,5} or minimum_cells != 50:
        raise ValueError('Registered days 2/3/4/5 and 50-cell threshold required')
    eligible = {day: {label for label,n in counts.items()
                      if n >= minimum_cells and label not in (*CONTROL_LABELS, 'unassigned')}
                for day,counts in counts_by_day.items()}
    return dict(H1_TFs=sorted(eligible[2] & eligible[3] & eligible[4]),
                H2_TFs=sorted(eligible[2] & eligible[3] & eligible[4] & eligible[5]),
                interval_TFs={f'{a}_to_{b}': sorted(eligible[a] & eligible[b])
                              for a,b in ((2,3),(3,4),(4,5),(2,4),(2,5),(3,5))})


def main():
    parser = argparse.ArgumentParser()
    for day in (2,3,4,5):
        root = '/data/yuchang/veloroute_w4/renge/processed_release_v2' if day in (2,3) else 'data/renge/processed_release_v1'
        parser.add_argument(f'--day{day}', default=f'{root}/day{day}/day{day}.h5ad')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    paths = {day: Path(getattr(args, f'day{day}')) for day in (2,3,4,5)}
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError('All four completed count files required; eligibility is not inferred from partial data')
    run = Artifacts(args.output, seed=None, inputs=[Path(__file__), Path(__file__).with_name('w4_artifacts.py'),
        Path(__file__).with_name('w4_shorttime.py')]+list(paths.values()))
    rows, counts = [], {}
    for day,path in paths.items():
        obs, _ = read_metadata(path)
        if set(obs['day'].astype(int)) != {day}:
            raise ValueError('Day label does not match declared input')
        selected = obs.loc[qc_mask(obs), 'condition'].astype(str)
        counts[day] = {str(label): int(n) for label,n in selected.value_counts().items()}
        for label in sorted(set(obs['condition'].astype(str))):
            rows.append(dict(day=day, condition=label, qualified_cells=counts[day].get(label,0),
                eligible_TF=counts[day].get(label,0) >= 50 and label not in (*CONTROL_LABELS,'unassigned')))
    result = dict(status='TECHNICAL_COHORT_QC_COMPLETE', preregistration_frozen=False,
        biological_effects_read=False, minimum_TF_cells_per_day=50,
        **eligible_cohorts(counts))
    with open(run.directory/'condition_counts.csv', 'x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    run.json('summary.json', result); run.json('status.json', dict(status='complete'))
    (run.directory/'RESULTS.md').write_text('# W4 technical eligibility\n\n'
        +f'H1 common TFs: {len(result["H1_TFs"])}; H2 common TFs: {len(result["H2_TFs"])}.\n\n'
        'Technical metadata only; this does not freeze the preregistration or authorize effects analysis.\n')


if __name__ == '__main__':
    main()
