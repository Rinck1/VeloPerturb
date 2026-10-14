"""Registered synthetic checks for W3 diagnostics; no biological inputs or GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from veloroute.artifacts import Run, load_config, save_csv, save_json
from veloroute.gpu_policy import enforce_gpu_policy
from w3_validation import conditional_score_test, two_stage_iv


def wilson(successes, trials):
    p = successes / trials
    z = 1.959963984540054
    den = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / den
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return float(center - half), float(center + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/w3_preflight_20261014.json')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    config = load_config(args.config)
    enforce_gpu_policy('cpu')
    script = Path(__file__).resolve()
    inputs = [Path(args.config), script, script.with_name('w3_validation.py')]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    start = time.monotonic()
    rng = np.random.default_rng(config['seed'])
    digest = hashlib.sha256()
    with Run(args.output, stage='W3_diagnostic_synthetic_calibration', kind='synthetic',
             config=config, inputs=inputs, seed=config['seed']) as run:
        rows, summary = [], []
        scenarios = config['null_scenarios'] + config['positive_scenarios']
        for scenario in scenarios:
            for rep in range(config['replicates']):
                n = config['units_per_replicate']
                state = rng.normal(size=(n, 4))
                extra = state @ rng.normal(size=(4, 6)) + rng.normal(size=(n, 6))
                noise = rng.normal(size=n)
                if scenario == 'heteroscedastic_null':
                    noise *= .3 + np.abs(state[:, 1])
                y = 4 * state[:, 0] - 2 * state[:, 1] + noise
                if scenario == 'sparse_increment':
                    y += .5 * extra[:, 0]
                elif scenario == 'mixed_sign_increment':
                    y += .5 * (extra[:, 0] - extra[:, 1])
                for value in (state, extra, y):
                    digest.update(np.ascontiguousarray(value).tobytes())
                test = conditional_score_test(y, extra, state,
                    simulations=config['conditional_bootstrap_simulations'], seed=config['seed'] + rep)
                rows.append(dict(scenario=scenario, replicate=rep, p_value=test['p_value'],
                                 rejected=test['p_value'] < config['alpha']))
            selected = [r for r in rows if r['scenario'] == scenario]
            successes = sum(r['rejected'] for r in selected)
            lo, hi = wilson(successes, len(selected))
            is_null = scenario in config['null_scenarios']
            passed = hi <= config['null_wilson95_upper_max'] if is_null else lo >= config['power_wilson95_lower_min']
            row = dict(scenario=scenario, replicates=len(selected), rejection_rate=successes / len(selected),
                       wilson95_low=lo, wilson95_high=hi, passed=passed)
            summary.append(row)
            print(json.dumps(row), flush=True)
            save_json(run.directory / 'status.json', dict(status='running', completed_scenarios=summary))
        iv_rows = []
        for rep in range(config['iv_replicates']):
            n = config['iv_genes_per_replicate']
            latent = rng.normal(size=n)
            ua = .8 * latent + rng.normal(size=n)
            sa, sb, sc = [latent + rng.normal(size=n) for _ in range(3)]
            y = 2 * latent + config['iv_extra_coefficient'] * ua + rng.normal(size=n)
            y_null = 2 * latent + rng.normal(size=n)
            for value in (latent, ua, sa, sb, sc, y, y_null):
                digest.update(np.ascontiguousarray(value).tobytes())
            fit = two_stage_iv(y, sb, ua, sc)
            control = two_stage_iv(y_null, sb, sa, sc)
            iv_rows.append(dict(replicate=rep, extra_coefficient=fit['extra_coefficient'],
                                first_stage_f=fit['first_stage_f'], self_control_coefficient=control['extra_coefficient']))
        estimated = float(np.mean([r['extra_coefficient'] for r in iv_rows]))
        self_control = float(np.mean([r['self_control_coefficient'] for r in iv_rows]))
        iv_pass = (abs(estimated - config['iv_extra_coefficient']) <= config['iv_mean_coefficient_error_max']
                   and abs(self_control) <= config['iv_self_control_mean_abs_max'])
        result = dict(status='CALIBRATION_PASS' if all(r['passed'] for r in summary) and iv_pass else 'CALIBRATION_FAIL',
                      conditional_score=summary, iv=dict(mean_extra=estimated, mean_self_control=self_control, passed=iv_pass),
                      synthetic_data_sha256=digest.hexdigest(), git_commit=commit,
                      runtime_seconds=time.monotonic() - start,
                      biological_conclusion='NONE',
                      scope='specified_linear_nulls_and_injections_only; not general conditional independence or proof of IV exclusion')
        save_csv(run.directory / 'conditional_replicates.csv', rows)
        save_csv(run.directory / 'iv_replicates.csv', iv_rows)
        save_json(run.directory / 'summary.json', result)
        save_json(run.directory / 'status.json', dict(status='complete', decision=result['status']))
        (run.directory / 'RESULTS.md').write_text('# W3 synthetic diagnostic calibration\n\n'
            + json.dumps(result, indent=2) + '\n\nNo biological dataset was used. '
              'A pass validates only the registered simulation families, not all nulls.\n')
        print(json.dumps(result), flush=True)
    if result['status'] != 'CALIBRATION_PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
