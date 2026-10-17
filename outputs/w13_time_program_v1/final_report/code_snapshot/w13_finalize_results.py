"""Keep the fixed 6h study intact and append its frozen-weight solver sensitivity."""
import argparse
import json
from pathlib import Path

from w4_artifacts import Artifacts,sha256
from w7_router_scaling import write_csv


def finalize(root):
    root = Path(root);original_path = root/'summary.json';fine_path = root/'ode_refinement/summary.json'
    original,fine = [json.loads(p.read_text()) for p in (original_path,fine_path)]
    run = Artifacts(root/'final_report',seed=[0,1,2],inputs=[original_path,
        root/'ode_refinement/horizon_summary.csv',root/'ode_refinement/solver_convergence.csv',Path(__file__)])
    rows = original['horizon_summary']+fine['horizon_summary'];comparisons = []
    for row in fine['horizon_summary']:
        hours = row['horizon_hours']
        by_name = {r['model']:r for r in rows if r['horizon_hours']==hours}
        real = row['real_energy']
        comparisons.append(dict(horizon_hours=hours,refined_ode_energy=real,
            gain_vs_matched_endpoint=1-real/by_name['endpoint_time_40k']['real_energy'],
            gain_vs_b1_endpoint36k=1-real/by_name['b1_endpoint36k']['real_energy'],
            relative_change_from_6h_solver=real/by_name['ode_40k']['real_energy']-1,
            gain_vs_static=row['gain_vs_static'],gain_vs_shuffled=row['gain_vs_shuffled']))
    result = dict(status='W13_ALL_REQUESTED_WORK_AND_NUMERICAL_REFINEMENT_COMPLETE',
        requested_steps_completed=[1,2,4],excluded_step=3,formal_training_runs=27,
        additional_refinement_training_steps=0,parameters=original['ode_parameters'],
        horizon_summary=rows,refined_ode_comparisons=comparisons,
        original_fixed_6h_study=dict(path=str(original_path),sha256=sha256(original_path)),
        frozen_weight_refinement=dict(path=str(fine_path),sha256=sha256(fine_path)),
        maximum_3h_vs_1_5h_relative_solver_difference=max(r['relative_to_movement'] for r in fine['solver_convergence']),
        interpretation='ODE improves absolute 5/10h fit but does not enlarge the kinetic-vs-static gap; 15h has no integration benefit',
        recommended_working_model='small fixed-route ODE for short horizons; 3h inference maximum step, 6h training maximum step',
        separate_reference='b1 endpoint36k, preserve original b0 endpoint36k as well',
        scope='same_series_development',original_gfg_used=False,mean_branch_frozen=True)
    write_csv(run.directory/'all_horizon_summary.csv',rows)
    write_csv(run.directory/'refined_ode_comparisons.csv',comparisons);run.json('summary.json',result)
    text = '# W13 final results: requested steps 1/2/4 complete\n\n'
    text += '27 formal runs completed; step 3 omitted. Source kinetics remain analytical, not original GFG/JVP.\n\n'
    text += 'Direct-distance endpoint36k: energy 0.595683; +4.137% vs static and +4.606% vs shuffled kinetics (5h).\n\n'
    text += '## Genuine small ODE, frozen weights, 3h inference step\n\n'
    text += '| Hours | Energy | vs own static | vs shuffled | vs matched endpoint time model |\n|---|---:|---:|---:|---:|\n'
    text += '\n'.join(f'| {r["horizon_hours"]} | {r["refined_ode_energy"]:.6f} | '
        f'{100*r["gain_vs_static"]:+.3f}% | {100*r["gain_vs_shuffled"]:+.3f}% | '
        f'{100*r["gain_vs_matched_endpoint"]:+.3f}% |' for r in comparisons)
    text += '\n\nThe fixed 6h study remains unchanged. 3h inference uses identical weights, with no training or target-dependent '
    text += 'checkpoint selection. The largest 3h vs 1.5h source-only difference is '
    text += f'{100*result["maximum_3h_vs_1_5h_relative_solver_difference"]:.3f}% of predicted movement.\n\n'
    text += 'ODE improves absolute 5/10h fit, including the static controls; it is not evidence of a larger '
    text += 'velocity-specific interaction. At 15h it is essentially tied with the time endpoint control '
    text += 'and slightly worse than the original b1 endpoint. Use the small ODE as the short-horizon candidate '
    text += 'and retain the endpoint reference, not a post-hoc per-window winner assembly.\n\n'
    text += 'Mean prediction is frozen throughout. Windows are same-series development, not independent replication '
    text += 'or unseen genetic perturbation validation.\n'
    (run.directory/'RESULTS.md').write_text(text);run.json('status.json',dict(status='complete'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root',required=True);args = parser.parse_args()
    finalize(args.root)
