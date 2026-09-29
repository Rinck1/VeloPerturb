"""Freeze all gate interventions before reading development outcomes."""
import argparse
import json
from pathlib import Path

from veloroute.artifacts import Run, save_csv, save_json, sha256
from veloroute.experiments import evaluate_prediction
from veloroute.full_experiments import predict_full_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fold', required=True)
    parser.add_argument('--conditions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    fold = Path(args.fold)
    arms = [('full', None), ('full_frozen_static', 0.), ('full_dynamic', 1.)]
    with Run(args.output, stage='exploratory_full_gate_interventions', kind='engineering',
             config={**vars(args), 'arms': arms, 'seed': 0, 'confirmation': 'sealed'},
             inputs=[__file__, args.checkpoint, fold/'provenance.json', args.conditions], seed=0) as run:
        predictions = []
        for arm, gate in arms:
            directory = run.directory/arm/'predict'
            predict_full_model(args.checkpoint, fold/'validation_source.npz', args.conditions, directory,
                transform_path=fold/'transform.npz', seed=0, device=args.device, force_gate=gate)
            predictions.append({'arm': arm, 'directory': str(directory), 'sha256': sha256(directory/'predictions.npz')})
        save_json(run.directory/'frozen_predictions.json', predictions)
        rows = []
        for record in predictions:
            directory = Path(record['directory'])
            if sha256(directory/'predictions.npz') != record['sha256']:
                raise ValueError('Frozen prediction modified before evaluation')
            rows.extend(evaluate_prediction(directory, fold/'validation_target.npz', directory.parent/'evaluate',
                                           target_genes_path=fold/'validation_target_genes.npz'))
        save_csv(run.directory/'metrics.csv', rows)
        means = {arm: sum(r['energy_distance'] for r in rows if r['arm'] == arm)/sum(r['arm'] == arm for r in rows)
                 for arm, _ in arms}
        summary = {'status': 'FULL_DEVELOPMENT_EVALUATED', 'rows': len(rows), 'training_seeds': 1,
                   'mean_energy_by_arm': means, 'gain_full_vs_frozen_static': means['full_frozen_static']-means['full'],
                   'no_router_only_causal_claim': True, 'confirmation_evaluated': False,
                   'research_status': 'exploratory_not_preregistered'}
        save_json(run.directory/'summary.json', summary)
        (run.directory/'RESULTS.md').write_text('# Full model gate interventions\n\n'+json.dumps(summary, indent=2)
            +'\n\nThree source-only predictions were frozen before evaluation. One training seed; '
             'a full system versus its frozen fallback is not isolated router evidence.\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
