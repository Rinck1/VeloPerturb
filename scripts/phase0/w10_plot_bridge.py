"""Plot all bridge recipes and the matched continuation without selecting checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from w4_artifacts import Artifacts, sha256


def read_csv(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))


def run(root):
    root = Path(root);manifest = json.loads((root/'study_manifest.json').read_text())
    report = json.loads((root/'report/summary.json').read_text())
    grid = read_csv(root/'report/grid.csv');last = manifest['config']['steps']
    recipes = [r['name'] for r in manifest['config']['recipes']]
    selected = report['selected_for_continuation'];next_root = root.parent/(root.name+'_continuation')
    inputs = [root/'study_manifest.json',root/'report/grid.csv',root/'report/comparisons.csv',Path(__file__)]
    comparisons = read_csv(root/'report/comparisons.csv')
    if (next_root/'report/comparisons.csv').is_file():
        inputs.append(next_root/'report/comparisons.csv')
        comparisons += read_csv(next_root/'report/comparisons.csv')
    run = Artifacts(root/'report/figures',seed=manifest['config']['seeds'],inputs=inputs)
    fig,axes = plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
    colors = {'static_router':'#777777','velocity_router':'#1f77b4','shuffled_velocity_router':'#e69f00'}
    labels = {'static_router':'Static routing','velocity_router':'Real dynamics','shuffled_velocity_router':'Shuffled dynamics'}
    for arm in manifest['config']['arms']:
        values = [next(float(r['energy_distance']) for r in grid if r['setting']==recipe and r['arm']==arm
                       and int(r['step'])==last) for recipe in recipes]
        axes[0].plot(range(len(recipes)),values,'o-',color=colors[arm],label=labels[arm])
    axes[0].set_xticks(range(len(recipes)),['W8','Direct\ndistance','Full\ncentering','Source\nU-stat','Full\nregularizer','LR\n1e-4'])
    axes[0].set_ylabel('Energy distance (lower is better)');axes[0].set_title('One-change recipe bridge, 12k steps')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    if selected is not None:
        curve = sorted([r for r in comparisons if r['setting']==selected['setting'] and int(r['step'])>=last],
                       key=lambda r:int(r['step']))
        for metric,label,color in (('gain_vs_static','vs matched static','#1f77b4'),
                                   ('gain_vs_shuffled','vs matched shuffled','#e69f00')):
            axes[1].plot([int(r['step'])/1000 for r in curve],[100*float(r[metric]) for r in curve],
                         'o-',label=label,color=color)
        axes[1].set_title(f'Matched continuation: {selected["setting"]}')
        axes[1].set_xlabel('Training steps (thousands)');axes[1].set_ylabel('Real-dynamics relative improvement (%)')
        axes[1].axhline(0,color='black',linewidth=.8);axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    else:
        axes[1].text(.5,.5,'No recipe beat both matched controls at 12k',ha='center',va='center',wrap=True)
        axes[1].set_axis_off()
    fig.suptitle('sci-FATE2: frozen mean, K=8, three-seed predictive mixture; same-series development',fontsize=11)
    path = run.directory/'bridge_and_continuation.png';fig.savefig(path,dpi=170);plt.close(fig)
    run.json('summary.json',dict(status='BRIDGE_FIGURE_READY',path=str(path),sha256=sha256(path)))
    run.json('status.json',dict(status='complete'));print(path,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root',required=True)
    args = parser.parse_args();run(args.root)
