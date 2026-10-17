"""Plot the completed W7 fixed grid; never selects models or modifies training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time


def plot(root, wait):
    root = Path(root); deadline = time.monotonic()+12*3600
    while not (root/'report'/'summary.json').exists():
        if not wait or time.monotonic() > deadline: raise TimeoutError('Scaling report incomplete')
        time.sleep(15)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = root/'report'
    rows = list(csv.DictReader((report/'grid.csv').open()))
    for row in rows:
        for key in ('experts','hidden','step'): row[key] = int(row[key])
        row['energy_distance'] = float(row['energy_distance'])
    config = json.loads((root/'study_manifest.json').read_text())['config']
    figure, axes = plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    colors = dict(static_router='#777777',velocity_router='#1679ab')
    selections = [
        ('experts',lambda r:r['hidden']==config['baseline_width'] and r['step']==config['steps'],'Number of experts K'),
        ('hidden',lambda r:r['experts']==config['baseline_experts'] and r['step']==config['steps'],'Backbone width'),
        ('step',lambda r:r['experts']==config['baseline_experts'] and r['hidden']==config['baseline_width'],'Training steps')]
    for axis,(key,selector,label) in zip(axes,selections):
        for arm in ('static_router','velocity_router'):
            selected = sorted([r for r in rows if selector(r) and r['arm']==arm], key=lambda r:r[key])
            if key == 'experts':
                selected = [r for r in rows if selector(r) and r['arm']=='k1']+selected
            axis.plot([r[key] for r in selected],[r['energy_distance'] for r in selected],
                marker='o',color=colors[arm],label=arm.replace('_',' '))
        axis.set_xscale('log',base=2 if key!='step' else 10)
        ticks = sorted({r[key] for r in rows if selector(r)})
        axis.set_xticks(ticks,labels=[str(t) for t in ticks]); axis.set_xlabel(label)
        axis.grid(alpha=.2); axis.set_ylabel('Energy distance (lower is better)')
    axes[0].legend(fontsize=8)
    figure.suptitle('sci-FATE2: fixed global mean; three-seed weighted mixture; same-series development')
    figure.savefig(report/'scaling_curves.png',dpi=160); plt.close(figure)
    print(json.dumps(dict(status='SCALING_PLOT_READY',path=str((report/'scaling_curves.png').resolve()))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root',required=True); parser.add_argument('--wait',action='store_true')
    args = parser.parse_args(); plot(args.root,args.wait)
