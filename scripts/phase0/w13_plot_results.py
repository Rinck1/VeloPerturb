"""Plot fixed W13 recipes and horizons without selecting a checkpoint."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root,continuation):
    root,continuation = Path(root),Path(continuation)
    result = json.loads((root/'summary.json').read_text())
    report = json.loads((continuation/'report/summary.json').read_text())
    fig,axes = plt.subplots(1,3,figsize=(15,4),layout='constrained')
    rows = report['comparisons']
    for key,label,color in (('real_energy','Real kinetics','#087e8b'),
                            ('static_energy','Static router','#777777'),
                            ('shuffled_energy','Shuffled kinetics','#c47b27')):
        axes[0].plot([r['step']/1000 for r in rows],[r[key] for r in rows],marker='o',label=label,color=color)
    axes[0].set(xlabel='Training steps (thousands)',ylabel='Energy distance (lower better)',
                title='Direct-distance continuation: 5h windows')
    axes[0].legend(fontsize=8)
    labels = {'b0_endpoint36k':'Original endpoint 36k','b1_endpoint36k':'Direct endpoint 36k',
              'endpoint_time_40k':'Time endpoint 40k','ode_40k':'RK4 ODE 40k'}
    colors = ['#888888','#087e8b','#8b5fbf','#c47b27']
    for (name,label),color in zip(labels.items(),colors):
        selected = [r for r in result['horizon_summary'] if r['model']==name]
        axes[1].plot([r['horizon_hours'] for r in selected],
                     [100*r['gain_vs_static'] for r in selected],marker='o',label=label,color=color)
    axes[1].axhline(0,color='black',linewidth=.7)
    axes[1].set(xlabel='Forecast horizon (hours)',ylabel='Energy reduction vs own static (%)',
                title='Kinetic increment; available-window means',xticks=[5,10,15])
    axes[1].legend(fontsize=8)
    integration = result['integration_comparisons']
    axes[2].bar([str(r['horizon_hours']) for r in integration],
                [100*r['gain_vs_matched_endpoint'] for r in integration],color='#c47b27')
    axes[2].axhline(0,color='black',linewidth=.7)
    axes[2].set(xlabel='Forecast horizon (hours)',ylabel='ODE energy reduction (%)',
                title='ODE vs matched endpoint time field')
    fig.suptitle('Same-series development: 3 seeds; 5/10/15h contain 3/2/1 overlapping windows',fontsize=11)
    output = root/'figures';output.mkdir(exist_ok=False)
    fig.savefig(output/'continuation_and_time_models.png',dpi=170)
    fig.savefig(output/'continuation_and_time_models.pdf')
    plt.close(fig)


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--continuation',required=True);args = parser.parse_args()
    plot(args.root,args.continuation)
