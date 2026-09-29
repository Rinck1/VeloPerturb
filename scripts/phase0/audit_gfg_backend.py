"""Compare ported GFG reconstruction/JVP with the actual local reference source."""
import ast
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from veloroute.artifacts import Run, load_config, save_csv, save_json
from veloroute.gfg import GFGCore


def audit(output):
    config = load_config('configs/veloroute_gfg_joint_20260914.yaml')
    root = Path(config['gfg_source'])
    files = [root/'model'/name for name in ('Encoder.py', 'Codebook.py', 'Decoder.py', 'model.py')]
    namespace = dict(torch=torch, nn=nn, F=F, np=np, Config=object, DataLoader=object,
        z_score=lambda x, mean, std: (x-mean)/std, re_z_score=lambda x, mean, std: x*std+mean)
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        # Avoid the reference plotting/scanpy imports and unrelated runtime setup.
        tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom)
            and node.module and (node.module.startswith('tool.') or node.module.startswith('model.')))]
        exec(compile(tree, str(path), 'exec'), namespace)
    reference = namespace['VeloModel'].__new__(namespace['VeloModel'])
    nn.Module.__init__(reference)
    hidden, genes = (256, 512, 512, 256), 11
    reference.manifold_encoder = namespace['Encoder'](genes, 8, hidden)
    reference.velocity_encoder = namespace['Encoder'](genes, 8, hidden)
    reference.manifold_codebook = namespace['SoftVectorQuantizer'](32, 8)
    reference.velocity_codebook = namespace['SoftVectorQuantizer'](32, 8, normalize=True)
    reference.decoder = namespace['Decoder'](8, genes, hidden)
    reference.G = genes
    reference.layer_states = {name: dict(mean=torch.linspace(.5, 1.5, genes), std=torch.linspace(.3, 1.3, genes))
                              for name in ('spliced', 'unspliced')}
    checkpoint = config['gfg_checkpoint']
    reference.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    port = GFGCore().eval()
    port.load_pretrained(checkpoint)
    reference.eval()
    torch.set_num_threads(2)
    torch.manual_seed(19)
    us = torch.rand(3, genes*2)*2
    u, s = us.chunk(2, -1)
    states = reference.layer_states
    with torch.no_grad():
        x, velocity, *_ = reference(us)
        result = port(u, s, states['unspliced']['mean'], states['unspliced']['std'],
                      states['spliced']['mean'], states['spliced']['std'])
    rows = []
    for name, actual, expected in [('reconstruction', torch.cat((result['uhat'], result['shat']), -1), x),
                                   ('JVP_velocity', torch.cat((result['vu'], result['vs']), -1), velocity)]:
        error = float((actual-expected).abs().max())
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        rows.append(dict(check=name, max_absolute_error=error, passed=True))
    with Run(output, stage='GFG_reference_source_parity', kind='engineering', config=config,
             inputs=[*files, checkpoint, Path(__file__)], seed=19) as run:
        save_csv(run.directory/'metrics.csv', rows)
        summary = dict(status='GFG_REFERENCE_PARITY_PASS', checks=rows,
            checkpoint_origin='local_MouseBrain_seed0', shared_environment_modified=False,
            adaptations=['dropout disabled for a single deterministic decoder/JVP map',
                         'training normalization refitted on RENGE source genes',
                         'ODE solve float64 and relative RNA-scale loss; no moments smoothing',
                         'router receives actual projected JVP velocity and backpropagates into GFG'])
        save_json(run.directory/'summary.json', summary)
        (run.directory/'RESULTS.md').write_text('# GFG reference parity\n\n'+json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    print(json.dumps(audit('outputs/veloroute_gfg_reference_parity_20260914')))
