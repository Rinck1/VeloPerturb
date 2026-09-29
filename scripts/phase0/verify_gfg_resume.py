"""Compare restored GFG training with uninterrupted same-seed training, not scores."""
import argparse
import json
from pathlib import Path
import torch

from veloroute.artifacts import Run, read_csv, save_csv, save_json

parser = argparse.ArgumentParser()
parser.add_argument('--original', required=True)
parser.add_argument('--resumed', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
a, b = Path(args.original), Path(args.resumed)
paths = [p/name for p in (a, b) for name in ('model.pt', 'training_trace.csv', 'provenance.json')]
for path in (a, b):
    if json.loads((path/'provenance.json').read_text())['status'] != 'complete':
        raise ValueError('Both training runs must have completed')
with Run(args.output, stage='GFG_actual_checkpoint_resume_equality', kind='engineering',
         config=dict(criterion='tensor_and_training_trace_equality_not_downstream_selection'), inputs=paths) as run:
    left = torch.load(a/'model.pt', map_location='cpu', weights_only=True)
    right = torch.load(b/'model.pt', map_location='cpu', weights_only=True)
    assert left['config'] == right['config'] and left['metadata'] == right['metadata']
    rows = []
    for name, value in left['model'].items():
        other = right['model'][name]
        delta = float((value.double()-other.double()).abs().max()) if value.numel() else 0.
        rows.append(dict(tensor=name, exact=torch.equal(value, other), maximum_absolute_difference=delta))
        torch.testing.assert_close(value, other, atol=1e-6, rtol=1e-5)
    trace_equal = read_csv(a/'training_trace.csv') == read_csv(b/'training_trace.csv')
    result = dict(status='GFG_ACTUAL_RESUME_PASS', tensors=len(rows),
        exact_tensors=sum(r['exact'] for r in rows), max_absolute_difference=max(r['maximum_absolute_difference'] for r in rows),
        training_trace_exact=trace_equal, full_future_scores_read=False)
    if not trace_equal:
        raise AssertionError('Resumed trace differs from uninterrupted run')
    save_csv(run.directory/'metrics.csv', rows)
    save_json(run.directory/'summary.json', result)
    (run.directory/'RESULTS.md').write_text('# Actual GFG resume verification\n\n'+json.dumps(result, indent=2))
print(json.dumps(result))
