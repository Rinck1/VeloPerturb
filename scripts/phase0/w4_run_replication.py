"""Run the fixed H1 predict->evaluate workflow without manual handoff."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from w4_artifacts import Artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-lock', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    runner = Path(__file__).with_name('w4_replication.py')
    run = Artifacts(args.output, seed=[0,1,2,3], inputs=[Path(__file__), runner, Path(args.protocol_lock)])
    try:
        for stage in ('predict', 'evaluate'):
            command = [sys.executable, str(runner), stage, '--protocol-lock', args.protocol_lock,
                       '--output', args.predictions if stage == 'predict' else args.evaluation]
            if stage == 'evaluate':
                command += ['--predictions', args.predictions]
            run.json('status.json', dict(status=stage, command=command))
            subprocess.run(command, check=True, timeout=21600)
        run.json('status.json', dict(status='complete', predictions=args.predictions, evaluation=args.evaluation))
    except BaseException as error:
        run.json('status.json', dict(status='failed', error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
