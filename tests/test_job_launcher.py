import json
from pathlib import Path
import subprocess
import sys
import pytest

LAUNCHER = Path(__file__).resolve().parents[1]/'scripts/phase0/launch_gpu_job.py'


def test_cpu_worker_clears_cuda_without_gpu_query(tmp_path):
    (tmp_path/'job.json').write_text(json.dumps(dict(physical_gpu=None, gpu_uuid=None,
        command=[sys.executable, '-c', 'import os; assert os.environ["CUDA_VISIBLE_DEVICES"] == ""'])))
    subprocess.run([sys.executable, str(LAUNCHER), '--worker', str(tmp_path)], check=True)
    status = json.loads((tmp_path/'status.json').read_text())
    assert status['status'] == 'complete'
    assert status['physical_gpu'] is None


@pytest.mark.parametrize('options', [[], ['--gpu', '4'], ['--gpu', '0', '--cpu']])
def test_launcher_rejects_missing_reserved_or_conflicting_devices(options):
    result = subprocess.run([sys.executable, str(LAUNCHER), *options], capture_output=True)
    assert result.returncode != 0
