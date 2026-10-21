from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w15i_followup_queue as module


def nvidia(monkeypatch, process_rows, memory):
    def query(command, **kwargs):
        if '--query-gpu=index,uuid,memory.used' in command:
            return f'0, GPU-approved-zero, {memory}\n6, GPU-reserved-six, 0\n'
        assert '--query-compute-apps=gpu_uuid,pid' in command
        return process_rows
    monkeypatch.setattr(module.subprocess, 'check_output', query)


def test_idle_gpu0_is_accepted_without_using_reserved_cards(monkeypatch):
    nvidia(monkeypatch, 'GPU-reserved-six, 123\n', 4)
    status = module.gpu0_status()
    assert status['physical_gpu'] == 0 and status['free']
    assert not status['active_pids']


def test_active_or_high_memory_gpu_is_not_accepted(monkeypatch):
    nvidia(monkeypatch, 'GPU-approved-zero, 234\n', 4)
    assert not module.gpu0_status()['free']
    nvidia(monkeypatch, '', 1024)
    assert not module.gpu0_status()['free']


def test_child_environment_restores_cuda_and_compatible_cudnn(tmp_path, monkeypatch):
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    env = module.worker_environment(tmp_path/'runtime')
    assert env['JAX_PLATFORMS'] == 'cuda'
    assert env['XLA_PYTHON_CLIENT_PREALLOCATE'] == 'false'
    assert env['LD_LIBRARY_PATH'].endswith('cuda_deps/nvidia/cudnn/lib')
    assert Path(env['TMPDIR']).is_dir()
