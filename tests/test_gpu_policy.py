import pytest
from veloroute import gpu_policy


def test_cpu_is_not_bound_to_GPU_visibility(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    assert gpu_policy.enforce_gpu_policy('cpu') == ()


@pytest.mark.parametrize('value', ['', '6', '0,7', '0,1,2,3,4,5,6', '0,0'])
def test_extra_or_implicit_GPUs_are_forbidden(monkeypatch, value):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', value)
    with pytest.raises(ValueError):
        gpu_policy.enforce_gpu_policy('cuda')


def test_six_card_limit_and_UUID_mapping(monkeypatch):
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5')
    assert gpu_policy.enforce_gpu_policy('cuda') == (0, 1, 2, 3, 4, 5)
    monkeypatch.setattr(gpu_policy, 'gpu_uuids', lambda: {5: 'GPU-allowed', 6: 'GPU-reserved'})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-allowed')
    assert gpu_policy.enforce_gpu_policy('cuda') == (5,)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-reserved')
    with pytest.raises(ValueError):
        gpu_policy.enforce_gpu_policy('cuda')
