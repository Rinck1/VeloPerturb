"""User resource contract: this project may use physical GPUs 0-3 only."""
import os
import subprocess
from functools import lru_cache

ALLOWED_PHYSICAL_GPUS = (0, 1, 2, 3)


@lru_cache(maxsize=1)
def gpu_uuids():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
        check=True, capture_output=True, text=True, timeout=15)
    return {int(row.split(',')[0].strip()): row.split(',')[1].strip()
            for row in result.stdout.strip().splitlines()}


def enforce_gpu_policy(device):
    if str(device).split(':')[0] != 'cuda':
        return ()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    tokens = [x.strip() for x in visible.split(',') if x.strip()]
    if not tokens or len(tokens) > 4 or len(set(tokens)) != len(tokens):
        raise ValueError('Explicit CUDA_VISIBLE_DEVICES within physical GPUs 0-3 is required; reserve GPUs 4-7')
    if all(x.isdigit() for x in tokens):
        if os.environ.get('CUDA_DEVICE_ORDER', 'PCI_BUS_ID') != 'PCI_BUS_ID':
            raise ValueError('Physical GPU indices require CUDA_DEVICE_ORDER=PCI_BUS_ID')
        indices = tuple(int(x) for x in tokens)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    else:
        lookup = {uuid: index for index, uuid in gpu_uuids().items()}
        if any(token not in lookup for token in tokens):
            raise ValueError('Unknown CUDA GPU UUID; cannot verify resource reservation')
        indices = tuple(lookup[token] for token in tokens)
    if not set(indices) <= set(ALLOWED_PHYSICAL_GPUS):
        raise ValueError('This project is restricted to physical GPUs 0-3; GPUs 4-7 are reserved')
    return indices
