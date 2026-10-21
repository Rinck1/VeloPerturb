"""User added physical GPU7 on 2026-10-21; GPU6 remains reserved.

Keep the earlier six-device visibility limit. Adding an approved alternative
device is not permission to expose every card at once.
"""
import os
import subprocess
from functools import lru_cache

ALLOWED_PHYSICAL_GPUS = (0, 1, 2, 3, 4, 5, 7)
RESERVED_PHYSICAL_GPUS = (6,)
MAX_VISIBLE_GPUS = 6


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
    if not tokens or len(tokens) > MAX_VISIBLE_GPUS or len(set(tokens)) != len(tokens):
        raise ValueError('Explicit CUDA_VISIBLE_DEVICES with at most six approved GPUs (0-5 or 7) is required; GPU6 is reserved')
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
        raise ValueError('This project is restricted to physical GPUs 0-5 or 7; GPU6 is reserved')
    return indices
