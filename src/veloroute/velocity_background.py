"""Read only versioned, source-aligned training-fit velocity backgrounds."""
import numpy as np

BACKGROUND_FORMAT = 'training_fit_background_v2'


def load_background(path, role, *, cell_ids=None):
    if role not in {'train', 'validation'}:
        raise ValueError('Unknown background role')
    with np.load(path, allow_pickle=False) as data:
        if 'format' not in data or str(data['format'].item()) != BACKGROUND_FORMAT:
            raise ValueError('Legacy/unversioned velocity background: regenerate with training-only normalization')
        values = {key: data[f'{role}_{key}_bg'].copy() for key in ('r', 'v')}
        values['cell_ids'] = data[f'{role}_cell_ids'].copy()
        if len(set(values['cell_ids'].tolist())) != len(values['cell_ids']):
            raise ValueError('Duplicate background cell IDs')
        for key in ('r', 'v'):
            if values[key].ndim != 2 or len(values[key]) != len(values['cell_ids']) or not np.isfinite(values[key]).all():
                raise ValueError('Invalid background arrays')
        if cell_ids is not None and list(cell_ids) != values['cell_ids'].tolist():
            raise ValueError('Velocity background/source cell order mismatch')
        return values
