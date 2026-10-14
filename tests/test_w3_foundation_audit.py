import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/phase0'))
from w3_7_foundation_audit import full_library_log_cpm


def test_pretrained_normalization_uses_full_transcriptome_denominator():
    counts = np.array([[10., 20.]])
    np.testing.assert_allclose(full_library_log_cpm(counts, [1000]), np.log1p([[100., 200.]]))
    np.testing.assert_array_equal(full_library_log_cpm(np.zeros((1, 2)), [1000]), [[0., 0.]])
    with pytest.raises(ValueError, match='exceed'):
        full_library_log_cpm(counts, [25])
