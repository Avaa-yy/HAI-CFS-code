"""Random splitting utilities used by the HAI-CFS analysis."""

import numpy as np


def random_train_test_split_indices(indices, fit_ratio, rng):
    """Randomly split row indices without consulting labels or outcomes."""
    if not 0 < fit_ratio < 1:
        raise ValueError('fit_ratio must be between 0 and 1.')

    shuffled_indices = rng.permutation(np.asarray(indices))
    n_train = int(len(shuffled_indices) * fit_ratio)
    if n_train < 3 or len(shuffled_indices) - n_train < 2:
        raise ValueError('Not enough rows for the requested train-test split.')

    return shuffled_indices[:n_train], shuffled_indices[n_train:]
