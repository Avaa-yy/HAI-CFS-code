import numpy as np
import pytest

from scripts.split_utils import random_train_test_split_indices


def split(seed, fit_ratio=0.4, n_rows=137):
    return random_train_test_split_indices(
        np.arange(n_rows),
        fit_ratio,
        np.random.default_rng(seed),
    )


def test_same_seed_reproduces_same_split():
    train_a, test_a = split(42)
    train_b, test_b = split(42)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(test_a, test_b)


def test_different_seeds_change_the_split():
    train_a, test_a = split(42)
    train_b, test_b = split(43)
    assert not np.array_equal(train_a, train_b)
    assert not np.array_equal(test_a, test_b)


def test_train_and_test_are_disjoint_and_complete():
    train, test = split(42)
    assert len(train) == 54
    assert len(test) == 83
    assert np.intersect1d(train, test).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate([train, test])),
        np.arange(137),
    )


def test_split_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        split(42, fit_ratio=0)

    with pytest.raises(ValueError):
        split(42, fit_ratio=1)
