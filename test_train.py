# pyright: reportMissingImports=false
"""Correctness checks for the training harness.

Run: uv run --with mlx --with pytest pytest test_train.py
"""
import mlx.core as mx

import train


def test_median_odd_and_even():
    assert train.median([3, 1, 2]) == 2
    assert train.median([4, 1, 3, 2]) == 2.5


def test_rand_batch_is_next_token_shifted():
    x, y = train.rand_batch(2, 5, vocab=50)
    assert x.shape == (2, 5) and y.shape == (2, 5)
    # targets are inputs shifted by one (next-token prediction)
    assert mx.array_equal(x[:, 1:], y[:, :-1])


def test_training_loop_learns():
    # Overfitting a fixed batch must drive the loss down substantially.
    first, last = train.correctness_check(steps=80)
    assert last < first * 0.5
