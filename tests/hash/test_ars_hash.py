"""Pytest suite for ARS-Hash Class.

Covers:
- simple sorted lists
- reversed list
- duplicates
- random lists
- negative integers
- 0-based & large numbers
- stability (count-preservation)
- large stress test
"""

import random
from collections.abc import Generator

import pytest

from arslib.hash.ars_hash import ARSHash


@pytest.fixture(autouse=True)
def deterministic_seed() -> Generator[None, None, None]:
    """Ensure deterministic behaviour for tests that use randomness."""
    random.seed(42)
    yield
    random.seed()


def run_sort_and_compare(arr: list[int]) -> None:
    """Test ARSHash sorting logic."""
    sorter = ARSHash()
    out = sorter.sort(arr)
    assert out == sorted(arr)


def test_simple_sorted() -> None:
    """Ensure that ARSHash works on simple sorted list (Best Case Scenario)."""
    data = [1, 2, 3, 4, 5]
    run_sort_and_compare(data)


def test_reversed() -> None:
    """Ensure that ARSHash works on simple reversed list."""
    data = [5, 4, 3, 2, 1]
    run_sort_and_compare(data)


def test_duplicates() -> None:
    """Ensure that ARSHash handle duplicates."""
    data = [3, 1, 2, 3, 2, 1, 1]
    run_sort_and_compare(data)


def test_random_lists_many_small() -> None:
    """Ensure that ARSHash works on random small lists."""
    for _ in range(200):
        n = random.randint(0, 50)
        arr = [random.randint(-20, 20) for _ in range(n)]
        run_sort_and_compare(arr)


def test_negative_integers_and_zero() -> None:
    """Ensure ARSHash handle negative integers and zero."""
    data = [-3, -1, -2, -2, 0, 0, -1]
    run_sort_and_compare(data)


def test_zero_based_and_large_numbers() -> None:
    """Ensure ARSHash handle zero-based and large numbers."""
    data = [0, 10**6, -(10**6), 999999999, 0, 2**31 - 1, -(2**31)]
    run_sort_and_compare(data)


def test_stability_count_preservation() -> None:
    """Ensure ARSHash Stability."""
    data = [2, 1, 2, 1, 2, 1, 1]
    run_sort_and_compare(data)
    out = ARSHash().sort(data)
    assert out.count(1) == data.count(1)
    assert out.count(2) == data.count(2)


def test_large_stress_medium() -> None:
    """Medium Stress Test for ARSHash."""
    n = 5000
    arr = [random.randint(-1000, 1000) for _ in range(n)]
    run_sort_and_compare(arr)


def test_large_stress_heavy() -> None:
    """Heavy Stress Test for ARSHash."""
    n = 50000
    arr = [random.randint(-5000, 5000) for _ in range(n)]
    run_sort_and_compare(arr)


def test_mixed_small_and_large_gaps() -> None:
    """Mix of Clusters and Large Gaps."""
    # Mix of clusters and large gaps
    arr: list[int] = []
    for base in [0, 1000, 2000000, -500000]:
        arr.extend([base + i for i in range(5)])  # small runs
    # add some random noise
    arr += [random.randint(-10000000, 10000000) for _ in range(50)]
    random.shuffle(arr)
    run_sort_and_compare(arr)
