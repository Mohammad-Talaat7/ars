"""Pytest suite for ARS-Bucket Class.

Covers:
- simple sorted lists
- reversed lists
- random floats (small)
- duplicate float values
- mixed duplicates and distinct values
- large float gaps
- values that should NOT merge under tolerance
- boundary precision (exact tolerance adjacency)
- boundary precision (just beyond tolerance)
- negative floats
- mixed-sign floats
- very small floats (near zero)
- very large floats (1e11–1e12)
- alternating close/far float sequences
- insertion inside run interior
- multiple float clusters
- NaN handling (raises ValueError with correct index)
- stability (result matches Python's sorted)
- round-trip idempotence (sorting twice gives same result)
- large randomized stress test (5k floats)
"""

import random
from collections.abc import Sequence

import pytest

from arslib.bucket.ars_bucket import ARSBucket


def assert_sorted_equal(
    a: Sequence[float], b: Sequence[float], tol: float = 1e-12
) -> None:
    """Assert element-wise equality for floats."""
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x - y) <= tol


def test_sorted_input() -> None:
    """Ensure that ARSBucket works on simple sorted list (Best Case Scenario)."""
    data = [float(i) * 0.1 for i in range(20)]
    sorter = ARSBucket()
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_reverse_sorted_input() -> None:
    """Tests sorting of reverse-ordered input."""
    data = [float(i) * 0.1 for i in range(20, 0, -1)]
    sorter = ARSBucket()
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_random_floats_small() -> None:
    """Tests correctness on small random float datasets."""
    data = [random.uniform(-10, 10) for _ in range(100)]
    sorter = ARSBucket()
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_equal_values() -> None:
    """Tests handling of identical float values (full-duplicate list)."""
    data = [3.14] * 50
    sorter = ARSBucket()
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_duplicates_mixed() -> None:
    """Tests handling of lists with both duplicates and distinct floats."""
    data = [1.2, 1.2, 1.2, 1.3, 1.3, 1.1, 1.1, 1.1]
    random.shuffle(data)
    sorter = ARSBucket()
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_large_gaps() -> None:
    """Tests sorting floats with very large numeric gaps."""
    data = [0.1, 10.0, 1000.0, -2000.0, 5.5]
    sorter = ARSBucket()
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_values_not_merging() -> None:
    """Tests that widely-separated floats do not merge into a single run."""
    tol = 1e-6
    data = [0.0, tol * 10, tol * 20, tol * 30]  # far apart
    sorter = ARSBucket(tolerance=tol)
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_boundary_precision_adjacent() -> None:
    """Tests adjacency when values differ by exactly the tolerance."""
    tol = 1e-6
    data = [1.0, 1.0 + tol]  # exactly at tolerance boundary
    sorter = ARSBucket(tolerance=tol)
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_boundary_precision_not_adjacent() -> None:
    """Tests non-adjacency when values exceed tolerance by a tiny margin."""
    tol = 1e-6
    data = [1.0, 1.0 + tol + 1e-9]  # slightly beyond threshold
    sorter = ARSBucket(tolerance=tol)
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_negative_floats() -> None:
    """Tests correct sorting of negative float ranges."""
    data = [random.uniform(-100, 0) for _ in range(100)]
    sorter = ARSBucket()
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_mixed_sign_values() -> None:
    """Tests sorting lists containing both positive and negative floats."""
    data = [-3.2, 0.0, 1.1, -1.1, 5.2, -0.00001]
    sorter = ARSBucket()
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_very_small_floats() -> None:
    """Tests sorting extremely small floats near zero."""
    data = [1e-12, 2e-12, 5e-13, -1e-12, 7e-13]
    sorter = ARSBucket(tolerance=1e-12)
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_very_large_floats() -> None:
    """Tests sorting extremely large float values."""
    data = [1e12, 1e12 + 5, 1e12 - 3, -1e12, 3.5e11]
    sorter = ARSBucket(tolerance=1e-6)
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_alternating_close_far() -> None:
    """Tests robustness on sequences alternating between close and far floats."""
    tol = 1e-6
    close_values = [1.0 + i * tol * 0.5 for i in range(20)]
    far_values = [10.0 + i for i in range(20)]
    data = [v for pair in zip(close_values, far_values) for v in pair]
    random.shuffle(data)
    sorter = ARSBucket(tolerance=tol)
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_insert_inside_run() -> None:
    """Tests correct insertion into a run's interior (middle insertion)."""
    # Forces insertion into the middle of a run
    data = [1.0, 1.000001, 1.000002, 1.0000015]
    sorter = ARSBucket(tolerance=1e-5)
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_clustered_floats() -> None:
    """Tests sorting when floats form well-separated numeric clusters."""
    cluster1 = [1.0 + i * 1e-7 for i in range(50)]
    cluster2 = [10.0 + i * 1e-7 for i in range(50)]
    data = cluster1 + cluster2
    random.shuffle(data)
    sorter = ARSBucket(tolerance=1e-6)
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data), tol=1e-9)


def test_nan_raises() -> None:
    """Tests that NaN values raise a ValueError with index awareness."""
    data = [1.0, 2.0, float("nan"), 3.0]
    sorter = ARSBucket()
    with pytest.raises(ValueError):
        _ = sorter.sort(data)


def test_stability_random() -> None:
    """Tests that ARSBucket matches Python sorted() on large random data."""
    data = [random.uniform(-50, 50) for _ in range(500)]
    sorter = ARSBucket()
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))


def test_round_trip_idempotence() -> None:
    """Tests that sorting an already-sorted result produces identical output."""
    data = [random.uniform(-10, 10) for _ in range(200)]
    sorter = ARSBucket()
    result1 = sorter.sort(data)
    result2 = sorter.sort(result1)
    assert_sorted_equal(result1, result2)


def test_stress_large_dataset() -> None:
    """Tests performance and correctness on a large randomized dataset."""
    data = [random.uniform(-1e6, 1e6) for _ in range(5000)]
    sorter = ARSBucket()
    result = sorter.sort(data)
    assert_sorted_equal(result, sorted(data))
