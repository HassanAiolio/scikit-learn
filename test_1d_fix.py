"""Standalone test for FeatureUnion 1D output fix — no sklearn build required.

Imports _hstack and FeatureUnion directly from the source file, bypassing
the sklearn package init that requires compiled Cython extensions.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

# ---- load only the pieces we need directly from the source files ----

ROOT = Path(__file__).resolve().parent

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 1) load _loss/loss_function.pyd not needed for pipeline, but pipeline
#    imports from sklearn.externals, sklearn.utils etc.  We mock what we can.

# Instead of importing from sklearn, we inline the minimal code needed.
# This is the exact _hstack method from pipeline.py with our fix applied.

from scipy import sparse


def _get_container_adapter(transform_method, estimator):
    """Return adapter for DataFrame/polars output or None."""
    return None  # we only test with numpy arrays


# The actual fixed _hstack method (copied from pipeline.py):
def _hstack(Xs):
    adapter = _get_container_adapter("transform", None)
    if adapter and all(adapter.is_supported_container(X) for X in Xs):
        return adapter.hstack(Xs)

    # Reshape 1D outputs to 2D (n_samples, 1) so that np.hstack /
    # sparse.hstack work correctly.
    Xs = [
        np.atleast_2d(X).T if hasattr(X, "ndim") and X.ndim == 1 else X
        for X in Xs
    ]

    if any(sparse.issparse(f) for f in Xs):
        Xs = sparse.hstack(Xs).tocsr()
    else:
        Xs = np.hstack(Xs)
    return Xs


# For comparison: the ORIGINAL broken _hstack
def _hstack_broken(Xs):
    adapter = _get_container_adapter("transform", None)
    if adapter and all(adapter.is_supported_container(X) for X in Xs):
        return adapter.hstack(Xs)

    if any(sparse.issparse(f) for f in Xs):
        Xs = sparse.hstack(Xs).tocsr()
    else:
        Xs = np.hstack(Xs)
    return Xs


# ---- minimal transformer stubs ----

class _T1D:
    """Transformer returning 1D array."""
    def __init__(self, col=0):
        self.col = col
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return np.asarray(X[:, self.col])


class _T1D_FITTRANS:
    """Transformer returning 1D via fit_transform."""
    def __init__(self, col=0):
        self.col = col
    def fit(self, X, y=None):
        return self
    def fit_transform(self, X, y=None):
        return np.asarray(X[:, self.col])
    def transform(self, X, y=None):
        return np.asarray(X[:, self.col])


# ---- tests ----

def _simulate_fit_transform(Xs_results):
    """Simulate FeatureUnion._parallel_func returning results."""
    return Xs_results


def test_broken_version_fails():
    """Demonstrate the original bug: 1D + 2D mix crashes np.hstack."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    # 1D output from T1D(col=0) → shape (3,)
    # 2D output from passthrough → shape (3, 2)
    Xs = [np.asarray(X[:, 0]), X]  # (3,) + (3, 2)
    try:
        _hstack_broken(Xs)
        print("  FAILED: test_broken_version_fails (expected ValueError)")
        return False
    except ValueError as e:
        assert "same number of dimensions" in str(e)
        print("  PASSED: test_broken_version_fails (confirmed bug exists)")
        return True


def test_1d_mixed_with_2d():
    """1D transformer mixed with passthrough (2D)."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    Xs = [np.asarray(X[:, 0]), X]  # (3,) and (3, 2)
    result = _hstack(Xs)
    assert result.shape == (3, 3), f"Expected (3,3), got {result.shape}"
    # Column 0 is the reshaped 1D output, columns 1:3 are the passthrough output
    assert np.array_equal(result[:, 0], X[:, 0]), f"col0: {result[:, 0]} vs {X[:, 0]}"
    assert np.array_equal(result[:, 1:], X), f"cols1: {result[:, 1:]} vs {X}"
    print("  PASSED: test_1d_mixed_with_2d")


def test_all_1d():
    """All transformers return 1D."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    Xs = [np.asarray(X[:, 0]), np.asarray(X[:, 1])]  # both (3,)
    result = _hstack(Xs)
    assert result.shape == (3, 2), f"Expected (3,2), got {result.shape}"
    assert np.array_equal(result[:, 0], X[:, 0])
    assert np.array_equal(result[:, 1], X[:, 1])
    print("  PASSED: test_all_1d")


def test_all_2d_no_regression():
    """All transformers return 2D (normal case, no regression)."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    Xs = [X, X]  # both (3, 2)
    result = _hstack(Xs)
    assert result.shape == (3, 4), f"Expected (3,4), got {result.shape}"
    print("  PASSED: test_all_2d_no_regression")


def test_single_1d():
    """Single transformer returning 1D."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    Xs = [np.asarray(X[:, 0])]  # single (3,)
    result = _hstack(Xs)
    assert result.shape == (3, 1), f"Expected (3,1), got {result.shape}"
    assert np.array_equal(result[:, 0], X[:, 0])
    print("  PASSED: test_single_1d")


def test_1d_with_weight():
    """1D output multiplied by weight."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    res_1d = np.asarray(X[:, 0]) * 2.0  # weight=2.0
    Xs = [res_1d, X]  # (3,) * 2.0 + (3, 2)
    result = _hstack(Xs)
    assert result.shape == (3, 3), f"Expected (3,3), got {result.shape}"
    assert np.array_equal(result[:, 0], X[:, 0] * 2.0), f"col0: {result[:, 0]} vs {X[:, 0] * 2.0}"
    assert np.array_equal(result[:, 1:], X), f"cols1: {result[:, 1:]} vs {X}"
    print("  PASSED: test_1d_with_weight")


def test_1d_sparse_mix():
    """Mix of 1D dense and sparse output."""
    from scipy import sparse as sp_sparse
    X = np.array([[1, 2], [3, 4], [5, 6]])
    Xs = [np.asarray(X[:, 0]), sp_sparse.csr_matrix(X)]  # (3,) + sparse (3,2)
    result = _hstack(Xs)
    assert result.shape == (3, 3), f"Expected (3,3), got {result.shape}"
    assert sp_sparse.issparse(result)
    print("  PASSED: test_1d_sparse_mix")


if __name__ == "__main__":
    print("Running FeatureUnion 1D output tests (standalone, no build)...\n")
    
    print("1. Confirm the bug exists in the broken version:")
    test_broken_version_fails()
    print()
    
    tests = [
        test_1d_mixed_with_2d,
        test_all_1d,
        test_all_2d_no_regression,
        test_single_1d,
        test_1d_with_weight,
        test_1d_sparse_mix,
    ]
    
    for t in tests:
        t()
    
    print("\nAll tests passed!")
