import numpy as np
import pytest

from resfrac import HolographicSublinearIndex


def test_import_and_fit():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(500, 4))
    idx = HolographicSublinearIndex(K=64, seed=123).fit(X)
    st = idx.stats()
    assert st is not None
    assert st.n == 500
    assert st.d == 4
    assert np.isfinite(st.h_min)
    assert np.isfinite(st.h_max)


def test_encode_and_query_window_size():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 5))
    idx = HolographicSublinearIndex(K=64, seed=7).fit(X)
    q = rng.normal(size=(5,))
    S = 25
    cands, meta = idx.query(q, S=S, gate=False)
    assert cands.shape[1] == 5
    assert 1 <= cands.shape[0] <= S
    assert 'loc' in meta and 'idx' in meta


def test_evaluate_callback_returns_index():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(200, 3))
    idx = HolographicSublinearIndex(K=64, seed=99).fit(X)
    q = rng.normal(size=(3,))

    def eval_nn(C, qq):
        d2 = np.sum((C - qq) ** 2, axis=1)
        return int(np.argmin(d2))

    j = idx.query(q, S=20, evaluate=eval_nn, gate=False)[0]
    assert isinstance(j, int)
    assert 0 <= j < 20


def _brute_nn(X, q):
    d2 = np.sum((X - q) ** 2, axis=1)
    return int(np.argmin(d2))


def test_full_window_contains_true_nn():
    rng = np.random.default_rng(2024)
    N, d = 800, 6
    X = rng.normal(size=(N, d))
    idx = HolographicSublinearIndex(K=96, seed=2024).fit(X)
    q = rng.normal(size=(d,))

    true_i = _brute_nn(X, q)
    cands, meta = idx.query(q, S=N, gate=False, return_indices=True)
    # When S >= N, all indices should be present
    assert len(cands) == N
    assert true_i in set(cands.tolist())


def test_gating_reject_forces_empty():
    rng = np.random.default_rng(555)
    X = rng.normal(size=(300, 5))
    idx = HolographicSublinearIndex(K=64, seed=555).fit(X)
    q = rng.normal(size=(5,))
    cands, meta = idx.query(q, S=20, gate=True, gate_mode='reject', gate_tol=-1.0)
    assert meta['gated'] is True
    assert cands is None or (hasattr(cands, 'size') and cands.size == 0)
