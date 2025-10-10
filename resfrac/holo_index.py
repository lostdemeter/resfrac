# resfrac/holo_index.py
# Holographic Sublinear Index: turns O(N) search into O(log N + S) via zeta-fringe encoding
#
# API
# - HolographicSublinearIndex(K=256, seed=42, normalize="auto", per_dim_phases=True)
# - fit(X[, weights]): X shape (N,) or (N, d); builds sorted fringe index
# - encode(q): returns fringe value h(q)
# - query(q, S=20, evaluate=None, return_indices=False, gate=False):
#       returns candidate indices or evaluate(candidates, q) result
# - stats(): quick summary
# - coherence_stats(values): phase variance + gap entropy + invariant proxy
#
# Notes
# - Zeros: uses get_zeta_fiducials(K). Phase keys are random per-dimension by default.
# - Vector data: h(x) = sum_j w_j * Re( mean_k exp(i (gamma_k * (x_j/Λ_j) + theta_{k,j})) )
# - Sorting: stores permutation and sorted h for binary search.
# - Coherence gating: optional; uses phase variance and a lightweight invariant.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from .holo_utils import get_zeta_fiducials, phase_retrieve, log_phi, fresnel_propagate2d

# Simple in-process cache for zeta zeros to avoid repeated costly computation
_GAMMA_CACHE: dict[int, np.ndarray] = {}

def _get_gammas_cached(K: int, dtype: str = "float32") -> np.ndarray:
    arr = _GAMMA_CACHE.get(int(K))
    if arr is None:
        g = get_zeta_fiducials(int(K))
        arr = np.asarray(g, dtype=np.float32 if str(dtype) == 'float32' else np.float64)
        _GAMMA_CACHE[int(K)] = arr
    return arr.astype(np.float32 if str(dtype) == 'float32' else np.float64, copy=False)

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class HoloIndexStats:
    n: int
    d: int
    K: int
    h_min: float
    h_max: float
    phase_var: Optional[float]
    gap_entropy: Optional[float]
    gap_entropy_norm: Optional[float]
    invariant: Optional[float]


class HolographicSublinearIndex:
    """
    Holographic Sublinear Index

    - Builds a 1D "fringe" index h(x) projected from input X via zeta-zero oscillators
      with random phases for ergodicity.
    - Supports scalar or vector X. For vectors, uses per-dimension random phases and
      optional per-dimension weights.
    - Query returns a local window of size S around the binary-search insertion point
      of the query fringe value h(q), enabling O(log N + S) approximate search.
    - Optional coherence gating via phase variance and a lightweight invariant.
    """

    def __init__(
        self,
        K: int = 256,
        seed: Optional[int] = 42,
        normalize: str = "auto",
        per_dim_phases: bool = True,
        gate_auto: bool = True,
        gate_quantile: float = 0.85,
        dtype: str = "float32",
        chunk_k: int = 0,
        prop: str = "none",
    ):
        self.K = int(K)
        self.seed = None if seed is None else int(seed)
        self.normalize = str(normalize)
        self.per_dim_phases = bool(per_dim_phases)
        self.gate_auto = bool(gate_auto)
        self.gate_quantile = float(gate_quantile)
        self._dtype = np.float32 if str(dtype).lower() == 'float32' else np.float64
        self._chunk_k = max(0, int(chunk_k))
        self._prop = str(prop).lower()

        self._rng = np.random.default_rng(self.seed)
        self._gammas = _get_gammas_cached(self.K, dtype='float32' if self._dtype==np.float32 else 'float64')  # shape (K,)
        # set later on fit once d is known
        self._theta = None  # shape (K, d) or (K, 1)
        self._w = None      # shape (d,)

        # dataset cache and index
        self._X = None
        self._X_min = None
        self._X_max = None
        self._d = None

        self._h = None              # shape (N,)
        self._perm = None           # permutation indices (argsort of h)
        self._h_sorted = None       # sorted h

        self._pre_stats: Optional[HoloIndexStats] = None
        self._gate_thr_pvar: Optional[float] = None
        self._gate_thr_Hn: Optional[float] = None

    # -----------------
    # Public API
    # -----------------

    def fit(self, X: ArrayLike, weights: Optional[ArrayLike] = None) -> "HolographicSublinearIndex":
        X = np.asarray(X, dtype=self._dtype)
        if X.ndim == 1:
            X = X[:, None]
        N, d = X.shape
        self._X = X
        self._d = d
        # normalization: map each dim to [0,1] (auto), or leave as-is (none)
        if self.normalize == "auto":
            self._X_min = X.min(axis=0)
            self._X_max = X.max(axis=0)
            span = self._X_max - self._X_min
            span[span == 0.0] = 1.0
            Xn = (X - self._X_min) / span
        elif self.normalize == "none":
            self._X_min = None
            self._X_max = None
            Xn = X
        else:
            raise ValueError(f"Unknown normalize='{self.normalize}' (use 'auto' or 'none')")

        # weights
        if weights is None:
            self._w = np.full(d, 1.0 / max(1, d), dtype=self._dtype)
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if w.size != d:
                raise ValueError(f"weights must have length {d}, got {w.size}")
            if not np.any(np.isfinite(w)):
                raise ValueError("weights must be finite")
            self._w = w / (np.sum(np.abs(w)) + 1e-12)

        # phases
        if self.per_dim_phases:
            self._theta = self._rng.uniform(0.0, 2 * np.pi, size=(self.K, d)).astype(self._dtype)
        else:
            base = self._rng.uniform(0.0, 2 * np.pi, size=(self.K, 1)).astype(self._dtype)
            self._theta = np.repeat(base, d, axis=1).astype(self._dtype, copy=False)

        # encode fringes
        self._h = self._encode_matrix(Xn)  # (N,)
        self._perm = np.argsort(self._h)
        self._h_sorted = self._h[self._perm]

        # pre-fit stats
        self._pre_stats = self._compute_stats(self._h_sorted)
        # auto-gating thresholds from baseline windows
        if self.gate_auto:
            self._gate_thr_pvar, self._gate_thr_Hn = self._compute_gate_thresholds(self._h_sorted)
        return self

    def encode(self, q: ArrayLike) -> float:
        q = np.asarray(q, dtype=self._dtype)
        if q.ndim == 0:
            q = q[None]
        if self._d is None:
            raise RuntimeError("fit() must be called before encode()")
        if q.size != self._d:
            if q.size == 1 and self._d == 1:
                pass
            else:
                raise ValueError(f"query has dim {q.size}, expected {self._d}")
        qn = self._normalize_query(q)
        return float(self._encode_row(qn[None, :])[0])

    def query(
        self,
        q: ArrayLike,
        S: int = 20,
        evaluate: Optional[Callable[[np.ndarray, np.ndarray], object]] = None,
        return_indices: bool = False,
        gate: bool = False,
        gate_mode: str = "flag",
        gate_tol: float = 0.15,
        probe_offset: int = 0,
    ) -> Tuple[object, dict]:
        """
        Query the index with point q.

        - Returns (result, meta) where result is either indices, candidate rows, or
          the evaluation result if `evaluate` is provided.
        - meta includes: {'h_q': float, 'loc': int, 'idx': np.ndarray, 'gated': bool, 'stats': {...}}
        """
        if self._h_sorted is None:
            raise RuntimeError("fit() must be called before query()")
        q = np.asarray(q, dtype=float)
        if q.ndim == 0:
            q = q[None]
        if q.size != self._d:
            if q.size == 1 and self._d == 1:
                pass
            else:
                raise ValueError(f"query has dim {q.size}, expected {self._d}")
        hq = self.encode(q)
        loc = int(np.searchsorted(self._h_sorted, hq, side="left"))
        # apply probe offset in sorted space
        loc = int(np.clip(loc + int(probe_offset), 0, self._h_sorted.size))
        N = self._h_sorted.size
        half = max(1, int(S) // 2)
        lo = max(0, loc - half)
        hi = min(N, lo + int(S))
        lo = max(0, hi - int(S))
        window_idx_sorted = np.arange(lo, hi, dtype=int)
        candidate_idx = self._perm[window_idx_sorted]

        # coherence gating (optional)
        gated = False
        cand_stats = None
        if gate:
            cand_values = self._h[candidate_idx]
            cand_sorted = np.sort(cand_values)
            cand_stats = self._compute_stats(cand_sorted)
            if self._pre_stats is not None and cand_stats is not None:
                cand_pv = float(cand_stats.phase_var if cand_stats.phase_var is not None else 0.0)
                # Optional Fresnel propagation-based coherence proxy on a 2D occupancy grid
                if self._prop == 'fresnel':
                    try:
                        # Use ranks to build a compact aperture grid
                        ranks = np.argsort(np.argsort(cand_values))
                        G = int(max(16, min(64, 2 ** int(np.ceil(np.log2(np.sqrt(max(1, ranks.size))))))))
                        # Map ranks to grid coordinates
                        ys = (ranks // max(1, int(np.sqrt(max(1, ranks.size))))) % G
                        xs = (ranks % max(1, int(np.sqrt(max(1, ranks.size))))) % G
                        grid = np.zeros((G, G), dtype=float)
                        for yy, xx in zip(ys, xs):
                            grid[int(yy) % G, int(xx) % G] += 1.0
                        I = fresnel_propagate2d(grid, z=1.0, lambda_w=1.0, dx=1.0, dy=1.0)
                        # Use phase variance of propagated intensity as an additional measure; combine conservatively
                        _, pv_f = phase_retrieve(I.ravel())
                        cand_pv = min(cand_pv, float(pv_f))
                    except Exception:
                        pass
                cand_Hn = float(cand_stats.gap_entropy_norm if cand_stats.gap_entropy_norm is not None else 0.0)

                if self.gate_auto and (self._gate_thr_pvar is not None or self._gate_thr_Hn is not None):
                    # Auto thresholds from baseline distributions
                    var_thresh = self._gate_thr_pvar if self._gate_thr_pvar is not None else float('inf')
                    ent_thresh = self._gate_thr_Hn if self._gate_thr_Hn is not None else float('inf')
                else:
                    # Manual relative thresholds vs pre-fit
                    pre_pv = float(self._pre_stats.phase_var if self._pre_stats.phase_var is not None else 0.0)
                    pre_Hn = float(self._pre_stats.gap_entropy_norm if self._pre_stats.gap_entropy_norm is not None else 0.0)
                    var_thresh = pre_pv + float(gate_tol)
                    if gate_tol >= 0:
                        var_thresh = max(var_thresh, 0.35)
                    ent_thresh = pre_Hn + float(gate_tol)

                # Phase-variance gating
                cond_pvar = cand_pv > var_thresh
                # Entropy gating only for sufficiently large candidate window
                n_cand = int(candidate_idx.size)
                cond_ent = (n_cand >= 64) and (cand_Hn > ent_thresh)

                gated = bool(cond_pvar or cond_ent)

        # Apply gate action
        if gated and str(gate_mode).lower() == "reject":
            # Return empty results while signaling gating via meta
            if return_indices and evaluate is None:
                result = np.array([], dtype=int)
            elif evaluate is None:
                result = self._X[np.array([], dtype=int)]
            else:
                result = None
        else:
            if return_indices and evaluate is None:
                result = candidate_idx
            elif evaluate is None:
                result = self._X[candidate_idx]
            else:
                result = evaluate(self._X[candidate_idx], q)

        meta = {
            "h_q": float(hq),
            "loc": loc,
            "idx": candidate_idx,
            "gated": bool(gated),
            "stats": {
                "pre": self._pre_stats.__dict__ if self._pre_stats else None,
                "cand": cand_stats.__dict__ if cand_stats else None,
            },
        }
        return result, meta

    def stats(self) -> Optional[HoloIndexStats]:
        return self._pre_stats

    # -----------------
    # Internal helpers
    # -----------------

    def _normalize_query(self, q: np.ndarray) -> np.ndarray:
        if self.normalize == "auto" and self._X_min is not None and self._X_max is not None:
            span = (self._X_max - self._X_min).copy()
            span[span == 0.0] = 1.0
            return (q - self._X_min) / span
        return q

    def _encode_matrix(self, Xn: np.ndarray) -> np.ndarray:
        """Encode fringe values in dtype with optional K-chunking to reduce memory and overhead.

        Xn shape (N, d); theta (K, d); gammas (K,)
        """
        N, d = Xn.shape
        acc = np.zeros(N, dtype=self._dtype)
        K = int(self.K)
        ck = int(self._chunk_k)
        # Process per-dimension to honor weights; vectorize over K with chunking
        for j in range(d):
            xj = Xn[:, j].astype(self._dtype)
            wj = self._w[j]
            if ck <= 0 or ck >= K:
                phases = (xj[:, None] * self._gammas[None, :]) + self._theta[:, j][None, :]
                acc += wj * np.cos(phases, dtype=self._dtype).mean(axis=1).astype(self._dtype)
            else:
                s = np.zeros(N, dtype=self._dtype)
                for start in range(0, K, ck):
                    end = min(K, start + ck)
                    g = self._gammas[start:end]
                    th = self._theta[start:end, j]
                    phases = (xj[:, None] * g[None, :]) + th[None, :]
                    s += np.cos(phases, dtype=self._dtype).sum(axis=1).astype(self._dtype)
                acc += wj * (s / K)
        return acc.astype(self._dtype, copy=False)

    def _encode_row(self, Xn_row: np.ndarray) -> np.ndarray:
        # Xn_row shape (1, d)
        return self._encode_matrix(Xn_row)

    def _compute_stats(self, values_sorted: np.ndarray) -> HoloIndexStats:
        if values_sorted is None or values_sorted.size == 0:
            return HoloIndexStats(n=0, d=int(self._d or 0), K=self.K,
                                   h_min=float("nan"), h_max=float("nan"),
                                   phase_var=None, gap_entropy=None, gap_entropy_norm=None, invariant=None)
        h_min = float(values_sorted[0])
        h_max = float(values_sorted[-1])
        # Phase variance on fringe values
        env, pvar = phase_retrieve(np.asarray(values_sorted, dtype=float))
        # Gap entropy (discrete proxy)
        gaps = np.diff(values_sorted)
        if gaps.size <= 1:
            H = 0.0
            Hn = 0.0
        else:
            # histogram-based entropy (bins via Freedman–Diaconis-ish cap)
            bins = int(max(4, min(64, np.sqrt(gaps.size))))
            hist, _ = np.histogram(gaps, bins=bins)
            p = hist.astype(float)
            p = p / (p.sum() + 1e-12)
            nz = p[p > 0]
            H = float(-np.sum(nz * np.log(nz + 1e-12)))  # natural log
            Hmax = float(np.log(max(2, bins)))
            Hn = float(H / (Hmax if Hmax > 0 else 1.0))
        # Lightweight invariant proxy: ln(n) + H/log(phi)
        n = int(values_sorted.size)
        inv = float(np.log(max(1, n)) + (H / (log_phi if log_phi != 0 else 1.0)))
        return HoloIndexStats(n=n, d=int(self._d or 0), K=self.K,
                              h_min=h_min, h_max=h_max,
                              phase_var=float(pvar), gap_entropy=H, gap_entropy_norm=Hn, invariant=inv)

    def _compute_gate_thresholds(self, h_sorted: np.ndarray, num_samples: int = 64,
                                 window: int = 64) -> Tuple[Optional[float], Optional[float]]:
        """Estimate auto-gating thresholds from baseline phase variance and entropy.

        Samples num_samples windows of size `window` uniformly across the sorted fringe
        values and returns quantile thresholds for phase variance and normalized gap entropy.
        """
        try:
            N = h_sorted.size
            if N <= 0:
                return None, None
            step = max(1, N // max(1, num_samples))
            pvars = []
            Hns = []
            for start in range(0, N - 1, step):
                end = min(N, start + window)
                if end - start < 8:
                    continue
                st = self._compute_stats(h_sorted[start:end])
                if st.phase_var is not None:
                    pvars.append(float(st.phase_var))
                if st.gap_entropy_norm is not None:
                    Hns.append(float(st.gap_entropy_norm))
            if len(pvars) == 0 and len(Hns) == 0:
                return None, None
            q = float(np.clip(self.gate_quantile, 0.5, 0.99))
            thr_pv = float(np.quantile(pvars, q)) if len(pvars) > 0 else None
            thr_Hn = float(np.quantile(Hns, q)) if len(Hns) > 0 else None
            return thr_pv, thr_Hn
        except Exception:
            return None, None
