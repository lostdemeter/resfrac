# resfrac/tools/holo_index_demo.py
# Minimal CLI to demonstrate the Holographic Sublinear Index

import argparse
import time
import numpy as np

from resfrac import HolographicSublinearIndex


def nn_bruteforce(X: np.ndarray, Q: np.ndarray, metric: str = "euclidean"):
    """Return indices of exact nearest neighbors for each q in Q.
    Uses chunking to limit memory usage.
    """
    N, d = X.shape
    M = Q.shape[0]
    idx = np.empty(M, dtype=int)
    bs = max(1, min(N, 20000))
    for i, q in enumerate(Q):
        best = (np.inf, -1)
        for s in range(0, N, bs):
            e = min(N, s + bs)
            if metric == "euclidean":
                d2 = np.sum((X[s:e] - q) ** 2, axis=1)
            elif metric == "cosine":
                x = X[s:e]
                num = np.sum(x * q, axis=1)
                den = (np.linalg.norm(x, axis=1) * (np.linalg.norm(q) + 1e-12) + 1e-12)
                sim = num / den
                d2 = 1.0 - sim
            else:
                raise ValueError("metric must be 'euclidean' or 'cosine'")
            j = int(np.argmin(d2))
            v = float(d2[j])
            if v < best[0]:
                best = (v, s + j)
        idx[i] = best[1]
    return idx


def evaluate_candidates(cands: np.ndarray, q: np.ndarray, metric: str = "euclidean"):
    if cands.size == 0:
        return -1
    if metric == "euclidean":
        d2 = np.sum((cands - q) ** 2, axis=1)
    elif metric == "cosine":
        num = np.sum(cands * q, axis=1)
        den = (np.linalg.norm(cands, axis=1) * (np.linalg.norm(q) + 1e-12) + 1e-12)
        d2 = 1.0 - (num / den)
    else:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    return int(np.argmin(d2))


def main():
    ap = argparse.ArgumentParser(description="Holographic Sublinear Index demo")
    ap.add_argument("--N", type=int, default=50000, help="Dataset size")
    ap.add_argument("--d", type=int, default=8, help="Dimensionality")
    ap.add_argument("--Q", type=int, default=200, help="Number of queries")
    ap.add_argument("--S", type=int, default=20, help="Sample size around insertion point")
    ap.add_argument("--K", type=int, default=256, help="Number of zeta zeros (fringe components)")
    ap.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"], help="Distance metric")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Computation dtype for index (affects speed)")
    ap.add_argument("--chunk-k", type=int, default=64, help="K-chunk size for vectorized encoding (0 disables chunking)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--prop", type=str, default="none", choices=["none", "fresnel"], help="Coherence propagation backend inside gating")
    ap.add_argument("--gate", action="store_true", help="Enable coherence gating")
    ap.add_argument("--gate-mode", type=str, default="flag", choices=["flag", "reject"], help="Gating behavior: flag or reject results when coherence drops")
    ap.add_argument("--gate-tol", type=float, default=0.15, help="Manual gating tolerance when auto gating is disabled")
    ap.add_argument("--gate-quantile", type=float, default=0.85, help="Auto-gating quantile for thresholds (0.5-0.99)")
    ap.add_argument("--no-gate-auto", action="store_true", help="Disable auto-gating thresholds; use gate-tol relative to baseline instead")
    ap.add_argument("--probes", type=int, default=1, help="Number of probe offsets around insertion point (>=1)")
    ap.add_argument("--ensembles", type=int, default=1, help="Number of independent indices to ensemble (>=1)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.N, args.d)).astype(float)
    # Normalize to [0,1] for a more uniform scale (optional)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    span = X_max - X_min
    span[span == 0.0] = 1.0
    Xn = (X - X_min) / span

    Qs = rng.normal(size=(args.Q, args.d)).astype(float)
    Qs = (Qs - X_min) / span

    # Fit index or ensemble of indices
    t0 = time.time()
    indices = []
    for e in range(max(1, int(args.ensembles))):
        idx = HolographicSublinearIndex(
            K=args.K,
            seed=int(args.seed) + e,
            normalize="none",
            gate_auto=(not args.no_gate_auto),
            gate_quantile=float(args.gate_quantile),
            dtype=str(args.dtype),
            chunk_k=int(args.chunk_k),
            prop=str(args.prop),
        ).fit(Xn)
        indices.append(idx)
    t_fit = time.time() - t0

    # Exact neighbors
    t0 = time.time()
    nn_exact = nn_bruteforce(Xn, Qs, metric=args.metric)
    t_exact = time.time() - t0

    # Approximate via holographic index
    t0 = time.time()
    hits = 0
    gated_count = 0
    for i, q in enumerate(Qs):
        # Union candidates across ensembles and probe offsets
        cand_indices = set()
        any_gated = False
        for e, idx in enumerate(indices):
            P = max(1, int(args.probes))
            # probe offsets spaced by half-window
            offsets = [ (p - (P//2)) * max(1, args.S//2) for p in range(P) ]
            for off in offsets:
                _, meta = idx.query(q, S=args.S, evaluate=None, return_indices=True,
                                    gate=args.gate, gate_mode=args.gate_mode,
                                    gate_tol=float(args.gate_tol), probe_offset=int(off))
                if meta.get("gated", False):
                    any_gated = True
                cand_indices.update(int(ii) for ii in meta["idx"].tolist())
        if any_gated:
            gated_count += 1
        if len(cand_indices) == 0:
            continue
        cand_list = sorted(cand_indices)
        cands = Xn[cand_list]
        j_local = evaluate_candidates(cands, q, metric=args.metric)
        if j_local >= 0:
            j_global = cand_list[j_local]
            if j_global == nn_exact[i]:
                hits += 1
    t_holo = time.time() - t0

    recall = hits / max(1, args.Q)

    print("== Holographic Sublinear Index Demo ==")
    print(f"N={args.N}, d={args.d}, Q={args.Q}, S={args.S}, K={args.K}, metric={args.metric}")
    print(f"fit: {t_fit:.3f}s | exact NN (brute): {t_exact:.3f}s | holo approx: {t_holo:.3f}s")
    print(f"top-1 recall: {recall:.3f}")
    if args.gate:
        print(f"coherence gated queries: {gated_count}/{args.Q} ({(100.0*gated_count/max(1,args.Q)):.1f}%)")


if __name__ == "__main__":
    main()
