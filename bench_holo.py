# bench_holo.py
# Benchmarks comparing fixed vs holographic mode on TSP, 3-SAT, and primes.
# Usage (from repo root):
#   python bench_holo.py --trials 10 --tsp-n 15 --sat uf50-0218.cnf --primeN 100000

import argparse
import time
import statistics as stats
import numpy as np

from resfrac3 import ResonantSolver, SATGraph, load_dimacs
from resfrac.primes.chudnovsky_backend import ChudnovskyBackend
from resfrac.holo_utils import get_zeta_fiducials, zero_calibrate
from resfrac import HolographicSublinearIndex


class TSPGraph:
    type = 'tsp'
    def __init__(self, coords):
        self.coords = np.array(coords, dtype=float)


def run_tsp_trial(n: int, holo: bool, seed: int = None, rh_constraint: bool = False, calib_weighting: str = 'none'):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    coords = rng.random((n, 2))
    g = TSPGraph(coords)
    solver = ResonantSolver(max_iters=50, holo=holo, rh_constraint=rh_constraint, calib_weighting=calib_weighting)
    tour, length, _ = solver.solve(g)
    bound = solver.invariant(g, tour)
    iters = max(0, len(solver.lengths) - 1)
    circ_var = float('nan')
    if holo:
        try:
            gaps = solver._get_gaps(tour, g)
            fid = get_zeta_fiducials(50)
            _, cv = zero_calibrate(gaps, fid, tol=0.1, weighting=calib_weighting)
            circ_var = float(cv)
        except Exception:
            pass
    return float(length), float(bound), int(iters), float(circ_var)


def run_sat_instance(path: str, holo: bool):
    g = load_dimacs(path)
    solver = ResonantSolver(max_iters=100, holo=holo)
    assign, unsat = solver.solve(g)
    bound = solver.invariant(g, assign)
    iters = max(0, len(solver.lengths) - 1)
    return int(unsat), float(bound), int(iters)


def run_primes(N: int, holo: bool):
    backend = ChudnovskyBackend(holo=holo)
    t0 = time.time()
    primes = backend.primes_up_to(int(N))
    dt = time.time() - t0
    try:
        from sympy import primepi
        expected = int(primepi(int(N)))
    except Exception:
        expected = None
    return len(primes), expected, dt


def summarize(vals):
    if len(vals) == 0:
        return "n/a"
    m = stats.mean(vals)
    s = stats.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.4f} ± {s:.4f}"


# ------------------------------
# Holographic Sublinear Index bench
# ------------------------------

def _nn_bruteforce(X: np.ndarray, Q: np.ndarray, metric: str = "euclidean"):
    N = X.shape[0]
    idx = np.empty(Q.shape[0], dtype=int)
    bs = max(1, min(N, 20000))
    for i, q in enumerate(Q):
        best = (np.inf, -1)
        for s in range(0, N, bs):
            e = min(N, s + bs)
            if metric == "euclidean":
                d2 = np.sum((X[s:e] - q) ** 2, axis=1)
            else:
                x = X[s:e]
                num = np.sum(x * q, axis=1)
                den = (np.linalg.norm(x, axis=1) * (np.linalg.norm(q) + 1e-12) + 1e-12)
                d2 = 1.0 - (num / den)
            j = int(np.argmin(d2))
            v = float(d2[j])
            if v < best[0]:
                best = (v, s + j)
        idx[i] = best[1]
    return idx

def _eval_local(cands: np.ndarray, q: np.ndarray, metric: str = "euclidean"):
    if cands.size == 0:
        return -1
    if metric == "euclidean":
        d2 = np.sum((cands - q) ** 2, axis=1)
    else:
        num = np.sum(cands * q, axis=1)
        den = (np.linalg.norm(cands, axis=1) * (np.linalg.norm(q) + 1e-12) + 1e-12)
        d2 = 1.0 - (num / den)
    return int(np.argmin(d2))

def run_holoindex_bench(N: int, d: int, Q: int, S: int, K: int, seed: int = 42, metric: str = "euclidean", gate: bool = True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, d)).astype(float)
    # Normalize to [0,1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    span = X_max - X_min
    span[span == 0.0] = 1.0
    Xn = (X - X_min) / span
    Qs = rng.normal(size=(Q, d)).astype(float)
    Qs = (Qs - X_min) / span

    t0 = time.time()
    index = HolographicSublinearIndex(K=K, seed=seed, normalize="none").fit(Xn)
    t_fit = time.time() - t0

    t0 = time.time()
    nn_exact = _nn_bruteforce(Xn, Qs, metric=metric)
    t_exact = time.time() - t0

    t0 = time.time()
    hits = 0
    gated_count = 0
    for i, q in enumerate(Qs):
        cands, meta = index.query(q, S=S, return_indices=False, gate=gate)
        if meta.get("gated", False):
            gated_count += 1
        j_local = _eval_local(cands, q, metric=metric)
        if j_local >= 0:
            if int(meta["idx"][j_local]) == int(nn_exact[i]):
                hits += 1
    t_holo = time.time() - t0

    recall = hits / max(1, Q)
    return {
        "fit_s": t_fit,
        "exact_s": t_exact,
        "holo_s": t_holo,
        "recall": recall,
        "gated": gated_count,
    }


def main():
    p = argparse.ArgumentParser(description="Holographic benchmark: TSP, 3-SAT, primes, HoloIndex")
    p.add_argument("--trials", type=int, default=10, help="Trials for TSP")
    p.add_argument("--tsp-n", type=int, default=15, help="Number of TSP points (10-20 recommended)")
    p.add_argument("--sat", type=str, default="uf50-0218.cnf", help="DIMACS CNF file path")
    p.add_argument("--primeN", type=int, default=100_000, help="Upper bound for prime sieve")
    p.add_argument("--rh-constraint", action="store_true", help="Enable RH-style penalty in holo_bound (TSP only)")
    p.add_argument("--calib-weighting", type=str, default="none", choices=["none", "inv_gamma", "pair_corr"], help="Weighting for zero_calibrate fringe synthesis")
    # HoloIndex options
    p.add_argument("--holoindex", action="store_true", help="Run Holographic Sublinear Index bench")
    p.add_argument("--hi-N", type=int, default=50_000, help="HoloIndex dataset size")
    p.add_argument("--hi-d", type=int, default=8, help="HoloIndex dimensionality")
    p.add_argument("--hi-Q", type=int, default=200, help="HoloIndex number of queries")
    p.add_argument("--hi-S", type=int, default=20, help="HoloIndex window size")
    p.add_argument("--hi-K", type=int, default=256, help="HoloIndex zeros (K)")
    p.add_argument("--hi-metric", type=str, default="euclidean", choices=["euclidean", "cosine"], help="HoloIndex metric")
    args = p.parse_args()

    print("== TSP ==")
    print(f"options: rh_constraint={args.rh_constraint}, calib_weighting={args.calib_weighting}")
    tsp_fixed_len, tsp_fixed_bound, tsp_fixed_iters = [], [], []
    tsp_holo_len, tsp_holo_bound, tsp_holo_iters, tsp_holo_circ = [], [], [], []
    for t in range(args.trials):
        L, B, I, _ = run_tsp_trial(args.tsp_n, holo=False, seed=1337 + t, rh_constraint=args.rh_constraint, calib_weighting=args.calib_weighting)
        tsp_fixed_len.append(L); tsp_fixed_bound.append(B); tsp_fixed_iters.append(I)
        Lh, Bh, Ih, CV = run_tsp_trial(args.tsp_n, holo=True, seed=9001 + t, rh_constraint=args.rh_constraint, calib_weighting=args.calib_weighting)
        tsp_holo_len.append(Lh); tsp_holo_bound.append(Bh); tsp_holo_iters.append(Ih); tsp_holo_circ.append(CV)
    print("fixed: length", summarize(tsp_fixed_len), "holo_bound", summarize(tsp_fixed_bound), "iters", summarize(tsp_fixed_iters))
    print(" holo: length", summarize(tsp_holo_len), "holo_bound", summarize(tsp_holo_bound), "iters", summarize(tsp_holo_iters))
    if len(tsp_holo_circ) > 0:
        print("       circ_var", summarize([v for v in tsp_holo_circ if np.isfinite(v)]))

    print("\n== 3-SAT ==")
    unsat_f, bound_f, it_f = run_sat_instance(args.sat, holo=False)
    unsat_h, bound_h, it_h = run_sat_instance(args.sat, holo=True)
    print(f"fixed: unsat {unsat_f}, holo_bound {bound_f:.3f}, iters {it_f}")
    print(f" holo: unsat {unsat_h}, holo_bound {bound_h:.3f}, iters {it_h}")

    print("\n== Primes ==")
    cnt_f, exp_pi, dt_f = run_primes(args.primeN, holo=False)
    cnt_h, _, dt_h = run_primes(args.primeN, holo=True)
    pi_text = f" (expected pi(N)={exp_pi})" if exp_pi is not None else ""
    print(f"fixed: count {cnt_f}{pi_text}, time {dt_f:.3f}s")
    print(f" holo: count {cnt_h}{pi_text}, time {dt_h:.3f}s")

    if args.holoindex:
        print("\n== HoloIndex ==")
        res = run_holoindex_bench(N=args.hi_N, d=args.hi_d, Q=args.hi_Q, S=args.hi_S, K=args.hi_K,
                                   seed=42, metric=args.hi_metric, gate=True)
        print(f"fit {res['fit_s']:.3f}s | exact NN {res['exact_s']:.3f}s | holo {res['holo_s']:.3f}s | recall {res['recall']:.3f} | gated {res['gated']}/{args.hi_Q}")


if __name__ == "__main__":
    main()
