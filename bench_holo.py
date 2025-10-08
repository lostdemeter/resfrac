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


class TSPGraph:
    type = 'tsp'
    def __init__(self, coords):
        self.coords = np.array(coords, dtype=float)


def run_tsp_trial(n: int, holo: bool, seed: int = None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    coords = rng.random((n, 2))
    g = TSPGraph(coords)
    solver = ResonantSolver(max_iters=50, holo=holo)
    tour, length, _ = solver.solve(g)
    bound = solver.invariant(g, tour)
    iters = max(0, len(solver.lengths) - 1)
    return float(length), float(bound), int(iters)


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


def main():
    p = argparse.ArgumentParser(description="Holographic benchmark: TSP, 3-SAT, primes")
    p.add_argument("--trials", type=int, default=10, help="Trials for TSP")
    p.add_argument("--tsp-n", type=int, default=15, help="Number of TSP points (10-20 recommended)")
    p.add_argument("--sat", type=str, default="uf50-0218.cnf", help="DIMACS CNF file path")
    p.add_argument("--primeN", type=int, default=100_000, help="Upper bound for prime sieve")
    args = p.parse_args()

    print("== TSP ==")
    tsp_fixed_len, tsp_fixed_bound, tsp_fixed_iters = [], [], []
    tsp_holo_len, tsp_holo_bound, tsp_holo_iters = [], [], []
    for t in range(args.trials):
        L, B, I = run_tsp_trial(args.tsp_n, holo=False, seed=1337 + t)
        tsp_fixed_len.append(L); tsp_fixed_bound.append(B); tsp_fixed_iters.append(I)
        Lh, Bh, Ih = run_tsp_trial(args.tsp_n, holo=True, seed=9001 + t)
        tsp_holo_len.append(Lh); tsp_holo_bound.append(Bh); tsp_holo_iters.append(Ih)
    print("fixed: length", summarize(tsp_fixed_len), "holo_bound", summarize(tsp_fixed_bound), "iters", summarize(tsp_fixed_iters))
    print(" holo: length", summarize(tsp_holo_len), "holo_bound", summarize(tsp_holo_bound), "iters", summarize(tsp_holo_iters))

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


if __name__ == "__main__":
    main()
