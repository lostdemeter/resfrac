# resfrac/primes/cli.py
# Usage: python -m resfrac.primes.cli --N 100000 --backend chudnovsky|srt [--max-num-override]

import argparse
import sys
import time

from resfrac.resfrac3 import PrimeGraph, ResonantSolver
from resfrac.primes.chudnovsky_backend import ChudnovskyBackend


def get_backend(name: str, max_num_override: int | None):
    name = (name or "chudnovsky").lower()
    if name in ("chudnovsky", "default"):
        return ChudnovskyBackend()
    if name in ("srt", "oracle", "srtoracle"):
        try:
            from resfrac.primes.srt_backend import SRTOracleBackend
        except Exception as e:
            print(f"Error: SRT backend requested but not available: {e}", file=sys.stderr)
            sys.exit(2)
        return SRTOracleBackend(max_num_override=max_num_override)
    print(f"Unknown backend: {name}", file=sys.stderr)
    sys.exit(2)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resfrac prime backend CLI")
    parser.add_argument("--N", type=int, required=True, help="Upper bound for prime generation (inclusive)")
    parser.add_argument(
        "--backend",
        type=str,
        default="chudnovsky",
        choices=["chudnovsky", "srt"],
        help="Prime backend to use",
    )
    parser.add_argument(
        "--max-num-override",
        type=int,
        default=None,
        help="Override safety clamp for SRT backend (expert use)",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="Show first K primes in output (0 to disable)",
    )
    args = parser.parse_args(argv)

    backend = get_backend(args.backend, args.max_num_override)
    g = PrimeGraph(N=args.N, backend=backend)
    solver = ResonantSolver(max_iters=0)

    t0 = time.time()
    primes, count = solver.solve(g)
    dt = time.time() - t0

    try:
        # Optional: expected via sympy if available in runtime of resfrac3
        from sympy import primepi
        expected = int(primepi(args.N))
    except Exception:
        expected = None

    print(f"Backend: {args.backend}")
    print(f"N: {args.N}")
    print(f"Primes found: {count}")
    if expected is not None:
        print(f"Expected (primepi): {expected}")
    print(f"Time: {dt:.3f}s")
    if args.show > 0:
        print("First primes:", primes[: args.show])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
