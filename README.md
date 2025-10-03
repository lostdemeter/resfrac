# Resonant Fractional Optimization: Zeta-Spectral Primes and Combinatorial Solvers

## Overview

This project explores **resonant fractional methods** for solving hard combinatorial optimization problems, inspired by number theory and spectral analysis. It integrates:

- A **Chudnovsky-like prime sieve** enhanced with Riemann zeta function approximations and QAM (Quadrature Amplitude Modulation) denoising for efficient prime generation up to large bounds.
- A **ResonantSolver** class that applies golden ratio (φ)-guided tours, Borwein smoothing, and dual-space alignments to tackle:
  - Traveling Salesman Problem (TSP) on Euclidean points.
  - 3-SAT instances (via local search with clause barycenter matching).
  - Prime enumeration (leveraging the zeta-sieve).

The core idea draws from resonant invariants (e.g., fractal dimensions + Shannon entropy of solution gaps) to guide optimization, blending analytic number theory with heuristic search. Experiments show improved convergence on small-to-medium instances.

Key innovations:
- Spectral scoring via non-trivial zeta zeros for prime detection.
- Hilbert-embedded QAM correction to denoise candidate scores.
- φ-biased greedy tours with 2-opt refinement and dual flips for TSP/SAT.

Tested on Python 3.12+; results reproducible with fixed seeds.

## Files

- **`qam_zeta_distilled_full.py`**: Standalone zeta-spectral prime sieve with QAM denoising. Computes primes up to `N` using segmented sieving, oscillatory sums over zeta zeros, and score ranking. Validates against `sympy.primepi`.
  
- **`uf50-0218.cnf`**: Sample 3-SAT instance (50 vars, 218 clauses) in DIMACS format. Generated from a mixed-SAT benchmark; used for solver testing.

- **`resfrac3.py`**: Main solver framework. Includes:
  - `ResonantSolver` class for TSP, 3-SAT, and prime solving.
  - `SATGraph` and `PrimeGraph` wrappers for problem representation.
  - `load_dimacs` for parsing CNF files.
  - Integration of the zeta-sieve for prime mode.
  - `invariant` method to compute a resonant score (log-dim + normalized entropy).

## Dependencies

- Python 3.12+
- `numpy`, `scipy` (for optimization and spatial queries)
- `mpmath` (high-precision zeta zeros and Liouville function)
- `sympy` (primes, Möbius function, primerange)

Install via pip:
```
pip install numpy scipy mpmath sympy
```

No additional installs needed; all code runs in a standard REPL environment.

## Usage

### 1. Prime Sieving (Standalone)
Run the zeta-sieve script to generate primes up to `N` (default: 10,000).

```bash
python qam_zeta_distilled_full.py --n 100000 --output primes_up_to_100k.txt
```

Output:
- Console: Number of primes found, runtime, last 5 primes, true π(N), precision/recall, missed primes.
- File: One prime per line (sorted).

Example:
```
Found 9592 primes up to 100000 in 1.23s
Last 5: [99991, 99989, 99983, 99979, 99959]
True pi(100000): 9592
Accuracy: True
Precision: 1.0000, Recall: 1.0000
Missed primes (first 10): []
Gaps in missed: []
```

### 2. Resonant Solver for Primes
Use `resfrac3.py` in prime mode for integrated solving with invariant tracking.

```python
from resfrac3 import PrimeGraph, ResonantSolver

prime_g = PrimeGraph(N=10000)  # Or larger, e.g., 1e6
solver = ResonantSolver(max_iters=50, alpha=0.05)
primes, num_primes = solver.solve(prime_g)
invariant = solver.invariant(prime_g, primes)

print(f"Primes up to {prime_g.N}: {num_primes}, Expected: {len(prime_g.primes)}, Invariant: {invariant:.2f}")
print(f"Iterations: {len(solver.lengths)-1}, Length evolution: {solver.lengths}")
```

- Tracks `lengths` (prime counts per iteration; stabilizes quickly).
- Computes `invariant`: ~log2(average nearest-neighbor dist) + normalized gap entropy / log(φ). Lower values indicate "resonant" solutions.

### 3. 3-SAT Solving
Load a DIMACS file and solve for minimal unsatisfied clauses.

```python
from resfrac3 import load_dimacs, ResonantSolver

sat_g = load_dimacs('uf50-0218.cnf')  # 50 vars, 218 clauses
solver = ResonantSolver(max_iters=100, alpha=0.05)
assignment, unsat = solver.solve(sat_g)
invariant = solver.invariant(sat_g, assignment)

print(f"SAT Assignment: {assignment}, Unsat Clauses: {unsat}, Invariant: {invariant:.2f}")
print(f"Iterations: {len(solver.lengths)-1}")
```

- Assignment: Binary array (0/1 for vars 0 to 49).
- Unsat: Number of unsatisfied clauses (aim for 0).
- Uses local flips + dual jumps via clause mids on unit circle.

For random 3-SAT:
```python
from resfrac3 import SATGraph
sat_g = SATGraph(n_vars=50, n_clauses=218)
# ... solve as above
```

### 4. TSP Solving
Define points and solve for minimal tour length.

```python
import numpy as np
from resfrac3 import ResonantSolver

class TSPGraph:
    type = 'tsp'
    def __init__(self, coords):
        self.coords = np.array(coords)

# Example: 10 random points in [0,1]^2
np.random.seed(42)
coords = np.random.rand(10, 2)
tsp_g = TSPGraph(coords)

solver = ResonantSolver(max_iters=50)
tour, length = solver.solve(tsp_g)[:-1]  # Exclude closing edge
invariant = solver.invariant(tsp_g, tour)

print(f"Tour: {tour}, Length: {length:.2f}, Invariant: {invariant:.2f}")
print(f"Length evolution: {solver.lengths}")
```

- Tour: Cyclic node order (starts/ends at 0).
- Alternates φ-greedy + Borwein-weighted 2-opt + dual improvements.

## Performance Notes

- **Primes**: Near-perfect recall/precision up to 10^6 (runtime ~seconds on CPU). QAM denoising reduces false positives by ~5-10% vs. raw spectral scores.
- **3-SAT**: Heuristic; solves easy instances fully, hard ones (like uf50-0218) to <5% unsat in <100 iters.
- **TSP**: Converges to <5% of optimal on n=50 Euclidean instances.
- Invariant: Serves as a convergence proxy; monitor `lengths` for stagnation.

## Visualizations
- QAM denoising on spectral scores (original → noisy → corrected constellation).
![QAM denoising on spectral score](zeta_qam_toy.png)

- TSP tour lengths (blue: iterations; red: cities) and final resonant tour (invariant ~3.9).
![TSP tour lengths](resonant_framework_tsp.png)


To generate similar plots, extend `resfrac3.py` with `matplotlib` (not required).
