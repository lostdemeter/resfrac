# AI Quick-Start (Resfrac)

This is a concise guide for assistants and contributors.

## Setup

- Python 3.12+
- Create/activate a venv and install deps:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# for tests
pip install pytest
```

## Prime sieve (sanity)

```bash
python qam_zeta_distilled_full.py --n 100000
```

## Holographic Sublinear Index

Turn O(N) search into O(log N + S) via zeta-fringe encoding.

Python usage:

```python
import numpy as np
from resfrac import HolographicSublinearIndex

rng = np.random.default_rng(42)
X = rng.normal(size=(50_000, 8))
idx = HolographicSublinearIndex(K=256, seed=42).fit(X)

q = rng.normal(size=(8,))
cands, meta = idx.query(q, S=20, gate=True)
print("loc=", meta["loc"], "gated=", meta["gated"]) 
```

CLI demo (run from project root):

```bash
python -m resfrac.tools.holo_index_demo --N 50000 --d 8 --Q 200 --S 20 --K 256 --gate
```

Notes:
- Increase `S` (window) or `K` (zeros) for higher recall.
- Gating is optional (`--gate`); to reject gated results in code, pass `gate_mode='reject'` to `query()`.

## Benchmarks

Run the combined holographic benchmarks (TSP, 3-SAT, primes, optional HoloIndex):

```bash
python bench_holo.py --trials 10 --tsp-n 15 --sat uf50-0218.cnf --primeN 100000 --holoindex
```

## Tests

Run unit tests:

```bash
pytest -q
```

What’s covered:
- Import and fit for `HolographicSublinearIndex`
- Basic `encode()`/`query()`
- Gating behavior (reject mode)
- Sanity recall with full-window search

## Repo tips

- Package root: `resfrac/` (inner package: `resfrac/resfrac/`).
- From repo root, use `python -m resfrac.tools.<module>`.
- The class `HolographicSublinearIndex` is re-exported at `resfrac.__init__`.
