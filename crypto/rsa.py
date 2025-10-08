# resfrac/crypto/rsa.py
import secrets
from math import gcd, log2
from typing import Callable, Tuple, List
import numpy as np

# Optional holographic helpers
try:
    from resfrac.primes.chudnovsky_backend import ChudnovskyBackend
    from resfrac.holo_utils import holographic_permanent
except Exception:  # pragma: no cover - optional
    ChudnovskyBackend = None  # type: ignore
    holographic_permanent = None  # type: ignore

_SMALL_PRIMES = [3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]

def _trial_division(n: int) -> bool:
    if n % 2 == 0:
        return n == 2
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    return True

def _decompose(n: int):
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    return r, d

def mr_is_probable_prime(n: int, rounds: int = 40) -> bool:
    if n < 2:
        return False
    if n in (2,3):
        return True
    if n % 2 == 0:
        return False
    if not _trial_division(n):
        return False
    r, d = _decompose(n)
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def _rand_odd_with_bitlen(bits: int) -> int:
    n = secrets.randbits(bits)
    n |= (1 << (bits - 1))
    n |= 1
    return n

def lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b

def _gen_prime(bits: int, e: int, is_prime_fn: Callable[[int], bool]) -> int:
    while True:
        cand = _rand_odd_with_bitlen(bits)
        if not is_prime_fn(cand):
            continue
        if gcd(cand - 1, e) == 1:
            return cand

def generate_rsa_keypair(bits: int = 2048, e: int = 65537,
                         is_prime_fn: Callable[[int], bool] = mr_is_probable_prime):
    assert bits >= 1024 and bits % 2 == 0
    p_bits = bits // 2
    q_bits = bits - p_bits
    p = _gen_prime(p_bits, e, is_prime_fn)
    q = _gen_prime(q_bits, e, is_prime_fn)
    while p == q:
        q = _gen_prime(q_bits, e, is_prime_fn)
    n = p * q
    lam = lcm(p - 1, q - 1)
    if gcd(e, lam) != 1:
        return generate_rsa_keypair(bits, e, is_prime_fn)
    d = pow(e, -1, lam)
    dp = d % (p - 1)
    dq = d % (q - 1)
    qinv = pow(q, -1, p)
    return (n, e), (n, d, p, q, dp, dq, qinv)


# -----------------------------
# Holographic helpers (demo only)
# -----------------------------
def sieve_primes_via_zeta(N: int = 50000, holo: bool = True) -> List[int]:
    """Return primes up to N using the Chudnovsky sieve (optionally with holo pruning).

    Note: This is for demo/visualization; not suitable for cryptographic prime generation.
    """
    if ChudnovskyBackend is None:
        raise RuntimeError("ChudnovskyBackend unavailable in runtime")
    backend = ChudnovskyBackend(holo=holo)
    return backend.primes_up_to(int(N))


def select_low_entropy_pair(primes: List[int]) -> Tuple[int, int]:
    """Select a prime pair (p, q) by minimizing a simple holographic-bound proxy.

    Proxy: Given consecutive primes p<q with gap g, build a 2x2 adjacency with off-diagonal
    weight g. The skew-det proxy yields |g|, so larger gaps lower the bound (sharper boundary).
    Returns the pair (p, q) that minimizes: -log2(|perm|+1e-10).
    """
    if len(primes) < 2:
        raise ValueError("Need at least two primes to select a pair")
    best_pair = (primes[0], primes[1])
    best_score = float("inf")
    for i in range(len(primes) - 1):
        p, q = int(primes[i]), int(primes[i + 1])
        g = q - p
        if g <= 0:
            continue
        if holographic_permanent is None:
            score = -log2(max(1e-10, float(g)))
        else:
            adj = np.array([[0.0, float(g)], [0.0, 0.0]], dtype=float)
            # skew will be [[0, g/2], [-g/2, 0]] leading to perm ~ |g|/sqrt(2); constant factors cancel in comparisons
            perm_proxy = float(holographic_permanent(adj))
            score = -log2(perm_proxy + 1e-10)
        if score < best_score:
            best_score = score
            best_pair = (p, q)
    return best_pair
