# resfrac/crypto/rsa.py
import secrets
from math import gcd
from typing import Callable, Tuple

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
