# resfrac/primes/chudnovsky_backend.py
from typing import List
from sympy import isprime
from .backends import PrimeBackend
from ..sieves.zeta_chudnovsky import chudnovsky_like_sieve

class ChudnovskyBackend(PrimeBackend):
    def primes_up_to(self, N: int) -> List[int]:
        return chudnovsky_like_sieve(N)

    def is_probable_prime(self, n: int) -> bool:
        # sympy.isprime is fine for big-int checks in tooling; for RSA, we’ll inject our own MR below.
        return bool(isprime(int(n)))
