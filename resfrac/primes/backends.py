# resfrac/primes/backends.py
from abc import ABC, abstractmethod
from typing import List

class PrimeBackend(ABC):
    @abstractmethod
    def primes_up_to(self, N: int) -> List[int]:
        ...

    @abstractmethod
    def is_probable_prime(self, n: int) -> bool:
        ...
