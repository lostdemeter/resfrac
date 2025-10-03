# resfrac/primes/srt_backend.py
from typing import List
from .backends import PrimeBackend
import os
import importlib

class SRTOracleBackend(PrimeBackend):
    def __init__(self, max_num_override=None):
        # Do NOT import at init; importing primes_oracle executes heavy code.
        self._srt = None
        self._max_num_override = max_num_override

    def _load(self, N: int):
        if self._srt is not None:
            return
        # Configure environment so primes_oracle respects our limits and stays light on import
        os.environ['SRT_MAX_NUM'] = str(int(N))
        # Keep import light: disable flow/eval unless user explicitly wants them
        os.environ.setdefault('SRT_USE_FLOW', '0')
        os.environ.setdefault('SRT_EVAL', '0')
        os.environ.setdefault('SRT_SELECTION', 'heap')
        # Make chunks small enough for quick runs
        os.environ.setdefault('SRT_CHUNK_SIZE', '2000')
        os.environ.setdefault('SRT_SUBSAMPLE', '50')
        os.environ.setdefault('SRT_HDR_M', '1')
        os.environ.setdefault('SRT_EIG_K', '16')
        # Now import
        self._srt = importlib.import_module('primes_oracle')  # primes_oracle.py must be importable

    def primes_up_to(self, N: int) -> List[int]:
        # Warning: srt primes_oracle scans [2..N] with heavy memory unless heap/memmap
        # For safety, clamp to <=1e9 unless caller overrides.
        if self._max_num_override is None and N > 10**9:
            raise ValueError("SRT oracle backend is not configured for N > 1e9.")
        # Import with configured environment for this N
        self._load(int(N))
        # Best-effort: patch globals dynamically where applicable (in case module reads them post-import)
        try:
            self._srt.max_num = int(N)
            if hasattr(self._srt, 'selection_mode'):
                self._srt.selection_mode = 'heap'
        except Exception:
            pass
        # Execute main path (module code writes generated_primes.txt)
        if hasattr(self._srt, 'get_generated_primes'):
            return list(map(int, self._srt.get_generated_primes(N)))
        # Fallback: read file written by primes_oracle
        with open('generated_primes.txt', 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]

    def is_probable_prime(self, n: int) -> bool:
        # Prefer not to import SRT just for primality; use sympy fallback unless already loaded
        if self._srt is not None and hasattr(self._srt, 'miller_rabin'):
            return bool(self._srt.miller_rabin(int(n)))
        try:
            from sympy import isprime
            return bool(isprime(int(n)))
        except Exception:
            return False
