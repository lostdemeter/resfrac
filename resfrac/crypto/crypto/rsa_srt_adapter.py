# resfrac/crypto/rsa_srt_adapter.py
# Cautious adapter which delegates only primality checks to SRT at small sizes.

def srt_is_prime(n: int) -> bool:
    try:
        import primes_oracle as srt  # srt repo on PYTHONPATH
        # srt.miller_rabin is deterministic for n < 4.759e9; for larger, fallback to MR
        if n < 4_759_123_141:
            return bool(srt.miller_rabin(int(n)))
    except Exception:
        pass
    from .rsa import mr_is_probable_prime
    return mr_is_probable_prime(n)
