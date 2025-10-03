# resfrac/sieves/zeta_chudnovsky.py
# Extracted from resfrac3.py / qam_zeta_distilled_full.py so it's importable
import numpy as np
from mpmath import im, li, mpf
from sympy import primerange, isprime, mobius, primepi
from math import exp, log, sqrt, ceil
import mpmath as mp
mp.dps = 20

def riemann_R(x, K=50):
    s = mpf(0)
    for n in range(1, K+1):
        mu = mobius(n)
        if mu == 0:
            continue
        s += mu / n * li(x ** (1/n))
    return float(s)

def get_gammas_dynamic(num_zeros):
    return np.array([float(im(mp.zetazero(k))) for k in range(1, num_zeros + 1)])

def segmented_pre_sieve(start_n, end_n, B):
    small_primes = list(primerange(2, B + 1))
    length = end_n - start_n + 1
    is_candidate = [True] * length
    for p in small_primes:
        start_multiple = max(p * p, ((start_n + p - 1) // p) * p)
        if start_multiple > end_n:
            continue
        idx = start_multiple - start_n
        for i in range(idx, length, p):
            is_candidate[i] = False
    return [start_n + i for i in range(length) if is_candidate[i]]

def compute_spectral_scores(candidates, gammas, h):
    log_ns = np.log(candidates)
    osc_plus = np.zeros(len(candidates), dtype=complex)
    osc_minus = np.zeros(len(candidates), dtype=complex)
    for gamma in gammas:
        taper = exp(-0.5 * h**2 * gamma**2)
        shift_plus = (0.5 + 1j * gamma) * h
        shift_minus = (0.5 + 1j * gamma) * (-h)
        phases_plus = np.exp(1j * gamma * log_ns + shift_plus)
        phases_minus = np.exp(1j * gamma * log_ns + shift_minus)
        osc_plus += taper * phases_plus / (0.5 + 1j * gamma)
        osc_minus += taper * phases_minus / (0.5 + 1j * gamma)
    psi_plus = np.array(candidates) * exp(h) - 2 * np.real(osc_plus)
    psi_minus = np.array(candidates) * exp(-h) - 2 * np.real(osc_minus)
    logn_arr = np.log(candidates)
    scores = (psi_plus - psi_minus) / (2 * h * np.array(candidates) * logn_arr)
    return scores

def chudnovsky_like_sieve(N, T=50, K=50, epsilon=1.2):
    gammas = get_gammas_dynamic(T)
    B = int(sqrt(N)) + 1
    mid = N / 2
    h = 0.05 / log(mid)
    approx = riemann_R(N, K)
    M = int(ceil(epsilon * approx))
    candidates = segmented_pre_sieve(2, N, B)
    scores = compute_spectral_scores(candidates, gammas, h)
    top_idx = np.argsort(-scores)[:M]
    top_candidates = np.array(candidates)[top_idx]
    primes = [int(c) for c in top_candidates if isprime(int(c))]
    return sorted(primes)
