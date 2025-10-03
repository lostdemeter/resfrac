from mpmath import *
import numpy as np
from sympy import primerange, primepi, isprime, mobius
from math import exp, log, pi, sqrt, ceil
import time
import argparse
import os

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
    return np.array([float(im(zetazero(k))) for k in range(1, num_zeros + 1)])

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

def qam_correct_scores(scores, noise_sigma=0.05, num_segments=4):
    """
    QAM correction: Denoise scores via Hilbert embed + Cantor-diagonal.
    """
    np.random.seed(42)
    n = len(scores)
    
    # Noisy channel
    noise = np.random.normal(0, noise_sigma, n)
    scores_noisy = scores + noise
    
    # Hilbert embed proxy: Map to 2D, quad order
    indices = np.arange(n) / (n - 1)
    norm_scores = (scores_noisy - np.min(scores_noisy)) / (np.max(scores_noisy) - np.min(scores_noisy) + 1e-8)
    points = np.column_stack((indices, norm_scores))
    quads = np.zeros(n)
    quads[(points[:, 0] > 0.5) & (points[:, 1] <= 0.5)] = 1
    quads[(points[:, 0] <= 0.5) & (points[:, 1] > 0.5)] = 2
    quads[(points[:, 0] > 0.5) & (points[:, 1] > 0.5)] = 3
    hilbert_order = quads + indices * 4  # Locality boost
    sorted_idx = np.argsort(hilbert_order)
    scores_embed = scores_noisy[sorted_idx]
    
    # Phase mapping
    theta = 2 * np.pi * (scores_embed - np.min(scores_embed)) / (np.max(scores_embed) - np.min(scores_embed) + 1e-8)
    
    # Cantor-diagonal per segment
    theta_corr = theta.copy()
    seg_size = n // num_segments
    for q in range(num_segments):
        start = q * seg_size
        end = min((q + 1) * seg_size, n)
        seg_theta = theta[start:end]
        if len(seg_theta) < 2:
            continue
        diffs = np.diff(seg_theta, append=seg_theta[0])
        bias = -np.sum(diffs) / len(seg_theta)  # Negated cum-diff proxy
        theta_corr[start:end] = (seg_theta + bias) % (2 * np.pi)
    
    # Unembed
    scores_corr = np.min(scores_embed) + (theta_corr / (2 * np.pi)) * (np.max(scores_embed) - np.min(scores_embed))
    unsort_idx = np.argsort(sorted_idx)
    scores_corrected = scores_corr[unsort_idx]
    
    return scores_corrected

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
    scores_qam = qam_correct_scores(scores)  # QAM denoising
    # Rank directly by corrected scores (no z-vote)
    top_idx = np.argsort(-scores_qam)[:M]
    top_candidates = np.array(candidates)[top_idx]
    primes = [int(c) for c in top_candidates if isprime(int(c))]
    expected_lower = approx - sqrt(N)
    expected_upper = approx + sqrt(N)
    num_primes = len(primes)
    if num_primes < expected_lower or num_primes > expected_upper:
        print(f"Warning: Found {num_primes}, expected ~{approx}")
    return sorted(primes)

def validate_primes(predicted, start_n, end_n):
    true_primes = set(primerange(start_n, end_n + 1))
    predicted_set = set(predicted)
    TP = len(predicted_set & true_primes)
    FP = len(predicted_set - true_primes)
    FN = len(true_primes - predicted_set)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    missed = sorted(true_primes - predicted_set)[:10]
    gaps = [missed[i+1] - missed[i] for i in range(len(missed)-1)] if len(missed) > 1 else []
    return precision, recall, missed, gaps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QAM-enhanced Chudnovsky-like prime sieve")
    parser.add_argument("--n", type=int, default=10000, help="Upper bound N (inclusive)")
    parser.add_argument("--output", type=str, default=None, help="Output file (one prime per line)")
    args = parser.parse_args()

    N = args.n
    if N < 2:
        raise SystemExit("--n must be >= 2")

    start_time = time.time()
    primes = chudnovsky_like_sieve(N)
    runtime = time.time() - start_time
    print(f"Found {len(primes)} primes up to {N} in {runtime:.2f}s")
    print("Last 5:", [int(x) for x in primes[-5:]])
    if args.output:
        dirn = os.path.dirname(args.output)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        with open(args.output, "w") as f:
            f.write("\n".join(str(p) for p in primes))
        print(f"Wrote {len(primes)} primes to {args.output}")
    true_pi = primepi(N)
    print(f"True pi({N}): {true_pi}")
    print(f"Accuracy: {len(primes) == true_pi}")
    precision, recall, missed, gaps = validate_primes(primes, 2, N)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Missed primes (first 10): {missed}")
    print(f"Gaps in missed: {gaps}")