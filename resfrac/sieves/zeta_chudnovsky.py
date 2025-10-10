# resfrac/sieves/zeta_chudnovsky.py
# Extracted from resfrac3.py / qam_zeta_distilled_full.py so it's importable
import numpy as np
from mpmath import im, li, mpf
from sympy import primerange, isprime, mobius, primepi
from math import exp, log, sqrt, ceil
import mpmath as mp
from resfrac.holo_utils import phase_retrieve, align_phase
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

def _gs_1d(intensity_meas: np.ndarray, target_amp: np.ndarray, n_iter: int = 30, tol: float = 1e-4) -> np.ndarray:
    """Gerchberg–Saxton for 1D signals using FFT.

    Parameters
    ----------
    intensity_meas : np.ndarray
        Measured Fourier-domain magnitudes (|F{u}|).
    target_amp : np.ndarray
        Desired time-domain amplitude constraint |u|.
    n_iter : int
        Maximum iterations.
    tol : float
        MSE tolerance on Fourier magnitude convergence.

    Returns
    -------
    np.ndarray
        Complex refined field u.
    """
    intensity_meas = np.asarray(intensity_meas, dtype=float).ravel()
    target_amp = np.asarray(target_amp, dtype=float).ravel()
    n = min(intensity_meas.size, target_amp.size)
    if n == 0:
        return target_amp.astype(complex)
    I = intensity_meas[:n]
    A = target_amp[:n]
    # initialize with random phase on target amplitude
    rng = np.random.default_rng(0)
    u = A * np.exp(1j * rng.uniform(0, 2 * np.pi, size=n))
    for _ in range(max(1, int(n_iter))):
        U = np.fft.fft(u)
        # enforce Fourier magnitude
        Up = I * np.exp(1j * np.angle(U))
        up = np.fft.ifft(Up)
        # enforce time-domain amplitude
        u_new = A * np.exp(1j * np.angle(up))
        # convergence check
        if np.mean((np.abs(np.fft.fft(u_new)) - I) ** 2) < float(tol):
            u = u_new
            break
        u = u_new
    return u

def chudnovsky_like_sieve(N, T=50, K=50, epsilon=1.2, holo: bool = False, phase_retrieval: str = "hilbert"):
    gammas = get_gammas_dynamic(T)
    B = int(sqrt(N)) + 1
    mid = N / 2
    h = 0.05 / log(mid)
    approx = riemann_R(N, K)
    M = int(ceil(epsilon * approx))
    candidates = segmented_pre_sieve(2, N, B)
    scores = compute_spectral_scores(candidates, gammas, h)
    if holo:
        try:
            # Object/Reference split: object = spectral scores; reference = smooth baseline ~ 1/log n
            obj = np.asarray(scores, dtype=float)
            ref = 1.0 / (np.log(np.asarray(candidates, dtype=float) + 1e-12) + 1e-12)
            # Phase/polarization alignment
            _theta, obj_aligned = align_phase(obj, ref)
            if str(phase_retrieval).lower() == "gs":
                # Fourier magnitude from object; time-domain amp from reference
                intensity_meas = np.abs(np.fft.fft(obj_aligned))
                target_amp = ref / (np.max(ref) + 1e-12)
                u_refined = _gs_1d(intensity_meas, target_amp, n_iter=30, tol=1e-4)
                refined_real = np.real(u_refined)
                env, phase_var = phase_retrieve(refined_real)
                env_norm = env / (np.max(env) + 1e-12)
                ref_norm = target_amp  # already normalized
                scores = 0.6 * refined_real * env_norm + 0.4 * ref_norm
                if phase_var > 0.12:
                    scores *= 0.85
            else:
                # Hilbert envelope path (default)
                env, phase_var = phase_retrieve(obj_aligned)
                env_norm = env / (np.max(env) + 1e-12)
                # Blend aligned object with normalized reference as a clean baseline
                ref_norm = ref / (np.max(ref) + 1e-12)
                scores = 0.6 * obj_aligned * env_norm + 0.4 * ref_norm
                # Stability gate under high phase variance
                if phase_var > 0.12:
                    scores *= 0.85
        except Exception:
            pass
    top_idx = np.argsort(-scores)[:M]
    top_candidates = np.array(candidates)[top_idx]
    primes = [int(c) for c in top_candidates if isprime(int(c))]
    return sorted(primes)
