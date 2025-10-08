import numpy as np
from scipy.linalg import det
from scipy.signal import hilbert
from scipy.optimize import minimize  # re-exported in case downstream wants it
from mpmath import zetazero

phi = (1 + np.sqrt(5)) / 2
log_phi = np.log(phi)

def holographic_permanent(adj):  # Valiant proxy: multiplicity via skew-det
    """Encode solution graph as boundary tensor; low val = unique reconstruction.

    For a given (weighted) adjacency matrix adj, we consider the skew-symmetric
    component and use the magnitude of its determinant as a proxy for boundary
    multiplicity. A lower value indicates a sharper, more unique reconstruction.
    """
    n = adj.shape[0]
    if n % 2 != 0:
        return 0.0
    skew = (adj - adj.T) / 2.0
    try:
        return float(np.abs(det(skew)) ** 0.5)
    except Exception:
        return 0.0


def phase_retrieve(signal_1d):  # Extend QAM: Hilbert envelope for fringe coherence
    """Retrieve amplitude/phase var; prune if var > 0.1 (incoherent).

    Parameters
    ----------
    signal_1d : array-like
        Real-valued 1D signal representing fringe magnitudes or scores.

    Returns
    -------
    envelope : np.ndarray
        Magnitude of the analytic signal (Hilbert envelope).
    phase_var : float
        Variance of the instantaneous phase; higher indicates incoherence.
    """
    signal_1d = np.asarray(signal_1d, dtype=float)
    if signal_1d.ndim != 1 or signal_1d.size == 0:
        return np.asarray(signal_1d, dtype=float), 0.0
    analytic = hilbert(signal_1d)
    env = np.abs(analytic)
    phase = np.angle(analytic)
    return env, float(np.var(phase))


def holo_bound(log_dim, H_boundary, adj=None):
    """Invariant as entropy surface: dim + disorder/φ - log(perm).

    Lower is better (sharper hologram). If adj is None, the permanent proxy is 1.
    Mixed log bases (log2 for dim/entropy; ln for perm) are acceptable for a
    heuristic invariant; only relative differences are used for acceptance.
    """
    perm = holographic_permanent(adj) if adj is not None else 1.0
    # Stabilize near-zero permanents to avoid -inf
    return float(log_dim + H_boundary / log_phi - np.log2(perm + 1e-10))


# ------------------------------
# RH zero fiducials + calibration
# ------------------------------

def get_zeta_fiducials(K: int = 50) -> np.ndarray:
    """Return the first K imaginary parts of nontrivial zeta zeros as float array."""
    return np.array([float(complex(zetazero(k)).imag) for k in range(1, K + 1)], dtype=float)


def zero_calibrate(gaps: np.ndarray, fiducials: np.ndarray, tol: float = 0.1):
    """Calibrate against RH zero fringes.

    Parameters
    ----------
    gaps : np.ndarray
        Positive gap-like sequence (e.g., TSP edge lengths).
    fiducials : np.ndarray
        Array of zeta zero imaginaries (gamma_k) to synthesize a fringe reference.
    tol : float
        Maximum circular variance tolerated to consider the signal coherent.

    Returns
    -------
    (alpha_shift, circ_var) : Tuple[Optional[float], float]
        alpha_shift is None if incoherent (circ_var > tol). Otherwise a small
        recommended additive shift to alpha in ~[-0.1, 0.1]. circ_var is the
        circular variance before shifting.
    """
    gaps = np.asarray(gaps, dtype=float).ravel()
    if gaps.size == 0:
        return None, 1.0
    # Construct a reference fringe using zeros scaled by the problem's typical gap scale
    t = float(np.mean(gaps))
    scale = np.log(max(t, 1e-12) + 1e-12)
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    zero_fringe = np.zeros_like(gaps, dtype=float)
    for z in np.asarray(fiducials, dtype=float):
        zero_fringe += np.sin(z * gaps / scale)
    # Analytic signals and phase difference
    analytic_gaps = hilbert(gaps)
    analytic_fringe = hilbert(zero_fringe)
    ph_g = np.angle(analytic_gaps)
    ph_f = np.angle(analytic_fringe)
    diff = ph_g - ph_f
    circ_var = float(1.0 - np.abs(np.mean(np.exp(1j * diff))))
    if not np.isfinite(circ_var):
        circ_var = 1.0
    if circ_var > tol:
        return None, float(circ_var)
    # Search a small additive phase shift to further tighten alignment
    grid = np.linspace(-0.1, 0.1, 41)
    best_s = 0.0
    best_val = np.inf
    for s in grid:
        d = (ph_g - s) - ph_f
        cv = 1.0 - np.abs(np.mean(np.exp(1j * d)))
        if cv < best_val:
            best_val = float(cv)
            best_s = float(s)
    return best_s, float(circ_var)
