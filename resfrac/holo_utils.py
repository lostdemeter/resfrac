import numpy as np
from scipy.linalg import det
from scipy.signal import hilbert
from scipy.optimize import minimize  # re-exported in case downstream wants it

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
