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


def align_phase(object_signal, reference_signal):
    """Polarization/phase alignment between object and reference signals.

    Computes analytic signals via Hilbert transform, estimates the mean phase
    offset between object and reference, and returns the rotation angle along
    with the phase-aligned real object signal.

    Parameters
    ----------
    object_signal : array-like
        Real-valued 1D signal representing the noisy/object path.
    reference_signal : array-like
        Real-valued 1D signal representing the clean/reference path.

    Returns
    -------
    theta : float
        Estimated phase rotation (radians) to align object with reference.
    aligned : np.ndarray
        Real-valued phase-aligned version of the object signal.
    """
    x = np.asarray(object_signal, dtype=float).ravel()
    r = np.asarray(reference_signal, dtype=float).ravel()
    n = min(x.size, r.size)
    if n == 0:
        return 0.0, x
    x = x[:n]
    r = r[:n]
    # Analytic signals
    ax = hilbert(x)
    ar = hilbert(r)
    ph_x = np.angle(ax)
    ph_r = np.angle(ar)
    d = ph_x - ph_r
    # Estimate mean phase difference on unit circle
    theta = float(np.angle(np.mean(np.exp(1j * d))))
    aligned_complex = ax * np.exp(-1j * theta)
    aligned = np.real(aligned_complex)
    return theta, aligned


def holo_bound(log_dim, H_boundary, adj=None, rh_constraint: bool = False):
    """Invariant as entropy surface: dim + disorder/φ - log(perm) [+ optional RH penalty].

    Lower is better (sharper hologram). If adj is None, the permanent proxy is 1.
    Mixed log bases (log2 for dim/entropy; ln for perm) are acceptable for a
    heuristic invariant; only relative differences are used for acceptance.

    If ``rh_constraint`` is True, add a soft penalty when the boundary signal implied
    by ``adj`` appears highly irregular (proxy for off-critical behavior). This is
    intentionally heuristic and lightweight.
    """
    perm = holographic_permanent(adj) if adj is not None else 1.0
    # Base invariant
    inv = float(log_dim + H_boundary / log_phi - np.log2(perm + 1e-10))
    if rh_constraint and adj is not None:
        try:
            # Proxy boundary signal: row-sum differences
            rs = np.sum(np.asarray(adj, dtype=float), axis=1)
            gaps = np.abs(np.diff(rs))
            if gaps.size > 0:
                # Threshold scaled by log(1+log_dim) to be dimension-aware
                thr = float(np.log(1.0 + max(0.0, float(log_dim))))
                if np.var(gaps) > thr:
                    inv += 10.0  # heavy but constant penalty
        except Exception:
            pass
    return float(inv)


# ------------------------------
# RH zero fiducials + calibration
# ------------------------------

def get_zeta_fiducials(K: int = 50) -> np.ndarray:
    """Return the first K imaginary parts of nontrivial zeta zeros as float array."""
    return np.array([float(complex(zetazero(k)).imag) for k in range(1, K + 1)], dtype=float)


def zero_calibrate(gaps: np.ndarray, fiducials: np.ndarray, tol: float = 0.1,
                   weighting: str = "none"):
    """Calibrate against RH zero fringes.

    Parameters
    ----------
    gaps : np.ndarray
        Positive gap-like sequence (e.g., TSP edge lengths).
    fiducials : np.ndarray
        Array of zeta zero imaginaries (gamma_k) to synthesize a fringe reference.
    tol : float
        Maximum circular variance tolerated to consider the signal coherent.

    Parameters
    ----------
    weighting : {"none", "inv_gamma", "pair_corr"}
        Optional weighting of zero contributions. "inv_gamma" scales by 1/γ_k.
        "pair_corr" uses a simple GUE-style pair-correlation weight 1 - sin^2(πγ)/(πγ)^2.

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
    zs = np.asarray(fiducials, dtype=float).ravel()
    if weighting == "inv_gamma":
        w = 1.0 / (np.abs(zs) + 1e-12)
    elif weighting == "pair_corr":
        # 1 - sin^2(pi z)/(pi z)^2, safe for large z
        zpi = np.pi * zs
        denom = np.where(np.abs(zpi) < 1e-12, 1e-12, zpi)
        w = 1.0 - (np.sin(zpi) ** 2) / (denom ** 2)
        w = np.clip(w, 0.0, 1.0)
    else:
        w = np.ones_like(zs)
    # Normalize weights to avoid scale blow-up
    w = w / (np.linalg.norm(w) + 1e-12)
    zero_fringe = np.zeros_like(gaps, dtype=float)
    for z, wz in zip(zs, w):
        zero_fringe += float(wz) * np.sin(z * gaps / scale)
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


# ------------------------------
# Fresnel diffraction (FFT-based, 2D)
# ------------------------------

def fresnel_propagate2d(field: np.ndarray, z: float = 1.0, lambda_w: float = 1.0,
                        dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Propagate a 2D scalar field to distance z using Fresnel approximation.

    Parameters
    ----------
    field : np.ndarray
        Real-valued 2D aperture field u(x, y, 0).
    z : float
        Propagation distance.
    lambda_w : float
        Wavelength.
    dx, dy : float
        Sample spacing in x and y.

    Returns
    -------
    np.ndarray
        Propagated intensity |u(x, y, z)|^2.
    """
    f0 = np.asarray(field, dtype=float)
    if f0.ndim != 2 or f0.size == 0:
        return np.abs(f0)**2
    ny, nx = f0.shape
    k = 2 * np.pi / max(lambda_w, 1e-12)
    # Quadratic phase factors (paraxial approx)
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    X, Y = np.meshgrid(x, y)
    Q1 = np.exp(1j * (k / (2 * max(z, 1e-12))) * (X**2 + Y**2))
    U0 = f0 * Q1
    U0_f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U0)))
    # Transfer kernel in frequency domain (Fresnel)
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(-1j * np.pi * lambda_w * max(z, 1e-12) * (FX**2 + FY**2))
    Uz_f = U0_f * H
    uz = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Uz_f)))
    # Output quadratic phase factor
    Q2 = np.exp(1j * (k / (2 * max(z, 1e-12))) * (X**2 + Y**2))
    uz = uz * Q2
    return np.abs(uz) ** 2
