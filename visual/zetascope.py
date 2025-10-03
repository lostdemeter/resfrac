# resfrac/visual/zetascope.py
# ZetaScope: Interactive and Cinematic visualizer for zeta-spectral prime scoring
# - Static mode: sliders + update button
# - Cinema mode: live animation with waterfall + hexbin + golden spiral overlay
#
# Deps: numpy, mpmath, sympy, scipy (for wav), matplotlib

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from mpmath import zetazero
from sympy import primerange, primepi
from scipy.io.wavfile import write as wav_write

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.animation import FuncAnimation

_phi = (1 + 5 ** 0.5) / 2.0

# ------------------------------
# Utilities: zeros, windows, spectra
# ------------------------------

_zeros_cache = {}

def riemann_zeros(K: int) -> np.ndarray:
    if K in _zeros_cache:
        return _zeros_cache[K]
    gammas = np.array([complex(zetazero(k)).imag for k in range(1, K + 1)], dtype=float)
    _zeros_cache[K] = gammas
    return gammas

def borwein_window(K: int, power: float = 1.0) -> np.ndarray:
    k = np.arange(1, K + 1, dtype=float)
    base = 1.0 - (k / (K + 1.0))
    base = np.clip(base, 0.0, 1.0)
    w = base ** max(1.0, power)
    w /= (np.linalg.norm(w) + 1e-12)
    return w

def compute_invariant(pred_primes: np.ndarray) -> float:
    if pred_primes is None or len(pred_primes) < 3:
        return float("nan")
    gaps = np.diff(pred_primes)
    if len(gaps) == 0:
        return float("nan")
    mean_gap = gaps.mean()
    vals, counts = np.unique(gaps, return_counts=True)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p + 1e-12))  # nats
    inv = math.log2(max(mean_gap, 1e-12)) + (H / abs(math.log(_phi)))
    return float(inv)

def evaluate_predictions(N: int, ranking_idx: np.ndarray, k_top: int) -> Tuple[float, float, np.ndarray]:
    true_primes = np.array(list(primerange(2, N + 1)), dtype=int)
    true_set = set(true_primes.tolist())
    n_vals = np.arange(2, N + 1, dtype=int)
    pred = n_vals[ranking_idx[:k_top]]
    pred_set = set(pred.tolist())
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(precision), float(recall), true_primes

# ------------------------------
# Sonification
# ------------------------------

def zeta_zero_choir(wav_path: str = "zeta_choir.wav",
                    K: int = 64,
                    seconds: float = 12.0,
                    sr: int = 44100,
                    gain: float = 0.9) -> str:
    gammas = riemann_zeros(K)
    f0 = 110.0
    freqs = f0 * (gammas / gammas[0]) ** 0.5
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = np.zeros_like(t)
    for i, f in enumerate(freqs):
        amp = 1.0 / (1.0 + 0.03 * i)
        detune = (i % 5) * 0.003 * f
        y += amp * np.sin(2 * np.pi * (f + detune) * t)
    y /= np.max(np.abs(y) + 1e-12)
    y *= gain
    wav_write(wav_path, sr, (y * 32767).astype(np.int16))
    return wav_path

# ------------------------------
# Static Scope (sliders + update)
# ------------------------------

def spectral_signature(n_vals: np.ndarray,
                       gammas: np.ndarray,
                       window_power: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = len(gammas)
    w = borwein_window(K, power=window_power)
    logn = np.log(n_vals)
    phases = np.outer(logn, gammas)
    I = (np.cos(phases) * w).sum(axis=1)
    Q = (np.sin(phases) * w).sum(axis=1)
    norm = np.linalg.norm(w) + 1e-12
    I /= norm
    Q /= norm
    mag = np.hypot(I, Q)
    return I, Q, mag

@dataclass
class ZetaScopeConfig:
    N: int = 100_000
    zeros: int = 256
    window_power: float = 1.5
    use_qam: bool = False  # static demo leaves denoise off by default
    seed: int = 42

class ZetaScopeApp:
    def __init__(self, cfg: ZetaScopeConfig):
        self.cfg = cfg
        self.n_vals = np.arange(2, cfg.N + 1, dtype=int)
        self.true_pi = primepi(cfg.N)
        self.gammas = riemann_zeros(cfg.zeros)

        matplotlib.rcParams["figure.figsize"] = (12, 8)
        self.fig = plt.figure(constrained_layout=False)
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.3])

        self.ax_mag = self.fig.add_subplot(gs[0, :])
        self.ax_iq  = self.fig.add_subplot(gs[1, 0])
        self.ax_iqc = self.fig.add_subplot(gs[1, 1])
        self.ax_metrics  = self.fig.add_subplot(gs[2, 1])
        self.ax_metrics.axis("off")
        self.fig.add_subplot(gs[2, 0]).axis("off")

        ax_zero = self.fig.add_axes([0.12, 0.08, 0.25, 0.03])
        self.s_zero = Slider(ax=ax_zero, label="zeros K", valmin=16, valmax=1024, valinit=cfg.zeros, valstep=16)
        ax_win = self.fig.add_axes([0.48, 0.08, 0.25, 0.03])
        self.s_win = Slider(ax=ax_win, label="window power", valmin=0.5, valmax=4.0, valinit=cfg.window_power, valstep=0.1)
        ax_btn = self.fig.add_axes([0.80, 0.05, 0.12, 0.05])
        self.btn = Button(ax_btn, "Update", color="#dddddd", hovercolor="#bbbbbb")

        self.I = self.Q = self.mag = None
        self.precision = self.recall = None
        self.invariant = None
        self.last_runtime = 0.0

        self.btn.on_clicked(lambda evt: self.on_update())
        self.recompute()
        self.redraw()

    def recompute(self):
        t0 = time.time()
        K = int(self.s_zero.val)
        self.gammas = riemann_zeros(K)
        window_power = float(self.s_win.val)
        self.I, self.Q, self.mag = spectral_signature(self.n_vals, self.gammas, window_power=window_power)

        idx = np.argsort(-self.mag)
        k_top = self.true_pi
        self.precision, self.recall, true_primes = evaluate_predictions(self.cfg.N, idx, k_top)
        pred_primes = self.n_vals[idx[:k_top]]
        pred_primes.sort()
        self.invariant = compute_invariant(pred_primes)

        self._idx = idx
        self._true_primes = true_primes
        self.last_runtime = time.time() - t0

    def redraw(self):
        self.ax_mag.clear(); self.ax_iq.clear(); self.ax_iqc.clear(); self.ax_metrics.clear()

        self.ax_mag.plot(self.n_vals, self.mag, color="#3366cc", lw=0.8, label="|S(n)|")
        self.ax_mag.vlines(self._true_primes, ymin=self.mag.min(), ymax=self.mag.max(), colors="#888888", alpha=0.06, linewidth=0.5)
        self.ax_mag.set_title(f"ZetaScope (static): N={self.cfg.N}, K={len(self.gammas)}, window={self.s_win.val:.2f}")
        self.ax_mag.set_xlabel("n"); self.ax_mag.set_ylabel("|S(n)|"); self.ax_mag.legend(loc="upper right", fontsize=9)

        s = 2.0 / (1.0 + np.sqrt(len(self.gammas)))
        self.ax_iq.hexbin(self.I, self.Q, gridsize=60, cmap="magma", mincnt=1)
        self.ax_iq.set_aspect("equal", adjustable="box")
        self.ax_iq.set_title("Constellation density (I/Q)")
        self.ax_iq.axhline(0, color="#aaaaaa", lw=0.5); self.ax_iq.axvline(0, color="#aaaaaa", lw=0.5)

        self.ax_iqc.scatter(self.I, self.Q, s=s, c=self.mag, cmap="viridis", alpha=0.5, edgecolor="none")
        self.ax_iqc.set_aspect("equal", adjustable="box")
        self.ax_iqc.set_title("Constellation colored by |S(n)|")
        self.ax_iqc.axhline(0, color="#aaaaaa", lw=0.5); self.ax_iqc.axvline(0, color="#aaaaaa", lw=0.5)

        txt = [
            f"pi(N) = {self.true_pi}",
            f"precision = {self.precision:.4f}",
            f"recall = {self.recall:.4f}",
            f"invariant ≈ {self.invariant:.3f}",
            f"compute time = {self.last_runtime:.2f}s"
        ]
        self.ax_metrics.text(0.02, 0.95, "\n".join(txt), va="top", ha="left", family="monospace", fontsize=10)
        self.ax_metrics.set_axis_off()
        self.fig.canvas.draw_idle()

    def on_update(self):
        self.recompute()
        self.redraw()

# ------------------------------
# Cinematic mode (animation)
# ------------------------------

class IncrementalSpectral:
    """
    Incrementally updates I/Q as K increases by adding contributions
    for gamma in (K_prev, K_new].
    """
    def __init__(self, n_vals: np.ndarray, Kmax: int, window_power: float):
        self.n_vals = n_vals
        self.logn = np.log(n_vals)
        self.Kmax = Kmax
        self.window_power = window_power
        self.gammas = riemann_zeros(Kmax)
        self.w_full = borwein_window(Kmax, power=window_power)
        self.I = np.zeros_like(self.logn)
        self.Q = np.zeros_like(self.logn)
        self.K_cur = 0
        self.norm = np.linalg.norm(self.w_full) + 1e-12

    def step(self, dK: int):
        K_new = min(self.K_cur + dK, self.Kmax)
        if K_new <= self.K_cur:
            return self.I, self.Q, self.magnitude()
        idx = np.arange(self.K_cur, K_new)  # 0-based indices
        gammas_batch = self.gammas[idx]      # shape (dK,)
        w_batch = self.w_full[idx]           # shape (dK,)
        phases = np.outer(self.logn, gammas_batch)  # (N, dK)
        self.I += (np.cos(phases) * w_batch).sum(axis=1)
        self.Q += (np.sin(phases) * w_batch).sum(axis=1)
        self.K_cur = K_new
        return self.I, self.Q, self.magnitude()

    def magnitude(self):
        return np.hypot(self.I, self.Q) / self.norm

def golden_spiral(ax, radius=1.0, turns=3, color="#ffaa00", lw=1.2, alpha=0.8):
    """
    Draw a golden spiral in current data coordinates.
    """
    theta = np.linspace(0, 2*np.pi*turns, 1200)
    # r grows by phi every quarter-turn (approx aesthetic)
    r = radius * (_phi ** (theta / (np.pi/2)))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, color=color, lw=lw, alpha=alpha)

class ZetaScopeCinematic:
    def __init__(self, N=50_000, Kmax=512, step=16, window_power=1.5, fps=20, waterfall_rows=120, rng_seed=42):
        self.N = N
        self.Kmax = Kmax
        self.stepK = step
        self.window_power = window_power
        self.fps = fps
        self.rows = waterfall_rows
        self.rng = np.random.default_rng(rng_seed)

        self.n_vals = np.arange(2, N + 1, dtype=int)
        self.true_pi = primepi(N)
        self.spec = IncrementalSpectral(self.n_vals, Kmax=Kmax, window_power=window_power)

        # Downsample factor for waterfall columns (≤ 1200 columns)
        self.ds = max(1, N // 1000)
        self.n_ds = self.n_vals[::self.ds]

        # Waterfall buffer (rolling)
        self.W = np.zeros((self.rows, len(self.n_ds)), dtype=float)
        self.w_row = 0

        matplotlib.rcParams["figure.figsize"] = (13, 8)
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.3])

        self.ax_mag = self.fig.add_subplot(gs[0, :])
        self.ax_iq  = self.fig.add_subplot(gs[1, 0])
        self.ax_wf  = self.fig.add_subplot(gs[1, 1])
        self.ax_metrics = self.fig.add_subplot(gs[2, :])
        self.ax_metrics.axis("off")

        # Static prime overlay lines for |S(n)|
        self.true_primes = np.array(list(primerange(2, N + 1)), dtype=int)

        # Initial artists
        self.mag_line, = self.ax_mag.plot([], [], color="#33aaff", lw=0.8, label="|S(n)|")
        self.ax_mag.vlines(self.true_primes, ymin=0, ymax=1, colors="#888888", alpha=0.06, linewidth=0.5)
        self.ax_mag.set_title(f"ZetaScope (cinema): N={N}, K→{Kmax}, step={step}, window={window_power}")
        self.ax_mag.set_xlim(2, N)
        self.ax_mag.set_ylim(0, 1)
        self.ax_mag.set_xlabel("n"); self.ax_mag.set_ylabel("|S(n)|")
        self.ax_mag.legend(loc="upper right", fontsize=9)

        # IQ hexbin
        self.hb = None
        self.iq_extent = None
        self._draw_iq(np.zeros(N-1), np.zeros(N-1))

        # Golden spiral overlay (draw later after we set extent)
        self._spiral_drawn = False

        # Waterfall heatmap
        self.im = self.ax_wf.imshow(self.W, aspect="auto", cmap="magma", origin="lower",
                                    extent=[self.n_ds[0], self.n_ds[-1], 0, self.rows])
        self.ax_wf.set_title("Spectral waterfall (rows ~ increasing K)")
        self.ax_wf.set_xlabel("n (downsampled)")
        self.ax_wf.set_ylabel("frame")

        # Metrics text
        self.txt_metrics = self.ax_metrics.text(0.01, 0.9, "", va="top", ha="left",
                                                family="monospace", fontsize=10)

        # State
        self._paused = False
        self._last_runtime = 0.0
        self._last_precision = 0.0
        self._last_recall = 0.0
        self._last_invariant = float("nan")
        self._last_idx = None

        # Key bindings
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Animation
        self.ani = FuncAnimation(self.fig, self._update_frame,
                                 init_func=self._init_frame,
                                 interval=int(1000 / fps), blit=False)

    def _init_frame(self):
        return []

    def _draw_iq(self, I, Q):
        # Draw hexbin; set extents for spiral scaling
        self.ax_iq.clear()
        hb = self.ax_iq.hexbin(I, Q, gridsize=60, cmap="viridis", mincnt=1)
        self.hb = hb
        self.ax_iq.set_aspect("equal", adjustable="box")
        self.ax_iq.set_title("Constellation density (I/Q) + golden spiral")
        self.ax_iq.axhline(0, color="#aaaaaa", lw=0.5); self.ax_iq.axvline(0, color="#aaaaaa", lw=0.5)
        xlim = self.ax_iq.get_xlim(); ylim = self.ax_iq.get_ylim()
        self.iq_extent = (xlim, ylim)
        # Draw golden spiral scaled to fit
        rad = 0.35 * max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]))
        golden_spiral(self.ax_iq, radius=max(1e-3, rad), turns=2.2, color="#ffaa00", lw=1.2, alpha=0.8)

    def _on_key(self, event):
        if event.key == " ":
            self._paused = not self._paused
        elif event.key == "right":
            self._step_once()
        elif event.key == "left":
            # small reverse step: re-init and fast-forward (simple approach)
            K_target = max(0, self.spec.K_cur - self.stepK)
            self._reinit_to_K(K_target)
        elif event.key.lower() == "s":
            fname = f"zetascope_frame_K{self.spec.K_cur}.png"
            self.fig.savefig(fname, dpi=150)
            print(f"[ZetaScope] Saved {fname}")
        elif event.key.lower() == "w":
            fname = f"zeta_choir_K{min(self.spec.K_cur,64)}.wav"
            zeta_zero_choir(fname, K=min(self.spec.K_cur, 64), seconds=8.0)
            print(f"[ZetaScope] Wrote {fname}")

    def _reinit_to_K(self, K_target):
        # rebuild spec up to K_target
        self.spec = IncrementalSpectral(self.n_vals, Kmax=self.Kmax, window_power=self.window_power)
        while self.spec.K_cur < K_target:
            self.spec.step(min(self.stepK, K_target - self.spec.K_cur))

    def _step_once(self):
        t0 = time.time()
        I, Q, mag = self.spec.step(self.stepK)
        self._last_runtime = time.time() - t0

        # Normalize y-limits adaptively
        y_max = max(1e-6, float(np.percentile(mag, 99.5)))
        self.ax_mag.set_ylim(0, y_max)
        self.mag_line.set_data(self.n_vals, mag)

        # IQ plot refresh every few frames to save time
        if (self.spec.K_cur // self.stepK) % 2 == 0:
            self._draw_iq(I, Q)

        # Waterfall: add a row (downsampled mag)
        mag_ds = mag[::self.ds]
        self.W[self.w_row % self.rows, :len(mag_ds)] = mag_ds
        self.im.set_data(self.W)
        self.im.set_clim(vmin=0, vmax=np.percentile(self.W, 99.0))
        self.w_row += 1

        # Predictions/metrics
        idx = np.argsort(-mag)
        k_top = self.true_pi
        precision, recall, _ = evaluate_predictions(self.N, idx, k_top)
        self._last_precision, self._last_recall = precision, recall
        pred_primes = self.n_vals[idx[:k_top]]
        pred_primes.sort()
        inv = compute_invariant(pred_primes)
        self._last_invariant = inv
        self._last_idx = idx

        txt = [
            f"K = {self.spec.K_cur}/{self.Kmax}  step={self.stepK}",
            f"precision = {precision:.4f}, recall = {recall:.4f}",
            f"invariant ≈ {inv:.3f}",
            f"frame compute = {self._last_runtime*1000:.1f} ms"
        ]
        self.txt_metrics.set_text("\n".join(txt))

    def _update_frame(self, _frame):
        if not self._paused and self.spec.K_cur < self.Kmax:
            self._step_once()
        return []

# ------------------------------
# Runner / CLI
# ------------------------------

def run_static(N: int, zeros: int, window_power: float, wav_out: Optional[str], wav_zeros: int, wav_seconds: float):
    if wav_out:
        print(f"[ZetaScope] Generating WAV: {wav_out} with K={wav_zeros}, seconds={wav_seconds} ...")
        path = zeta_zero_choir(wav_path=wav_out, K=wav_zeros, seconds=wav_seconds)
        print(f"[ZetaScope] Wrote {path}")
    cfg = ZetaScopeConfig(N=N, zeros=zeros, window_power=window_power)
    app = ZetaScopeApp(cfg)
    plt.show()

def run_cinema(N: int, Kmax: int, step: int, window_power: float, fps: int, rows: int):
    # Tip: for smoothness, keep N ≤ 80k, Kmax ≤ 1024, step ∈ {8,16,32}
    app = ZetaScopeCinematic(N=N, Kmax=Kmax, step=step, window_power=window_power, fps=fps, waterfall_rows=rows)
    plt.show()

def main():
    p = argparse.ArgumentParser(description="ZetaScope: zeta-spectral visualizer (static + cinema)")
    p.add_argument("--N", type=int, default=100000, help="Upper bound for n (default 100k).")
    p.add_argument("--zeros", type=int, default=256, help="K for static mode.")
    p.add_argument("--window", type=float, default=1.5, help="Window power (Borwein-like).")
    p.add_argument("--wav", type=str, default=None, help="Optional: output WAV filename (static mode).")
    p.add_argument("--wav-zeros", type=int, default=64, help="Zeros used for WAV (default 64).")
    p.add_argument("--wav-seconds", type=float, default=12.0, help="Length of WAV in seconds (default 12).")

    # Cinema options
    p.add_argument("--cinema", action="store_true", help="Run animated cinematic mode.")
    p.add_argument("--Kmax", type=int, default=512, help="Max zeros for cinema sweep.")
    p.add_argument("--step", type=int, default=16, help="Zeros added per frame in cinema.")
    p.add_argument("--fps", type=int, default=20, help="Target frames per second for cinema.")
    p.add_argument("--rows", type=int, default=120, help="Waterfall rows (rolling window).")

    args = p.parse_args()

    if args.cinema:
        run_cinema(N=args.N, Kmax=args.Kmax, step=args.step, window_power=args.window, fps=args.fps, rows=args.rows)
    else:
        run_static(N=args.N, zeros=args.zeros, window_power=args.window,
                   wav_out=args.wav, wav_zeros=args.wav_zeros, wav_seconds=args.wav_seconds)

if __name__ == "__main__":
    main()
