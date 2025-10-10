#!/usr/bin/env python3
"""
Zeta-fiducial bootstrap utilities for holographic resolution enhancement.

Functions
- zeta_points(K): zeta zeros -> 2D fiducial points in [-0.5,0.5]^2
- zeta_fringe_cartographer(points, grid, carrier, sigma): synthesize |O+R|^2
- zeta_sfft(H, carrier, crop): crop +1 sideband and reconstruct intensity
- tune_walltime(K, grid, carriers, sigmas, crop): tiny grid search

CLI
  python -m resfrac.tools.zeta_fiducial --K 50 --grid 256 --carrier 0.15 --sigma 0.05 --visualize
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict

import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

from resfrac.holo_utils import get_zeta_fiducials


@dataclass
class ZetaParams:
    grid: int = 256
    carrier: float = 0.15  # cycles/pixel along x
    sigma: float = 0.05    # Gaussian width in normalized coords
    crop: int = 32         # sideband crop (pixels)


def zeta_points(K: int = 50) -> np.ndarray:
    """Return K points (K,2) from the first K RH zero imaginaries.
    x is spread over [-0.5,0.5], y is imag/max(imag) mapped to [-0.5,0.5]."""
    zs = get_zeta_fiducials(K).astype(float)
    if K <= 0:
        return np.zeros((0, 2), dtype=float)
    y = zs / float(np.max(zs))  # [0,1]
    y = y * 1.0 - 0.5          # [-0.5,0.5]
    x = np.linspace(-0.5, 0.5, K)
    return np.stack([x, y], axis=1)


def zeta_fringe_cartographer(points: np.ndarray, grid: int = 256,
                              carrier: float = 0.15, sigma: float = 0.05) -> np.ndarray:
    """Synthesize an off-axis hologram H = |O + R|^2 from fiducial points.
    Object wave O is a sum of Gaussians with phase ramps; reference R tilted along x."""
    points = np.asarray(points, dtype=float)
    N = int(grid)
    # Normalized spatial coords in [-0.5,0.5]
    u = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(u, u)
    # Object field
    if points.size == 0:
        O = np.zeros((N, N), dtype=complex)
    else:
        O = np.zeros((N, N), dtype=complex)
        two_s2 = 2.0 * max(sigma, 1e-6) ** 2
        for px, py in points:
            G = np.exp(-((X - px) ** 2 + (Y - py) ** 2) / two_s2)
            P = np.exp(1j * 2 * np.pi * (px * X + py * Y))
            O += G * P
    # Reference wave (pixel-frequency carrier along x)
    xpix = (np.arange(N, dtype=float) / N)
    R = np.exp(1j * 2 * np.pi * carrier * xpix)[None, :]
    R = np.repeat(R, N, axis=0)
    H = np.abs(O + R) ** 2
    return H.astype(np.float32)


def zeta_sfft(H: np.ndarray, carrier: float = 0.15, crop: int = 32) -> Tuple[np.ndarray, Dict[str, float]]:
    """Crop the +1 sideband around fx=+carrier and reconstruct by IFFT."""
    H = np.asarray(H, dtype=float)
    N = H.shape[1]
    Hf = fftshift(fft2(H))
    c = N // 2
    ox = int(round(carrier * N))
    half = int(crop // 2)
    x0 = max(0, c + ox - half)
    x1 = min(N, x0 + crop)
    y0 = c - half
    y1 = y0 + (x1 - x0)
    y0 = max(0, y0)
    y1 = min(N, y1)
    side = Hf[y0:y1, x0:x1]
    recon = np.abs(ifft2(fftshift(side))) ** 2
    stats = {
        "max_intensity": float(np.max(recon)) if recon.size else 0.0,
        "crop_w": int(x1 - x0),
        "crop_h": int(y1 - y0),
    }
    return recon.astype(np.float32), stats


def tune_walltime(K: int = 50, grid: int = 256,
                   carriers: Sequence[float] = (0.1, 0.15, 0.2),
                   sigmas: Sequence[float] = (0.03, 0.05, 0.07), crop: int = 32) -> Dict[str, float]:
    """Tiny grid search over (carrier, sigma), returns best by max_intensity/time."""
    pts = zeta_points(K)
    best = {"score": -1.0}
    for carr in carriers:
        for s in sigmas:
            t0 = time.time()
            H = zeta_fringe_cartographer(pts, grid=grid, carrier=carr, sigma=s)
            recon, st = zeta_sfft(H, carrier=carr, crop=crop)
            dt = max(time.time() - t0, 1e-6)
            score = st["max_intensity"] / dt
            if score > best.get("score", -1.0):
                best = {"carrier": carr, "sigma": s, "score": score, "time": dt,
                        "max_intensity": st["max_intensity"]}
    return best


def main():
    p = argparse.ArgumentParser(description="Zeta-fiducial hologram tools")
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--grid", type=int, default=256)
    p.add_argument("--carrier", type=float, default=0.15)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--crop", type=int, default=32)
    p.add_argument("--tune", action="store_true", help="Run tiny (carrier,sigma) tuner")
    p.add_argument("--visualize", action="store_true")
    args = p.parse_args()

    if args.tune:
        best = tune_walltime(args.K, args.grid, crop=args.crop)
        print(f"Best: carrier={best['carrier']:.3f}, sigma={best['sigma']:.3f}, maxI={best['max_intensity']:.2f}, time={best['time']:.4f}s")
        return

    # Synthesize and reconstruct
    pts = zeta_points(args.K)
    t0 = time.time()
    H = zeta_fringe_cartographer(pts, grid=args.grid, carrier=args.carrier, sigma=args.sigma)
    recon, st = zeta_sfft(H, carrier=args.carrier, crop=args.crop)
    dt = time.time() - t0

    print("== ZETA-FIDUCIAL BOOTSTRAP ==")
    print(f"K={args.K}, grid={args.grid}, carrier={args.carrier}, sigma={args.sigma}, crop={args.crop}")
    print(f"Time: {dt:.4f}s, Recon max: {st['max_intensity']:.2f}, Crop: {st['crop_w']}x{st['crop_h']}")

    if args.visualize:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        # Show points
        axes[0].scatter(pts[:,0], pts[:,1], s=12, c="tab:blue")
        axes[0].set_title("Zeta Fiducials")
        axes[0].set_xlim(-0.55, 0.55)
        axes[0].set_ylim(-0.55, 0.55)
        axes[0].set_aspect("equal")
        # Hologram
        axes[1].imshow(H, cmap="gray")
        axes[1].set_title("Hologram |O+R|^2")
        axes[1].axis("off")
        # Reconstruction
        axes[2].imshow(recon, cmap="inferno")
        axes[2].set_title("Reconstruction (SFFT crop)")
        axes[2].axis("off")
        plt.tight_layout()
        out = Path("zeta_fiducial_viz.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {out}")


if __name__ == "__main__":
    main()
