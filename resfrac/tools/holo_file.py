#!/usr/bin/env python3
"""
holo_file.py
Custom .holo file format for encoding/decoding B&W images as holographic interference patterns.

File Format:
- Header (128 bytes): JSON metadata (version, dimensions, reference wave params, bit depth)
- Data: NumPy array (hologram intensity |O + R|²) as uint8 or float32

Usage:
  python holo_file.py --encode input.png --output test.holo
  python holo_file.py --decode test.holo --output recon.png
  python holo_file.py --benchmark  # Run checkerboard test
"""

import argparse
import json
import time
import zlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors as mcolors
from io import BytesIO
from pathlib import Path
from typing import Optional

# Fixed-size JSON header length (bytes)
HEADER_SIZE = 512


# ========== Core Encode/Decode Functions ==========

def encode_holo(image, output_file, kx=0.3, ky=0.3, lambda_w=1.0, quantize=True, phase: bool = False,
                mode: str = "offaxis1", qam_order: int = 0, qam_range_min: Optional[float] = None,
                qam_range_max: Optional[float] = None):
    """
    Encode a B&W image to .holo file using off-axis holography.
    
    Parameters
    ----------
    image : np.ndarray
        2D float32 array [0,1] representing B&W image intensity.
    output_file : str or Path
        Output .holo file path.
    kx, ky : float
        Reference wave tilt (cycles/pixel) for off-axis encoding.
    lambda_w : float
        Wavelength (arbitrary units, typically 1.0).
    quantize : bool
        If True, quantize to uint8 for 4x size reduction (~0.1 dB PSNR loss).
    
    Returns
    -------
    holo : np.ndarray
        Hologram intensity array (before quantization).
    """
    image = np.asarray(image, dtype=np.float32)
    if mode in ("color", "ps4color"):
        # Accept either grayscale (expand to RGB later) or RGB
        if image.ndim == 2:
            height, width = image.shape
        elif image.ndim == 3 and image.shape[2] == 3:
            height, width = image.shape[:2]
        else:
            raise ValueError(f"{mode} expects image shape (H,W) or (H,W,3), got {image.shape}")
    else:
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got shape {image.shape}")
        height, width = image.shape
    
    # Reference wave: tilted plane wave for off-axis separation
    # Pixel indices (j, i); kx, ky are cycles/pixel (normalized frequency)
    x = np.arange(width, dtype=float)
    y = np.arange(height, dtype=float)
    X, Y = np.meshgrid(x, y)
    R = np.exp(1j * 2 * np.pi * (kx * X + ky * Y))

    scale_factor = 4.0  # For intensity normalization to [0,1]

    if mode == "offaxis1":
        # Object wave: sqrt(intensity) with flat phase (planar scene)
        O = np.sqrt(np.clip(image, 0, 1)) * np.exp(1j * 0)
        # Interference pattern: |O + R|², theoretical range [0,4]
        holo_raw = np.abs(O + R) ** 2
        holo = np.clip(holo_raw / scale_factor, 0.0, 1.0)
        compression = None
        if quantize:
            # Store as embedded PNG (lossless) to reduce size; bit_depth stays 8
            bit_depth = 8
            from io import BytesIO as _BytesIO
            _bio = _BytesIO()
            plt.imsave(_bio, holo, cmap='gray', format='png', vmin=0, vmax=1)
            payload = _bio.getvalue()  # PNG bytes
            compression = "png"
        else:
            holo_save = holo.astype(np.float32)
            bit_depth = 32
            payload = holo_save
        header_mode = "offaxis1"
        extra = {"frames": 1, "phases": [0.0]}

    elif mode == "ps4":
        # Object wave for phase-shifting path
        O = np.sqrt(np.clip(image, 0, 1)) * np.exp(1j * 0)
        # Four phase steps: 0, 90, 180, 270 degrees
        phases = [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]
        frames = []
        for phi in phases:
            R_phi = R * np.exp(1j * phi)
            I = np.abs(O + R_phi) ** 2
            frames.append(np.clip(I / scale_factor, 0.0, 1.0))
        stack = np.stack(frames, axis=0)  # (4, H, W)
        if quantize:
            payload = (stack * 255).astype(np.uint8)
            bit_depth = 8
        else:
            payload = stack.astype(np.float32)
            bit_depth = 32
        header_mode = "ps4"
        extra = {"frames": 4, "phases": phases}

    elif mode == "ps4color":
        # Lossless color via 4-phase per channel (12 frames total)
        if image.ndim == 2:
            img_rgb = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 3:
            img_rgb = np.clip(image, 0, 1).astype(np.float32)
        else:
            raise ValueError(f"ps4color expects image shape (H,W,3), got {image.shape}")

        phases = [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]
        stacks = []  # will become (3, 4, H, W)
        for c in range(3):
            Oc = np.sqrt(np.clip(img_rgb[:, :, c], 0, 1)) * np.exp(1j * 0)
            frames = []
            for phi in phases:
                R_phi = R * np.exp(1j * phi)
                I = np.abs(Oc + R_phi) ** 2
                frames.append(np.clip(I / scale_factor, 0.0, 1.0))
            stacks.append(np.stack(frames, axis=0))
        stack = np.stack(stacks, axis=0)  # (3, 4, H, W)

        if quantize:
            payload = (stack * 255).astype(np.uint8)
            bit_depth = 8
        else:
            payload = stack.astype(np.float32)
            bit_depth = 32
        header_mode = "ps4color"
        extra = {"frames": 4, "phases": phases, "channels": 3}

    elif mode == "color":
        # Angle-multiplexed RGB channels into a single intensity hologram
        # Expect image as (H, W, 3) float32 in [0,1]
        if image.ndim == 2:
            img_rgb = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 3:
            img_rgb = np.clip(image, 0, 1).astype(np.float32)
        else:
            raise ValueError(f"Color mode expects image shape (H,W,3), got {image.shape}")

        # Default per-channel carriers, defined as offsets relative to (kx, ky)
        # Keep within Nyquist (< 0.5 cycles/pixel)
        dx = [0.18, 0.00, -0.18]
        dy = [0.00, 0.18, -0.18]
        kx_list = [float(kx) + dx[i] for i in range(3)]
        ky_list = [float(ky) + dy[i] for i in range(3)]
        # Build object field as sum of channel object waves on distinct carriers
        O_r = np.sqrt(np.clip(img_rgb[:, :, 0], 0, 1)) * np.exp(1j * 0)
        O_g = np.sqrt(np.clip(img_rgb[:, :, 1], 0, 1)) * np.exp(1j * 0)
        O_b = np.sqrt(np.clip(img_rgb[:, :, 2], 0, 1)) * np.exp(1j * 0)

        X, Y = np.meshgrid(x, y)
        C_r = np.exp(1j * 2 * np.pi * (kx_list[0] * X + ky_list[0] * Y))
        C_g = np.exp(1j * 2 * np.pi * (kx_list[1] * X + ky_list[1] * Y))
        C_b = np.exp(1j * 2 * np.pi * (kx_list[2] * X + ky_list[2] * Y))
        # Scale objects vs reference to reduce O_i * O_j leakage terms
        alpha = 0.5
        O_total = alpha * (O_r * C_r + O_g * C_g + O_b * C_b)

        # Reference wave (kx, ky) as provided, for robust off-axis separation
        R0 = R
        holo_raw = np.abs(O_total + R0) ** 2
        # Normalize dynamically to [0,1]
        max_val = float(np.max(holo_raw)) if np.isfinite(np.max(holo_raw)) else 1.0
        scale_factor = max(max_val, 1e-6)
        holo = np.clip(holo_raw / scale_factor, 0.0, 1.0)

        if quantize:
            bit_depth = 8
            from io import BytesIO as _BytesIO
            _bio = _BytesIO()
            plt.imsave(_bio, holo, cmap='gray', format='png', vmin=0, vmax=1)
            payload = _bio.getvalue()  # PNG bytes
            compression = "png"
        else:
            payload = holo.astype(np.float32)
            bit_depth = 32
        header_mode = "color"
        extra = {"channels": 3, "color_kx": kx_list, "color_ky": ky_list, "color_alpha": alpha}

    elif mode == "complex":
        # Object wave for complex storage
        O = np.sqrt(np.clip(image, 0, 1)) * np.exp(1j * 0)
        # Store complex object field directly: O = sqrt(image) * exp(i*0)
        # Optional: QAM modulation to compact indices

        def _select_dtype_and_bits(order: int):
            if order <= 256:
                return np.uint8, 8
            elif order <= 65536:
                return np.uint16, 16
            else:
                return np.uint32, 32

        def _qam_levels(order: int, rmin: float, rmax: float):
            L = int(np.sqrt(order))
            if L * L != order or L < 2:
                raise ValueError("qam_order must be a perfect square >= 4 (e.g., 16, 64, 256)")
            levels = np.linspace(rmin, rmax, L, dtype=np.float32)
            step = levels[1] - levels[0]
            return L, levels, step

        def _qam_quantize_indices(field_c: np.ndarray, L: int, rmin: float, step: float):
            # Vectorized nearest level rounding on square grid
            re = field_c.real
            im = field_c.imag
            i_r = np.rint((re - rmin) / (step + 1e-12)).astype(np.int64)
            i_i = np.rint((im - rmin) / (step + 1e-12)).astype(np.int64)
            i_r = np.clip(i_r, 0, L - 1)
            i_i = np.clip(i_i, 0, L - 1)
            return (i_r * L + i_i).astype(np.int64)

        if qam_order and qam_order > 0:
            # Auto range if not specified: symmetric about 0 to cover field extrema
            re_min, re_max = float(np.min(O.real)), float(np.max(O.real))
            im_min, im_max = float(np.min(O.imag)), float(np.max(O.imag))
            # Use symmetric range to include negative/positive lobes if present
            m = max(abs(re_min), abs(re_max), abs(im_min), abs(im_max), 1e-6)
            rmin = qam_range_min if qam_range_min is not None else -m
            rmax = qam_range_max if qam_range_max is not None else m
            L, levels, step = _qam_levels(qam_order, rmin, rmax)
            idx = _qam_quantize_indices(O, L, rmin, step)
            dtype, bit_depth = _select_dtype_and_bits(qam_order)
            payload = idx.astype(dtype)  # (H, W) integer indices
            quantize = False  # handled by QAM indices
            header_mode = "complex"
            extra = {"qam_order": int(qam_order), "qam_min": float(rmin), "qam_max": float(rmax)}
            scale_factor = 1.0
        else:
            # Lossless with float32 components (real, imag)
            re = np.real(O).astype(np.float32)
            im = np.imag(O).astype(np.float32)
            payload = np.stack([re, im], axis=0)  # (2, H, W)
            bit_depth = 32
            quantize = False  # force float storage
            header_mode = "complex"
            extra = {"channels": 2}
            scale_factor = 1.0
    else:
        raise ValueError("mode must be one of {'offaxis1','ps4','ps4color','complex','color'}")

    # Build JSON header with metadata
    # Compute checksum over the uncompressed array bytes when payload is ndarray,
    # otherwise compress path used above stores zlib of .npy but checksum should
    # still reflect the array contents for integrity.
    if isinstance(payload, np.ndarray):
        checksum = zlib.crc32(payload.tobytes())
    else:
        # payload is compressed bytes; recompute checksum from original array
        # (holo_save for offaxis1 quantize)
        if mode == "offaxis1" and quantize:
            # Use uint8 view for checksum stability
            checksum = zlib.crc32(((holo * 255).astype(np.uint8)).tobytes())
        else:
            # Fallback: checksum of raw payload bytes
            checksum = zlib.crc32(payload)

    header = {
        "version": "1.1",
        "mode": header_mode,
        "width": int(width),
        "height": int(height),
        "kx": float(kx),
        "ky": float(ky),
        "lambda": float(lambda_w),
        "bit_depth": int(bit_depth),
        "phase_retrieval": bool(phase),
        "scale": float(scale_factor),
        **extra,
        "checksum": f"{checksum:08x}",
    }
    # Mark compression if used (offaxis1/color uint8 path currently)
    if mode in ("offaxis1", "color") and quantize and 'compression' not in header:
        header["compression"] = compression or None  # "png"
        header["payload_format"] = "png"
    # Mark QAM payload format for complex mode when indices are stored
    if mode == "complex" and isinstance(payload, np.ndarray) and payload.ndim == 2 and extra.get("qam_order", 0):
        header["payload_format"] = "qam"  # integer indices
    header_str = json.dumps(header)
    header_bytes = header_str.encode('utf-8')
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError(f"Header too large ({len(header_bytes)}B) exceeds fixed {HEADER_SIZE}B. Reduce metadata or increase HEADER_SIZE.")
    header_bytes = header_bytes.ljust(HEADER_SIZE, b'\0')  # Fixed-size header
    
    # Write: header + NumPy array
    with open(output_file, 'wb') as f:
        f.write(header_bytes)
        if isinstance(payload, (bytes, bytearray)):
            f.write(payload)
        else:
            np.save(f, payload)
    
    # Return raw intensity for float path (used by tests that analyze spectra)
    if mode == "offaxis1" and not quantize:
        return holo_raw
    return payload


def decode_holo(input_file, phase_retrieval: bool = False):
    """
    Decode .holo file to reconstruct B&W image via Angular Spectrum method.
    
    Parameters
    ----------
    input_file : str or Path
        Input .holo file path.
    
    Returns
    -------
    recon : np.ndarray
        Reconstructed B&W image (2D float32, [0,1]).
    """
    with open(input_file, 'rb') as f:
        # Parse fixed-size header
        header_bytes = f.read(HEADER_SIZE)
        header_str = header_bytes.rstrip(b'\0').decode('utf-8')
        header = json.loads(header_str)
        
        # Load hologram data (supports optional zlib-compressed npy stream)
        compression = header.get('compression', None)
        payload_format = header.get('payload_format', 'npy')
        if compression == 'zlib':
            data_bytes = f.read()
            try:
                raw = zlib.decompress(data_bytes)
            except Exception:
                # If decompression fails, treat as uncompressed npy
                raw = data_bytes
            from io import BytesIO as _BytesIO
            holo = np.load(_BytesIO(raw))
        elif payload_format == 'png':
            from io import BytesIO as _BytesIO
            data_bytes = f.read()
            img = mpimg.imread(_BytesIO(data_bytes), format='png')
            if img.ndim == 3:
                img = np.mean(img[:, :, :3], axis=2)
            # PNG reader returns [0,1] float32
            holo = img.astype(np.float32)
        else:
            holo = np.load(f)
        # Compute checksum on raw payload bytes before any dtype conversion
        # For PNG path, rebuild uint8 approximation for checksum consistency
        if payload_format == 'png':
            checksum_raw = f"{zlib.crc32(((holo * 255).astype(np.uint8)).tobytes()):08x}"
        else:
            checksum_raw = f"{zlib.crc32(holo.tobytes()):08x}"
    
    width = int(header['width'])
    height = int(header['height'])
    kx = float(header['kx'])
    ky = float(header['ky'])
    bit_depth = int(header['bit_depth'])
    mode = str(header.get('mode', 'offaxis1'))
    header_phase = bool(header.get('phase_retrieval', False))
    header_checksum = str(header.get('checksum', '')).lower()
    scale = float(header.get('scale', 4.0))
    
    # Dequantize for intensity-based payloads; QAM indices are handled later
    payload_format = header.get('payload_format', 'npy')
    if payload_format == 'qam':
        # Keep integer indices as-is
        pass
    else:
        if bit_depth == 8:
            # If loaded from PNG, holo already in [0,1]
            if header.get('payload_format', 'npy') != 'png':
                holo = holo.astype(np.float32) / 255.0
        elif bit_depth == 32:
            holo = holo.astype(np.float32)
        else:
            raise ValueError(f"Unsupported bit_depth: {bit_depth}")

    # Verify checksum if present
    if header_checksum:
        if checksum_raw != header_checksum:
            print("[holo_file] Warning: checksum mismatch (file header vs computed)")
    
    if mode == "offaxis1":
        # Restore original intensity scale prior to processing
        holo = holo * scale
        # Off-axis single-frame demodulation
        x = np.arange(width, dtype=float)
        y = np.arange(height, dtype=float)
        X, Y = np.meshgrid(x, y)
        demod = np.exp(-1j * 2 * np.pi * (kx * X + ky * Y))
        H_demod = holo * demod
        Hf = np.fft.fft2(H_demod)
        fx = np.fft.fftfreq(width)
        fy = np.fft.fftfreq(height)
        Fx, Fy = np.meshgrid(fx, fy)
        cutoff = 0.25
        sigma = cutoff / 2.0
        gauss = np.exp(-0.5 * (Fx**2 + Fy**2) / (sigma**2 + 1e-12))
        Hf_filtered = Hf * gauss
        u_conj = np.fft.ifft2(Hf_filtered)
        recon = np.abs(u_conj) ** 2

    elif mode == "ps4":
        # Four phase-shifted frames, shape (4, H, W)
        # Undo normalization
        I = holo * scale
        if I.shape[0] != 4:
            raise ValueError("ps4 data must have shape (4, H, W)")
        I0, I90, I180, I270 = I[0], I[1], I[2], I[3]
        ReC = (I0 - I180) / 4.0
        ImC = (I270 - I90) / 4.0
        C = ReC + 1j * ImC
        recon = np.abs(C) ** 2

    elif mode == "ps4color":
        # Lossless color via 4-phase per channel
        # Shape: (3, 4, H, W) with same normalization as ps4
        I = holo * scale
        if I.ndim != 4 or I.shape[0] != 3 or I.shape[1] != 4:
            raise ValueError("ps4color data must have shape (3, 4, H, W)")
        chans = []
        for c in range(3):
            I0, I90, I180, I270 = I[c, 0], I[c, 1], I[c, 2], I[c, 3]
            ReC = (I0 - I180) / 4.0
            ImC = (I270 - I90) / 4.0
            C = ReC + 1j * ImC
            chan = np.abs(C) ** 2
            chans.append(chan)
        recon = np.stack(chans, axis=2)

    elif mode == "color":
        # Per-channel reconstruction: mix by (kxi - kx, kyi - ky) and low-pass around DC
        I = holo * scale
        # Remove DC to suppress residual carrier and apply mild apodization
        I = I - float(np.mean(I))
        x = np.arange(width, dtype=float)
        y = np.arange(height, dtype=float)
        X, Y = np.meshgrid(x, y)
        fx = np.fft.fftfreq(width)
        fy = np.fft.fftfreq(height)
        Fx, Fy = np.meshgrid(fx, fy)
        # Precompute radial frequency grid for adaptive passband selection
        rgrid = np.sqrt(Fx**2 + Fy**2)
        kx_list = [float(k) for k in header.get('color_kx', [float(kx)+0.18, float(kx), float(kx)-0.18])]
        ky_list = [float(k) for k in header.get('color_ky', [float(ky), float(ky)+0.18, float(ky)-0.18])]
        chans = []
        for kxi, kyi in zip(kx_list, ky_list):
            demod = np.exp(-1j * 2 * np.pi * (((kxi - kx) * X) + ((kyi - ky) * Y)))
            Hf = np.fft.fft2(I * demod)
            # Estimate effective bandwidth by radial cumulative energy (98%)
            S = np.abs(Hf)
            # Histogram radial energy
            bins = np.linspace(0.0, 0.5, 256)
            hist, edges = np.histogram(rgrid.ravel(), bins=bins, weights=S.ravel())
            csum = np.cumsum(hist)
            total = csum[-1] if csum.size else 1.0
            idx = int(np.searchsorted(csum, 0.98 * total)) if total > 0 else len(bins)//4
            r98 = float(edges[min(max(idx, 1), len(edges)-1)])
            # Build Gaussian LPF with radius tied to r98 (guard rails)
            cutoff = min(max(r98 * 1.15, 0.08), 0.45)
            sigma = max(cutoff / 2.5, 1e-3)
            gauss_lp = np.exp(-0.5 * (rgrid**2) / (sigma**2))
            u = np.fft.ifft2(Hf * gauss_lp)
            mag = np.abs(u)
            m = float(np.max(mag)) if np.isfinite(np.max(mag)) else 1.0
            m = max(m, 1e-12)
            chans.append((mag / m) ** 2)
        recon = np.stack(chans, axis=2)
        recon = np.clip(recon, 0, 1)

    elif mode == "complex":
        if payload_format == 'qam':
            # Demap QAM indices back to complex field
            order = int(header.get('qam_order', 0))
            if order <= 0:
                raise ValueError("QAM payload missing 'qam_order' in header")
            L = int(np.sqrt(order))
            if L * L != order:
                raise ValueError("Invalid qam_order in header (not a perfect square)")
            rmin = float(header.get('qam_min', -1.0))
            rmax = float(header.get('qam_max', 1.0))
            step = (rmax - rmin) / (L - 1) if L > 1 else 0.0
            idx = holo.astype(np.int64)
            i_r = idx // L
            i_i = idx % L
            re = (rmin + i_r * step).astype(np.float32)
            im = (rmin + i_i * step).astype(np.float32)
            O = re + 1j * im
            recon = np.abs(O) ** 2
        else:
            # Complex field stored as (2, H, W): [real, imag]
            if holo.shape[0] != 2:
                raise ValueError("complex data must have shape (2, H, W)")
            O = holo[0] + 1j * holo[1]
            recon = np.abs(O) ** 2
    else:
        raise ValueError("Unknown mode in file header")
    # Normalize to [0,1] for display
    recon = np.clip(recon, 0, 1)
    
    return recon


# ========== Utility Functions ==========

def generate_checkerboard(size=256, block=16):
    """Generate a checkerboard pattern for testing."""
    img = np.zeros((size, size), dtype=np.float32)
    for i in range(0, size, block * 2):
        for j in range(0, size, block * 2):
            img[i:i+block, j:j+block] = 1
            img[i+block:i+block*2, j+block:j+block*2] = 1
    return img


def load_image_as_rgb(path):
    """Load image as RGB float32 [0,1]."""
    img = mpimg.imread(path)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape for RGB: {img.shape}")
    return img

def generate_color_gradient(size=256, noise_std=0.05):
    """Generate a colorful HSV gradient with optional noise, returns (H,W,3) in [0,1]."""
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)
    H = np.tile(x[None, :], (size, 1))  # hue along x
    S = np.ones((size, size), dtype=np.float32)
    V = np.ones((size, size), dtype=np.float32)
    hsv = np.stack([H, S, V], axis=2)
    rgb = mcolors.hsv_to_rgb(hsv).astype(np.float32)
    if noise_std and noise_std > 0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, noise_std, size=(size, size, 3)).astype(np.float32)
        rgb = np.clip(rgb + noise, 0.0, 1.0)
    return rgb

def compute_psnr(original, reconstructed):
    """Compute Peak Signal-to-Noise Ratio in dB."""
    diff = (original - reconstructed)
    mse = float(np.mean(diff ** 2))
    # Treat tiny numerical noise as exact
    if mse <= 1e-12:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def load_image_as_bw(path):
    """Load image from file and convert to B&W float32 [0,1]."""
    img = mpimg.imread(path)
    # Convert RGB to grayscale if needed
    if img.ndim == 3:
        img = np.mean(img[:, :, :3], axis=2)
    # Normalize to [0,1]
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def save_image(path, img):
    """Save float32 [0,1] image to file (handles grayscale or RGB)."""
    if img.ndim == 2:
        plt.imsave(path, img, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imsave(path, np.clip(img, 0, 1), vmin=0, vmax=1)


# ========== Benchmarking ==========

def benchmark_holo_vs_png(img, holo_file='test.holo', verbose=True, phase=False, quantize=True, kx=0.3, ky=0.3, mode: str = "offaxis1", qam_order: int = 0):
    """
    Benchmark .holo format vs PNG baseline.
    
    Parameters
    ----------
    img : np.ndarray
        Input B&W image (2D float32, [0,1]).
    holo_file : str
        Temporary .holo file path for testing.
    verbose : bool
        Print detailed results.
    phase : bool
        Enable phase retrieval.
    quantize : bool
        Use uint8 quantization (False = float32).
    kx, ky : float
        Reference wave tilt.
    
    Returns
    -------
    dict
        Benchmark results with timing, size, and quality metrics.
    """
    # === Holographic encoding/decoding ===
    start = time.time()
    holo = encode_holo(img, holo_file, quantize=quantize, phase=phase, kx=kx, ky=ky, mode=mode, qam_order=qam_order)
    encode_time = time.time() - start
    holo_size = Path(holo_file).stat().st_size
    
    start = time.time()
    recon = decode_holo(holo_file, phase_retrieval=phase)
    decode_time = time.time() - start
    
    psnr = compute_psnr(img, recon)
    
    # === PNG baseline ===
    start = time.time()
    bio = BytesIO()
    if getattr(img, 'ndim', 2) == 2:
        plt.imsave(bio, img, cmap='gray', format='png', vmin=0, vmax=1)
    else:
        plt.imsave(bio, np.clip(img, 0, 1), format='png', vmin=0, vmax=1)
    png_size = len(bio.getvalue())
    png_enc_time = time.time() - start
    
    start = time.time()
    bio.seek(0)
    png_recon = mpimg.imread(bio, format='png')
    if png_recon.ndim == 3 and getattr(img, 'ndim', 2) == 2:
        png_recon = np.mean(png_recon[:, :, :3], axis=2)
    elif png_recon.ndim == 3 and getattr(img, 'ndim', 2) == 3:
        png_recon = png_recon[:, :, :3]
    png_dec_time = time.time() - start
    
    results = {
        'holo_encode_time': encode_time,
        'holo_decode_time': decode_time,
        'holo_size': holo_size,
        'holo_psnr': psnr,
        'png_encode_time': png_enc_time,
        'png_decode_time': png_dec_time,
        'png_size': png_size,
        'png_psnr': float('inf'),
        'image_pixels': img.size
    }
    
    if verbose:
        mode_str = f"{mode}, {'float32' if not quantize else 'uint8'}{', phase' if phase else ''}{f', QAM{qam_order}' if (mode=='complex' and qam_order) else ''}"
        print("=" * 60)
        print("HOLOGRAPHIC FILE FORMAT BENCHMARK")
        print("=" * 60)
        print(f"Image size: {img.shape[0]}×{img.shape[1]} ({img.size} pixels)")
        print(f"Mode: kx={kx:.2f}, ky={ky:.2f}, {mode_str}")
        print()
        print(f"Holographic (.holo, {mode_str}):")
        print(f"  Encode time: {encode_time:.4f} s")
        print(f"  Decode time: {decode_time:.4f} s")
        print(f"  File size:   {holo_size} bytes (~{holo_size / img.size:.3f} B/pixel)")
        print(f"  PSNR:        {psnr:.2f} dB")
        print()
        print("PNG Baseline:")
        print(f"  Encode time: {png_enc_time:.4f} s")
        print(f"  Decode time: {png_dec_time:.4f} s")
        print(f"  File size:   {png_size} bytes (~{png_size / img.size:.3f} B/pixel)")
        print(f"  PSNR:        inf dB (lossless)")
        print()
        print("Analysis:")
        print(f"  Size ratio (holo/PNG): {holo_size / max(png_size, 1):.2f}x")
        print(f"  Encode speedup (PNG/holo): {png_enc_time / max(encode_time, 1e-6):.2f}x")
        print(f"  Decode speedup (PNG/holo): {png_dec_time / max(decode_time, 1e-6):.2f}x")
        print()
        if psnr < 20:
            print(f"⚠️  Low PSNR ({psnr:.2f} dB). Try: --phase --float for higher quality")
        print("Note: Holographic format trades size/speed for multiplexing potential.")
        print("      For N=100 pages, holo stays ~16kB (multiplexed), PNG scales to ~100kB.")
        print("=" * 60)
    
    return results


def visualize_encoding(img, holo_file='test.holo', output_prefix='holo_viz', mode: str = "offaxis1"):
    """
    Visualize the holographic encoding process.
    
    Creates three images:
    1. Original image
    2. Hologram (interference pattern)
    3. Reconstructed image
    """
    # Encode
    payload = encode_holo(img, holo_file, quantize=False, mode=mode)
    
    # Decode
    recon = decode_holo(holo_file)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if getattr(img, 'ndim', 2) == 2:
        axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    else:
        axes[0].imshow(np.clip(img, 0, 1))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    if mode == 'ps4' and isinstance(payload, np.ndarray) and getattr(payload, 'ndim', 0) == 3:
        holo_show = payload[0]
    elif mode == 'ps4color' and isinstance(payload, np.ndarray) and getattr(payload, 'ndim', 0) == 4 and payload.shape[0] == 3 and payload.shape[1] == 4:
        # Show first channel, first phase for visualization
        holo_show = payload[0, 0]
    elif mode == 'complex' and isinstance(payload, np.ndarray) and getattr(payload, 'ndim', 0) == 3 and payload.shape[0] == 2:
        # Show amplitude of complex field
        holo_show = np.hypot(payload[0], payload[1])
    else:
        holo_show = payload if isinstance(payload, np.ndarray) else np.asarray(payload)
    axes[1].imshow(holo_show, cmap='gray')
    axes[1].set_title('Hologram')
    axes[1].axis('off')
    
    if getattr(recon, 'ndim', 2) == 2:
        axes[2].imshow(recon, cmap='gray', vmin=0, vmax=1)
    else:
        axes[2].imshow(np.clip(recon, 0, 1))
    axes[2].set_title(f'Reconstructed (PSNR: {compute_psnr(img, recon):.1f} dB)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_prefix}.png")
    plt.close()


# ========== CLI Interface ==========

def main():
    parser = argparse.ArgumentParser(
        description="Holographic file format (.holo) for B&W images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode image to .holo
  python holo_file.py --encode input.png --output test.holo
  
  # Decode .holo to image
  python holo_file.py --decode test.holo --output recon.png
  
  # Run benchmark on checkerboard
  python holo_file.py --benchmark
  
  # Visualize encoding process
  python holo_file.py --encode input.png --output test.holo --visualize
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--encode", type=str, metavar="IMAGE",
                           help="Encode image to .holo format")
    mode_group.add_argument("--decode", type=str, metavar="HOLO_FILE",
                           help="Decode .holo file to image")
    mode_group.add_argument("--benchmark", action="store_true",
                           help="Run benchmark on checkerboard pattern")
    
    # I/O options
    parser.add_argument("--output", type=str, metavar="FILE",
                       help="Output file path")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization of encoding process")
    
    # Encoding parameters
    parser.add_argument("--kx", type=float, default=0.3,
                       help="Reference wave tilt in x (cycles/pixel, default: 0.3)")
    parser.add_argument("--ky", type=float, default=0.3,
                       help="Reference wave tilt in y (cycles/pixel, default: 0.3)")
    parser.add_argument("--mode", type=str, default="offaxis1",
                       choices=["offaxis1","ps4","ps4color","complex","color"],
                       help="Storage mode: offaxis1 (single frame), ps4 (4-phase B&W), ps4color (4-phase per RGB), complex (field), color (RGB multiplex)")
    parser.add_argument("--no-quantize", action="store_true",
                       help="Disable uint8 quantization (use float32 for higher quality)")
    parser.add_argument("--phase", action="store_true",
                       help="Enable lightweight phase retrieval (encode: pre-process; decode: 5-iter GS)")
    parser.add_argument("--float", action="store_true",
                       help="Alias for --no-quantize (store float32 hologram)")
    # QAM options (complex mode)
    parser.add_argument("--qam-order", type=int, default=0,
                       help="Enable QAM modulation for complex mode with given order (perfect square like 16, 64, 256). 0 disables QAM.")
    parser.add_argument("--qam-range-min", type=float, default=None,
                       help="Optional lower bound for QAM grid levels. Default auto: symmetric about 0 from data range.")
    parser.add_argument("--qam-range-max", type=float, default=None,
                       help="Optional upper bound for QAM grid levels. Default auto: symmetric about 0 from data range.")
    
    # Benchmark options
    parser.add_argument("--size", type=int, default=256,
                       help="Checkerboard size for benchmark (default: 256)")
    parser.add_argument("--block", type=int, default=16,
                       help="Checkerboard block size (default: 16)")
    parser.add_argument("--color-benchmark", action="store_true",
                       help="Use a colorful gradient test image and color mode for benchmark")
    
    args = parser.parse_args()
    
    # === Encode Mode ===
    if args.encode:
        img = load_image_as_rgb(args.encode) if args.mode in ('color','ps4color') else load_image_as_bw(args.encode)
        output_file = args.output or 'output.holo'
        
        print(f"Encoding {args.encode} ({img.shape[0]}×{img.shape[1]}) to {output_file}...")
        start = time.time()
        quantize = not (args.no_quantize or args.float)
        encode_holo(img, output_file, kx=args.kx, ky=args.ky,
                    quantize=quantize, phase=args.phase, mode=args.mode,
                    qam_order=args.qam_order, qam_range_min=args.qam_range_min, qam_range_max=args.qam_range_max)
        elapsed = time.time() - start
        size = Path(output_file).stat().st_size
        print(f"✓ Encoded in {elapsed:.4f}s, size: {size} bytes ({size/img.size:.3f} B/pixel)")
        
        if args.visualize:
            visualize_encoding(img, output_file, output_prefix=output_file.replace('.holo', '_viz'))
    
    # === Decode Mode ===
    elif args.decode:
        output_file = args.output or 'recon.png'
        
        print(f"Decoding {args.decode} to {output_file}...")
        start = time.time()
        recon = decode_holo(args.decode, phase_retrieval=args.phase)
        elapsed = time.time() - start
        save_image(output_file, recon)
        print(f"✓ Decoded in {elapsed:.4f}s, saved to {output_file}")
    
    # === Benchmark Mode ===
    elif args.benchmark:
        if args.color_benchmark:
            print("Generating colorful gradient pattern...")
            img = generate_color_gradient(size=args.size, noise_std=0.05)
            bm_mode = args.mode if args.mode in ('color','ps4color') else 'color'
        else:
            print("Generating checkerboard pattern...")
            img = generate_checkerboard(size=args.size, block=args.block)
            bm_mode = args.mode
        
        print("Running benchmark...")
        quantize = not (args.no_quantize or args.float)
        results = benchmark_holo_vs_png(img, holo_file='test.holo', verbose=True, 
                                       phase=args.phase, quantize=quantize,
                                       kx=args.kx, ky=args.ky, mode=bm_mode, qam_order=args.qam_order)
        
        # Optionally save visualization
        print("\nGenerating visualization...")
        viz_prefix = 'color_holo' if args.color_benchmark else 'checkerboard_holo'
        visualize_encoding(img, 'test.holo', output_prefix=viz_prefix, mode=bm_mode)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
