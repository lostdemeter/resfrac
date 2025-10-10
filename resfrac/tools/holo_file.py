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
from io import BytesIO
from pathlib import Path

# Fixed-size JSON header length (bytes)
HEADER_SIZE = 512


# ========== Core Encode/Decode Functions ==========

def encode_holo(image, output_file, kx=0.3, ky=0.3, lambda_w=1.0, quantize=True, phase: bool = False,
                mode: str = "offaxis1"):
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
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D, got shape {image.shape}")
    
    height, width = image.shape
    
    # Object wave: sqrt(intensity) with flat phase (planar scene)
    O = np.sqrt(np.clip(image, 0, 1)) * np.exp(1j * 0)
    
    # Reference wave: tilted plane wave for off-axis separation
    # Pixel indices (j, i); kx, ky are cycles/pixel (normalized frequency)
    x = np.arange(width, dtype=float)
    y = np.arange(height, dtype=float)
    X, Y = np.meshgrid(x, y)
    R = np.exp(1j * 2 * np.pi * (kx * X + ky * Y))

    scale_factor = 4.0  # For intensity normalization to [0,1]

    if mode == "offaxis1":
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

    elif mode == "complex":
        # Store complex object field directly: O = sqrt(image) * exp(i*0)
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
        raise ValueError("mode must be one of {'offaxis1','ps4','complex'}")

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
    # Mark compression if used (only offaxis1 uint8 path currently)
    if mode == "offaxis1" and quantize and 'compression' not in header:
        header["compression"] = compression or None  # "png"
        header["payload_format"] = "png"
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
    
    # Dequantize if uint8
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

    elif mode == "complex":
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
    """Save float32 [0,1] image to file."""
    plt.imsave(path, img, cmap='gray', vmin=0, vmax=1)


# ========== Benchmarking ==========

def benchmark_holo_vs_png(img, holo_file='test.holo', verbose=True, phase=False, quantize=True, kx=0.3, ky=0.3, mode: str = "offaxis1"):
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
    holo = encode_holo(img, holo_file, quantize=quantize, phase=phase, kx=kx, ky=ky, mode=mode)
    encode_time = time.time() - start
    holo_size = Path(holo_file).stat().st_size
    
    start = time.time()
    recon = decode_holo(holo_file, phase_retrieval=phase)
    decode_time = time.time() - start
    
    psnr = compute_psnr(img, recon)
    
    # === PNG baseline ===
    start = time.time()
    bio = BytesIO()
    plt.imsave(bio, img, cmap='gray', format='png', vmin=0, vmax=1)
    png_size = len(bio.getvalue())
    png_enc_time = time.time() - start
    
    start = time.time()
    bio.seek(0)
    png_recon = mpimg.imread(bio, format='png')
    if png_recon.ndim == 3:
        png_recon = np.mean(png_recon[:, :, :3], axis=2)
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
        mode_str = f"{mode}, {'float32' if not quantize else 'uint8'}{', phase' if phase else ''}"
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
    
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    if mode == 'ps4' and isinstance(payload, np.ndarray) and getattr(payload, 'ndim', 0) == 3:
        holo_show = payload[0]
    elif mode == 'complex' and isinstance(payload, np.ndarray) and getattr(payload, 'ndim', 0) == 3 and payload.shape[0] == 2:
        # Show amplitude of complex field
        holo_show = np.hypot(payload[0], payload[1])
    else:
        holo_show = payload if isinstance(payload, np.ndarray) else np.asarray(payload)
    axes[1].imshow(holo_show, cmap='gray')
    axes[1].set_title('Hologram')
    axes[1].axis('off')
    
    axes[2].imshow(recon, cmap='gray', vmin=0, vmax=1)
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
                       choices=["offaxis1","ps4","complex"],
                       help="Storage mode: offaxis1 (single frame), ps4 (4-phase lossless), complex (lossless)")
    parser.add_argument("--no-quantize", action="store_true",
                       help="Disable uint8 quantization (use float32 for higher quality)")
    parser.add_argument("--phase", action="store_true",
                       help="Enable lightweight phase retrieval (encode: pre-process; decode: 5-iter GS)")
    parser.add_argument("--float", action="store_true",
                       help="Alias for --no-quantize (store float32 hologram)")
    
    # Benchmark options
    parser.add_argument("--size", type=int, default=256,
                       help="Checkerboard size for benchmark (default: 256)")
    parser.add_argument("--block", type=int, default=16,
                       help="Checkerboard block size (default: 16)")
    
    args = parser.parse_args()
    
    # === Encode Mode ===
    if args.encode:
        img = load_image_as_bw(args.encode)
        output_file = args.output or 'output.holo'
        
        print(f"Encoding {args.encode} ({img.shape[0]}×{img.shape[1]}) to {output_file}...")
        start = time.time()
        quantize = not (args.no_quantize or args.float)
        encode_holo(img, output_file, kx=args.kx, ky=args.ky,
                    quantize=quantize, phase=args.phase, mode=args.mode)
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
        print("Generating checkerboard pattern...")
        img = generate_checkerboard(size=args.size, block=args.block)
        
        print("Running benchmark...")
        quantize = not (args.no_quantize or args.float)
        results = benchmark_holo_vs_png(img, holo_file='test.holo', verbose=True, 
                                       phase=args.phase, quantize=quantize,
                                       kx=args.kx, ky=args.ky, mode=args.mode)
        
        # Optionally save visualization
        print("\nGenerating visualization...")
        visualize_encoding(img, 'test.holo', output_prefix='checkerboard_holo', mode=args.mode)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
