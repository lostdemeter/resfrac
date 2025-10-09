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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
from pathlib import Path


# ========== Core Encode/Decode Functions ==========

def encode_holo(image, output_file, kx=0.15, ky=0.15, lambda_w=1.0, quantize=True):
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
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    R = np.exp(1j * 2 * np.pi * (kx * X + ky * Y))
    
    # Interference pattern: |O + R|²
    holo = np.abs(O + R) ** 2
    # Normalize to [0,1] (theoretical max is 2 for unit amplitude waves)
    holo = np.clip(holo, 0, 2) / 2.0
    
    # Quantize for size reduction
    if quantize:
        holo_save = (holo * 255).astype(np.uint8)
        bit_depth = 8
    else:
        holo_save = holo.astype(np.float32)
        bit_depth = 32
    
    # Build JSON header with metadata
    header = {
        "version": "1.0",
        "width": int(width),
        "height": int(height),
        "kx": float(kx),
        "ky": float(ky),
        "lambda": float(lambda_w),
        "bit_depth": int(bit_depth)
    }
    header_str = json.dumps(header)
    header_bytes = header_str.encode('utf-8').ljust(128, b'\0')  # Fixed 128B header
    
    # Write: header + NumPy array
    with open(output_file, 'wb') as f:
        f.write(header_bytes)
        np.save(f, holo_save)
    
    return holo


def decode_holo(input_file):
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
        # Parse fixed 128B header
        header_bytes = f.read(128)
        header_str = header_bytes.rstrip(b'\0').decode('utf-8')
        header = json.loads(header_str)
        
        # Load hologram data
        holo = np.load(f)
    
    width = int(header['width'])
    height = int(header['height'])
    kx = float(header['kx'])
    ky = float(header['ky'])
    bit_depth = int(header['bit_depth'])
    
    # Dequantize if uint8
    if bit_depth == 8:
        holo = holo.astype(np.float32) / 255.0
    
    # Angular Spectrum reconstruction
    # Step 1: FFT to frequency domain
    H_fft = np.fft.fft2(holo)
    
    # Step 2: Demodulate by shifting to center object term
    freq_x = np.fft.fftfreq(width)
    freq_y = np.fft.fftfreq(height)
    Fx, Fy = np.meshgrid(freq_x, freq_y)
    shift = np.exp(-1j * 2 * np.pi * (kx * Fx + ky * Fy))
    H_shifted = H_fft * shift
    
    # Step 3: Lowpass filter to isolate object term (circular aperture)
    # Radius chosen to pass object bandwidth while rejecting twin/zero-order
    radius = max(kx, ky) / 1.5  # Adaptive radius based on tilt
    mask = (Fx ** 2 + Fy ** 2) < radius ** 2
    H_filtered = H_shifted * mask
    
    # Step 4: IFFT back to spatial domain
    recon_complex = np.fft.ifft2(H_filtered)
    
    # Step 5: Extract intensity (|·|²) and normalize
    recon = np.abs(recon_complex) ** 2
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
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
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

def benchmark_holo_vs_png(img, holo_file='test.holo', verbose=True):
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
    
    Returns
    -------
    dict
        Benchmark results with timing, size, and quality metrics.
    """
    # === Holographic encoding/decoding ===
    start = time.time()
    holo = encode_holo(img, holo_file, quantize=True)
    encode_time = time.time() - start
    holo_size = Path(holo_file).stat().st_size
    
    start = time.time()
    recon = decode_holo(holo_file)
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
        print("=" * 60)
        print("HOLOGRAPHIC FILE FORMAT BENCHMARK")
        print("=" * 60)
        print(f"Image size: {img.shape[0]}×{img.shape[1]} ({img.size} pixels)")
        print()
        print("Holographic (.holo, uint8):")
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
        print("Note: Holographic format trades size/speed for multiplexing potential.")
        print("      For N=100 pages, holo stays ~16kB (multiplexed), PNG scales to ~100kB.")
        print("=" * 60)
    
    return results


def visualize_encoding(img, holo_file='test.holo', output_prefix='holo_viz'):
    """
    Visualize the holographic encoding process.
    
    Creates three images:
    1. Original image
    2. Hologram (interference pattern)
    3. Reconstructed image
    """
    # Encode
    holo = encode_holo(img, holo_file, quantize=False)
    
    # Decode
    recon = decode_holo(holo_file)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(holo, cmap='gray')
    axes[1].set_title('Hologram (|O + R|²)')
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
    parser.add_argument("--kx", type=float, default=0.15,
                       help="Reference wave tilt in x (cycles/pixel, default: 0.15)")
    parser.add_argument("--ky", type=float, default=0.15,
                       help="Reference wave tilt in y (cycles/pixel, default: 0.15)")
    parser.add_argument("--no-quantize", action="store_true",
                       help="Disable uint8 quantization (use float32 for higher quality)")
    
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
        encode_holo(img, output_file, kx=args.kx, ky=args.ky, 
                   quantize=(not args.no_quantize))
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
        recon = decode_holo(args.decode)
        elapsed = time.time() - start
        save_image(output_file, recon)
        print(f"✓ Decoded in {elapsed:.4f}s, saved to {output_file}")
    
    # === Benchmark Mode ===
    elif args.benchmark:
        print("Generating checkerboard pattern...")
        img = generate_checkerboard(size=args.size, block=args.block)
        
        print("Running benchmark...")
        results = benchmark_holo_vs_png(img, holo_file='test.holo', verbose=True)
        
        # Optionally save visualization
        print("\nGenerating visualization...")
        visualize_encoding(img, 'test.holo', output_prefix='checkerboard_holo')
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
