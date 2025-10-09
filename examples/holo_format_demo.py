#!/usr/bin/env python3
"""
holo_format_demo.py
Demonstration of .holo file format integration with resfrac holographic tools.

This script shows:
1. Encoding TSP tours as B&W images and saving as .holo
2. Encoding prime patterns as holograms
3. Using .holo format for holographic data storage
4. Multiplexing multiple images in a single hologram
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from resfrac.tools.holo_file import (
    encode_holo, decode_holo, compute_psnr, 
    generate_checkerboard, save_image
)
from resfrac.holo_utils import get_zeta_fiducials


# ========== Demo 1: Basic Encoding/Decoding ==========

def demo_basic():
    """Basic demonstration of .holo format."""
    print("=" * 70)
    print("DEMO 1: Basic Holographic Encoding/Decoding")
    print("=" * 70)
    
    # Create test pattern
    img = generate_checkerboard(size=256, block=16)
    
    # Encode to .holo
    print("\n1. Encoding checkerboard to .holo format...")
    holo_file = "demo_basic.holo"
    encode_holo(img, holo_file, kx=0.15, ky=0.15, quantize=True)
    size = Path(holo_file).stat().st_size
    print(f"   ✓ Saved to {holo_file} ({size} bytes, {size/img.size:.3f} B/pixel)")
    
    # Decode from .holo
    print("\n2. Decoding from .holo format...")
    recon = decode_holo(holo_file)
    psnr = compute_psnr(img, recon)
    print(f"   ✓ Reconstructed with PSNR: {psnr:.2f} dB")
    
    # Save visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show hologram (reload for visualization)
    with open(holo_file, 'rb') as f:
        import json
        f.read(128)  # Skip header
        holo = np.load(f).astype(np.float32) / 255.0
    axes[1].imshow(holo, cmap='gray')
    axes[1].set_title('Hologram (Interference Pattern)')
    axes[1].axis('off')
    
    axes[2].imshow(recon, cmap='gray')
    axes[2].set_title(f'Reconstructed (PSNR: {psnr:.1f} dB)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_basic_viz.png', dpi=150, bbox_inches='tight')
    print(f"\n3. Visualization saved to demo_basic_viz.png")
    plt.close()


# ========== Demo 2: Prime Pattern Encoding ==========

def demo_primes():
    """Encode prime number patterns as holograms."""
    print("\n" + "=" * 70)
    print("DEMO 2: Prime Pattern Holographic Encoding")
    print("=" * 70)
    
    # Generate prime pattern (Ulam spiral-like)
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    size = 128
    img = np.zeros((size, size), dtype=np.float32)
    
    # Simple grid pattern: white if (i*size + j) is prime
    print("\n1. Generating prime pattern...")
    for i in range(size):
        for j in range(size):
            n = i * size + j
            if is_prime(n):
                img[i, j] = 1.0
    
    prime_count = int(np.sum(img))
    print(f"   ✓ Pattern contains {prime_count} primes")
    
    # Encode to .holo
    print("\n2. Encoding to holographic format...")
    holo_file = "demo_primes.holo"
    encode_holo(img, holo_file, kx=0.2, ky=0.2, quantize=True)
    size_bytes = Path(holo_file).stat().st_size
    print(f"   ✓ Saved to {holo_file} ({size_bytes} bytes)")
    
    # Decode and verify
    print("\n3. Decoding and verifying...")
    recon = decode_holo(holo_file)
    psnr = compute_psnr(img, recon)
    print(f"   ✓ PSNR: {psnr:.2f} dB")
    
    # Threshold reconstruction to count recovered primes
    recon_binary = (recon > 0.5).astype(np.float32)
    recovered_count = int(np.sum(recon_binary))
    accuracy = np.mean(img == recon_binary)
    print(f"   ✓ Recovered {recovered_count}/{prime_count} primes ({accuracy*100:.1f}% accuracy)")
    
    # Save comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Original Prime Pattern ({prime_count} primes)')
    axes[0].axis('off')
    
    axes[1].imshow(recon, cmap='gray')
    axes[1].set_title(f'Reconstructed (PSNR: {psnr:.1f} dB)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_primes_viz.png', dpi=150, bbox_inches='tight')
    print(f"\n4. Visualization saved to demo_primes_viz.png")
    plt.close()


# ========== Demo 3: Zeta Zero Visualization ==========

def demo_zeta_zeros():
    """Encode zeta zero patterns as holograms."""
    print("\n" + "=" * 70)
    print("DEMO 3: Zeta Zero Pattern Encoding")
    print("=" * 70)
    
    # Get first 100 zeta zeros
    print("\n1. Computing zeta zero fiducials...")
    zeros = get_zeta_fiducials(K=100)
    print(f"   ✓ Loaded {len(zeros)} zeta zeros")
    print(f"   First 5: {zeros[:5]}")
    
    # Create visualization: zeros as vertical lines at x = gamma_k
    size = 256
    img = np.zeros((size, size), dtype=np.float32)
    
    # Map zeros to image coordinates
    z_min, z_max = zeros.min(), zeros.max()
    for z in zeros:
        x = int((z - z_min) / (z_max - z_min) * (size - 1))
        img[:, x] = 1.0
    
    print(f"\n2. Created {size}×{size} visualization of zeta zeros")
    
    # Encode to .holo
    print("\n3. Encoding to holographic format...")
    holo_file = "demo_zeta.holo"
    encode_holo(img, holo_file, kx=0.18, ky=0.18, quantize=True)
    size_bytes = Path(holo_file).stat().st_size
    print(f"   ✓ Saved to {holo_file} ({size_bytes} bytes)")
    
    # Decode
    print("\n4. Decoding...")
    recon = decode_holo(holo_file)
    psnr = compute_psnr(img, recon)
    print(f"   ✓ PSNR: {psnr:.2f} dB")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap='hot')
    axes[0].set_title('Zeta Zero Positions')
    axes[0].axis('off')
    
    axes[1].imshow(recon, cmap='hot')
    axes[1].set_title(f'Reconstructed (PSNR: {psnr:.1f} dB)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_zeta_viz.png', dpi=150, bbox_inches='tight')
    print(f"\n5. Visualization saved to demo_zeta_viz.png")
    plt.close()


# ========== Demo 4: Angle Multiplexing ==========

def demo_multiplexing():
    """Demonstrate angle multiplexing: multiple images in one hologram."""
    print("\n" + "=" * 70)
    print("DEMO 4: Angle Multiplexing (Multiple Images in One Hologram)")
    print("=" * 70)
    
    print("\n1. Creating three test patterns...")
    size = 128
    
    # Pattern 1: Horizontal stripes
    img1 = np.zeros((size, size), dtype=np.float32)
    for i in range(0, size, 16):
        img1[i:i+8, :] = 1.0
    
    # Pattern 2: Vertical stripes
    img2 = np.zeros((size, size), dtype=np.float32)
    for j in range(0, size, 16):
        img2[:, j:j+8] = 1.0
    
    # Pattern 3: Diagonal stripes
    img3 = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            if (i + j) % 16 < 8:
                img3[i, j] = 1.0
    
    print("   ✓ Created 3 patterns: horizontal, vertical, diagonal stripes")
    
    # Encode each with different reference angles
    print("\n2. Encoding with different reference wave angles...")
    angles = [(0.1, 0.1), (0.2, 0.0), (0.0, 0.2)]
    images = [img1, img2, img3]
    holo_files = []
    
    for i, (img, (kx, ky)) in enumerate(zip(images, angles)):
        holo_file = f"demo_mux_{i}.holo"
        encode_holo(img, holo_file, kx=kx, ky=ky, quantize=True)
        holo_files.append(holo_file)
        print(f"   ✓ Pattern {i+1} encoded with angle ({kx}, {ky})")
    
    # Simulate multiplexing by summing holograms
    print("\n3. Multiplexing: combining holograms...")
    import json
    
    holo_sum = None
    for holo_file in holo_files:
        with open(holo_file, 'rb') as f:
            f.read(128)  # Skip header
            holo = np.load(f).astype(np.float32)
        if holo_sum is None:
            holo_sum = holo
        else:
            holo_sum += holo
    
    # Normalize
    holo_sum = holo_sum / len(holo_files)
    
    # Save multiplexed hologram
    mux_file = "demo_multiplexed.holo"
    # Use first pattern's metadata for header
    with open(holo_files[0], 'rb') as f:
        header_bytes = f.read(128)
    with open(mux_file, 'wb') as f:
        f.write(header_bytes)
        np.save(f, holo_sum.astype(np.uint8))
    
    print(f"   ✓ Multiplexed hologram saved to {mux_file}")
    
    # Decode each angle
    print("\n4. Decoding individual patterns from multiplexed hologram...")
    recons = []
    psnrs = []
    
    for i, (img, (kx, ky)) in enumerate(zip(images, angles)):
        # Manually decode with specific angle
        import json
        with open(mux_file, 'rb') as f:
            header_bytes = f.read(128)
            header_str = header_bytes.rstrip(b'\0').decode('utf-8')
            header = json.loads(header_str)
            holo = np.load(f).astype(np.float32)
        
        # Override angle in header
        header['kx'] = kx
        header['ky'] = ky
        
        # Save temporary file with correct angle
        temp_file = f"temp_decode_{i}.holo"
        header_str = json.dumps(header)
        header_bytes = header_str.encode('utf-8').ljust(128, b'\0')
        with open(temp_file, 'wb') as f:
            f.write(header_bytes)
            np.save(f, holo.astype(np.uint8))
        
        recon = decode_holo(temp_file)
        recons.append(recon)
        psnr = compute_psnr(img, recon)
        psnrs.append(psnr)
        print(f"   ✓ Pattern {i+1} decoded with PSNR: {psnr:.2f} dB")
        
        Path(temp_file).unlink()  # Clean up
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(3):
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].set_title(f'Original Pattern {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recons[i], cmap='gray')
        axes[1, i].set_title(f'Reconstructed (PSNR: {psnrs[i]:.1f} dB)')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_multiplexing_viz.png', dpi=150, bbox_inches='tight')
    print(f"\n5. Visualization saved to demo_multiplexing_viz.png")
    plt.close()
    
    print("\nNote: Multiplexing quality depends on angular separation.")
    print("      Larger angular differences improve isolation between patterns.")


# ========== Main Demo Runner ==========

def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("HOLOGRAPHIC FILE FORMAT (.holo) DEMONSTRATION")
    print("Integrating with resfrac holographic tools")
    print("=" * 70)
    
    try:
        demo_basic()
        demo_primes()
        demo_zeta_zeros()
        demo_multiplexing()
        
        print("\n" + "=" * 70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - demo_basic.holo, demo_basic_viz.png")
        print("  - demo_primes.holo, demo_primes_viz.png")
        print("  - demo_zeta.holo, demo_zeta_viz.png")
        print("  - demo_multiplexed.holo, demo_multiplexing_viz.png")
        print("\nNext steps:")
        print("  1. Run: python -m resfrac.tools.holo_file --benchmark")
        print("  2. Integrate with HolographicSublinearIndex for data storage")
        print("  3. Explore phase retrieval for improved reconstruction")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
