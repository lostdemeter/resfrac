#!/usr/bin/env python3
"""
test_holo_file.py
Unit tests for .holo file format encoding/decoding.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from resfrac.tools.holo_file import (
    encode_holo, decode_holo, generate_checkerboard,
    compute_psnr, benchmark_holo_vs_png, generate_color_gradient
)


class TestHoloFileFormat(unittest.TestCase):
    """Test suite for .holo file format."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_encode_decode_roundtrip(self):
        """Test that encode->decode preserves image structure."""
        # Create simple test image
        img = np.zeros((64, 64), dtype=np.float32)
        img[16:48, 16:48] = 1.0  # White square in center
        
        holo_file = self.temp_path / "test.holo"
        
        # Encode
        encode_holo(img, holo_file, quantize=True)
        self.assertTrue(holo_file.exists())
        
        # Decode
        recon = decode_holo(holo_file)
        
        # Check shape preservation
        self.assertEqual(recon.shape, img.shape)
        
        # Check PSNR is reasonable (>10 dB for structured pattern)
        psnr = compute_psnr(img, recon)
        self.assertGreater(psnr, 10.0, f"PSNR too low: {psnr:.2f} dB")
    
    def test_checkerboard_fidelity(self):
        """Test encoding/decoding of checkerboard pattern."""
        img = generate_checkerboard(size=128, block=8)
        holo_file = self.temp_path / "checker.holo"
        
        encode_holo(img, holo_file, kx=0.15, ky=0.15, quantize=True)
        recon = decode_holo(holo_file)
        
        psnr = compute_psnr(img, recon)
        # Checkerboard should reconstruct well with off-axis holography
        self.assertGreater(psnr, 8.0, f"Checkerboard PSNR too low: {psnr:.2f} dB")
    
    def test_quantization_quality(self):
        """Test that uint8 quantization maintains acceptable quality."""
        img = generate_checkerboard(size=128, block=16)
        holo_file_q = self.temp_path / "quantized.holo"
        holo_file_f = self.temp_path / "float.holo"
        
        # Encode with quantization
        encode_holo(img, holo_file_q, quantize=True)
        recon_q = decode_holo(holo_file_q)
        psnr_q = compute_psnr(img, recon_q)
        
        # Encode without quantization
        encode_holo(img, holo_file_f, quantize=False)
        recon_f = decode_holo(holo_file_f)
        psnr_f = compute_psnr(img, recon_f)
        
        # Quantized should be within ~1-2 dB of float32
        self.assertGreater(psnr_q, psnr_f - 3.0, 
                          f"Quantization loss too high: {psnr_f - psnr_q:.2f} dB")
        
        # Quantized file should be ~4x smaller
        size_q = holo_file_q.stat().st_size
        size_f = holo_file_f.stat().st_size
        ratio = size_f / size_q
        self.assertGreater(ratio, 2.5, f"Quantization size reduction too low: {ratio:.2f}x")
    
    def test_file_size_efficiency(self):
        """Test that .holo file size is reasonable."""
        img = generate_checkerboard(size=256, block=16)
        holo_file = self.temp_path / "size_test.holo"
        
        encode_holo(img, holo_file, quantize=True)
        size = holo_file.stat().st_size
        
        # Should be <50kB for 256x256 uint8 (target from spec)
        self.assertLess(size, 50_000, f"File size too large: {size} bytes")
        
        # Should be >1kB (not over-compressed)
        self.assertGreater(size, 1_000, f"File size suspiciously small: {size} bytes")
    
    def test_different_tilts(self):
        """Test encoding with different reference wave tilts."""
        img = generate_checkerboard(size=128, block=16)
        
        tilts = [(0.1, 0.1), (0.15, 0.15), (0.2, 0.2), (0.3, 0.3)]
        psnrs = []
        
        for kx, ky in tilts:
            holo_file = self.temp_path / f"tilt_{kx}_{ky}.holo"
            encode_holo(img, holo_file, kx=kx, ky=ky, quantize=True)
            recon = decode_holo(holo_file)
            psnr = compute_psnr(img, recon)
            psnrs.append(psnr)
        
        # All tilts should produce reasonable reconstructions
        for i, psnr in enumerate(psnrs):
            self.assertGreater(psnr, 7.0, 
                             f"Tilt {tilts[i]} PSNR too low: {psnr:.2f} dB")
    
    def test_header_metadata(self):
        """Test that header metadata is correctly stored and retrieved."""
        img = np.random.rand(100, 100).astype(np.float32)
        holo_file = self.temp_path / "metadata.holo"
        
        kx, ky = 0.25, 0.18
        encode_holo(img, holo_file, kx=kx, ky=ky, quantize=True)
        
        # Read header manually
        import json
        with open(holo_file, 'rb') as f:
            header_bytes = f.read(512)  # fixed header size in current spec
            header_str = header_bytes.rstrip(b'\0').decode('utf-8')
            header = json.loads(header_str)
        
        # Verify metadata
        self.assertIn(header['version'], ('1.0', '1.1'))  # allow minor bump
        self.assertEqual(header['width'], 100)
        self.assertEqual(header['height'], 100)
        self.assertAlmostEqual(header['kx'], kx, places=5)
        self.assertAlmostEqual(header['ky'], ky, places=5)
        self.assertEqual(header['bit_depth'], 8)
        # Default mode should be offaxis1 unless changed by encode
        self.assertIn(header.get('mode', 'offaxis1'), ('offaxis1', 'ps4', 'complex'))
    
    def test_edge_cases(self):
        """Test edge cases: all black, all white, single pixel."""
        test_cases = [
            ("all_black", np.zeros((64, 64), dtype=np.float32)),
            ("all_white", np.ones((64, 64), dtype=np.float32)),
            ("single_pixel", np.array([[1.0]], dtype=np.float32)),
        ]
        
        for name, img in test_cases:
            holo_file = self.temp_path / f"{name}.holo"
            encode_holo(img, holo_file, quantize=True)
            recon = decode_holo(holo_file)
            
            # Should not crash and should preserve shape
            self.assertEqual(recon.shape, img.shape)
    
    def test_benchmark_runs(self):
        """Test that benchmark function runs without errors."""
        img = generate_checkerboard(size=128, block=16)
        # Default (offaxis1)
        results = benchmark_holo_vs_png(
            img,
            holo_file=str(self.temp_path / "bench.holo"),
            verbose=False,
            mode='offaxis1'
        )
        
        # Check that all expected keys are present
        expected_keys = [
            'holo_encode_time', 'holo_decode_time', 'holo_size', 'holo_psnr',
            'png_encode_time', 'png_decode_time', 'png_size', 'png_psnr'
        ]
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check that times are positive
        self.assertGreater(results['holo_encode_time'], 0)
        self.assertGreater(results['holo_decode_time'], 0)
        
        # Check that PSNR is finite
        self.assertTrue(np.isfinite(results['holo_psnr']))

        # PS4 near-lossless (uint8) should achieve high PSNR
        results_ps4 = benchmark_holo_vs_png(
            img,
            holo_file=str(self.temp_path / "bench_ps4.holo"),
            verbose=False,
            mode='ps4',
            quantize=True
        )
        self.assertGreater(results_ps4['holo_psnr'], 30.0)

        # Complex float32 should be lossless (infinite PSNR)
        results_cplx = benchmark_holo_vs_png(
            img,
            holo_file=str(self.temp_path / "bench_complex.holo"),
            verbose=False,
            mode='complex',
            quantize=False
        )
        self.assertTrue(np.isinf(results_cplx['holo_psnr']))

    def test_lossless_modes(self):
        """Lossless modes (ps4 float32, complex float32) reconstruct exactly."""
        img = generate_checkerboard(size=64, block=8)
        p_ps4 = self.temp_path / "lossless_ps4.holo"
        p_cplx = self.temp_path / "lossless_complex.holo"

        # PS4 float32
        encode_holo(img, p_ps4, quantize=False, mode='ps4')
        recon_ps4 = decode_holo(p_ps4)
        self.assertTrue(np.isinf(compute_psnr(img, recon_ps4)))

        # Complex float32
        encode_holo(img, p_cplx, quantize=False, mode='complex')
        recon_cplx = decode_holo(p_cplx)
        self.assertTrue(np.isinf(compute_psnr(img, recon_cplx)))

    def test_ps4color_lossless(self):
        """Lossless color via ps4color should reconstruct exactly with float32 payload."""
        img_rgb = generate_color_gradient(size=64, noise_std=0.0)
        p = self.temp_path / "lossless_ps4color.holo"
        encode_holo(img_rgb, p, quantize=False, mode='ps4color')
        recon = decode_holo(p)
        self.assertEqual(recon.shape, img_rgb.shape)
        self.assertTrue(np.isinf(compute_psnr(img_rgb, recon)))

    def test_color_mode_smoke(self):
        """Angle-multiplexed color should run encode/decode without errors and return RGB image."""
        img_rgb = generate_color_gradient(size=64, noise_std=0.05)
        p = self.temp_path / "color_smoke.holo"
        # Use uint8 quantization for small file; not necessarily high PSNR
        encode_holo(img_rgb, p, quantize=True, mode='color')
        recon = decode_holo(p)
        # Should reconstruct an RGB image in [0,1]
        self.assertEqual(recon.ndim, 3)
        self.assertEqual(recon.shape[:2], img_rgb.shape[:2])
        self.assertGreater(compute_psnr(img_rgb, recon), 3.0)


class TestHolographicProperties(unittest.TestCase):
    """Test holographic-specific properties."""
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_interference_pattern_structure(self):
        """Test that hologram shows interference fringes."""
        # Simple binary pattern
        img = np.zeros((128, 128), dtype=np.float32)
        img[48:80, 48:80] = 1.0
        
        holo_file = self.temp_path / "interference.holo"
        holo = encode_holo(img, holo_file, quantize=False)
        
        # Hologram should have higher frequency content than original
        # (due to interference fringes)
        img_fft = np.abs(np.fft.fft2(img))
        holo_fft = np.abs(np.fft.fft2(holo))
        
        # High frequency energy ratio
        h, w = img.shape
        high_freq_mask = np.zeros_like(img_fft, dtype=bool)
        high_freq_mask[h//4:3*h//4, w//4:3*w//4] = False
        high_freq_mask = ~high_freq_mask
        
        img_hf_energy = np.sum(img_fft[high_freq_mask])
        holo_hf_energy = np.sum(holo_fft[high_freq_mask])
        
        # Hologram should have more high-frequency content
        self.assertGreater(holo_hf_energy, img_hf_energy,
                          "Hologram should contain interference fringes")
    
    def test_off_axis_separation(self):
        """Test that off-axis encoding enables frequency separation."""
        img = generate_checkerboard(size=128, block=16)
        holo_file = self.temp_path / "offaxis.holo"
        
        # Encode with significant tilt
        kx, ky = 0.2, 0.2
        holo = encode_holo(img, holo_file, kx=kx, ky=ky, quantize=False)
        
        # FFT of hologram should show three distinct orders:
        # - Zero order (DC)
        # - Object term (shifted by +k)
        # - Twin image (shifted by -k)
        holo_fft = np.fft.fftshift(np.fft.fft2(holo))
        
        # Check for energy concentration away from center
        h, w = holo.shape
        center_mask = np.zeros_like(holo_fft, dtype=bool)
        cy, cx = h // 2, w // 2
        r = 10
        y, x = np.ogrid[:h, :w]
        center_mask = (x - cx)**2 + (y - cy)**2 < r**2
        
        center_energy = np.sum(np.abs(holo_fft[center_mask]))
        total_energy = np.sum(np.abs(holo_fft))
        
        # Most energy should be outside center (in shifted orders)
        self.assertLess(center_energy / total_energy, 0.5,
                       "Off-axis hologram should have shifted frequency content")


def run_tests():
    """Run all tests and print results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestHoloFileFormat))
    suite.addTests(loader.loadTestsFromTestCase(TestHolographicProperties))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
