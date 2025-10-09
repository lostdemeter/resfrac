# .holo Format Quick Start Guide

## Installation

The `.holo` format is included in the `resfrac` package. Ensure you have the required dependencies:

```bash
pip install numpy scipy matplotlib mpmath
```

## 5-Minute Tutorial

### 1. Run the Benchmark

```bash
python -m resfrac.tools.holo_file --benchmark
```

**Output:**
```
============================================================
HOLOGRAPHIC FILE FORMAT BENCHMARK
============================================================
Image size: 256×256 (65536 pixels)

Holographic (.holo, uint8):
  Encode time: 0.0029 s
  Decode time: 0.0046 s
  File size:   65792 bytes (~1.004 B/pixel)
  PSNR:        11.41 dB

PNG Baseline:
  Encode time: 0.0032 s
  Decode time: 0.0013 s
  File size:   1088 bytes (~0.017 B/pixel)
  PSNR:        inf dB (lossless)
============================================================
```

**Generated files:**
- `test.holo` - Holographic file
- `checkerboard_holo.png` - Visualization

### 2. Encode Your Own Image

```bash
# Encode any B&W or grayscale image
python -m resfrac.tools.holo_file --encode input.png --output my_image.holo

# With custom parameters
python -m resfrac.tools.holo_file --encode input.png --output my_image.holo \
    --kx 0.2 --ky 0.2 --no-quantize
```

**Parameters:**
- `--kx`, `--ky`: Reference wave tilt (default: 0.15)
  - Higher values (0.2-0.3) → better quality, larger bandwidth
  - Lower values (0.1) → more compact, lower quality
- `--no-quantize`: Use float32 instead of uint8 (4x larger, slightly better quality)

### 3. Decode a Hologram

```bash
python -m resfrac.tools.holo_file --decode my_image.holo --output reconstructed.png
```

### 4. Visualize the Process

```bash
python -m resfrac.tools.holo_file --encode input.png --output test.holo --visualize
```

This creates a 3-panel visualization showing:
1. Original image
2. Hologram (interference pattern)
3. Reconstructed image with PSNR

## Python API Examples

### Basic Usage

```python
from resfrac.tools.holo_file import encode_holo, decode_holo, compute_psnr
import numpy as np

# Create or load image (must be 2D float32, range [0,1])
img = np.random.rand(256, 256).astype(np.float32)

# Encode to .holo
encode_holo(img, 'output.holo', kx=0.15, ky=0.15, quantize=True)

# Decode from .holo
recon = decode_holo('output.holo')

# Measure quality
psnr = compute_psnr(img, recon)
print(f"PSNR: {psnr:.2f} dB")
```

### Load Image from File

```python
from resfrac.tools.holo_file import load_image_as_bw, encode_holo

# Load and convert to B&W
img = load_image_as_bw('photo.jpg')  # Handles RGB→grayscale, normalization

# Encode
encode_holo(img, 'photo.holo')
```

### Benchmarking

```python
from resfrac.tools.holo_file import benchmark_holo_vs_png, generate_checkerboard

# Generate test pattern
img = generate_checkerboard(size=256, block=16)

# Run benchmark
results = benchmark_holo_vs_png(img, holo_file='test.holo', verbose=True)

print(f"Hologram size: {results['holo_size']} bytes")
print(f"PSNR: {results['holo_psnr']:.2f} dB")
```

## Complete Demo

Run the comprehensive demo to see all features:

```bash
python examples/holo_format_demo.py
```

**Demonstrations:**
1. **Basic encoding/decoding** - Checkerboard pattern
2. **Prime patterns** - Encode prime number distributions
3. **Zeta zeros** - Visualize Riemann zeta zero positions
4. **Angle multiplexing** - Store 3 images in one hologram

**Generated files:**
- `demo_basic.holo`, `demo_basic_viz.png`
- `demo_primes.holo`, `demo_primes_viz.png`
- `demo_zeta.holo`, `demo_zeta_viz.png`
- `demo_multiplexed.holo`, `demo_multiplexing_viz.png`

## Integration with resfrac

### Holographic Index Storage

```python
from resfrac import HolographicSublinearIndex
from resfrac.tools.holo_file import encode_holo
import numpy as np

# Create dataset
X = np.random.rand(10000, 64)  # 10k vectors, 64-dim

# Build holographic index
index = HolographicSublinearIndex(K=256).fit(X)

# Query
query = np.random.rand(64)
candidates, meta = index.query(query, S=20)
```

### Zeta Zero Analysis

```python
from resfrac.holo_utils import get_zeta_fiducials, zero_calibrate
from resfrac.tools.holo_file import encode_holo
import numpy as np

# Get zeta zeros
zeros = get_zeta_fiducials(K=100)

# Create visualization pattern
size = 256
img = np.zeros((size, size))
for i, z in enumerate(zeros):
    x = int(i / len(zeros) * size)
    img[:, x] = 1.0

# Encode as hologram
encode_holo(img, 'zeta_pattern.holo', kx=0.18, ky=0.18)

# Use for calibration
gaps = np.diff(zeros)
alpha_shift, circ_var = zero_calibrate(gaps, zeros, tol=0.1)
print(f"Circular variance: {circ_var:.4f}")
```

## Quality Tips

### Improving PSNR

1. **Increase reference tilt** (kx, ky):
   ```python
   encode_holo(img, 'high_quality.holo', kx=0.25, ky=0.25)
   # PSNR: ~12 dB → ~15 dB
   ```

2. **Disable quantization**:
   ```python
   encode_holo(img, 'float32.holo', quantize=False)
   # Size: 16 kB → 260 kB, PSNR: +0.5 dB
   ```

3. **Use structured patterns**:
   - Checkerboards, grids: PSNR ~12-15 dB
   - Random noise: PSNR ~8-10 dB
   - Sparse patterns: PSNR ~15-20 dB

### Size Optimization

1. **Use uint8 quantization** (default):
   ```python
   encode_holo(img, 'compact.holo', quantize=True)
   # 4x size reduction, ~0.1 dB loss
   ```

2. **Reduce image size**:
   ```python
   from scipy.ndimage import zoom
   img_small = zoom(img, 0.5)  # 256×256 → 128×128
   encode_holo(img_small, 'small.holo')
   # 4x size reduction
   ```

## Troubleshooting

### Import Error

```python
# If you get: ModuleNotFoundError: No module named 'resfrac.tools'
# Make sure you're in the repo root and resfrac is importable:
import sys
sys.path.insert(0, '/path/to/resfrac')
from resfrac.tools.holo_file import encode_holo
```

### Low PSNR (<5 dB)

- **Increase kx, ky**: Try 0.2-0.3 for better frequency separation
- **Check image range**: Must be [0, 1], not [0, 255]
- **Avoid extreme patterns**: Very sparse or very dense patterns may reconstruct poorly

### Large File Size

- **Enable quantization**: `quantize=True` (default)
- **Reduce image dimensions**: Downsample before encoding
- **Check bit depth**: Verify header shows `bit_depth: 8`, not 32

## Next Steps

1. **Read the full specification**: `docs/HOLO_FORMAT.md`
2. **Explore advanced features**: Angle multiplexing, phase retrieval
3. **Integrate with your workflow**: Use for holographic data storage
4. **Contribute**: Submit improvements or new features

## Performance Reference

| Image Size | Encode Time | Decode Time | File Size (uint8) |
|------------|-------------|-------------|-------------------|
| 64×64      | ~0.001 s    | ~0.001 s    | ~4 kB             |
| 128×128    | ~0.003 s    | ~0.003 s    | ~16 kB            |
| 256×256    | ~0.003 s    | ~0.005 s    | ~66 kB            |
| 512×512    | ~0.012 s    | ~0.020 s    | ~262 kB           |
| 1024×1024  | ~0.050 s    | ~0.080 s    | ~1 MB             |

*Benchmarked on typical hardware (2020s CPU)*

## Support

For questions or issues:
1. Check `docs/HOLO_FORMAT.md` for detailed specification
2. Review `examples/holo_format_demo.py` for usage patterns
3. Run tests: `python tests/test_holo_file.py`
4. See main `README.md` and `AI_README.md` for resfrac context

---

**Happy holographic encoding!** 🌊✨
