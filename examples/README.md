# Holographic File Format Examples

This directory contains demonstrations of the `.holo` file format for holographic image encoding.

## Quick Start

Run all demonstrations:

```bash
python examples/holo_format_demo.py
```

This will generate 8 files showing different use cases.

## Demonstrations

### Demo 1: Basic Encoding/Decoding
**Pattern**: 256×256 checkerboard  
**PSNR**: 11.41 dB  
**Output**: `demo_basic.holo`, `demo_basic_viz.png`

Shows the fundamental encoding process:
- Original image (left)
- Hologram interference pattern (center)
- Reconstructed image (right)

### Demo 2: Prime Pattern Encoding
**Pattern**: 1900 primes in 128×128 grid  
**PSNR**: 8.67 dB  
**Accuracy**: 88.4% (binary threshold)  
**Output**: `demo_primes.holo`, `demo_primes_viz.png`

Demonstrates encoding sparse patterns (prime number positions).

### Demo 3: Zeta Zero Visualization
**Pattern**: 100 Riemann zeta zeros  
**PSNR**: 6.37 dB  
**Output**: `demo_zeta.holo`, `demo_zeta_viz.png`

Integration with `resfrac.holo_utils.get_zeta_fiducials()` to visualize zeta zero positions as vertical lines.

### Demo 4: Angle Multiplexing
**Patterns**: 3 stripe patterns (horizontal, vertical, diagonal)  
**PSNR**: 7.16-7.50 dB per pattern  
**Output**: `demo_multiplexed.holo`, `demo_multiplexing_viz.png`

Shows how multiple images can be stored in a single hologram using different reference wave angles.

## Generated Files

After running the demo:

```
examples/
├── holo_format_demo.py          # Demo script
├── README.md                     # This file
└── (generated files in repo root:)
    ├── demo_basic.holo           # 66 KB
    ├── demo_basic_viz.png        # 204 KB
    ├── demo_primes.holo          # 17 KB
    ├── demo_primes_viz.png       # 68 KB
    ├── demo_zeta.holo            # 66 KB
    ├── demo_zeta_viz.png         # 127 KB
    ├── demo_multiplexed.holo     # 17 KB
    └── demo_multiplexing_viz.png # 76 KB
```

## Customization

Edit `holo_format_demo.py` to:
- Change image sizes
- Adjust reference wave angles (kx, ky)
- Try different patterns
- Experiment with quantization settings

## Integration Examples

### With Holographic Index

```python
from resfrac import HolographicSublinearIndex
from resfrac.tools.holo_file import encode_holo
import numpy as np

# Create dataset
X = np.random.rand(10000, 64)

# Build index
index = HolographicSublinearIndex(K=256).fit(X)

# Encode for storage
X_2d = X.reshape(100, 100, 64).mean(axis=2)
encode_holo(X_2d, 'dataset.holo')
```

### With Zeta Calibration

```python
from resfrac.holo_utils import get_zeta_fiducials, zero_calibrate
from resfrac.tools.holo_file import encode_holo

# Get zeta zeros
zeros = get_zeta_fiducials(K=100)

# Create pattern and encode
img = create_zeta_pattern(zeros)
encode_holo(img, 'zeta_pattern.holo')

# Use for calibration
gaps = compute_gaps(tour)
alpha_shift, circ_var = zero_calibrate(gaps, zeros)
```

## Performance

All demos run in ~2 seconds total on typical hardware.

Individual encoding times:
- 128×128: ~0.003 s
- 256×256: ~0.003 s

## Documentation

See:
- `docs/HOLO_FORMAT.md` - Complete specification
- `docs/HOLO_QUICKSTART.md` - 5-minute tutorial
- `HOLO_DELIVERY.md` - Delivery summary

## Support

For issues or questions, see the main repository documentation.
