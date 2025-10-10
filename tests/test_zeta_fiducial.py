#!/usr/bin/env python3
"""
Unit tests for zeta-fiducial holographic utilities.
"""
import time
import numpy as np
import unittest

from resfrac.tools import (
    zeta_points,
    zeta_fringe_cartographer,
    zeta_sfft,
    tune_walltime,
)


class TestZetaFiducial(unittest.TestCase):
    def test_points_shape_and_range(self):
        K = 50
        pts = zeta_points(K)
        self.assertEqual(pts.shape, (K, 2))
        # x monotonic and within [-0.5, 0.5]
        xs = pts[:, 0]
        ys = pts[:, 1]
        self.assertTrue(np.all(np.diff(xs) > 0))
        self.assertTrue(np.all(xs >= -0.5) and np.all(xs <= 0.5))
        self.assertTrue(np.all(ys >= -0.5) and np.all(ys <= 0.5))

    def test_cartographer_and_sfft_smoke(self):
        K = 30
        grid = 128
        carrier = 0.15
        sigma = 0.05
        crop = 32
        pts = zeta_points(K)
        t0 = time.time()
        H = zeta_fringe_cartographer(pts, grid=grid, carrier=carrier, sigma=sigma)
        recon, stats = zeta_sfft(H, carrier=carrier, crop=crop)
        dt = time.time() - t0
        self.assertEqual(H.shape, (grid, grid))
        self.assertGreater(stats["max_intensity"], 0.0)
        self.assertLess(dt, 1.0)  # should be quick

    def test_tuner_returns_reasonable_params(self):
        best = tune_walltime(K=20, grid=128, crop=32)
        self.assertIn("carrier", best)
        self.assertIn("sigma", best)
        self.assertIn("max_intensity", best)
        self.assertIn("time", best)
        self.assertGreater(best["max_intensity"], 0.0)


if __name__ == "__main__":
    unittest.main()
