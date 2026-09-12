import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from histogram_equalization import (
    compute_histogram,
    compute_cdf,
    equalize_histogram,
    clip_histogram,
    adaptive_equalize,
)


class TestHistogram(unittest.TestCase):
    def test_histogram_counts_pixels_correctly(self):
        pixels = [0, 0, 128, 255, 255, 255]
        hist = compute_histogram(pixels)
        self.assertEqual(hist[0], 2)
        self.assertEqual(hist[128], 1)
        self.assertEqual(hist[255], 3)
        self.assertEqual(sum(hist), 6)

    def test_histogram_all_same_value(self):
        pixels = [100] * 50
        hist = compute_histogram(pixels)
        self.assertEqual(hist[100], 50)
        self.assertEqual(sum(hist), 50)


class TestCdf(unittest.TestCase):
    def test_cdf_ends_at_one(self):
        hist = compute_histogram(list(range(256)))
        cdf = compute_cdf(hist)
        self.assertAlmostEqual(cdf[-1], 1.0, places=5)

    def test_cdf_is_monotonically_non_decreasing(self):
        import random
        pixels = [random.randint(0, 255) for _ in range(1000)]
        cdf = compute_cdf(compute_histogram(pixels))
        for i in range(1, len(cdf)):
            self.assertGreaterEqual(cdf[i], cdf[i - 1])

    def test_cdf_on_empty_histogram_returns_zeros(self):
        cdf = compute_cdf([0] * 256)
        self.assertEqual(cdf, [0.0] * 256)


class TestGlobalEqualization(unittest.TestCase):
    def test_output_length_matches_input(self):
        pixels = list(range(256))
        equalized = equalize_histogram(pixels)
        self.assertEqual(len(equalized), len(pixels))

    def test_output_values_in_valid_range(self):
        import random
        pixels = [random.randint(0, 255) for _ in range(512)]
        equalized = equalize_histogram(pixels)
        for p in equalized:
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 255)

    def test_constant_image_equalization(self):
        pixels = [128] * 100
        equalized = equalize_histogram(pixels)
        # All constant — result should be 255 (CDF = 1.0)
        self.assertTrue(all(p == 255 for p in equalized))


class TestClipHistogram(unittest.TestCase):
    def test_clip_limit_invalid_raises(self):
        with self.assertRaises(ValueError):
            clip_histogram([10] * 256, clip_limit=0)

    def test_total_pixel_count_preserved(self):
        hist = [20] * 128 + [5] * 128
        original_total = sum(hist)
        clipped, _ = clip_histogram(hist, clip_limit=1.5)
        self.assertEqual(sum(clipped), original_total)


class TestAdaptiveEqualize(unittest.TestCase):
    def test_output_length_matches_input(self):
        pixels = [(i * 13 + 50) % 256 for i in range(64 * 64)]
        result = adaptive_equalize(pixels, width=64, height=64, tile_size=8)
        self.assertEqual(len(result), len(pixels))


    def test_raises_on_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            adaptive_equalize(list(range(100)), width=20, height=20)

    def test_output_in_valid_range(self):
        import random
        pixels = [random.randint(0, 255) for _ in range(32 * 32)]
        result = adaptive_equalize(pixels, width=32, height=32, tile_size=8)
        for p in result:
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 255)


if __name__ == '__main__':
    unittest.main()
