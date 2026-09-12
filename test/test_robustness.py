import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from robustness import (
    add_gaussian_noise,
    apply_brightness_shift,
    jpeg_quantize_block,
    compute_ber,
    robustness_report,
)
from watermark_pipeline import embed_lsb, extract_lsb


class TestDegradationFunctions(unittest.TestCase):
    def test_gaussian_noise_clamps_range(self):
        pixels = [128] * 512
        noisy = add_gaussian_noise(pixels, std_dev=50)
        for p in noisy:
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 255)

    def test_brightness_shift_positive(self):
        pixels = [100, 150, 200]
        shifted = apply_brightness_shift(pixels, 60)
        self.assertEqual(shifted, [160, 210, 255])

    def test_brightness_shift_negative_clamped(self):
        pixels = [10, 20, 30]
        shifted = apply_brightness_shift(pixels, -50)
        self.assertEqual(shifted, [0, 0, 0])

    def test_jpeg_quantize_reduces_precision(self):
        pixels = [123, 145, 167, 189]
        quantized = jpeg_quantize_block(pixels, quality=50)
        step = max(1, round((100 - 50) / 4))
        for orig, quant in zip(pixels, quantized):
            self.assertEqual(quant % step, 0)

    def test_jpeg_quantize_quality_100_is_lossless(self):
        pixels = list(range(0, 256, 16))
        quantized = jpeg_quantize_block(pixels, quality=100)
        self.assertEqual(quantized, pixels)

    def test_jpeg_quantize_invalid_quality_raises(self):
        with self.assertRaises(ValueError):
            jpeg_quantize_block([128], quality=0)
        with self.assertRaises(ValueError):
            jpeg_quantize_block([128], quality=101)


class TestBitErrorRate(unittest.TestCase):
    def test_ber_identical_sequences(self):
        bits = [0, 1, 1, 0, 1]
        self.assertEqual(compute_ber(bits, bits), 0.0)

    def test_ber_all_flipped(self):
        original = [0, 1, 0, 1]
        flipped  = [1, 0, 1, 0]
        self.assertAlmostEqual(compute_ber(original, flipped), 1.0)

    def test_ber_half_flipped(self):
        original = [0, 1, 0, 1]
        modified = [1, 1, 1, 1]
        self.assertAlmostEqual(compute_ber(original, modified), 0.5)

    def test_ber_raises_on_length_mismatch(self):
        with self.assertRaises(ValueError):
            compute_ber([0, 1], [0, 1, 0])


class TestRobustnessReport(unittest.TestCase):
    def _make_pixels(self, size=8192):
        return [(i * 37 + 64) % 200 + 30 for i in range(size)]

    def test_perfect_recovery_without_degradation(self):
        pixels = self._make_pixels()
        msg = "MARK"
        report = robustness_report(
            pixels,
            extract_fn=extract_lsb,
            embed_fn=embed_lsb,
            message=msg,
            degradations=[('no_degradation', lambda x: list(x))],
        )
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]['recovered'], msg)
        self.assertEqual(report[0]['ber'], 0.0)
        self.assertTrue(report[0]['intact'])


if __name__ == '__main__':
    unittest.main()
