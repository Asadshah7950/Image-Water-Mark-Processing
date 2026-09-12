import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from color_channel import decompose_channels, interleave_channels, compute_luminance

class TestColorChannel(unittest.TestCase):
    def test_decompose_and_interleave_roundtrip(self):
        original = [10, 20, 30, 40, 50, 60]
        channels = decompose_channels(original)
        self.assertEqual(channels['R'], [10, 40])
        self.assertEqual(channels['G'], [20, 50])
        self.assertEqual(channels['B'], [30, 60])
        rebuilt = interleave_channels(channels['R'], channels['G'], channels['B'])
        self.assertEqual(rebuilt, original)

    def test_invalid_rgb_length_raises(self):
        with self.assertRaises(ValueError):
            decompose_channels([1, 2])

    def test_luminance_calculation(self):
        lum = compute_luminance(255, 255, 255)
        self.assertAlmostEqual(lum, 255.0, places=3)
        lum_green = compute_luminance(0, 255, 0)
        self.assertGreater(lum_green, compute_luminance(0, 0, 255))

if __name__ == '__main__':
    unittest.main()
