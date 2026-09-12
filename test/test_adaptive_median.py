import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from adaptive_median import apply_median_filter, get_neighborhood_window

class TestMedianFilter(unittest.TestCase):
    def test_window_extraction_bounds(self):
        pixels = list(range(9)) # 3x3
        win = get_neighborhood_window(pixels, 3, 3, 0, 0, 1)
        self.assertEqual(len(win), 4) # Corner: 4 neighbors

    def test_removes_single_salt_pixel(self):
        # 3x3 image with 0s and one 255 noise pixel at center
        pixels = [0, 0, 0, 0, 255, 0, 0, 0, 0]
        filtered = apply_median_filter(pixels, 3, 3, radius=1)
        self.assertEqual(filtered[4], 0) # Center noise replaced by median 0

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            apply_median_filter([1, 2], 3, 3)

if __name__ == '__main__':
    unittest.main()
