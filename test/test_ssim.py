import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from ssim import ssim_block, ssim_global

class TestSSIM(unittest.TestCase):
    def test_identical_blocks_have_score_one(self):
        block = [100, 120, 140, 160] * 16
        score = ssim_block(block, block)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_slightly_perturbed_blocks_have_high_ssim(self):
        b1 = [100, 120, 140, 160] * 16
        b2 = [p + 1 for p in b1]
        score = ssim_block(b1, b2)
        self.assertGreater(score, 0.98)

    def test_drastically_different_blocks_have_low_ssim(self):
        b1 = [0] * 64
        b2 = [255] * 64
        score = ssim_block(b1, b2)
        self.assertLess(score, 0.1)

    def test_length_mismatch_raises_error(self):
        with self.assertRaises(ValueError):
            ssim_block([100], [100, 100])

    def test_global_ssim_range(self):
        img1 = [(i * 17) % 256 for i in range(512)]
        img2 = [(p + 2) % 256 for p in img1]
        score = ssim_global(img1, img2, block_size=64)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

if __name__ == '__main__':
    unittest.main()
