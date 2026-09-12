import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from entropy import calculate_block_entropy, select_optimal_embedding_blocks

class TestEntropy(unittest.TestCase):
    def test_uniform_block_has_zero_entropy(self):
        block = [128] * 64
        self.assertEqual(calculate_block_entropy(block), 0.0)

    def test_high_variance_block_has_high_entropy(self):
        block = list(range(64))
        ent = calculate_block_entropy(block)
        self.assertGreater(ent, 5.0)

    def test_select_optimal_blocks(self):
        # First 64 uniform (entropy 0), second 64 noisy (entropy high)
        pixels = [100] * 64 + [(i * 37) % 256 for i in range(64)]
        selected = select_optimal_embedding_blocks(pixels, block_size=64, min_entropy=2.0)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['block_index'], 1)

if __name__ == '__main__':
    unittest.main()
