import unittest
from scripts.robustness_benchmarker import RobustnessBenchmarker

class TestRobustnessBenchmarker(unittest.TestCase):
    def test_zero_ber_for_perfect_recovery(self):
        bits = [1, 0, 1, 1, 0, 0, 1]
        self.assertEqual(RobustnessBenchmarker.compute_bit_error_rate(bits, bits), 0.0)

    def test_calculated_ber_mismatch(self):
        b1 = [1, 0, 1, 0]
        b2 = [1, 0, 0, 1] # 2 mismatches out of 4 = 0.5
        self.assertEqual(RobustnessBenchmarker.compute_bit_error_rate(b1, b2), 0.5)

if __name__ == '__main__':
    unittest.main()
