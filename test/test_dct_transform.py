import unittest
from scripts.dct_transform import DCTProcessor

class TestDCTProcessor(unittest.TestCase):
    def test_dct_and_inverse_reconstruction(self):
        signal = [12.0, 34.0, 56.0, 78.0, 90.0, 110.0, 130.0, 150.0]
        coeffs = DCTProcessor.dct_1d(signal)
        reconstructed = DCTProcessor.idct_1d(coeffs)
        for orig, rec in zip(signal, reconstructed):
            self.assertAlmostEqual(orig, rec, places=2)

if __name__ == '__main__':
    unittest.main()
