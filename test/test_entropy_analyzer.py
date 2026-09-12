import unittest
from scripts.entropy_analyzer import EntropyAnalyzer

class TestEntropyAnalyzer(unittest.TestCase):
    def test_zero_entropy_for_uniform_region(self):
        uniform = [128] * 64
        self.assertEqual(EntropyAnalyzer.calculate_shannon_entropy(uniform), 0.0)
        self.assertEqual(EntropyAnalyzer.classify_region_texture(0.0), "SMOOTH")

    def test_high_entropy_for_textured_region(self):
        # 64 unique values has log2(64) = 6.0 entropy
        textured = list(range(64))
        ent = EntropyAnalyzer.calculate_shannon_entropy(textured)
        self.assertEqual(ent, 6.0)
        self.assertEqual(EntropyAnalyzer.classify_region_texture(ent), "TEXTURED")

if __name__ == '__main__':
    unittest.main()
