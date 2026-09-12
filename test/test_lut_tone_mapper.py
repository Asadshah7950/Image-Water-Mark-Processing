import unittest
from scripts.lut_tone_mapper import LUTToneMapper

class TestLUTToneMapper(unittest.TestCase):
    def test_linear_lut_with_gamma_one(self):
        lut = LUTToneMapper.build_gamma_lut(1.0)
        for i in range(256):
            self.assertEqual(lut[i], i)

    def test_apply_lut_transformation(self):
        lut = LUTToneMapper.build_gamma_lut(2.2)
        pixels = [0, 128, 255]
        out = LUTToneMapper.apply_lut(pixels, lut)
        self.assertEqual(out[0], 0)
        self.assertEqual(out[2], 255)
        self.assertGreater(out[1], 128) # gamma > 1 brightens midtones

if __name__ == '__main__':
    unittest.main()
