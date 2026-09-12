import unittest
from scripts.color_space_converter import ColorSpaceConverter

class TestColorSpaceConverter(unittest.TestCase):
    def test_roundtrip_color_conversion(self):
        r_orig, g_orig, b_orig = 120, 80, 200
        y, cb, cr = ColorSpaceConverter.rgb_to_ycbcr(r_orig, g_orig, b_orig)
        r_out, g_out, b_out = ColorSpaceConverter.ycbcr_to_rgb(y, cb, cr)
        self.assertAlmostEqual(r_orig, r_out, delta=2)
        self.assertAlmostEqual(g_orig, g_out, delta=2)
        self.assertAlmostEqual(b_orig, b_out, delta=2)

if __name__ == '__main__':
    unittest.main()
