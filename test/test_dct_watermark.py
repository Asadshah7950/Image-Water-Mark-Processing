import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from dct_watermark import (
    dct8,
    idct8,
    embed_dct_watermark,
    extract_dct_watermark_bit,
    embed_dct_message,
    extract_dct_message,
    BLOCK_SIZE,
    EMBED_DELTA,
)

class TestDctTransform(unittest.TestCase):
    def test_dct_idct_roundtrip(self):
        """DCT followed by IDCT should reconstruct the original block within floating-point tolerance."""
        block = [120, 135, 145, 130, 125, 110, 140, 150]
        coeffs = dct8(block)
        recovered = idct8(coeffs)
        for orig, rec in zip(block, recovered):
            self.assertAlmostEqual(orig, rec, places=6)

    def test_dct_energy_compaction(self):
        """DC coefficient should carry the dominant energy for a smooth block."""
        block = [100] * BLOCK_SIZE  # Flat block — all energy in DC
        coeffs = dct8(block)
        # DC component should be dominant vs sum of all AC components
        dc_energy = coeffs[0] ** 2
        ac_energy = sum(c ** 2 for c in coeffs[1:])
        self.assertGreater(dc_energy, ac_energy)


class TestDctWatermarkBit(unittest.TestCase):
    def test_embed_and_extract_bit_1(self):
        block = list(range(100, 100 + BLOCK_SIZE))
        modified = embed_dct_watermark(block, 1)
        extracted = extract_dct_watermark_bit(modified)
        self.assertEqual(extracted, 1)

    def test_embed_and_extract_bit_0(self):
        block = list(range(80, 80 + BLOCK_SIZE))
        modified = embed_dct_watermark(block, 0)
        extracted = extract_dct_watermark_bit(modified)
        self.assertEqual(extracted, 0)

    def test_output_clamped_to_valid_range(self):
        block = [255] * BLOCK_SIZE
        modified = embed_dct_watermark(block, 1)
        for pixel in modified:
            self.assertGreaterEqual(pixel, 0)
            self.assertLessEqual(pixel, 255)

    def test_invalid_bit_raises(self):
        block = list(range(BLOCK_SIZE))
        with self.assertRaises(ValueError):
            embed_dct_watermark(block, 2)


class TestDctMessageRoundtrip(unittest.TestCase):
    def _make_pixel_data(self, size=4096):
        return [(i * 37 + 128) % 256 for i in range(size)]

    def test_embed_and_extract_short_message(self):
        pixels = self._make_pixel_data()
        msg = "DCT_WATERMARK"
        watermarked = embed_dct_message(pixels, msg)
        recovered = extract_dct_message(watermarked)
        self.assertEqual(recovered, msg)

    def test_embed_and_extract_alphanumeric(self):
        pixels = self._make_pixel_data(8192)
        msg = "SecureWatermark2026"
        watermarked = embed_dct_message(pixels, msg)
        recovered = extract_dct_message(watermarked)
        self.assertEqual(recovered, msg)

    def test_embed_raises_when_buffer_too_small(self):
        pixels = list(range(16))  # Only 2 blocks — can't hold even 1 ASCII char
        with self.assertRaises(ValueError):
            embed_dct_message(pixels, "TOOLONGMESSAGE")

    def test_extract_stops_at_null_terminator(self):
        pixels = self._make_pixel_data()
        msg = "STOP"
        watermarked = embed_dct_message(pixels, msg)
        recovered = extract_dct_message(watermarked, max_chars=256)
        self.assertEqual(recovered, "STOP")
        self.assertFalse(recovered.endswith("\x00"))


if __name__ == '__main__':
    unittest.main()
