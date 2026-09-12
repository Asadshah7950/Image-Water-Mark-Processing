import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from image_validator import (
    detect_image_format,
    calculate_shannon_entropy,
    sanitize_watermark_payload,
    validate_image_buffer,
)

class TestImageValidator(unittest.TestCase):
    def test_detect_image_formats(self):
        png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        jpeg_header = b'\xff\xd8\xff\xe0\x00\x10JFIF'
        bmp_header = b'BM\x36\x00\x0c\x00'
        webp_header = b'RIFF\x24\x00\x00\x00WEBPVP8 '

        self.assertEqual(detect_image_format(png_header), 'PNG')
        self.assertEqual(detect_image_format(jpeg_header), 'JPEG')
        self.assertEqual(detect_image_format(bmp_header), 'BMP')
        self.assertEqual(detect_image_format(webp_header), 'WEBP')
        self.assertEqual(detect_image_format(b'\x00\x01\x02\x03'), 'UNKNOWN')

    def test_shannon_entropy(self):
        uniform = b'\xAA' * 1000
        self.assertEqual(calculate_shannon_entropy(uniform), 0.0)

        import random
        random.seed(42)
        random_bytes = bytes(random.randint(0, 255) for _ in range(10000))
        entropy = calculate_shannon_entropy(random_bytes)
        # Random bytes have entropy very close to 8.0 bits/byte
        self.assertGreater(entropy, 7.9)

    def test_sanitize_watermark_payload(self):
        dirty = "WATERMARK\x00\x01\x1b[31m_SECURE_2026\r\n"
        cleaned = sanitize_watermark_payload(dirty, max_chars=20)
        self.assertEqual(cleaned, "WATERMARK[31m_SECURE")

        truncated = sanitize_watermark_payload("A" * 1000, max_chars=50)
        self.assertEqual(len(truncated), 50)

    def test_validate_image_buffer(self):
        tiny = b'\x01\x02'
        valid, msg = validate_image_buffer(tiny, min_size=32)
        self.assertFalse(valid)
        self.assertIn("below minimum threshold", msg)

        flat = b'\x00' * 64
        valid, msg = validate_image_buffer(flat)
        self.assertFalse(valid)
        self.assertIn("degenerate entropy", msg)

        png_mock = b'\x89PNG\r\n\x1a\n' + bytes(range(40))
        valid, fmt = validate_image_buffer(png_mock)
        self.assertTrue(valid)
        self.assertEqual(fmt, 'PNG')

if __name__ == '__main__':
    unittest.main()
