import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from watermark_pipeline import (
    calculate_psnr,
    calculate_mse,
    embed_lsb,
    extract_lsb,
    process_batch_concurrent
)

class TestWatermarkPipeline(unittest.TestCase):
    def test_psnr_identical_images(self):
        data = b'\x10\x20\x30\x40' * 100
        self.assertEqual(calculate_psnr(data, data), 100.0)

    def test_lsb_embed_and_extract(self):
        buffer_size = 1024
        raw = bytearray(range(256)) * 4
        secret = 'CONFIDENTIAL_DATA_2026'

        watermarked = embed_lsb(raw, secret)
        recovered = extract_lsb(bytes(watermarked))

        self.assertEqual(recovered, secret)

    def test_psnr_quality_threshold(self):
        buffer_size = 4096
        raw = bytearray((i % 256) for i in range(buffer_size))
        secret = 'AUTH_WATERMARK'

        watermarked = embed_lsb(raw, secret)
        psnr = calculate_psnr(bytes(raw), bytes(watermarked))

        # LSB modifications should maintain exceptionally high visual fidelity (> 50 dB)
        self.assertGreater(psnr, 50.0)

    def test_concurrent_batch_processing(self):
        results = process_batch_concurrent(image_count=8, img_size=2048, message='BATCH_TAG', max_workers=4)
        self.assertEqual(len(results), 8)
        for idx, psnr in results:
            self.assertGreater(psnr, 50.0)

if __name__ == '__main__':
    unittest.main()
