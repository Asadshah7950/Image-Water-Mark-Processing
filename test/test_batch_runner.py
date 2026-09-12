import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from batch_runner import BatchPipelineRunner
from watermark_pipeline import embed_lsb

class TestBatchPipelineRunner(unittest.TestCase):
    def test_processes_batch_successfully(self):
        runner = BatchPipelineRunner(max_workers=2)
        items = [
            {'id': f'img_{i}', 'pixels': [(j * 17) % 256 for j in range(256)]}
            for i in range(8)
        ]
        summary = runner.process_batch(items, embed_lsb, message='OK')
        self.assertEqual(summary['total_items'], 8)
        self.assertEqual(summary['succeeded'], 8)
        self.assertEqual(summary['failed'], 0)
        self.assertGreater(summary['throughput_items_per_sec'], 0.0)

    def test_handles_failing_items_gracefully(self):
        runner = BatchPipelineRunner(max_workers=2)
        def faulty_embed(pixels, msg):
            if len(pixels) < 10:
                raise ValueError("Too small")
            return embed_lsb(pixels, msg)

        items = [
            {'id': 'good', 'pixels': [100] * 256},
            {'id': 'bad', 'pixels': [100] * 4},
        ]
        summary = runner.process_batch(items, faulty_embed, message='A')
        self.assertEqual(summary['succeeded'], 1)
        self.assertEqual(summary['failed'], 1)

if __name__ == '__main__':
    unittest.main()
