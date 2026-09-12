import unittest
from scripts.image_quality_metrics import ImageQualityMetrics

class TestImageQualityMetrics(unittest.TestCase):
    def test_infinite_psnr_for_identical_images(self):
        img = [100.0, 150.0, 200.0]
        self.assertEqual(ImageQualityMetrics.compute_mse(img, img), 0.0)
        self.assertEqual(ImageQualityMetrics.compute_psnr(img, img), float('inf'))

    def test_computed_psnr_value(self):
        img_a = [100.0, 150.0, 200.0]
        img_b = [101.0, 149.0, 200.0] # mse = (1 + 1 + 0)/3 = 0.6667
        psnr = ImageQualityMetrics.compute_psnr(img_a, img_b)
        self.assertGreater(psnr, 45.0)

if __name__ == '__main__':
    unittest.main()
