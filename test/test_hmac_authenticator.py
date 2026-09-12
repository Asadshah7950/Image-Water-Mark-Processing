import unittest
from scripts.hmac_authenticator import HMACAuthenticator

class TestHMACAuthenticator(unittest.TestCase):
    def test_compute_and_verify(self):
        key = b"super_secret_watermark_key_123"
        data = b"image_pixel_payload_data"
        tag = HMACAuthenticator.compute_tag(data, key)
        self.assertEqual(len(tag), 64)
        self.assertTrue(HMACAuthenticator.verify_tag(data, tag, key))

    def test_tamper_rejection(self):
        key = b"super_secret_watermark_key_123"
        data = b"image_pixel_payload_data"
        tag = HMACAuthenticator.compute_tag(data, key)
        self.assertFalse(HMACAuthenticator.verify_tag(b"tampered_data", tag, key))

if __name__ == '__main__':
    unittest.main()
