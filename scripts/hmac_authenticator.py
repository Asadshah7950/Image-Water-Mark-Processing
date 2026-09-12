import hmac
import hashlib

class HMACAuthenticator:
    @staticmethod
    def compute_tag(data_bytes: bytes, secret_key: bytes) -> str:
        if not secret_key:
            raise ValueError("Secret key cannot be empty")
        return hmac.new(secret_key, data_bytes, hashlib.sha256).hexdigest()

    @staticmethod
    def verify_tag(data_bytes: bytes, expected_tag: str, secret_key: bytes) -> bool:
        computed = HMACAuthenticator.compute_tag(data_bytes, secret_key)
        return hmac.compare_digest(computed, expected_tag)
