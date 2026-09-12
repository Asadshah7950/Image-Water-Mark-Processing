from typing import List

class RobustnessBenchmarker:
    @staticmethod
    def compute_bit_error_rate(original_bits: List[int], recovered_bits: List[int]) -> float:
        if len(original_bits) != len(recovered_bits) or not original_bits:
            raise ValueError("Bit arrays must be of equal non-zero length")
        mismatches = sum(1 for a, b in zip(original_bits, recovered_bits) if a != b)
        return round(mismatches / len(original_bits), 4)

    @staticmethod
    def simulate_salt_pepper_noise(pixels: List[int], noise_ratio: float = 0.05) -> List[int]:
        result = list(pixels)
        total_noisy = int(len(pixels) * noise_ratio)
        step = max(1, len(pixels) // max(1, total_noisy))
        for i in range(0, len(pixels), step):
            result[i] = 255 if (i % 2 == 0) else 0
        return result
