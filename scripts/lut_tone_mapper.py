from typing import List

class LUTToneMapper:
    @staticmethod
    def build_gamma_lut(gamma: float) -> List[int]:
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        lut = [0] * 256
        for i in range(256):
            normalized = i / 255.0
            corrected = normalized ** (1.0 / gamma)
            lut[i] = int(round(corrected * 255.0))
        return lut

    @staticmethod
    def apply_lut(pixels: List[int], lut: List[int]) -> List[int]:
        return [lut[max(0, min(255, p))] for p in pixels]
