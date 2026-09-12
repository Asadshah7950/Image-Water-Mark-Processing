import math
from typing import List

class ImageQualityMetrics:
    @staticmethod
    def compute_mse(img_a: List[float], img_b: List[float]) -> float:
        if len(img_a) != len(img_b) or not img_a:
            raise ValueError("Arrays must be non-empty and of equal length")
        squared_err = sum((a - b) ** 2 for a, b in zip(img_a, img_b))
        return squared_err / len(img_a)

    @staticmethod
    def compute_psnr(img_a: List[float], img_b: List[float], max_val: float = 255.0) -> float:
        mse = ImageQualityMetrics.compute_mse(img_a, img_b)
        if mse == 0:
            return float('inf')
        return round(20 * math.log10(max_val) - 10 * math.log10(mse), 2)
