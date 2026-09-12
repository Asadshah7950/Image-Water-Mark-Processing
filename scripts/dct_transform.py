import math
from typing import List

class DCTProcessor:
    @staticmethod
    def dct_1d(vector: List[float]) -> List[float]:
        n = len(vector)
        out = [0.0] * n
        factor = math.pi / (2.0 * n)
        for u in range(n):
            alpha = math.sqrt(1.0 / n) if u == 0 else math.sqrt(2.0 / n)
            acc = sum(vector[x] * math.cos((2 * x + 1) * u * factor) for x in range(n))
            out[u] = round(alpha * acc, 4)
        return out

    @staticmethod
    def idct_1d(coeffs: List[float]) -> List[float]:
        n = len(coeffs)
        out = [0.0] * n
        factor = math.pi / (2.0 * n)
        for x in range(n):
            acc = 0.0
            for u in range(n):
                alpha = math.sqrt(1.0 / n) if u == 0 else math.sqrt(2.0 / n)
                acc += alpha * coeffs[u] * math.cos((2 * x + 1) * u * factor)
            out[x] = round(acc, 4)
        return out
