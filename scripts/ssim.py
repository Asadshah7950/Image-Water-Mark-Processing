"""
Structural Similarity Index Measure (SSIM) — pure Python implementation.
Evaluates perceptual visual degradation of watermark embedding.
"""
import math
from typing import List, Tuple

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def variance_and_covariance(img1: List[float], img2: List[float], m1: float, m2: float) -> Tuple[float, float, float]:
    n = len(img1)
    if n <= 1:
        return 0.0, 0.0, 0.0
    var1 = sum((x - m1) ** 2 for x in img1) / (n - 1)
    var2 = sum((y - m2) ** 2 for y in img2) / (n - 1)
    covar = sum((x - m1) * (y - m2) for x, y in zip(img1, img2)) / (n - 1)
    return var1, var2, covar

def ssim_block(block1: List[int], block2: List[int], k1: float = 0.01, k2: float = 0.03, l_max: int = 255) -> float:
    """Computes SSIM between two identical-size 1-D pixel blocks."""
    if len(block1) != len(block2):
        raise ValueError("Blocks must be equal length")
    b1 = [float(p) for p in block1]
    b2 = [float(p) for p in block2]

    c1 = (k1 * l_max) ** 2
    c2 = (k2 * l_max) ** 2

    m1 = mean(b1)
    m2 = mean(b2)
    var1, var2, covar = variance_and_covariance(b1, b2, m1, m2)

    numerator = (2 * m1 * m2 + c1) * (2 * covar + c2)
    denominator = (m1 ** 2 + m2 ** 2 + c1) * (var1 + var2 + c2)

    if denominator == 0:
        return 1.0
    return numerator / denominator

def ssim_global(img1: List[int], img2: List[int], block_size: int = 64) -> float:
    """Computes Mean SSIM across segmented blocks."""
    if len(img1) != len(img2):
        raise ValueError("Images must have identical pixel count")
    scores = []
    for i in range(0, len(img1), block_size):
        chunk1 = img1[i:i + block_size]
        chunk2 = img2[i:i + block_size]
        if len(chunk1) == block_size:
            scores.append(ssim_block(chunk1, chunk2))
    return mean(scores) if scores else 1.0
