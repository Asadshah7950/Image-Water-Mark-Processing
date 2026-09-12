import math
from typing import List

class EntropyAnalyzer:
    @staticmethod
    def calculate_shannon_entropy(pixels: List[int]) -> float:
        if not pixels:
            return 0.0
        n = len(pixels)
        freq = {}
        for p in pixels:
            freq[p] = freq.get(p, 0) + 1
        
        entropy = 0.0
        for count in freq.values():
            prob = count / n
            entropy -= prob * math.log2(prob)
        return round(entropy, 4)

    @staticmethod
    def classify_region_texture(entropy: float, threshold: float = 4.5) -> str:
        return "TEXTURED" if entropy >= threshold else "SMOOTH"
