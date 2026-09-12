"""
Local Shannon Entropy Region Analyzer.
Identifies high-entropy image regions suitable for perceptual watermark embedding.
"""
import math
from typing import List, Dict

def calculate_block_entropy(block: List[int], bins: int = 256) -> float:
    """Computes Shannon entropy in bits for a discrete 1-D block of pixels."""
    if not block:
        return 0.0
    counts = [0] * bins
    for p in block:
        if 0 <= p < bins:
            counts[p] += 1
    n = len(block)
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log2(p)
    return round(entropy, 4)

def select_optimal_embedding_blocks(
    pixels: List[int],
    block_size: int = 64,
    min_entropy: float = 3.0
) -> List[Dict]:
    """
    Scans image pixels in blocks and returns indices of blocks with sufficient texture complexity.
    """
    suitable_blocks = []
    num_blocks = len(pixels) // block_size
    for i in range(num_blocks):
        start = i * block_size
        chunk = pixels[start:start + block_size]
        ent = calculate_block_entropy(chunk)
        if ent >= min_entropy:
            suitable_blocks.append({
                'block_index': i,
                'start_offset': start,
                'entropy': ent
            })
    return suitable_blocks
