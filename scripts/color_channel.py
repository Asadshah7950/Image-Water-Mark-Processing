"""
Multi-Channel RGB Spatial Watermarking.
Embeds watermark payloads into weighted luminance-optimal color channels (R, G, or B).
"""
from typing import List, Dict

def decompose_channels(rgb_pixels: List[int]) -> Dict[str, List[int]]:
    """Splits interleaved RGB pixel array [R, G, B, R, G, B...] into separate channels."""
    if len(rgb_pixels) % 3 != 0:
        raise ValueError("rgb_pixels length must be a multiple of 3")
    return {
        'R': rgb_pixels[0::3],
        'G': rgb_pixels[1::3],
        'B': rgb_pixels[2::3],
    }

def interleave_channels(r: List[int], g: List[int], b: List[int]) -> List[int]:
    """Combines separated R, G, B channel lists back into an interleaved RGB array."""
    if not (len(r) == len(g) == len(b)):
        raise ValueError("Channels must have equal length")
    out = []
    for red, green, blue in zip(r, g, b):
        out.extend([red, green, blue])
    return out

def compute_luminance(r: int, g: int, b: int) -> float:
    """Computes ITU-R BT.601 perceptual luminance value."""
    return 0.299 * r + 0.587 * g + 0.114 * b
