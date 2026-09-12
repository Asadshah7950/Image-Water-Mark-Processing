"""
Spatial Adaptive Median Filter for Image Noise Removal.
Removes salt-and-pepper noise while preserving sharp watermark boundaries.
"""
from typing import List

def get_neighborhood_window(
    pixels: List[int],
    width: int,
    height: int,
    center_r: int,
    center_c: int,
    window_radius: int
) -> List[int]:
    """Extracts pixel values in a rectangular spatial window around (center_r, center_c)."""
    window = []
    for r in range(max(0, center_r - window_radius), min(height, center_r + window_radius + 1)):
        for c in range(max(0, center_c - window_radius), min(width, center_c + window_radius + 1)):
            window.append(pixels[r * width + c])
    return window

def apply_median_filter(
    pixels: List[int],
    width: int,
    height: int,
    radius: int = 1
) -> List[int]:
    """Applies a standard median filter over a 2D image represented as a flat row-major array."""
    if len(pixels) != width * height:
        raise ValueError("Pixel count must equal width * height")
    output = list(pixels)
    for r in range(height):
        for c in range(width):
            window = get_neighborhood_window(pixels, width, height, r, c, radius)
            window.sort()
            median_val = window[len(window) // 2]
            output[r * width + c] = median_val
    return output
