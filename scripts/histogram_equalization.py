"""
Histogram equalization and CLAHE-inspired adaptive local contrast enhancement.

Used as a pre-processing step in the watermark pipeline to normalize image
luminance before embedding, improving watermark perceptual uniformity across
images with varying exposure (very dark or very bright images cause
non-uniform embedding capacity when using intensity-based selection).

References:
  - CLAHE: K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization",
    Graphics Gems IV, Academic Press, 1994.
"""
from typing import List, Tuple


def compute_histogram(pixel_data: List[int], bins: int = 256) -> List[int]:
    """
    Compute the pixel intensity histogram.

    Args:
        pixel_data: 1-D list of pixel values in [0, 255].
        bins: Number of histogram bins (default 256 for 8-bit grayscale).

    Returns:
        List of length `bins` with frequency counts.
    """
    hist = [0] * bins
    for p in pixel_data:
        if 0 <= p < bins:
            hist[p] += 1
    return hist


def compute_cdf(histogram: List[int]) -> List[float]:
    """
    Compute the normalized Cumulative Distribution Function (CDF) from a histogram.
    The CDF is used to build the histogram equalization mapping table.

    Returns:
        List of floats in [0.0, 1.0], one per bin.
    """
    total = sum(histogram)
    if total == 0:
        return [0.0] * len(histogram)

    cdf = []
    cumulative = 0
    for count in histogram:
        cumulative += count
        cdf.append(cumulative / total)
    return cdf


def equalize_histogram(pixel_data: List[int]) -> List[int]:
    """
    Apply global histogram equalization to a 1-D pixel array.

    Stretches contrast by remapping pixel intensities using the CDF:
        new_pixel = round(CDF(p) * 255)

    This improves watermark embedding capacity uniformity across dark/bright images
    by spreading intensity values more evenly across the [0, 255] range.

    Args:
        pixel_data: 1-D list of pixel values in [0, 255].

    Returns:
        Equalized pixel values in [0, 255].
    """
    hist = compute_histogram(pixel_data)
    cdf = compute_cdf(hist)
    return [round(cdf[p] * 255) for p in pixel_data]


def clip_histogram(histogram: List[int], clip_limit: float) -> Tuple[List[int], int]:
    """
    Clip a histogram at `clip_limit * average_count` and return the clipped histogram
    and the total number of redistributed pixels (used for CLAHE redistribution).

    Args:
        histogram: Raw frequency histogram.
        clip_limit: Multiplier for the average bin count (e.g. 2.0 = 2x average).

    Returns:
        (clipped_histogram, total_redistributed_pixels)
    """
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be positive, got {clip_limit}")

    n_pixels = sum(histogram)
    n_bins = len(histogram)
    avg = n_pixels / n_bins
    threshold = max(1, round(avg * clip_limit))

    clipped = []
    redistributed = 0
    for count in histogram:
        if count > threshold:
            redistributed += count - threshold
            clipped.append(threshold)
        else:
            clipped.append(count)

    # Redistribute excess uniformly
    per_bin = redistributed // n_bins
    remainder = redistributed % n_bins
    result = [c + per_bin for c in clipped]
    for i in range(remainder):
        result[i] += 1

    return result, redistributed


def adaptive_equalize(
    pixel_data: List[int],
    width: int,
    height: int,
    tile_size: int = 8,
    clip_limit: float = 2.0,
) -> List[int]:
    """
    Simplified CLAHE-inspired adaptive local histogram equalization.

    Divides the image into non-overlapping tiles of `tile_size x tile_size` pixels,
    applies clipped histogram equalization independently per tile, then stitches
    results back into a flat pixel array.

    Args:
        pixel_data: Flat 1-D pixel array (row-major, length = width * height).
        width: Image width in pixels.
        height: Image height in pixels.
        tile_size: Size of each local tile in pixels.
        clip_limit: CLAHE clip multiplier (higher = more contrast, less noise amplification).

    Returns:
        Processed pixel array with locally equalized contrast.
    """
    if len(pixel_data) != width * height:
        raise ValueError(
            f"pixel_data length {len(pixel_data)} != width*height ({width}*{height}={width*height})"
        )

    output = list(pixel_data)

    for row_start in range(0, height, tile_size):
        for col_start in range(0, width, tile_size):
            row_end = min(row_start + tile_size, height)
            col_end = min(col_start + tile_size, width)

            # Extract tile pixels
            tile_pixels = [
                pixel_data[r * width + c]
                for r in range(row_start, row_end)
                for c in range(col_start, col_end)
            ]

            # Clip and equalize the tile
            hist = compute_histogram(tile_pixels)
            clipped_hist, _ = clip_histogram(hist, clip_limit)
            cdf = compute_cdf(clipped_hist)

            # Write back equalized pixels
            idx = 0
            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    orig = tile_pixels[idx]
                    output[r * width + c] = round(cdf[orig] * 255)
                    idx += 1

    return output
