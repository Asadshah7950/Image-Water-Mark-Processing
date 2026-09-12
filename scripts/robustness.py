"""
Robustness analysis utilities for digital watermarking pipelines.

Computes Bit Error Rate (BER) after common image degradation operations
(Gaussian noise, JPEG-like quantization, brightness shifts) to evaluate
how well the embedded watermark survives real-world transformations.
"""
import math
import random
from typing import List, Tuple


def add_gaussian_noise(pixel_data: List[int], std_dev: float, seed: int = 42) -> List[int]:
    """Add Gaussian noise with given standard deviation to pixel data."""
    rng = random.Random(seed)
    return [max(0, min(255, round(p + rng.gauss(0, std_dev)))) for p in pixel_data]


def apply_brightness_shift(pixel_data: List[int], delta: int) -> List[int]:
    """Shift all pixel values by a constant brightness offset (clamped to [0,255])."""
    return [max(0, min(255, p + delta)) for p in pixel_data]


def jpeg_quantize_block(pixel_data: List[int], quality: int = 75) -> List[int]:
    """
    Simulate JPEG quantization by rounding pixel values to a step size derived from quality.
    Quality 100 = no loss (step=1), Quality 1 = maximum loss (step~25).
    """
    if not (1 <= quality <= 100):
        raise ValueError(f"Quality must be in [1, 100], got {quality}")
    step = max(1, round((100 - quality) / 4))
    return [round(p / step) * step for p in pixel_data]


def compute_ber(
    original_bits: List[int],
    recovered_bits: List[int],
) -> float:
    """
    Compute Bit Error Rate (BER) between original and recovered binary watermark sequences.
    BER = (number of flipped bits) / (total bits).
    Returns 0.0 if lists are empty or identical.
    """
    if len(original_bits) != len(recovered_bits):
        raise ValueError("Bit sequences must have identical length")
    if not original_bits:
        return 0.0
    errors = sum(a != b for a, b in zip(original_bits, recovered_bits))
    return errors / len(original_bits)


def robustness_report(
    original_pixels: List[int],
    extract_fn,
    embed_fn,
    message: str,
    degradations: List[Tuple[str, callable]],
) -> List[dict]:
    """
    Run a watermark robustness evaluation across multiple degradation scenarios.

    Args:
        original_pixels: Raw pixel data before watermarking.
        extract_fn: Callable(pixels) -> str to extract the watermark message.
        embed_fn: Callable(pixels, message) -> pixels to embed the watermark.
        message: The string to embed as a watermark.
        degradations: List of (label, transform_fn) where transform_fn: pixels -> pixels.

    Returns:
        List of dicts with {scenario, recovered, ber, intact}.
    """
    # Embed the watermark into clean pixels
    watermarked = embed_fn(original_pixels, message)

    results = []
    original_bits = [int(b) for char in message for b in f'{ord(char):08b}']

    for label, transform in degradations:
        degraded = transform(watermarked)
        try:
            recovered_msg = extract_fn(degraded)
            recovered_bits = [int(b) for char in recovered_msg for b in f'{ord(char):08b}']
            # Pad or truncate to match lengths
            n = min(len(original_bits), len(recovered_bits))
            ber = compute_ber(original_bits[:n], recovered_bits[:n]) if n > 0 else 1.0
        except Exception:
            recovered_msg = ''
            ber = 1.0

        results.append({
            'scenario': label,
            'recovered': recovered_msg,
            'ber': round(ber, 4),
            'intact': recovered_msg == message,
        })

    return results
