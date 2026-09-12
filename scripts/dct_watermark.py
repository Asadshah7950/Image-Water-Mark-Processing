import math
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1D DCT-II and inverse (IDCT-II) — pure Python, no external dependencies.
#
# The 8-point DCT is the standard used in JPEG baseline (ITU-T T.81).
# We use it here to illustrate frequency-domain steganographic embedding:
#   - Watermark bits are encoded into mid-frequency DCT coefficients (AC[1]..AC[5]).
#   - These bands survive mild low-pass filtering but are perceptually invisible.
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_SIZE = 8

def dct8(block: List[float]) -> List[float]:
    """8-point DCT-II: maps spatial block → frequency coefficients."""
    n = BLOCK_SIZE
    result = []
    for k in range(n):
        ck = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
        total = sum(block[i] * math.cos(math.pi * k * (2 * i + 1) / (2 * n)) for i in range(n))
        result.append(ck * total)
    return result

def idct8(coeffs: List[float]) -> List[float]:
    """8-point IDCT-II (inverse): maps frequency coefficients → spatial block."""
    n = BLOCK_SIZE
    result = []
    for i in range(n):
        total = 0.0
        for k in range(n):
            ck = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
            total += ck * coeffs[k] * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
        result.append(total)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Watermark embedding / extraction
# ─────────────────────────────────────────────────────────────────────────────

# Delta: the coefficient perturbation magnitude.
# Larger = more robust, smaller = less visible. 8.0 is a common baseline.
EMBED_DELTA = 8.0

# Mid-frequency AC indices to use (avoid DC=0 and high-freq 6,7)
MID_FREQ_INDICES = [2, 3, 4]

def embed_dct_watermark(pixel_block: List[int], bit: int) -> List[int]:
    """
    Embed one watermark bit into an 8-pixel block using frequency-domain quantization.

    Strategy: quantise the selected AC coefficient to an even/odd multiple of EMBED_DELTA.
      - bit == 1 → round coefficient to nearest odd  multiple of EMBED_DELTA
      - bit == 0 → round coefficient to nearest even multiple of EMBED_DELTA

    Returns the modified spatial block as integers clamped to [0, 255].
    """
    if bit not in (0, 1):
        raise ValueError(f"Watermark bit must be 0 or 1, got {bit}")

    block_float = [float(p) for p in pixel_block]
    coeffs = dct8(block_float)

    target_idx = MID_FREQ_INDICES[0]
    coeff = coeffs[target_idx]
    quantized = round(coeff / EMBED_DELTA)
    if bit == 1:
        if quantized % 2 == 0:
            quantized += 1
    else:
        if quantized % 2 != 0:
            quantized += 1
    coeffs[target_idx] = quantized * EMBED_DELTA

    spatial = idct8(coeffs)
    return [max(0, min(255, round(v))) for v in spatial]

def extract_dct_watermark_bit(pixel_block: List[int]) -> int:
    """
    Extract one watermark bit from an 8-pixel block.
    Returns 1 if the mid-frequency coefficient is an odd multiple of EMBED_DELTA, else 0.
    """
    block_float = [float(p) for p in pixel_block]
    coeffs = dct8(block_float)
    target_idx = MID_FREQ_INDICES[0]
    quantized = round(coeffs[target_idx] / EMBED_DELTA)
    return quantized % 2

def embed_dct_message(pixel_data: List[int], message: str) -> List[int]:
    """
    Embed a UTF-8 string message into a 1-D pixel array using DCT frequency-domain watermarking.

    Encodes each character as 8 bits; each bit is embedded into one BLOCK_SIZE-pixel segment.
    A null byte (8 zero-bits) is appended as a terminator.

    Raises ValueError if the pixel array is too small to hold the message.
    """
    binary_msg = ''.join(f'{ord(c):08b}' for c in message) + '00000000'  # null terminator
    required_pixels = len(binary_msg) * BLOCK_SIZE
    if required_pixels > len(pixel_data):
        raise ValueError(
            f"Message requires {required_pixels} pixels, but only {len(pixel_data)} available."
        )

    output = list(pixel_data)
    for i, bit in enumerate(binary_msg):
        start = i * BLOCK_SIZE
        block = output[start:start + BLOCK_SIZE]
        modified = embed_dct_watermark(block, int(bit))
        output[start:start + BLOCK_SIZE] = modified
    return output

def extract_dct_message(pixel_data: List[int], max_chars: int = 256) -> str:
    """
    Extract a UTF-8 string message from a DCT-watermarked 1-D pixel array.
    Stops at the null-byte terminator or after max_chars characters.
    """
    bits = []
    for block_idx in range(min(len(pixel_data) // BLOCK_SIZE, max_chars * 8 + 8)):
        start = block_idx * BLOCK_SIZE
        block = pixel_data[start:start + BLOCK_SIZE]
        bits.append(extract_dct_watermark_bit(block))

    chars = []
    for byte_idx in range(len(bits) // 8):
        byte = bits[byte_idx * 8: byte_idx * 8 + 8]
        code = int(''.join(str(b) for b in byte), 2)
        if code == 0:
            break
        chars.append(chr(code))
        if len(chars) >= max_chars:
            break

    return ''.join(chars)
