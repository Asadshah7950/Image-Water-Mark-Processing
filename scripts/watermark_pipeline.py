import math
import concurrent.futures
from typing import List, Tuple

def calculate_mse(orig: bytes, watermarked: bytes) -> float:
    if len(orig) != len(watermarked) or len(orig) == 0:
        return 0.0
    total_sq_err = sum((a - b) ** 2 for a, b in zip(orig, watermarked))
    return total_sq_err / len(orig)

def calculate_psnr(orig: bytes, watermarked: bytes) -> float:
    mse = calculate_mse(orig, watermarked)
    if mse == 0:
        return 100.0  # Perfect match
    max_pixel = 255.0
    return 20.0 * math.log10(max_pixel / math.sqrt(mse))

def embed_lsb(pixel_data: bytearray, message: str) -> bytearray:
    binary_msg = ''.join(f'{ord(c):08b}' for c in message) + '00000000'  # null terminator
    if len(binary_msg) > len(pixel_data):
        raise ValueError('Message exceeds payload capacity of image buffer')

    out = bytearray(pixel_data)
    for i, bit in enumerate(binary_msg):
        out[i] = (out[i] & ~1) | int(bit)
    return out

def extract_lsb(pixel_data: bytes, max_chars: int = 1024) -> str:
    bits = []
    for byte in pixel_data:
        bits.append(str(byte & 1))
        if len(bits) % 8 == 0:
            char_code = int(''.join(bits[-8:]), 2)
            if char_code == 0:
                break
            if len(bits) // 8 >= max_chars:
                break

    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte_bits = ''.join(bits[i:i+8])
        code = int(byte_bits, 2)
        if code == 0:
            break
        chars.append(chr(code))
    return ''.join(chars)

def process_single_image(idx: int, size: int, message: str) -> Tuple[int, float]:
    data = bytearray((i * 37 + idx) % 256 for i in range(size))
    watermarked = embed_lsb(data, message)
    psnr = calculate_psnr(bytes(data), bytes(watermarked))
    return idx, psnr

def process_batch_concurrent(image_count: int, img_size: int, message: str, max_workers: int = 4) -> List[Tuple[int, float]]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_image, i, img_size, message)
            for i in range(image_count)
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    return sorted(results, key=lambda x: x[0])
