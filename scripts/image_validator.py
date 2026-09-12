import math
from typing import Dict, Any, Tuple

MAGIC_NUMBERS = {
    b'\x89PNG\r\n\x1a\n': 'PNG',
    b'\xff\xd8\xff': 'JPEG',
    b'BM': 'BMP',
    b'GIF87a': 'GIF',
    b'GIF89a': 'GIF',
}

def detect_image_format(header: bytes) -> str:
    """Detect image MIME/format type based on binary magic numbers."""
    if not header:
        return 'UNKNOWN'
    for magic, fmt in MAGIC_NUMBERS.items():
        if header.startswith(magic):
            return fmt
    if len(header) >= 12 and header[:4] == b'RIFF' and header[8:12] == b'WEBP':
        return 'WEBP'
    return 'UNKNOWN'

def calculate_shannon_entropy(data: bytes) -> float:
    """Calculate Shannon entropy (bits per byte) across a binary payload."""
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    total = len(data)
    entropy = 0.0
    for count in freq:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return round(entropy, 4)

def sanitize_watermark_payload(payload: str, max_chars: int = 512) -> str:
    """Sanitize watermark payload to printable ASCII and truncate to maximum allowed characters."""
    if not payload:
        return ''
    cleaned = ''.join(c for c in payload if 32 <= ord(c) <= 126)
    return cleaned[:max_chars]

def validate_image_buffer(buffer: bytes, min_size: int = 32) -> Tuple[bool, str]:
    """Validate buffer meets minimum payload size, format integrity, and non-trivial entropy."""
    if len(buffer) < min_size:
        return False, f"Buffer size ({len(buffer)}B) below minimum threshold ({min_size}B)"
    fmt = detect_image_format(buffer[:16])
    if fmt == 'UNKNOWN':
        entropy = calculate_shannon_entropy(buffer)
        if entropy < 0.1:
            return False, "Buffer exhibits zero or degenerate entropy (uniform flat bytes)"
        return True, "RAW_PIXELS"
    return True, fmt
