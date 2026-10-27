"""Photo checks that don't need a model: quality (for the retake loop) and a
perceptual hash (for duplicate detection).

Quality thresholds are starting points chosen on synthetic images; Phase 5
calibrates them on real photos.
"""

import io

import numpy as np
from PIL import Image, ImageOps

# Decompression-bomb limit, as in vision. Pillow only raises above 2x this, so
# open_image checks the header's size itself before decoding.
MAX_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

MIN_SIDE = 480  # px; smaller and dents are a few pixels wide
MIN_SHARPNESS = 50.0  # variance of the Laplacian on a 512 px copy
MIN_BRIGHTNESS, MAX_BRIGHTNESS = 40, 225  # mean grey level, 0-255
ANALYSIS_SIDE = 512


def open_image(data):
    image = Image.open(io.BytesIO(data))  # reads the header only
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Image has too many pixels")
    image.load()
    return ImageOps.exif_transpose(image).convert("RGB")


def sharpness(gray):
    """Variance of the Laplacian: edges give large second derivatives, blur flattens them."""
    g = gray.astype(np.float64)
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def assess(image):
    """Return {"ok", "problems", "width", "height", "sharpness", "brightness"}."""
    width, height = image.size
    small = image.copy()
    small.thumbnail((ANALYSIS_SIDE, ANALYSIS_SIDE))
    gray = np.asarray(small.convert("L"))
    sharp = sharpness(gray)
    bright = float(gray.mean())
    problems = []
    if min(width, height) < MIN_SIDE:
        problems.append(
            f"Photo is too small ({width}x{height}); take it closer or at higher resolution."
        )
    if sharp < MIN_SHARPNESS:
        problems.append("Photo is blurry; hold the phone steady and tap to focus.")
    if bright < MIN_BRIGHTNESS:
        problems.append("Photo is too dark; take it in daylight or with more light.")
    elif bright > MAX_BRIGHTNESS:
        problems.append("Photo is over-exposed; avoid direct sun or flash glare.")
    return {
        "ok": not problems,
        "problems": problems,
        "width": width,
        "height": height,
        "sharpness": round(sharp, 1),
        "brightness": round(bright, 1),
    }


def _dct_matrix(n):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    d = np.sqrt(2 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    d[0] /= np.sqrt(2)
    return d


_DCT32 = _dct_matrix(32)


def phash(image):
    """64-bit perceptual hash (the classic pHash, as in the `imagehash` library):
    shrink to 32x32 grey, take the 2-D DCT, keep the 8x8 lowest frequencies, and
    set each bit by whether that coefficient is above their median. Low
    frequencies describe the overall layout, which survives re-saving, resizing
    and mild edits; that's why near-identical photos get near-identical hashes."""
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    low = (_DCT32 @ pixels @ _DCT32.T)[:8, :8]
    bits = (low > np.median(low)).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a, b):
    return (int(a, 16) ^ int(b, 16)).bit_count()
