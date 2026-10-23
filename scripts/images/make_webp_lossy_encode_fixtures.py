"""Fixtures for the lossy WebP ENCODER (byte-identical with libwebp).

For each case it writes into SpawnDev.Codecs.Demo.Shared/TestData/WebPLossyEnc:
  <src>.yuv.gz      the YUV 4:2:0 planes LIBWEBP imports for the (opaque RGB) source picture (WebPPictureImportRGBA, the same
                    conversion WebPEncode runs), Y then U then V, tightly packed; one per source picture
  <name>.webp       Pillow's output (libwebp WebPEncode with WebPConfigInit defaults + quality + method), the expected bytes
  manifest.tsv      name, source, width, height, quality, method
Truth never comes from our code.

Usage: python -I make_webp_lossy_encode_fixtures.py   (venv with Pillow, numpy, webp)
"""
import gzip
import io
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import libwebp_picture  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "WebPLossyEnc"))
os.makedirs(OUT, exist_ok=True)


def photo(w, h, seed):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.zeros((h, w, 3), np.float64)
    a[..., 0] = 128 + 100 * np.sin(xx / 9.0 + seed) * np.cos(yy / 13.0)
    a[..., 1] = 128 + 80 * np.sin(yy / 7.0 - seed) + 30 * np.sin(xx / 4.0)
    a[..., 2] = 255 * ((xx / max(w, 1) + yy / max(h, 1)) / 2)
    for _ in range(6):
        cx, cy = r.integers(0, max(w, 1)), r.integers(0, max(h, 1))
        rad = r.integers(2, max(3, min(w, h) // 3 + 2))
        m = (xx - cx) ** 2 + (yy - cy) ** 2 < rad * rad
        a[m] = r.integers(0, 256, 3)
    a += r.normal(0, 7, a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


def graphic(w, h, seed):
    """Flat areas, hard edges, thin lines: I16 / I4 / skip decisions and the flat-source special cases."""
    r = np.random.default_rng(seed)
    a = np.full((h, w, 3), 240, np.uint8)
    for _ in range(8):
        x0, y0 = r.integers(0, w), r.integers(0, h)
        x1, y1 = min(w, x0 + r.integers(1, max(2, w // 2))), min(h, y0 + r.integers(1, max(2, h // 2)))
        a[y0:y1, x0:x1] = r.integers(0, 256, 3)
    a[::7, :] = (20, 20, 20)
    a[:, ::11] = (200, 30, 30)
    return a


def noise(w, h, seed):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


SOURCES = []
for (w, h) in [(1, 1), (16, 16), (17, 9), (33, 47), (64, 64), (100, 75), (255, 129)]:
    SOURCES.append((f"photo_{w}x{h}", w, h, photo(w, h, w * 3 + h)))
for (w, h) in [(48, 48), (130, 66)]:
    SOURCES.append((f"graphic_{w}x{h}", w, h, graphic(w, h, w + h)))
SOURCES.append(("noise_40x24", 40, 24, noise(40, 24, 7)))
SOURCES.append(("flat_32x32", 32, 32, np.full((32, 32, 3), 77, np.uint8)))
SOURCES.append(("photo_640x480", 640, 480, photo(640, 480, 99)))

QUALITIES = [0, 5, 25, 50, 75, 80, 90, 98, 99, 100]
rows = []
for (src, w, h, rgb) in SOURCES:
    rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)])
    Y, U, V, A = libwebp_picture.import_rgba(rgba)
    with open(os.path.join(OUT, src + ".yuv.gz"), "wb") as f:
        f.write(gzip.compress(Y.tobytes() + U.tobytes() + V.tobytes(), 6, mtime=0))
    big = w * h > 100000
    for q in ([75, 90] if big else QUALITIES):
        for m in ([4] if big else [0, 1, 2, 3, 4]):
            name = f"{src}_q{q}_m{m}"
            buf = io.BytesIO()
            Image.fromarray(rgb, "RGB").save(buf, "WEBP", lossless=False, quality=q, method=m)
            data = buf.getvalue()
            assert data[12:16] == b"VP8 ", (name, data[12:16])   # simple format for an opaque picture
            with open(os.path.join(OUT, name + ".webp"), "wb") as f:
                f.write(data)
            rows.append((name, src, w, h, q, m))

with open(os.path.join(OUT, "manifest.tsv"), "w", newline="\n") as f:
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print(len(rows), "encoded fixtures from", len(SOURCES), "sources")
