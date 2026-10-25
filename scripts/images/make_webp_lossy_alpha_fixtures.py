"""Fixtures for the lossy WebP ENCODER with transparency (VP8X + ALPH + VP8).

For each case it writes into SpawnDev.Codecs.Demo.Shared/TestData/WebPLossyAlpha:
  <src>.rgba.gz     the RGBA8 source picture (straight alpha, rows top to bottom); one per source picture
  <name>.webp       Pillow's output (libwebp WebPEncode with WebPConfigInit defaults + quality + method: alpha_compression 1,
                    alpha_filtering 1, alpha_quality 100, exact 0), the reference file
  manifest.tsv      name, source, width, height, quality, method
Truth never comes from our code. What the test holds our file to:
  - the VP8 chunk is BYTE-IDENTICAL to libwebp's (the colour part does not depend on how the alpha is coded);
  - the ALPH chunk decodes to exactly the source alpha (libwebp's own ALPH is lossless at alpha_quality 100 - asserted below);
  - the container is RIFF + VP8X (alpha flag, canvas size) + ALPH + VP8, chunk sizes and padding as the spec says.
The ALPH bytes themselves are not compared: our VP8L coder is not libwebp's.

Usage: python -I make_webp_lossy_alpha_fixtures.py   (venv with Pillow, numpy)
"""
import gzip
import io
import os
import struct

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "WebPLossyAlpha"))
os.makedirs(OUT, exist_ok=True)


def photo(w, h, seed):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.zeros((h, w, 3), np.float64)
    a[..., 0] = 128 + 100 * np.sin(xx / 9.0 + seed) * np.cos(yy / 13.0)
    a[..., 1] = 128 + 80 * np.sin(yy / 7.0 - seed) + 30 * np.sin(xx / 4.0)
    a[..., 2] = 255 * ((xx / max(w, 1) + yy / max(h, 1)) / 2)
    a += r.normal(0, 7, a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


def alpha_gradient(w, h, seed):
    """Semi-transparent everywhere: a diagonal ramp plus noise (filters, no fully transparent block)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = 255 * (xx / max(w - 1, 1) * 0.6 + yy / max(h - 1, 1) * 0.4)
    a += np.random.default_rng(seed).normal(0, 3, a.shape)
    return np.clip(a, 1, 254).astype(np.uint8)


def alpha_mask(w, h, seed):
    """A cut-out: opaque disc and bars, fully transparent elsewhere (libwebp clears the RGB under alpha 0 per block)."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.zeros((h, w), np.uint8)
    a[(xx - w / 2) ** 2 + (yy - h / 2) ** 2 < (min(w, h) / 2.5) ** 2] = 255
    a[:, ::9] = 255
    for _ in range(4):
        x0, y0 = r.integers(0, w), r.integers(0, h)
        a[y0:y0 + max(1, h // 6), x0:x0 + max(1, w // 5)] = 0
    return a


def alpha_sparse(w, h, seed):
    """Opaque with scattered transparent and half-transparent pixels."""
    r = np.random.default_rng(seed)
    a = np.full((h, w), 255, np.uint8)
    k = max(1, (w * h) // 23)
    a[r.integers(0, h, k), r.integers(0, w, k)] = 0
    a[r.integers(0, h, k), r.integers(0, w, k)] = 128
    return a


SOURCES = []
for (w, h) in [(1, 1), (17, 9), (33, 47), (64, 64), (100, 75), (255, 129)]:
    for kind, fn in [("grad", alpha_gradient), ("mask", alpha_mask), ("sparse", alpha_sparse)]:
        a = fn(w, h, w * 7 + h)
        if a.min() == 255:  # 1x1 sparse can come out opaque: force transparency
            a[0, 0] = 0
        SOURCES.append((f"{kind}_{w}x{h}", w, h, np.dstack([photo(w, h, w + 3 * h), a])))
SOURCES.append(("mask_640x480", 640, 480, np.dstack([photo(640, 480, 5), alpha_mask(640, 480, 5)])))


def chunks(data):
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    assert struct.unpack("<I", data[4:8])[0] == len(data) - 8
    out, p = [], 12
    while p < len(data):
        tag, size = data[p:p + 4], struct.unpack("<I", data[p + 4:p + 8])[0]
        out.append((tag, data[p + 8:p + 8 + size]))
        p += 8 + size + (size & 1)
    assert p == len(data)
    return out


QUALITIES = [0, 50, 75, 100]
rows = []
for (src, w, h, rgba) in SOURCES:
    with open(os.path.join(OUT, src + ".rgba.gz"), "wb") as f:
        f.write(gzip.compress(rgba.tobytes(), 6, mtime=0))
    big = w * h > 100000
    for q in ([75] if big else QUALITIES):
        for m in ([4] if big else [3, 4]):
            name = f"{src}_q{q}_m{m}"
            buf = io.BytesIO()
            Image.fromarray(rgba, "RGBA").save(buf, "WEBP", lossless=False, quality=q, method=m)
            data = buf.getvalue()
            tags = [t for t, _ in chunks(data)]
            assert tags == [b"VP8X", b"ALPH", b"VP8 "], (name, tags)
            back = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
            assert np.array_equal(back[..., 3], rgba[..., 3]), (name, "libwebp's ALPH is not lossless here")
            with open(os.path.join(OUT, name + ".webp"), "wb") as f:
                f.write(data)
            rows.append((name, src, w, h, q, m))

with open(os.path.join(OUT, "manifest.tsv"), "w", newline="\n") as f:
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print(len(rows), "encoded fixtures from", len(SOURCES), "sources")
