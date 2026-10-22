"""Fixtures for the libwebp-exact YUV 4:2:0 -> RGBA conversion (Images/Yuv/WebPYuvToRgbaGpu).

For every (size, quality) it writes into SpawnDev.Codecs.Demo.Shared/TestData/WebPYuv:
  <name>.webp   a lossy WebP made by Pillow/libwebp
  <name>.yuv    the decoded planes (Y, then U, then V; chroma (w+1)/2 x (h+1)/2) from FFMPEG's native VP8 decoder
  <name>.rgba8  the RGBA8 picture libwebp itself produces (Pillow decode: default config = fancy upsampling)

Truth is therefore independent of our code on both sides: ffmpeg gives the YUV, libwebp gives the RGB, and the
converter under test must map one to the other bit-exactly.

Usage: python -I make_webp_yuv_fixtures.py
Needs: Pillow with WebP, numpy, ffmpeg (path below).
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

FFMPEG = r"C:\Users\TJ\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "WebPYuv"))
os.makedirs(OUT, exist_ok=True)


def picture(w, h, seed):
    """Smooth gradients + colour blobs + a little noise: chroma varies, so upsampling errors show."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.zeros((h, w, 3), np.float64)
    a[..., 0] = 128 + 100 * np.sin(xx / 7.0 + seed) * np.cos(yy / 11.0)
    a[..., 1] = 128 + 90 * np.sin(yy / 5.0 - seed) + 20 * np.sin(xx / 3.0)
    a[..., 2] = 255 * ((xx / max(w, 1) + yy / max(h, 1)) / 2)
    for _ in range(4):
        cx, cy = r.integers(0, max(w, 1)), r.integers(0, max(h, 1))
        rad = r.integers(2, max(3, min(w, h) // 2 + 2))
        col = r.integers(0, 256, 3)
        m = (xx - cx) ** 2 + (yy - cy) ** 2 < rad * rad
        a[m] = col
    a += r.normal(0, 6, a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


CASES = []
for (w, h) in [(1, 1), (2, 2), (3, 3), (4, 1), (1, 4), (16, 16), (17, 33), (31, 47), (64, 64), (100, 75), (255, 129), (320, 200)]:
    for q in (30, 90):
        CASES.append((w, h, q))

n = 0
for (w, h, q) in CASES:
    name = f"yuv_{w}x{h}_q{q}"
    img = Image.fromarray(picture(w, h, w * 31 + h + q), "RGB")
    path = os.path.join(OUT, name + ".webp")
    img.save(path, "WEBP", lossless=False, quality=q, method=4)
    with Image.open(path) as d:
        d.load()
        rgba = np.asarray(d.convert("RGBA"), dtype=np.uint8)
    assert rgba.shape == (h, w, 4), (name, rgba.shape)
    open(os.path.join(OUT, name + ".rgba8"), "wb").write(rgba.tobytes())
    yuv_path = os.path.join(OUT, name + ".yuv")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", yuv_path], check=True)
    cw, ch = (w + 1) // 2, (h + 1) // 2
    expect = w * h + 2 * cw * ch
    got = os.path.getsize(yuv_path)
    if got != expect:
        sys.exit(f"{name}: ffmpeg wrote {got} bytes, expected {expect}")
    n += 1
print(f"wrote {n} fixtures to {OUT}")
