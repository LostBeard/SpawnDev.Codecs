"""Fixtures for the lossy WebP DECODER (Images/WebP/WebPDecoderGpu, VP8 + ALPH path).

Truth is what LIBWEBP shows: Pillow's decode (default config = fancy upsampling, no dithering) of each file, as RGBA.
Nothing here comes from our code.

Writes into SpawnDev.Codecs.Demo.Shared/TestData/WebPLossy:
  <name>.webp                   new files (stills with alpha, animations incl. mixed lossy/lossless frames, big pictures)
  <name>.rgba8.gz / .f<k>.rgba8.gz   the RGBA canvas per frame (only for pictures of <= 70000 pixels)
  manifest.tsv                  name <TAB> folder (WebP|WebPLossy) <TAB> frames <TAB> width <TAB> height <TAB> sha256 of frame 0 .. (comma list per frame) <TAB> truthFile(1/0)
It also covers every lossy file already in TestData/WebP (the crafted-ALPH family, lossy_32x32, anim_lossy_32x32, the
metadata files) and the yuv_* family of TestData/WebPYuv, so one manifest drives the decoder test.

Usage: python -I make_webp_lossy_decode_fixtures.py
"""
import gzip
import hashlib
import io
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData"))
OUT = os.path.join(DATA, "WebPLossy")
os.makedirs(OUT, exist_ok=True)
FULL_TRUTH_MAX_PIXELS = 70000


def picture(w, h, seed, alpha_kind):
    """Gradients + blobs + noise (so chroma and detail vary) with an alpha channel of the requested kind."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.zeros((h, w, 4), np.float64)
    a[..., 0] = 128 + 100 * np.sin(xx / 7.0 + seed) * np.cos(yy / 11.0)
    a[..., 1] = 128 + 90 * np.sin(yy / 5.0 - seed) + 20 * np.sin(xx / 3.0)
    a[..., 2] = 255 * ((xx / max(w, 1) + yy / max(h, 1)) / 2)
    for _ in range(5):
        cx, cy = r.integers(0, max(w, 1)), r.integers(0, max(h, 1))
        rad = r.integers(2, max(3, min(w, h) // 2 + 2))
        col = r.integers(0, 256, 3)
        m = (xx - cx) ** 2 + (yy - cy) ** 2 < rad * rad
        a[m, :3] = col
    a[..., :3] += r.normal(0, 5, a[..., :3].shape)
    if alpha_kind == "opaque":
        a[..., 3] = 255
    elif alpha_kind == "ramp":
        a[..., 3] = 255 * (xx / max(w - 1, 1))
    elif alpha_kind == "disc":
        d = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
        a[..., 3] = np.where(d < min(w, h) / 3, 255, np.where(d < min(w, h) / 2.2, 128, 0))
    elif alpha_kind == "noise":
        a[..., 3] = r.integers(0, 256, (h, w))
    return np.clip(a, 0, 255).astype(np.uint8)


def pillow_frames(path):
    out = []
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
        for k in range(n):
            im.seek(k)
            out.append(np.asarray(im.convert("RGBA"), dtype=np.uint8))
    return out


rows = []


def record(name, folder, path):
    frames = pillow_frames(path)
    h, w = frames[0].shape[:2]
    hashes = ",".join(hashlib.sha256(f.tobytes()).hexdigest() for f in frames)
    truth = 1 if w * h <= FULL_TRUTH_MAX_PIXELS else 0
    if truth:
        for k, f in enumerate(frames):
            suffix = ".rgba8.gz" if len(frames) == 1 else f".f{k}.rgba8.gz"
            with open(os.path.join(OUT, name + suffix), "wb") as fh:
                fh.write(gzip.compress(f.tobytes(), 6, mtime=0))
    rows.append((name, folder, len(frames), w, h, hashes, truth))


def save(name, arr, **kw):
    path = os.path.join(OUT, name + ".webp")
    Image.fromarray(arr, "RGBA").save(path, "WEBP", lossless=False, **kw)
    record(name, "WebPLossy", path)


# 1. Stills: sizes x quality x alpha kind. Odd sizes hit the chroma edge rules; 1x1 is the degenerate case.
sizes = [(1, 1), (2, 3), (7, 5), (16, 16), (17, 33), (33, 17), (64, 48), (101, 67), (255, 129)]
n = 0
for (w, h) in sizes:
    for q, aq, kind in [(20, 100, "ramp"), (75, 40, "disc"), (100, 100, "noise"), (60, 100, "opaque")]:
        save(f"ls_{w}x{h}_q{q}_aq{aq}_{kind}", picture(w, h, w * 7 + h * 3 + q, kind), quality=q, alpha_quality=aq, method=4)
        n += 1

# 2. Methods 0..6 on one picture (different segment / filter / mode decisions in the encoder).
for m in range(7):
    save(f"ls_m{m}_97x61", picture(97, 61, 500 + m, "disc"), quality=70, alpha_quality=90, method=m)

# 3. Production-scale pictures (hash truth): a VGA photo-like frame, a 1080p frame, a tall and a wide strip.
for (w, h, q) in [(640, 480, 80), (1920, 1080, 75), (123, 1600, 50), (1800, 90, 85)]:
    save(f"ls_big_{w}x{h}_q{q}", picture(w, h, w + h, "disc"), quality=q, alpha_quality=80, method=3)

# 4. Animations: lossy frames (sub-rectangles, blend/dispose chosen by libwebp's animation encoder), and mixed frames.
def anim(name, w, h, count, frames_kw, **kw):
    frames = []
    for k in range(count):
        base = picture(w, h, 900 + k, "disc")
        # a block that moves, over a static background, so the encoder emits sub-rectangle frames
        bx = (k * 7) % max(w - 12, 1)
        base2 = picture(w, h, 900, "disc")
        base2[4:4 + 12, bx:bx + 12] = base[4:4 + 12, bx:bx + 12]
        frames.append(Image.fromarray(base2, "RGBA"))
    path = os.path.join(OUT, name + ".webp")
    frames[0].save(path, "WEBP", save_all=True, append_images=frames[1:], duration=[40 + 10 * k for k in range(count)],
                   loop=0, **kw)
    record(name, "WebPLossy", path)


anim("la_lossy_48x40", 48, 40, 5, None, lossless=False, quality=70, alpha_quality=90, method=3, kmin=0, kmax=0)
anim("la_lossy_odd_37x29", 37, 29, 4, None, lossless=False, quality=40, alpha_quality=100, method=2, minimize_size=True)
anim("la_mixed_64x48", 64, 48, 6, None, lossless=False, quality=60, alpha_quality=100, method=3, allow_mixed=True)
anim("la_lossy_keyframes_80x60", 80, 60, 6, None, lossless=False, quality=85, alpha_quality=100, method=4, kmin=1, kmax=2)

# 5. Existing lossy files: their truth only (the manifest points at the folder holding the file).
src = os.path.join(DATA, "WebP")
for fn in sorted(os.listdir(src)):
    if not fn.endswith(".webp"):
        continue
    blob = open(os.path.join(src, fn), "rb").read()
    if b"VP8 " not in blob:
        continue
    try:
        record(fn[:-5], "WebP", os.path.join(src, fn))
    except Exception as ex:   # a deliberately malformed file: the decoder tests cover those as rejections
        print("skip (Pillow cannot decode):", fn, type(ex).__name__, ex)

src = os.path.join(DATA, "WebPYuv")
for fn in sorted(os.listdir(src)):
    if fn.endswith(".webp"):
        record(fn[:-5], "WebPYuv", os.path.join(src, fn))

with open(os.path.join(OUT, "manifest.tsv"), "w", newline="\n") as fh:
    for r in rows:
        fh.write("\t".join(str(x) for x in r) + "\n")
print(len(rows), "entries;", sum(1 for r in rows if r[6]), "with full truth")
