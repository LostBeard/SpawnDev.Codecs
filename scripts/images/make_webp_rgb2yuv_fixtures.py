"""Fixtures for WebPRgbaToYuvGpu (Images/Yuv): RGBA8 -> YUV 4:2:0 + alpha exactly as libwebp's lossy encoder imports it.

Truth:
  <name>.yuva.gz        Y, U, V (chroma (w+1)/2 x (h+1)/2) and the alpha plane from LIBWEBP ITSELF: WebPPictureImportRGBA
                        with use_argb = 0 (ImportYUVAFromRGBA), read through the `webp` cffi package (libwebp_picture.py).
  <name>.clean.yuva.gz  the same after WebPCleanupTransparentArea (libwebp runs it unless config->exact). That function is not
                        exported by the bindings, so it is ported below line by line from src/enc/picture_tools_enc.c (1.6.0)
                        and applied to libwebp's planes.
  <name>.rgba.gz        the input.
  manifest.tsv          name, width, height
Nothing here comes from SpawnDev.Codecs.

Usage: python -I make_webp_rgb2yuv_fixtures.py     (venv with numpy + webp)
"""
import gzip
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import libwebp_picture  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "WebPRgbToYuv"))
os.makedirs(OUT, exist_ok=True)


def smoothen_block(a, y, x0, y0, w, h):
    blk_a = a[y0:y0 + h, x0:x0 + w]
    blk_y = y[y0:y0 + h, x0:x0 + w]
    vis = blk_a != 0
    count = int(vis.sum())
    if 0 < count < w * h:
        avg = int(blk_y[vis].astype(np.int64).sum()) // count
        blk_y[~vis] = avg
    return count == 0


def cleanup_transparent_area(Y, U, V, A):
    """src/enc/picture_tools_enc.c WebPCleanupTransparentArea, the !use_argb branch, SIZE = 8."""
    Y, U, V = Y.copy(), U.copy(), V.copy()
    h, w = Y.shape
    y = 0
    while y + 8 <= h:
        need_reset = True
        values = [0, 0, 0]
        x = 0
        while x + 8 <= w:
            if smoothen_block(A, Y, x, y, 8, 8):
                if need_reset:
                    values = [int(Y[y, x]), int(U[y >> 1, x >> 1]), int(V[y >> 1, x >> 1])]
                    need_reset = False
                Y[y:y + 8, x:x + 8] = values[0]
                U[(y >> 1):(y >> 1) + 4, (x >> 1):(x >> 1) + 4] = values[1]
                V[(y >> 1):(y >> 1) + 4, (x >> 1):(x >> 1) + 4] = values[2]
            else:
                need_reset = True
            x += 8
        if x < w:
            smoothen_block(A, Y, x, y, w - x, 8)
        y += 8
    if y < h:
        sub_h = h - y
        x = 0
        while x + 8 <= w:
            smoothen_block(A, Y, x, y, 8, sub_h)
            x += 8
        if x < w:
            smoothen_block(A, Y, x, y, w - x, sub_h)
    return Y, U, V


def picture(w, h, seed, kind, noise=18):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.zeros((h, w, 4), np.float64)
    a[..., 0] = 128 + 110 * np.sin(xx / 5.0 + seed) * np.cos(yy / 9.0)
    a[..., 1] = 128 + 100 * np.sin(yy / 4.0 - seed) + 25 * np.sin(xx / 2.0)
    a[..., 2] = 255 * ((xx / max(w, 1) + yy / max(h, 1)) / 2)
    if noise: a[..., :3] += r.normal(0, noise, a[..., :3].shape)
    if kind == "opaque":
        a[..., 3] = 255
    elif kind == "ramp":
        a[..., 3] = 255 * (xx / max(w - 1, 1))
    elif kind == "disc":
        d = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
        a[..., 3] = np.where(d < min(w, h) / 3, 255, np.where(d < min(w, h) / 2.2, 97, 0))
    elif kind == "holes":
        # fully transparent runs of whole 8x8 blocks (flatten path) plus ragged transparent pixels (smoothen path)
        a[..., 3] = 255
        bx, by = (xx // 8).astype(int), (yy // 8).astype(int)
        a[((bx % 5) < 3) & ((by % 3) == 1), 3] = 0
        a[(r.random((h, w)) < 0.15), 3] = 0
        a[(xx + yy) % 11 == 0, 3] = 40
    elif kind == "noise":
        a[..., 3] = r.integers(0, 256, (h, w))
    elif kind == "clear":
        a[..., 3] = 0
    return np.clip(a, 0, 255).astype(np.uint8)


def procedural(w, h):
    """Integer-only picture for the production-scale cases; the C# test builds the SAME bytes (ProceduralRgba)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.int64)
    rgba = np.zeros((h, w, 4), np.int64)
    rgba[..., 0] = (xx * 5 + yy * 3) & 255
    rgba[..., 1] = ((xx ^ yy) * 7) & 255
    rgba[..., 2] = ((xx * yy) >> 3) & 255
    R = min(w, h)
    d2 = (2 * xx - w) ** 2 + (2 * yy - h) ** 2
    alpha = np.where(9 * d2 < 4 * R * R, 255, np.where(121 * d2 < 100 * R * R, 97, 0))
    holes = (((xx >> 3) % 5) < 3) & (((yy >> 3) % 3) == 1) & (9 * d2 >= 4 * R * R)
    alpha = np.where(holes, 0, alpha)
    rgba[..., 3] = alpha
    return rgba.astype(np.uint8)


def gz(path, data):
    with open(path, "wb") as f:
        f.write(gzip.compress(data, 6, mtime=0))


rows = []
sizes = [(1, 1), (2, 2), (3, 5), (5, 3), (8, 8), (9, 9), (16, 8), (17, 33), (33, 17), (64, 48), (100, 75), (255, 129)]
kinds = ["opaque", "ramp", "disc", "holes", "noise", "clear"]
for (w, h) in sizes:
    for k in kinds:
        rows.append((f"c_{w}x{h}_{k}", w, h, k))
rows.append(("c_640x480_proc", 640, 480, "proc"))
rows.append(("c_1920x1080_proc", 1920, 1080, "proc"))

with open(os.path.join(OUT, "manifest.tsv"), "w", newline="\n") as man:
    for (name, w, h, k) in rows:
        big = w * h > 100000
        rgba = procedural(w, h) if big else picture(w, h, w * 13 + h * 7 + len(k), k)
        Y, U, V, A = libwebp_picture.import_rgba(rgba)
        if A is None:
            A = np.full((h, w), 255, np.uint8)   # libwebp allocates no alpha plane for an opaque picture
        assert np.array_equal(A, rgba[..., 3]), name
        Yc, Uc, Vc = cleanup_transparent_area(Y, U, V, A)
        raw = Y.tobytes() + U.tobytes() + V.tobytes() + A.tobytes()
        clean = Yc.tobytes() + Uc.tobytes() + Vc.tobytes() + A.tobytes()
        if big:   # production scale: the input is procedural (rebuilt by the test), the truth is a hash
            lines = [hashlib.sha256(rgba.tobytes()).hexdigest(), hashlib.sha256(raw).hexdigest(), hashlib.sha256(clean).hexdigest()]
            with open(os.path.join(OUT, name + ".sha256"), "w", newline="\n") as f:
                f.write("\n".join(lines) + "\n")
        else:
            gz(os.path.join(OUT, name + ".rgba.gz"), rgba.tobytes())
            gz(os.path.join(OUT, name + ".yuva.gz"), raw)
            gz(os.path.join(OUT, name + ".clean.yuva.gz"), clean)
        man.write(f"{name}\t{w}\t{h}\n")
print(len(rows), "fixtures")
