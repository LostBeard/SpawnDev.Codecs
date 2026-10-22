#!/usr/bin/env python3
"""
Generates the SOURCES of the animated-WebP ENCODER tests into SpawnDev.Codecs.Demo.Shared/TestData/WebP, all named animenc_*:

  animenc_<name>.f<k>.rgba8.gz       frame k's pixels (RGBA8, exactly the frame's rectangle) - what the encoder is given
  animenc_<name>.frames.tsv          frame k: x y w h duration_ms dispose(0 none, 1 background) blend(0 source, 1 over)
  animenc_<name>.canvas<k>.rgba8.gz  TRUTH: the canvas as it must look while frame k is shown, computed here by an independent
                                     numpy compositor (WebP spec semantics) and cross-checked against libwebp (see below)
  animenc_manifest.tsv               name, canvas width, canvas height, frame count

Blend "over" frames use alpha 0 / 255 only, where every blend formula agrees (the decoder tests cover partial-alpha blending
against libwebp's integer arithmetic). Frames with blend "source" may carry any alpha.

Cross-check: each animation is also assembled into a WebP file with libwebp's own lossless encoder (Pillow) for the frames and
hand-written ANMF chunks; libwebp's animation decoder must show exactly the canvases computed here. The encoder under test is
never involved in the truth.

Run:   <venv>/python -I scripts/images/make_webp_anim_encode_fixtures.py        (needs numpy + pillow)
"""
import gzip
import io
import os
import struct
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import webp_ref as W  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "WebP")
for f in os.listdir(OUT):
    if f.startswith("animenc_"):
        os.remove(os.path.join(OUT, f))

manifest = []


def gz(raw):
    return gzip.compress(raw, 9, mtime=0)


def write(name, data):
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(data)


def photo(h, w, seed, level=6):
    """A smooth colourful picture with a little noise (compresses with the predictor / cross colour transforms)."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.zeros((h, w, 4), np.uint8)
    a[..., 0] = (xx * 3 + seed * 7 + r.integers(0, level, (h, w))) & 255
    a[..., 1] = (yy * 4 + xx + r.integers(0, level, (h, w))) & 255
    a[..., 2] = ((xx + yy) * 2 + seed * 13 + r.integers(0, level, (h, w))) & 255
    a[..., 3] = 255
    return a


def binary_alpha(h, w, seed):
    a = photo(h, w, seed)
    yy, xx = np.mgrid[0:h, 0:w]
    a[..., 3] = np.where(np.hypot(xx - w / 2, yy - h / 2) < min(h, w) * 0.45, 255, 0)
    # RGB under alpha 0 is 0: libwebp copies (not blends) an "over" frame into the area the previous frame disposed, so junk
    # RGB under alpha 0 would survive there and the spec compositor (alpha 0 = keep the transparent canvas) would differ
    a[a[..., 3] == 0, :3] = 0
    return a


def any_alpha(h, w, seed):
    a = photo(h, w, seed, 40)
    r = np.random.default_rng(seed + 1)
    a[..., 3] = np.where(r.random((h, w)) < 0.3, 0, r.integers(1, 256, (h, w)))  # many fully transparent pixels WITH rgb
    return a


def frame(x, y, arr, dur, dispose_bg=False, over=False):
    return dict(x=x, y=y, arr=arr, dur=dur, dispose_bg=dispose_bg, over=over)


def composite(cw, ch, frames):
    canvas = np.zeros((ch, cw, 4), np.uint8)
    out = []
    prev = None
    for f in frames:
        if prev is not None and prev["dispose_bg"]:
            ph, pw = prev["arr"].shape[:2]
            canvas[prev["y"]:prev["y"] + ph, prev["x"]:prev["x"] + pw] = 0
        arr = f["arr"]
        h, w = arr.shape[:2]
        region = canvas[f["y"]:f["y"] + h, f["x"]:f["x"] + w]
        if f["over"]:
            assert set(np.unique(arr[..., 3])) <= {0, 255}, "blend over is only exact for alpha 0 / 255"
            m = arr[..., 3] == 255
            region[m] = arr[m]
        else:
            region[...] = arr
        out.append(canvas.copy())
        prev = f
    return out


def libwebp_canvases(cw, ch, frames):
    """Assemble the animation with libwebp-encoded frames and return libwebp's decoded canvases."""
    any_a = any((f["arr"][..., 3] != 255).any() for f in frames)
    vp8x = bytes([0x02 | (0x10 if any_a else 0), 0, 0, 0]) + struct.pack("<I", cw - 1)[:3] + struct.pack("<I", ch - 1)[:3]
    chunks = [W.make_chunk(b"VP8X", vp8x), W.make_chunk(b"ANIM", struct.pack("<IH", 0, 0))]
    for f in frames:
        h, w = f["arr"].shape[:2]
        b = io.BytesIO()
        Image.fromarray(f["arr"], "RGBA").save(b, format="WEBP", lossless=True, exact=True)
        pl = [p for fc, p in W.riff_chunks(b.getvalue()) if fc == b"VP8L"][0]
        hdr = struct.pack("<I", f["x"] // 2)[:3] + struct.pack("<I", f["y"] // 2)[:3] + struct.pack("<I", w - 1)[:3] \
            + struct.pack("<I", h - 1)[:3] + struct.pack("<I", f["dur"])[:3] \
            + bytes([(1 if f["dispose_bg"] else 0) | (0 if f["over"] else 2)])
        chunks.append(W.make_chunk(b"ANMF", hdr + W.make_chunk(b"VP8L", pl)))
    im = Image.open(io.BytesIO(W.make_riff(chunks)))
    res = []
    for k in range(im.n_frames):
        im.seek(k)
        res.append(np.array(im.convert("RGBA")))
    return res


def add(name, cw, ch, frames):
    canvases = composite(cw, ch, frames)
    lib = libwebp_canvases(cw, ch, frames)
    assert len(lib) == len(canvases)
    for k, (a, b) in enumerate(zip(canvases, lib)):
        assert (a == b).all(), "%s: the numpy compositor and libwebp disagree on frame %d" % (name, k)
    rows = []
    for k, f in enumerate(frames):
        h, w = f["arr"].shape[:2]
        assert f["x"] % 2 == 0 and f["y"] % 2 == 0
        write("animenc_%s.f%d.rgba8.gz" % (name, k), gz(f["arr"].tobytes()))
        write("animenc_%s.canvas%d.rgba8.gz" % (name, k), gz(canvases[k].tobytes()))
        rows.append((f["x"], f["y"], w, h, f["dur"], 1 if f["dispose_bg"] else 0, 1 if f["over"] else 0))
    write("animenc_%s.frames.tsv" % name, ("\n".join("\t".join(str(v) for v in r) for r in rows) + "\n").encode())
    manifest.append((name, cw, ch, len(frames)))
    print("%-28s canvas %dx%d  %d frames" % (name, cw, ch, len(frames)))


CW, CH = 64, 48
# full-canvas opaque frames, different durations
add("opaque_64x48", CW, CH, [frame(0, 0, photo(CH, CW, 300 + k), 40 + 10 * k) for k in range(5)])

# sub-rectangles, every blend / dispose combination, binary and arbitrary alpha
add("rects_64x48", CW, CH, [
    frame(0, 0, photo(CH, CW, 310), 80),
    frame(8, 6, binary_alpha(20, 30, 311), 100, over=True),
    frame(20, 10, any_alpha(30, 30, 312), 120, dispose_bg=True),
    frame(0, 0, binary_alpha(40, 40, 313), 90, over=True),
    frame(10, 2, any_alpha(20, 20, 314), 70, dispose_bg=True),
    frame(30, 20, binary_alpha(26, 34, 315), 60, over=True),
    frame(4, 4, photo(12, 14, 316), 50, dispose_bg=True, over=True),
])

# thin and odd-sized frames at even offsets
add("thin_7x9", 7, 9, [
    frame(0, 0, binary_alpha(9, 7, 320), 30),
    frame(2, 0, photo(9, 1, 321), 30, over=True),
    frame(0, 4, photo(1, 7, 322), 30, dispose_bg=True),
    frame(4, 2, photo(1, 1, 323), 30, over=True),
    frame(6, 8, photo(1, 1, 324), 30),
])

# one frame only
add("single_32x24", 32, 24, [frame(0, 0, photo(24, 32, 330), 1000)])

# bigger frames: several prefix groups, longer matches, transform choices
cw, ch = 96, 72
big0 = photo(ch, cw, 340)
big1 = big0.copy()
big1[20:50, 24:72, :3] = np.array([250, 20, 30], np.uint8)
big2 = big1.copy()
big2[6:30, 60:90, :3] = np.array([20, 220, 40], np.uint8)
add("grow_96x72", cw, ch, [frame(0, 0, big0, 100), frame(0, 0, big1, 100), frame(0, 0, big2, 250)])

# maximum duration (24 bits) and a duration of 0
add("durations_16x16", 16, 16, [frame(0, 0, photo(16, 16, 350), 0), frame(0, 0, photo(16, 16, 351), (1 << 24) - 1)])

with open(os.path.join(OUT, "animenc_manifest.tsv"), "w") as fh:
    for row in manifest:
        fh.write("\t".join(str(v) for v in row) + "\n")
print("fixtures:", len(manifest))
