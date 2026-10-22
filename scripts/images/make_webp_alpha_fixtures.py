#!/usr/bin/env python3
"""
Generates the ALPH (alpha plane of lossy WebP) fixtures into SpawnDev.Codecs.Demo.Shared/TestData/WebP, all named alph_*:

  alph_<name>.webp                a VP8X file: VP8X + ALPH + VP8 (the VP8 colour data is a small lossy image from libwebp)
  alph_<name>.alpha8.gz           TRUTH: the alpha plane (w*h bytes) as libwebp decodes the file (Pillow RGBA channel 3)
  alph_anim_*.webp                an animation whose frames are ALPH + VP8, truth per frame: alph_anim_*.f<k>.alpha8.gz
  alph_manifest.tsv               name, kind, width, height, frames, method, filter, preprocessing, features

Two families:
  * crafted ALPH chunks: the plane is filtered by an independent numpy implementation of the spec's four prediction
    filters, then stored raw (method 0) or compressed by libwebp's own VP8L encoder (method 1, with the 5-byte VP8L
    header stripped, as the spec requires). Truth is the SOURCE plane, and libwebp must decode the file to exactly it.
  * libwebp's own ALPH chunks (Pillow RGBA lossy save, alpha_quality 100 and a reduced one): truth is libwebp's decode.

The library's own code is never used as truth.

Run:   <venv>/python -I scripts/images/make_webp_alpha_fixtures.py        (needs numpy + pillow)
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
    if f.startswith("alph_"):
        os.remove(os.path.join(OUT, f))

manifest = []


def gz(raw):
    return gzip.compress(raw, 9, mtime=0)


def write(name, data):
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(data)


# --------------------------------------------------------------------------------------------------
# planes
# --------------------------------------------------------------------------------------------------

def p_smooth(h, w, seed=1):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx * 255 // max(1, w - 1) + yy * 255 // max(1, h - 1)) // 2 + r.integers(0, 3, (h, w))).clip(0, 255).astype(np.uint8)


def p_mask(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - w / 2, yy - h / 2)
    r = min(h, w) * 0.4
    a = np.where(d < r, 255, 0)
    ring = (d >= r) & (d < r + 3)
    a = np.where(ring, (255 * (1 - (d - r) / 3)).astype(int), a)
    return a.astype(np.uint8)


def p_noise(h, w, seed=2):
    return np.random.default_rng(seed).integers(0, 256, (h, w)).astype(np.uint8)


def p_steps(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx // 9) % 4) * 80 + ((yy // 7) % 2) * 15).astype(np.uint8)


def p_flat(h, w, v):
    return np.full((h, w), v, np.uint8)


PLANES = [
    ("smooth_64x48", p_smooth(48, 64)),
    ("mask_61x45", p_mask(45, 61)),
    ("noise_33x29", p_noise(29, 33)),
    ("steps_50x40", p_steps(40, 50)),
    ("flat255_32x24", p_flat(24, 32, 255)),
    ("flat0_17x11", p_flat(11, 17, 0)),
    ("tall_1x37", p_smooth(37, 1, 3)),
    ("wide_41x1", p_smooth(1, 41, 4)),
    ("one_1x1", p_flat(1, 1, 77)),
    ("big_300x200", p_smooth(200, 300, 5)),
]

# --------------------------------------------------------------------------------------------------
# independent implementation of the ALPH prediction filters (spec: "Alpha" chunk, filtering methods)
# --------------------------------------------------------------------------------------------------

def predictor(plane, f):
    p = plane.astype(np.int32)
    left = np.zeros_like(p)
    left[:, 1:] = p[:, :-1]
    top = np.zeros_like(p)
    top[1:, :] = p[:-1, :]
    tl = np.zeros_like(p)
    tl[1:, 1:] = p[:-1, :-1]
    if f == 0:
        return np.zeros_like(p)
    if f == 1:
        pred = left.copy()
    elif f == 2:
        pred = top.copy()
    else:
        pred = np.clip(left + top - tl, 0, 255)
    pred[0, 1:] = left[0, 1:]        # the top row is predicted from the left sample
    pred[1:, 0] = top[1:, 0]         # the left column from the sample above
    pred[0, 0] = 0                   # the first sample from 0
    return pred


def apply_filter(plane, f):
    if f == 0:
        return plane.copy()
    return ((plane.astype(np.int32) - predictor(plane, f)) & 255).astype(np.uint8)


# --------------------------------------------------------------------------------------------------
# file builders
# --------------------------------------------------------------------------------------------------

_vp8_cache = {}


def lossy_vp8(w, h):
    """A lossy image's VP8 chunk payload (the colour half of the file)."""
    if (w, h) not in _vp8_cache:
        r = np.random.default_rng(w * 1000 + h)
        yy, xx = np.mgrid[0:h, 0:w]
        rgb = np.dstack([(xx * 4 + yy) & 255, (yy * 5) & 255, r.integers(0, 256, (h, w))]).astype(np.uint8)
        b = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(b, format="WEBP", lossless=False, quality=60)
        _vp8_cache[(w, h)] = [p for f, p in W.riff_chunks(b.getvalue()) if f == b"VP8 "][0]
    return _vp8_cache[(w, h)]


def vp8x(w, h, flags):
    return W.make_chunk(b"VP8X", bytes([flags, 0, 0, 0]) + struct.pack("<I", w - 1)[:3] + struct.pack("<I", h - 1)[:3])


def build_still(w, h, alph):
    return W.make_riff([vp8x(w, h, 0x10), W.make_chunk(b"ALPH", alph), W.make_chunk(b"VP8 ", lossy_vp8(w, h))])


def alph_raw(res, f, pre=0):
    return bytes([(pre << 4) | (f << 2)]) + res.tobytes()


def alph_lossless(res, f, wmethod, quality, pre=0, rich=False):
    """ALPH payload: header byte + the headerless VP8L stream libwebp's encoder makes of an image whose green is the residual.

    rich=True also fills red and blue (correlated with green, plus noise). A decoder only reads green, but the stream
    then needs subtract-green / cross-colour / predictor to compress, so every VP8L transform is exercised in the
    headerless ALPH entry point."""
    h, w = res.shape
    argb = np.zeros((h, w, 4), np.uint8)
    argb[..., 1] = res
    argb[..., 3] = 255
    if rich:
        r = np.random.default_rng(h * 31 + w)
        argb[..., 0] = ((res.astype(int) * 3 // 4) + r.integers(0, 3, (h, w))) & 255
        argb[..., 2] = ((res.astype(int) // 2) + r.integers(0, 3, (h, w))) & 255
    b = io.BytesIO()
    Image.fromarray(argb, "RGBA").save(b, format="WEBP", lossless=True, exact=True, method=wmethod, quality=quality)
    payload = [p for fcc, p in W.riff_chunks(b.getvalue()) if fcc == b"VP8L"][0]
    return bytes([(pre << 4) | (f << 2) | 1]) + payload[5:]


def alph_features(alph, w, h):
    """Which VP8L features a method-1 ALPH chunk uses, via the independent pure-Python decoder (re-adding the 5-byte header)."""
    if alph[0] & 3 != 1:
        return "raw"
    hdr = 0x2F | ((w - 1) << 8) | ((h - 1) << 22)
    _, _, pix, st = W.decode_vp8l(hdr.to_bytes(5, "little") + alph[1:])
    g = np.array([(q >> 8) & 255 for q in pix], np.uint8).reshape(h, w)
    return st.summary(), g


def lib_alpha(data):
    """libwebp's decode of every frame: list of (H,W) alpha channels."""
    im = Image.open(io.BytesIO(data))
    out = []
    for k in range(getattr(im, "n_frames", 1)):
        im.seek(k)
        out.append(np.array(im.convert("RGBA"))[..., 3])
    return out


def add(name, kind, data, truth, method, f, pre, feat):
    h, w = truth.shape
    got = lib_alpha(data)[0]
    assert got.shape == truth.shape and (got == truth).all(), name + ": libwebp alpha differs from the truth plane"
    write("alph_%s.webp" % name, data)
    write("alph_%s.alpha8.gz" % name, gz(truth.tobytes()))
    manifest.append(("alph_" + name, kind, w, h, 1, method, f, pre, feat))
    print("%-34s %6d B  method=%d filter=%d pre=%d  %s" % (name, len(data), method, f, pre, feat[:90]))


# 1. crafted chunks: every plane x every filter x {raw, lossless}
idx = 0
for pname, plane in PLANES:
    h, w = plane.shape
    for f in range(4):
        if pname.startswith("big") and f in (0, 2):
            continue  # the big plane only needs the two filters that are not cheap to get right by accident
        res = apply_filter(plane, f)
        if not pname.startswith("big"):
            add("%s_f%d_raw" % (pname, f), "still", build_still(w, h, alph_raw(res, f)), plane, 0, f, 0, "raw")
        wm = idx % 7
        idx += 1
        chunk = alph_lossless(res, f, wm, 75 if wm != 0 else 20)
        feat, g = alph_features(chunk, w, h)
        assert (g == res).all(), "webp_ref disagrees with the residual for " + pname
        add("%s_f%d_vp8l" % (pname, f), "still", build_still(w, h, chunk), plane, 1, f, 0, feat)

# VP8L streams that use subtract green, cross colour, predictor, colour cache, meta prefix codes inside an ALPH chunk
rich_seen = set()
for pname, plane in (PLANES[0], PLANES[1], PLANES[2]):
    h, w = plane.shape
    for f, wm, q in ((0, 4, 75), (3, 6, 100)):
        res = apply_filter(plane, f)
        chunk = alph_lossless(res, f, wm, q, rich=True)
        feat, g = alph_features(chunk, w, h)
        assert (g == res).all()
        rich_seen.update(feat.split(" ")[0][3:-1].split(","))
        add("rich_%s_f%d_m%d" % (pname, f, wm), "still", build_still(w, h, chunk), plane, 1, f, 0, feat)
print("transforms in rich ALPH streams:", sorted(rich_seen))
assert any(t.startswith("G") for t in rich_seen) and any(t.startswith("C") for t in rich_seen) \
    and any(t.startswith("P") for t in rich_seen), "rich streams must cover subtract green, cross colour and predictor"

# pre-processing field 1 (level reduction) is informative only: the same raw data must decode identically
plane = PLANES[0][1]
res = apply_filter(plane, 3)
add("smooth_64x48_f3_raw_pre1", "still", build_still(64, 48, alph_raw(res, 3, pre=1)), plane, 0, 3, 1, "raw")
# a raw chunk with trailing bytes after the plane (libwebp ignores them)
add("smooth_64x48_f0_raw_trailing", "still", build_still(64, 48, alph_raw(plane, 0) + b"\x01\x02\x03\x04\x05"), plane, 0, 0, 0, "raw+trailing")

# 2. libwebp's own ALPH chunks
for pname, plane in [PLANES[0], PLANES[1], PLANES[3]]:
    h, w = plane.shape
    rgb = np.random.default_rng(h * w).integers(0, 256, (h, w, 3)).astype(np.uint8)
    rgba = np.dstack([rgb, plane])
    for aq in (100, 30):
        b = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(b, format="WEBP", lossless=False, quality=60, alpha_quality=aq)
        data = b.getvalue()
        alph = [p for fcc, p in W.riff_chunks(data) if fcc == b"ALPH"]
        assert alph, "libwebp wrote no ALPH chunk"
        a0 = alph[0]
        truth = lib_alpha(data)[0]
        if aq == 100:
            assert (truth == plane).all(), "alpha_quality 100 must be lossless"
        feat = alph_features(a0, w, h)[0] if (a0[0] & 3) == 1 else "raw"
        add("libwebp_%s_q%d" % (pname, aq), "still", data, truth, a0[0] & 3, (a0[0] >> 2) & 3, (a0[0] >> 4) & 3, feat)

# 3. an animation: three lossy frames with ALPH (offsets even, no blend, dispose none => each frame replaces its rectangle)
CW, CH = 48, 40
frame_defs = [
    (0, 0, p_smooth(CH, CW, 11), 3, 1),
    (8, 6, p_mask(16, 20), 1, 0),
    (4, 2, p_steps(20, 30), 2, 1),
]
chunks = [vp8x(CW, CH, 0x02 | 0x10), W.make_chunk(b"ANIM", struct.pack("<IH", 0, 0))]
planes = []
for k, (x, y, plane, f, method) in enumerate(frame_defs):
    h, w = plane.shape
    res = apply_filter(plane, f)
    alph = alph_lossless(res, f, 4, 75) if method == 1 else alph_raw(res, f)
    sub = W.make_chunk(b"ALPH", alph) + W.make_chunk(b"VP8 ", lossy_vp8(w, h))
    hdr = struct.pack("<I", x // 2)[:3] + struct.pack("<I", y // 2)[:3] + struct.pack("<I", w - 1)[:3] \
        + struct.pack("<I", h - 1)[:3] + struct.pack("<I", 50)[:3] + bytes([2])  # bit 1: do not blend
    chunks.append(W.make_chunk(b"ANMF", hdr + sub))
    planes.append((x, y, plane))
data = W.make_riff(chunks)
canvases = lib_alpha(data)
for k, (x, y, plane) in enumerate(planes):
    h, w = plane.shape
    assert (canvases[k][y:y + h, x:x + w] == plane).all(), "animation frame %d: libwebp alpha differs from the plane" % k
    write("alph_anim_48x40.f%d.alpha8.gz" % k, gz(plane.tobytes()))
write("alph_anim_48x40.webp", data)
manifest.append(("alph_anim_48x40", "anim", CW, CH, len(planes), 1, 0, 0, "frames=%d" % len(planes)))
print("%-34s %6d B  %d frames" % ("anim_48x40", len(data), len(planes)))

# 4. what libwebp itself does with the malformed ALPH chunks the decoder tests feed our decoder (the tests' oracle)
def libwebp_accepts(alph):
    try:
        lib_alpha(build_still(64, 48, alph))
        return True
    except Exception:  # noqa: BLE001
        return False


base_alph = alph_raw(PLANES[0][1], 0)
expect_reject = {
    "method 2": bytes([2]) + base_alph[1:], "method 3": bytes([3]) + base_alph[1:], "pre-processing 2": bytes([0x20]) + base_alph[1:],
    "reserved bit 6": bytes([0x40]) + base_alph[1:], "reserved bit 7": bytes([0x80]) + base_alph[1:], "empty": b"",
    "header only": bytes([0]), "raw one byte short": base_alph[:-1],
    "vp8l cut": alph_lossless(apply_filter(PLANES[0][1], 3), 3, 4, 75)[:60],
}
for what, a in expect_reject.items():
    assert not libwebp_accepts(a), "libwebp accepted the malformed ALPH chunk: " + what
assert libwebp_accepts(base_alph) and libwebp_accepts(alph_raw(PLANES[0][1], 0, pre=1)) and libwebp_accepts(base_alph + b"xyz")
print("libwebp rejects %d malformed ALPH variants and accepts the valid ones" % len(expect_reject))

# 5. webp_ref's ALPH decoder (the third oracle of the encoder tests) must agree with libwebp on every still fixture
checked = 0
for row in manifest:
    if row[1] != "still":
        continue
    fdata = open(os.path.join(OUT, row[0] + ".webp"), "rb").read()
    truth_plane = np.frombuffer(gzip.decompress(open(os.path.join(OUT, row[0] + ".alpha8.gz"), "rb").read()), np.uint8).reshape(row[3], row[2])
    alph_payload = [p for fcc, p in W.riff_chunks(fdata) if fcc == b"ALPH"][0]
    got_plane = np.array(W.alpha_decode(alph_payload, row[2], row[3])[0], np.uint8)
    assert (got_plane == truth_plane).all(), "webp_ref ALPH decode differs from libwebp: " + row[0]
    checked += 1
print("webp_ref ALPH decode agrees with libwebp on %d fixtures" % checked)

with open(os.path.join(OUT, "alph_manifest.tsv"), "w") as fh:
    for row in manifest:
        fh.write("\t".join(str(v) for v in row) + "\n")
print("fixtures:", len(manifest))
