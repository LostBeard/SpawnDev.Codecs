#!/usr/bin/env python3
"""
Generates the WebP decoder/encoder test fixtures into SpawnDev.Codecs.Demo.Shared/TestData/WebP:

  <name>.webp                  the file (libwebp lossless via Pillow, or hand-assembled RIFF for animations)
  <name>.rgba8.gz              the SOURCE pixels the file was built from (RGBA8, rows top to bottom), gzip'd
  <name>.f<k>.rgba8.gz         animations: the composited canvas after frame k, as libwebp's WebPAnimDecoder
                               (Pillow seek/convert) shows it
  manifest.tsv                 one row per file: name, kind, width, height, has_alpha, frames, orientation, features
  <name>.frames.tsv            animations: index x y w h duration_ms dispose blend  (parsed from the RIFF ANMF chunks)
  sources/<name>.rgba8.gz      (encoder tests) the source images the encoder is asked to compress

Truth: lossless => the decoded pixels equal the source pixels EXACTLY (exact=True keeps RGB under alpha 0). Every
file is verified here with libwebp (Pillow): its decode must equal the source. Animations: the expected canvases
are libwebp's own anim decoder output. The library's own code is never used as truth.

webp_ref.py (an independent pure-Python VP8L decoder, itself cross-checked against libwebp per file) is used to
REPORT and ASSERT which bitstream features the fixture set exercises.

Run:   <venv>/python -I scripts/images/make_webp_fixtures.py        (needs numpy + pillow)
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
SRC = os.path.join(OUT, "sources")
os.makedirs(OUT, exist_ok=True)
os.makedirs(SRC, exist_ok=True)
for d in (OUT, SRC):
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isfile(p):
            os.remove(p)

rng = np.random.default_rng(20261022)
manifest = []
stats_all = []


def gz(raw):
    return gzip.compress(raw, 9, mtime=0)


def write(name, data):
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(data)


def rgba(h, w):
    a = np.zeros((h, w, 4), np.uint8)
    a[..., 3] = 255
    return a


def pil_save(arr, **kw):
    b = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(b, format="WEBP", lossless=True, exact=True, **kw)
    return b.getvalue()


def lib_decode(data):
    """libwebp's decode (through Pillow): (RGBA array, has_alpha, n_frames)."""
    im = Image.open(io.BytesIO(data))
    has_alpha = im.mode == "RGBA"
    n = getattr(im, "n_frames", 1)
    frames = []
    for k in range(n):
        im.seek(k)
        frames.append(np.array(im.convert("RGBA")))
    return frames, has_alpha, n


def vp8l_payload(data):
    for f, p in W.riff_chunks(data):
        if f == b"VP8L":
            return p
    return None


def exif_tiff(orientation):
    # little-endian TIFF, one IFD0 entry: Orientation (0x0112), SHORT, count 1
    return b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", 1) + struct.pack("<HHII", 0x0112, 3, 1, orientation) + struct.pack("<I", 0)


def check_features(name, data):
    p = vp8l_payload(data)
    if p is None:
        return None
    w, h, pix, st = W.decode_vp8l(p)
    stats_all.append((name, st))
    return st


def add_still(name, arr, kind="still", features=True, orientation=1, truth=None, **kw):
    data = kw.pop("_data", None) or pil_save(arr, **kw)
    frames, has_alpha, n = lib_decode(data)
    assert n == 1
    assert (frames[0] == arr).all(), name + ": libwebp decode differs from the source (not lossless?)"
    h, w = arr.shape[:2]
    write(name + ".webp", data)
    truth = truth or name
    if not os.path.exists(os.path.join(OUT, truth + ".rgba8.gz")):
        write(truth + ".rgba8.gz", gz(arr.tobytes()))
    st = check_features(name, data)
    if st is not None:
        w2, h2, pix, _ = W.decode_vp8l(vp8l_payload(data))
        ref = np.array([[(q >> 16) & 255, (q >> 8) & 255, q & 255, q >> 24] for q in pix], np.uint8).reshape(h, w, 4)
        assert (ref == arr).all(), name + ": webp_ref differs from the source"
    manifest.append((name, kind, w, h, int(has_alpha), 1, orientation, truth, st.summary() if st else "-"))
    print("%-28s %6d B  %s" % (name, len(data), st.summary() if st else ""))
    return data


def add_source(name, arr):
    """A source image for the encoder tests (no .webp)."""
    with open(os.path.join(SRC, name + ".rgba8.gz"), "wb") as f:
        f.write(gz(arr.tobytes()))
    manifest.append((name, "source", arr.shape[1], arr.shape[0], int((arr[..., 3] != 255).any()), 1, 1, name, "-"))


# --------------------------------------------------------------------------------------------------
# image makers
# --------------------------------------------------------------------------------------------------

def mk_flat(h, w, c):
    a = rgba(h, w)
    a[...] = c
    return a


def mk_photo(h, w, seed=1, noise=6):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    a = rgba(h, w)
    a[..., 0] = (xx * 255 // max(1, w - 1) + r.integers(0, noise, (h, w))) & 255
    a[..., 1] = (yy * 255 // max(1, h - 1) + r.integers(0, noise, (h, w))) & 255
    a[..., 2] = (((xx + yy) * 3) // 2 + r.integers(0, noise, (h, w))) & 255
    return a


def mk_mosaic(h, w, seed=2, tile=12):
    """Tiles of very different texture, so a predictor-per-tile encoder uses many modes."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    a = rgba(h, w)
    for ty in range(0, h, tile):
        for tx in range(0, w, tile):
            kind = int(r.integers(0, 12))
            sl = (slice(ty, min(h, ty + tile)), slice(tx, min(w, tx + tile)))
            y = yy[sl] - ty
            x = xx[sl] - tx
            base = r.integers(0, 256, 3)
            if kind == 0:
                v = np.zeros(y.shape + (3,), int) + base
            elif kind == 1:
                v = (base + np.stack([x * 9, x * 4, x * 6], -1))
            elif kind == 2:
                v = (base + np.stack([y * 7, y * 9, y * 3], -1))
            elif kind == 3:
                v = (base + np.stack([(x + y) * 5, (x + y) * 3, (x + y) * 7], -1))
            elif kind == 4:
                v = (base + np.stack([(x - y) * 6, (y - x) * 4, (x - y) * 2], -1))
            elif kind == 5:
                v = r.integers(0, 256, y.shape + (3,))
            elif kind == 6:
                v = base + r.integers(0, 4, y.shape + (3,))
            elif kind == 7:
                v = base + ((x ^ y) & 1)[..., None] * 90
            elif kind == 8:
                v = base + np.stack([x * y // 3, x * 2 + y, y * 2 - x], -1)
            elif kind == 9:
                v = np.stack([(x // 3) * 40, (y // 3) * 40, ((x + y) // 4) * 30], -1) + base // 4
            elif kind == 10:
                v = base + np.cumsum(r.integers(-3, 4, y.shape + (3,)), axis=1)
            else:
                v = base + np.cumsum(r.integers(-3, 4, y.shape + (3,)), axis=0)
            a[sl + (slice(0, 3),)] = (v & 255).astype(np.uint8)
    return a


def mk_palette(h, w, ncolors, seed=3, smooth=False):
    r = np.random.default_rng(seed)
    pal = r.integers(0, 256, (ncolors, 4)).astype(np.uint8)
    pal[:, 3] = 255
    if smooth:
        idx = ((np.mgrid[0:h, 0:w][1] + np.mgrid[0:h, 0:w][0]) // 2) % ncolors
    else:
        idx = r.integers(0, ncolors, (h, w))
    # make sure every colour appears
    idx.flat[:ncolors] = np.arange(ncolors)
    return pal[idx]


def mk_alpha(h, w, seed=4):
    r = np.random.default_rng(seed)
    a = mk_photo(h, w, seed, 10)
    alpha = r.integers(0, 256, (h, w)).astype(np.uint8)
    alpha[r.random((h, w)) < 0.25] = 0
    alpha[r.random((h, w)) < 0.15] = 255
    a[..., 3] = alpha
    # arbitrary colour under alpha 0: exact=True must preserve it
    return a


def mk_cache_heavy(h, w, seed=5):
    r = np.random.default_rng(seed)
    cols = r.integers(0, 256, (300, 4)).astype(np.uint8)
    cols[:, 3] = 255
    a = rgba(h, w)
    last = 0
    for y in range(h):
        for x in range(w):
            if r.random() < 0.35:
                last = int(r.integers(0, 300))
            a[y, x] = cols[(last + (x * 7 + y * 3) % 5) % 300]
    return a


def mk_large(h, w, seed=6):
    """Structured and compressible: shapes, gradients, stripes, a little noise."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    a = rgba(h, w)
    a[..., 0] = (xx * 255 // w)
    a[..., 1] = (yy * 255 // h)
    a[..., 2] = 128
    for i in range(40):
        cx, cy, rad = int(r.integers(0, w)), int(r.integers(0, h)), int(r.integers(10, 90))
        col = r.integers(0, 256, 3)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < rad * rad
        a[mask, :3] = col
    stripes = ((xx // 24) % 2 == 0) & (yy > h * 3 // 4)
    a[stripes, :3] = (a[stripes, :3] // 2)
    box = (xx > w // 3) & (xx < w // 3 + 300) & (yy > h // 8) & (yy < h // 8 + 120)
    a[box, 2] = (xx[box] + yy[box]) & 255
    return a


# --------------------------------------------------------------------------------------------------
# stills
# --------------------------------------------------------------------------------------------------

add_still("flat_32x24", mk_flat(24, 32, (200, 30, 90, 255)))
add_still("flat_alpha_16x16", mk_flat(16, 16, (10, 200, 60, 77)))
add_still("flat_transparent_20x9", mk_flat(9, 20, (0, 0, 0, 0)))
add_still("two_color_70x45", mk_palette(45, 70, 2, seed=11))
add_still("pal3_53x29", mk_palette(29, 53, 3, seed=12))
add_still("pal4_64x40", mk_palette(40, 64, 4, seed=13))
add_still("pal5_47x31", mk_palette(31, 47, 5, seed=14))
add_still("pal16_61x33", mk_palette(33, 61, 16, seed=15))
add_still("pal16_smooth_100x30", mk_palette(30, 100, 16, seed=16, smooth=True))
add_still("pal17_59x37", mk_palette(37, 59, 17, seed=17))
add_still("pal100_80x50", mk_palette(50, 80, 100, seed=18))
add_still("pal256_90x70", mk_palette(70, 90, 256, seed=19))
pa = mk_palette(40, 50, 12, seed=20)
pa[..., 3] = np.array([0, 40, 255, 128] * 3, np.uint8)[(np.arange(40 * 50) % 12).reshape(40, 50) % 12]
add_still("pal12_alpha_50x40", pa)

add_still("grad_h_128x32", np.dstack([np.tile(np.arange(128, dtype=np.uint8) * 2, (32, 1))] * 3 + [np.full((32, 128), 255, np.uint8)]))
add_still("grad_v_32x128", np.dstack([np.tile((np.arange(128, dtype=np.uint8) * 2)[:, None], (1, 32))] * 3 + [np.full((128, 32), 255, np.uint8)]))
add_still("grad_diag_64x64", mk_photo(64, 64, 7, 1))
add_still("mosaic_96x72", mk_mosaic(72, 96, 21))
add_still("mosaic_200x150_m6", mk_mosaic(150, 200, 22, 20), method=6)
add_still("mosaic_200x150_m0", mk_mosaic(150, 200, 22, 20), method=0, truth="mosaic_200x150_m6")

for m in range(7):
    add_still("photo_80x60_m%d" % m, mk_photo(60, 80, 30, 8), method=m, truth="photo_80x60")
add_still("photo_q100_80x60", mk_photo(60, 80, 30, 8), method=6, quality=100, truth="photo_80x60")
add_still("photo_q0_80x60", mk_photo(60, 80, 30, 8), method=0, quality=0, truth="photo_80x60")

add_still("alpha_random_64x40", mk_alpha(40, 64))
add_still("alpha_random_m6_64x40", mk_alpha(40, 64, 41), method=6)
add_still("cache_heavy_80x60", mk_cache_heavy(60, 80))

add_still("one_pixel_1x1", mk_flat(1, 1, (12, 34, 56, 255)))
add_still("one_pixel_alpha_1x1", mk_flat(1, 1, (12, 34, 56, 100)))
add_still("tall_1x37", mk_photo(37, 1, 50, 40))
add_still("wide_41x1", mk_photo(1, 41, 51, 40))
add_still("odd_13x7", mk_photo(7, 13, 52, 60))
add_still("odd_255x3", mk_photo(3, 255, 53, 60))
add_still("odd_3x255", mk_photo(255, 3, 54, 60))
add_still("odd_17x19_m6", mk_mosaic(19, 17, 55, 5), method=6)
add_still("large_1024x768", mk_large(768, 1024), method=3)

# tiles of 4 different statistics => the encoder should pick several prefix-code groups (meta prefix codes)
meta_img = rgba(256, 256)
meta_img[:128, :128, :3] = mk_photo(128, 128, 60, 3)[..., :3]
meta_img[:128, 128:, :3] = r_noise = np.random.default_rng(61).integers(0, 256, (128, 128, 3))
meta_img[128:, :128, :3] = np.array([30, 60, 90], np.uint8)
meta_img[128:, 128:, :3] = mk_palette(128, 128, 6, seed=62)[..., :3]

# ICC / EXIF / XMP carrying files (VP8X wrapper)
photo = mk_photo(24, 32, 70, 5)
icc = b"\0\0\0\x80ADBE" + bytes(range(120))
xmp = b"<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?><x:xmpmeta xmlns:x='adobe:ns:meta/'/><?xpacket end='w'?>"
add_still("meta_icc_exif_xmp_32x24", photo, kind="still", orientation=6, icc_profile=icc, exif=b"Exif\0\0" + exif_tiff(6), xmp=xmp)
add_still("meta_exif_alpha_40x30", mk_alpha(30, 40, 71), orientation=3, exif=b"Exif\0\0" + exif_tiff(3))
add_still("meta_xmp_only_16x16", mk_photo(16, 16, 72, 5), xmp=xmp)


def meta_sidecars(name):
    """Expected metadata payloads as libwebp's mux stored them (<name>.icc / .exif / .xmp)."""
    data = open(os.path.join(OUT, name + ".webp"), "rb").read()
    for fcc, ext in ((b"ICCP", "icc"), (b"EXIF", "exif"), (b"XMP ", "xmp")):
        for f, p in W.riff_chunks(data):
            if f == fcc:
                write("%s.%s" % (name, ext), p)


for n in ("meta_icc_exif_xmp_32x24", "meta_exif_alpha_40x30", "meta_xmp_only_16x16"):
    meta_sidecars(n)

# a lossy file: the decoder must refuse it cleanly (not yet supported)
lossy_src = np.array(Image.fromarray(mk_photo(32, 32, 80, 4), "RGBA"))
bl = io.BytesIO()
Image.fromarray(lossy_src, "RGBA").save(bl, format="WEBP", lossless=False, quality=80)
write("lossy_32x32.webp", bl.getvalue())
manifest.append(("lossy_32x32", "lossy", 32, 32, int(lib_decode(bl.getvalue())[1]), 1, 1, "-", "-"))
bl = io.BytesIO()
Image.fromarray(mk_photo(32, 32, 80, 4)[..., :3].copy(), "RGB").save(bl, format="WEBP", lossless=False, quality=80)
write("lossy_rgb_32x32.webp", bl.getvalue())
manifest.append(("lossy_rgb_32x32", "lossy", 32, 32, int(lib_decode(bl.getvalue())[1]), 1, 1, "-", "-"))

# --------------------------------------------------------------------------------------------------
# animations
# --------------------------------------------------------------------------------------------------

def anmf_chunk(x, y, w, h, dur, dispose_bg, no_blend, vp8l):
    flags = (1 if dispose_bg else 0) | (2 if no_blend else 0)
    hdr = struct.pack("<I", x // 2)[:3] + struct.pack("<I", y // 2)[:3] + struct.pack("<I", w - 1)[:3] \
        + struct.pack("<I", h - 1)[:3] + struct.pack("<I", dur)[:3] + bytes([flags])
    return W.make_chunk(b"ANMF", hdr + W.make_chunk(b"VP8L", vp8l))


def build_anim(canvas_w, canvas_h, frames, bg=0, loops=0, extra_flags=0, icc=None):
    """frames: list of dicts(x,y,arr,dur,dispose_bg,no_blend). arr = RGBA frame image (lossless VP8L)."""
    any_alpha = any((f["arr"][..., 3] != 255).any() for f in frames)
    flags = 0x02 | (0x10 if any_alpha else 0) | extra_flags
    vp8x = bytes([flags, 0, 0, 0]) + struct.pack("<I", canvas_w - 1)[:3] + struct.pack("<I", canvas_h - 1)[:3]
    chunks = [W.make_chunk(b"VP8X", vp8x), W.make_chunk(b"ANIM", struct.pack("<IH", bg, loops))]
    for f in frames:
        data = pil_save(f["arr"], method=int(f.get("method", 4)))
        pl = vp8l_payload(data)
        chunks.append(anmf_chunk(f["x"], f["y"], f["arr"].shape[1], f["arr"].shape[0], f["dur"], f["dispose_bg"], f["no_blend"], pl))
    return W.make_riff(chunks)


def parse_anmf(data):
    rows = []
    for fcc, p in W.riff_chunks(data):
        if fcc == b"ANMF":
            x = (p[0] | p[1] << 8 | p[2] << 16) * 2
            y = (p[3] | p[4] << 8 | p[5] << 16) * 2
            w = (p[6] | p[7] << 8 | p[8] << 16) + 1
            h = (p[9] | p[10] << 8 | p[11] << 16) + 1
            dur = p[12] | p[13] << 8 | p[14] << 16
            fl = p[15]
            rows.append((len(rows), x, y, w, h, dur, fl & 1, 0 if (fl & 2) else 1))
    return rows


def add_anim(name, data, w, h):
    frames, has_alpha, n = lib_decode(data)
    write(name + ".webp", data)
    for k, fr in enumerate(frames):
        write("%s.f%d.rgba8.gz" % (name, k), gz(fr.tobytes()))
    rows = parse_anmf(data)
    assert len(rows) == n, (name, len(rows), n)
    write(name + ".frames.tsv", ("\n".join("\t".join(str(v) for v in r) for r in rows) + "\n").encode())
    feats = []
    for fcc, p in W.riff_chunks(data):
        if fcc == b"ANMF":
            for f2, p2 in W.riff_chunks(b"RIFF\0\0\0\0WEBP" + p[16:]):
                if f2 == b"VP8L":
                    _, _, _, st = W.decode_vp8l(p2)
                    stats_all.append((name, st))
    manifest.append((name, "anim", w, h, int(has_alpha), n, 1, name, "frames=%d" % n))
    print("%-28s %6d B  %d frames" % (name, len(data), n))


def rand_rgba(h, w, seed, alpha_mode="random"):
    r = np.random.default_rng(seed)
    a = np.zeros((h, w, 4), np.uint8)
    a[..., :3] = r.integers(0, 256, (h, w, 3))
    if alpha_mode == "random":
        a[..., 3] = r.integers(0, 256, (h, w))
    elif alpha_mode == "binary":
        a[..., 3] = (r.random((h, w)) < 0.6) * 255
    elif alpha_mode == "low":
        a[..., 3] = r.integers(1, 12, (h, w))
        a[..., :3] = 255 - r.integers(0, 3, (h, w, 3))
    else:
        a[..., 3] = 255
    return a


CW, CH = 64, 48
base = mk_photo(CH, CW, 90, 5)
f_over = [
    dict(x=0, y=0, arr=base, dur=80, dispose_bg=False, no_blend=False),
    dict(x=8, y=6, arr=rand_rgba(20, 30, 91), dur=100, dispose_bg=False, no_blend=False),
    dict(x=20, y=10, arr=rand_rgba(30, 30, 92), dur=120, dispose_bg=True, no_blend=False),
    dict(x=0, y=0, arr=rand_rgba(40, 40, 93), dur=90, dispose_bg=False, no_blend=False),
    dict(x=10, y=2, arr=rand_rgba(20, 20, 94), dur=70, dispose_bg=True, no_blend=True),
    dict(x=30, y=20, arr=rand_rgba(26, 34, 95, "binary"), dur=60, dispose_bg=False, no_blend=False),
]
add_anim("anim_blend_dispose_64x48", build_anim(CW, CH, f_over), CW, CH)

# keyframe path of libwebp's anim decoder: a full-canvas frame disposed to background, then a partial-alpha frame
f_key = [
    dict(x=0, y=0, arr=rand_rgba(CH, CW, 100, "random"), dur=50, dispose_bg=True, no_blend=False),
    dict(x=6, y=4, arr=rand_rgba(24, 30, 101, "low"), dur=50, dispose_bg=False, no_blend=False),
    dict(x=0, y=0, arr=rand_rgba(CH, CW, 102, "low"), dur=50, dispose_bg=True, no_blend=False),
    dict(x=10, y=10, arr=rand_rgba(12, 16, 103, "low"), dur=50, dispose_bg=True, no_blend=False),
    dict(x=20, y=14, arr=rand_rgba(12, 16, 104, "low"), dur=50, dispose_bg=False, no_blend=False),
    dict(x=2, y=2, arr=rand_rgba(30, 30, 105, "random"), dur=50, dispose_bg=False, no_blend=False),
]
add_anim("anim_keyframe_blend_64x48", build_anim(CW, CH, f_key), CW, CH)

# dispose none + opaque sub-rectangle frames (typical GIF-like animation), nonzero ANIM background colour (ignored by libwebp)
f_opaque = []
cur = mk_photo(CH, CW, 110, 3)
f_opaque.append(dict(x=0, y=0, arr=cur, dur=40, dispose_bg=False, no_blend=True))
for k in range(1, 5):
    f_opaque.append(dict(x=2 * k * 3, y=2 * k, arr=rand_rgba(16 + k, 18 + 2 * k, 110 + k, "opaque"), dur=40 + k, dispose_bg=False, no_blend=False))
add_anim("anim_opaque_rects_bgcolor_64x48", build_anim(CW, CH, f_opaque, bg=0xFF3366CC, loops=3), CW, CH)

# 1-pixel wide / tall frames, odd sizes
f_thin = [
    dict(x=0, y=0, arr=rand_rgba(9, 7, 120, "binary"), dur=30, dispose_bg=False, no_blend=True),
    dict(x=2, y=0, arr=rand_rgba(9, 1, 121, "random"), dur=30, dispose_bg=False, no_blend=False),
    dict(x=0, y=4, arr=rand_rgba(1, 7, 122, "random"), dur=30, dispose_bg=True, no_blend=False),
    dict(x=4, y=2, arr=rand_rgba(1, 1, 123, "random"), dur=30, dispose_bg=False, no_blend=False),
]
add_anim("anim_thin_frames_7x9", build_anim(7, 9, f_thin), 7, 9)

# libwebp's own animation encoder (sub-rectangles, blend and dispose chosen by libwebp)
shapes = []
for k in range(5):
    fr = mk_photo(40, 56, 130, 2).copy()
    fr[8 + 4 * k:20 + 4 * k, 6 + 8 * k:20 + 8 * k, :3] = np.array([250, 20 + 40 * k, 30], np.uint8)
    shapes.append(Image.fromarray(fr, "RGBA"))
b = io.BytesIO()
shapes[0].save(b, format="WEBP", save_all=True, append_images=shapes[1:], duration=[100, 80, 60, 40, 20], loop=0,
               lossless=True, exact=True, minimize_size=False, kmin=0, kmax=0)
add_anim("anim_libwebp_56x40", b.getvalue(), 56, 40)
shapes = []
for k in range(4):
    fr = np.zeros((36, 44, 4), np.uint8)
    fr[4 + 5 * k:16 + 5 * k, 3 + 6 * k:18 + 6 * k] = (200, 40 * k, 90, 255)
    fr[20:30, 10:30] = (30, 60, 200, 128 + 20 * k)
    shapes.append(Image.fromarray(fr, "RGBA"))
b = io.BytesIO()
shapes[0].save(b, format="WEBP", save_all=True, append_images=shapes[1:], duration=50, loop=2,
               lossless=True, exact=True, minimize_size=True)
add_anim("anim_libwebp_alpha_44x36", b.getvalue(), 44, 36)

# an animation whose first frame is lossy: the decoder must refuse it cleanly
b = io.BytesIO()
sh = [Image.fromarray(mk_photo(32, 32, 140 + i, 4), "RGBA") for i in range(3)]
sh[0].save(b, format="WEBP", save_all=True, append_images=sh[1:], duration=50, lossless=False, quality=70)
write("anim_lossy_32x32.webp", b.getvalue())
manifest.append(("anim_lossy_32x32", "anim_lossy", 32, 32, int(lib_decode(b.getvalue())[1]), lib_decode(b.getvalue())[2], 1, "-", "-"))

# --------------------------------------------------------------------------------------------------
# sources for the ENCODER tests (libwebp compresses the same image, for the size comparison)
# --------------------------------------------------------------------------------------------------

add_source("enc_flat_32x24", mk_flat(24, 32, (200, 30, 90, 255)))
add_source("enc_two_color_70x45", mk_palette(45, 70, 2, seed=211))
add_source("enc_pal4_64x40", mk_palette(40, 64, 4, seed=213))
add_source("enc_pal16_61x33", mk_palette(33, 61, 16, seed=215))
add_source("enc_pal256_90x70", mk_palette(70, 90, 256, seed=219))
add_source("enc_pal12_alpha_50x40", pa)
add_source("enc_photo_80x60", mk_photo(60, 80, 230, 8))
add_source("enc_noise_64x48", np.dstack([np.random.default_rng(231).integers(0, 256, (48, 64, 3)).astype(np.uint8), np.full((48, 64), 255, np.uint8)]))
add_source("enc_grad_h_128x32", np.dstack([np.tile(np.arange(128, dtype=np.uint8) * 2, (32, 1))] * 3 + [np.full((32, 128), 255, np.uint8)]))
add_source("enc_mosaic_96x72", mk_mosaic(72, 96, 221))
add_source("enc_alpha_64x40", mk_alpha(40, 64, 241))
add_source("enc_cache_heavy_80x60", mk_cache_heavy(60, 80, 245))
add_source("enc_one_pixel_1x1", mk_flat(1, 1, (12, 34, 56, 255)))
add_source("enc_tall_1x37", mk_photo(37, 1, 250, 40))
add_source("enc_wide_41x1", mk_photo(1, 41, 251, 40))
add_source("enc_odd_13x7", mk_photo(7, 13, 252, 60))
add_source("enc_large_320x240", mk_large(240, 320, 260))
add_source("enc_meta_groups_128x128", meta_img[64:192, 64:192].copy())

# libwebp's size for each source (reference ratio), written beside the manifest
with open(os.path.join(OUT, "sources", "libwebp_sizes.tsv"), "w") as f:
    for name, kind, w, h, ha, fr, ori, tr, ft in manifest:
        if kind != "source":
            continue
        with gzip.open(os.path.join(SRC, name + ".rgba8.gz")) as g:
            arr = np.frombuffer(g.read(), np.uint8).reshape(h, w, 4)
        sizes = []
        for m in (0, 4, 6):
            sizes.append(len(pil_save(arr, method=m)))
        f.write("%s\t%d\t%d\t%d\n" % ((name,) + tuple(sizes)))

# --------------------------------------------------------------------------------------------------
# manifest + coverage report
# --------------------------------------------------------------------------------------------------

with open(os.path.join(OUT, "manifest.tsv"), "w") as f:
    for row in manifest:
        f.write("\t".join(str(v) for v in row) + "\n")

union = lambda attr: set().union(*[getattr(s, attr) for _, s in stats_all])  # noqa: E731
kinds = set()
for _, s in stats_all:
    for k, b in s.transforms:
        kinds.add(k)
modes = union("predictor_modes")
bundle = {s.bundle_bits for _, s in stats_all if s.palette_size}
print()
print("transforms used          :", sorted(kinds), "(want G P C I)")
print("predictor modes used     :", "".join("%x" % m for m in sorted(modes)), "(want 0-d)")
print("colour cache             :", any(s.cache_bits for _, s in stats_all))
print("meta prefix groups > 1   :", any(s.groups > 1 for _, s in stats_all), "max groups", max(s.groups for _, s in stats_all))
print("bundling width_bits      :", sorted(bundle), "(want 0 1 2 3)")
print("short distance codes     :", len(union("plane_codes")), "of 120")
print("far distances            :", any(s.far_dists for _, s in stats_all))
print("overlapping copies       :", any(s.has_overlap_copy for _, s in stats_all))
print("prefix code kinds        :", sorted(union("code_kinds")))
total = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(OUT) for fn in fns)
print("fixture folder bytes     :", total)
missing = []
if kinds != {"G", "P", "C", "I"}: missing.append("transforms")
if modes != set(range(14)): missing.append("predictor modes")
if not any(s.cache_bits for _, s in stats_all): missing.append("colour cache")
if not any(s.groups > 1 for _, s in stats_all): missing.append("meta groups")
if not {0, 1, 2, 3} <= bundle: missing.append("bundling")
if not any(s.far_dists for _, s in stats_all): missing.append("far dists")
if not {"simple1", "simple2", "normal"} <= union("code_kinds"): missing.append("code kinds")
if missing:
    print("COVERAGE GAPS:", missing)
    sys.exit(1)
print("coverage OK")
