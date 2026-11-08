"""Fixtures for the ANIMATED lossy WebP encoder (WebPLossyEncoderGpu.EncodeAnimationAsync) and the filter_strength option
of the still lossy encoder.

TestData/WebPAnimLossy:
  <source>.frames.gz  the source frames, full canvas RGBA8 each, concatenated (cases with the same frames share one)
  <name>.webp       Pillow's animated lossy save of those frames (libwebp 1.6.0 WebPAnimEncoder) = the expected bytes
  manifest.tsv      name, source, width, height, frames, durations (comma list), quality, method, kmin, kmax (-999 =
                    Pillow's default), exif (0/1), loop, background (hex 0xAARRGGBB)
  The EXIF payload of a case with exif=1 is comfy_exif() below (Pillow Image.Exif.tobytes(), what ComfyUI passes), stored
  as <name>.exif.

TestData/WebPLossyEncFilter:
  <name>.yuv.gz     libwebp's own imported YUV 4:2:0 planes of the source (WebPPictureImportRGBA), Y then U then V
  <name>.webp       libwebp WebPEncode (the webp cffi package, same bytes as Pillow's libwebp 1.6.0 for the default
                    config, checked below) with WebPConfigInit defaults + quality + method + filter_strength
  manifest.tsv      name, width, height, quality, method, filter_strength

Truth never comes from our code.

Usage: python -I make_webp_anim_lossy_fixtures.py   (venv with Pillow, numpy, webp)
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
import webp  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData"))
ANIM = os.path.join(ROOT, "WebPAnimLossy")
FILT = os.path.join(ROOT, "WebPLossyEncFilter")
os.makedirs(ANIM, exist_ok=True)
os.makedirs(FILT, exist_ok=True)


def photo(w, h, seed):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.zeros((h, w, 3), np.float64)
    a[..., 0] = 128 + 100 * np.sin(xx / 9.0 + seed) * np.cos(yy / 13.0)
    a[..., 1] = 128 + 80 * np.sin(yy / 7.0 - seed) + 30 * np.sin(xx / 4.0)
    a[..., 2] = 255 * ((xx / max(w, 1) + yy / max(h, 1)) / 2)
    a += r.normal(0, 3, a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


def opaque(rgb):
    return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])


def sprite(size, seed):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    m = (xx - c) ** 2 + (yy - c) ** 2 <= (size / 2) ** 2
    s = np.zeros((size, size, 4), np.uint8)
    s[..., :3] = r.integers(0, 256, (size, size, 3))
    s[m, 3] = 255
    return s, m


def comfy_exif():
    ex = Image.Exif()
    ex[0x0110] = 'prompt:{"3": {"inputs": {"seed": 42, "steps": 20}, "class_type": "KSampler"}}'
    ex[0x010F] = 'workflow:{"nodes": [], "links": []}'
    return ex.tobytes()


cases = []


def case(name, frames, durations, quality=80, method=4, kmin=None, kmax=None, exif=False, loop=0, background=(0, 0, 0, 0)):
    cases.append((name, frames, durations, quality, method, kmin, kmax, exif, loop, background))


# 1. moving noisy patch over a noisy background (no flattening: libwebp writes no ALPH)
rng = np.random.default_rng(1)
base = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
fr = []
for k in range(8):
    f = base.copy()
    f[10:30, 8 + 4 * k:40 + 4 * k] = rng.integers(0, 256, (20, 32, 3), dtype=np.uint8)
    fr.append(opaque(f))
case("patch_q80_m4", fr, 125)
case("patch_q60_m3", fr, 100, quality=60, method=3)

# 2. sprite moving over a photo (static background: blended frames get flattened blocks -> ALPH in libwebp's file)
bg = photo(128, 96, 2)
spr, m = sprite(20, 3)
fr = []
for k in range(12):
    f = opaque(bg.copy())
    x, y = 6 + 8 * k, 10 + 5 * (k % 6)
    reg = f[y:y + 20, x:x + 20]
    reg[m] = spr[m]
    fr.append(f)
case("sprite_q80_m4", fr, 80)
case("sprite_q80_m4_nokey", fr, 80, kmin=0, kmax=0)
case("sprite_q80_m4_allkey", fr, 80, kmin=0, kmax=1)
case("sprite_q90_m3_k2_6", fr, [40, 80, 120, 40, 80, 120, 40, 80, 120, 40, 80, 120], quality=90, method=3, kmin=2, kmax=6)

# 3. repeated frames: skipped, their durations merged into the previous frame
fr = [opaque(photo(64, 48, 4))] * 3 + [opaque(photo(64, 48, 5))] * 2 + [opaque(photo(64, 48, 4))]
case("repeats_q75_m4", fr, [100, 200, 300, 50, 60, 70], quality=75)

# 4. every frame the same -> one frame -> a still file (with EXIF: VP8X + EXIF)
fr = [opaque(photo(48, 40, 6))] * 3
case("still_q80_m4", fr, 100)
case("still_exif_q80_m4", fr, 100, exif=True)

# 5. transparent sprite moving on a transparent canvas (alpha in the source; dispose-to-background decisions)
fr = []
spr, m = sprite(16, 7)
for k in range(8):
    f = np.zeros((48, 80, 4), np.uint8)
    x, y = 4 + 7 * k, 6 + 3 * (k % 4)
    f[y:y + 16, x:x + 16][m] = spr[m]
    fr.append(f)
case("alpha_sprite_q80_m4_nokey", fr, 100, kmin=0, kmax=0)
case("alpha_sprite_q80_m4", fr, 100)

# 6. a lone frame that does not cover the canvas (transparent border): OptimizeSingleFrame decides
f = np.zeros((40, 56, 4), np.uint8)
f[8:30, 10:44, :3] = photo(34, 22, 8)
f[8:30, 10:44, 3] = 255
case("lone_subrect_q80_m4", [f, f.copy()], 150)

# 7. odd canvas, small scattered changes, varied durations, quality 50 method 3
rng = np.random.default_rng(9)
cur = opaque(photo(37, 23, 9))
fr = []
for k in range(10):
    cur = cur.copy()
    y, x = rng.integers(0, 23), rng.integers(0, 37)
    cur[y:y + 3, x:x + 5, :3] = rng.integers(0, 256, 3)
    fr.append(cur)
case("odd_q50_m3", fr, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], quality=50, method=3)

# 8. scene cuts: every frame unrelated (full-canvas rectangles every time)
rng = np.random.default_rng(10)
fr = [opaque(photo(48, 32, 20 + k)) for k in range(7)]
case("cuts_q80_m4", fr, 125)

# 9. 1x1 canvas
case("tiny_q80_m4", [opaque(np.array([[[200, 10, 10]]], np.uint8)), opaque(np.array([[[10, 200, 10]]], np.uint8)),
                     opaque(np.array([[[10, 10, 200]]], np.uint8))], 100)

# 10. ComfyUI-like: hue-shifting gradient (changes everywhere), EXIF prompt / workflow, loop 0
yy, xx = np.mgrid[0:72, 0:112].astype(np.float64)
fr = []
for k in range(9):
    a = np.zeros((72, 112, 3))
    a[..., 0] = 128 + 120 * np.sin(xx / 17.0 + k * 0.4)
    a[..., 1] = 128 + 120 * np.sin(yy / 11.0 - k * 0.3)
    a[..., 2] = 128 + 100 * np.cos((xx + yy) / 23.0 + k * 0.2)
    fr.append(opaque(np.clip(a, 0, 255).astype(np.uint8)))
case("comfy_gradient_exif_q80_m4", fr, 125, exif=True)

# 11. durations that overflow 24 bits when merged: libwebp splits with its 1x1 transparent frame
fr = [opaque(photo(32, 32, 30))] * 3 + [opaque(photo(32, 32, 31))]
case("longdur_q80_m4", fr, [(1 << 24) - 1, (1 << 24) - 1, 5, 100])

# 12. larger picture, object over photo, loop count and background colour
bg = photo(256, 192, 40)
spr, m = sprite(40, 41)
fr = []
for k in range(6):
    f = opaque(bg.copy())
    x, y = 10 + 31 * k, 30 + 9 * (k % 5)
    f[y:y + 40, x:x + 40][m] = spr[m]
    fr.append(f)
case("big_q75_m4", fr, 66, quality=75, loop=3, background=(10, 20, 30, 255))

# 13. sensor-like jitter below / around the lossy tolerance (QualityToMaxDiff: q80 -> 4, q30 -> 15, q95 -> 2) plus a moving
# sprite: the lossy rectangle is smaller than the lossless one, blocks flatten by SIMILARITY, near-repeats are skipped
rng = np.random.default_rng(60)
bg = photo(96, 72, 61)
spr, m = sprite(14, 62)
fr = []
for k in range(10):
    jitter = rng.integers(-3, 4, bg.shape)
    f = opaque(np.clip(bg.astype(int) + jitter, 0, 255).astype(np.uint8))
    if k not in (3, 4):   # frames 3 and 4: jitter only (near-repeats of the previous canvas)
        x, y = 4 + 8 * k, 8 + 4 * (k % 5)
        f[y:y + 14, x:x + 14][m] = spr[m]
    fr.append(f)
case("jitter_q80_m4", fr, 100)
case("jitter_q80_m4_nokey", fr, 100, kmin=0, kmax=0)
case("jitter_q30_m4_nokey", fr, 100, quality=30, kmin=0, kmax=0)
case("jitter_q95_m3_nokey", fr, 100, quality=95, method=3, kmin=0, kmax=0)
# bigger jitter (up to 10): similar at q30 only
fr2 = []
for k in range(6):
    jitter = rng.integers(-10, 11, bg.shape)
    fr2.append(opaque(np.clip(bg.astype(int) + jitter, 0, 255).astype(np.uint8)))
case("jitter10_q30_m4", fr2, 100, quality=30)
case("jitter10_q80_m4", fr2, 100)

# 14. semi-transparent layer with jitter: PixelsAreSimilar scales the colour difference by the destination alpha
fr = []
base_a = np.zeros((40, 64, 4), np.uint8)
base_a[..., :3] = photo(64, 40, 70)
base_a[..., 3] = np.tile(np.linspace(0, 255, 64).astype(np.uint8), (40, 1))
for k in range(7):
    f = base_a.copy()
    f[..., :3] = np.clip(f[..., :3].astype(int) + rng.integers(-6, 7, f[..., :3].shape), 0, 255).astype(np.uint8)
    f[10:20, 5 + 6 * k:15 + 6 * k, 3] = 255
    fr.append(f)
case("semi_alpha_jitter_q80_m4_nokey", fr, 100, kmin=0, kmax=0)
case("semi_alpha_jitter_q50_m4_nokey", fr, 100, quality=50, kmin=0, kmax=0)

# 15. a faint layer (alpha 40) with slight colour jitter under a moving opaque sprite: every non-opaque pixel is
# SIMILAR to the previous frame (lossy blending possible) but not EQUAL (lossless blending impossible)
fr = []
faint = np.zeros((40, 56, 4), np.uint8)
faint[..., :3] = photo(56, 40, 90)
faint[..., 3] = 40
spr, m = sprite(10, 91)
stamped = faint.copy()
for k in range(6):
    # opaque sprites are only ever ADDED (stamped, never erased), so no pixel goes from opaque back to faint
    x, y = 3 + 7 * k, 5 + 3 * (k % 3)
    stamped[y:y + 10, x:x + 10][m] = spr[m]
    f = stamped.copy()
    faint_px = f[..., 3] != 255
    jit = np.clip(f[..., :3].astype(int) + rng.integers(-3, 4, f[..., :3].shape), 0, 255).astype(np.uint8)
    f[..., :3][faint_px] = jit[faint_px]
    fr.append(f)
case("lowalpha_jitter_q80_m4", fr, 100)
case("lowalpha_jitter_q80_m4_nokey", fr, 100, kmin=0, kmax=0)

manifest = []
sources = {}
for (name, frames, durations, q, meth, kmin, kmax, exif, loop, background) in cases:
    h, w = frames[0].shape[:2]
    n = len(frames)
    durs = durations if isinstance(durations, list) else [durations] * n
    # opaque frames go in as RGB images, as ComfyUI gives them (Pillow imports them with alpha 255)
    imgs = [Image.fromarray(np.ascontiguousarray(f), "RGBA") if (f[..., 3] != 255).any()
            else Image.fromarray(np.ascontiguousarray(f[..., :3]), "RGB") for f in frames]
    kw = dict(save_all=True, append_images=imgs[1:], duration=durs, lossless=False, quality=q, method=meth,
              loop=loop, background=background)
    if kmin is not None:
        kw["kmin"] = kmin
    if kmax is not None:
        kw["kmax"] = kmax
    exif_bytes = comfy_exif() if exif else None
    if exif_bytes is not None:
        kw["exif"] = exif_bytes
    buf = io.BytesIO()
    imgs[0].save(buf, "WEBP", **kw)
    data = buf.getvalue()
    with open(os.path.join(ANIM, name + ".webp"), "wb") as fh:
        fh.write(data)
    raw = b"".join(np.ascontiguousarray(f).tobytes() for f in frames)
    source = sources.get(raw)
    if source is None:
        source = sources[raw] = name
        with open(os.path.join(ANIM, source + ".frames.gz"), "wb") as fh:
            fh.write(gzip.compress(raw, mtime=0))
    if exif_bytes is not None:
        with open(os.path.join(ANIM, name + ".exif"), "wb") as fh:
            fh.write(exif_bytes)
    bgv = (background[3] << 24) | (background[0] << 16) | (background[1] << 8) | background[2]
    kmin_s = -999 if kmin is None else kmin
    kmax_s = -999 if kmax is None else kmax
    manifest.append(f"{name}\t{source}\t{w}\t{h}\t{n}\t{','.join(map(str, durs))}\t{q}\t{meth}\t{kmin_s}\t{kmax_s}\t{1 if exif else 0}\t{loop}\t{bgv:08X}")
    print(name, w, h, n, len(data))
with open(os.path.join(ANIM, "manifest.tsv"), "w", newline="\n") as fh:
    fh.write("\n".join(manifest) + "\n")

# ---- still files with filter_strength (libwebp through cffi), checked against Pillow for the default strength
fman = []
srcs = [("photo", photo(97, 61, 50)), ("photo_small", photo(17, 13, 51)), ("noise", np.random.default_rng(52).integers(0, 256, (40, 64, 3), dtype=np.uint8))]
for sname, rgb in srcs:
    h, w = rgb.shape[:2]
    rgba = opaque(rgb)
    Y, U, V, _ = libwebp_picture.import_rgba(rgba)
    with open(os.path.join(FILT, sname + ".yuv.gz"), "wb") as fh:
        fh.write(gzip.compress(Y.tobytes() + U.tobytes() + V.tobytes(), mtime=0))
    for q in (30, 80, 95):
        for meth in (3, 4):
            for fs in (0, 25, 60, 100):
                pic = webp.WebPPicture.from_numpy(rgba)
                cfg = webp.WebPConfig.new(quality=q, lossless=False)
                cfg.ptr.method = meth
                cfg.ptr.filter_strength = fs
                assert webp.lib.WebPValidateConfig(cfg.ptr)
                data = bytes(pic.encode(cfg).buffer())
                if fs == 60:
                    b = io.BytesIO()
                    Image.fromarray(rgb).save(b, "WEBP", quality=q, method=meth)
                    assert b.getvalue() == data, f"webp cffi and Pillow disagree for {sname} q{q} m{meth}"
                name = f"{sname}_q{q}_m{meth}_f{fs}"
                with open(os.path.join(FILT, name + ".webp"), "wb") as fh:
                    fh.write(data)
                fman.append(f"{name}\t{sname}\t{w}\t{h}\t{q}\t{meth}\t{fs}")
with open(os.path.join(FILT, "manifest.tsv"), "w", newline="\n") as fh:
    fh.write("\n".join(fman) + "\n")
print(len(fman), "filter fixtures")

# ---- still lossy files with ICCP / EXIF / XMP (Pillow's still save muxes them with libwebp's WebPMux)
META = os.path.join(ROOT, "WebPLossyMeta")
os.makedirs(META, exist_ok=True)
icc = bytes(range(256)) * 2 + b"spawn-icc"          # opaque payload: the muxer copies it, nobody parses it here
xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF/></x:xmpmeta>'
exif_full = comfy_exif()                              # "Exif\0\0" + TIFF, what ComfyUI hands Pillow
exif_stripped = exif_full[6:]                         # what Pillow's still save passes to libwebp
mman = []
srcs = {"meta_photo": opaque(photo(61, 43, 80))}
a = opaque(photo(50, 34, 81))
a[..., 3] = np.tile(np.linspace(0, 255, 50).astype(np.uint8), (34, 1))
srcs["meta_alpha"] = a
for name, kw, src in [
    ("still_exif", dict(exif=exif_full), "meta_photo"),
    ("still_icc_exif_xmp", dict(icc_profile=icc, exif=exif_full, xmp=xmp), "meta_photo"),
    ("still_alpha_exif_xmp", dict(exif=exif_full, xmp=xmp), "meta_alpha"),
]:
    rgba = srcs[src]
    img = Image.fromarray(rgba, "RGBA") if (rgba[..., 3] != 255).any() else Image.fromarray(np.ascontiguousarray(rgba[..., :3]), "RGB")
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=80, method=4, **kw)
    with open(os.path.join(META, name + ".webp"), "wb") as fh:
        fh.write(buf.getvalue())
    if not os.path.exists(os.path.join(META, src + ".rgba.gz")):
        with open(os.path.join(META, src + ".rgba.gz"), "wb") as fh:
            fh.write(gzip.compress(np.ascontiguousarray(rgba).tobytes(), mtime=0))
    h, w = rgba.shape[:2]
    mman.append(f"{name}\t{src}\t{w}\t{h}\t{1 if 'icc_profile' in kw else 0}\t{1 if 'exif' in kw else 0}\t{1 if 'xmp' in kw else 0}")
    print(name, len(buf.getvalue()))
for nm, data in (("icc", icc), ("exif", exif_stripped), ("xmp", xmp)):
    with open(os.path.join(META, nm + ".bin"), "wb") as fh:
        fh.write(data)
with open(os.path.join(META, "manifest.tsv"), "w", newline="\n") as fh:
    fh.write("\n".join(mman) + "\n")

# ---- animated LOSSLESS (Pillow save_all=True, lossless=True: ComfyUI's Save Animated WEBP default), same sources.
# TestData/WebPAnimLossless: <name>.webp (Pillow's file) + manifest.tsv in the lossy manifest's format; the frames are
# the lossy fixtures' <source>.frames.gz.
LL = os.path.join(ROOT, "WebPAnimLossless")
os.makedirs(LL, exist_ok=True)
lman = []
for line in manifest:
    c = line.split("\t")
    name, source = c[0], c[1]
    w, h, n = int(c[2]), int(c[3]), int(c[4])
    durs = [int(x) for x in c[5].split(",")]
    kmin, kmax = int(c[8]), int(c[9])
    if int(c[7]) != 4 or name.startswith("jitter10"):
        continue                                     # lossless: method 4 / quality 80 (ComfyUI's default)
    raw = gzip.decompress(open(os.path.join(ANIM, source + ".frames.gz"), "rb").read())
    frames = [np.frombuffer(raw, np.uint8, w * h * 4, k * w * h * 4).reshape(h, w, 4) for k in range(n)]
    imgs = [Image.fromarray(np.ascontiguousarray(f), "RGBA") if (f[..., 3] != 255).any()
            else Image.fromarray(np.ascontiguousarray(f[..., :3]), "RGB") for f in frames]
    kw = dict(save_all=True, append_images=imgs[1:], duration=durs, lossless=True, quality=80, method=4,
              loop=int(c[11]), background=tuple(int(c[12][i:i + 2], 16) for i in (2, 4, 6, 0)))
    if kmin != -999: kw["kmin"] = kmin
    if kmax != -999: kw["kmax"] = kmax
    if c[10] == "1": kw["exif"] = open(os.path.join(ANIM, name + ".exif"), "rb").read()
    buf = io.BytesIO()
    imgs[0].save(buf, "WEBP", **kw)
    with open(os.path.join(LL, name + ".webp"), "wb") as fh:
        fh.write(buf.getvalue())
    lman.append("\t".join([name, source] + c[2:6] + ["80", "4"] + c[8:]))
    print("lossless", name, len(buf.getvalue()))
with open(os.path.join(LL, "manifest.tsv"), "w", newline="\n") as fh:
    fh.write("\n".join(lman) + "\n")
