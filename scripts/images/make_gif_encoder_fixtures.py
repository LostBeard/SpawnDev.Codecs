#!/usr/bin/env python3
"""
Generates the GIF ENCODER test fixtures in SpawnDev.Codecs.Demo.Shared/TestData/GifEnc/ :

  <name>.src.rgba8       the source frames, RGBA8, back to back (frame k at k * W * H * 4)
  <name>.expected.gif    the file GifEncoderGpu must produce, BYTE FOR BYTE
  <name>.canvas.rgba8    what every decoder must show for each frame of that file (palette colour, alpha 255; transparent
                         index -> 0, 0, 0, 0); checked against Pillow's decode of expected.gif at generation time
  <name>.json            width, height, frames and the encoder options (maxColors, dither, alphaThreshold, loop, delaysMs)

This script is an INDEPENDENT implementation (plain Python / numpy) of the algorithms documented in
SpawnDev.Codecs/Images/Gif/GifEncoderKernels.cs:
  * exact palette: the sorted set of opaque 0xRRGGBB colours of all frames when it fits maxColors (minus one index
    reserved for transparency when any pixel is transparent);
  * otherwise per frame: exact if it fits, else median cut (5-5-5 histogram bins; split the box with the highest
    (population >> s) * longest extent at the population median of its longest axis; palette entry = pixel weighted mean
    colour of the box; entries sorted ascending);
  * nearest colour by squared RGB distance (lowest index on a tie), optional 8x8 Bayer ordered dither (+-16 levels) or
    Floyd-Steinberg (pull form, weights 7/3/5/1, (sum + 8) >> 4);
  * LZW in segments of 32768 pixels (clear code between segments, end code after the last), min code size
    max(2, log2(table size)), 255 byte sub-blocks, table sizes padded with zeros, GCE on every frame of an animation or
    when anything is transparent (transparent index = number of colours), NETSCAPE2.0 loop extension for animations.
Every expected.gif is also opened with Pillow and each frame compared with the canvas: a file the oracle cannot read, or
reads differently, aborts the run.

Run:   <venv>/python -I scripts/images/make_gif_encoder_fixtures.py        (needs numpy + pillow)
"""
import io
import json
import math
import os
import struct
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "GifEnc")
PHOTO = os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "bbb_first_frame_visual.png")
os.makedirs(OUT, exist_ok=True)

SEG_PIXELS = 1 << 15


# ---------------------------------------------------------------- LZW (segmented)

def lzw_codes(idx, mcs):
    """(code, width) pairs of the whole LZW stream of the index sequence, segmented like the encoder kernel."""
    clear = 1 << mcs
    eoi = clear + 1
    out = []
    n = len(idx)
    segs = (n + SEG_PIXELS - 1) // SEG_PIXELS
    for s in range(segs):
        seq = [int(v) for v in idx[s * SEG_PIXELS:(s + 1) * SEG_PIXELS]]
        table = {}
        nxt = eoi + 1
        width = mcs + 1
        if s == 0:
            out.append((clear, width))
        prefix = seq[0]
        for k in seq[1:]:
            key = (prefix, k)
            c = table.get(key)
            if c is not None:
                prefix = c
                continue
            out.append((prefix, width))
            table[key] = nxt
            nxt += 1
            if nxt > (1 << width) and width < 12:
                width += 1
            if nxt == 4096:
                out.append((clear, width))
                table = {}
                nxt = eoi + 1
                width = mcs + 1
            prefix = k
        out.append((prefix, width))
        if nxt < 4096:
            nxt += 1
            if nxt > (1 << width) and width < 12:
                width += 1
        out.append((eoi if s == segs - 1 else clear, width))
    return out


def pack_codes(codes):
    acc = 0
    n = 0
    out = bytearray()
    for c, wd in codes:
        acc |= c << n
        n += wd
        while n >= 8:
            out.append(acc & 255)
            acc >>= 8
            n -= 8
    if n:
        out.append(acc & 255)
    return bytes(out)


def sub_blocks(data):
    out = bytearray()
    for i in range(0, len(data), 255):
        chunk = data[i:i + 255]
        out.append(len(chunk))
        out += chunk
    out.append(0)
    return bytes(out)


# ---------------------------------------------------------------- palette

def pack_rgb(c):
    return (c[0] << 16) | (c[1] << 8) | c[2]


def exact_palette(frames_rgba, thr):
    cols = set()
    for fr in frames_rgba:
        px = fr.reshape(-1, 4)
        op = px[px[:, 3] >= thr]
        for r, g, b in {(int(r), int(g), int(b)) for r, g, b in op[:, :3]}:
            cols.add((r, g, b))
    return sorted(cols, key=pack_rgb)


def median_cut(opaque_rgb, K):
    """opaque_rgb: (N,3) uint8 of the opaque pixels. Returns the sorted palette (list of (r,g,b))."""
    r = opaque_rgb[:, 0].astype(np.int64); g = opaque_rgb[:, 1].astype(np.int64); b = opaque_rgb[:, 2].astype(np.int64)
    bins_all = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)
    cnt = np.bincount(bins_all, minlength=32768)
    sr = np.bincount(bins_all, weights=(r & 7), minlength=32768).astype(np.int64)
    sg = np.bincount(bins_all, weights=(g & 7), minlength=32768).astype(np.int64)
    sb = np.bincount(bins_all, weights=(b & 7), minlength=32768).astype(np.int64)
    bins = [i for i in range(32768) if cnt[i] != 0]
    total = int(cnt.sum())
    sh = 0
    while (total >> sh) >= (1 << 26):
        sh += 1

    def chan(bid, a):
        return (bid >> (10 - 5 * a)) & 31

    def stats(lo, hi):
        c = 0
        mn = [99, 99, 99]
        mx = [-1, -1, -1]
        for bid in bins[lo:hi]:
            c += int(cnt[bid])
            for a in range(3):
                v = chan(bid, a)
                mn[a] = min(mn[a], v)
                mx[a] = max(mx[a], v)
        return c, mn, mx

    boxes = []
    st = []
    if bins:
        boxes.append((0, len(bins)))
        st.append(stats(0, len(bins)))
    while boxes and len(boxes) < K:
        best, best_score = -1, -1
        for i, (lo, hi) in enumerate(boxes):
            if hi - lo < 2:
                continue
            c, mn, mx = st[i]
            ext = max(mx[a] - mn[a] for a in range(3))
            score = (c >> sh) * ext
            if score > best_score:
                best_score, best = score, i
        if best < 0:
            break
        lo, hi = boxes[best]
        c, mn, mx = st[best]
        ext = [mx[a] - mn[a] for a in range(3)]
        axis = 0
        if ext[1] > ext[axis]:
            axis = 1
        if ext[2] > ext[axis]:
            axis = 2
        h1 = [0] * 32
        for bid in bins[lo:hi]:
            h1[chan(bid, axis)] += int(cnt[bid])
        cum = 0
        split = mn[axis]
        for v in range(mn[axis], mx[axis] + 1):
            cum += h1[v]
            if cum * 2 >= c:
                split = v
                break
        if split >= mx[axis]:
            split = mx[axis] - 1
        left = [bid for bid in bins[lo:hi] if chan(bid, axis) <= split]
        right = [bid for bid in bins[lo:hi] if chan(bid, axis) > split]
        bins[lo:hi] = left + right
        mid = lo + len(left)
        boxes[best] = (lo, mid)
        st[best] = stats(lo, mid)
        boxes.append((mid, hi))
        st.append(stats(mid, hi))
    pal = []
    for lo, hi in boxes:
        C = sum(int(cnt[bid]) for bid in bins[lo:hi])
        s = 0
        while (C >> s) >= (1 << 23):
            s += 1
        W = 0
        acc = [0, 0, 0]
        for bid in bins[lo:hi]:
            n = int(cnt[bid])
            w = n >> s
            res = (int(sr[bid]), int(sg[bid]), int(sb[bid]))
            for a in range(3):
                col = (chan(bid, a) << 3) + (res[a] + n // 2) // n
                acc[a] += w * col
            W += w
        pal.append(tuple((acc[a] + W // 2) // W for a in range(3)))
    return sorted(pal, key=pack_rgb)


# ---------------------------------------------------------------- mapping

def bayer_recursive(n):
    m = np.array([[0, 2], [3, 1]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return m


BAYER = bayer_recursive(8)


def bayer8(x, y):
    xv, yv = x & 7, y & 7
    xy = xv ^ yv
    return ((xy & 1) << 5) | ((yv & 1) << 4) | ((xy & 2) << 2) | ((yv & 2) << 1) | ((xy & 4) >> 1) | ((yv & 4) >> 2)


for _y in range(8):
    for _x in range(8):
        assert bayer8(_x, _y) == BAYER[_y, _x], "Bayer formula mismatch"


def nearest(pal_arr, r, g, b):
    d = (pal_arr[:, 0] - r) ** 2 + (pal_arr[:, 1] - g) ** 2 + (pal_arr[:, 2] - b) ** 2
    return int(np.argmin(d))      # first minimum = lowest index


def map_frame(frame, palette, exact, trans_idx, thr, dither):
    h, w, _ = frame.shape
    idx = np.zeros((h, w), dtype=np.int64)
    pal_arr = np.array(palette, dtype=np.int64).reshape(-1, 3)
    lookup = {pack_rgb(c): i for i, c in enumerate(palette)}
    err = np.zeros((h, w, 3), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            r, g, b, a = (int(v) for v in frame[y, x])
            if a < thr:
                idx[y, x] = trans_idx
                continue
            if exact:
                idx[y, x] = lookup[pack_rgb((r, g, b))]
                continue
            if dither == 1:
                t = ((2 * bayer8(x, y) - 63) * 16) >> 6
                r, g, b = (min(255, max(0, v + t)) for v in (r, g, b))
            elif dither == 2:
                acc = np.zeros(3, dtype=np.int64)
                if x > 0:
                    acc += 7 * err[y, x - 1]
                if y > 0:
                    if x + 1 < w:
                        acc += 3 * err[y - 1, x + 1]
                    acc += 5 * err[y - 1, x]
                    if x > 0:
                        acc += err[y - 1, x - 1]
                r, g, b = (min(255, max(0, v + ((int(s) + 8) >> 4))) for v, s in zip((r, g, b), acc))
            i = nearest(pal_arr, r, g, b)
            idx[y, x] = i
            if dither == 2:
                err[y, x] = (r - palette[i][0], g - palette[i][1], b - palette[i][2])
    return idx


# ---------------------------------------------------------------- the encoder (reference)

def table_for(entries):
    size = 2
    while size < entries:
        size <<= 1
    return size


def encode_reference(frames, opts):
    """frames: list of (H,W,4) uint8. Returns (gif bytes, canvases list of (H,W,4) uint8)."""
    F = len(frames)
    h, w, _ = frames[0].shape
    thr = opts["alphaThreshold"]
    maxc = opts["maxColors"]
    anim_trans = any(bool((f[..., 3] < thr).any()) for f in frames)
    limit = maxc - (1 if anim_trans else 0)
    glob = exact_palette(frames, thr)
    global_exact = len(glob) <= limit
    palettes, exacts = [], []
    for f in frames:
        if global_exact:
            palettes.append(glob)
            exacts.append(True)
            continue
        own = exact_palette([f], thr)
        if len(own) <= limit:
            palettes.append(own)
            exacts.append(True)
        else:
            px = f.reshape(-1, 4)
            palettes.append(median_cut(px[px[:, 3] >= thr][:, :3], limit))
            exacts.append(False)
    single = global_exact or F == 1
    gce = F > 1 or anim_trans
    loop_ext = F > 1 and opts["loop"] >= 0
    disp = [-1] * F if opts.get("disposals") is None else opts["disposals"]
    delays = opts.get("delaysMs") or [opts["frameDelayMs"]] * F

    out = bytearray(b"GIF89a")
    tables = []
    for p in palettes:
        entries = len(p) + (1 if anim_trans else 0)
        size = table_for(entries)
        tab = bytearray()
        for i in range(size):
            c = p[i] if i < len(p) else (0, 0, 0)
            tab += bytes(c)
        tables.append((size, bytes(tab)))
    if single:
        size, tab = tables[0]
        out += struct.pack("<HHBBB", w, h, 0x80 | (7 << 4) | (int(math.log2(size)) - 1), 0, 0) + tab
    else:
        out += struct.pack("<HHBBB", w, h, 0, 0, 0)
    if loop_ext:
        out += b"\x21\xff\x0bNETSCAPE2.0\x03\x01" + struct.pack("<H", opts["loop"]) + b"\x00"
    canvases = []
    for k, f in enumerate(frames):
        pal = palettes[k]
        trans_idx = len(pal) if anim_trans else None
        idx = map_frame(f, pal, exacts[k], trans_idx if trans_idx is not None else -1, thr, opts["dither"])
        size, tab = tables[k]
        mcs = max(2, int(math.log2(size)))
        d = disp[k]
        if d < 0:
            d = (2 if anim_trans else 1) if F > 1 else 0
        if gce:
            cs = min(65535, (delays[k] + 5) // 10)
            out += b"\x21\xf9\x04" + bytes([(d << 2) | (1 if anim_trans else 0)]) + struct.pack("<H", cs) + \
                bytes([trans_idx if anim_trans else 0]) + b"\x00"
        local = not single
        out += b"\x2c" + struct.pack("<HHHHB", 0, 0, w, h, (0x80 | (int(math.log2(size)) - 1)) if local else 0)
        if local:
            out += tab
        out += bytes([mcs]) + sub_blocks(pack_codes(lzw_codes(idx.reshape(-1), mcs)))
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                i = int(idx[y, x])
                if anim_trans and i == trans_idx:
                    continue
                canvas[y, x] = (*pal[i], 255)
        canvases.append(canvas)
    out += b"\x3b"
    return bytes(out), canvases, dict(exacts=exacts, palettes=palettes, limit=limit, anim_trans=anim_trans)


# ---------------------------------------------------------------- fixtures

def save(name, frames, **kw):
    opts = dict(maxColors=256, dither=0, alphaThreshold=128, loop=0, frameDelayMs=100, delaysMs=None, disposals=None)
    opts.update(kw)
    h, w, _ = frames[0].shape
    gif, canvases, info = encode_reference(frames, opts)
    # the oracle must read the expected file exactly the way the canvases say
    im = Image.open(io.BytesIO(gif))
    assert im.n_frames == len(frames), f"{name}: Pillow sees {im.n_frames} frames"
    for k in range(len(frames)):
        im.seek(k)
        got = np.asarray(im.convert("RGBA"), dtype=np.uint8)
        if not np.array_equal(got, canvases[k]):
            bad = np.argwhere((got != canvases[k]).any(axis=2))
            raise SystemExit(f"{name}: Pillow disagrees with the reference canvas on frame {k} at {bad[0]}: "
                             f"{got[tuple(bad[0])]} vs {canvases[k][tuple(bad[0])]}")
    with open(os.path.join(OUT, name + ".src.rgba8"), "wb") as fp:
        fp.write(b"".join(f.tobytes() for f in frames))
    with open(os.path.join(OUT, name + ".expected.gif"), "wb") as fp:
        fp.write(gif)
    with open(os.path.join(OUT, name + ".canvas.rgba8"), "wb") as fp:
        fp.write(b"".join(c.tobytes() for c in canvases))
    extra = {}
    if len(frames) == 1 and opts["dither"] == 0 and not info["exacts"][0]:
        # Pillow's own quantizers on the same picture and colour budget, measured the way the test measures ours:
        # PSNR over the three colour channels of the opaque pixels.
        f = frames[0]
        opaque = f[..., 3] >= opts["alphaThreshold"]
        src_rgb = f[..., :3].astype(np.float64)

        def psnr(mapped):
            diff = (mapped.astype(np.float64) - src_rgb)[opaque]
            mse = (diff ** 2).sum() / (3 * diff.shape[0] // 3 * 3 / 3) if diff.size else 0.0
            n = diff.shape[0]
            sq = float((diff ** 2).sum())
            return 99.0 if sq == 0 else 10 * math.log10(255.0 * 255.0 * 3 * n / sq)

        base = Image.fromarray(f[..., :3])
        k = info["limit"]
        mc = base.quantize(colors=k, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert("RGB")
        oc = base.quantize(colors=k, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).convert("RGB")
        ours = canvases[0][..., :3]
        extra = dict(pillowMedianCutPsnr=psnr(np.asarray(mc)), pillowOctreePsnr=psnr(np.asarray(oc)), oursPsnr=psnr(ours))
    with open(os.path.join(OUT, name + ".json"), "w") as fp:
        json.dump(dict(width=w, height=h, frames=len(frames), **opts, **extra), fp)
    note = "" if not extra else "  PSNR ours %.2f dB, Pillow median cut %.2f dB, octree %.2f dB" % (extra["oursPsnr"], extra["pillowMedianCutPsnr"], extra["pillowOctreePsnr"])
    print(f"{name}: {w}x{h} x{len(frames)}  {len(gif)} bytes  (Pillow agrees){note}")


def rgba(rgb, alpha=255):
    a = np.full(rgb.shape[:2] + (1,), alpha, dtype=np.uint8) if np.isscalar(alpha) else alpha[..., None].astype(np.uint8)
    return np.concatenate([rgb.astype(np.uint8), a], axis=2)


def main():
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    rng = np.random.default_rng(0x47494645)

    def random_colours(n):
        return rng.integers(0, 256, size=(n, 3))

    # ---- exact palettes, single picture --------------------------------------------------------------
    save("exact_1x1", [rgba(np.array([[[12, 200, 77]]]))])
    save("exact_two_colours_5x3", [rgba(random_colours(2)[rng.integers(0, 2, size=(3, 5))])])
    pal = random_colours(40)
    save("exact_40_colours_31x17", [rgba(pal[rng.integers(0, 40, size=(17, 31))])])
    pal = np.array([[i, (i * 7) % 256, (i * 13) % 256] for i in range(256)])
    pix = pal[(np.arange(40 * 40) % 256)].reshape(40, 40, 3)
    save("exact_256_colours", [rgba(pix)])
    pal = random_colours(255)
    a = np.where(rng.random((32, 48)) < 0.2, 0, 255)
    pix = pal[rng.integers(0, 255, size=(32, 48))]
    pix[0, :255 % 48] = 0
    k = 0
    for y in range(32):
        for x in range(48):
            if k < 255 and a[y, x] == 255:
                pix[y, x] = pal[k]
                k += 1
    save("exact_255_colours_plus_transparent", [rgba(pix, a)])
    save("exact_all_transparent_6x6", [rgba(np.zeros((6, 6, 3)), 0)])
    pal = random_colours(9)
    a = rng.integers(0, 256, size=(11, 13))
    save("exact_alpha_threshold_boundary", [rgba(pal[rng.integers(0, 9, size=(11, 13))], a)], alphaThreshold=100)
    save("exact_alpha_threshold_0", [rgba(pal[rng.integers(0, 9, size=(11, 13))], a)], alphaThreshold=0)
    pal = random_colours(100)
    save("exact_single_row_300x1", [rgba(pal[rng.integers(0, 100, size=(1, 300))])])
    save("exact_single_column_1x300", [rgba(pal[rng.integers(0, 100, size=(300, 1))])])

    # ---- LZW stress: tables that fill and clear, and several segments -----------------------------------
    pal = random_colours(256)
    save("exact_noise_256_100x100", [rgba(pal[rng.integers(0, 256, size=(100, 100))])])
    yy, xx = np.mgrid[0:120, 0:300]
    smooth = np.stack([(xx // 3) % 40 * 6, (yy // 2) % 40 * 6, ((xx + yy) // 5) % 6 * 40], axis=2)
    save("quant_smooth_300x120_multi_segment", [rgba(smooth)])
    pal = random_colours(256)
    noise = pal[rng.integers(0, 256, size=(300, 400))]
    save("exact_multi_segment_noise_400x300", [rgba(noise)])
    save("exact_runs_90x90", [rgba(np.zeros((90, 90, 3), dtype=np.int64) + (np.arange(90) // 7 % 2 * 200)[:, None, None])])

    # ---- exact animations (one shared palette) -----------------------------------------------------------
    pal = random_colours(60)
    frames = []
    for k in range(8):
        pix = pal[rng.integers(0, 60, size=(16, 24))]
        a = np.where(rng.random((16, 24)) < 0.25, 0, 255)
        frames.append(rgba(pix, a))
    save("anim_exact_transparent_8x", frames, delaysMs=[10 * (k + 1) for k in range(8)], loop=0)
    frames = [rgba(pal[rng.integers(0, 60, size=(10, 12))]) for _ in range(5)]
    save("anim_exact_opaque_5x", frames, loop=3, frameDelayMs=70)
    save("anim_exact_no_loop_extension", frames[:3], loop=-1)
    save("anim_exact_explicit_disposals", frames[:3], disposals=[1, 2, 3])
    frames = [rgba(pal[rng.integers(0, 60, size=(9, 9))], np.where(rng.random((9, 9)) < 0.3, 0, 255)) for _ in range(70)]
    save("anim_exact_70_frames", frames, frameDelayMs=40)

    # ---- quantized pictures --------------------------------------------------------------------------------
    photo = np.asarray(Image.open(PHOTO).convert("RGB"), dtype=np.uint8)
    small = np.asarray(Image.fromarray(photo).resize((160, 90), Image.Resampling.BOX), dtype=np.uint8)
    for dither, nm in ((0, "none"), (1, "ordered"), (2, "floyd")):
        save(f"quant_photo_160x90_256_{nm}", [rgba(small)], dither=dither)
    for k, dither in ((64, 0), (16, 1), (16, 2), (2, 0), (3, 2)):
        save(f"quant_photo_160x90_max{k}_d{dither}", [rgba(small)], maxColors=k, dither=dither)
    save("quant_photo_320x180_256_none_multi_segment", [rgba(photo)])
    save("quant_photo_320x180_64_floyd_multi_segment", [rgba(photo)], maxColors=64, dither=2)
    # transparency + quantization: the transparent index is reserved
    holes = np.zeros((90, 160), dtype=np.int64)
    holes[(((xx[:90, :160] // 12) + (yy[:90, :160] // 9)) % 5) == 0] = 1
    a = np.where(holes == 1, 0, 255)
    for dither in (0, 1, 2):
        save(f"quant_photo_transparent_d{dither}", [rgba(small, a)], dither=dither)
    save("quant_photo_transparent_max8", [rgba(small, a)], maxColors=8)
    save("quant_photo_transparent_max2", [rgba(small, a)], maxColors=2)
    save("quant_photo_transparent_max3_floyd", [rgba(small, a)], maxColors=3, dither=2)

    # ---- quantized animations: per-frame local palettes -----------------------------------------------------
    frames = []
    for k in range(5):
        crop = photo[k * 12:k * 12 + 72, k * 20:k * 20 + 128]
        frames.append(rgba(crop))
    save("anim_quant_5x_local_palettes", frames, dither=0, frameDelayMs=80)
    save("anim_quant_5x_ordered", frames, dither=1)
    save("anim_quant_3x_floyd", frames[:3], dither=2)
    # a frame that fits exactly next to frames that do not (its own exact local palette)
    mixed = [frames[0], rgba(random_colours(30)[rng.integers(0, 30, size=(72, 128))]), frames[2]]
    save("anim_quant_mixed_exact_and_quantized", mixed, dither=1)
    tr = [rgba(f[..., :3], np.where(rng.random(f.shape[:2]) < 0.15, 0, 255)) for f in frames[:4]]
    save("anim_quant_transparent_4x", tr, dither=0, delaysMs=[50, 60, 70, 80])
    print("done")


if __name__ == "__main__":
    main()
