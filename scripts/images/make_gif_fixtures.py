#!/usr/bin/env python3
"""
Generates the GIF test fixtures in SpawnDev.Codecs.Demo.Shared/TestData/Gif/ :

  <name>.gif           the file
  <name>.json          canvas size, loop count, per-frame rectangle / delay / disposal / blend / interlace, flags
  <name>.web.rgba8     ALL frames' canvases concatenated (frame k at k * W * H * 4), RGBA, rows top to bottom:
                       what GifCompositing.Web must produce. Derived from the index arrays the file was BUILT
                       from (never from a decoder) and the compositing rules in `composite` below.
  <name>.pil.rgba8     the same, but exactly what PILLOW shows: Image.open, then for every k Image.seek(k) and
                       convert("RGBA") (Pillow 13, default GifImagePlugin.LOADING_STRATEGY). This is the truth for
                       GifCompositing.Pillow. Absent when Pillow cannot be the oracle for the file (it grows the
                       canvas for frames that extend past the screen, or it refuses the file); the JSON says why
                       ("pillow": "ok" | "canvas-differs" | "error: ...").
  bad_*.gif/.json      hostile streams the decoder must reject with a specific reason: {"reject": text, "frame": k}
                       (+ "okFrames" decodable before the corrupt frame) and what Pillow does with them ("pillow").

Why a hand-rolled GIF *writer* and not only Pillow: the tests need every LZW corner (min code size 2..8, table
fills and clear, deferred clear = no clear code when the table is full, 1-byte sub-blocks, no leading clear, no end
code), local + global palettes, interlacing at every height, all four disposal methods and the sticky/unspecified
ones, frames that are partial / offset / clipped, comment / plain-text / unknown application extensions, GIF87a, a
missing trailer, trailing garbage, and files built to be rejected. Files written by Pillow itself are included too
(real-world LZW shapes, optimize=True).

Web truth rules (`composite`): initial canvas transparent; a transparent index leaves the canvas alone; disposal 2
clears the frame rectangle to transparent (the logical screen background index is ignored, like browsers);
disposal 3 restores the canvas from before the frame; 0 / 1 / 4..7 leave it; an index beyond the colour table is
opaque black; frame pixels outside the logical screen are clipped.

The fuzz_* files are random animations (random rectangles, palettes, transparency, disposal 0..7, GCE-less
frames, interlacing, indices beyond the table). fuzz_pal_* use ordinary palettes, fuzz_gray_* palettes that are the
identity gray ramp (Pillow's "L" mode), fuzz_mixed_* ordinary first frames with gray-ramp frames after. Their
Pillow truth is what the Pillow-mode tests check: a few hundred frames of real Pillow output on random inputs.

Run:   <venv>/python -I scripts/images/make_gif_fixtures.py        (needs numpy + pillow)
"""
import io
import json
import os
import struct
import sys

import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "SpawnDev.Codecs.Demo.Shared", "TestData", "Gif")
os.makedirs(OUT, exist_ok=True)

DISP_NAMES = {2: "Background", 3: "Previous"}      # everything else is "None"


# ---------------------------------------------------------------- LZW

LZW_STATS = {"full": 0}   # how many times a code table reached 4096 entries (fixtures must exercise this)


def lzw_compress(idx, mcs, mode="clear_full", lead_clear=True, emit_end=True, trailing=b""):
    """GIF LZW. mode: 'clear_full' = emit a clear code when the table fills; 'deferred' = never clear."""
    clear = 1 << mcs
    eoi = clear + 1
    out = bytearray()
    acc = 0
    nbits = 0

    def emit(code, width):
        nonlocal acc, nbits
        acc |= code << nbits
        nbits += width
        while nbits >= 8:
            out.append(acc & 255)
            acc >>= 8
            nbits -= 8

    table = {}
    nxt = eoi + 1
    width = mcs + 1
    if lead_clear:
        emit(clear, width)
    seq = [int(v) for v in idx]
    prefix = seq[0]
    for k in seq[1:]:
        key = (prefix, k)
        c = table.get(key)
        if c is not None:
            prefix = c
            continue
        emit(prefix, width)
        if nxt < 4096:
            table[key] = nxt
            nxt += 1
            if nxt > (1 << width) and width < 12:
                width += 1
            if nxt == 4096:
                LZW_STATS["full"] += 1
            if nxt == 4096 and mode == "clear_full":
                emit(clear, width)
                table = {}
                nxt = eoi + 1
                width = mcs + 1
        prefix = k
    emit(prefix, width)
    # The decoder adds an entry after the last data code too, and may widen before it reads the end code.
    if nxt < 4096:
        nxt += 1
        if nxt > (1 << width) and width < 12:
            width += 1
    if emit_end:
        emit(eoi, width)
    if nbits:
        out.append(acc & 255)
    return bytes(out) + trailing


def pack_codes(codes):
    """[(code, width), ...] -> bytes, LSB first."""
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


def sub_blocks(data, block=255):
    out = bytearray()
    for i in range(0, len(data), block):
        chunk = data[i:i + block]
        out.append(len(chunk))
        out += chunk
    out.append(0)
    return bytes(out)


def interlace_rows(h):
    return list(range(0, h, 8)) + list(range(4, h, 8)) + list(range(2, h, 4)) + list(range(1, h, 2))


# ---------------------------------------------------------------- GIF writer


def table_bits(n):
    """colour table size field for n entries (table has 2 << field entries, at least 2)."""
    size = 2
    field = 0
    while size < n:
        size <<= 1
        field += 1
    return field, size


def pack_palette(pal, size):
    p = bytearray()
    for i in range(size):
        r, g, b = pal[i] if i < len(pal) else (0, 0, 0)
        p += bytes([int(r), int(g), int(b)])
    return bytes(p)


def frame_dict(idx, x=0, y=0, palette=None, delay=10, disp=0, trans=None, interlace=False, mcs=None,
               mode="clear_full", block=255, pre=None, gce=True, lzw_raw=None, hdr_wh=None):
    """One frame. A frame without a graphic control extension has no delay, disposal or transparent index.
    lzw_raw = (min code size, payload bytes) replaces the generated LZW data; hdr_wh = (w, h) lies in the descriptor."""
    if not gce:
        trans, disp, delay = None, 0, 0
    return dict(idx=np.asarray(idx), x=x, y=y, palette=palette, delay=delay, disp=disp, trans=trans,
                interlace=interlace, mcs=mcs, mode=mode, block=block, pre=pre or [], gce=gce,
                lzw_raw=lzw_raw, hdr_wh=hdr_wh)


def write_gif(W, H, frames, global_palette=None, loop=None, version=b"GIF89a", bg=0, trailer=True,
              garbage=b"", post_header=b""):
    out = bytearray(version)
    gfield = gsize = 0
    flags = 0
    if global_palette is not None:
        gfield, gsize = table_bits(len(global_palette))
        flags = 0x80 | (7 << 4) | gfield
    out += struct.pack("<HHBBB", W, H, flags, bg, 0)
    if global_palette is not None:
        out += pack_palette(global_palette, gsize)
    out += post_header
    if loop is not None:
        out += b"\x21\xff\x0bNETSCAPE2.0\x03\x01" + struct.pack("<H", loop) + b"\x00"
    for f in frames:
        for blob in f["pre"]:
            out += blob
        if f["gce"]:
            pk = (f["disp"] << 2) | (1 if f["trans"] is not None else 0)
            out += b"\x21\xf9\x04" + bytes([pk]) + struct.pack("<H", f["delay"]) + \
                bytes([f["trans"] if f["trans"] is not None else 0]) + b"\x00"
        h, w = f["idx"].shape
        dw, dh = f["hdr_wh"] if f["hdr_wh"] else (w, h)
        fl = 0x40 if f["interlace"] else 0
        lfield = lsize = 0
        if f["palette"] is not None:
            lfield, lsize = table_bits(len(f["palette"]))
            fl |= 0x80 | lfield
        out += b"\x2c" + struct.pack("<HHHHB", f["x"], f["y"], dw, dh, fl)
        if f["palette"] is not None:
            out += pack_palette(f["palette"], lsize)
        pal_n = lsize if f["palette"] is not None else gsize
        if f["lzw_raw"] is not None:
            mcs, payload = f["lzw_raw"]
            out += bytes([mcs]) + sub_blocks(payload, f["block"])
            continue
        mcs = f["mcs"] if f["mcs"] is not None else max(2, (pal_n - 1).bit_length())
        rows = interlace_rows(h) if f["interlace"] else range(h)
        stream = np.concatenate([f["idx"][r] for r in rows]) if h else np.array([], dtype=int)
        out += bytes([mcs]) + sub_blocks(lzw_compress(stream, mcs, f["mode"]), f["block"])
    if trailer:
        out += b"\x3b"
    out += garbage
    return bytes(out)


# ---------------------------------------------------------------- spec compositor (the Web truth)


def composite(W, H, frames, global_palette):
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    outs = []
    saved = None
    for i, f in enumerate(frames):
        if i > 0:
            p = frames[i - 1]
            ph, pw = p["idx"].shape
            if p["disp"] == 2:
                canvas[p["y"]:p["y"] + ph, p["x"]:p["x"] + pw] = 0
            elif p["disp"] == 3 and saved is not None:
                canvas = saved.copy()
        if f["disp"] == 3:
            saved = canvas.copy()
        pal = f["palette"] if f["palette"] is not None else global_palette
        table_n = table_bits(len(pal))[1]
        h, w = f["idx"].shape
        for yy in range(h):
            for xx in range(w):
                cx, cy = f["x"] + xx, f["y"] + yy
                if cx >= W or cy >= H:
                    continue
                v = int(f["idx"][yy, xx])
                if f["trans"] is not None and v == f["trans"]:
                    continue
                if v < table_n and v < len(pal):
                    r, g, b = pal[v]
                else:
                    r, g, b = 0, 0, 0
                canvas[cy, cx] = (r, g, b, 255)
        outs.append(canvas.copy())
    return outs


def has_alpha_web(W, H, frames):
    f0 = frames[0]
    h, w = f0["idx"].shape
    if f0["x"] != 0 or f0["y"] != 0 or w < W or h < H:
        return True
    return any(f["trans"] is not None or f["disp"] == 2 for f in frames)


def pillow_truth(data, W, H, nframes):
    """Pillow's view of every frame, or (None, reason)."""
    try:
        im = Image.open(io.BytesIO(data))
        n = getattr(im, "n_frames", 1)
        got = []
        for k in range(n):
            im.seek(k)
            got.append(np.asarray(im.convert("RGBA"), dtype=np.uint8).copy())
        loop = Image.open(io.BytesIO(data)).info.get("loop")
    except Exception as ex:
        return None, f"error: {type(ex).__name__}: {str(ex)[:70]}", None
    if n != nframes:
        return None, f"error: Pillow sees {n} frames, expected {nframes}", None
    if any(g.shape != (H, W, 4) for g in got):
        return None, "canvas-differs", loop
    return got, "ok", loop


def save(name, data, W, H, frames, global_palette, loop=None, web_truth="auto", exact_frames=True, require_pillow=True):
    """Write <name>.gif/.json/.web.rgba8/.pil.rgba8. web_truth=None: no Web truth (Pillow-written files)."""
    web = composite(W, H, frames, global_palette) if web_truth == "auto" else web_truth
    with open(os.path.join(OUT, name + ".gif"), "wb") as fp:
        fp.write(data)
    if web is not None:
        with open(os.path.join(OUT, name + ".web.rgba8"), "wb") as fp:
            fp.write(b"".join(w.tobytes() for w in web))
    pil, status, pil_loop = pillow_truth(data, W, H, len(frames))
    if pil is not None:
        with open(os.path.join(OUT, name + ".pil.rgba8"), "wb") as fp:
            fp.write(b"".join(p.tobytes() for p in pil))
        if pil_loop != loop:
            raise SystemExit(f"{name}: loop count {loop} but Pillow reports {pil_loop}")
    meta = dict(width=W, height=H, loop=loop, exactFrames=exact_frames, pillow=status, web=web is not None,
                hasTransparentIndex=any(f["trans"] is not None for f in frames),
                hasAlphaWeb=has_alpha_web(W, H, frames),
                frames=[dict(x=f["x"], y=f["y"], w=int(f["idx"].shape[1]), h=int(f["idx"].shape[0]),
                             delayMs=f["delay"] * 10, dispose=DISP_NAMES.get(f["disp"], "None"),
                             blend="Over" if f["trans"] is not None else "Source",
                             interlaced=bool(f["interlace"])) for f in frames])
    with open(os.path.join(OUT, name + ".json"), "w") as fp:
        json.dump(meta, fp)
    if require_pillow and pil is None:
        print(f"{name}: {len(data)} bytes, {len(frames)} frame(s)  Pillow: {status}")
    return pil is not None


def rpal(rng, n):
    return [tuple(int(c) for c in rng.integers(0, 256, size=3)) for _ in range(n)]


def gray_pal(n):
    return [(i, i, i) for i in range(n)]


def still(name, rng, w, h, n, **kw):
    pal = rpal(rng, n)
    idx = rng.integers(0, n, size=(h, w))
    frames = [frame_dict(idx, **kw)]
    save(name, write_gif(w, h, frames, pal), w, h, frames, pal)


# ---------------------------------------------------------------- hostile streams

def block_offsets(data):
    """[(kind, file offset)] of the top-level blocks of a well-formed GIF ('gce', 'ext', 'image'); used to place cuts."""
    pos = 13
    if data[10] & 0x80:
        pos += 3 * (2 << (data[10] & 7))
    out = []

    def skip(p):
        while data[p]:
            p += 1 + data[p]
        return p + 1

    while pos < len(data) and data[pos] != 0x3B:
        if data[pos] == 0x21:
            out.append(("gce" if data[pos + 1] == 0xF9 else "ext", pos))
            pos = skip(pos + 2)
        elif data[pos] == 0x2C:
            out.append(("image", pos))
            fl = data[pos + 9]
            p = pos + 10
            if fl & 0x80:
                p += 3 * (2 << (fl & 7))
            pos = skip(p + 1)
        else:
            raise ValueError("not a well-formed GIF")
    return out


def tiny_gif(lzw, mcs=2, w=2, h=2, pal=((0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)), term=True, gpal=True,
             hdr_wh=None, screen_wh=None):
    """A 2x2-class GIF with a 4 entry palette and one frame whose LZW payload is exactly `lzw`.
    With min code size 2 the codes are 3 bits wide: clear = 4, end = 5, literals 0..3, first free code 6."""
    f = bytearray(b"GIF89a")
    sw, sh = screen_wh if screen_wh else (w, h)
    f += struct.pack("<HHBBB", sw, sh, (0x80 | 1) if gpal else 0, 0, 0)
    if gpal:
        f += pack_palette(list(pal), 4)
    dw, dh = hdr_wh if hdr_wh else (w, h)
    f += b"\x2c" + struct.pack("<HHHHB", 0, 0, dw, dh, 0)
    f += bytes([mcs]) + (bytes([len(lzw)]) + bytes(lzw) if len(lzw) else b"") + (b"\x00" if term else b"")
    f += b";"
    return bytes(f)


def hostile_set(rng):
    """(name, bytes, reject text or None, frame to decode, okFrames) - cut from valid files / crafted code streams."""
    out = []
    pal16 = rpal(rng, 16)
    idx = rng.integers(0, 16, size=(7, 19))
    base = write_gif(19, 7, [frame_dict(idx)], pal16)
    out.append(("bad_signature_x", b"GIX89a" + base[6:], "bad signature", 0, []))
    out.append(("bad_version_85a", b"GIF85a" + base[6:], "bad signature", 0, []))
    out.append(("bad_trunc_screen_descriptor", base[:9], "truncated", 0, []))
    out.append(("bad_trunc_global_palette", base[:20], "truncated", 0, []))
    out.append(("bad_trunc_image_descriptor", base[:13 + 48 + 8 + 4], "truncated", 0, []))
    out.append(("bad_trunc_lzw_half", base[:len(base) - (len(base) - (13 + 48 + 8 + 10 + 2)) // 2], "truncated", 0, []))
    out.append(("bad_trunc_no_subblock_terminator", base[:-2], "truncated", 0, []))
    out.append(("bad_unknown_block_0x55", base[:-1] + b"\x55", "invalid block introducer", 0, []))
    out.append(("bad_unknown_block_0x00", base[:-1] + b"\x00\x3b", "invalid block introducer", 0, []))
    out.append(("bad_no_frames", base[:13 + 48] + b"\x3b", "no image data", 0, []))
    out.append(("bad_header_only", base[:13 + 48], "truncated", 0, []))
    # cut inside a graphic control extension / a local palette / the second frame of an animation
    lp = rpal(rng, 8)
    fr = [frame_dict(rng.integers(0, 16, size=(6, 6)), disp=1, delay=3),
          frame_dict(rng.integers(0, 8, size=(5, 5)), palette=lp, disp=2, delay=4, trans=1),
          frame_dict(rng.integers(0, 16, size=(4, 4)), disp=1, delay=5)]
    anim = write_gif(6, 6, fr, pal16, loop=0)
    gce_at = anim.index(b"\x21\xf9", 13 + 48 + 19)          # second GCE
    out.append(("bad_trunc_in_gce", anim[:gce_at + 4], "truncated", 0, []))
    desc_at = anim.index(b"\x2c", gce_at)
    out.append(("bad_trunc_in_local_palette", anim[:desc_at + 10 + 9], "truncated", 0, []))
    out.append(("bad_trunc_second_frame", anim[:desc_at + 10 + 24 + 6], "truncated", 0, []))
    # LZW code streams (min code size 2: 3-bit codes, clear 4, end 5)
    out.append(("bad_lzw_code_beyond_table", tiny_gif(pack_codes([(4, 3), (1, 3), (7, 3)])), "invalid LZW code", 0, []))
    out.append(("bad_lzw_first_code_not_literal", tiny_gif(pack_codes([(4, 3), (6, 3)])), "invalid LZW code", 0, []))
    out.append(("bad_lzw_early_end_code", tiny_gif(pack_codes([(4, 3), (1, 3), (5, 3)])), "LZW end code after 1 of 4", 0, []))
    out.append(("bad_lzw_data_ends", tiny_gif(pack_codes([(4, 3), (1, 3)])), "LZW data ends after 1 of 4", 0, []))
    out.append(("bad_lzw_no_data_bytes", tiny_gif(b""), "can describe at most 0", 0, []))
    for m in (0, 1, 9, 12, 255):
        out.append((f"bad_lzw_mincode_{m}", tiny_gif(pack_codes([(4, 3), (1, 3), (5, 3)]), mcs=m),
                    f"invalid LZW minimum code size {m}", 0, []))
    out.append(("bad_zero_width_frame", tiny_gif(b"\x04", hdr_wh=(0, 2)), "zero size", 0, []))
    out.append(("bad_zero_height_frame", tiny_gif(b"\x04", hdr_wh=(2, 0)), "zero size", 0, []))
    out.append(("bad_zero_screen", tiny_gif(b"\x04", screen_wh=(0, 0)), "zero size", 0, []))
    out.append(("bad_no_colour_table", tiny_gif(pack_codes([(4, 3), (0, 3), (1, 3), (2, 3), (3, 4), (5, 4)]), gpal=False),
                "no colour table", 0, []))
    # a header that claims far more pixels than the data can describe must not size a giant index buffer
    out.append(("bad_frame_larger_than_data", tiny_gif(pack_codes([(4, 3), (1, 3), (5, 3)]), hdr_wh=(30000, 30000), screen_wh=(30000, 100)),
                "can describe at most", 0, []))
    out.append(("bad_screen_too_large", tiny_gif(pack_codes([(4, 3), (1, 3), (5, 3)]), screen_wh=(65535, 65535)),
                "too large", 0, []))
    # an LZW error in frame 1 only: frame 0 still decodes
    fr2 = [frame_dict(rng.integers(0, 16, size=(6, 6)), disp=1), frame_dict(rng.integers(0, 16, size=(6, 6)), disp=1)]
    two = bytearray(write_gif(6, 6, fr2, pal16))
    d2 = two.index(b"\x2c", two.index(b"\x2c", 13 + 48 + 8) + 10)     # second image descriptor
    for i in range(6):
        two[d2 + 10 + 2 + i] = 0xFF
    out.append(("bad_lzw_error_in_frame1", bytes(two), "frame 1", 1, [0]))
    return out


def save_hostile(rng):
    for name, data, reject, frame, ok_frames in hostile_set(rng):
        with open(os.path.join(OUT, name + ".gif"), "wb") as fp:
            fp.write(data)
        try:
            im = Image.open(io.BytesIO(data))
            for k in range(getattr(im, "n_frames", 1)):
                im.seek(k)
                im.convert("RGBA")
            verdict = "accepts"
        except Exception as ex:
            verdict = f"rejects: {type(ex).__name__}"
        with open(os.path.join(OUT, name + ".json"), "w") as fp:
            json.dump(dict(reject=reject, frame=frame, okFrames=ok_frames, pillow=verdict), fp)
        print(f"{name}: {len(data)} bytes  must be rejected: {reject!r}  Pillow {verdict}")


# ---------------------------------------------------------------- random animations (fuzz_*)

def fuzz_case(rng, kind):
    """kind: 'pal' ordinary palettes, 'gray' identity gray ramps, 'mixed' ordinary first frame + gray frames after."""
    W = int(rng.integers(3, 14))
    H = int(rng.integers(3, 14))
    nf = int(rng.integers(1, 9))

    def mkpal(n, gray):
        if gray:
            return gray_pal(n)
        p = rpal(rng, n)
        if all(c == 0 for c in p[0]) or all(p[i] == (i, i, i) for i in range(len(p))):
            p[0] = (p[0][0], 1, p[0][2])
        # a table whose entries happen to be the identity ramp would be gray; make entry 1 differ
        if len(p) > 1 and p[1] == (1, 1, 1):
            p[1] = (1, 2, 1)
        return p

    first_gray = kind == "gray"
    sizes = [2, 4, 16, 256] if kind == "gray" else [2, 3, 4, 16, 100, 256]
    gp = None if rng.random() < 0.15 else mkpal(int(rng.choice(sizes)), first_gray)
    frames = []
    for i in range(nf):
        if rng.random() < 0.3 and i > 0:
            w, h, x, y = W, H, 0, 0
        else:
            w = int(rng.integers(1, W + 1))
            h = int(rng.integers(1, H + 1))
            x = int(rng.integers(0, W - w + 1))
            y = int(rng.integers(0, H - h + 1))
        lp = None
        if gp is None or rng.random() < 0.3:
            lp = mkpal(int(rng.choice([2, 4, 16, 256])), first_gray)
        if kind == "mixed" and i > 0 and rng.random() < 0.5:
            lp = gray_pal(int(rng.choice([4, 16, 256])))
        n = len(lp) if lp is not None else len(gp)
        tsz = table_bits(n)[1]
        hi = min(tsz + (3 if rng.random() < 0.2 else 0), 256, 1 << max(2, (tsz - 1).bit_length()))
        idx = rng.integers(0, max(hi, 1), size=(h, w))
        trans = int(rng.integers(0, min(tsz, 256))) if rng.random() < 0.5 else None
        if trans is not None and rng.random() < 0.7:
            idx = np.where(rng.random(idx.shape) < 0.4, trans, idx)
        disp = int(rng.choice([0, 1, 2, 3, 3, 2, 4, 7, 0]))
        frames.append(frame_dict(idx, x=x, y=y, palette=lp, delay=int(rng.integers(0, 50)), disp=disp, trans=trans,
                                 interlace=bool(rng.random() < 0.3), gce=bool(rng.random() < 0.85)))
    if gp is None:
        for f in frames:
            if f["palette"] is None:
                f["palette"] = mkpal(16, first_gray)
    bg = int(rng.integers(0, 6)) if gp is not None else 0
    return W, H, frames, gp, bg


def make_fuzz(count_pal=40, count_gray=14, count_mixed=10):
    made = 0
    for kind, count in (("pal", count_pal), ("gray", count_gray), ("mixed", count_mixed)):
        for i in range(count):
            rng = np.random.default_rng(0x66757a7a + 1000 * len(kind) + i)
            W, H, frames, gp, bg = fuzz_case(rng, kind)
            data = write_gif(W, H, frames, gp, bg=bg)
            ok = save(f"fuzz_{kind}_{i:03d}", data, W, H, frames, gp, require_pillow=False)
            if not ok:
                raise SystemExit(f"fuzz_{kind}_{i:03d}: Pillow could not be the oracle")
            made += 1
    print(f"{made} fuzz animations written")


# ---------------------------------------------------------------- main

def main():
    rng = np.random.default_rng(0x474946)

    # ---- palette sizes, odd sizes -----------------------------------------------------------------------
    for n, w, h in ((2, 13, 9), (4, 17, 11), (16, 19, 7), (256, 23, 21), (3, 9, 9), (100, 15, 15), (7, 8, 8)):
        still(f"pal{n}_{w}x{h}", rng, w, h, n)
    for w, h in ((1, 1), (1, 37), (41, 1), (5, 3), (2, 2), (600, 3), (3, 300)):
        still(f"size_{w}x{h}", rng, w, h, 16)
    # 87a, no extensions at all
    pal = rpal(rng, 8)
    idx = rng.integers(0, 8, size=(12, 14))
    fr = [frame_dict(idx, gce=False)]
    save("gif87a", write_gif(14, 12, fr, pal, version=b"GIF87a"), 14, 12, fr, pal)

    # ---- interlacing at every interesting height -------------------------------------------------------------
    for h in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 23, 33, 130):
        pal = rpal(rng, 256)
        idx = rng.integers(0, 256, size=(h, 11))
        fr = [frame_dict(idx, interlace=True)]
        save(f"interlaced_h{h}", write_gif(11, h, fr, pal), 11, h, fr, pal)

    # ---- colour tables -------------------------------------------------------------------------------
    pal = rpal(rng, 16)
    idx = rng.integers(0, 16, size=(10, 12))
    fr = [frame_dict(idx, palette=pal)]
    save("local_only", write_gif(12, 10, fr, None), 12, 10, fr, None)
    gp = rpal(rng, 4)
    fr = [frame_dict(idx, palette=pal)]
    save("local_over_global", write_gif(12, 10, fr, gp), 12, 10, fr, gp)
    # identity gray ramps (Pillow's "L" mode): the index IS the gray level, also beyond a short table
    for n in (256, 16, 4):
        idx = rng.integers(0, n, size=(9, 13))
        fr = [frame_dict(idx)]
        save(f"gray_ramp_{n}", write_gif(13, 9, fr, gray_pal(n)), 13, 9, fr, gray_pal(n))
    gp4 = gray_pal(4)
    idx = rng.integers(0, 8, size=(8, 8))
    fr = [frame_dict(idx, mcs=3)]
    save("gray_ramp_4_index_beyond_table", write_gif(8, 8, fr, gp4), 8, 8, fr, gp4)

    # ---- transparency -------------------------------------------------------------------------------------
    for tr in (0, 3, 255):
        pal = rpal(rng, 256)
        idx = rng.integers(0, 6, size=(9, 13)) if tr != 255 else rng.integers(250, 256, size=(9, 13))
        fr = [frame_dict(idx, trans=tr)]
        save(f"trans_index{tr}", write_gif(13, 9, fr, pal), 13, 9, fr, pal)

    # ---- LZW corners ---------------------------------------------------------------------------------
    pal2 = rpal(rng, 2)
    idx = rng.integers(0, 2, size=(9, 11))
    for mcs in (2, 5, 8):
        fr = [frame_dict(idx, mcs=mcs)]
        save(f"mincode{mcs}_pal2", write_gif(11, 9, fr, pal2), 11, 9, fr, pal2)
    pal4 = rpal(rng, 4)
    idx = rng.integers(0, 10, size=(8, 8))   # indices beyond the 4-entry table: opaque black
    fr = [frame_dict(idx, mcs=4)]
    save("index_beyond_palette", write_gif(8, 8, fr, pal4), 8, 8, fr, pal4)

    def noise_fixture(name, n, w, h, mode, min_fills):
        pal = rpal(rng, n)
        idx = rng.integers(0, n, size=(h, w))
        fr = [frame_dict(idx, mode=mode)]
        before = LZW_STATS["full"]
        data = write_gif(w, h, fr, pal)
        fills = LZW_STATS["full"] - before
        assert fills >= min_fills, f"{name}: the code table filled {fills} times, wanted {min_fills}"
        save(name, data, w, h, fr, pal)
        print(f"    ({name}: LZW table reached 4096 entries {fills} time(s))")

    # Noise big enough to fill the 4096-entry table (clear code when full, or the deferred-clear shape where the
    # encoder just keeps coding with the frozen table).
    for mode in ("clear_full", "deferred"):
        noise_fixture(f"noise256_{mode}", 256, 100, 80, mode, 2 if mode == "clear_full" else 1)
        noise_fixture(f"noise16_{mode}", 16, 100, 140, mode, 1)
    noise_fixture("noise4_clear_full", 4, 200, 140, "clear_full", 1)
    noise_fixture("noise4_deferred", 4, 100, 280, "deferred", 1)
    # long runs: strings of maximal length (4096-entry table of ever longer strings; also the KwKwK case again and again)
    idx = np.zeros((90, 90), dtype=int)
    idx[::7] = 1
    fr = [frame_dict(idx, mode="clear_full")]
    save("runs_pal2", write_gif(90, 90, fr, pal2), 90, 90, fr, pal2)
    idx = np.zeros((40, 40), dtype=int)
    fr = [frame_dict(idx)]
    save("runs_single_colour", write_gif(40, 40, fr, pal2), 40, 40, fr, pal2)
    idx = rng.integers(0, 16, size=(24, 24))
    for blk in (1, 7, 255):
        fr = [frame_dict(idx, block=blk)]
        pal = rpal(rng, 16)
        save(f"subblock{blk}", write_gif(24, 24, fr, pal), 24, 24, fr, pal)
    # valid streams other decoders (Pillow) accept: no leading clear, no end code, junk after the last pixel
    pal = rpal(rng, 16)
    idx = rng.integers(0, 16, size=(9, 9))
    stream = idx.reshape(-1)
    for nm, kw in (("lzw_no_leading_clear", dict(lead_clear=False)), ("lzw_no_end_code", dict(emit_end=False)),
                   ("lzw_junk_after_pixels", dict(trailing=b"\x55\xaa\x01\x02\x03"))):
        fr = [frame_dict(idx, lzw_raw=(4, lzw_compress(stream, 4, **kw)))]
        save(nm, write_gif(9, 9, fr, pal), 9, 9, fr, pal)

    # ---- extensions the decoder must skip -------------------------------------------------------------
    comment = b"\x21\xfe" + sub_blocks(b"hello comment " * 30)
    plain = b"\x21\x01\x0c" + struct.pack("<HHHH", 0, 0, 8, 8) + b"\x08\x08\x01\x00" + sub_blocks(b"text!")
    unknown_app = b"\x21\xff\x0bXMP DataXMP" + sub_blocks(b"<x:xmpmeta>" + bytes(range(1, 200)) + b"</x:xmpmeta>")
    unknown_ext = b"\x21\x99" + sub_blocks(b"\x01\x02\x03")
    pal = rpal(rng, 16)
    idx = rng.integers(0, 16, size=(10, 10))
    idx2 = rng.integers(0, 16, size=(10, 10))
    fr = [frame_dict(idx, pre=[comment, plain, unknown_app]), frame_dict(idx2, pre=[unknown_ext, comment])]
    save("extensions_skipped", write_gif(10, 10, fr, pal, loop=0, post_header=comment), 10, 10, fr, pal, loop=0)
    # two graphic control extensions before one image (never written by an encoder): a later "unspecified" disposal
    # and a later GCE without a transparent flag do not override the earlier ones (Pillow's merge).
    gce_a = b"\x21\xf9\x04" + bytes([(2 << 2) | 1]) + struct.pack("<H", 7) + bytes([2]) + b"\x00"
    gce_b = b"\x21\xf9\x04" + bytes([0]) + struct.pack("<H", 9) + bytes([5]) + b"\x00"
    fr = [frame_dict(rng.integers(0, 16, size=(8, 8)), gce=False, pre=[gce_a, gce_b]),
          frame_dict(rng.integers(0, 16, size=(5, 5)), x=1, y=1, gce=False)]
    fr[0]["trans"], fr[0]["disp"], fr[0]["delay"] = 2, 2, 9
    save("double_gce", write_gif(8, 8, fr, pal), 8, 8, fr, pal)

    # ---- loop counts ---------------------------------------------------------------------------------
    for lp in (0, 3, 65535):
        fr = [frame_dict(rng.integers(0, 16, size=(6, 6)), delay=d) for d in (5, 10)]
        save(f"loop{lp}", write_gif(6, 6, fr, pal, loop=lp), 6, 6, fr, pal, loop=lp)
    fr = [frame_dict(rng.integers(0, 16, size=(6, 6)), delay=d) for d in (5, 10)]
    save("loop_none", write_gif(6, 6, fr, pal), 6, 6, fr, pal, loop=None)
    # a loop extension AFTER the first image is not the animation's loop count
    fr = [frame_dict(rng.integers(0, 16, size=(6, 6)), delay=5),
          frame_dict(rng.integers(0, 16, size=(6, 6)), delay=5, pre=[b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x09\x00\x00"])]
    save("loop_after_first_frame", write_gif(6, 6, fr, pal), 6, 6, fr, pal, loop=None)

    # ---- trailer handling -----------------------------------------------------------------------------
    idx = rng.integers(0, 16, size=(7, 9))
    fr = [frame_dict(idx)]
    save("no_trailer", write_gif(9, 7, fr, pal, trailer=False), 9, 7, fr, pal)
    save("trailing_garbage", write_gif(9, 7, fr, pal, garbage=b"\x00\x01garbage after the trailer\xff" * 3), 9, 7, fr, pal)

    # ---- animations: disposal -----------------------------------------------------------------------------
    W, H = 20, 14
    pal = rpal(rng, 16)

    def noise(w, h, n=16):
        return rng.integers(1, n, size=(h, w))

    # disposal 0/1 : frames accumulate on the canvas (partial rectangles, transparency lets earlier pixels show)
    for disp in (0, 1):
        fr = [frame_dict(noise(W, H), disp=disp, delay=7),
              frame_dict(noise(8, 6), x=3, y=2, disp=disp, delay=1, trans=1),
              frame_dict(noise(10, 5), x=9, y=8, disp=disp, delay=100, trans=2),
              frame_dict(noise(W, H), disp=disp, delay=0, trans=3)]
        save(f"anim_dispose{disp}", write_gif(W, H, fr, pal, loop=0), W, H, fr, pal, loop=0)
    # disposal 2 : each frame's rectangle is cleared before the next
    fr = [frame_dict(noise(W, H), disp=1),
          frame_dict(noise(8, 6), x=3, y=2, disp=2, delay=4),
          frame_dict(noise(9, 4), x=10, y=9, disp=2, delay=4, trans=1),
          frame_dict(noise(5, 5), x=0, y=0, disp=2, delay=4),
          frame_dict(noise(6, 6), x=14, y=8, disp=1, delay=4, trans=2)]
    save("anim_dispose2", write_gif(W, H, fr, pal, loop=0), W, H, fr, pal, loop=0)
    # disposal 2 with a global palette whose background index is used, and frames with / without transparency
    fr = [frame_dict(noise(W, H), disp=1),
          frame_dict(noise(8, 6), x=3, y=2, disp=2, delay=4),
          frame_dict(noise(4, 4), x=12, y=1, disp=1, delay=4, trans=5)]
    save("anim_dispose2_background_index", write_gif(W, H, fr, pal, loop=0, bg=3), W, H, fr, pal, loop=0)
    # disposal 3 : the canvas is restored to what it was before the frame
    fr = [frame_dict(noise(W, H), disp=1),
          frame_dict(noise(8, 6), x=3, y=2, disp=3, delay=4),
          frame_dict(noise(9, 4), x=10, y=9, disp=3, delay=4, trans=1),
          frame_dict(noise(7, 7), x=1, y=1, disp=1, delay=4, trans=2),
          frame_dict(noise(10, 10), x=5, y=2, disp=3, delay=4),
          frame_dict(noise(4, 4), x=8, y=4, disp=1, delay=4)]
    save("anim_dispose3", write_gif(W, H, fr, pal, loop=0), W, H, fr, pal, loop=0)
    # disposal 3 on the FIRST frame (with and without a transparent index)
    fr = [frame_dict(noise(8, 6), x=2, y=2, disp=3, delay=4), frame_dict(noise(6, 5), x=9, y=3, disp=1, delay=4)]
    save("anim_dispose3_first", write_gif(W, H, fr, pal), W, H, fr, pal)
    fr = [frame_dict(noise(8, 6), x=2, y=2, disp=3, delay=4, trans=1), frame_dict(noise(6, 5), x=9, y=3, disp=1, delay=4)]
    save("anim_dispose3_first_transparent", write_gif(W, H, fr, pal), W, H, fr, pal)
    # reserved disposal values 4..7 and the unspecified value 0 after a specified one (Pillow keeps the old method)
    fr = [frame_dict(noise(W, H), disp=1),
          frame_dict(noise(8, 6), x=3, y=2, disp=4, delay=4),
          frame_dict(noise(5, 5), x=9, y=8, disp=0, delay=4),
          frame_dict(noise(6, 4), x=1, y=9, disp=7, delay=4, trans=1),
          frame_dict(noise(4, 4), x=0, y=0, disp=2, delay=4),
          frame_dict(noise(4, 4), x=5, y=5, disp=0, delay=4, gce=False),
          frame_dict(noise(W, H), disp=0, delay=4, trans=6)]
    save("anim_dispose_unspecified_and_reserved", write_gif(W, H, fr, pal), W, H, fr, pal)
    # mixed disposal sequence, local palettes of different sizes, an interlaced frame, delays across the range
    lp1 = rpal(rng, 4)
    lp2 = rpal(rng, 256)
    fr = [frame_dict(noise(W, H), disp=1, delay=65535),
          frame_dict(rng.integers(0, 4, size=(9, 9)), x=4, y=3, palette=lp1, disp=2, delay=1),
          frame_dict(rng.integers(0, 256, size=(14, 12)), x=5, y=0, palette=lp2, interlace=True, disp=3, trans=7, delay=33),
          frame_dict(noise(W, H), disp=0, delay=2, trans=5),
          frame_dict(rng.integers(0, 4, size=(6, 4)), x=0, y=8, palette=lp1, disp=2, delay=9, trans=0)]
    save("anim_mixed", write_gif(W, H, fr, pal, loop=7), W, H, fr, pal, loop=7)
    # first frame smaller than the screen: the rest of the canvas is transparent (Web) / palette entry 0 (Pillow)
    fr = [frame_dict(noise(8, 6), x=6, y=4, disp=1), frame_dict(noise(5, 5), x=0, y=0, disp=1, trans=4)]
    save("anim_first_partial", write_gif(W, H, fr, pal), W, H, fr, pal)
    fr = [frame_dict(noise(8, 6), x=6, y=4, disp=1, trans=3), frame_dict(noise(5, 5), x=0, y=0, disp=1)]
    save("anim_first_partial_transparent", write_gif(W, H, fr, pal), W, H, fr, pal)
    # frames that extend past the logical screen are clipped (Pillow grows its canvas instead: no Pillow truth)
    fr = [frame_dict(noise(12, 9), x=14, y=8, disp=1), frame_dict(noise(30, 30), x=2, y=1, disp=1, trans=2)]
    save("anim_clipped", write_gif(W, H, fr, pal), W, H, fr, pal)
    # frames entirely outside the screen draw nothing
    fr = [frame_dict(noise(W, H), disp=1), frame_dict(noise(4, 4), x=25, y=3, disp=2), frame_dict(noise(4, 4), x=3, y=30, disp=3)]
    save("anim_outside_screen", write_gif(W, H, fr, pal), W, H, fr, pal)
    # interlaced animation frames of different heights
    fr = [frame_dict(rng.integers(0, 16, size=(h, 13)), x=1, y=0, interlace=True, disp=1, delay=3, trans=0) for h in (14, 9, 5, 3, 1)]
    save("anim_interlaced", write_gif(14, 14, fr, pal, loop=0), 14, 14, fr, pal, loop=0)
    # many frames: crosses the decoder's 64-frame batch boundary, with every disposal mixed in
    fr = []
    for i in range(70):
        w, h = int(rng.integers(2, 7)), int(rng.integers(2, 6))
        fr.append(frame_dict(rng.integers(0, 16, size=(h, w)), x=int(rng.integers(0, 14 - w + 1)), y=int(rng.integers(0, 12 - h + 1)),
                             disp=[1, 2, 3, 0][i % 4], delay=1 + i, trans=(int(rng.integers(0, 16)) if i % 3 == 0 else None)))
    save("anim_70_frames", write_gif(14, 12, fr, pal, loop=0), 14, 12, fr, pal, loop=0)
    # noise frames big enough to fill the LZW table in every frame
    fr = [frame_dict(rng.integers(0, 256, size=(60, 80)), palette=rpal(rng, 256), disp=1, delay=2, mode=m)
          for m in ("clear_full", "deferred")]
    save("anim_noise_lzw", write_gif(80, 60, fr, None, loop=0), 80, 60, fr, None, loop=0)

    # ---- files written by Pillow's own encoder (truth = Pillow's decode; real-world LZW stream shapes) ---------
    yy, xx = np.mgrid[0:40, 0:56]
    frames_img = []
    for k in range(5):
        a = np.zeros((40, 56, 3), dtype=np.uint8)
        a[..., 0] = (xx * 4 + k * 30) % 256
        a[..., 1] = (yy * 6) % 256
        a[..., 2] = (xx * yy + k * 50) % 256
        frames_img.append(Image.fromarray(a).quantize(colors=64, dither=Image.Dither.NONE))
    for opt, nm in ((False, "pillow_anim"), (True, "pillow_anim_optimized")):
        buf = io.BytesIO()
        frames_img[0].save(buf, format="GIF", save_all=True, append_images=frames_img[1:], duration=[40, 80, 120, 60, 20],
                           loop=2, optimize=opt, disposal=1)
        pillow_written(nm, buf.getvalue(), 56, 40, loop=2)
    # RGBA frames with holes: Pillow writes transparency + (with disposal=2) restore-to-background
    for disposal, nm in ((2, "pillow_anim_rgba_dispose2"), (1, "pillow_anim_rgba_dispose1"), (3, "pillow_anim_rgba_dispose3")):
        rgba = []
        for k in range(4):
            a = np.zeros((30, 40, 4), dtype=np.uint8)
            a[..., 0] = (xx[:30, :40] * 6 + k * 40) % 256
            a[..., 1] = (yy[:30, :40] * 8) % 256
            a[..., 2] = 90 + 30 * k
            a[..., 3] = np.where(((xx[:30, :40] // 5 + yy[:30, :40] // 5 + k) % 3) == 0, 0, 255)
            rgba.append(Image.fromarray(a, "RGBA"))
        buf = io.BytesIO()
        rgba[0].save(buf, format="GIF", save_all=True, append_images=rgba[1:], duration=50, loop=0, disposal=disposal)
        pillow_written(nm, buf.getvalue(), 40, 30, loop=0)
    # a smooth 256 x 256 picture (a few thousand LZW codes per row band, table clears)
    gy, gx = np.mgrid[0:256, 0:256]
    a = np.zeros((256, 256, 3), dtype=np.uint8)
    a[..., 0] = gx
    a[..., 1] = gy
    a[..., 2] = (gx ^ gy)
    img = Image.fromarray(a).quantize(colors=256, dither=Image.Dither.FLOYDSTEINBERG)
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    pillow_written("pillow_still_256x256_dithered", buf.getvalue(), 256, 256, loop=None)
    # a Pillow-written still image with transparency
    a = np.zeros((20, 30, 4), dtype=np.uint8)
    a[..., 0] = np.arange(30)[None, :] * 8
    a[..., 1] = np.arange(20)[:, None] * 12
    a[..., 3] = np.where((np.arange(30)[None, :] + np.arange(20)[:, None]) % 5 == 0, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(a, "RGBA").save(buf, format="GIF", transparency=0)
    pillow_written("pillow_still_transparency", buf.getvalue(), 30, 20, loop=None)
    # a Pillow-written grayscale animation (mode "L" frames: identity gray ramp)
    gl = []
    for k in range(4):
        gl.append(Image.fromarray(((gx[:32, :48] * 5 + gy[:32, :48] * 3 + k * 60) % 256).astype(np.uint8), "L"))
    buf = io.BytesIO()
    gl[0].save(buf, format="GIF", save_all=True, append_images=gl[1:], duration=30, loop=0)
    pillow_written("pillow_anim_gray", buf.getvalue(), 48, 32, loop=0)

    make_fuzz()
    save_hostile(rng)


def pillow_written(name, data, W, H, loop):
    """A file Pillow wrote: the structure is read back from Pillow; BOTH truths are Pillow's output (the Web
    compositing of a Pillow-written animation is not derivable from the arguments, so no .web file is made)."""
    pim = Image.open(io.BytesIO(data))
    n = pim.n_frames
    descr = []
    for k in range(n):
        pim.seek(k)
        x0, y0, x1, y1 = pim.tile[0][1] if pim.tile else (0, 0, W, H)
        # Pillow may drop the fixed-delay GCE for stills; the disposal / transparency are what the file says
        descr.append(frame_dict(np.zeros((y1 - y0, x1 - x0), dtype=int), x=x0, y=y0,
                                delay=pim.info.get("duration", 0) // 10, disp=1 if n > 1 else 0,
                                trans=pim.info.get("transparency"), gce=True))
    got, status, pil_loop = pillow_truth(data, W, H, n)
    if got is None:
        raise SystemExit(f"{name}: Pillow could not be the oracle: {status}")
    save(name, data, W, H, descr, None, loop=loop, web_truth=None, exact_frames=False)


def fuzz_dir(directory, count, seed0):
    """Bulk random animations OUTSIDE the repo for a big one-off comparison:
         python -I make_gif_fixtures.py --fuzz-dir <dir> <count> <first seed>
       writes <dir>/f<seed>.gif and f<seed>.pil.rgba8 (all frames, Pillow's output) and f<seed>.json {width,height,frames};
       the test GifDecoderGpu_FuzzDirectory_MatchesPillow compares the decoder with them when SPAWNDEV_GIF_FUZZ_DIR is set."""
    os.makedirs(directory, exist_ok=True)
    made = 0
    for seed in range(seed0, seed0 + count):
        rng = np.random.default_rng(seed)
        kind = ("pal", "pal", "pal", "gray", "mixed")[seed % 5]
        W, H, frames, gp, bg = fuzz_case(rng, kind)
        data = write_gif(W, H, frames, gp, bg=bg)
        pil, status, _ = pillow_truth(data, W, H, len(frames))
        if pil is None:
            continue
        base = os.path.join(directory, f"f{seed}")
        with open(base + ".gif", "wb") as fp:
            fp.write(data)
        with open(base + ".pil.rgba8", "wb") as fp:
            fp.write(b"".join(p.tobytes() for p in pil))
        with open(base + ".json", "w") as fp:
            json.dump(dict(width=W, height=H, frames=len(frames)), fp)
        made += 1
    print(f"{made} of {count} random animations written to {directory}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fuzz-dir":
        fuzz_dir(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        sys.exit(0)
    # start from an empty fixture folder: stale files from an older layout would be embedded too
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    sys.exit(main())
