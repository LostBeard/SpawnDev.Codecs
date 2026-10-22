#!/usr/bin/env python3
"""
Generates the PNG decoder test fixtures: SpawnDev.Codecs.Demo.Shared/TestData/Png/<name>.png plus the
exact expected pixels <name>.rgba8 (and <name>.rgba16 for 16-bit files).

Why a hand-rolled PNG *writer* and not Pillow: the tests need every colour type x bit depth x interlace
combination, every scanline filter type forced on purpose, multi-chunk IDAT, tRNS colour keys and palette
alpha. Pillow cannot write gray 2/4-bit or force filters. The truth is the sample array the file is built
FROM (not any decoder), converted to RGBA by the PNG spec's definitions:
  - samples below 8 bits scale by 255 / (2^depth - 1) (exact); 16-bit -> 8-bit takes the high byte
    (libpng png_set_strip_16, Pillow); 8-bit -> 16-bit replicates (v * 257); <8 -> 16: v * 65535 / max.
  - tRNS colour key: pixel is fully transparent when ALL its samples equal the key.
As an independent cross-check every file is also opened with Pillow (libpng-equivalent decoder) and its
RGBA8 must equal the truth wherever Pillow can represent the file.

Run:   <venv>/python -I scripts/images/make_png_fixtures.py        (needs numpy + pillow)
"""
import os
import struct
import sys
import zlib

import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "SpawnDev.Codecs.Demo.Shared", "TestData", "Png")
os.makedirs(OUT, exist_ok=True)

ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def pack_row(samples: np.ndarray, depth: int) -> bytes:
    """samples: 1-D int array of per-channel samples for one scanline (already interleaved)."""
    if depth == 8:
        return samples.astype(np.uint8).tobytes()
    if depth == 16:
        return samples.astype(">u2").tobytes()
    per_byte = 8 // depth
    n = len(samples)
    padded = np.zeros(((n + per_byte - 1) // per_byte) * per_byte, dtype=np.uint8)
    padded[:n] = samples
    out = bytearray()
    for i in range(0, len(padded), per_byte):
        b = 0
        for k in range(per_byte):
            b = (b << depth) | int(padded[i + k])
        out.append(b)
    return bytes(out)


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def filter_row(ftype: int, cur: bytes, prev: bytes, bpp: int) -> bytes:
    out = bytearray(len(cur))
    for i in range(len(cur)):
        x = cur[i]
        a = cur[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        if ftype == 0:
            v = x
        elif ftype == 1:
            v = x - a
        elif ftype == 2:
            v = x - b
        elif ftype == 3:
            v = x - ((a + b) >> 1)
        else:
            v = x - paeth(a, b, c)
        out[i] = v & 0xFF
    return bytes(out)


def encode_image(samples: np.ndarray, ctype: int, depth: int, interlace: bool, filters, idat_chunk: int,
                 plte=None, trns=None) -> bytes:
    h, w, ch = samples.shape
    assert ch == CHANNELS[ctype]
    bpp = max(1, (ch * depth + 7) // 8)
    raw = bytearray()
    passes = ADAM7 if interlace else [(0, 0, 1, 1)]
    row_counter = 0
    for xs, ys, dx, dy in passes:
        sub = samples[ys::dy, xs::dx, :]
        if sub.shape[0] == 0 or sub.shape[1] == 0:
            continue
        prev = bytes(len(pack_row(sub[0].reshape(-1), depth)))
        for r in range(sub.shape[0]):
            cur = pack_row(sub[r].reshape(-1), depth)
            ft = filters[row_counter % len(filters)]
            row_counter += 1
            raw.append(ft)
            raw += filter_row(ft, cur, prev, bpp)
            prev = cur
    comp = zlib.compress(bytes(raw), 6)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 1 if interlace else 0))
    if plte is not None:
        png += chunk(b"PLTE", bytes(np.asarray(plte, dtype=np.uint8).reshape(-1)))
    if trns is not None:
        png += chunk(b"tRNS", trns)
    for i in range(0, len(comp), idat_chunk):
        png += chunk(b"IDAT", comp[i:i + idat_chunk])
    png += chunk(b"IEND", b"")
    return png


def to_rgba(samples, ctype, depth, plte=None, palpha=None, key=None):
    """Truth: RGBA at 8 and 16 bits from the sample array, by the PNG spec's definitions."""
    h, w, _ = samples.shape
    mx = (1 << depth) - 1
    s = samples.astype(np.int64)

    def scale8(v):
        if depth == 16:
            return v >> 8
        if depth == 8:
            return v
        return v * (255 // mx)

    def scale16(v):
        if depth == 16:
            return v
        if depth == 8:
            return v * 257
        return v * (65535 // mx)

    rgba8 = np.zeros((h, w, 4), dtype=np.int64)
    rgba16 = np.zeros((h, w, 4), dtype=np.int64)
    if ctype == 3:
        pal = np.asarray(plte, dtype=np.int64)
        idx = s[:, :, 0]
        for c in range(3):
            rgba8[:, :, c] = pal[idx, c]
            rgba16[:, :, c] = pal[idx, c] * 257
        al = np.full(256, 255, dtype=np.int64)
        if palpha is not None:
            al[:len(palpha)] = palpha
        rgba8[:, :, 3] = al[idx]
        rgba16[:, :, 3] = al[idx] * 257
        return rgba8, rgba16
    if ctype in (0, 4):
        for c in range(3):
            rgba8[:, :, c] = scale8(s[:, :, 0])
            rgba16[:, :, c] = scale16(s[:, :, 0])
        a8 = scale8(s[:, :, 1]) if ctype == 4 else np.full((h, w), 255)
        a16 = scale16(s[:, :, 1]) if ctype == 4 else np.full((h, w), 65535)
    else:
        for c in range(3):
            rgba8[:, :, c] = scale8(s[:, :, c])
            rgba16[:, :, c] = scale16(s[:, :, c])
        a8 = scale8(s[:, :, 3]) if ctype == 6 else np.full((h, w), 255)
        a16 = scale16(s[:, :, 3]) if ctype == 6 else np.full((h, w), 65535)
    if key is not None:
        match = np.ones((h, w), dtype=bool)
        for c, kv in enumerate(key):
            match &= (s[:, :, c] == (kv & mx))
        a8 = np.where(match, 0, a8)
        a16 = np.where(match, 0, a16)
    rgba8[:, :, 3] = a8
    rgba16[:, :, 3] = a16
    return rgba8, rgba16


def make_samples(rng, w, h, ch, depth, smooth=False):
    mx = (1 << depth) - 1
    if smooth:
        yy, xx = np.mgrid[0:h, 0:w]
        planes = []
        for c in range(ch):
            planes.append(((xx * (3 + c) + yy * (5 - c) + c * 40) * mx // max(1, (w * 5 + h * 5))) & mx)
        arr = np.stack(planes, axis=2).astype(np.int64)
        # sprinkle noise so filters have something to chew on
        noise = rng.integers(0, 2, size=arr.shape)
        return np.clip(arr + noise, 0, mx)
    return rng.integers(0, mx + 1, size=(h, w, ch)).astype(np.int64)


def write_fixture(name, samples, ctype, depth, interlace=False, filters=(0, 1, 2, 3, 4), idat_chunk=1 << 16,
                  plte=None, palpha=None, key=None):
    trns = None
    if ctype == 3 and palpha is not None:
        trns = bytes(palpha)
    if key is not None:
        trns = b"".join(struct.pack(">H", k) for k in key)
    png = encode_image(samples, ctype, depth, interlace, list(filters), idat_chunk, plte, trns)
    rgba8, rgba16 = to_rgba(samples, ctype, depth, plte, palpha, key)
    with open(os.path.join(OUT, name + ".png"), "wb") as f:
        f.write(png)
    with open(os.path.join(OUT, name + ".rgba8"), "wb") as f:
        f.write(rgba8.astype(np.uint8).tobytes())
    if depth == 16 or ctype == 3:
        with open(os.path.join(OUT, name + ".rgba16"), "wb") as f:
            f.write(rgba16.astype("<u2").tobytes())
    # Independent cross-check with Pillow.
    im = Image.open(os.path.join(OUT, name + ".png"))
    im.load()
    got = np.asarray(im.convert("RGBA"), dtype=np.int64)
    if not np.array_equal(got, rgba8):
        d = np.argwhere(got != rgba8)
        print(f"  NOTE {name}: Pillow RGBA8 differs from truth at {len(d)} samples (first {d[0].tolist()}: "
              f"pillow={got[tuple(d[0])]} truth={rgba8[tuple(d[0])]})")
    print(f"{name}: {len(png)} bytes")


def main():
    rng = np.random.default_rng(0x504E47)

    # Every colour type x bit depth x interlace.
    combos = [
        (0, 1), (0, 2), (0, 4), (0, 8), (0, 16),
        (2, 8), (2, 16),
        (3, 1), (3, 2), (3, 4), (3, 8),
        (4, 8), (4, 16),
        (6, 8), (6, 16),
    ]
    names = {0: "gray", 2: "rgb", 3: "pal", 4: "graya", 6: "rgba"}
    for ctype, depth in combos:
        for il in (False, True):
            w, h = (19, 11) if il else (13, 9)
            ch = CHANNELS[ctype]
            s = make_samples(rng, w, h, ch, depth)
            plte = palpha = None
            if ctype == 3:
                n = 1 << depth
                plte = rng.integers(0, 256, size=(n, 3))
                palpha = list(rng.integers(0, 256, size=max(1, n // 2))) if depth in (2, 8) else None
            nm = f"{names[ctype]}{depth}{'_il' if il else ''}"
            write_fixture(nm, s, ctype, depth, il, plte=plte, palpha=palpha)

    # Colour keys.
    for ctype, depth in ((0, 8), (0, 16), (0, 4), (2, 8), (2, 16)):
        w, h = 16, 8
        ch = CHANNELS[ctype]
        mx = (1 << depth) - 1
        s = rng.integers(0, 4, size=(h, w, ch)).astype(np.int64) * (mx // 3)   # few distinct values -> many key hits
        keyvals = [int(s[2, 3, c]) for c in range(ch)]
        write_fixture(f"{names[ctype]}{depth}_key", s, ctype, depth, False, key=keyvals)

    # Each filter type forced for every row, on an RGBA image with a smooth + noisy picture.
    for ft in range(5):
        s = make_samples(rng, 40, 24, 4, 8, smooth=True)
        write_fixture(f"rgba8_filter{ft}", s, 6, 8, False, filters=(ft,))
    for ft in range(5):
        s = make_samples(rng, 21, 17, 1, 4, smooth=True)
        write_fixture(f"gray4_filter{ft}", s, 0, 4, False, filters=(ft,))

    # Degenerate sizes.
    write_fixture("rgba8_1x1", make_samples(rng, 1, 1, 4, 8), 6, 8)
    write_fixture("rgba8_64x1", make_samples(rng, 64, 1, 4, 8), 6, 8)
    write_fixture("rgba8_1x64", make_samples(rng, 1, 64, 4, 8), 6, 8)
    write_fixture("rgb8_3x2_il", make_samples(rng, 3, 2, 3, 8), 2, 8, True)   # most Adam7 passes empty
    write_fixture("gray1_1x1_il", make_samples(rng, 1, 1, 1, 1), 0, 1, True)

    # Many IDAT chunks, bigger picture.
    write_fixture("rgba8_multiidat", make_samples(rng, 300, 200, 4, 8, smooth=True), 6, 8, False, idat_chunk=1500)
    write_fixture("rgb8_il_multiidat", make_samples(rng, 211, 97, 3, 8, smooth=True), 2, 8, True, idat_chunk=997)


if __name__ == "__main__":
    sys.exit(main())
