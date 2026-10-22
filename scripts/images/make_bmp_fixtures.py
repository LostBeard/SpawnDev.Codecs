#!/usr/bin/env python3
"""
Generates the BMP decoder test fixtures: SpawnDev.Codecs.Demo.Shared/TestData/Bmp/<name>.bmp plus the exact
expected pixels <name>.rgba8 (RGBA, rows top to bottom).

Why a hand-rolled BMP *writer* and not Pillow: the tests need every format Pillow cannot write (RLE4 / RLE8,
RGB555 / RGB565 / 4-4-4-4 / 1-5-5-5 / 10-10-10-2 bitfields, V2..V5 headers, BITMAPCOREHEADER, top-down, palettes
shorter than 2^bpp). The truth is the sample array a file is built FROM (never a decoder), converted to RGBA by
the conventions the decoder documents:
  - channels narrower than 8 bits expand by bit replication (the pattern repeated, top 8 bits kept; computed
    below with a string trick, independent of the C# integer code);
  - a 32-bit image with an alpha channel whose values are ALL zero is opaque (browser behaviour);
  - palette indices beyond the palette are opaque black; the reserved byte of a palette entry is ignored;
  - RLE pixels the stream never covers are transparent black (0, 0, 0, 0).
As an independent cross-check every file is also opened with Pillow wherever Pillow can read it; differences are
printed as NOTE lines (Pillow differs on purpose for RLE gaps and bit expansion, see the notes in the output).

Run:   <venv>/python -I scripts/images/make_bmp_fixtures.py        (needs numpy + pillow)
"""
import os
import struct
import sys

import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "SpawnDev.Codecs.Demo.Shared", "TestData", "Bmp")
os.makedirs(OUT, exist_ok=True)

BI_RGB, BI_RLE8, BI_RLE4, BI_BITFIELDS, BI_ALPHABITFIELDS = 0, 1, 2, 3, 6


def expand8(v: int, bits: int) -> int:
    """Bit replication to 8 bits: repeat the bit pattern until >= 8 bits, keep the first 8 (string trick)."""
    if bits == 0:
        return 0
    if bits >= 8:
        return v >> (bits - 8)
    pat = format(v, "0%db" % bits)
    s = pat * (8 // bits + 1)
    return int(s[:8], 2)


def dib_header(size, w, h, bpp, comp, image_size, clr_used, masks):
    """masks: (r, g, b, a) used for header sizes >= 52 (and appended by the caller for size 40)."""
    if size == 12:
        return struct.pack("<IHHHH", 12, w, h, 1, bpp)
    base = struct.pack("<IiiHHIIiiII", 40, w, h, 1, bpp, comp, image_size, 2835, 2835, clr_used, 0)
    if size == 40:
        return base
    r, g, b, a = masks
    if size == 52:
        return struct.pack("<IiiHHIIiiII", 52, w, h, 1, bpp, comp, image_size, 2835, 2835, clr_used, 0) + \
            struct.pack("<III", r, g, b)
    if size == 56:
        return struct.pack("<IiiHHIIiiII", 56, w, h, 1, bpp, comp, image_size, 2835, 2835, clr_used, 0) + \
            struct.pack("<IIII", r, g, b, a)
    v4 = struct.pack("<IiiHHIIiiII", size, w, h, 1, bpp, comp, image_size, 2835, 2835, clr_used, 0) + \
        struct.pack("<IIII", r, g, b, a) + struct.pack("<I", 0x57696E20) + bytes(36 + 12)
    if size == 108:
        return v4
    assert size == 124
    return v4 + struct.pack("<IIII", 4, 0, 0, 0)   # intent, profile data, profile size, reserved


def build_bmp(w, h_signed, bpp, comp, pixels, palette_entries=None, header=40, masks=(0, 0, 0, 0),
              clr_used=0, entry_size=4, reserved_byte=0, rng=None):
    """palette_entries: (n, 3) RGB array or None. pixels: the raw pixel-data bytes (already padded)."""
    hdr = dib_header(header, w, h_signed, bpp, comp, len(pixels), clr_used, masks)
    extra = b""
    if header == 40 and comp == BI_BITFIELDS:
        extra = struct.pack("<III", masks[0], masks[1], masks[2])
    if header == 40 and comp == BI_ALPHABITFIELDS:
        extra = struct.pack("<IIII", *masks)
    pal = b""
    if palette_entries is not None:
        for i, (r, g, b) in enumerate(palette_entries):
            rb = reserved_byte if rng is None else int(rng.integers(0, 256))
            pal += bytes([b, g, r]) if entry_size == 3 else bytes([b, g, r, rb])
    off = 14 + len(hdr) + len(extra) + len(pal)
    total = off + len(pixels)
    return b"BM" + struct.pack("<IHHI", total, 0, 0, off) + hdr + extra + pal + pixels


def pad_rows(rows, align=4):
    out = bytearray()
    for r in rows:
        r = bytes(r)
        out += r + bytes((-len(r)) % align)
    return bytes(out)


def pack_indices(idx, bpp):
    """idx: (h, w) ints; returns list of unpadded row bytes."""
    rows = []
    for row in idx:
        if bpp == 8:
            rows.append(bytes(int(v) for v in row))
        else:
            per = 8 // bpp
            out = bytearray()
            for i in range(0, len(row), per):
                byte = 0
                for k in range(per):
                    v = int(row[i + k]) if i + k < len(row) else 0
                    byte = (byte << bpp) | v
                out.append(byte)
            rows.append(bytes(out))
    return rows


def order(rows, top_down):
    return list(rows) if top_down else list(reversed(rows))


def save(name, bmp, rgba, notes=""):
    with open(os.path.join(OUT, name + ".bmp"), "wb") as f:
        f.write(bmp)
    with open(os.path.join(OUT, name + ".rgba8"), "wb") as f:
        f.write(np.asarray(rgba, dtype=np.uint8).tobytes())
    status = ""
    try:
        im = Image.open(os.path.join(OUT, name + ".bmp"))
        im.load()
        got = np.asarray(im.convert("RGBA"), dtype=np.int64)
        truth = np.asarray(rgba, dtype=np.int64)
        if got.shape != truth.shape:
            status = f"  NOTE {name}: Pillow size {got.shape} != truth {truth.shape}"
        elif not np.array_equal(got, truth):
            d = np.argwhere(got != truth)
            status = (f"  NOTE {name}: Pillow differs at {len(d)} samples (first {d[0].tolist()}: "
                      f"pillow={got[tuple(d[0])]} truth={truth[tuple(d[0])]}) {notes}")
        else:
            status = "  (Pillow agrees)"
    except Exception as ex:  # Pillow cannot read every variant
        status = f"  (Pillow cannot read: {type(ex).__name__}: {str(ex)[:60]})"
    print(f"{name}: {len(bmp)} bytes{status}")


def palette_truth(idx, palette):
    h, w = idx.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, 3] = 255
    pal = np.asarray(palette, dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            v = int(idx[y, x])
            if v < len(pal):
                out[y, x, :3] = pal[v]
    return out


def mask_of(shift, bits):
    return ((1 << bits) - 1) << shift if bits else 0


def bitfield_image(rng, w, h, bpp, layout, alpha_all_zero=False):
    """layout: dict channel -> (shift, bits). Returns (pixel_bytes_rows, truth_rgba, masks)."""
    samples = {}
    for ch in "rgba":
        sh, bits = layout.get(ch, (0, 0))
        if bits == 0:
            samples[ch] = np.zeros((h, w), dtype=np.int64)
        elif ch == "a" and alpha_all_zero:
            samples[ch] = np.zeros((h, w), dtype=np.int64)
        else:
            samples[ch] = rng.integers(0, 1 << bits, size=(h, w)).astype(np.int64)
    rows = []
    for y in range(h):
        row = bytearray()
        for x in range(w):
            v = 0
            for ch in "rgba":
                sh, bits = layout.get(ch, (0, 0))
                v |= int(samples[ch][y, x]) << sh
            row += struct.pack("<H" if bpp == 16 else "<I", v)
        rows.append(bytes(row))
    truth = np.zeros((h, w, 4), dtype=np.uint8)
    has_alpha = layout.get("a", (0, 0))[1] > 0
    any_alpha = bool((samples["a"] != 0).any())
    for y in range(h):
        for x in range(w):
            for ci, ch in enumerate("rgb"):
                truth[y, x, ci] = expand8(int(samples[ch][y, x]), layout[ch][1])
            if has_alpha and any_alpha:
                truth[y, x, 3] = expand8(int(samples["a"][y, x]), layout["a"][1])
            else:
                truth[y, x, 3] = 255
    masks = tuple(mask_of(*layout.get(ch, (0, 0))) for ch in "rgba")
    return rows, truth, masks


# ---------------------------------------------------------------- RLE


def rle_bytes(ops, rle4):
    out = bytearray()
    for op in ops:
        kind = op[0]
        if kind == "run":
            out += bytes([op[1], op[2]])
        elif kind == "abs":
            vals = op[1]
            assert len(vals) >= 3
            out += bytes([0, len(vals)])
            if rle4:
                data = bytearray()
                for i in range(0, len(vals), 2):
                    hi = vals[i]
                    lo = vals[i + 1] if i + 1 < len(vals) else 0
                    data.append((hi << 4) | lo)
            else:
                data = bytearray(vals)
            out += data
            if len(data) % 2:
                out.append(0)
        elif kind == "eol":
            out += bytes([0, 0])
        elif kind == "eob":
            out += bytes([0, 1])
        elif kind == "delta":
            out += bytes([0, 2, op[1], op[2]])
    return bytes(out)


def rle_truth(ops, w, h, rle4, palette):
    """Simulate the stream: x, y count from the bottom-left; uncovered pixels stay (0, 0, 0, 0)."""
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    def put(x, y, idx):
        assert 0 <= x < w and 0 <= y < h, (x, y)
        r, g, b = palette[idx] if idx < len(palette) else (0, 0, 0)
        canvas[h - 1 - y, x] = (r, g, b, 255)

    x = y = 0
    for op in ops:
        kind = op[0]
        if kind == "run":
            for k in range(op[1]):
                idx = op[2] if not rle4 else ((op[2] >> 4) if k % 2 == 0 else (op[2] & 15))
                put(x, y, idx)
                x += 1
        elif kind == "abs":
            for v in op[1]:
                put(x, y, v)
                x += 1
        elif kind == "eol":
            x = 0
            y += 1
        elif kind == "delta":
            x += op[1]
            y += op[2]
        elif kind == "eob":
            break
    return canvas


def main():
    rng = np.random.default_rng(0x424D50)

    # ---- palette images, bottom-up and top-down, odd sizes (row padding) -------------------------------
    for bpp, w, h in ((1, 19, 7), (1, 32, 5), (1, 33, 4), (1, 8, 3), (4, 21, 9), (4, 8, 4), (8, 17, 13), (8, 4, 5)):
        n = 1 << bpp
        pal = rng.integers(0, 256, size=(n, 3))
        idx = rng.integers(0, n, size=(h, w))
        rows = pack_indices(idx, bpp)
        truth = palette_truth(idx, pal)
        save(f"pal{bpp}_{w}x{h}",
             build_bmp(w, h, bpp, BI_RGB, pad_rows(order(rows, False)), pal, rng=rng), truth)
    # top-down palette image (negative height)
    pal = rng.integers(0, 256, size=(256, 3))
    idx = rng.integers(0, 256, size=(9, 15))
    save("pal8_topdown", build_bmp(15, -9, 8, BI_RGB, pad_rows(order(pack_indices(idx, 8), True)), pal),
         palette_truth(idx, pal))
    # palette shorter than 2^bpp: indices beyond it are opaque black
    for bpp, used in ((4, 5), (8, 100), (1, 1)):
        n = 1 << bpp
        pal = rng.integers(0, 256, size=(used, 3))
        idx = rng.integers(0, n, size=(7, 13))
        rows = pack_indices(idx, bpp)
        save(f"pal{bpp}_short{used}",
             build_bmp(13, 7, bpp, BI_RGB, pad_rows(order(rows, False)), pal, clr_used=used),
             palette_truth(idx, pal))
    # clrUsed larger than 2^bpp is clamped
    pal = rng.integers(0, 256, size=(16, 3))
    idx = rng.integers(0, 16, size=(5, 9))
    save("pal4_clrused_too_big",
         build_bmp(9, 5, 4, BI_RGB, pad_rows(order(pack_indices(idx, 4), False)), pal, clr_used=16),
         palette_truth(idx, pal))
    # 1x1
    pal = rng.integers(0, 256, size=(2, 3))
    idx = np.array([[1]])
    save("pal1_1x1", build_bmp(1, 1, 1, BI_RGB, pad_rows(pack_indices(idx, 1)), pal), palette_truth(idx, pal))

    # ---- 24-bit --------------------------------------------------------------------------------------
    for w, h, td in ((13, 11, False), (13, 11, True), (1, 1, False), (4, 3, False), (7, 1, False), (1, 9, False)):
        img = rng.integers(0, 256, size=(h, w, 3)).astype(np.uint8)
        rows = [bytes(px for p in row for px in (p[2], p[1], p[0])) for row in img]
        truth = np.concatenate([img, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)
        save(f"rgb24_{w}x{h}{'_topdown' if td else ''}",
             build_bmp(w, -h if td else h, 24, BI_RGB, pad_rows(order(rows, td))), truth)

    # ---- 32-bit BI_RGB: 4th byte is alpha (non-zero) / padding (all zero => opaque) -----------------------
    w, h = 11, 6
    img = rng.integers(0, 256, size=(h, w, 4)).astype(np.uint8)
    img[0, 0, 3] = 7   # guarantee a non-zero alpha value
    rows = [bytes(px for p in row for px in (p[2], p[1], p[0], p[3])) for row in img]
    save("rgb32_alpha", build_bmp(w, h, 32, BI_RGB, pad_rows(order(rows, False))), img)
    img0 = img.copy()
    img0[:, :, 3] = 0
    rows = [bytes(px for p in row for px in (p[2], p[1], p[0], p[3])) for row in img0]
    truth = img0.copy()
    truth[:, :, 3] = 255
    save("rgb32_alpha_all_zero_is_opaque", build_bmp(w, h, 32, BI_RGB, pad_rows(order(rows, False))), truth)
    # V4 / V5 header, BI_RGB 32-bit with alpha (masks in the header are not used for BI_RGB)
    for hs in (108, 124):
        rows = [bytes(px for p in row for px in (p[2], p[1], p[0], p[3])) for row in img]
        save(f"rgb32_alpha_v{4 if hs == 108 else 5}",
             build_bmp(w, h, 32, BI_RGB, pad_rows(order(rows, False)), header=hs,
                       masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)), img)

    # ---- BITFIELDS / ALPHABITFIELDS, every header layout ---------------------------------------------
    argb32 = {"r": (16, 8), "g": (8, 8), "b": (0, 8), "a": (24, 8)}
    rgb32 = {"r": (16, 8), "g": (8, 8), "b": (0, 8)}
    rows, truth, masks = bitfield_image(rng, 9, 5, 32, argb32)
    for hs, nm in ((56, "v3"), (108, "v4"), (124, "v5")):
        save(f"bf32_argb_{nm}", build_bmp(9, 5, 32, BI_BITFIELDS, pad_rows(order(rows, False)), header=hs, masks=masks), truth)
    save("bf32_argb_info40_alphabitfields",
         build_bmp(9, 5, 32, BI_ALPHABITFIELDS, pad_rows(order(rows, False)), header=40, masks=masks), truth)
    rows3, truth3, masks3 = bitfield_image(rng, 9, 5, 32, rgb32)
    save("bf32_rgb_info40", build_bmp(9, 5, 32, BI_BITFIELDS, pad_rows(order(rows3, False)), header=40, masks=masks3), truth3)
    save("bf32_rgb_v2", build_bmp(9, 5, 32, BI_BITFIELDS, pad_rows(order(rows3, False)), header=52, masks=masks3), truth3)
    # alpha mask present but every alpha value 0: opaque
    rowsz, truthz, masksz = bitfield_image(rng, 9, 5, 32, argb32, alpha_all_zero=True)
    save("bf32_argb_v4_alpha_all_zero_is_opaque",
         build_bmp(9, 5, 32, BI_BITFIELDS, pad_rows(order(rowsz, False)), header=108, masks=masksz), truthz)
    # generic masks: 10-10-10-2, and a swapped channel order (BGRA in the dword)
    l1010 = {"r": (20, 10), "g": (10, 10), "b": (0, 10), "a": (30, 2)}
    r1, t1, m1 = bitfield_image(rng, 8, 4, 32, l1010)
    save("bf32_10_10_10_2", build_bmp(8, 4, 32, BI_BITFIELDS, pad_rows(order(r1, False)), header=108, masks=m1), t1)
    lswap = {"r": (0, 8), "g": (8, 8), "b": (16, 8), "a": (24, 8)}
    r2, t2, m2 = bitfield_image(rng, 6, 5, 32, lswap)
    save("bf32_abgr_swapped", build_bmp(6, 5, 32, BI_BITFIELDS, pad_rows(order(r2, False)), header=108, masks=m2), t2)

    # 16-bit: BI_RGB (555), BITFIELDS 565, 4-4-4-4, 1-5-5-5, 5-5-5 BITFIELDS with a V4 header
    r555 = {"r": (10, 5), "g": (5, 5), "b": (0, 5)}
    rows, truth, _ = bitfield_image(rng, 11, 5, 16, r555)
    save("rgb16_555", build_bmp(11, 5, 16, BI_RGB, pad_rows(order(rows, False))), truth)
    save("rgb16_555_topdown", build_bmp(11, -5, 16, BI_RGB, pad_rows(order(rows, True))), truth)
    r565 = {"r": (11, 5), "g": (5, 6), "b": (0, 5)}
    rows, truth, masks = bitfield_image(rng, 13, 4, 16, r565)
    save("bf16_565", build_bmp(13, 4, 16, BI_BITFIELDS, pad_rows(order(rows, False)), header=40, masks=masks), truth)
    save("bf16_565_v4", build_bmp(13, 4, 16, BI_BITFIELDS, pad_rows(order(rows, False)), header=108, masks=masks), truth)
    r4444 = {"r": (8, 4), "g": (4, 4), "b": (0, 4), "a": (12, 4)}
    rows, truth, masks = bitfield_image(rng, 7, 5, 16, r4444)
    save("bf16_4444_alpha", build_bmp(7, 5, 16, BI_BITFIELDS, pad_rows(order(rows, False)), header=108, masks=masks), truth)
    r1555 = {"r": (10, 5), "g": (5, 5), "b": (0, 5), "a": (15, 1)}
    rows, truth, masks = bitfield_image(rng, 9, 4, 16, r1555)
    save("bf16_1555_alpha", build_bmp(9, 4, 16, BI_ALPHABITFIELDS, pad_rows(order(rows, False)), header=40, masks=masks), truth)

    # ---- BITMAPCOREHEADER (OS/2 1.x): 3-byte palette entries, unsigned 16 bit dimensions ---------------
    for bpp, w, h in ((1, 10, 5), (4, 9, 5), (8, 7, 6)):
        n = 1 << bpp
        pal = rng.integers(0, 256, size=(n, 3))
        idx = rng.integers(0, n, size=(h, w))
        save(f"core_pal{bpp}", build_bmp(w, h, bpp, BI_RGB, pad_rows(order(pack_indices(idx, bpp), False)), pal,
                                         header=12, entry_size=3), palette_truth(idx, pal))
    img = rng.integers(0, 256, size=(5, 7, 3)).astype(np.uint8)
    rows = [bytes(px for p in row for px in (p[2], p[1], p[0])) for row in img]
    save("core_rgb24", build_bmp(7, 5, 24, BI_RGB, pad_rows(order(rows, False)), header=12),
         np.concatenate([img, np.full((5, 7, 1), 255, dtype=np.uint8)], axis=2))

    # ---- RLE8 / RLE4 -------------------------------------------------------------------------------------
    pal = rng.integers(0, 256, size=(256, 3))
    w, h = 16, 6
    ops = []
    for row in range(h):
        if row == 0:
            ops += [("run", 16, 3)]
        elif row == 1:
            ops += [("run", 4, 200), ("abs", [1, 2, 3, 4, 5]), ("run", 7, 9)]       # odd absolute run (padded)
        elif row == 2:
            ops += [("abs", [9, 8, 7, 6, 5, 4]), ("run", 10, 255)]                  # even absolute run
        elif row == 3:
            ops += [("run", 1, 17), ("run", 15, 18)]
        elif row == 4:
            ops += [("abs", list(range(10, 26)))]
        else:
            ops += [("run", 8, 0), ("run", 8, 1)]
        ops += [("eol",)] if row < h - 1 else []
    ops += [("eol",), ("eob",)]
    truth = rle_truth(ops, w, h, False, pal)
    save("rle8_basic", build_bmp(w, h, 8, BI_RLE8, rle_bytes(ops, False), pal), truth)
    # delta / end-of-line skipping leaves uncovered (transparent) pixels; ends without EOL before EOB
    w, h = 12, 7
    ops = [("run", 3, 5), ("delta", 4, 0), ("run", 2, 6), ("eol",),
           ("delta", 0, 2), ("run", 12, 7), ("eol",),
           ("abs", [1, 2, 3]), ("delta", 5, 1), ("abs", [4, 5, 6, 7]), ("eol",), ("run", 1, 9), ("eob",)]
    truth = rle_truth(ops, w, h, False, pal)
    save("rle8_delta_gaps", build_bmp(w, h, 8, BI_RLE8, rle_bytes(ops, False), pal), truth,
         "(Pillow fills gaps with palette index 0; the decoder leaves them transparent)")
    # stream with no end-of-bitmap code
    ops = [("run", 4, 2), ("eol",), ("run", 4, 3), ("eol",)]
    truth = rle_truth(ops, 4, 2, False, pal)
    save("rle8_no_eob", build_bmp(4, 2, 8, BI_RLE8, rle_bytes(ops, False), pal), truth)

    pal16 = rng.integers(0, 256, size=(16, 3))
    w, h = 17, 5
    ops = [("run", 17, 0x12), ("eol",),
           ("abs", [1, 2, 3]), ("run", 5, 0xAB), ("abs", [15, 14, 13, 12, 11]), ("run", 4, 0x7C), ("eol",),
           ("abs", [3, 2, 1, 0]), ("abs", [9, 9, 9, 9, 9, 9, 9]), ("run", 6, 0x45), ("eol",),
           ("run", 1, 0x90), ("delta", 2, 0), ("run", 3, 0xF1), ("eol",),
           ("run", 16, 0x55), ("run", 1, 0x30), ("eol",), ("eob",)]
    truth = rle_truth(ops, w, h, True, pal16)
    save("rle4_basic", build_bmp(w, h, 4, BI_RLE4, rle_bytes(ops, True), pal16), truth,
         "(gap handling differs from Pillow)")

    # ---- a larger noisy picture of each common kind -----------------------------------------------------
    img = rng.integers(0, 256, size=(48, 64, 4)).astype(np.uint8)
    img[5, 5, 3] = 1
    rows = [bytes(px for p in row for px in (p[2], p[1], p[0], p[3])) for row in img]
    save("rgb32_noise_64x48", build_bmp(64, 48, 32, BI_RGB, pad_rows(order(rows, False))), img)


if __name__ == "__main__":
    sys.exit(main())
