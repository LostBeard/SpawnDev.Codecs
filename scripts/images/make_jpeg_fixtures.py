#!/usr/bin/env python3
"""
Generates the JPEG codec test fixtures: SpawnDev.Codecs.Demo.Shared/TestData/Jpeg/<name>.jpg plus <name>.rgba8,
the exact pixels Pillow (libjpeg-turbo, ISLOW IDCT, fancy upsampling) decodes the file to, converted to RGBA8.
That decode is the oracle the GPU decoder must match BIT-EXACT.

Where each .jpg comes from:
  - Pillow's own libjpeg-turbo encoder: gray, 4:4:4 / 4:2:2 / 4:2:0, odd sizes, quality range, progressive
    (libjpeg's default progressive script: DC first + refine, AC first + refine, spectral selection and
    successive approximation), optimize=True tables, restart intervals, EXIF orientation 1..8, Adobe CMYK, Adobe RGB.
  - mini_encode() below, a small baseline JPEG writer, for what Pillow cannot write: 4:4:0 and 4:1:1 sampling,
    YCCK (Adobe transform 2), RGB without any marker (component ids 'R','G','B'), extended sequential (SOF1),
    other sampling layouts, a 12-bit header stub, and the deliberately broken files used by the error tests.
The writer only has to produce VALID files; correctness of the pixels is always Pillow's decode of the file.

Also writes TestData/Jpeg/std_tables.txt-free C# constants on stdout when run with --print-tables (the standard
Annex K Huffman and quantization tables are read back out of a Pillow-written file, so they cannot be mistyped).

Run:   <venv>/python -I scripts/images/make_jpeg_fixtures.py        (needs numpy + pillow)
"""
import hashlib
import io
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageOps

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "SpawnDev.Codecs.Demo.Shared", "TestData", "Jpeg")

ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7,
          14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39,
          46, 53, 60, 61, 54, 47, 55, 62, 63]


# ---------------------------------------------------------------------------------------------------------------
# Source pictures
# ---------------------------------------------------------------------------------------------------------------

def photo(w, h, seed, noise=6.0):
    """Photographic-style synthetic RGB picture: smooth gradients, soft blobs, hard edges, fine texture, noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        img[:, :, c] = 90 + 70 * np.sin(xx * (0.05 + 0.02 * c) + c) * np.cos(yy * (0.04 + 0.01 * c))
        for _ in range(4):
            cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(max(2, min(w, h) / 6), max(3, min(w, h) / 2))
            amp = rng.uniform(-70, 70)
            img[:, :, c] += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    # hard edges and a fine stripe texture so high-frequency coefficients are exercised
    img[:, : max(1, w // 3), 0] += 40
    img[(yy.astype(int) // 5) % 2 == 0, 1] -= 25
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def gray_photo(w, h, seed):
    return np.asarray(Image.fromarray(photo(w, h, seed)).convert("L"))


# ---------------------------------------------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------------------------------------------

written = []


def write_raw(name, data):
    with open(os.path.join(OUT, name + ".jpg"), "wb") as f:
        f.write(data)


def put_fixture(name, data, truth=True):
    """Write <name>.jpg and, when the file is decodable, <name>.rgba8 = Pillow's decode as RGBA."""
    write_raw(name, data)
    if truth:
        im = Image.open(io.BytesIO(data))
        im.load()
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
        with open(os.path.join(OUT, name + ".rgba8"), "wb") as f:
            f.write(rgba.tobytes())
        print(f"{name}: {len(data)} bytes, {im.size[0]}x{im.size[1]} {im.mode}")
    else:
        print(f"{name}: {len(data)} bytes (no truth)")
    written.append(name)


def pillow_save(arr, mode, **kw):
    im = Image.fromarray(arr, mode)
    b = io.BytesIO()
    im.save(b, "JPEG", **kw)
    return b.getvalue()


# ---------------------------------------------------------------------------------------------------------------
# Standard tables, read back out of a Pillow-written file (so they cannot be mistyped)
# ---------------------------------------------------------------------------------------------------------------

def segments(data):
    """Yield (marker, payload_offset, payload_length) for the header segments up to SOS."""
    i = 2
    while i < len(data):
        assert data[i] == 0xFF, (i, data[i])
        m = data[i + 1]
        if m == 0xDA:
            return
        n = (data[i + 2] << 8) | data[i + 3]
        yield m, i + 4, n - 2
        i += 2 + n


def extract_std_tables():
    d = pillow_save(photo(16, 16, 1), "RGB", quality=50, subsampling=2)
    quant = {}
    huff = {}
    for m, off, n in segments(d):
        if m == 0xDB:
            tq = d[off] & 15
            quant[tq] = list(d[off + 1: off + 65])          # zigzag order
        elif m == 0xC4:
            p = off
            end = off + n
            while p < end:
                tc, th = d[p] >> 4, d[p] & 15
                bits = list(d[p + 1: p + 17])
                cnt = sum(bits)
                huff[(tc, th)] = (bits, list(d[p + 17: p + 17 + cnt]))
                p += 17 + cnt
    return quant, huff


STD_QUANT, STD_HUFF = extract_std_tables()


def print_csharp_tables():
    def arr(a):
        return ", ".join(str(x) for x in a)
    print("// STD_QUANT (zigzag order), quality 50")
    for t in (0, 1):
        print(f"Quant{t}: {{ {arr(STD_QUANT[t])} }}")
    for (tc, th), (bits, vals) in sorted(STD_HUFF.items()):
        print(f"Huff tc={tc} th={th} bits: {{ {arr(bits)} }}")
        print(f"Huff tc={tc} th={th} vals: {{ {arr(vals)} }}")


def scaled_quant(q):
    """libjpeg jpeg_quality_scaling + jpeg_add_quant_table (force_baseline)."""
    q = max(1, min(100, q))
    scale = 5000 // q if q < 50 else 200 - q * 2
    out = {}
    for t in (0, 1):
        # STD_QUANT[t] is the quality-50 table: (base * 100 + 50) // 100 == base, so it IS the base table.
        out[t] = [max(1, min(255, (b * scale + 50) // 100)) for b in STD_QUANT[t]]
    return out


# ---------------------------------------------------------------------------------------------------------------
# Mini baseline JPEG writer (arbitrary sampling, restart intervals, Adobe / JFIF / no markers)
# ---------------------------------------------------------------------------------------------------------------

def huff_codes(bits, vals):
    codes = {}
    code = 0
    k = 0
    for ln in range(1, 17):
        for _ in range(bits[ln - 1]):
            codes[vals[k]] = (code, ln)
            code += 1
            k += 1
        code <<= 1
    return codes


class BitWriter:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.n = 0

    def put(self, code, ln):
        self.acc = (self.acc << ln) | (code & ((1 << ln) - 1))
        self.n += ln
        while self.n >= 8:
            b = (self.acc >> (self.n - 8)) & 0xFF
            self.out.append(b)
            if b == 0xFF:
                self.out.append(0)
            self.n -= 8
        self.acc &= (1 << self.n) - 1

    def flush(self):
        if self.n:
            self.put(0x7F, 8 - self.n)


DCT_MAT = np.array([[(np.sqrt(1 / 8) if u == 0 else np.sqrt(2 / 8)) * np.cos((2 * x + 1) * u * np.pi / 16)
                     for x in range(8)] for u in range(8)])


def bit_category(v):
    a = abs(v)
    return a.bit_length()


def encode_block(bw, coef, pred, dc_codes, ac_codes):
    """coef: 64 ints in zigzag order. Returns the new DC predictor."""
    diff = coef[0] - pred
    s = bit_category(diff)
    bw.put(*dc_codes[s])
    if s:
        bw.put(diff if diff > 0 else diff + (1 << s) - 1, s)
    run = 0
    for k in range(1, 64):
        v = coef[k]
        if v == 0:
            run += 1
            continue
        while run > 15:
            bw.put(*ac_codes[0xF0])
            run -= 16
        s = bit_category(v)
        bw.put(*ac_codes[(run << 4) | s])
        bw.put(v if v > 0 else v + (1 << s) - 1, s)
        run = 0
    if run:
        bw.put(*ac_codes[0x00])
    return coef[0]


def seg(marker, payload):
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


def mini_encode(planes, samp, width, height, quality=85, comp_ids=None, qsel=None, jfif=True, adobe=None,
                restart=0, sof=0xC0, precision=8, raw_scan=None, omit=(), separate_scans=False, extra_app=b"",
                qt16=False):
    """
    planes : list of 2-D uint8 arrays, one per component, ALREADY downsampled to their own size
             (ceil(width*h/maxh) x ceil(height*v/maxv)).
    samp   : list of (h, v) per component.
    adobe  : None, or the Adobe transform byte (0 / 1 / 2).
    raw_scan : bytes to use verbatim as the entropy-coded data (for deliberately broken files).
    omit   : header pieces to leave out, from {'dqt', 'dht', 'sof'}.
    separate_scans : one non-interleaved scan per component (real blocks only), instead of one interleaved scan.
    extra_app : raw bytes (whole marker segments) written right after the JFIF / Adobe markers.
    qt16   : write the DQT segments with 16-bit entries (Pq = 1).
    """
    nc = len(planes)
    maxh = max(h for h, _ in samp)
    maxv = max(v for _, v in samp)
    comp_ids = comp_ids or list(range(1, nc + 1))
    qsel = qsel or [0 if i == 0 else 1 for i in range(nc)]
    qt = scaled_quant(quality)
    mcus_x = -(-width // (8 * maxh))
    mcus_y = -(-height // (8 * maxv))

    # coefficient blocks per component, padded to whole MCUs by edge replication
    blocks = []
    for ci, p in enumerate(planes):
        h, v = samp[ci]
        bw_, bh_ = mcus_x * h, mcus_y * v
        padded = np.pad(p.astype(np.float64), ((0, bh_ * 8 - p.shape[0]), (0, bw_ * 8 - p.shape[1])), mode="edge")
        t = padded.reshape(bh_, 8, bw_, 8).transpose(0, 2, 1, 3) - 128.0
        c = np.einsum("ux,abxy,vy->abuv", DCT_MAT, t, DCT_MAT)
        qv = np.array(qt[qsel[ci]], dtype=np.float64)
        nat = np.zeros(64)
        for k in range(64):
            nat[ZIGZAG[k]] = qv[k]
        q = np.round(c.reshape(bh_, bw_, 64) / nat).astype(np.int64)
        blocks.append(q[:, :, ZIGZAG])           # zigzag order

    out = bytearray(b"\xff\xd8")
    if jfif:
        out += seg(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    if adobe is not None:
        out += seg(0xEE, b"Adobe\x00\x64\x00\x00\x00\x00" + bytes([adobe]))
    out += extra_app
    if 'dqt' not in omit:
        for t in sorted(set(qsel)):
            if qt16:
                out += seg(0xDB, bytes([0x10 | t]) + b"".join(struct.pack(">H", v) for v in qt[t]))
            else:
                out += seg(0xDB, bytes([t]) + bytes(qt[t]))
    if 'sof' not in omit:
        sofp = bytes([precision]) + struct.pack(">HH", height, width) + bytes([nc])
        for ci in range(nc):
            sofp += bytes([comp_ids[ci], (samp[ci][0] << 4) | samp[ci][1], qsel[ci]])
        out += seg(sof, sofp)
    use_tbl = sorted(set(qsel))
    if 'dht' not in omit:
        for th in use_tbl:
            for tc in (0, 1):
                bits, vals = STD_HUFF[(tc, th)]
                out += seg(0xC4, bytes([(tc << 4) | th]) + bytes(bits) + bytes(vals))
    if restart:
        out += seg(0xDD, struct.pack(">H", restart))
    def sos_for(cis):
        sosp = bytes([len(cis)])
        for ci in cis:
            sosp += bytes([comp_ids[ci], (qsel[ci] << 4) | qsel[ci]])
        return seg(0xDA, sosp + bytes([0, 63, 0]))

    def mcu_list(cis):
        """MCUs of a scan: each a list of (component, block row, block column)."""
        if len(cis) == 1:
            ci = cis[0]
            wib, hib = -(-planes[ci].shape[1] // 8), -(-planes[ci].shape[0] // 8)
            return [[(ci, r, c)] for r in range(hib) for c in range(wib)]
        res = []
        for my in range(mcus_y):
            for mx in range(mcus_x):
                res.append([(ci, my * samp[ci][1] + by, mx * samp[ci][0] + bx)
                            for ci in cis for by in range(samp[ci][1]) for bx in range(samp[ci][0])])
        return res

    scans = [[ci] for ci in range(nc)] if separate_scans else [list(range(nc))]
    dcs = {t: huff_codes(*STD_HUFF[(0, t)]) for t in use_tbl}
    acs = {t: huff_codes(*STD_HUFF[(1, t)]) for t in use_tbl}
    for si, cis in enumerate(scans):
        out += sos_for(cis)
        if raw_scan is not None:
            out += raw_scan
            continue
        preds = {ci: 0 for ci in cis}
        bw = BitWriter()
        rst = 0
        for mcu, blks in enumerate(mcu_list(cis)):
            if restart and mcu and mcu % restart == 0:
                bw.flush()
                out += bw.out
                out += bytes([0xFF, 0xD0 + (rst & 7)])
                rst += 1
                bw = BitWriter()
                preds = {ci: 0 for ci in cis}
            for ci, r, c in blks:
                preds[ci] = encode_block(bw, [int(x) for x in blocks[ci][r, c]], preds[ci], dcs[qsel[ci]], acs[qsel[ci]])
        bw.flush()
        out += bw.out
    out += b"\xff\xd9"
    return bytes(out)


def downsample(plane, fh, fv):
    """Box-average a full-resolution plane by integer factors (fh, fv); edge replicated to a whole multiple."""
    h, w = plane.shape
    ph, pw = -(-h // fv) * fv, -(-w // fh) * fh
    p = np.pad(plane.astype(np.float64), ((0, ph - h), (0, pw - w)), mode="edge")
    return np.round(p.reshape(ph // fv, fv, pw // fh, fh).mean(axis=(1, 3))).astype(np.uint8)


def rgb_to_ycc(rgb):
    r, g, b = [rgb[:, :, i].astype(np.float64) for i in range(3)]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    return [np.clip(np.round(p), 0, 255).astype(np.uint8) for p in (y, cb, cr)]


def sampled_planes(rgb, samp):
    maxh = max(h for h, _ in samp)
    maxv = max(v for _, v in samp)
    full = rgb_to_ycc(rgb)
    return [downsample(full[i], maxh // samp[i][0], maxv // samp[i][1]) for i in range(3)]


def cropped(plane, width, height, h, v, maxh, maxv):
    cw = -(-width * h // maxh)
    ch = -(-height * v // maxv)
    return plane[:ch, :cw]


# ---------------------------------------------------------------------------------------------------------------
# Fixture set
# ---------------------------------------------------------------------------------------------------------------

def main():
    if "--print-tables" in sys.argv:
        print_csharp_tables()
        return
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))

    sizes = [(1, 1), (7, 5), (17, 33), (100, 75)]

    # --- grayscale ---------------------------------------------------------------------------------------------
    for w, h in sizes:
        put_fixture(f"gray_{w}x{h}", pillow_save(gray_photo(w, h, 11), "L", quality=90))

    # --- Pillow colour: 4:4:4 / 4:2:2 / 4:2:0 x odd sizes ------------------------------------------------------
    for sub, nm in ((0, "444"), (1, "422"), (2, "420")):
        for w, h in sizes:
            put_fixture(f"c{nm}_{w}x{h}", pillow_save(photo(w, h, 20 + sub), "RGB", quality=90, subsampling=sub))

    # --- 4:4:0 / 4:1:1 / odd layouts via the mini writer -----------------------------------------------------
    layouts = {
        "c440": [(1, 2), (1, 1), (1, 1)],
        "c411": [(4, 1), (1, 1), (1, 1)],
        "c4x2": [(2, 2), (2, 1), (1, 1)],      # chroma planes of different ratios (h2v2 luma, Cb h1v2-ish, Cr 2x)
        "c1x4": [(1, 4), (1, 1), (1, 1)],      # integral 1x4 vertical
    }
    for nm, samp in layouts.items():
        maxh = max(h for h, _ in samp)
        maxv = max(v for _, v in samp)
        for w, h in ((17, 33), (66, 50)) if nm in ("c440", "c411") else ((17, 33),):
            rgb = photo(w, h, 30)
            planes = [cropped(p, w, h, *samp[i], maxh, maxv) for i, p in enumerate(sampled_planes(rgb, samp))]
            put_fixture(f"{nm}_{w}x{h}", mini_encode(planes, samp, w, h, quality=90))
    for w, h in ((7, 5), (1, 1)):
        rgb = photo(w, h, 31)
        samp = [(1, 2), (1, 1), (1, 1)]
        planes = [cropped(p, w, h, *samp[i], 1, 2) for i, p in enumerate(sampled_planes(rgb, samp))]
        put_fixture(f"c440_{w}x{h}", mini_encode(planes, samp, w, h, quality=90))

    # --- quality range -------------------------------------------------------------------------------------------
    base = photo(64, 48, 40)
    for q in (1, 10, 25, 50, 75, 90, 95, 100):
        put_fixture(f"q{q:03d}_420", pillow_save(base, "RGB", quality=q, subsampling=2))
    put_fixture("q100_444", pillow_save(base, "RGB", quality=100, subsampling=0))
    put_fixture("q001_gray", pillow_save(gray_photo(64, 48, 41), "L", quality=1))

    # --- progressive (libjpeg default script: spectral selection + successive approximation) ------------------
    for sub, nm in ((0, "444"), (1, "422"), (2, "420")):
        for w, h in ((17, 33), (66, 50)):
            put_fixture(f"prog_c{nm}_{w}x{h}", pillow_save(photo(w, h, 50 + sub), "RGB", quality=85,
                                                           subsampling=sub, progressive=True))
    for w, h in ((1, 1), (7, 5), (66, 50)):
        put_fixture(f"prog_gray_{w}x{h}", pillow_save(gray_photo(w, h, 55), "L", quality=85, progressive=True))
    put_fixture("prog_c420_7x5", pillow_save(photo(7, 5, 56), "RGB", quality=85, subsampling=2, progressive=True))
    put_fixture("prog_c420_q100", pillow_save(base, "RGB", quality=100, subsampling=2, progressive=True))
    put_fixture("prog_c420_q5", pillow_save(base, "RGB", quality=5, subsampling=2, progressive=True))

    # --- optimized Huffman tables ----------------------------------------------------------------------------------
    for sub, nm in ((0, "444"), (2, "420")):
        put_fixture(f"opt_c{nm}_66x50", pillow_save(photo(66, 50, 60 + sub), "RGB", quality=85, subsampling=sub,
                                                     optimize=True))
    put_fixture("opt_gray_66x50", pillow_save(gray_photo(66, 50, 63), "L", quality=85, optimize=True))

    # --- restart intervals (baseline + progressive; Pillow: MCUs per interval via blocks / rows) -------------------
    rbase = photo(66, 50, 70)
    for blocks_ in (1, 2, 5, 7):
        put_fixture(f"rst_b{blocks_}_c420", pillow_save(rbase, "RGB", quality=85, subsampling=2,
                                                        restart_marker_blocks=blocks_))
    put_fixture("rst_rows1_c444", pillow_save(rbase, "RGB", quality=85, subsampling=0, restart_marker_rows=1))
    put_fixture("rst_b3_gray", pillow_save(gray_photo(66, 50, 71), "L", quality=85, restart_marker_blocks=3))
    put_fixture("prog_rst_b4_c420", pillow_save(rbase, "RGB", quality=85, subsampling=2, progressive=True,
                                                restart_marker_blocks=4))
    put_fixture("prog_rst_b1_gray", pillow_save(gray_photo(66, 50, 72), "L", quality=85, progressive=True,
                                                restart_marker_blocks=1))
    put_fixture("opt_rst_b6_c422", pillow_save(rbase, "RGB", quality=85, subsampling=1, optimize=True,
                                               restart_marker_blocks=6))
    # a restart interval written by the mini writer on a 4:1:1 picture
    rgb = photo(66, 50, 73)
    samp = [(4, 1), (1, 1), (1, 1)]
    planes = [cropped(p, 66, 50, *samp[i], 4, 1) for i, p in enumerate(sampled_planes(rgb, samp))]
    put_fixture("rst_n3_c411", mini_encode(planes, samp, 66, 50, quality=88, restart=3))

    # --- EXIF orientation 1..8 (24x16 so the files stay tiny) ------------------------------------------------------
    obase = photo(24, 16, 80)
    for o in range(1, 9):
        exif = Image.Exif()
        exif[0x0112] = o
        data = pillow_save(obase, "RGB", quality=90, subsampling=2, exif=exif.tobytes())
        put_fixture(f"orient_{o}", data)
        im = Image.open(io.BytesIO(data))
        im.load()
        # Pillow's own transpose for the EXIF value (independent of our kernel)
        t = ImageOps.exif_transpose(im.copy())
        rgba = np.asarray(t.convert("RGBA"), dtype=np.uint8)
        with open(os.path.join(OUT, f"orient_{o}.oriented.rgba8"), "wb") as f:
            f.write(np.array([t.size[0], t.size[1]], dtype="<i4").tobytes() + rgba.tobytes())

    # --- CMYK / YCCK / RGB / extended sequential / odd marker layouts --------------------------------------------
    rgbs = photo(33, 17, 90)
    cmyk = np.concatenate([255 - rgbs, np.zeros((17, 33, 1), dtype=np.uint8) + 30], axis=2)
    cmyk[:, :, 3] = np.clip(gray_photo(33, 17, 91).astype(np.int32) // 3, 0, 255).astype(np.uint8)
    put_fixture("cmyk_adobe_33x17", pillow_save(cmyk, "CMYK", quality=90))
    put_fixture("cmyk_adobe_q50_33x17", pillow_save(cmyk, "CMYK", quality=50))
    # YCCK: Adobe transform 2. The YCC triple encodes the (inverted) RGB-form of C,M,Y; K is stored as is.
    inv = 255 - cmyk[:, :, :3]
    ycc = rgb_to_ycc(inv)
    kplane = cmyk[:, :, 3]
    # Pillow inverts CMYK on save, so what lies in the file is 255-C ...: for YCCK we write stored values directly.
    stored_k = 255 - kplane
    put_fixture("ycck_adobe_33x17", mini_encode([ycc[0], ycc[1], ycc[2], stored_k],
                                                [(1, 1)] * 4, 33, 17, quality=90, qsel=[0, 1, 1, 0],
                                                jfif=False, adobe=2))
    put_fixture("ycck_420_adobe_33x17", mini_encode(
        [ycc[0], downsample(ycc[1], 2, 2)[:9, :17], downsample(ycc[2], 2, 2)[:9, :17], stored_k],
        [(2, 2), (1, 1), (1, 1), (2, 2)], 33, 17, quality=90, qsel=[0, 1, 1, 0], jfif=False, adobe=2))
    put_fixture("rgb_adobe0_33x17", pillow_save(rgbs, "RGB", quality=90, keep_rgb=True))
    full = [rgbs[:, :, i] for i in range(3)]
    put_fixture("rgb_ids_nomarker_33x17", mini_encode(full, [(1, 1)] * 3, 33, 17, quality=90,
                                                      comp_ids=[ord('R'), ord('G'), ord('B')], qsel=[0, 0, 0],
                                                      jfif=False, adobe=None))
    put_fixture("rgb_adobe0_nomarkerids_33x17", mini_encode(full, [(1, 1)] * 3, 33, 17, quality=90, qsel=[0, 0, 0],
                                                            jfif=False, adobe=0))
    ycc3 = rgb_to_ycc(rgbs)
    put_fixture("ycc_nomarker_33x17", mini_encode(ycc3, [(1, 1)] * 3, 33, 17, quality=90, jfif=False, adobe=None))
    put_fixture("ycc_adobe1_33x17", mini_encode(ycc3, [(1, 1)] * 3, 33, 17, quality=90, jfif=False, adobe=1))
    put_fixture("extseq_c420_33x17", mini_encode(
        [ycc3[0], downsample(ycc3[1], 2, 2)[:9, :17], downsample(ycc3[2], 2, 2)[:9, :17]],
        [(2, 2), (1, 1), (1, 1)], 33, 17, quality=90, sof=0xC1))

    # --- structure variants the Pillow encoder cannot write ------------------------------------------------------------
    put_fixture("cmyk_noadobe_33x17", mini_encode([cmyk[:, :, i] for i in range(4)], [(1, 1)] * 4, 33, 17, quality=90,
                                                  qsel=[0, 0, 0, 0], jfif=False, adobe=None))
    put_fixture("prog_cmyk_33x17", pillow_save(cmyk, "CMYK", quality=90, progressive=True))
    g17 = gray_photo(17, 33, 95)
    put_fixture("gray_sampling22_17x33", mini_encode([g17], [(2, 2)], 17, 33, quality=90))
    rgb33 = photo(33, 17, 96)
    y3 = rgb_to_ycc(rgb33)
    s420 = [(2, 2), (1, 1), (1, 1)]
    p420 = [y3[0], downsample(y3[1], 2, 2)[:9, :17], downsample(y3[2], 2, 2)[:9, :17]]
    put_fixture("seq_multiscan_c420_33x17", mini_encode(p420, s420, 33, 17, quality=90, separate_scans=True))
    put_fixture("seq_multiscan_c444_33x17", mini_encode(y3, [(1, 1)] * 3, 33, 17, quality=90, separate_scans=True))
    put_fixture("seq_multiscan_rst_c420_33x17", mini_encode(p420, s420, 33, 17, quality=90, separate_scans=True, restart=3))
    put_fixture("qt16_c420_33x17", mini_encode(p420, s420, 33, 17, quality=90, qt16=True))
    exif_le = bytes.fromhex("457869660000" "49492a0008000000" "0100" "120103000100000006000000" "00000000")
    xmp = b"http://ns.adobe.com/xap/1.0/" + bytes([0]) + b"<x:xmpmeta xmlns:x='adobe:ns:meta/'/>"
    put_fixture("orient_le_6", mini_encode(p420, s420, 33, 17, quality=90, extra_app=seg(0xE1, exif_le)))
    put_fixture("orient_xmp_first_6", mini_encode(p420, s420, 33, 17, quality=90,
                                                  extra_app=seg(0xE1, xmp) + seg(0xE1, exif_le) + seg(0xFE, b"comment")))

    # --- larger photographic-style picture -------------------------------------------------------------------------
    big = photo(160, 120, 100)
    put_fixture("photo_c420_160x120", pillow_save(big, "RGB", quality=85, subsampling=2))
    put_fixture("photo_prog_c420_160x120", pillow_save(big, "RGB", quality=85, subsampling=2, progressive=True,
                                                       optimize=True))

    # --- encoder fixtures: the SAME source pixels encoded by Pillow (libjpeg-turbo); the GPU encoder must produce
    #     the identical bytes. Sources are RGBA with a noisy alpha channel (alpha must be ignored). ---------------
    def enc_source(name, rgb):
        h, w = rgb.shape[:2]
        alpha = np.random.default_rng(w * 1000 + h).integers(0, 256, (h, w, 1), dtype=np.uint8)
        with open(os.path.join(OUT, f"encsrc_{name}.rgba8"), "wb") as f:
            f.write(np.concatenate([rgb, alpha], axis=2).tobytes())

    esizes = [(1, 1), (7, 5), (17, 33), (100, 75), (64, 48)]
    for w, h in esizes:
        rgb = photo(w, h, 200 + w)
        enc_source(f"{w}x{h}", rgb)
        for sub, nm in ((0, "444"), (1, "422"), (2, "420")):
            for q in ((50, 90) if (w, h) != (64, 48) else (1, 25, 75, 100)):
                write_raw(f"enc_c{nm}_q{q:03d}_{w}x{h}", pillow_save(rgb, "RGB", quality=q, subsampling=sub))
    for w, h in ((17, 33), (100, 75), (1, 1)):
        g = gray_photo(w, h, 210 + w)
        rgb = np.stack([g, g, g], axis=2)
        enc_source(f"gray_{w}x{h}", rgb)
        write_raw(f"enc_gray_q075_{w}x{h}", pillow_save(g, "L", quality=75))
    rgb = photo(100, 75, 300)
    enc_source("rst100x75", rgb)
    for n in (1, 3, 7):
        write_raw(f"enc_rst_n{n}_c420_100x75", pillow_save(rgb, "RGB", quality=85, subsampling=2, restart_marker_blocks=n))
    write_raw("enc_rst_n5_c444_100x75", pillow_save(rgb, "RGB", quality=85, subsampling=0, restart_marker_blocks=5))
    write_raw("enc_rst_n2_c422_100x75", pillow_save(rgb, "RGB", quality=85, subsampling=1, restart_marker_blocks=2))
    g = gray_photo(100, 75, 301)
    enc_source("rstgray100x75", np.stack([g, g, g], axis=2))
    write_raw("enc_rst_n4_gray_100x75", pillow_save(g, "L", quality=85, restart_marker_blocks=4))
    rgb = photo(24, 16, 302)
    enc_source("orient24x16", rgb)
    for o in (3, 6, 8):
        e = Image.Exif()
        e[0x0112] = o
        write_raw(f"enc_orient{o}_c420_24x16", pillow_save(rgb, "RGB", quality=90, subsampling=2, exif=e.tobytes()))

    # --- production-scale pictures (1024x768): the truth is a SHA-256 of Pillow's RGBA decode, so the folder stays small --
    bigrgb = photo(1024, 768, 400, noise=2.0)

    def put_big(name, data):
        write_raw(name, data)
        im = Image.open(io.BytesIO(data))
        im.load()
        digest = hashlib.sha256(np.asarray(im.convert("RGBA"), dtype=np.uint8).tobytes()).hexdigest()
        with open(os.path.join(OUT, name + ".sha256"), "w") as f:
            f.write(digest)
        print(f"{name}: {len(data)} bytes, sha256 {digest[:16]}")

    put_big("big_c420_1024x768", pillow_save(bigrgb, "RGB", quality=80, subsampling=2))
    put_big("big_prog_c444_1024x768", pillow_save(bigrgb, "RGB", quality=75, subsampling=0, progressive=True, optimize=True))
    put_big("big_rst_c422_1024x768", pillow_save(bigrgb, "RGB", quality=80, subsampling=1, restart_marker_blocks=64))
    # The encoder gets the pixels Pillow decoded from big_c420 (our decoder reproduces them bit-exact) and must reproduce
    # Pillow's encoding of those pixels byte for byte (SHA-256 of the file).
    dec_big = Image.open(io.BytesIO(pillow_save(bigrgb, "RGB", quality=80, subsampling=2))).convert("RGB")
    for nm, kw in (("bigenc_c420_q85", dict(quality=85, subsampling=2)),
                   ("bigenc_c444_q95", dict(quality=95, subsampling=0)),
                   ("bigenc_rst64_c420_q85", dict(quality=85, subsampling=2, restart_marker_blocks=64))):
        b = io.BytesIO()
        dec_big.save(b, "JPEG", **kw)
        with open(os.path.join(OUT, nm + ".sha256"), "w") as f:
            f.write(hashlib.sha256(b.getvalue()).hexdigest())
        print(f"{nm}: {len(b.getvalue())} bytes")

    # --- broken files (error tests; no truth) ------------------------------------------------------------------------
    tiny = photo(16, 16, 110)
    good = pillow_save(tiny, "RGB", quality=85, subsampling=2)
    goodp = pillow_save(tiny, "RGB", quality=85, subsampling=2, progressive=True)
    goodr = pillow_save(photo(48, 32, 111), "RGB", quality=85, subsampling=2, restart_marker_blocks=2)

    def sos_end(d):
        i = 2
        while True:
            m = d[i + 1]
            n = (d[i + 2] << 8) | d[i + 3]
            if m == 0xDA:
                return i + 2 + n
            i += 2 + n

    def find_marker(d, m, start=2):
        i = start
        while i < len(d) - 3:
            if d[i] == 0xFF and d[i + 1] == m:
                return i
            i += 1
        raise ValueError(m)

    put_fixture("bad_soi", b"\x00" + good[1:], truth=False)
    put_fixture("bad_cut_in_dqt", good[:find_marker(good, 0xDB) + 20], truth=False)
    se = sos_end(good)
    put_fixture("bad_cut_in_scan", good[:se + (len(good) - se) // 2], truth=False)
    put_fixture("bad_cut_in_scan_prog", goodp[:sos_end(goodp) + 40], truth=False)
    pp = goodp[:find_marker(goodp, 0xDA, find_marker(goodp, 0xDA) + 2)]       # progressive, whole first scan only
    put_fixture("bad_prog_no_eoi", pp, truth=False)
    put_fixture("bad_no_sof", good[:find_marker(good, 0xC0) + 1] + b"\xfe" + good[find_marker(good, 0xC0) + 2:],
                truth=False)
    put_fixture("bad_no_dht", good[:find_marker(good, 0xC4) + 1] + b"\xfe" + good[find_marker(good, 0xC4) + 2:],
                truth=False)
    sof = find_marker(good, 0xC0)
    put_fixture("bad_12bit", good[:sof + 4] + bytes([12]) + good[sof + 5:], truth=False)
    put_fixture("bad_arithmetic", good[:sof + 1] + b"\xc9" + good[sof + 2:], truth=False)
    put_fixture("bad_lossless", good[:sof + 1] + b"\xc3" + good[sof + 2:], truth=False)
    put_fixture("bad_zero_width", good[:sof + 7] + b"\x00\x00" + good[sof + 9:], truth=False)
    put_fixture("bad_zero_height", good[:sof + 5] + b"\x00\x00" + good[sof + 7:], truth=False)
    put_fixture("bad_sampling_factor", good[:sof + 11] + bytes([0x51]) + good[sof + 12:], truth=False)
    # missing DQT for the component's table
    put_fixture("bad_no_dqt", good[:find_marker(good, 0xDB) + 1] + b"\xfe" + good[find_marker(good, 0xDB) + 2:],
                truth=False)
    # restart marker removed: 2 bytes cut out of the entropy data at the first RST
    r1 = find_marker(goodr, 0xD0, sos_end(goodr))
    put_fixture("bad_restart_missing", goodr[:r1] + goodr[r1 + 2:], truth=False)
    # restart marker with the wrong number
    put_fixture("bad_restart_order", goodr[:r1 + 1] + bytes([0xD3]) + goodr[r1 + 2:], truth=False)
    # an AC symbol stream that can only be a bad Huffman code (all ones: the longest code of every std table is not all ones)
    samp = [(1, 1)] * 3
    planes = sampled_planes(tiny, samp)
    put_fixture("bad_huffman_code", mini_encode(planes, samp, 16, 16, raw_scan=b"\xff\x00" * 40), truth=False)
    # DC category 0, then five ZRL symbols (0xF0): run walks past coefficient 63
    bw = BitWriter()
    dc0 = huff_codes(*STD_HUFF[(0, 0)])
    ac0 = huff_codes(*STD_HUFF[(1, 0)])
    bw.put(*dc0[0])
    for _ in range(5):
        bw.put(*ac0[0xF0])
    bw.flush()
    gray_tiny = gray_photo(8, 8, 112)
    put_fixture("bad_run_overflow", mini_encode([gray_tiny], [(1, 1)], 8, 8, qsel=[0], jfif=False,
                                                 raw_scan=bytes(bw.out)), truth=False)
    # 12-bit extended sequential header (decoder must refuse with a clear reason)
    put_fixture("bad_12bit_sof1", mini_encode([gray_tiny], [(1, 1)], 8, 8, qsel=[0], jfif=False, sof=0xC1,
                                               precision=12), truth=False)
    # three components, one with sampling the decoder cannot upsample (fractional ratio 3:2)
    samp = [(3, 1), (2, 1), (1, 1)]
    frac = mini_encode([np.zeros((8, 24), np.uint8), np.zeros((8, 16), np.uint8), np.zeros((8, 8), np.uint8)],
                       samp, 24, 8, quality=50)
    put_fixture("bad_fractional_sampling", frac, truth=False)

    # more structural breakage, each a one-field change of a good file
    dht0 = find_marker(good, 0xC4)                      # first DHT: DC table 0, counts at +5 .. +21
    cnt = bytearray(good)
    cnt[dht0 + 5:dht0 + 21] = bytes([4, 4, 4] + [0] * 13)   # 12 codes: 4 of length 1 is impossible
    put_fixture("bad_huffman_oversubscribed", bytes(cnt), truth=False)
    sos = find_marker(good, 0xDA)
    put_fixture("bad_sos_unknown_component", good[:sos + 5] + bytes([9]) + good[sos + 6:], truth=False)
    dqt = find_marker(good, 0xDB)
    put_fixture("bad_dqt_short", good[:dqt + 2] + bytes([0, 60]) + good[dqt + 4:], truth=False)
    put_fixture("bad_marker_junk", good[:dqt] + bytes([0x41]) + good[dqt:], truth=False)
    sofpos = find_marker(good, 0xC0)
    sof_len = (good[sofpos + 2] << 8) | good[sofpos + 3]
    put_fixture("bad_sof_twice", good[:sofpos] + good[sofpos:sofpos + 2 + sof_len] * 2 + good[sofpos + 2 + sof_len:], truth=False)
    psos = find_marker(goodp, 0xDA)
    ns = goodp[psos + 4]
    se_at = psos + 4 + 1 + 2 * ns + 1                   # Ss at +1+2ns, Se right after
    bad_prog = bytearray(goodp)
    bad_prog[se_at] = 5                                  # a DC scan (Ss = 0) must have Se = 0
    put_fixture("bad_progressive_params", bytes(bad_prog), truth=False)
    put_fixture("bad_no_scan", good[:find_marker(good, 0xDA)] + bytes([0xFF, 0xD9]), truth=False)
    put_fixture("bad_only_soi_eoi", bytes([0xFF, 0xD8, 0xFF, 0xD9]), truth=False)
    put_fixture("bad_too_short", bytes([0xFF, 0xD8, 0xFF]), truth=False)

    dcbad = bytearray(good)
    dcbad[dht0 + 21 + 11] = 200                         # last symbol of DC table 0 (a DC category must be <= 15)
    put_fixture("bad_huffman_dc_symbol", bytes(dcbad), truth=False)
    ms = mini_encode(y3, [(1, 1)] * 3, 33, 17, quality=90, separate_scans=True)
    sos_positions = [i for i in range(len(ms) - 1) if ms[i] == 0xFF and ms[i + 1] == 0xDA]
    put_fixture("bad_missing_component_scan", ms[:sos_positions[2]] + bytes([0xFF, 0xD9]), truth=False)
    one = mini_encode([gray_tiny], [(1, 1)], 8, 8, qsel=[0], jfif=False)
    one_sos = [i for i in range(len(one) - 1) if one[i] == 0xFF and one[i + 1] == 0xDA][0]
    put_fixture("bad_too_many_scans", one[:one_sos] + one[one_sos:-2] * 130 + bytes([0xFF, 0xD9]), truth=False)
    big_dims = bytearray(good)
    big_dims[sofpos + 5:sofpos + 9] = bytes([0xFF, 0xFF, 0xFF, 0xFF])
    put_fixture("bad_too_large", bytes(big_dims), truth=False)
    dup = bytearray(good)
    dup[sos + 7] = dup[sos + 5]
    put_fixture("bad_sos_duplicate_component", bytes(dup), truth=False)
    tsel = bytearray(good)
    tsel[sos + 6] = 0x40
    put_fixture("bad_sos_table_selector", bytes(tsel), truth=False)
    put_fixture("bad_mcu_too_many_blocks", mini_encode([np.zeros((16, 16), np.uint8)] * 4, [(2, 2)] * 4, 16, 16,
                                                       qsel=[0, 0, 0, 0], jfif=False), truth=False)

    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"\n{len(written)} fixtures, {total / 1024:.0f} KiB total")


if __name__ == "__main__":
    sys.exit(main())
