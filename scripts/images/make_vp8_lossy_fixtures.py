#!/usr/bin/env python3
"""
Generates the VP8 lossy keyframe fixtures: SpawnDev.Codecs.Demo.Shared/TestData/Vp8Lossy/<name>.vp8 (one raw VP8 keyframe,
exactly what sits in a WebP 'VP8 ' chunk or an IVF frame), plus the exact decoded planes as the truth:
    <name>.yuv     Y plane (w*h), then U, then V (ceil(w/2) x ceil(h/2) each)   -- when w*h <= SMALL_PIXELS
    <name>.sha256  hex SHA-256 of that same yuv420p byte string                -- for larger pictures
and manifest.csv, one row per fixture with the header facts read by an INDEPENDENT python VP8 header parser below
(filter type / level / sharpness / lf deltas / segmentation / partitions / quantizer / skip), so the C# tests can assert
that the corpus really covers each keyframe feature.

THE TRUTH NEVER COMES FROM OUR OWN CODE. It is the native VP8 decoder of ffmpeg (`-c:v vp8`, yuv420p out, verified native
below: the stream is already yuv420p, so no scaler runs). For fixtures that came from libwebp the script also decodes the
.webp with ffmpeg's webp decoder and with libwebp's own `dwebp -yuv -nodither`, and aborts unless all three agree
byte for byte (ffmpeg-vp8 on the extracted payload, ffmpeg-webp on the file, libwebp).

Where the .vp8 files come from:
  * Pillow (libwebp 1.6): lossy, quality 1..100, method 0..6, default 4 segments + normal filter. Sizes 1x1 .. 640x480.
  * cwebp 1.5.0 (CLI): simple vs normal filter (-nostrong), sharpness 0..7, filter strength, autofilter, segments 1..4, sns.
    (libwebp's encoder writes ONE token partition always: WebPConfig.partitions has no effect, verified by identical
    output for partitions 0..3; multi-partition streams therefore come from libvpx below.)
  * ffmpeg + libvpx (VP8 encoder, one keyframe): token partitions, -sharpness, versions 0..3 (version 1 = simple filter),
    and libvpx's mode_ref_lf_delta (B_PRED loop-filter delta) which libwebp never writes.

Tools (env vars override the defaults): FFMPEG, CWEBP, DWEBP. python needs numpy, pillow.
Run:   <venv>/python -I scripts/images/make_vp8_lossy_fixtures.py
"""
import hashlib
import io
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "Vp8Lossy")
SCRATCH = os.environ.get("VP8_SCRATCH") or tempfile.mkdtemp(prefix="vp8fix_")

FFMPEG = os.environ.get("FFMPEG", r"C:\Users\TJ\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe")
CWEBP = os.environ.get("CWEBP", "cwebp")
DWEBP = os.environ.get("DWEBP", "dwebp")

SMALL_PIXELS = 40000   # at most this many pixels: store the full yuv; larger: store only its SHA-256


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
    img[:, : max(1, w // 3), 0] += 40
    img[(yy.astype(int) // 5) % 2 == 0, 1] -= 25
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def noise_pic(w, h, seed):
    """Heavy white noise plus a gradient: huge coefficients (DCT_CAT6 tokens), almost every MB is B_PRED."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h, w, 3))
    yy, xx = np.mgrid[0:h, 0:w]
    base[:, :, 0] = (base[:, :, 0] * 0.7 + xx * 255 / max(1, w - 1) * 0.3)
    return np.clip(base, 0, 255).astype(np.uint8)


def text_pic(w, h, seed):
    """Hard black/white edges at many angles: exercises every 4x4 sub-block predictor and strong loop filtering."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 235, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(max(6, (w * h) // 600)):
        kind = int(rng.integers(0, 3))
        col = rng.integers(0, 120, 3)
        if kind == 0:
            x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
            ww, hh = int(rng.integers(1, max(2, w // 3))), int(rng.integers(1, max(2, h // 3)))
            img[y0:y0 + hh, x0:x0 + ww] = col
        elif kind == 1:
            a, b, c = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, w)
            mask = np.abs(a * xx + b * yy - c) < 1.2
            img[mask] = col
        else:
            cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(2, max(3, min(w, h) / 3))
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
            img[mask] = col
    return img


def flat_pic(w, h, seed):
    """Large flat regions with a few shapes: almost every macroblock has no coefficients (mb_skip_coeff)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (60, 140, 200)
    img[h // 2:, :] = (200, 180, 40)
    for _ in range(3):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        ww, hh = int(rng.integers(2, max(3, w // 4))), int(rng.integers(2, max(3, h // 4)))
        img[y0:y0 + hh, x0:x0 + ww] = rng.integers(0, 256, 3)
    return img


KINDS = {"photo": photo, "noise": noise_pic, "text": text_pic, "flat": flat_pic}


def make(kind, w, h, seed):
    return KINDS[kind](w, h, seed)


# ---------------------------------------------------------------------------------------------------------------
# Independent VP8 header parser (RFC 6386 sec 9.2 - 9.11, 19.2). Only used for the manifest and the self-checks.
# ---------------------------------------------------------------------------------------------------------------

class BoolDec:
    def __init__(self, data):
        self.d = data
        self.p = 2
        self.value = (data[0] << 8 | data[1]) if len(data) >= 2 else 0
        self.range = 255
        self.bits = 0

    def bool(self, prob):
        split = 1 + (((self.range - 1) * prob) >> 8)
        big = split << 8
        if self.value >= big:
            ret = 1
            self.range -= split
            self.value -= big
        else:
            ret = 0
            self.range = split
        while self.range < 128:
            self.value <<= 1
            self.range <<= 1
            self.bits += 1
            if self.bits == 8:
                self.bits = 0
                if self.p < len(self.d):
                    self.value |= self.d[self.p]
                self.p += 1
        return ret

    def lit(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bool(128)
        return v

    def flag_signed(self, n):
        if self.lit(1):
            v = self.lit(n)
            return -v if self.lit(1) else v
        return 0


def parse_frame_tag_and_header(frame):
    tag = frame[0] | frame[1] << 8 | frame[2] << 16
    if tag & 1:
        raise ValueError("not a keyframe")
    version = (tag >> 1) & 7
    part0 = tag >> 5
    if frame[3:6] != b"\x9d\x01\x2a":
        raise ValueError("bad start code")
    w = (frame[6] | frame[7] << 8) & 0x3FFF
    h = (frame[8] | frame[9] << 8) & 0x3FFF
    bd = BoolDec(frame[10:10 + part0])
    hdr = {"version": version, "width": w, "height": h, "part0": part0}
    hdr["color_space"] = bd.lit(1)
    hdr["clamp"] = bd.lit(1)
    seg = bd.lit(1)
    hdr["seg"] = seg
    hdr["seg_map"] = hdr["seg_data"] = hdr["seg_abs"] = 0
    hdr["seg_q"] = [0] * 4
    hdr["seg_lf"] = [0] * 4
    if seg:
        hdr["seg_map"] = bd.lit(1)
        hdr["seg_data"] = bd.lit(1)
        if hdr["seg_data"]:
            hdr["seg_abs"] = bd.lit(1)
            hdr["seg_q"] = [bd.flag_signed(7) for _ in range(4)]
            hdr["seg_lf"] = [bd.flag_signed(6) for _ in range(4)]
        if hdr["seg_map"]:
            for _ in range(3):
                if bd.lit(1):
                    bd.lit(8)
    hdr["filter_type"] = bd.lit(1)
    hdr["filter_level"] = bd.lit(6)
    hdr["sharpness"] = bd.lit(3)
    hdr["lf_delta"] = bd.lit(1)
    hdr["ref_delta"] = [0] * 4
    hdr["mode_delta"] = [0] * 4
    if hdr["lf_delta"]:
        if bd.lit(1):
            hdr["ref_delta"] = [bd.flag_signed(6) for _ in range(4)]
            hdr["mode_delta"] = [bd.flag_signed(6) for _ in range(4)]
    hdr["log2_parts"] = bd.lit(2)
    hdr["q"] = bd.lit(7)
    hdr["qdeltas"] = [bd.flag_signed(4) for _ in range(5)]
    hdr["refresh_entropy"] = bd.lit(1)
    # token prob updates: needs the 1056 update probabilities; the manifest does not need what follows, so skip it.
    return hdr


# ---------------------------------------------------------------------------------------------------------------
# Tooling
# ---------------------------------------------------------------------------------------------------------------

def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, **kw)
    if r.returncode != 0:
        raise RuntimeError("command failed: %s\n%s" % (cmd, (r.stderr or b"").decode("utf-8", "replace")[-2000:]))
    return r


def riff_vp8_payload(webp):
    """Returns the payload of the 'VP8 ' chunk of a simple lossy WebP (no VP8X, no alpha)."""
    assert webp[0:4] == b"RIFF" and webp[8:12] == b"WEBP", "not a RIFF WebP"
    pos = 12
    while pos + 8 <= len(webp):
        fourcc = webp[pos:pos + 4]
        size = struct.unpack("<I", webp[pos + 4:pos + 8])[0]
        if fourcc == b"VP8 ":
            return webp[pos + 8:pos + 8 + size]
        if fourcc in (b"VP8X", b"ALPH", b"VP8L"):
            raise ValueError("unexpected chunk %r in a fixture" % fourcc)
        pos += 8 + size + (size & 1)
    raise ValueError("no VP8 chunk")


def ivf_wrap(frame, w, h):
    hdr = b"DKIF" + struct.pack("<HH4sHHIIII", 0, 32, b"VP80", w, h, 30, 1, 1, 0)
    return hdr + struct.pack("<IQ", len(frame), 0) + frame


def ivf_first_frame(ivf):
    assert ivf[0:4] == b"DKIF"
    hlen = struct.unpack("<H", ivf[6:8])[0]
    size = struct.unpack("<I", ivf[hlen:hlen + 4])[0]
    return ivf[hlen + 12:hlen + 12 + size]


def ffmpeg_decode(path, decoder):
    """Decodes the single picture in path with the named native decoder; returns the yuv420p bytes. Fails unless the
    stream is natively yuv420p (no scaler / conversion in the way)."""
    out = os.path.join(SCRATCH, "ffout.yuv")
    if os.path.exists(out):
        os.remove(out)
    r = run([FFMPEG, "-hide_banner", "-loglevel", "info", "-y", "-c:v", decoder, "-i", path, "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "yuv420p", out])
    log = r.stderr.decode("utf-8", "replace")
    stream_lines = [l for l in log.splitlines() if "Stream #0:0" in l and "Video:" in l]
    if not stream_lines:
        raise RuntimeError("no video stream in ffmpeg log:\n" + log)
    inp = [l for l in stream_lines][0]
    if "yuv420p" not in inp:
        raise RuntimeError("ffmpeg decoder %s did not output native yuv420p: %s" % (decoder, inp))
    if decoder == "vp8" and "Video: vp8" not in log:
        raise RuntimeError("expected the native vp8 decoder: " + inp)
    with open(out, "rb") as f:
        return f.read()


def dwebp_decode(webp_path, w, h):
    out = os.path.join(SCRATCH, "dw.yuv")
    if os.path.exists(out):
        os.remove(out)
    run([DWEBP, "-nodither", "-yuv", webp_path, "-o", out])
    with open(out, "rb") as f:
        return f.read()


def yuv_size(w, h):
    return w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2)


MANIFEST = []


def add_fixture(name, src, frame, truth, webp_path=None, check_webp=False):
    """Writes <name>.vp8 and its truth after cross-checking decoders; records the manifest row."""
    hdr = parse_frame_tag_and_header(frame)
    w, h = hdr["width"], hdr["height"]
    # ffmpeg native vp8 on exactly the bytes the C# test will see (wrapped in an IVF container)
    ivf_path = os.path.join(SCRATCH, "f.ivf")
    with open(ivf_path, "wb") as f:
        f.write(ivf_wrap(frame, w, h))
    yuv = ffmpeg_decode(ivf_path, "vp8")
    assert len(yuv) == yuv_size(w, h), "%s: ffmpeg yuv size %d != %d" % (name, len(yuv), yuv_size(w, h))
    if check_webp:
        yuv_webp = ffmpeg_decode(webp_path, "webp")
        assert yuv_webp == yuv, "%s: ffmpeg webp decode differs from ffmpeg vp8 decode of the extracted payload" % name
        yuv_lib = dwebp_decode(webp_path, w, h)
        assert yuv_lib == yuv, "%s: libwebp (dwebp) differs from ffmpeg (first diff at %d)" % (
            name, next((i for i in range(min(len(yuv), len(yuv_lib))) if yuv[i] != yuv_lib[i]), -1))
    assert truth is None or truth == yuv
    with open(os.path.join(OUT, name + ".vp8"), "wb") as f:
        f.write(frame)
    if w * h <= SMALL_PIXELS:
        with open(os.path.join(OUT, name + ".yuv"), "wb") as f:
            f.write(yuv)
    else:
        with open(os.path.join(OUT, name + ".sha256"), "w") as f:
            f.write(hashlib.sha256(yuv).hexdigest() + "\n")
    MANIFEST.append((name, src, w, h, hdr["version"], hdr["filter_type"], hdr["filter_level"], hdr["sharpness"],
                     hdr["lf_delta"], "/".join(map(str, hdr["ref_delta"])) + ";" + "/".join(map(str, hdr["mode_delta"])),
                     hdr["seg"], hdr["seg_map"], hdr["seg_data"], hdr["seg_abs"],
                     "/".join(map(str, hdr["seg_lf"])), "/".join(map(str, hdr["seg_q"])),
                     hdr["log2_parts"], hdr["q"], "/".join(map(str, hdr["qdeltas"])), len(frame)))


def from_webp_bytes(name, src, webp_bytes):
    p = os.path.join(SCRATCH, name + ".webp")
    with open(p, "wb") as f:
        f.write(webp_bytes)
    frame = riff_vp8_payload(webp_bytes)
    add_fixture(name, src, frame, None, webp_path=p, check_webp=True)


# ---------------------------------------------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------
# Random-syntax keyframes. Every byte string that starts with a valid frame header is a valid VP8 keyframe: the mode and
# token partitions below are RANDOM BYTES, which the bool decoder turns into random (but model-distributed) segment ids,
# skip flags, 16x16 modes, all ten 4x4 sub-modes in every context, chroma modes and coefficient tokens. This reaches
# decoder paths no real encoder produces (skipped B_PRED macroblocks, prob 0, probability updates everywhere, cat6
# tokens, deltas that clamp). The truth is still ffmpeg's decode of the same bytes; the python writer only has to emit a
# header the decoders parse the same way (the manifest parser reads it back).
# ---------------------------------------------------------------------------------------------------------------

class BoolEnc:
    """RFC 6386 section 7.3 boolean encoder."""
    def __init__(self):
        self.out = bytearray()
        self.range = 255
        self.bottom = 0
        self.bit_count = 24

    def _add_one(self):
        i = len(self.out) - 1
        while i >= 0 and self.out[i] == 255:
            self.out[i] = 0
            i -= 1
        self.out[i] += 1

    def put(self, prob, bit):
        split = 1 + (((self.range - 1) * prob) >> 8)
        if bit:
            self.bottom += split
            self.range -= split
        else:
            self.range = split
        while self.range < 128:
            self.range <<= 1
            if self.bottom & (1 << 31):
                self._add_one()
            self.bottom = (self.bottom << 1) & 0xFFFFFFFF
            self.bit_count -= 1
            if self.bit_count == 0:
                self.out.append(self.bottom >> 24)
                self.bottom &= (1 << 24) - 1
                self.bit_count = 8

    def lit(self, v, n):
        for b in range(n - 1, -1, -1):
            self.put(128, (v >> b) & 1)

    def flag_signed(self, v, n):
        if v == 0:
            self.lit(0, 1)
        else:
            self.lit(1, 1)
            self.lit(abs(v), n)
            self.lit(1 if v < 0 else 0, 1)

    def finish(self):
        c = self.bit_count
        v = self.bottom
        if v & (1 << (32 - c)):
            self._add_one()
        v = (v << (c & 7)) & 0xFFFFFFFF
        c >>= 3
        while c > 0:
            v = (v << 8) & 0xFFFFFFFF
            c -= 1
        for _ in range(4):
            self.out.append(v >> 24)
            v = (v << 8) & 0xFFFFFFFF
        return bytes(self.out)


_UPDATE_PROBS = None


def coef_update_probs():
    """The 4*8*3*11 coefficient-probability update probabilities, read from the library's own table (a wrong table would
    desynchronise our decoder from ffmpeg and fail the fixture)."""
    global _UPDATE_PROBS
    if _UPDATE_PROBS is None:
        import re
        path = os.path.join(HERE, "..", "..", "SpawnDev.Codecs", "Video", "Vp8", "Vp8CoefUpdateProbs.cs")
        text = open(path, encoding="utf-8").read()
        text = text[text.index("UpdateProbs = new byte["):]
        text = text[:text.index("};")]
        text = re.sub(r"//[^\n]*", "", text)
        nums = [int(x) for x in re.findall(r"\b\d+\b", text)]
        assert len(nums) == 1056, len(nums)
        _UPDATE_PROBS = nums
    return _UPDATE_PROBS


def random_keyframe(rng, w, h, p):
    """p: dict of header choices (see gen_random). Returns the raw VP8 keyframe bytes."""
    mb_cols, mb_rows = (w + 15) // 16, (h + 15) // 16
    e = BoolEnc()
    e.lit(0, 1)                       # color space
    e.lit(p.get("clamp", 0), 1)       # clamping type
    seg = p.get("seg")
    e.lit(1 if seg else 0, 1)
    if seg:
        e.lit(seg["map"], 1)
        e.lit(seg["data"], 1)
        if seg["data"]:
            e.lit(seg["abs"], 1)
            for v in seg["q"]:
                e.flag_signed(v, 7)
            for v in seg["lf"]:
                e.flag_signed(v, 6)
        if seg["map"]:
            for pr in seg["probs"]:
                if pr is None:
                    e.lit(0, 1)
                else:
                    e.lit(1, 1)
                    e.lit(pr, 8)
    e.lit(p["filter_type"], 1)
    e.lit(p["level"], 6)
    e.lit(p["sharp"], 3)
    d = p.get("delta")
    e.lit(1 if d else 0, 1)
    if d:
        e.lit(1 if d["update"] else 0, 1)
        if d["update"]:
            for v in d["ref"]:
                e.flag_signed(v, 6)
            for v in d["mode"]:
                e.flag_signed(v, 6)
    e.lit(p["log2_parts"], 2)
    e.lit(p["q"], 7)
    for v in p["qdeltas"]:
        e.flag_signed(v, 4)
    e.lit(1, 1)                       # refresh_entropy_probs
    up = coef_update_probs()
    for i in range(1056):
        if rng.random() < p["prob_update_rate"]:
            e.put(up[i], 1)
            e.lit(int(rng.integers(0, 256)), 8)
        else:
            e.put(up[i], 0)
    skip = p.get("skip_prob")         # None = mb_no_coeff_skip off
    e.lit(0 if skip is None else 1, 1)
    if skip is not None:
        e.lit(skip, 8)
    header = e.finish()
    modes = header + bytes(rng.integers(0, 256, mb_cols * mb_rows * p.get("mode_bytes_per_mb", 48), dtype=np.uint8))
    nparts = 1 << p["log2_parts"]
    part_sizes = []
    for i in range(nparts):
        rows = len(range(i, mb_rows, nparts))
        part_sizes.append(64 + rows * mb_cols * p.get("token_bytes_per_mb", 160))
    parts = [bytes(rng.integers(0, 256, s, dtype=np.uint8)) for s in part_sizes]
    table = b"".join(struct.pack("<I", s)[:3] for s in part_sizes[:-1])
    tag = (0 | (p.get("version", 0) << 1) | (1 << 4) | (len(modes) << 5))
    frame = struct.pack("<I", tag)[:3] + b"\x9d\x01\x2a" + struct.pack("<HH", w, h) + modes + table + b"".join(parts)
    return frame


def gen_random():
    rng = np.random.default_rng(20261022)
    sizes = [(16, 16), (17, 33), (31, 47), (48, 32), (64, 64), (100, 75), (33, 100), (1, 1), (7, 5), (80, 48), (128, 96), (65, 17)]
    count = 0
    for i in range(80):
        w, h = sizes[i % len(sizes)]
        if i >= 48:
            # skip-heavy set (prob_skip_false tiny => most macroblocks carry the skip flag): skipped B_PRED macroblocks
            # next to non-skipped Y2 macroblocks exercise the Y2 entropy-context rule and the inner-edge filter rule
            w, h = [(112, 112), (128, 96), (96, 128), (80, 80)][i % 4]
        p = {"version": int(rng.integers(0, 4)), "filter_type": int(rng.integers(0, 2)),
             "level": int(rng.choice([0, 1, 5, 12, 20, 33, 45, 63])), "sharp": int(rng.integers(0, 8)),
             "log2_parts": int(rng.integers(0, 4)), "q": int(rng.integers(0, 60)),
             "qdeltas": [int(rng.choice([0, 0, -7, 5, 15, -15])) for _ in range(5)],
             "prob_update_rate": float(rng.choice([0.0, 0.01, 0.1])),
             "skip_prob": None if rng.random() < 0.3 else int(rng.choice([0, 1, 60, 128, 200, 255])),
             "clamp": int(rng.random() < 0.15)}
        if i >= 48:
            p["skip_prob"] = int(rng.choice([1, 3, 100, 128, 160]))
            p["level"] = int(rng.choice([8, 20, 40, 63]))
        if rng.random() < 0.6:
            p["delta"] = {"update": rng.random() < 0.85, "ref": [int(rng.integers(-12, 13)) for _ in range(4)],
                          "mode": [int(rng.integers(-12, 13)) for _ in range(4)]}
        if rng.random() < 0.65:
            absolute = int(rng.random() < 0.5)
            data = int(rng.random() < 0.85)
            if absolute:
                qv = [int(rng.integers(0, 80)) for _ in range(4)]
                lv = [int(rng.integers(0, 64)) for _ in range(4)]
            else:
                # keep every intermediate value in range: ffmpeg does not clamp q / level between the segment step and
                # the plane / mode deltas, libvpx does; they only agree when no intermediate clamp is needed
                qv = [int(rng.integers(-p["q"] // 2, 20)) for _ in range(4)]
                lv = [int(rng.integers(-p["level"], 63 - p["level"] + 1)) if p["level"] > 0 else 0 for _ in range(4)]
            if not data:
                qv, lv = [0] * 4, [0] * 4
            p["seg"] = {"map": int(rng.random() < 0.8), "data": data, "abs": absolute if data else 0, "q": qv, "lf": lv,
                        "probs": [None if rng.random() < 0.3 else int(rng.integers(1, 256)) for _ in range(3)]}
        # With deltas the per-MB level can leave 0..63 only through the final clamp, which both decoders apply identically.
        # Keep q + deltas inside 0..127 for the same reason.
        qtot = p["q"]
        if "seg" in p and p["seg"]["data"]:
            qs = p["seg"]["q"]
            ok = all(0 <= (v if p["seg"]["abs"] else qtot + v) + dq <= 127 for v in qs for dq in p["qdeltas"] + [0]) \
                and all(0 <= (v if p["seg"]["abs"] else qtot + v) <= 127 for v in qs)
            if not ok:
                p["qdeltas"] = [0] * 5
        elif not all(0 <= qtot + dq <= 127 for dq in p["qdeltas"]):
            p["qdeltas"] = [0] * 5
        frame = random_keyframe(rng, w, h, p)
        name = "rnd_%02d_%dx%d_%s_L%d_S%d_P%d%s%s%s" % (
            i, w, h, "simple" if p["filter_type"] else "normal", p["level"], p["sharp"], 1 << p["log2_parts"],
            "_delta" if "delta" in p else "", "_seg" if "seg" in p else "", "_noskip" if p["skip_prob"] is None else "")
        add_fixture(name, "random syntax (python header writer + random partitions)", frame, None)
        count += 1


def gen_pillow():
    sizes = [(1, 1), (7, 5), (16, 16), (17, 33), (100, 75), (255, 129)]
    kinds = ["photo", "noise", "text", "flat"]
    qs = [1, 20, 50, 75, 100]
    i = 0
    for (w, h) in sizes:
        for q in qs:
            kind = kinds[i % 4]
            method = i % 7
            img = Image.fromarray(make(kind, w, h, 1000 + i), "RGB")
            buf = io.BytesIO()
            img.save(buf, "WEBP", lossless=False, quality=q, method=method)
            from_webp_bytes("pil_%s_%dx%d_q%d_m%d" % (kind, w, h, q, method), "pillow q=%d m=%d" % (q, method), buf.getvalue())
            i += 1
    # every method on one picture
    for m in range(7):
        img = Image.fromarray(make("photo", 100, 75, 77), "RGB")
        buf = io.BytesIO()
        img.save(buf, "WEBP", lossless=False, quality=75, method=m)
        from_webp_bytes("pil_methods_photo_100x75_m%d" % m, "pillow q=75 m=%d" % m, buf.getvalue())
    # sizes around the macroblock grid
    for (w, h) in [(2, 2), (15, 17), (31, 47), (48, 32), (160, 16), (16, 160), (33, 1), (1, 33)]:
        img = Image.fromarray(make("text", w, h, 5000 + w * 7 + h), "RGB")
        buf = io.BytesIO()
        img.save(buf, "WEBP", lossless=False, quality=60, method=4)
        from_webp_bytes("pil_text_%dx%d_q60_m4" % (w, h), "pillow q=60 m=4", buf.getvalue())
    # larger photographic pictures
    for (q, m) in [(75, 4), (30, 6), (95, 2)]:
        img = Image.fromarray(make("photo", 640, 480, 4242 + q), "RGB")
        buf = io.BytesIO()
        img.save(buf, "WEBP", lossless=False, quality=q, method=m)
        from_webp_bytes("pil_photo_640x480_q%d_m%d" % (q, m), "pillow q=%d m=%d" % (q, m), buf.getvalue())
    img = Image.fromarray(make("noise", 255, 129, 99), "RGB")
    buf = io.BytesIO()
    img.save(buf, "WEBP", lossless=False, quality=100, method=6)
    from_webp_bytes("pil_noise_255x129_q100_m6_big", "pillow q=100 m=6", buf.getvalue())


def gen_cwebp():
    sets = [
        ("simple_f60_s0", ["-nostrong", "-f", "60", "-sharpness", "0"]),
        ("simple_f100_s7", ["-nostrong", "-f", "100", "-sharpness", "7"]),
        ("simple_f20_s3", ["-nostrong", "-f", "20", "-sharpness", "3"]),
        ("strong_f80_s3", ["-strong", "-f", "80", "-sharpness", "3"]),
        ("strong_f100_s1", ["-strong", "-f", "100", "-sharpness", "1"]),
        ("strong_f100_s5", ["-strong", "-f", "100", "-sharpness", "5"]),
        ("strong_f100_s7", ["-strong", "-f", "100", "-sharpness", "7"]),
        ("nofilter", ["-f", "0"]),
        ("autofilter", ["-af", "-f", "50"]),
        ("seg1", ["-segments", "1", "-f", "70"]),
        ("seg2", ["-segments", "2", "-f", "70"]),
        ("seg3", ["-segments", "3", "-f", "70", "-sharpness", "2"]),
        ("sns0", ["-sns", "0", "-f", "60"]),
        ("sns100", ["-sns", "100", "-f", "60", "-segments", "4"]),
        ("m6q100f100", ["-m", "6", "-f", "100"]),
    ]
    imgs = [("photo", 100, 75), ("text", 100, 75), ("photo", 255, 129), ("noise", 66, 50)]
    k = 0
    for si, (tag, opts) in enumerate(sets):
        # keep the corpus bounded: every option set on the 100x75 photo and on one rotating other picture
        for (kind, w, h) in (imgs[0], imgs[1 + (si % 3)]):
            k += 1
            png = os.path.join(SCRATCH, "src.png")
            Image.fromarray(make(kind, w, h, 3000 + k), "RGB").save(png)
            q = "60" if kind != "noise" else "90"
            webp = os.path.join(SCRATCH, "c.webp")
            run([CWEBP, "-quiet", "-q", q, "-m", "4"] + opts + [png, "-o", webp])
            with open(webp, "rb") as f:
                data = f.read()
            from_webp_bytes("cw_%s_%s_%dx%d" % (tag, kind, w, h), "cwebp -q %s %s" % (q, " ".join(opts)), data)


def gen_libvpx():
    sets = [
        ("p0_q10", ["-crf", "10"], "photo", 100, 75),
        ("p0_q30", ["-crf", "30"], "photo", 255, 129),
        ("p0_q50", ["-crf", "50"], "text", 100, 75),
        ("p0_q63", ["-crf", "63"], "photo", 100, 75),
        ("p0_s4", ["-crf", "25", "-sharpness", "4"], "photo", 100, 75),
        ("p0_s7", ["-crf", "25", "-sharpness", "7"], "text", 100, 75),
        ("p1_simple", ["-crf", "25", "-profile:v", "1"], "photo", 100, 75),
        ("p1_simple_text", ["-crf", "40", "-profile:v", "1"], "text", 255, 129),
        ("p2", ["-crf", "25", "-profile:v", "2"], "photo", 100, 75),
        ("p3", ["-crf", "25", "-profile:v", "3"], "photo", 100, 75),
        ("slices2", ["-crf", "25", "-slices", "2"], "photo", 255, 129),
        ("slices4", ["-crf", "25", "-slices", "4"], "photo", 255, 129),
        ("slices8", ["-crf", "25", "-slices", "8"], "text", 255, 129),
        ("slices2_odd", ["-crf", "30", "-slices", "2"], "text", 17, 33),
        ("slices4_tiny", ["-crf", "30", "-slices", "4"], "photo", 33, 20),
        ("slices8_big", ["-crf", "35", "-slices", "8"], "photo", 640, 480),
        ("slices4_s5_p1", ["-crf", "30", "-slices", "4", "-sharpness", "5", "-profile:v", "1"], "photo", 160, 120),
        ("rt", ["-crf", "30", "-deadline", "realtime", "-cpu-used", "8"], "photo", 255, 129),
        ("odd_17x33", ["-crf", "25"], "photo", 17, 33),
        ("tiny_1x1", ["-crf", "25"], "photo", 1, 1),
        ("flat", ["-crf", "35"], "flat", 100, 75),
        ("big_640x480", ["-crf", "20"], "photo", 640, 480),
        ("big_noise", ["-crf", "15"], "noise", 320, 240),
    ]
    for i, (tag, opts, kind, w, h) in enumerate(sets):
        # fixed quantizer: crf == qmin == qmax (otherwise the rate controller picks its own q, which was 4 here)
        crf = opts[opts.index("-crf") + 1]
        opts = opts + ["-qmin", crf, "-qmax", crf]
        png = os.path.join(SCRATCH, "src.png")
        Image.fromarray(make(kind, w, h, 9000 + i), "RGB").save(png)
        ivf = os.path.join(SCRATCH, "v.ivf")
        if os.path.exists(ivf):
            os.remove(ivf)
        run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", png, "-c:v", "libvpx", "-b:v", "0", "-g", "1",
             "-auto-alt-ref", "0", "-lag-in-frames", "0", "-frames:v", "1"] + opts + ["-f", "ivf", ivf])
        with open(ivf, "rb") as f:
            frame = ivf_first_frame(f.read())
        add_fixture("vpx_%s_%s_%dx%d" % (tag, kind, w, h), "ffmpeg libvpx " + " ".join(opts), frame, None)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    gen_pillow()
    gen_cwebp()
    gen_libvpx()
    gen_random()
    cols = ["name", "source", "width", "height", "version", "filter_type", "filter_level", "sharpness", "lf_delta",
            "ref_delta;mode_delta", "seg", "seg_map", "seg_data", "seg_abs", "seg_lf", "seg_q", "log2_parts", "q",
            "qdeltas", "bytes"]
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        for row in sorted(MANIFEST):
            f.write(",".join(str(x).replace(",", " ") for x in row) + "\n")
    total = sum(os.path.getsize(os.path.join(OUT, n)) for n in os.listdir(OUT))
    print("wrote %d fixtures, %d bytes total, into %s" % (len(MANIFEST), total, OUT))
    # coverage summary
    def count(pred):
        return sum(1 for r in MANIFEST if pred(r))
    # the corpus must really cover what the tests claim it covers
    for label, pred in [("simple filter", lambda r: r[5] == 1 and r[6] > 0), ("normal filter", lambda r: r[5] == 0 and r[6] > 0),
                        ("filter level 0", lambda r: r[6] == 0), ("sharpness > 0", lambda r: r[7] > 0 and r[6] > 0),
                        ("mode/ref lf deltas", lambda r: r[8] == 1), ("segmentation", lambda r: r[10] == 1),
                        ("2 partitions", lambda r: r[16] == 1), ("4 partitions", lambda r: r[16] == 2),
                        ("8 partitions", lambda r: r[16] == 3)]:
        assert count(pred) > 0, "no fixture covers: " + label
    print("simple filter:", count(lambda r: r[5] == 1), " normal:", count(lambda r: r[5] == 0 and r[6] > 0),
          " level0:", count(lambda r: r[6] == 0), " sharpness>0:", count(lambda r: r[7] > 0),
          " lf_delta:", count(lambda r: r[8] == 1), " segmentation:", count(lambda r: r[10] == 1),
          " partitions 2/4/8:", [count(lambda r, p=p: r[16] == p) for p in (1, 2, 3)])


if __name__ == "__main__":
    main()
