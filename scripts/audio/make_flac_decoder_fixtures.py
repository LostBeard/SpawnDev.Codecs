"""Fixtures for the full FLAC decoder (FlacDecoderGpu.DecodeAsync): FLAC files from several encoders with the samples
they hold (FLAC is lossless, so the truth is the encoder's input).

  <src>.s32.gz      source samples, interleaved int32 little-endian (one per source)
  <case>.flac       a FLAC file of a source
  manifest.tsv      case, source, channels, rate, bits, samples, what the file exercises

Encoders:
  * FFmpeg 8.1 flacenc: compression levels 0 / 5 / 8 / 12, fixed-only prediction, every stereo mode, block sizes, 16 / 24
    (and 32 when FFmpeg writes it) bits, 1..8 channels, wasted bits;
  * libFLAC (through libsndfile / soundfile): 8 / 16 / 24 bits;
  * variable block size: frames of two FFmpeg encodes (different block sizes) joined, their headers rewritten to the
    variable-blocking form (sample number) with fresh CRCs;
  * a small writer here for what encoders rarely write: escaped partitions (Rice and Rice2, raw widths 0..n), Rice2
    parameters, LPC up to order 32 with negative shifts avoided, 8-bit, constant / verbatim subframes, wasted bits.
Every file is checked to decode to its source with libFLAC (soundfile) before it is written.

Usage: python -I make_flac_decoder_fixtures.py   (ffmpeg on PATH, numpy, soundfile)
"""
import gzip
import os
import struct
import subprocess
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "FlacDec"))
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

rng = np.random.default_rng(1108)


def tones(n, rate, ch, bits, freqs, noise=0.01, amp=0.6):
    t = np.arange(n) / rate
    full = (1 << (bits - 1)) - 1
    out = np.zeros((n, ch))
    for c in range(ch):
        s = sum(np.sin(2 * np.pi * f * (1 + 0.07 * c) * t + rng.random() * 6) for f in freqs) / len(freqs)
        s = s * amp + rng.normal(0, noise, n)
        out[:, c] = s * full
    return np.clip(np.round(out), -full - 1, full).astype(np.int64)


sources = {}
manifest = []


def add_source(name, data, rate, bits):
    sources[name] = (data, rate, bits)
    with open(os.path.join(OUT, name + ".s32.gz"), "wb") as fh:
        fh.write(gzip.compress(data.astype("<i4").tobytes(), mtime=0))


def check_and_add(case, src, path_or_bytes, note, oracle="libflac"):
    data, rate, bits = sources[src]
    if isinstance(path_or_bytes, (bytes, bytearray)):
        b = bytes(path_or_bytes)
    else:
        b = open(path_or_bytes, "rb").read()
    out = os.path.join(OUT, case + ".flac")
    with open(out, "wb") as fh:
        fh.write(b)
    # libFLAC must decode it to the source exactly (FFmpeg for 32-bit files, which libsndfile does not read, and for
    # files of unknown length, which libsndfile cannot seek)
    if bits == 32 or oracle == "ffmpeg":
        raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", out, "-f", "s32le", "-"],
                             check=True, capture_output=True, stdin=subprocess.DEVNULL).stdout
        dec = np.frombuffer(raw, "<i4").astype(np.int64).reshape(-1, data.shape[1]) >> (32 - bits)
        r = rate
    else:
        dec, r = sf.read(out, dtype="int32", always_2d=True)
        dec = dec.astype(np.int64) >> (32 - bits)
    assert r == rate, (case, r, rate)
    assert dec.shape == data.shape, (case, dec.shape, data.shape)
    assert np.array_equal(dec, data), (case, "libFLAC decodes other samples")
    manifest.append([case, src, str(data.shape[1]), str(rate), str(bits), str(data.shape[0]), note])
    print(case, len(b), "bytes:", note)


def raw_of(data, bits):
    if bits <= 16:
        return data.astype("<i2").tobytes(), "s16le"
    return (data.astype(np.int64) << (32 - bits)).astype("<i4").tobytes(), "s32le"


def ffmpeg_flac(src, extra, tmp, name):
    data, rate, bits = sources[src]
    raw, fmt = raw_of(data, bits)
    rp = os.path.join(tmp, name + ".raw")
    with open(rp, "wb") as fh:
        fh.write(raw)
    out = os.path.join(tmp, name + ".flac")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", fmt, "-ar", str(rate), "-ac", str(data.shape[1]),
           "-i", rp, "-c:a", "flac"]
    if bits > 16:
        cmd += ["-sample_fmt", "s32", "-bits_per_raw_sample", str(bits)]
    if bits == 32:
        cmd += ["-strict", "experimental"]
    cmd += extra + ["-fflags", "+bitexact", "-flags", "+bitexact", out]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    return out


# ---------------------------------------------------------------- bit writer + CRCs for the hand-made / rewritten files
def crc8(b):
    c = 0
    for x in b:
        c ^= x
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def crc16(b):
    c = 0
    for x in b:
        c ^= x << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x8005) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return c


class Bits:
    def __init__(self):
        self.v = 0
        self.n = 0

    def put(self, val, n):
        if n == 0:
            return
        self.v = (self.v << n) | (val & ((1 << n) - 1))
        self.n += n

    def put_signed(self, val, n):
        self.put(val & ((1 << n) - 1), n)

    def unary(self, q):
        self.put(0, q)
        self.put(1, 1)

    def align(self):
        if self.n % 8:
            self.put(0, 8 - self.n % 8)

    def bytes(self):
        assert self.n % 8 == 0
        return self.v.to_bytes(self.n // 8, "big") if self.n else b""


def utf8_num(v):
    if v < 0x80:
        return bytes([v])
    n = 2
    while v >= (1 << (5 * n + 1)):
        n += 1
    out = []
    for _ in range(n - 1):
        out.append(0x80 | (v & 0x3F))
        v >>= 6
    first = ((0xFF << (8 - n)) & 0xFF) | v
    return bytes([first] + out[::-1])


RATE_CODES = {88200: 1, 176400: 2, 192000: 3, 8000: 4, 16000: 5, 22050: 6, 24000: 7, 32000: 8, 44100: 9, 48000: 10,
              96000: 11}
BITS_CODES = {8: 1, 12: 2, 16: 4, 20: 5, 24: 6, 32: 7}
BLOCK_CODES = {192: 1, 576: 2, 1152: 3, 2304: 4, 4608: 5, 256: 8, 512: 9, 1024: 10, 2048: 11, 4096: 12, 8192: 13,
               16384: 14, 32768: 15}


def frame_header(variable, number, block, rate, chan_assign, bits):
    h = Bits()
    h.put(0x3FFE, 14)
    h.put(0, 1)
    h.put(1 if variable else 0, 1)
    tail = b""
    if block in BLOCK_CODES:
        h.put(BLOCK_CODES[block], 4)
    elif block <= 256:
        h.put(6, 4)
        tail += bytes([block - 1])
    else:
        h.put(7, 4)
        tail += struct.pack(">H", block - 1)
    if rate in RATE_CODES:
        h.put(RATE_CODES[rate], 4)
        rtail = b""
    elif rate % 1000 == 0 and rate // 1000 < 256:
        h.put(12, 4)
        rtail = bytes([rate // 1000])
    elif rate < 65536:
        h.put(13, 4)
        rtail = struct.pack(">H", rate)
    else:
        h.put(14, 4)
        rtail = struct.pack(">H", rate // 10)
    h.put(chan_assign, 4)
    h.put(BITS_CODES[bits], 3)
    h.put(0, 1)
    b = h.bytes() + utf8_num(number) + tail + rtail
    return b + bytes([crc8(b)])


def streaminfo(minb, maxb, minf, maxf, rate, ch, bits, total, md5=b"\0" * 16, last=True, extra=b""):
    s = Bits()
    s.put(minb, 16)
    s.put(maxb, 16)
    s.put(minf, 24)
    s.put(maxf, 24)
    s.put(rate, 20)
    s.put(ch - 1, 3)
    s.put(bits - 1, 5)
    s.put(total, 36)
    body = s.bytes() + md5
    hdr = bytes([(0x80 if last and not extra else 0) | 0]) + struct.pack(">I", len(body))[1:]
    return b"fLaC" + hdr + body + extra


def md5_of(data, bits):
    import hashlib
    nbytes = (bits + 7) // 8
    a = data.astype(np.int64)
    out = bytearray()
    flat = a.reshape(-1)
    for i in range(nbytes):
        out_b = ((flat >> (8 * i)) & 0xFF).astype(np.uint8)
        if i == 0:
            arr = np.zeros((flat.size, nbytes), np.uint8)
        arr[:, i] = out_b
    return hashlib.md5(arr.tobytes()).digest()


def rice_partitions(b, res, order, porder, params, method):
    """Write a residual section: params[p] = rice parameter or ('esc', rawbits)."""
    b.put(method, 2)
    b.put(porder, 4)
    n = len(res) + order
    psize = n >> porder
    at = 0
    pbits = 4 if method == 0 else 5
    esc = 15 if method == 0 else 31
    for p in range(1 << porder):
        cnt = psize - order if p == 0 else psize
        part = res[at:at + cnt]
        at += cnt
        prm = params[p]
        if isinstance(prm, tuple):
            raw = prm[1]
            b.put(esc, pbits)
            b.put(raw, 5)
            for r in part:
                assert raw > 0 or r == 0
                if raw:
                    assert -(1 << (raw - 1)) <= r < (1 << (raw - 1)), (r, raw)
                    b.put_signed(int(r), raw)
        else:
            b.put(prm, pbits)
            for r in part:
                u = (int(r) << 1) ^ (int(r) >> 63) if r < 0 else int(r) << 1
                if r < 0:
                    u = ((-int(r)) << 1) - 1
                b.unary(u >> prm)
                b.put(u & ((1 << prm) - 1), prm)


def lpc_predict_forward(warm, coefs, shift, res):
    """x from warm-up samples + residuals, as a decoder computes it."""
    x = list(warm)
    order = len(coefs)
    for r in res:
        acc = 0
        for j in range(order):
            acc += coefs[j] * x[-1 - j]
        x.append(r + (acc >> shift))
    return x


FIXED = [[], [1], [2, -1], [3, -3, 1], [4, -6, 4, -1]]


# ---------------------------------------------------------------- sources
with tempfile.TemporaryDirectory() as tmp:
    add_source("st16_44k", tones(44100 + 9876, 44100, 2, 16, [220, 554, 1320, 3100]), 44100, 16)
    add_source("mono16_22k", tones(33333, 22050, 1, 16, [300, 1700], noise=0.05), 22050, 16)
    add_source("st24_96k", tones(48000 + 77, 96000, 2, 24, [440, 5000, 18000]), 96000, 24)
    add_source("mono8_8k", tones(8000 + 5, 8000, 1, 8, [400, 900], noise=0.05), 8000, 8)
    add_source("ch6_48k", tones(48000 // 2 + 3, 48000, 6, 16, [200, 700, 2500]), 48000, 16)
    loud = np.clip(rng.normal(0, 30000, (20000, 2)), -32768, 32767).round().astype(np.int64)
    add_source("noise16", loud, 44100, 16)
    add_source("noise24", np.clip(rng.normal(0, 7e6, (12000, 2)), -(1 << 23), (1 << 23) - 1).round().astype(np.int64), 48000, 24)
    wasted = (tones(15000, 44100, 2, 16, [500, 1500]) >> 3) << 3
    wasted[:, 1] = (wasted[:, 1] >> 5) << 5
    add_source("wasted16", wasted, 44100, 16)
    add_source("silence_dc", np.concatenate([np.zeros((5000, 2), np.int64), np.full((6000, 2), 1234, np.int64),
                                             np.full((5000, 2), -32768, np.int64)]), 44100, 16)
    add_source("st32_48k", tones(12000, 48000, 2, 32, [330, 2900]), 48000, 32)

    # FFmpeg
    for lvl in (0, 5, 8, 12):
        check_and_add(f"ff_st16_l{lvl}", "st16_44k", ffmpeg_flac("st16_44k", ["-compression_level", str(lvl)], tmp, f"a{lvl}"),
                      f"ffmpeg level {lvl}")
    for mode, nm in ((0, "indep"), (1, "left_side"), (2, "right_side"), (3, "mid_side")):
        check_and_add(f"ff_st16_{nm}", "st16_44k", ffmpeg_flac("st16_44k", ["-ch_mode", str(mode)], tmp, "m" + nm),
                      f"ffmpeg stereo mode {nm}")
    check_and_add("ff_st16_fixed", "st16_44k", ffmpeg_flac("st16_44k", ["-lpc_type", "0"], tmp, "fx"), "fixed predictors only")
    check_and_add("ff_st16_exact_rice", "st16_44k", ffmpeg_flac("st16_44k", ["-compression_level", "8", "-exact_rice_parameters", "1"], tmp, "er"),
                  "exact rice parameters")
    for fs in (192, 1000, 16384):
        check_and_add(f"ff_st16_frame{fs}", "st16_44k", ffmpeg_flac("st16_44k", ["-frame_size", str(fs)], tmp, f"fs{fs}"),
                      f"frame size {fs}")
    check_and_add("ff_mono16_22k", "mono16_22k", ffmpeg_flac("mono16_22k", [], tmp, "mono"), "mono, odd length, 22.05 kHz")
    check_and_add("ff_st24_96k_l8", "st24_96k", ffmpeg_flac("st24_96k", ["-compression_level", "8"], tmp, "s24"), "24-bit 96 kHz")
    check_and_add("ff_ch6_48k", "ch6_48k", ffmpeg_flac("ch6_48k", [], tmp, "c6"), "6 channels")
    check_and_add("ff_noise16", "noise16", ffmpeg_flac("noise16", [], tmp, "n16"), "white noise (verbatim subframes)")
    check_and_add("ff_noise24", "noise24", ffmpeg_flac("noise24", ["-compression_level", "8"], tmp, "n24"), "24-bit noise (Rice2)")
    check_and_add("ff_wasted16", "wasted16", ffmpeg_flac("wasted16", [], tmp, "w16"), "wasted bits")
    check_and_add("ff_silence_dc", "silence_dc", ffmpeg_flac("silence_dc", [], tmp, "sdc"), "constant subframes")
    try:
        check_and_add("ff_st32_48k", "st32_48k", ffmpeg_flac("st32_48k", ["-compression_level", "5"], tmp, "s32"), "32-bit")
    except Exception as e:  # FFmpeg builds without the experimental 32-bit encoder
        print("no 32-bit FFmpeg FLAC:", e)

    # libFLAC (soundfile)
    for src, sub in (("st16_44k", "PCM_16"), ("st24_96k", "PCM_24"), ("mono8_8k", "PCM_S8"), ("ch6_48k", "PCM_16"),
                     ("wasted16", "PCM_16"), ("noise24", "PCM_24")):
        data, rate, bits = sources[src]
        p = os.path.join(tmp, f"lf_{src}.flac")
        sf.write(p, (data << (32 - bits)).astype(np.int32), rate, subtype=sub, format="FLAC")
        check_and_add(f"libflac_{src}", src, p, f"libFLAC {sub}")

    # variable block size: frames of two FFmpeg encodes joined, headers rewritten with sample numbers
    def frames_of(path):
        b = open(path, "rb").read()
        p = 4
        while True:
            last = b[p] & 0x80
            ln = int.from_bytes(b[p + 1:p + 4], "big")
            p += 4 + ln
            if last:
                break
        # split at verified frame boundaries: sync + header CRC-8 + frame CRC-16
        starts = []
        i = p
        while i < len(b):
            starts.append(i)
            j = i + 2
            while True:
                j = b.find(b"\xff\xf8", j)
                if j < 0:
                    j = len(b)
                    break
                if crc16(b[i:j]) == 0:
                    break
                j += 1
            i = j
        return [b[s:e] for s, e in zip(starts, starts[1:] + [len(b)])]

    data, rate, bits = sources["st16_44k"]
    split = 30000
    va, vb = data[:split], data[split:]
    sources["var_a"] = (va, rate, bits)
    sources["var_b"] = (vb, rate, bits)
    fa = frames_of(ffmpeg_flac("var_a", ["-frame_size", "4096"], tmp, "va"))
    fb = frames_of(ffmpeg_flac("var_b", ["-frame_size", "1152"], tmp, "vb"))
    del sources["var_a"], sources["var_b"]

    def rewrite(frame, sample_number):
        # parse the fixed-blocking header, rebuild it in variable-blocking form
        h = Bits()
        b1 = frame[2]
        b2 = frame[3]
        bs_code, rate_code = b1 >> 4, b1 & 15
        p = 4
        # skip the UTF-8 frame number
        first = frame[p]
        n = 1
        while first & (0x80 >> (n - 1)) and n < 8:
            n += 1
        n = 1 if first < 0x80 else n - 1
        p += n
        extra = 0
        if bs_code == 6:
            extra += 1
        elif bs_code == 7:
            extra += 2
        if rate_code == 12:
            extra += 1
        elif rate_code in (13, 14):
            extra += 2
        tail = frame[p:p + extra]
        p += extra + 1   # + CRC-8
        hdr = bytes([0xFF, 0xF9, b1, b2]) + utf8_num(sample_number) + tail
        hdr += bytes([crc8(hdr)])
        body = hdr + frame[p:-2]
        return body + struct.pack(">H", crc16(body))

    # sample numbers from the block sizes the two encodes used (each one's last frame is shorter)
    frames = []
    pos = 0
    for part, fl in ((va, fa), (vb, fb)):
        n = part.shape[0]
        bs = 4096 if fl is fa else 1152
        for k, f in enumerate(fl):
            frames.append(rewrite(f, pos))
            pos += min(bs, n - k * bs)
    assert pos == data.shape[0]
    allf = b"".join(frames)
    sizes = [len(f) for f in frames]
    si = streaminfo(1152, 4096, min(sizes), max(sizes), rate, 2, 16, data.shape[0], md5_of(data, 16))
    check_and_add("var_blocksize", "st16_44k", si + allf, "variable block size (4096 then 1152), MD5 set")

    # hand-made frames: escapes, Rice2, high-order LPC, constant / verbatim, wasted bits. Each subframe follows a target
    # signal: warm-up = target, then residual = target - prediction, clamped to an escaped partition's raw width, so the
    # samples stay in range whatever the predictor
    hrate, hbits = 32000, 16
    n = 1024
    t_axis = np.arange(n)

    def target(seed, amp, freqs, noise=0.01):
        r = np.random.default_rng(seed)
        s = sum(np.sin(2 * np.pi * f * t_axis / hrate + r.random() * 6) for f in freqs) / len(freqs)
        return [int(v) for v in np.round(s * amp + r.normal(0, amp * noise, n))]

    def follow(tgt, order, predict, porder, spec):
        psize = n >> porder
        x = list(tgt[:order])
        res = []
        for i in range(order, n):
            pred = predict(x)
            r = tgt[i] - pred
            s = spec[i // psize]
            if isinstance(s, tuple):
                w = s[1]
                lo, hi = (0, 0) if w == 0 else (-(1 << (w - 1)), (1 << (w - 1)) - 1)
                r = min(max(r, lo), hi)
            res.append(r)
            x.append(pred + r)
        return x, res

    def params_of(res, order, porder, spec, method):
        psize = n >> porder
        out = []
        at = 0
        for p, s in enumerate(spec):
            cnt = psize - order if p == 0 else psize
            part = res[at:at + cnt]
            at += cnt
            if s == "auto":
                m = sum(abs(2 * v) for v in part) / max(1, len(part))
                k = 0
                while (1 << (k + 1)) <= m:
                    k += 1
                out.append(min(k, 14 if method == 0 else 30))
            else:
                out.append(s)
        return out

    def fixed_pred(order):
        cs = FIXED[order]
        return lambda x: sum(c * x[-1 - j] for j, c in enumerate(cs))

    def lpc_pred(coefs, shift):
        return lambda x: sum(c * x[-1 - j] for j, c in enumerate(coefs)) >> shift

    def subframe_fixed(b, tgt, order, porder, spec, method, bps, wasted=0):
        shifted = [v >> wasted for v in tgt]
        x, res = follow(shifted, order, fixed_pred(order), porder, spec)
        b.put(0, 1)
        b.put(8 + order, 6)
        if wasted:
            b.put(1, 1)
            b.unary(wasted - 1)
        else:
            b.put(0, 1)
        for w in x[:order]:
            b.put_signed(int(w), bps - wasted)
        rice_partitions(b, res, order, porder, params_of(res, order, porder, spec, method), method)
        return [v << wasted for v in x]

    def subframe_lpc(b, tgt, coefs, prec, shift, porder, spec, method, bps):
        order = len(coefs)
        x, res = follow(tgt, order, lpc_pred(coefs, shift), porder, spec)
        b.put(0, 1)
        b.put(32 + order - 1, 6)
        b.put(0, 1)
        for w in x[:order]:
            b.put_signed(int(w), bps)
        b.put(prec - 1, 4)
        b.put_signed(shift, 5)
        for c in coefs:
            b.put_signed(c, prec)
        rice_partitions(b, res, order, porder, params_of(res, order, porder, spec, method), method)
        return x

    def subframe_const(b, v, bps):
        b.put(0, 1)
        b.put(0, 6)
        b.put(0, 1)
        b.put_signed(v, bps)
        return [v] * n

    def subframe_verbatim(b, xs, bps):
        b.put(0, 1)
        b.put(1, 6)
        b.put(0, 1)
        for v in xs:
            b.put_signed(int(v), bps)
        return list(xs)

    hand = []
    # frame 0: FIXED order 2 with partitions [rice 3, escape raw 12, rice 0, escape raw 0]; LPC order 32 (precision 15)
    b = Bits()
    x0 = subframe_fixed(b, target(1, 3000, [40]), 2, 2, [3, ("esc", 12), 0, ("esc", 0)], 0, hbits)
    coefs = [8192, -4096] + [int(v) for v in rng.integers(-15, 16, 30)]
    x1 = subframe_lpc(b, target(2, 9000, [300, 1200]), coefs, 15, 12, 0, ["auto"], 0, hbits)
    hand.append((b, [x0, x1]))
    # frame 1: Rice2: parameter 17 then an escape (raw 16); order 0 with parameter 20, escapes of 3 / 1 / 0 bits
    b = Bits()
    xA = subframe_fixed(b, target(3, 20000, [150, 2500], 0.2), 1, 1, [17, ("esc", 16)], 1, hbits)
    xB = subframe_fixed(b, target(4, 9000, [100]), 0, 3, [20, ("esc", 3), "auto", ("esc", 1), "auto", "auto", ("esc", 0), "auto"], 1, hbits)
    hand.append((b, [xA, xB]))
    # frame 2: constant + verbatim
    b = Bits()
    xC = subframe_const(b, -1234, hbits)
    xV = subframe_verbatim(b, [int(v) for v in rng.integers(-32768, 32768, n)], hbits)
    hand.append((b, [xC, xV]))
    # frame 3: 4 wasted bits on FIXED order 4; LPC order 3 with negative coefficients and an escaped (raw 8) half
    b = Bits()
    xW = subframe_fixed(b, [(v >> 4) << 4 for v in target(5, 12000, [250])], 4, 0, ["auto"], 0, hbits, wasted=4)
    xL = subframe_lpc(b, target(6, 7000, [600, 1700]), [-256, 100, -30], 12, 9, 1, ["auto", ("esc", 8)], 0, hbits)
    hand.append((b, [xW, xL]))
    for k, (b, xs) in enumerate(hand):
        for c in range(2):
            mx = max(abs(int(v)) for v in xs[c])
            assert mx <= 32768, ("hand frame", k, "channel", c, "exceeds 16 bits:", mx)
    frames = []
    chans = [[], []]
    for k, (b, xs) in enumerate(hand):
        hdr = frame_header(False, k, n, hrate, 1, hbits)
        b.align()
        body = hdr + b.bytes()
        frames.append(body + struct.pack(">H", crc16(body)))
        for c in range(2):
            chans[c].extend(int(v) for v in xs[c])
    hd = np.stack([np.array(chans[0], np.int64), np.array(chans[1], np.int64)], axis=1)
    add_source("hand16", hd, hrate, hbits)
    sizes = [len(f) for f in frames]
    # STREAMINFO with total samples 0 (unknown) and no MD5, then a PADDING block
    pad = bytes([0x81, 0, 0, 16]) + bytes(16)
    # libFLAC checks the frames with the length in STREAMINFO; the fixture says 0 (unknown), checked with FFmpeg
    known = streaminfo(n, n, min(sizes), max(sizes), hrate, 2, hbits, hd.shape[0], md5_of(hd, 16), last=False, extra=pad)
    check_and_add("hand_escapes_rice2_lpc32_md5", "hand16", known + b"".join(frames),
                  "escaped partitions, Rice2, LPC order 32, constant, verbatim, wasted bits; MD5 set")
    si = streaminfo(n, n, min(sizes), max(sizes), hrate, 2, hbits, 0, last=False, extra=pad)
    check_and_add("hand_escapes_rice2_lpc32", "hand16", si + b"".join(frames),
                  "the same frames, STREAMINFO total samples 0 and no MD5", oracle="ffmpeg")

with open(os.path.join(OUT, "manifest.tsv"), "w", newline="\n") as fh:
    for m in manifest:
        fh.write("\t".join(m) + "\n")
print(len(manifest), "cases")
