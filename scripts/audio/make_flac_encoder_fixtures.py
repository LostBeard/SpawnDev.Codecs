"""Fixtures for the FLAC ENCODER (FlacEncoderGpu.EncodeAsync, a port of FFmpeg 8.1's flacenc + FLAC muxer).

For each source signal: <src>.s32.gz (the interleaved samples as int32 little-endian, the value range of the bit depth)
and FFmpeg's .flac of it at several compression levels (`-fflags +bitexact -flags +bitexact`, deterministic), the
expected file, made with `-cpuflags 0` (FFmpeg's C code, the same on every CPU; its x86 SIMD autocorrelation rounds
differently). Our port of that C code must write the same bytes.

Writes SpawnDev.Codecs.Demo.Shared/TestData/FlacEnc/: <src>.s32.gz, <src>_l<level>.flac, manifest.tsv
(name, source, channels, rate, bits, samples per channel, level).

Usage: python -I make_flac_encoder_fixtures.py   (ffmpeg on PATH, numpy)
"""
import gzip
import os
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "FlacEnc"))
os.makedirs(OUT, exist_ok=True)


def tone(n, rate, freqs, amp, seed, noise=0.0):
    r = np.random.default_rng(seed)
    t = np.arange(n) / rate
    s = sum(np.sin(2 * np.pi * f * t + r.random() * 6.28) for f in freqs) / len(freqs)
    return s * amp + r.normal(0, noise * amp if noise else 0, n)


def to_int(x, bits):
    lim = (1 << (bits - 1)) - 1
    return np.clip(np.round(x), -lim - 1, lim).astype(np.int64)


sources = []


def source(name, chans, rate, bits):
    """chans: list of float arrays (equal length) in sample units."""
    data = np.stack([to_int(c, bits) for c in chans], axis=1)   # (n, channels)
    sources.append((name, data, rate, bits))


rng = np.random.default_rng(7)
n = int(44100 * 1.3) + 17                       # not a multiple of the block size
source("stereo_tone_16", [tone(n, 44100, [440, 660], 12000, 1), tone(n, 44100, [445, 990], 9000, 2)], 44100, 16)
source("stereo_corr_16", [tone(n, 44100, [300, 1200], 15000, 3, 0.01)] * 1 + [tone(n, 44100, [300, 1200], 15000, 3, 0.01) * 0.98], 44100, 16)
m = int(48000 * 0.9) + 5
source("mono_noise_16", [rng.normal(0, 4000, m)], 48000, 16)
q = int(48000 * 0.8) + 3
source("stereo_tone_24", [tone(q, 48000, [220, 880, 3000], 3_000_000, 4, 0.001), tone(q, 48000, [230, 870], 2_500_000, 5, 0.001)], 48000, 24)
w = 9000
sil = np.zeros(w)
sil[3000:6000] = tone(3000, 22050, [500], 8000, 6)
source("mono_silence_mix_16", [sil], 22050, 16)
wb = (tone(7000, 32000, [400], 6000, 8) // 8) * 8           # every sample a multiple of 8: 3 wasted bits
source("mono_wasted_16", [wb], 32000, 16)
source("tiny_3_16", [np.array([100.0, -200.0, 300.0]), np.array([5.0, 6.0, 7.0])], 44100, 16)
k = 5000
source("five_ch_16", [tone(k, 44100, [200 + 100 * c], 7000, 10 + c) for c in range(5)], 44100, 16)
source("odd_rate_16", [tone(6000, 44056, [1000], 9000, 20), tone(6000, 44056, [1100], 9000, 21)], 44056, 16)
source("hi_rate_24", [tone(20000, 96000, [1000, 7000], 2_000_000, 30)], 96000, 24)
# full-scale white noise: no predictor beats verbatim coding -> FFmpeg's verbatim fallback (frame larger than verbatim)
wn = np.random.default_rng(40).integers(-32768, 32768, (2, 7000)).astype(np.float64)
source("white_full_16", [wn[0], wn[1]], 44100, 16)
source("white_full_mono_16", [np.random.default_rng(41).integers(-32768, 32768, 6000).astype(np.float64)], 44100, 16)
const = np.full(4000, 1234.0)
source("constant_16", [const, -const], 44100, 16)

LEVELS = [0, 1, 2, 5, 8]
manifest = []
with tempfile.TemporaryDirectory() as tmp:
    for name, data, rate, bits in sources:
        n_, ch = data.shape
        with open(os.path.join(OUT, name + ".s32.gz"), "wb") as fh:
            fh.write(gzip.compress(data.astype("<i4").tobytes(), mtime=0))
        raw = os.path.join(tmp, name + ".raw")
        if bits == 16:
            data.astype("<i2").tofile(raw)
            fmt = "s16le"
        else:
            (data.astype(np.int64) << 8).astype("<i4").tofile(raw)   # 24-bit in the top of s32, as FFmpeg reads s32
            fmt = "s32le"
        for level in LEVELS:
            out = os.path.join(OUT, f"{name}_l{level}.flac")
            # -cpuflags 0: FFmpeg's C code paths (its x86 SIMD window / autocorrelation round differently, and differ by CPU)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-cpuflags", "0", "-f", fmt, "-ar", str(rate), "-ac", str(ch),
                   "-i", raw, "-c:a", "flac", "-compression_level", str(level), "-fflags", "+bitexact", "-flags", "+bitexact", out]
            subprocess.run(cmd, check=True)
            manifest.append(f"{name}_l{level}\t{name}\t{ch}\t{rate}\t{bits}\t{n_}\t{level}")
            print(name, level, os.path.getsize(out))
with open(os.path.join(OUT, "manifest.tsv"), "w", newline="\n") as fh:
    fh.write("\n".join(manifest) + "\n")
