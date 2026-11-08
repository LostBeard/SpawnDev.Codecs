"""Fixtures for the MP3 decoder (Mp3DecoderGpu): MP3 files from encoders used as black boxes (FFmpeg's libmp3lame and
Windows' MediaFoundation MP3 encoder, mp3_mf), each with minimp3's decode of it.

  <case>.mp3          the file
  <case>.mm3.f32.gz   minimp3's decode (float output; tags and the Info frame skipped by minimp3_ex's helpers; delay + 529
                      dropped at the start, padding - 529 at the end of the decoded frames, FFmpeg's convention),
                      interleaved float32 little-endian, gzip
  manifest.tsv        case, rate, channels, source samples per channel (-1 when the file is not a gapless encode of a
                      known length), minimp3's samples per channel, minimp3_ex mp3dec_load_buf's samples per channel
                      (it cuts to the Info frame's frame count: different only when that count is wrong), what the file
                      exercises

minimp3 (github.com/lieff/minimp3, CC0) runs as scripts/audio/minimp3_ref.c built with MSVC x64 (see that file). No MP3
encoder or decoder code is read here: FFmpeg / LAME / MediaFoundation only produce files.

Coverage: CBR 128 / 320, VBR V0 / V4 / V6, mono / stereo, MPEG-1 (44.1 / 48 / 32 kHz), MPEG-2 (22.05 / 24 / 16 kHz),
MPEG-2.5 (8 / 11.025 / 12 kHz), short-block-heavy transients, joint (M/S) and plain L/R stereo, intensity stereo
(mp3_mf at a low rate), ID3v2 with TXXX tags + Info frame + ID3v1, no tags and no Info frame, free-format (a CBR
file's bitrate index set to 0 in every header). The side info of every file is parsed here to report what it holds.

Usage: python -I make_mp3_decoder_fixtures.py <minimp3_ref.exe>    (ffmpeg on PATH with libmp3lame, numpy)
"""
import gzip
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "Mp3Dec"))
REF = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))
rng = np.random.default_rng(1108)
TMP = tempfile.mkdtemp(prefix="mp3dec-fx-")


def music(n, rate, ch):
    """Harmonic tones with vibrato, a shared (centre) part and a per-channel part, light noise."""
    t = np.arange(n) / rate
    out = np.zeros((n, ch))
    centre = np.zeros(n)
    for k, f in enumerate([196.0, 293.7, 392.0, 587.3]):
        centre += 0.08 / (k + 1) * np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 5 * t))
    for c in range(ch):
        own = np.zeros(n)
        for k, f in enumerate([440.0 * (1 + 0.25 * c), 1250.0 + 400 * c, min(5200.0, rate / 3.2)]):
            own += 0.06 / (k + 1) * np.sin(2 * np.pi * f * t + c)
        env = 0.6 + 0.4 * np.sin(2 * np.pi * (1.5 + c) * t)
        out[:, c] = centre + own * env + 0.004 * rng.standard_normal(n)
    return out


def transients(n, rate, ch):
    """Sharp noise bursts every 70 ms over a quiet tone: pre-echo control wants short blocks."""
    t = np.arange(n) / rate
    out = np.zeros((n, ch))
    for c in range(ch):
        x = 0.05 * np.sin(2 * np.pi * (330 + 110 * c) * t)
        period = int(0.07 * rate)
        for start in range(int(0.01 * rate) + 37 * c, n, period):
            m = min(n - start, int(0.012 * rate))
            burst = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.002 * rate)) * 0.7
            x[start:start + m] += burst
        out[:, c] = x
    return out


def wide(n, rate, ch):
    """Different content per channel in the highs (intensity stereo codes the highs as one channel plus positions)."""
    t = np.arange(n) / rate
    out = music(n, rate, ch)
    for c in range(ch):
        out[:, c] += 0.05 * np.sin(2 * np.pi * (6000 + 2500 * c) * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t + c))
    return out


def centred(n, rate, ch):
    """Nearly the same signal in both channels (a small side part): joint stereo codes it as mid / side."""
    x = music(n, rate, 1)[:, 0]
    t = np.arange(n) / rate
    side = 0.01 * np.sin(2 * np.pi * 700 * t)
    return np.stack([x + side, x - side], axis=1)


def write_wav(path, x, rate):
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def run(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0 or (args[0] == "ffmpeg" and r.stderr.strip()):
        raise RuntimeError(f"{args}: {r.stderr}")
    return r.stdout


def minimp3(path):
    out = os.path.join(TMP, "ref.f32")
    ch, hz, n, ex = (int(v) for v in run([REF, path, out]).split())
    data = open(out, "rb").read()
    assert len(data) == n * ch * 4, (path, len(data), n, ch)
    return ch, hz, n, ex, data


# ------------------------------------------------------------------ side info stats (report only)

BR = {1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
      2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]}
SR = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


class Bits:
    def __init__(self, d, at):
        self.d, self.p = d, at * 8

    def get(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | ((self.d[self.p >> 3] >> (7 - (self.p & 7))) & 1)
            self.p += 1
        return v


def stats(d):
    """Frames, short-block granule-channels, M/S frames, intensity frames, free-format frames."""
    p = 0
    if d[:3] == b"ID3":
        p = 10 + ((d[6] & 0x7F) << 21 | (d[7] & 0x7F) << 14 | (d[8] & 0x7F) << 7 | (d[9] & 0x7F))
    frames = short = units = ms = intensity = free = 0
    free_size = 0
    while p + 4 <= len(d) and d[p] == 0xFF and (d[p + 1] & 0xE0) == 0xE0:
        ver = (d[p + 1] >> 3) & 3
        mpeg1 = ver == 3
        bri, sri, pad = d[p + 2] >> 4, (d[p + 2] >> 2) & 3, (d[p + 2] >> 1) & 1
        mode, ext = d[p + 3] >> 6, (d[p + 3] >> 4) & 3
        rate = SR[ver][sri]
        spf = 1152 if mpeg1 else 576
        if bri == 0:
            free += 1
            if free_size == 0:
                q = p + 4
                while not (d[q] == 0xFF and d[q + 1] == d[p + 1] and (d[q + 2] & 0x0C) == (d[p + 2] & 0x0C)):
                    q += 1
                free_size = q - p - pad
            size = free_size + pad
        else:
            size = spf // 8 * BR[1 if mpeg1 else 2][bri] * 1000 // rate + pad
        nch = 1 if mode == 3 else 2
        b = Bits(d, p + 4 + (0 if (d[p + 1] & 1) else 2))
        if mpeg1:
            b.get(9)
            b.get(5 if nch == 1 else 3)
            b.get(4 * nch)
        else:
            b.get(8)
            b.get(nch)
        for _ in range(2 if mpeg1 else 1):
            for _ in range(nch):
                b.get(12 + 9 + 8 + (4 if mpeg1 else 9))
                units += 1
                if b.get(1):
                    bt = b.get(2)
                    b.get(1 + 10 + 9)
                    short += bt == 2
                else:
                    b.get(15 + 4 + 3)
                b.get((1 if mpeg1 else 0) + 2)
        frames += 1
        ms += mode == 1 and (ext & 2) != 0
        intensity += mode == 1 and (ext & 1) != 0
        p += size
    return frames, short, units, ms, intensity, free


# ------------------------------------------------------------------ cases

# name, source, rate, channels, seconds, encoder args, note
CASES = [
    ("cbr128_st44", music, 44100, 2, 0.55, ["-c:a", "libmp3lame", "-b:a", "128k"], "CBR 128k stereo 44.1 kHz, joint stereo"),
    ("cbr320_st44", music, 44100, 2, 0.5, ["-c:a", "libmp3lame", "-b:a", "320k"], "CBR 320k stereo 44.1 kHz"),
    ("v0_st44", music, 44100, 2, 0.5, ["-c:a", "libmp3lame", "-q:a", "0"], "VBR V0 stereo 44.1 kHz"),
    ("v4_mono48", music, 48000, 1, 0.6, ["-c:a", "libmp3lame", "-q:a", "4"], "VBR V4 mono 48 kHz"),
    ("cbr96_st32", music, 32000, 2, 0.6, ["-c:a", "libmp3lame", "-b:a", "96k"], "CBR 96k stereo 32 kHz"),
    ("lr_st48", music, 48000, 2, 0.45, ["-c:a", "libmp3lame", "-b:a", "192k", "-joint_stereo", "0"], "plain L/R stereo 48 kHz"),
    ("lsf_st22", music, 22050, 2, 0.8, ["-c:a", "libmp3lame", "-b:a", "64k"], "MPEG-2 22.05 kHz stereo 64k"),
    ("lsf_v4_mono24", music, 24000, 1, 0.8, ["-c:a", "libmp3lame", "-q:a", "4"], "MPEG-2 24 kHz mono VBR V4"),
    ("lsf_mono16", music, 16000, 1, 0.9, ["-c:a", "libmp3lame", "-b:a", "32k"], "MPEG-2 16 kHz mono 32k"),
    ("m25_mono8", music, 8000, 1, 1.2, ["-c:a", "libmp3lame", "-b:a", "16k"], "MPEG-2.5 8 kHz mono 16k"),
    ("m25_st11", music, 11025, 2, 1.0, ["-c:a", "libmp3lame", "-b:a", "24k"], "MPEG-2.5 11.025 kHz stereo 24k"),
    ("m25_v6_mono12", music, 12000, 1, 1.0, ["-c:a", "libmp3lame", "-q:a", "6"], "MPEG-2.5 12 kHz mono VBR V6"),
    ("short_st44", transients, 44100, 2, 0.55, ["-c:a", "libmp3lame", "-b:a", "160k"], "transients: short blocks, 44.1 kHz stereo"),
    ("short_lsf_mono22", transients, 22050, 1, 0.8, ["-c:a", "libmp3lame", "-b:a", "48k"], "transients: short blocks, MPEG-2 mono"),
    ("short_m25_st12", transients, 12000, 2, 1.0, ["-c:a", "libmp3lame", "-b:a", "32k"], "transients: short blocks, MPEG-2.5 stereo"),
    ("ms_st44", centred, 44100, 2, 0.5, ["-c:a", "libmp3lame", "-b:a", "96k", "-joint_stereo", "1"], "joint stereo at 96k (M/S frames)"),
    ("tags_st44", music, 44100, 2, 0.4, ["-c:a", "libmp3lame", "-b:a", "128k", "-write_id3v1", "1",
                                          "-metadata", "title=MP3 decoder fixture",
                                          "-metadata", "prompt={\"3\": {\"class_type\": \"KSampler\"}}",
                                          "-metadata", "workflow=spawndev"], "ID3v2 (TXXX) + Info frame + ID3v1"),
    ("notag_mono44", music, 44100, 1, 0.4, ["-c:a", "libmp3lame", "-b:a", "64k", "-write_xing", "0", "-id3v2_version", "0"],
     "no ID3v2, no Info frame (no gapless trim)"),
    ("is_st44", wide, 44100, 2, 0.6, ["-c:a", "mp3_mf", "-b:a", "40k"], "MediaFoundation 40k: intensity stereo frames"),
    ("is_st48", wide, 48000, 2, 0.6, ["-c:a", "mp3_mf", "-b:a", "32k"], "MediaFoundation 32k 48 kHz: intensity stereo frames"),
]

manifest = []
total = 0
for name, gen, rate, ch, secs, args, note in CASES:
    n = int(round(rate * secs))
    src = gen(n, rate, ch)
    wav = os.path.join(TMP, name + ".wav")
    write_wav(wav, src, rate)
    mp3 = os.path.join(OUT, name + ".mp3")
    run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", wav] + args + [mp3])
    gapless = args[1] == "libmp3lame" and "-write_xing" not in args
    files = [(name, mp3, gapless, note)]
    if name == "cbr128_st44":
        # free format: the same frames with bitrate index 0 (each frame's size is then the distance between headers)
        d = bytearray(open(mp3, "rb").read())
        p = 10 + ((d[6] & 0x7F) << 21 | (d[7] & 0x7F) << 14 | (d[8] & 0x7F) << 7 | (d[9] & 0x7F)) if d[:3] == b"ID3" else 0
        count = 0
        while p + 4 <= len(d) and d[p] == 0xFF and (d[p + 1] & 0xE0) == 0xE0:
            size = 144 * 128000 // 44100 + ((d[p + 2] >> 1) & 1)
            d[p + 2] &= 0x0F
            p += size
            count += 1
        free = os.path.join(OUT, "free_st44.mp3")
        open(free, "wb").write(bytes(d))
        files.append(("free_st44", free, True, f"free format ({count} frames of cbr128_st44 with bitrate index 0)"))
    for fname, path, gl, fnote in files:
        mch, hz, mn, ex, data = minimp3(path)
        assert mch == ch and hz == rate, (fname, mch, hz)
        if gl:
            assert mn == n and ex == n, f"{fname}: minimp3 gives {mn} / {ex} samples, the source has {n}"
        fr, short, units, ms, ist, free = stats(open(path, "rb").read())
        fnote += f"; mp3dec_load_buf gives {ex} samples" if ex != mn else ""
        fnote += f"; {fr} frames, {short}/{units} short granule-channels, {ms} M/S, {ist} intensity, {free} free-format"
        gz = gzip.compress(data, 9, mtime=0)
        open(os.path.join(OUT, fname + ".mm3.f32.gz"), "wb").write(gz)
        size = os.path.getsize(path) + len(gz)
        total += size
        manifest.append(f"{fname}\t{rate}\t{ch}\t{n if gl else -1}\t{mn}\t{ex}\t{fnote}")
        print(f"{fname}: {fnote} ({size} bytes)")
        if fname.startswith("is_"):
            assert ist > 0, f"{fname}: no intensity stereo frame"
        if fname.startswith("short_"):
            assert short > 0, f"{fname}: no short block"

open(os.path.join(OUT, "manifest.tsv"), "w", newline="\n").write("\n".join(manifest) + "\n")
print(f"{len(manifest)} fixtures, {total} bytes")
