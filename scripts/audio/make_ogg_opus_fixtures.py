"""Fixtures for the Ogg Opus writer (Container/Ogg/OggOpusWriter.cs, a port of FFmpeg's ogg muxer) and the Ogg Opus
encode (OpusEncoderGpu + OggOpusWriter, as FFmpeg's libopus wrapper drives libopus).

For each case FFmpeg 8.1 encodes source PCM (s16) with libopus at a bitrate (`-fflags +bitexact -flags +bitexact`):
  <name>.opus      FFmpeg's file (the truth)
  <name>.json      what is inside: OpusHead fields, OpusTags, and every audio packet (offset / length in the file) with the
                   granule of the page it ends on (or -1)
  <src>.s16.gz     the source samples (interleaved int16 little-endian), shared by cases of the same source
  manifest.txt     the case names

Usage: python -I make_ogg_opus_fixtures.py   (ffmpeg on PATH, numpy)
"""
import gzip
import json
import os
import struct
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "OggOpus"))
os.makedirs(OUT, exist_ok=True)


def pages(b):
    p = 0
    while p < len(b):
        assert b[p:p + 4] == b"OggS"
        flags = b[p + 5]
        granule = struct.unpack("<q", b[p + 6:p + 14])[0]
        nseg = b[p + 26]
        segs = b[p + 27:p + 27 + nseg]
        data = p + 27 + nseg
        yield flags, granule, segs, data
        p = data + sum(segs)


def packets(b):
    """Every packet with the granule of the page it ends on (-1 when it does not end the page's last packet)."""
    out = []
    cur_start, cur_len = None, 0
    for flags, granule, segs, data in pages(b):
        at = data
        ends = []
        for s in segs:
            if cur_start is None:
                cur_start = at
            cur_len += s
            at += s
            if s < 255:
                ends.append([cur_start, cur_len])
                cur_start, cur_len = None, 0
        for i, e in enumerate(ends):
            e.append(granule if i == len(ends) - 1 else -1)
            out.append(e)
    return out


def tone(n, rate, freqs, amp, seed):
    r = np.random.default_rng(seed)
    t = np.arange(n) / rate
    return sum(np.sin(2 * np.pi * f * t + r.random()) for f in freqs) / len(freqs) * amp + r.normal(0, amp * 0.02, n)


sources = {
    "stereo48": (48000, 2, np.stack([tone(48000 * 3 + 517, 48000, [220, 660, 1800], 9000, 1),
                                     tone(48000 * 3 + 517, 48000, [330, 990], 8000, 2)], axis=1)),
    "mono48": (48000, 1, tone(48000 * 2 + 77, 48000, [440, 3000], 10000, 3)[:, None]),
    "stereo24": (24000, 2, np.stack([tone(24000 * 2 + 11, 24000, [300, 900], 9000, 4),
                                     tone(24000 * 2 + 11, 24000, [310, 700], 9000, 5)], axis=1)),
}
# (source, kbps, complexity): 10 is libopus' (and FFmpeg's, PyAV's, ComfyUI's) default; 9 is the highest OpusEncoderGpu
# supports (10 adds libopus' tonality analysis)
cases = [("stereo48", 64, 10), ("stereo48", 96, 10), ("stereo48", 128, 10), ("stereo48", 192, 10), ("stereo48", 320, 10),
         ("mono48", 64, 10), ("mono48", 128, 10), ("stereo24", 96, 10),
         ("stereo48", 128, 9), ("mono48", 64, 9), ("stereo24", 96, 9)]
manifest = []
with tempfile.TemporaryDirectory() as tmp:
    for src, (rate, ch, data) in sources.items():
        pcm = np.clip(np.round(data), -32768, 32767).astype("<i2")
        with open(os.path.join(OUT, src + ".s16.gz"), "wb") as fh:
            fh.write(gzip.compress(pcm.tobytes(), mtime=0))
        pcm.tofile(os.path.join(tmp, src + ".raw"))
    for src, kbps, cx in cases:
        rate, ch, data = sources[src]
        name = f"{src}_{kbps}k" + ("" if cx == 10 else f"_c{cx}")
        out = os.path.join(OUT, name + ".opus")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "s16le", "-ar", str(rate), "-ac", str(ch),
                        "-i", os.path.join(tmp, src + ".raw"), "-c:a", "libopus", "-b:a", f"{kbps}k", "-compression_level", str(cx),
                        "-fflags", "+bitexact", "-flags", "+bitexact", out], check=True)
        b = open(out, "rb").read()
        pk = packets(b)
        head = b[pk[0][0]:pk[0][0] + pk[0][1]]
        tags = b[pk[1][0]:pk[1][0] + pk[1][1]]
        doc = {
            "source": src, "rate": rate, "channels": ch, "samples": int(data.shape[0]), "bitrate": kbps * 1000, "complexity": cx,
            "preSkip": struct.unpack("<H", head[10:12])[0], "inputRate": struct.unpack("<I", head[12:16])[0],
            "tags": tags.hex(), "packets": pk[2:],
        }
        with open(os.path.join(OUT, name + ".json"), "w", newline="\n") as fh:
            json.dump(doc, fh)
        manifest.append(name)
        print(name, len(b), "bytes", len(pk) - 2, "packets", "preskip", doc["preSkip"], "last granule", pk[-1][2])
with open(os.path.join(OUT, "manifest.txt"), "w", newline="\n") as fh:
    fh.write("\n".join(manifest) + "\n")
