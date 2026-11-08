"""Page layout fixtures for the Ogg Opus writer (Container/Ogg/OggOpusWriter.cs): the edge cases the encode fixtures
(make_ogg_opus_fixtures.py) never reach.

Each case is a crafted Ogg Opus stream (written here per RFC 3533 / RFC 7845, one packet per page; every packet is a
20 ms CELT TOC byte 0xFC followed by filler bytes of a chosen length) that the ffmpeg binary remuxes
(`-c copy -fflags +bitexact -flags +bitexact -page_duration <us>`). FFmpeg's output is the truth: given its packets
(pts -preSkip + 960 k, the last packet's duration from the final granule) and the page duration, OggOpusWriter must
write it byte for byte. The cases cover: a packet larger than a page (granule -1 page), the 65025-byte flush rule, a
packet split across pages (continued flag; the continued page's start time), a zero lacing value on its own page, the
first page's start time (PreSkip before the first pts) with microsecond rounding (halves up), a full last page
(end-of-stream flag), an OpusTags packet spanning pages, page duration 0 (no limit) and a stream with no audio packets.

Writes SpawnDev.Codecs.Demo.Shared/TestData/OggOpusLayout/: <name>.opus.gz and manifest.txt (name, page duration us).

Usage: python -I make_ogg_opus_layout_fixtures.py   (ffmpeg on PATH)
"""
import gzip
import os
import struct
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "OggOpusLayout"))
os.makedirs(OUT, exist_ok=True)


def crc_table():
    t = []
    for i in range(256):
        r = i << 24
        for _ in range(8):
            r = ((r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if r & 0x80000000 else (r << 1) & 0xFFFFFFFF
        t.append(r)
    return t


CRC = crc_table()


def crc(b):
    c = 0
    for x in b:
        c = ((c << 8) & 0xFFFFFFFF) ^ CRC[((c >> 24) & 0xFF) ^ x]
    return c


def pages_of(flags, granule, seq, packet):
    """The pages of one packet on pages of its own (continued pages when it needs more than 255 lacing values)."""
    lacing = [255] * (len(packet) // 255) + [len(packet) % 255]
    out, pos, first = [], 0, True
    while True:
        chunk, lacing = lacing[:255], lacing[255:]
        f = flags | (0 if first else 1)
        g = granule if not lacing else -1
        h = bytearray(b"OggS" + bytes([0, f]) + struct.pack("<qII", g, 0, seq) + b"\0\0\0\0" + bytes([len(chunk)]) + bytes(chunk))
        body = packet[pos:pos + sum(chunk)]
        pos += sum(chunk)
        h[22:26] = struct.pack("<I", crc(bytes(h) + body))
        out.append(bytes(h) + body)
        seq += 1
        first = False
        if not lacing:
            return b"".join(out), seq


def crafted(sizes, pre_skip, comment_len=0):
    head = b"OpusHead" + bytes([1, 2]) + struct.pack("<HIhB", pre_skip, 48000, 0, 0)
    comments = [b"comment=" + b"x" * comment_len] if comment_len else []
    tags = b"OpusTags" + struct.pack("<I", 6) + b"ffmpeg" + struct.pack("<I", len(comments))
    for c in comments:
        tags += struct.pack("<I", len(c)) + c
    data, seq = b"", 0
    for flags, packet in ((2, head), (0, tags)):
        p, seq = pages_of(flags, 0, seq, packet)
        data += p
    for i, s in enumerate(sizes):
        packet = bytes([0xFC]) + bytes(((i * 7 + j) & 0xFF) for j in range(s - 1))
        p, seq = pages_of(4 if i == len(sizes) - 1 else 0, 960 * (i + 1), seq, packet)
        data += p
    return data


def n(count, size):
    return [size] * count


# name: (packet sizes, pre-skip, page duration in us, OpusTags comment length, extra ffmpeg output args)
cases = {
    "spill_then_duration": (n(64, 1000) + n(150, 10), 312, 1500000, 0, []),
    "zero_lacing_and_huge": (n(63, 1000) + [10, 510] + n(5, 10) + [66000] + n(5, 10), 312, 0, 0, []),
    "byte_limit_flush": (n(63, 1000) + [2026, 10], 312, 0, 0, []),
    "byte_limit_split": (n(63, 1000) + [2025, 10], 312, 0, 0, []),
    "first_page_start_in": (n(130, 10), 5000, 1104167, 0, []),
    "first_page_start_out": (n(130, 10), 5000, 1104168, 0, []),
    "first_page_half_in": (n(130, 10), 3, 1000063, 0, []),
    "first_page_half_out": (n(130, 10), 3, 1000064, 0, []),
    "full_last_page": (n(5, 10) + [65024], 312, 1000000, 0, []),
    "zero_lacing_last": ([65025], 312, 1000000, 0, []),
    "tags_span_pages": (n(60, 300), 312, 1000000, 70000, []),
    "short_pages": (n(40, 900), 312, 30000, 0, []),
    "no_audio": (n(3, 10), 312, 1000000, 0, ["-frames:a", "0"]),
}
manifest = []
with tempfile.TemporaryDirectory() as tmp:
    for name, (sizes, pre, pd, clen, extra) in cases.items():
        src = os.path.join(tmp, name + ".in.opus")
        out = os.path.join(tmp, name + ".opus")
        with open(src, "wb") as fh:
            fh.write(crafted(sizes, pre, clen))
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-c", "copy", *extra, "-fflags", "+bitexact",
                        "-flags", "+bitexact", "-page_duration", str(pd), out], check=True)
        b = open(out, "rb").read()
        with open(os.path.join(OUT, name + ".opus.gz"), "wb") as fh:
            fh.write(gzip.compress(b, mtime=0))
        manifest.append(f"{name} {pd}")
        print(name, len(b), "bytes")
with open(os.path.join(OUT, "manifest.txt"), "w", newline="\n") as fh:
    fh.write("\n".join(manifest) + "\n")
