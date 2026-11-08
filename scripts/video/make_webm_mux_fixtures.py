"""Fixtures for the WebM MUXER (Container/Matroska/WebMWriter.cs, a clean-room WebM muxer).

For each case ffmpeg (8.1) encodes a short synthetic clip and muxes it with `-fflags +bitexact` (deterministic: no
random UIDs, "Lavf" app strings). That file is the truth. A small EBML reader below lists what the muxer was given:
the tracks (codec, CodecPrivate, size, colour, rates, delay, tags) and every packet (track, timestamp, key frame flag,
payload, BlockAdditional, duration, DiscardPadding), payloads as offsets into the truth file. The test hands exactly
those to WebMWriter and requires the same bytes.

Writes SpawnDev.Codecs.Demo.Shared/TestData/WebMMux/<case>.webm + <case>.json and manifest.txt.
Usage: python -I make_webm_mux_fixtures.py [case ...]   (ffmpeg on PATH; no case names = every case)
"""
import json
import os
import struct
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "WebMMux"))
os.makedirs(OUT, exist_ok=True)

BX = ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact"]


def ff(args, out, pipe=False):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args + BX
    if pipe:
        cmd += ["-f", "webm", "pipe:1"]
        with open(out, "wb") as fh:
            subprocess.run(cmd, check=True, stdout=fh)
    else:
        subprocess.run(cmd + [out], check=True)


# ---- minimal EBML reader
def vint(b, p):
    first = b[p]
    n = 1
    while n <= 8 and not (first & (0x80 >> (n - 1))):
        n += 1
    val = first & ((0x80 >> (n - 1)) - 1)
    for i in range(1, n):
        val = (val << 8) | b[p + i]
    unknown = val == (1 << (7 * n)) - 1
    return val, n, unknown


def elid(b, p):
    first = b[p]
    n = 1
    while not (first & (0x80 >> (n - 1))):
        n += 1
    return int.from_bytes(b[p:p + n], "big"), n


MASTERS = {0x18538067, 0x1549A966, 0x1654AE6B, 0xAE, 0xE0, 0xE1, 0x55B0, 0x1254C367, 0x7373, 0x63C0, 0x67C8,
           0x1F43B675, 0xA0, 0x75A1, 0xA6, 0x1C53BB6B, 0x114D9B74}


def children(b, start, end):
    p = start
    while p < end:
        i, n = elid(b, p)
        size, m, unk = vint(b, p + n)
        data = p + n + m
        e = end if unk else data + size
        yield i, data, e
        p = e


def children_h(b, start, end):
    """Like children(), also giving where each element's header starts."""
    p = start
    while p < end:
        i, n = elid(b, p)
        size, m, unk = vint(b, p + n)
        data = p + n + m
        e = end if unk else data + size
        yield i, p, data, e
        p = e


def lensize(n):
    k = 1
    while (n + 1) >> (7 * k):
        k += 1
    return k


def uint(b, s, e):
    return int.from_bytes(b[s:e], "big") if e > s else 0


def sint(b, s, e):
    return int.from_bytes(b[s:e], "big", signed=True) if e > s else 0


def flt(b, s, e):
    return struct.unpack(">d", b[s:e])[0] if e - s == 8 else struct.unpack(">f", b[s:e])[0]


def strv(b, s, e):
    return b[s:e].decode("utf-8")


def parse(path):
    b = open(path, "rb").read()
    seg = next((d, e) for i, d, e in children(b, 0, len(b)) if i == 0x18538067)
    info = {"title": None, "muxing": None, "writing": None}
    tracks, gtags, ttags = [], [], {}
    packets = []
    for i, s, e in children(b, *seg):
        if i == 0x1549A966:
            for j, s2, e2 in children(b, s, e):
                if j == 0x7BA9: info["title"] = strv(b, s2, e2)
                if j == 0x4D80: info["muxing"] = strv(b, s2, e2)
                if j == 0x5741: info["writing"] = strv(b, s2, e2)
        elif i == 0x1654AE6B:
            for j, s2, e2 in children(b, s, e):
                t = {"default": True, "forced": False}
                # room reserved for CodecPrivate: the CodecPrivate element plus a Void right after it (or a lone Void)
                prev_cp, cp_total = None, 0
                for k, h3, s3, e3 in children_h(b, s2, e2):
                    if k == 0x63A2:
                        cp_total = e3 - h3
                        prev_cp = True
                    elif k == 0xEC and (prev_cp or "codecPrivate" not in t):
                        total = cp_total + (e3 - h3)
                        t["codecPrivateReserve"] = next(m for m in range(total, 0, -1) if 2 + lensize(m) + m == total)
                        prev_cp = False
                    else:
                        prev_cp = False
                for k, s3, e3 in children(b, s2, e2):
                    if k == 0xD7: t["number"] = uint(b, s3, e3)
                    elif k == 0x73C5: t["uid"] = uint(b, s3, e3)
                    elif k == 0x536E: t["name"] = strv(b, s3, e3)
                    elif k == 0x22B59C: t["language"] = strv(b, s3, e3)
                    elif k == 0x88: t["default"] = uint(b, s3, e3) != 0
                    elif k == 0x55AA: t["forced"] = uint(b, s3, e3) != 0
                    elif k == 0x86: t["codec"] = strv(b, s3, e3)
                    elif k == 0x83: t["type"] = uint(b, s3, e3)
                    elif k == 0x23E383: t["defaultDurationNs"] = uint(b, s3, e3)
                    elif k == 0x56AA: t["codecDelayNs"] = uint(b, s3, e3)
                    elif k == 0x63A2: t["codecPrivate"] = b[s3:e3].hex()
                    elif k == 0xE0:
                        for v, s4, e4 in children(b, s3, e3):
                            if v == 0xB0: t["width"] = uint(b, s4, e4)
                            elif v == 0xBA: t["height"] = uint(b, s4, e4)
                            elif v == 0x9A: t["interlaced"] = uint(b, s4, e4) == 1
                            elif v == 0x53C0: t["alphaMode"] = uint(b, s4, e4) == 1
                            elif v == 0x54B0: t["displayWidth"] = uint(b, s4, e4)
                            elif v == 0x54BA: t["displayHeight"] = uint(b, s4, e4)
                            elif v == 0x55B0:
                                c = {}
                                names = {0x55B1: "matrix", 0x55B7: "sitingH", 0x55B8: "sitingV", 0x55B9: "range",
                                         0x55BA: "transfer", 0x55BB: "primaries"}
                                for w, s5, e5 in children(b, s4, e4):
                                    c[names[w]] = uint(b, s5, e5)
                                t["colour"] = c
                    elif k == 0xE1:
                        for a, s4, e4 in children(b, s3, e3):
                            if a == 0x9F: t["channels"] = uint(b, s4, e4)
                            elif a == 0xB5: t["sampleRate"] = flt(b, s4, e4)
                            elif a == 0x6264: t["bitDepth"] = uint(b, s4, e4)
                tracks.append(t)
        elif i == 0x1254C367:
            for j, s2, e2 in children(b, s, e):   # Tag
                target, simple = None, []
                for k, s3, e3 in children(b, s2, e2):
                    if k == 0x63C0:
                        for q, s4, e4 in children(b, s3, e3):
                            if q == 0x63C5: target = uint(b, s4, e4)
                    elif k == 0x67C8:
                        name = val = None
                        for q, s4, e4 in children(b, s3, e3):
                            if q == 0x45A3: name = strv(b, s4, e4)
                            if q == 0x4487: val = strv(b, s4, e4)
                        if name != "DURATION":
                            simple.append([name, val])
                if target is None: gtags += simple
                else: ttags[target] = simple
        elif i == 0x1F43B675:
            cts = 0
            for j, s2, e2 in children(b, s, e):
                if j == 0xE7: cts = uint(b, s2, e2)
                elif j in (0xA3, 0xA0):
                    pk = {"duration": None, "add": None, "discardNs": 0}
                    blk = (s2, e2) if j == 0xA3 else None
                    if j == 0xA0:
                        for k, s3, e3 in children(b, s2, e2):
                            if k == 0xA1: blk = (s3, e3)
                            elif k == 0x9B: pk["duration"] = uint(b, s3, e3)
                            elif k == 0x75A2: pk["discardNs"] = sint(b, s3, e3)
                            elif k == 0xFB: pk["ref"] = sint(b, s3, e3)
                            elif k == 0x75A1:
                                for m_, s4, e4 in children(b, s3, e3):
                                    for q, s5, e5 in children(b, s4, e4):
                                        if q == 0xA5: pk["add"] = [s5, e5 - s5]
                    tn, n, _ = vint(b, blk[0])
                    rel = struct.unpack(">h", b[blk[0] + n:blk[0] + n + 2])[0]
                    flags = b[blk[0] + n + 2]
                    pk["track"] = tn
                    pk["ts"] = cts + rel
                    pk["key"] = (flags & 0x80) != 0 if j == 0xA3 else ("ref" not in pk)
                    pk["data"] = [blk[0] + n + 3, blk[1] - (blk[0] + n + 3)]
                    packets.append(pk)
    return b, info, tracks, gtags, ttags, packets


OPUS_DUR = [480, 960, 1920, 2880] * 3 + [480, 960, 480, 960] + [120, 240, 480, 960] * 4


def opus_ms(b, off, n):
    if n == 0: return -1
    code, cfg = b[off] & 3, b[off] >> 3
    frames = 1 if code == 0 else 2 if code < 3 else (b[off + 1] & 0x3F)
    s = frames * OPUS_DUR[cfg]
    return (s * 1000 + 24000) // 48000


def manifest(name, path, live=False):
    b, info, tracks, gtags, ttags, packets = parse(path)
    by_num = {t["number"]: t for t in tracks}
    for t in tracks:
        t["tags"] = ttags.get(t.get("uid"), [])
        if t.get("codecDelayNs"):
            rate = t["sampleRate"]
            t["initialPaddingSamples"] = round(t["codecDelayNs"] * rate / 1e9)
    for p in packets:
        t = by_num[p["track"]]
        # the duration the muxer was given (it shows only when it differs from the default / coded one)
        if p["duration"] is None:
            if t["codec"] == "A_OPUS":
                p["duration"] = opus_ms(b, *p["data"])
            elif t.get("defaultDurationNs"):
                p["duration"] = (t["defaultDurationNs"] + 500000) // 1000000
            else:
                p["duration"] = 0
        # timestamps as handed to the muxer: before the track's initial-padding offset
        if t.get("codecDelayNs"):
            off = (t["initialPaddingSamples"] * 1000 * 2 + int(t["sampleRate"])) // (2 * int(t["sampleRate"]))
            p["ts"] -= off
    doc = {"info": info, "globalTags": gtags, "tracks": tracks, "packets": packets, "live": live}
    with open(os.path.join(OUT, name + ".json"), "w", newline="\n") as fh:
        json.dump(doc, fh, indent=0)
    print(name, len(b), "bytes", len(tracks), "tracks", len(packets), "packets")
    return name


def main(only=None):
    """Make every case (only=None) or the named ones; manifest.txt always lists every case, in this order."""
    vp9 = ["-c:v", "libvpx-vp9", "-crf", "40", "-b:v", "0", "-deadline", "realtime", "-cpu-used", "8", "-g", "12", "-threads", "1"]
    opus = ["-c:a", "libopus", "-b:a", "48k"]
    tsrc = lambda s, r, d: ["-f", "lavfi", "-i", f"testsrc2=size={s}:rate={r}:duration={d}"]
    sine = lambda d, rate=48000: ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={rate}:duration={d}"]
    vp9g = lambda g, crf=40: ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-deadline", "realtime", "-cpu-used", "8",
                              "-threads", "1", "-g", str(g)]

    cases = [  # (name, ffmpeg args, live)
        ("vp9_only", tsrc("64x48", 24, 2) + vp9, False),
        ("vp9_opus", tsrc("64x48", 24, 2) + sine(2) + vp9 + opus, False),
        ("vp9_alpha", tsrc("64x48", 12, 1) + ["-vf", "format=yuva420p,geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a='if(gt(X,32),255,64)'"]
         + vp9 + ["-pix_fmt", "yuva420p"], False),
        ("long_opus", tsrc("32x24", 10, 12) + sine(12) + vp9 + opus, False),
        ("opus_only", sine(3) + opus, False),
        ("opus_441", sine(1.3, 44100) + opus, False),
        ("vp8_only", tsrc("48x32", 15, 1) + ["-c:v", "libvpx", "-crf", "30", "-b:v", "200k", "-g", "5", "-threads", "1"], False),
        ("av1_only", tsrc("64x64", 8, 1) + ["-c:v", "libaom-av1", "-crf", "50", "-cpu-used", "8", "-g", "4", "-threads", "1", "-row-mt", "0"], False),
        ("tags_title", tsrc("32x32", 6, 1) + sine(1) + vp9 + opus + ["-metadata", "title=SpawnGen test", "-metadata",
         'comment={"prompt": {"3": {"class_type": "KSampler"}}}', "-metadata:s:v:0", "language=eng",
         "-metadata:s:a:0", "handler_name=ComfyUI audio"], False),
        # two video tracks at the same rate: every frame time is a tie, broken by track order in the interleaver
        ("two_video", tsrc("32x24", 8, 1) + ["-f", "lavfi", "-i", "testsrc=size=32x24:rate=8:duration=1", "-map", "0:v", "-map", "1:v"] + vp9, False),
        ("live_vp9_opus", tsrc("32x24", 10, 3) + sine(3) + vp9 + opus, True),
        # audio only, 6 s: the 5 s cluster time limit and one cue per cluster
        ("opus_6s", sine(6) + opus, False),
        # audio only, pipe: 1 s clusters
        ("opus_live", sine(3) + opus, True),
        # video without key frames after the first: clusters split by time alone (5 s seekable, 1 s pipe)
        ("vp9_longgop", tsrc("64x48", 10, 6) + vp9g(1000), False),
        ("vp9_longgop_live", tsrc("64x48", 10, 3) + vp9g(1000), True),
        # key frames every 3 frames around the 4 KiB key-frame cluster threshold (kept at 4088 bytes, split at 4112)
        ("vp9_keyframe_4k", tsrc("80x64", 10, 6) + vp9g(3, 44), False),
        ("vp9_keyframe_live", tsrc("64x48", 10, 4) + vp9g(3), True),
        # pipe output past the 32 KiB cluster size limit
        ("vp9_size_live", tsrc("192x144", 25, 1) + vp9g(1000, 4), True),
        # track Name, Language, FlagDefault / FlagForced, DisplayWidth / DisplayHeight, chroma siting
        ("meta_tracks", tsrc("64x48", 10, 1) + sine(1) + vp9 + opus + ["-metadata:s:v:0", "title=MyVideo", "-metadata:s:a:0",
         "title=MyAudio", "-metadata:s:a:0", "language=fre", "-disposition:v:0", "default+forced", "-disposition:a:0", "forced",
         "-aspect", "16:9", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
         "-chroma_sample_location", "left"], False),
        # every Colour field
        ("meta_colour", tsrc("64x48", 10, 1) + vp9 + ["-vf", "setparams=color_primaries=bt709:color_trc=smpte2084:colorspace=bt709:range=tv",
         "-color_primaries", "bt709", "-color_trc", "smpte2084", "-chroma_sample_location", "left", "-aspect", "4:3"], False),
        # AlphaMode next to DisplayWidth / DisplayHeight
        ("meta_alpha_display", tsrc("64x48", 10, 1) + ["-vf", "format=yuva420p"] + vp9 + ["-pix_fmt", "yuva420p", "-aspect", "2:1"], False),
    ]
    names = [c[0] for c in cases]
    for name, args, live in cases:
        if only and name not in only:
            continue
        p = os.path.join(OUT, name + ".webm")
        ff(args, p, pipe=live)
        manifest(name, p, live=live)

    with open(os.path.join(OUT, "manifest.txt"), "w", newline="\n") as fh:
        fh.write("\n".join(names) + "\n")


if __name__ == "__main__":
    import sys
    main(set(sys.argv[1:]) or None)
