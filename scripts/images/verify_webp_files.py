#!/usr/bin/env python3
"""
Verifies WebP files written by the library's encoder, using libwebp through Pillow as the independent oracle.

For every <dir>/<name>.webp:
  * Pillow must open it and report the size in <name>.size ("W H"),
  * its decode (RGBA) must equal <name>.rgba8 EXACTLY (the expected pixels the test wrote),
  * metadata sidecars <name>.icc / .exif / .xmp, when present, must equal what libwebp's demuxer reports,
  * the size is compared with libwebp's own lossless encode of the same pixels (method 4, quality 75) and reported
    as a ratio (only a catastrophic ratio fails: this is a sanity check, not a tight bound).

Other sidecars switch the check: <name>.alpha8 (an ALPH test: the lossy file's alpha plane must equal it in libwebp AND in the
pure-Python ALPH decoder), <name>.anim + <name>.canvases (an animation: every frame canvas, duration, loop count and background
colour as libwebp's animation decoder shows them). Metadata sidecars are checked against libwebp's demuxer (Pillow info) and the
VP8X flag bits.

Prints one line per file ("OK name ours libwebp ratio" / "FAIL name reason") and exits 1 if any file failed.

Run:   <venv>/python -I scripts/images/verify_webp_files.py <dir>
"""
import io
import os
import struct
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import webp_ref as W  # noqa: E402  (independent pure-Python VP8L decoder, used for the feature report)


def chunks(data):
    pos = 12
    while pos + 8 <= len(data):
        fcc = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        yield fcc, data[pos + 8:pos + 8 + size]
        pos += 8 + size + (size & 1)


def check_metadata(path, data, im):
    """
    Metadata sidecars <name>.icc / .exif / .xmp, when present, must equal what libwebp's demuxer reports (Pillow info), and the
    RIFF must hold exactly one such chunk with exactly that payload; the VP8X flag bits must say so. Without a sidecar the
    file must carry no such chunk (a stray chunk would be wrong too).
    """
    want_flags = 0
    for ext, fcc, key, bit in (("icc", b"ICCP", "icc_profile", 0x20), ("exif", b"EXIF", "exif", 0x08), ("xmp", b"XMP ", "xmp", 0x04)):
        side = path + "." + ext
        found = [p for f, p in chunks(data) if f == fcc]
        if os.path.exists(side):
            expected = open(side, "rb").read()
            if len(found) != 1 or found[0] != expected:
                raise Exception("%s chunk missing, duplicated or different" % ext)
            if im.info.get(key) != expected:
                raise Exception("libwebp (demux) reports a different %s payload" % ext)
            want_flags |= bit
        elif found:
            raise Exception("unexpected %s chunk" % ext)
    vp8x = [p for f, p in chunks(data) if f == b"VP8X"]
    if vp8x and (vp8x[0][0] & 0x2C) != want_flags:
        raise Exception("VP8X flags 0x%02x do not match the metadata present (0x%02x)" % (vp8x[0][0] & 0x2C, want_flags))
    if not vp8x and want_flags:
        raise Exception("metadata present but no VP8X chunk")


def main(d):
    names = sorted(f[:-5] for f in os.listdir(d) if f.endswith(".webp"))
    bad = 0
    for name in names:
        path = os.path.join(d, name)
        try:
            data = open(path + ".webp", "rb").read()
            w, h = (int(v) for v in open(path + ".size").read().split())
            im = Image.open(io.BytesIO(data))
            if im.size != (w, h):
                raise Exception("size %s != %dx%d" % (im.size, w, h))
            if os.path.exists(path + ".anim"):
                # an animation our encoder wrote. <name>.anim: "loop L", "bg R G B A", "dur d0 d1 ..."; <name>.canvases: the expected
                # canvas of every frame, RGBA8, concatenated. libwebp's animation decoder (through Pillow) must show exactly those.
                spec = {}
                for line in open(path + ".anim").read().splitlines():
                    key, _, rest = line.partition(" ")
                    spec[key] = [int(v) for v in rest.split()]
                durs = spec["dur"]
                n = len(durs)
                if getattr(im, "n_frames", 1) != n:
                    raise Exception("libwebp sees %d frames, expected %d" % (getattr(im, "n_frames", 1), n))
                want = np.frombuffer(open(path + ".canvases", "rb").read(), np.uint8).reshape(n, h, w, 4)
                for k in range(n):
                    im.seek(k)
                    frame = np.array(im.convert("RGBA"))
                    if not (frame == want[k]).all():
                        ys, xs = np.nonzero((frame != want[k]).any(axis=2))
                        raise Exception("frame %d canvas mismatch at x=%d y=%d: got %s expected %s (%d pixels differ)" % (
                            k, xs[0], ys[0], frame[ys[0], xs[0]], want[k][ys[0], xs[0]], len(xs)))
                    if im.info.get("duration") != durs[k]:
                        raise Exception("frame %d duration %s != %d" % (k, im.info.get("duration"), durs[k]))
                if im.info.get("loop") != spec["loop"][0]:
                    raise Exception("loop count %s != %d" % (im.info.get("loop"), spec["loop"][0]))
                if tuple(im.info.get("background", ())) != tuple(spec["bg"]):
                    raise Exception("background %s != %s" % (im.info.get("background"), tuple(spec["bg"])))
                check_metadata(path, data, im)
                st_total = 0
                for fcc, p in chunks(data):
                    if fcc == b"ANMF":
                        sub = [pp for f2, pp in W.riff_chunks(b"RIFF\0\0\0\0WEBP" + p[16:]) if f2 == b"VP8L"][0]
                        W.decode_vp8l(sub)   # an independent bitstream parser must accept every frame's stream too
                        st_total += 1
                print("OK %s %d 0 0.000" % (name, len(data)))
                continue
            got = np.array(im.convert("RGBA"))
            if os.path.exists(path + ".alpha8"):
                # an ALPH test: <name>.webp is a lossy VP8X file whose ALPH chunk our encoder wrote. libwebp's alpha plane must
                # equal the truth plane, and so must an independent pure-Python decode of the ALPH chunk (webp_ref).
                want_a = np.frombuffer(open(path + ".alpha8", "rb").read(), np.uint8).reshape(h, w)
                if not (got[..., 3] == want_a).all():
                    ys, xs = np.nonzero(got[..., 3] != want_a)
                    raise Exception("libwebp alpha mismatch at x=%d y=%d: got %d expected %d (%d samples differ)" % (
                        xs[0], ys[0], got[ys[0], xs[0], 3], want_a[ys[0], xs[0]], len(xs)))
                alph = [p for f, p in chunks(data) if f == b"ALPH"][0]
                plane, st, hd = W.alpha_decode(alph, w, h)
                if not (np.array(plane, np.uint8) == want_a).all():
                    raise Exception("webp_ref ALPH decode differs from the truth")
                print("STATS %s method=%d filter=%d pre=%d %s" % (name, hd["method"], hd["filter"], hd["pre"], st.summary() if st else "raw"))
                print("OK %s %d 0 0.000" % (name, len(data)))
                continue
            want = np.frombuffer(open(path + ".rgba8", "rb").read(), np.uint8).reshape(h, w, 4)
            if not (got == want).all():
                ys, xs = np.nonzero((got != want).any(axis=2))
                raise Exception("pixel mismatch at x=%d y=%d: got %s expected %s (%d pixels differ)" % (
                    xs[0], ys[0], got[ys[0], xs[0]], want[ys[0], xs[0]], len(xs)))
            check_metadata(path, data, im)
            if os.path.exists(path + ".stats"):
                # second, independent decoder + which bitstream features the encoder actually used
                payload = [p for f, p in chunks(data) if f == b"VP8L"][0]
                w2, h2, pix, st = W.decode_vp8l(payload)
                ref_rgba = np.array([[(q >> 16) & 255, (q >> 8) & 255, q & 255, q >> 24] for q in pix], np.uint8).reshape(h, w, 4)
                if not (ref_rgba == want).all():
                    raise Exception("webp_ref decode differs from the source")
                print("STATS %s %s" % (name, st.summary()))
            # the size comparison is about the image: leave the metadata chunks (header + payload + pad) out
            ours = len(data) - sum(8 + len(p) + (len(p) & 1) for f, p in chunks(data) if f in (b"ICCP", b"EXIF", b"XMP "))
            if max(w, h) > 16383:
                # libwebp's ENCODER refuses 16384 (its limit is 16383) although its decoder accepts the format maximum
                print("OK %s %d 0 0.000" % (name, ours))
                continue
            ref = io.BytesIO()
            Image.fromarray(want, "RGBA").save(ref, format="WEBP", lossless=True, method=4, quality=75, exact=True)
            theirs = len(ref.getvalue())
            ratio = ours / theirs
            if ratio > 12 and ours > theirs + 512:
                raise Exception("catastrophic size %d vs libwebp %d" % (ours, theirs))
            print("OK %s %d %d %.3f" % (name, ours, theirs, ratio))
        except Exception as e:  # noqa: BLE001
            bad += 1
            print("FAIL %s %s" % (name, e))
    print("SUMMARY %d files, %d failed" % (len(names), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
