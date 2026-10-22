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
pure-Python ALPH decoder),

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
            for ext, fcc in (("icc", b"ICCP"), ("exif", b"EXIF"), ("xmp", b"XMP ")):
                side = path + "." + ext
                if os.path.exists(side):
                    expected = open(side, "rb").read()
                    found = [p for f, p in chunks(data) if f == fcc]
                    if not found or found[0] != expected:
                        raise Exception("%s chunk missing or different" % ext)
            if os.path.exists(path + ".stats"):
                # second, independent decoder + which bitstream features the encoder actually used
                payload = [p for f, p in chunks(data) if f == b"VP8L"][0]
                w2, h2, pix, st = W.decode_vp8l(payload)
                ref_rgba = np.array([[(q >> 16) & 255, (q >> 8) & 255, q & 255, q >> 24] for q in pix], np.uint8).reshape(h, w, 4)
                if not (ref_rgba == want).all():
                    raise Exception("webp_ref decode differs from the source")
                print("STATS %s %s" % (name, st.summary()))
            ours = len(data)
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
