#!/usr/bin/env python3
"""
Generates the ICCP / EXIF / XMP metadata fixtures into SpawnDev.Codecs.Demo.Shared/TestData/WebP, all named meta2_*:

  meta2_<case>.webp               the file
  meta2_<case>.icc/.exif/.xmp     TRUTH: the payload libwebp's demuxer reports for the chunk (Pillow Image.info), absent when libwebp
                                  reports none
  meta2_manifest.tsv              case, width, height, frame count, orientation we must report, lossless (1) or lossy (0),
                                  has_icc, has_exif, has_xmp (as libwebp reports them), note

Cases cover what a reader meets in the wild: every chunk subset, VP8X flag bits that do not match the chunks present (libwebp
only sees a chunk whose flag is set), chunk order, a duplicate (the first one wins), an unknown chunk in between, odd payload
lengths (RIFF padding), metadata on a lossy file with alpha and on an animation (written by libwebp's own muxer), and EXIF
orientation in both byte orders, with and without the "Exif\\0\\0" prefix, valid 1..8 and several malformed forms.

Orientation truth: for a valid value it is what Pillow's own EXIF parser reads (asserted below); a malformed orientation entry
(value outside 1..8, wrong type or count, truncated IFD, bad TIFF header) is reported as 1 ("upright") by the library.

Run:   <venv>/python -I scripts/images/make_webp_meta_fixtures.py        (needs numpy + pillow)
"""
import io
import os
import struct
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import webp_ref as W  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "SpawnDev.Codecs.Demo.Shared", "TestData", "WebP")
for f in os.listdir(OUT):
    if f.startswith("meta2_"):
        os.remove(os.path.join(OUT, f))

manifest = []

# the image part of the lossless cases: a flat 8x8 RGBA picture compressed by libwebp
FLAT = np.zeros((8, 8, 4), np.uint8)
FLAT[...] = (200, 100, 50, 255)
_b = io.BytesIO()
Image.fromarray(FLAT, "RGBA").save(_b, format="WEBP", lossless=True, exact=True)
VP8L = [p for f, p in W.riff_chunks(_b.getvalue()) if f == b"VP8L"][0]
IMG = W.make_chunk(b"VP8L", VP8L)

ICC = b"\0\0\0\x80ADBE" + bytes(range(120))                      # 128 bytes
ICC_ODD = ICC + b"\x07"                                           # 129 bytes: padded in the RIFF
XMP = b"<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:RDF/></x:xmpmeta>"
XMP_ODD = XMP + b" "
assert len(ICC_ODD) % 2 == 1


def tiff_le(orientation):
    return b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", 1) + struct.pack("<HHII", 0x0112, 3, 1, orientation) + struct.pack("<I", 0)


def tiff_be(orientation):
    return b"MM\0*" + struct.pack(">I", 8) + struct.pack(">H", 1) + struct.pack(">HHI", 0x0112, 3, 1) \
        + struct.pack(">HH", orientation, 0) + struct.pack(">I", 0)


EXIF = tiff_le(6)
EXIF_ODD = EXIF + b"\0"


def vp8x(flags, w=8, h=8):
    return W.make_chunk(b"VP8X", bytes([flags, 0, 0, 0]) + struct.pack("<I", w - 1)[:3] + struct.pack("<I", h - 1)[:3])


def libwebp_view(data):
    im = Image.open(io.BytesIO(data))
    im.load()
    got = {k: im.info.get(k) for k in ("icc_profile", "exif", "xmp")}
    try:
        ori = im.getexif().get(0x0112)
    except Exception:  # noqa: BLE001  (Pillow's own EXIF parser rejects some of the malformed blocks)
        ori = None
    return im, got, ori


def add(case, data, orientation, lossless=True, note="", expect_orientation_from_pillow=False):
    im, got, pillow_ori = libwebp_view(data)
    frames = getattr(im, "n_frames", 1)
    if expect_orientation_from_pillow:
        assert pillow_ori == orientation, "%s: Pillow reads orientation %s, the manifest says %s" % (case, pillow_ori, orientation)
    with open(os.path.join(OUT, "meta2_%s.webp" % case), "wb") as f:
        f.write(data)
    flags = []
    for key, ext in (("icc_profile", "icc"), ("exif", "exif"), ("xmp", "xmp")):
        flags.append(int(got[key] is not None))
        if got[key] is not None:
            with open(os.path.join(OUT, "meta2_%s.%s" % (case, ext)), "wb") as f:
                f.write(got[key])
    manifest.append((case, im.size[0], im.size[1], frames, orientation, int(lossless), flags[0], flags[1], flags[2], note or "-"))
    print("%-34s ori=%d icc=%d exif=%d xmp=%d %s" % (case, orientation, flags[0], flags[1], flags[2], note))


I_, E_, X_ = W.make_chunk(b"ICCP", ICC), W.make_chunk(b"EXIF", EXIF), W.make_chunk(b"XMP ", XMP)
ALL = 0x20 | 0x08 | 0x04 | 0x10

# ---- chunk subsets, flags and order ----
add("canonical_all", W.make_riff([vp8x(ALL), I_, IMG, E_, X_]), 6, expect_orientation_from_pillow=True)
add("icc_only", W.make_riff([vp8x(0x20 | 0x10), I_, IMG]), 1)
add("exif_only", W.make_riff([vp8x(0x08 | 0x10), IMG, E_]), 6, expect_orientation_from_pillow=True)
add("xmp_only", W.make_riff([vp8x(0x04 | 0x10), IMG, X_]), 1)
add("exif_xmp", W.make_riff([vp8x(0x0C | 0x10), IMG, E_, X_]), 6, expect_orientation_from_pillow=True)
add("icc_xmp", W.make_riff([vp8x(0x24 | 0x10), I_, IMG, X_]), 1)
add("flags_zero_chunks_present", W.make_riff([vp8x(0x10), I_, IMG, E_, X_]), 1, note="libwebp ignores a chunk whose VP8X flag is clear")
add("icc_flag_only_all_chunks", W.make_riff([vp8x(0x30), I_, IMG, E_, X_]), 1, note="only the ICCP flag is set")
add("exif_flag_only_all_chunks", W.make_riff([vp8x(0x18), I_, IMG, E_, X_]), 6, note="only the EXIF flag is set")
add("flags_set_chunks_missing", W.make_riff([vp8x(ALL), IMG]), 1)
add("all_after_image", W.make_riff([vp8x(ALL), IMG, I_, E_, X_]), 6)
add("exif_xmp_before_icc", W.make_riff([vp8x(ALL), E_, X_, I_, IMG]), 6)
add("unknown_chunk_between", W.make_riff([vp8x(ALL), W.make_chunk(b"ZZZZ", b"abc"), I_, IMG, W.make_chunk(b"QQQQ", b"xy"), E_, X_]), 6)
add("duplicate_chunks_first_wins", W.make_riff([vp8x(ALL), I_, IMG, E_, X_, W.make_chunk(b"XMP ", b"<second/>"),
                                                W.make_chunk(b"ICCP", ICC[::-1]), W.make_chunk(b"EXIF", tiff_le(3))]), 6,
    note="a second chunk of a kind is ignored")
add("odd_lengths_padded", W.make_riff([vp8x(ALL), W.make_chunk(b"ICCP", ICC_ODD), IMG, W.make_chunk(b"EXIF", EXIF_ODD), W.make_chunk(b"XMP ", XMP_ODD)]), 6,
    note="ICC 129, EXIF 27 and XMP odd: RIFF pad bytes")

# ---- libwebp's own muxer: lossy with alpha, and an animation ----
rng = np.random.default_rng(7)
photo = np.dstack([rng.integers(0, 256, (24, 32, 3)).astype(np.uint8), rng.integers(0, 256, (24, 32)).astype(np.uint8)])
b = io.BytesIO()
Image.fromarray(photo, "RGBA").save(b, format="WEBP", lossless=False, quality=60, icc_profile=ICC, exif=b"Exif\0\0" + tiff_le(8), xmp=XMP)
add("lossy_alpha_all_libwebp", b.getvalue(), 8, lossless=False, note="VP8X + ICCP + ALPH + VP8 + EXIF + XMP", expect_orientation_from_pillow=True)
frames = [Image.fromarray(np.dstack([rng.integers(0, 256, (16, 16, 3)).astype(np.uint8), np.full((16, 16), 255, np.uint8)]), "RGBA") for _ in range(3)]
b = io.BytesIO()
frames[0].save(b, format="WEBP", save_all=True, append_images=frames[1:], duration=40, loop=5, lossless=True,
               icc_profile=ICC, exif=b"Exif\0\0" + tiff_be(2), xmp=XMP)
add("anim_all_libwebp", b.getvalue(), 2, note="animation written by libwebp's muxer", expect_orientation_from_pillow=True)

# ---- EXIF orientation ----
for o in range(1, 9):
    add("orientation_%d_le" % o, W.make_riff([vp8x(0x08 | 0x10), IMG, W.make_chunk(b"EXIF", tiff_le(o))]), o, expect_orientation_from_pillow=True)
for o in (2, 5, 8):
    add("orientation_%d_be_prefix" % o, W.make_riff([vp8x(0x08 | 0x10), IMG, W.make_chunk(b"EXIF", b"Exif\0\0" + tiff_be(o))]), o,
        expect_orientation_from_pillow=True)
add("orientation_7_le_prefix", W.make_riff([vp8x(0x08 | 0x10), IMG, W.make_chunk(b"EXIF", b"Exif\0\0" + tiff_le(7))]), 7, expect_orientation_from_pillow=True)

# an IFD with several entries: orientation is not the first
multi = b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", 3) \
    + struct.pack("<HHII", 0x010F, 2, 4, 0x41424300) + struct.pack("<HHII", 0x0112, 3, 1, 4) + struct.pack("<HHII", 0x0131, 3, 1, 9) + struct.pack("<I", 0)
add("orientation_in_multi_entry_ifd", W.make_riff([vp8x(0x08 | 0x10), IMG, W.make_chunk(b"EXIF", multi)]), 4, expect_orientation_from_pillow=True)

# malformed: reported as 1
bad = {
    "value_0": tiff_le(0),
    "value_9": tiff_le(9),
    "value_65535": tiff_le(65535),
    "type_long": b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", 1) + struct.pack("<HHII", 0x0112, 4, 1, 6) + struct.pack("<I", 0),
    "count_2": b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", 1) + struct.pack("<HHII", 0x0112, 3, 2, 6) + struct.pack("<I", 0),
    "bad_magic": b"II+\0" + tiff_le(6)[4:],
    "bad_byte_order": b"XX*\0" + tiff_le(6)[4:],
    "truncated_ifd": tiff_le(6)[:14],
    "ifd_offset_past_end": b"II*\0" + struct.pack("<I", 4000) + b"\0" * 8,
    "no_orientation_tag": b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", 1) + struct.pack("<HHII", 0x010F, 3, 1, 1) + struct.pack("<I", 0),
}
for label, ex in bad.items():
    data = W.make_riff([vp8x(0x08 | 0x10), IMG, W.make_chunk(b"EXIF", ex)])
    add("orientation_malformed_" + label, data, 1, note="malformed orientation entry")

with open(os.path.join(OUT, "meta2_manifest.tsv"), "w") as fh:
    for row in manifest:
        fh.write("\t".join(str(v) for v in row) + "\n")
print("fixtures:", len(manifest), " bytes:", sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.startswith("meta2_")))
