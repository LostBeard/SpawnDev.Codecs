#!/usr/bin/env python3
"""
Opens every PNG the GPU encoder tests dumped (CODECS_DUMP_DIR=<dir> while running the desktop
PngEncoderGpu_* tests) with Pillow / libpng-equivalent decoding and checks the pixels equal the source
the encoder was given. Colour-type reduction (RGBA -> RGB / gray) is invisible to this check on purpose:
only the decoded RGBA must match.

Run:  <venv>/python -I scripts/images/verify_png_encoder.py <dump dir>
"""
import glob
import os
import sys

import numpy as np
from PIL import Image


def main():
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(d, "png_*.png")))
    if not files:
        print("no dumped PNGs found in", d)
        return 2
    bad = 0
    for f in files:
        stem = f[:-4]
        name = os.path.basename(stem)
        sixteen = name.endswith("rgba16")
        src = open(stem + ".src", "rb").read()
        im = Image.open(f)
        im.load()
        if sixteen:
            # Pillow reads 16-bit RGBA as 8-bit (high byte): compare against the high bytes of the source.
            expect = np.frombuffer(src, dtype="<u2").reshape(im.height, im.width, 4) >> 8
            got = np.asarray(im.convert("RGBA")).astype(np.int64)
            ok = np.array_equal(got, expect.astype(np.int64))
        else:
            expect = np.frombuffer(src, dtype=np.uint8).reshape(im.height, im.width, 4)
            got = np.asarray(im.convert("RGBA"))
            ok = np.array_equal(got, expect)
        print(("OK   " if ok else "FAIL ") + name + f"  mode={im.mode} size={im.size} info_keys={sorted(k for k in im.info if isinstance(k, str))}")
        if not ok:
            bad += 1
    print(f"--- {len(files) - bad} ok, {bad} failed of {len(files)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
