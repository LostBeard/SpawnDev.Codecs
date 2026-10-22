#!/usr/bin/env python3
"""
Independent oracle for the encoder tests: opens an image with Pillow and writes what Pillow sees.

Usage:  <venv>/python -I scripts/images/pillow_decode.py <image file> <output dir>

Writes <output dir>/<file name>.json  {format, width, height, mode, frames, durations, loop, background}
and    <output dir>/<file name>.f<k>.rgba8   the RGBA canvas Pillow shows for frame k (rows top to bottom).
Exit code 0 when Pillow opened and loaded every frame, 1 otherwise (the reason is printed).
"""
import json
import os
import sys

import numpy as np
from PIL import Image


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, os.path.basename(src))
    try:
        im = Image.open(src)
        frames = getattr(im, "n_frames", 1)
        info = {"format": im.format, "width": im.size[0], "height": im.size[1], "mode": im.mode,
                "frames": frames, "durations": [], "loop": im.info.get("loop"),
                "background": im.info.get("background")}
        for k in range(frames):
            im.seek(k)
            rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
            with open(f"{base}.f{k}.rgba8", "wb") as f:
                f.write(rgba.tobytes())
            info["durations"].append(im.info.get("duration"))
        with open(base + ".json", "w") as f:
            json.dump(info, f)
        return 0
    except Exception as ex:
        print(f"PILLOW FAILED: {type(ex).__name__}: {ex}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
