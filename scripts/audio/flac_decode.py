"""Test oracle: decode a .flac with libFLAC (through libsndfile / soundfile) - not our code.

Usage: python -I flac_decode.py <input.flac> <output.raw>
Writes the decoded samples as interleaved int32 little-endian in the stream's own bit depth (16-bit values for a 16-bit
file, 24-bit values for 24-bit), and prints JSON: {"channels", "rate", "frames", "subtype"}.
libFLAC verifies every frame's CRCs while decoding; a corrupt stream fails here.
"""
import json
import sys

import numpy as np
import soundfile as sf

src, dst = sys.argv[1], sys.argv[2]
info = sf.info(src)
data, rate = sf.read(src, dtype="int32", always_2d=True)    # libsndfile scales to the full int32 range
shift = {"PCM_16": 16, "PCM_24": 8, "PCM_S8": 24, "PCM_U8": 24}.get(info.subtype, 0)
(data >> shift).astype("<i4").tofile(dst)
print(json.dumps({"channels": info.channels, "rate": rate, "frames": int(data.shape[0]), "subtype": info.subtype}))
