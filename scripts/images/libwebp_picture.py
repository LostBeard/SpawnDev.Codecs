"""libwebp's own RGBA -> YUVA import (WebPPictureImportRGBA with use_argb = 0 = ImportYUVAFromRGBA), read through cffi.

The webp package's cdef hides y/u/v/a, so they are read at their offsets in struct WebPPicture (src/webp/encode.h,
64-bit): use_argb 0, colorspace 4, width 8, height 12, y 16, u 24, v 32, y_stride 40, uv_stride 44, a 48, a_stride 56,
pad1[2] 60, argb 72, argb_stride 80, pad2[3] 84, writer 96. The offsets of width and writer are checked against cffi.
"""
import struct
import numpy as np
from webp import ffi, lib

assert ffi.offsetof("WebPPicture", "width") == 8, ffi.offsetof("WebPPicture", "width")
assert ffi.offsetof("WebPPicture", "writer") == 96, ffi.offsetof("WebPPicture", "writer")


def import_rgba(rgba):
    """rgba: (h, w, 4) uint8. Returns (Y, U, V, A or None) as numpy arrays, exactly libwebp's planes."""
    h, w = rgba.shape[:2]
    rgba = np.ascontiguousarray(rgba)
    pic = ffi.new("WebPPicture*")
    assert lib.WebPPictureInit(pic)
    pic.width, pic.height, pic.use_argb = w, h, 0
    buf = ffi.from_buffer(rgba)
    assert lib.WebPPictureImportRGBA(pic, ffi.cast("uint8_t*", buf), w * 4)
    raw = bytes(ffi.buffer(ffi.cast("uint8_t*", pic), 96))
    yp, up, vp = struct.unpack_from("<QQQ", raw, 16)
    ys, uvs = struct.unpack_from("<ii", raw, 40)
    ap, = struct.unpack_from("<Q", raw, 48)
    as_, = struct.unpack_from("<i", raw, 56)
    cw, ch = (w + 1) // 2, (h + 1) // 2

    def plane(ptr, stride, pw, ph):
        b = bytes(ffi.buffer(ffi.cast("uint8_t*", ptr), stride * (ph - 1) + pw))
        return np.array([np.frombuffer(b, np.uint8, pw, r * stride) for r in range(ph)])

    Y = plane(yp, ys, w, h)
    U = plane(up, uvs, cw, ch)
    V = plane(vp, uvs, cw, ch)
    A = plane(ap, as_, w, h) if ap else None
    lib.WebPPictureFree(pic)
    return Y, U, V, A
