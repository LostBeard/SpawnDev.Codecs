"""BCn ENCODER fixtures (SpawnDev.Codecs texture track): etcpak's blocks for our encoder to match byte for byte.

usage: python make_bcn_encoder_fixtures.py <SpawnDev.Codecs.Demo.Shared/TestData>

Writes TestData/BcnEnc/: <name>.rgba.gz (the input picture, RGBA8), <name>.<fmt> (etcpak 0.9.15's blocks for it) and
manifest.tsv (name, format, width, height). The etcpak 0.9.15 Windows wheel runs etcpak's portable (non-SIMD) paths
(its BC4 matches that arithmetic block for block, not the SSE4.1 one), whose BC1 ignores alpha: "alpha" is also coded as
BC1. Widths include a multiple of 8 and one that is not (SIMD builds take different paths for those).
BC7 is bc7enc at its default settings (compress_bc7 without params). The wheel's SIMD modules are tagged cp37 and do
not load on newer Pythons, so etcpak runs its portable module (_etcpak_none); the script checks that.
"""
import gzip, os, sys
import numpy as np
import etcpak
from PIL import Image

COMPRESS = {"bc1": etcpak.compress_bc1, "bc3": etcpak.compress_bc3, "bc4": etcpak.compress_bc4, "bc5": etcpak.compress_bc5,
            "bc7": etcpak.compress_bc7}


def main():
    assert "etcpak._etcpak_none" in sys.modules, "etcpak loaded a SIMD module; the fixtures are made with the portable one"
    root = os.path.join(sys.argv[1], "BcnEnc")
    os.makedirs(root, exist_ok=True)
    rows = []

    def save(name, rgba, formats):
        h, w = rgba.shape[:2]
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        with gzip.GzipFile(os.path.join(root, f"{name}.rgba.gz"), "wb", mtime=0) as f:
            f.write(rgba.tobytes())
        for fmt in formats:
            blocks = COMPRESS[fmt](rgba.tobytes(), w, h)
            with open(os.path.join(root, f"{name}.{fmt}"), "wb") as f:
                f.write(blocks)
            rows.append(f"{name}\t{fmt}\t{w}\t{h}")
        print(name, w, h, formats, flush=True)

    bbb = np.asarray(Image.open(os.path.join(sys.argv[1], "bbb_first_frame_visual.png")).convert("RGBA"))
    bh, bw = (bbb.shape[0] // 4) * 4, (bbb.shape[1] // 8) * 8
    bbb = bbb[:bh, :bw].copy()
    bbb[..., 3] = 255
    save("bbb", bbb, ["bc1", "bc3", "bc4", "bc5", "bc7"])
    save("bbb_w4", bbb[:, :bw - 4], ["bc1", "bc3", "bc7"])

    rng = np.random.default_rng(0xBC1)
    w, h = 64, 48
    yy, xx = np.mgrid[0:h, 0:w]
    syn = np.zeros((h, w, 4), np.uint8)
    syn[..., 0] = (xx * 4) & 255
    syn[..., 1] = (yy * 5) & 255
    syn[..., 2] = rng.integers(0, 256, (h, w))
    syn[..., 3] = np.where((xx // 8 + yy // 8) % 3 == 0, rng.integers(0, 256, (h, w)), np.where(xx < 16, 0, 255))
    save("alpha", syn, ["bc1", "bc3", "bc4", "bc5", "bc7"])
    opaque = syn.copy()
    opaque[..., 3] = 255
    save("gradient", opaque, ["bc1", "bc7"])

    noise = rng.integers(0, 256, (64, 64, 4), dtype=np.uint8)
    save("noise", noise, ["bc3", "bc4", "bc5", "bc7"])
    noise[..., 3] = 255
    save("noise_opaque", noise, ["bc1", "bc7"])

    # Flat and nearly flat blocks: solid colours, 5:6:5-equal neighbours (solid), one-step ramps, a single odd pixel.
    flat = np.zeros((32, 64, 4), np.uint8)
    flat[..., 3] = 255
    for by in range(8):
        for bx in range(16):
            base = rng.integers(0, 256, 4)
            blk = np.broadcast_to(base, (4, 4, 4)).copy()
            kind = (by * 16 + bx) % 4
            if kind == 1:
                blk[..., :3] = (blk[..., :3] & np.array([0xF8, 0xFC, 0xF8])) | rng.integers(0, 4, (4, 4, 3)) & np.array([7, 3, 7])
            elif kind == 2:
                blk[..., 0] = np.clip(int(base[0]) + np.arange(16).reshape(4, 4) // 4, 0, 255)
            elif kind == 3:
                blk[1, 2, :3] = 255 - blk[1, 2, :3]
            blk[..., 3] = 255 if by < 4 else base[3]
            flat[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4] = blk
    save("flat", flat, ["bc3", "bc4", "bc5", "bc7"])
    flat_opaque = flat.copy()
    flat_opaque[..., 3] = 255
    save("flat_opaque", flat_opaque, ["bc1", "bc7"])

    # BC7 only: grey ramps (mode 1's degenerate-endpoint fix), two-colour splits along partitions, alpha ramps with
    # equal colour (mode 5's alpha search) and alpha steps (mode 7).
    ramps = np.zeros((32, 64, 4), np.uint8)
    for by in range(8):
        for bx in range(16):
            kind = (by * 16 + bx) % 4
            yy4, xx4 = np.mgrid[0:4, 0:4]
            base = rng.integers(0, 200, 4)
            blk = np.zeros((4, 4, 4), np.int32)
            if kind == 0:
                v = base[0] + (xx4 + 4 * yy4) * int(rng.integers(1, 4))
                blk[..., 0] = v; blk[..., 1] = v; blk[..., 2] = v; blk[..., 3] = 255
            elif kind == 1:
                c0, c1 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
                m = (xx4 + yy4 * int(rng.integers(0, 3))) % 4 < 2
                blk[..., :3] = np.where(m[..., None], c0, c1); blk[..., 3] = 255
            elif kind == 2:
                blk[..., :3] = base[:3]; blk[..., 3] = np.clip(base[3] + (xx4 + 4 * yy4) * 3, 0, 255)
            else:
                blk[..., :3] = base[:3] + xx4[..., None] * 5; blk[..., 3] = np.where(yy4 < 2, 255, int(rng.integers(0, 200)))
            ramps[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4] = np.clip(blk, 0, 255)
    save("ramps", ramps, ["bc7"])

    with open(os.path.join(root, "manifest.tsv"), "w", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    print(f"{len(rows)} cases in {root}")


if __name__ == "__main__":
    main()
