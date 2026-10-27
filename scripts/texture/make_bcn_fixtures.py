"""BCn decoder fixtures (SpawnDev.Codecs texture track).

usage: python make_bcn_fixtures.py <SpawnDev.Codecs.Demo.Shared/TestData>

Writes TestData/Bcn/: for every case a raw block file <name>.<fmt>, the RGBA8 the reference decoder produces for it
<name>.<fmt>.rgba.gz, and manifest.tsv (name, format, width, height, source). Blocks come from the reference encoders
(etcpak: etcpak's BC1/BC3/BC4/BC5 encoders, bc7enc for BC7, every BC7 mode on its own through the mode mask) over a real
picture and a synthetic alpha picture, plus random blocks (every BC7 mode, partition, rotation and the reserved mode).
The expected pixels are texture2ddecoder's, and the script asserts Pillow's BCn decoder (a DDS) gives the same, except
BC3: there the D3D spec reads the colour block in 4-colour mode whatever c0 vs c1 (DirectXTex does), Pillow does that and
texture2ddecoder applies BC1's 3-colour rule to blocks with c0 <= c1, so BC3's expected pixels are Pillow's and
texture2ddecoder must agree on every other block; and BC7's reserved mode (no mode bit set) decodes to (0, 0, 0, 0) by
the spec (DirectXTex, texture2ddecoder) where Pillow gives alpha 255, so those blocks are checked on RGB only.
Output convention (D3D): BC1 / BC3 / BC7 RGBA; BC4 (r, 0, 0, 255); BC5 (r, g, 0, 255). BC1 with c0 <= c1 decodes index
3 as transparent black (0, 0, 0, 0) by the spec (DirectXTex, Pillow); texture2ddecoder makes it opaque, so BC1's alpha is
Pillow's (its colour must agree with texture2ddecoder's).
"""
import gzip, io, os, struct, sys
import numpy as np
import etcpak, texture2ddecoder as t2d
from PIL import Image

DXGI = {"bc1": 71, "bc3": 77, "bc4": 80, "bc5": 83, "bc7": 98}
BLOCK_BYTES = {"bc1": 8, "bc3": 16, "bc4": 8, "bc5": 16, "bc7": 16}
T2D = {"bc1": t2d.decode_bc1, "bc3": t2d.decode_bc3, "bc4": t2d.decode_bc4, "bc5": t2d.decode_bc5, "bc7": t2d.decode_bc7}


def blocks_of(w, h, fmt):
    return ((w + 3) // 4) * ((h + 3) // 4) * BLOCK_BYTES[fmt]


def dds(blocks, fmt, w, h):
    hdr = struct.pack("<4sIIIIIII44sIIIIIIIIIIIII", b"DDS ", 124, 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000, h, w,
                      len(blocks), 0, 1, b"\0" * 44, 32, 0x4, struct.unpack("<I", b"DX10")[0], 0, 0, 0, 0, 0,
                      0x1000, 0, 0, 0, 0)
    return hdr + struct.pack("<IIIII", DXGI[fmt], 3, 0, 1, 0) + blocks


def reference_rgba(blocks, fmt, w, h):
    """texture2ddecoder's pixels in the D3D convention, checked against Pillow's decoder."""
    bgra = np.frombuffer(T2D[fmt](blocks, w, h), np.uint8).reshape(h, w, 4)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 3] = 255
    if fmt in ("bc1", "bc3", "bc7"):
        out[...] = bgra[..., [2, 1, 0, 3]]
    elif fmt == "bc4":
        out[..., 0] = bgra[..., 2]
    else:
        out[..., 0] = bgra[..., 2]
        out[..., 1] = bgra[..., 1]
    pil = np.asarray(Image.open(io.BytesIO(dds(blocks, fmt, w, h))).convert("RGBA"))
    if fmt == "bc3":
        # 4-colour blocks (c0 > c1) and alpha must agree; the c0 <= c1 colour is the spec's (Pillow's).
        bw = (w + 3) // 4
        for b in range(len(blocks) // 16):
            bx, by = b % bw, b // bw
            c0 = blocks[b * 16 + 8] | blocks[b * 16 + 9] << 8
            c1 = blocks[b * 16 + 10] | blocks[b * 16 + 11] << 8
            ys, xs = slice(by * 4, by * 4 + 4), slice(bx * 4, bx * 4 + 4)
            assert np.array_equal(pil[ys, xs, 3], out[ys, xs, 3]), "bc3: alpha disagrees"
            if c0 > c1:
                assert np.array_equal(pil[ys, xs], out[ys, xs]), "bc3: a 4-colour block disagrees"
        out = pil.copy()
    elif fmt == "bc7":
        bw = (w + 3) // 4
        for b in range(len(blocks) // 16):
            bx, by = b % bw, b // bw
            ys, xs = slice(by * 4, by * 4 + 4), slice(bx * 4, bx * 4 + 4)
            ch = slice(0, 3) if blocks[b * 16] == 0 else slice(0, 4)
            assert np.array_equal(pil[ys, xs, ch], out[ys, xs, ch]), "bc7: Pillow and texture2ddecoder disagree"
    elif fmt == "bc1":
        # Colours must agree; the alpha is the spec's (Pillow's: index 3 of a 3-colour block is transparent).
        assert np.array_equal(pil[..., :3], out[..., :3]), "bc1: Pillow and texture2ddecoder disagree (colour)"
        out[..., 3] = pil[..., 3]
    elif fmt == "bc4":
        assert np.array_equal(pil[..., 0], out[..., 0]), "bc4: Pillow and texture2ddecoder disagree"
    else:
        assert np.array_equal(pil[..., :2], out[..., :2]), "bc5: Pillow and texture2ddecoder disagree"
    return out


def encode(rgba, fmt, w, h, mode_mask=None):
    raw = np.ascontiguousarray(rgba).tobytes()
    if fmt == "bc1":
        return etcpak.compress_bc1(raw, w, h)
    if fmt == "bc3":
        return etcpak.compress_bc3(raw, w, h)
    if fmt == "bc4":
        return etcpak.compress_bc4(raw, w, h)
    if fmt == "bc5":
        return etcpak.compress_bc5(raw, w, h)
    params = etcpak.BC7CompressBlockParams()
    if mode_mask is not None:
        params.m_mode_mask = mode_mask
    return etcpak.compress_bc7(raw, w, h, params)


def main():
    root = os.path.join(sys.argv[1], "Bcn")
    os.makedirs(root, exist_ok=True)
    rows = []

    def save(name, fmt, w, h, blocks, source):
        print(f"{name}.{fmt} {w}x{h}", flush=True)
        assert len(blocks) == blocks_of(w, h, fmt), (name, fmt, len(blocks))
        rgba = reference_rgba(blocks, fmt, w, h)
        with open(os.path.join(root, f"{name}.{fmt}"), "wb") as f:
            f.write(blocks)
        with gzip.GzipFile(os.path.join(root, f"{name}.{fmt}.rgba.gz"), "wb", mtime=0) as f:
            f.write(rgba.tobytes())
        rows.append(f"{name}\t{fmt}\t{w}\t{h}\t{source}")

    # A real picture (multiple of 4 in both directions: the encoders want whole blocks).
    bbb = np.asarray(Image.open(os.path.join(sys.argv[1], "bbb_first_frame_visual.png")).convert("RGBA"))
    bh, bw = (bbb.shape[0] // 4) * 4, (bbb.shape[1] // 4) * 4
    bbb = np.ascontiguousarray(bbb[:bh, :bw])
    for fmt in ("bc1", "bc3", "bc4", "bc5", "bc7"):
        save("bbb", fmt, bw, bh, encode(bbb, fmt, bw, bh), "bbb_first_frame_visual.png, reference encoder")
    for mode in range(8):
        save(f"bbb_mode{mode}", "bc7", bw, bh, encode(bbb, "bc7", bw, bh, 1 << mode), f"bbb, bc7enc mode mask 1 << {mode}")

    # A synthetic picture with alpha: gradients, a hard mask, noisy alpha.
    rng = np.random.default_rng(0xBC7)
    w, h = 64, 48
    yy, xx = np.mgrid[0:h, 0:w]
    syn = np.zeros((h, w, 4), np.uint8)
    syn[..., 0] = (xx * 4) & 255
    syn[..., 1] = (yy * 5) & 255
    syn[..., 2] = rng.integers(0, 256, (h, w))
    syn[..., 3] = np.where((xx // 8 + yy // 8) % 3 == 0, rng.integers(0, 256, (h, w)), np.where(xx < 16, 0, 255))
    for fmt in ("bc1", "bc3", "bc4", "bc5", "bc7"):
        save("alpha", fmt, w, h, encode(syn, fmt, w, h), "synthetic RGBA, reference encoder")
    # (Mode 5 alone on this picture crashes the reference encoder (etcpak 0.9.15's bc7enc, a native fault): mode 5 is
    # covered by bbb_mode5 and the every_mode blocks. The mode-mask encodes are also not the same from run to run, so a
    # regeneration rewrites them; every bit pattern is a valid BC7 block and the expected pixels are the reference
    # decoders' for exactly the blocks stored, so the fixtures stay consistent.)
    for mode in (4, 6, 7):
        save(f"alpha_mode{mode}", "bc7", w, h, encode(syn, "bc7", w, h, 1 << mode), f"synthetic RGBA, bc7enc mode mask 1 << {mode}")

    # Random blocks: every bit pattern is a valid block (BC7: all modes, partitions, rotations, the reserved mode).
    for fmt in ("bc1", "bc3", "bc4", "bc5", "bc7"):
        rw, rh = 64, 64
        save("random", fmt, rw, rh, rng.integers(0, 256, blocks_of(rw, rh, fmt), dtype=np.uint8).tobytes(), "random blocks")
    # BC7: every mode byte explicitly (mode m = lowest set bit m of byte 0), random rest.
    rnd = bytearray(rng.integers(0, 256, blocks_of(64, 64, "bc7"), dtype=np.uint8).tobytes())
    for b in range(len(rnd) // 16):
        m = b % 9   # 8 = the reserved all-zero mode bits
        rnd[b * 16] = 0 if m == 8 else ((rnd[b * 16] & ~((2 << m) - 1)) | (1 << m)) & 0xFF
    save("every_mode", "bc7", 64, 64, bytes(rnd), "random blocks, modes 0-7 and reserved in turn")
    # Partial edge blocks: a picture whose size is not a multiple of 4 (the decoder writes only width x height).
    save("edge", "bc7", 37, 21, rng.integers(0, 256, blocks_of(37, 21, "bc7"), dtype=np.uint8).tobytes(), "random blocks, 37x21")
    save("edge", "bc1", 37, 21, rng.integers(0, 256, blocks_of(37, 21, "bc1"), dtype=np.uint8).tobytes(), "random blocks, 37x21")

    with open(os.path.join(root, "manifest.tsv"), "w", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    print(f"{len(rows)} cases in {root}")


if __name__ == "__main__":
    main()
