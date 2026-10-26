# Truth fixtures for Compression/GDeflate (GDeflateDecompressorGpu): GDeflate tile streams written by Microsoft's
# reference implementation (github.com/microsoft/DirectStorage, GDeflate/, Apache 2.0, on NVIDIA's libdeflate fork),
# never by our code. The reference is built from source in the session scratchpad (gdeflate_cli: cmake + MSVC, a
# 60-line driver around GDeflate::Compress / GDeflate::Decompress); pass its path as the second argument.
#
# Stream format (TileStream.h): 8-byte header {u8 id = 4, u8 magic = id ^ 0xFF, u16 numTiles, u32 tileSizeIdx:2 = 1 |
# lastTileSize:18 << 2}, then numTiles u32 (entry 0 = the LAST tile's compressed size, entry i > 0 = tile i's offset),
# then the tiles. Each tile is 64 KB of input as one GDeflate stream (DEFLATE blocks, symbols swizzled over 32
# sub-streams).
#
# Inputs are procedural (gen_* from make_lz4_fixtures.py / make_snappy_fixtures.py; the tests rebuild them) or real files
# embedded in the test assembly, at compression levels 1, 6 and 12 (stored, static and dynamic blocks all occur: noise
# gives stored blocks, tiny inputs static ones). Every stream is decoded by the reference here and must give the input.
# Damaged streams: the reference must refuse them too (or, where it cannot see the damage, the case is not added).
#
# manifest.tsv columns: name, content kind, content length, seed, level, stream length, expected ("ok" or a
# GDeflateError name), description. Streams: <name>.gdf.
#
# usage: python make_gdeflate_fixtures.py <TestData dir> <path to gdcli.exe>

import os, struct, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_lz4_fixtures import gen_text, gen_noise, gen_struct
from make_snappy_fixtures import gen_mixed, gen_far

GEN = {"text": gen_text, "noise": gen_noise, "struct": gen_struct, "mixed": gen_mixed, "far": gen_far}

def content(kind, n, seed):
    if kind == "struct" and seed < 0: return bytes(n)
    return GEN[kind](n, seed)

def main(testdata, cli):
    out = os.path.join(testdata, "GDeflate")
    os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="gdf_")
    rows = []

    def compress(data, level):
        src = os.path.join(tmp, "in.bin"); dst = os.path.join(tmp, "out.gdf")
        open(src, "wb").write(data)
        subprocess.run([cli, "c", str(level), src, dst], check=True, capture_output=True)
        return open(dst, "rb").read()

    def decodes(stream):
        src = os.path.join(tmp, "s.gdf"); dst = os.path.join(tmp, "s.out")
        open(src, "wb").write(stream)
        r = subprocess.run([cli, "d", src, dst], capture_output=True)
        return open(dst, "rb").read() if r.returncode == 0 else None

    def add(name, kind, n, seed, level, stream, expected="ok", desc=""):
        got = decodes(stream)
        if expected == "ok":
            assert got == content_of(kind, n, seed), f"{name}: the reference does not decode its own stream to the content"
        else:
            assert got is None, f"{name}: the reference accepted a stream we expect refused ({expected})"
        open(os.path.join(out, name + ".gdf"), "wb").write(stream)
        rows.append((name, kind, n, seed, level, len(stream), expected, desc))

    files = {}
    def content_of(kind, n, seed):
        if kind.startswith("file:"):
            return files[kind]
        return content(kind, n, seed)

    # (GDeflate::Compress refuses empty input, so there is no empty stream.)
    for level in (1, 6, 12):
        for n in (1, 100, 65535, 65536, 65537, 300_000):
            d = content("text", n, 7 + n % 13)
            add(f"text_{n}_l{level}", "text", n, 7 + n % 13, level, compress(d, level))
        d = content("noise", 200_000, 99)
        add(f"noise_200000_l{level}", "noise", 200_000, 99, level, compress(d, level), desc="incompressible: stored blocks")
        d = content("struct", 1_000_003, 5)
        add(f"struct_1000003_l{level}", "struct", 1_000_003, 5, level, compress(d, level))
    add("zeros_2000000_l6", "struct", 2_000_000, -1, 6, compress(bytes(2_000_000), 6), desc="one long match run per tile")
    add("mixed_200000_l9", "mixed", 200_000, 3, 9, compress(content("mixed", 200_000, 3), 9))
    add("far_700000_l12", "far", 700_000, 9, 12, compress(content("far", 700_000, 9), 12))
    add("text_8000000_l6", "text", 8_000_000, 31, 6, compress(content("text", 8_000_000, 31), 6), desc="123 tiles")
    for rel in ("Big_Buck_Bunny_180_10s.webm", "bbb_first_frame.yuv", "bbb_180_2s.ivf"):
        data = open(os.path.join(testdata, rel), "rb").read()
        files["file:" + rel] = data
        add("file_" + rel.replace(".", "_") + "_l12", "file:" + rel, len(data), 0, 12, compress(data, 12))

    # Damaged streams (the reference refuses each one).
    good = compress(content("text", 200_000, 44), 6)
    def flip(b, at, bit=1):
        b = bytearray(b); b[at] ^= bit; return bytes(b)
    add("bad_magic", "text", 200_000, 44, 6, flip(good, 1), "BadHeader", "magic != id ^ 0xFF")
    add("bad_codec_id", "text", 200_000, 44, 6, bytes([5, 5 ^ 0xFF]) + good[2:], "BadHeader", "codec id 5 (not GDeflate)")
    tiles = struct.unpack_from("<H", good, 2)[0]
    data_at = 8 + 4 * tiles
    # The first tile's first block header: BTYPE = 3 (reserved). Bits 1..2 of the first 32-bit packet of sub-stream 0.
    bt = bytearray(good); bt[data_at] |= 0x06
    add("bad_block_type", "text", 200_000, 44, 6, bytes(bt), "BadBlockType", "first block BTYPE set to reserved 3")

    with open(os.path.join(out, "manifest.tsv"), "w", newline="\n") as m:
        m.write("# Microsoft DirectStorage GDeflate reference (NVIDIA libdeflate fork) via gdcli\n")
        for r in rows: m.write("\t".join(str(x) for x in r) + "\n")
    print(f"{len(rows)} streams -> {out}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
