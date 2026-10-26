# Truth fixtures for Compression/Lz4 (Lz4FrameCompressorGpu): LZ4 frames made by liblz4 (python-lz4's bundled
# library, LZ4F_compressFrame), never by our code. Independent 64 KB blocks (block_linked=False), the configuration our
# compressor reproduces byte for byte.
#
# Inputs are procedural (gen_* below; Lz4FrameCompressorGpuTests rebuilds the same bytes) or real files already embedded
# in the test assembly (TestData/*.webm, *.yuv, *.ivf). Truth: the whole frame (<name>.lz4) when it is under 512 KiB, else
# its length and SHA-256 in manifest.tsv.
#
# manifest.tsv columns: name, input kind, input length, seed, store_size, block_checksum, content_checksum,
# acceleration, frame length, frame SHA-256, stored (1 = <name>.lz4 exists).
#
# usage: python make_lz4_fixtures.py <TestData dir>     (needs lz4)

import hashlib, os, sys
import lz4, lz4.frame

WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "pack", "my", "box", "with", "five", "dozen",
         "liquor", "jugs", "sphinx", "of", "black", "quartz"]

def gen_text(n, seed):
    out = bytearray()
    x = seed & 0x7FFFFFFF
    while len(out) < n:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out += WORDS[x % 20].encode()
        out += b".\n" if (x >> 8) % 9 == 0 else b" "
    return bytes(out[:n])

def gen_noise(n, seed):
    out = bytearray(n)
    x = (seed | 1) & 0xFFFFFFFF
    for i in range(n):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out[i] = x & 0xFF
    return bytes(out)

def gen_struct(n, seed):
    # runs, ramps, repeated records and a few noisy bytes: long matches, overlapping offsets, short literals
    out = bytearray(n)
    for i in range(n):
        r = (i // 37) % 11
        if r < 3: v = 0
        elif r < 6: v = (i * 3 + seed) & 0xFF
        elif r < 9: v = ((i % 23) * 7 + (i // 4096)) & 0xFF
        else: v = ((i * 2654435761 + seed) >> 11) & 0xFF
        out[i] = v
    return bytes(out)

GEN = {"text": gen_text, "noise": gen_noise, "struct": gen_struct}

def frame(data, store_size, block_checksum, content_checksum, accel):
    return lz4.frame.compress(data, compression_level=0 if accel == 1 else -(accel - 1),
                              block_size=lz4.frame.BLOCKSIZE_MAX64KB, block_linked=False,
                              content_checksum=bool(content_checksum), block_checksum=bool(block_checksum),
                              store_size=bool(store_size))

def main(testdata):
    out = os.path.join(testdata, "Lz4")
    os.makedirs(out, exist_ok=True)
    rows = []
    def add(name, kind, data, seed, ss=1, bc=0, cc=0, accel=1):
        f = frame(data, ss, bc, cc, accel)
        stored = len(f) < 512 * 1024
        if stored:
            open(os.path.join(out, name + ".lz4"), "wb").write(f)
        rows.append((name, kind, len(data), seed, ss, bc, cc, accel, len(f), hashlib.sha256(f).hexdigest(), int(stored)))

    for n in (0, 1, 12, 13, 14, 100, 4096, 65535, 65536, 65537, 200_000):
        add(f"text_{n}", "text", gen_text(n, 7 + n % 13), 7 + n % 13)
    for n in (13, 1000, 65536, 300_000):
        add(f"noise_{n}", "noise", gen_noise(n, 99 + n % 7), 99 + n % 7)
    for n in (16, 5000, 65536, 1_000_003):
        add(f"struct_{n}", "struct", gen_struct(n, 5 + n % 3), 5 + n % 3)
    add("zeros_200000", "struct", bytes(200_000), -1)
    # options
    add("text_150000_bc", "text", gen_text(150_000, 21), 21, bc=1)
    add("text_150000_cc", "text", gen_text(150_000, 22), 22, cc=1)
    add("text_150001_bc_cc_nosize", "text", gen_text(150_001, 23), 23, ss=0, bc=1, cc=1)
    add("struct_700000_accel8", "struct", gen_struct(700_000, 4), 4, accel=8)
    add("text_300000_accel3_cc", "text", gen_text(300_000, 25), 25, cc=1, accel=3)
    # multi-wave (the tests run this one with small waves) and production scale
    add("text_40000000", "text", gen_text(40_000_000, 31), 31)
    # real files embedded in the test assembly
    for rel in ("Big_Buck_Bunny_180_10s.webm", "bbb_first_frame.yuv", "bbb_180_2s.ivf"):
        data = open(os.path.join(testdata, rel), "rb").read()
        add("file_" + rel.replace(".", "_"), "file:" + rel, data, 0, cc=1)

    with open(os.path.join(out, "manifest.tsv"), "w", newline="\n") as m:
        m.write("# liblz4 " + lz4.library_version_string() + "\n")
        for r in rows:
            m.write("\t".join(str(x) for x in r) + "\n")
    print(f"{len(rows)} frames -> {out} (liblz4 {lz4.library_version_string()})")

if __name__ == "__main__":
    main(sys.argv[1])
