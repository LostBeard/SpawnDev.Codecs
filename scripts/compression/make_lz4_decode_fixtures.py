# Decoder fixtures for Compression/Lz4 (Lz4FrameDecompressorGpu): LZ4 frames written by liblz4 (python-lz4) in every
# configuration a decoder meets, plus hand-damaged frames that must be refused with a specific reason.
#
# Content is procedural (the same generators as make_lz4_fixtures.py; the tests rebuild it), so the truth of a good frame
# is its content; a bad frame's truth is the error name. Frames go to TestData/Lz4Dec/<name>.lz4.
#
# manifest.tsv columns: name, content kind, content length, seed, expected ("ok" or an Lz4Error name), description.
#
# usage: python make_lz4_decode_fixtures.py <TestData dir>     (needs lz4; imports the generators next to this file)

import os, struct, sys
import lz4, lz4.frame, lz4.block
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_lz4_fixtures import gen_text, gen_noise, gen_struct

GEN = {"text": gen_text, "noise": gen_noise, "struct": gen_struct}
SIZES = {64: lz4.frame.BLOCKSIZE_MAX64KB, 256: lz4.frame.BLOCKSIZE_MAX256KB, 1024: lz4.frame.BLOCKSIZE_MAX1MB, 4096: lz4.frame.BLOCKSIZE_MAX4MB}

def legacy(data):
    out = bytearray(struct.pack('<I', 0x184C2102))
    for i in range(0, len(data), 8 << 20):
        c = lz4.block.compress(data[i:i + (8 << 20)], mode='default', store_size=False)
        out += struct.pack('<I', len(c)) + c
    return bytes(out)

def skippable(payload, nibble=0):
    return struct.pack('<II', 0x184D2A50 + nibble, len(payload)) + payload

def main(testdata):
    out = os.path.join(testdata, "Lz4Dec")
    os.makedirs(out, exist_ok=True)
    rows = []
    def add(name, kind, n, seed, frame, expected="ok", desc=""):
        open(os.path.join(out, name + ".lz4"), "wb").write(frame)
        rows.append((name, kind, n, seed, expected, desc))
    def content(kind, n, seed):
        return GEN[kind](n, seed)

    # liblz4 configurations
    for bs in (64, 256, 1024, 4096):
        for linked in (True, False):
            n = 2_500_000 if bs >= 1024 else 700_000
            d = content("text", n, bs + linked)
            add(f"text_{n}_b{bs}_{'linked' if linked else 'indep'}", "text", n, bs + linked,
                lz4.frame.compress(d, block_size=SIZES[bs], block_linked=linked), desc=f"{bs} KB blocks, linked={linked}")
    d = content("struct", 2_000_000, 3)
    add("struct_2000000_hc9_linked", "struct", 2_000_000, 3, lz4.frame.compress(d, compression_level=9, block_linked=True), desc="HC level 9, linked")
    add("struct_2000000_hc16_b4096", "struct", 2_000_000, 3, lz4.frame.compress(d, compression_level=16, block_size=SIZES[4096], block_linked=False), desc="HC level 16 (optimal parser), 4 MB blocks")
    d = content("text", 500_000, 11)
    add("text_500000_allchecks", "text", 500_000, 11, lz4.frame.compress(d, block_checksum=True, content_checksum=True, block_linked=True), desc="block + content checksums, linked")
    add("text_500000_nosize", "text", 500_000, 11, lz4.frame.compress(d, store_size=False), desc="no content size")
    d = content("noise", 300_000, 5)
    add("noise_300000_raw", "noise", 300_000, 5, lz4.frame.compress(d, block_checksum=True), desc="incompressible: raw blocks")
    add("zeros_5000000_b4096", "struct", 5_000_000, -1, lz4.frame.compress(bytes(5_000_000), block_size=SIZES[4096], content_checksum=True), desc="one 4 MB match run per block")
    add("empty", "text", 0, 0, lz4.frame.compress(b""), desc="empty frame")
    d = content("struct", 9_000_000, 13)
    add("struct_9000000_legacy", "struct", 9_000_000, 13, legacy(d), desc="legacy frame, two 8 MB blocks")
    # concatenation and skippable frames
    a = content("text", 100_000, 17)
    b = content("text", 100_000, 18)
    cat = skippable(b"metadata!") + lz4.frame.compress(a, block_linked=False) + skippable(b"", 15) + lz4.frame.compress(b, content_checksum=True)
    add("concat_2x100000_skippable", "text2", 200_000, 17, cat, desc="skippable + frame + empty skippable + frame (content = text(17) + text(18))")

    # damaged frames: must be refused, never decoded to garbage
    good = lz4.frame.compress(content("text", 200_000, 21), block_checksum=True, content_checksum=True, block_linked=False)
    def flip(f, at, bit=1):
        x = bytearray(f); x[at] ^= bit; return bytes(x)
    add("bad_header_checksum", "text", 200_000, 21, flip(good, 14), "HeaderChecksum")
    add("bad_magic", "text", 200_000, 21, flip(good, 0), "BadMagic")
    add("bad_truncated", "text", 200_000, 21, good[:len(good) - 3], "Truncated")
    # the first block's stored checksum (after its data) off by one bit
    first_stored = struct.unpack('<I', good[15:19])[0] & 0x7FFFFFFF
    add("bad_block_checksum", "text", 200_000, 21, flip(good, 15 + 4 + first_stored), "BlockChecksum")
    # content checksum wrong
    add("bad_content_checksum", "text", 200_000, 21, flip(good, len(good) - 1), "ContentChecksum")
    nochk = lz4.frame.compress(content("text", 200_000, 22), block_linked=False)
    # first block's first match offset pushed past the block start (byte after the first literal run)
    blk = bytearray(nochk)
    p = 15 + 4                      # header (with content size) + block size
    token = blk[p]; lit = token >> 4; q = p + 1
    if lit == 15:
        while True:
            lit += blk[q]; q += 1
            if blk[q - 1] != 255: break
    off_at = q + lit
    blk[off_at] = 0xFF; blk[off_at + 1] = 0xFF
    add("bad_offset", "text", 200_000, 22, bytes(blk), "BadOffset")
    add("bad_dictionary", "text", 200_000, 22, flip(nochk, 4, 0x01), "DictionaryNotSupported")

    with open(os.path.join(out, "manifest.tsv"), "w", newline="\n") as m:
        m.write("# liblz4 " + lz4.library_version_string() + "\n")
        for r in rows:
            m.write("\t".join(str(x) for x in r) + "\n")
    print(f"{len(rows)} frames -> {out}")

if __name__ == "__main__":
    main(sys.argv[1])
