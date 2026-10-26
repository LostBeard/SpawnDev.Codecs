# Truth fixtures for Compression/Snappy (SnappyCompressorGpu, SnappyDecompressorGpu).
#
# Compressor (TestData/Snappy): raw Snappy streams and Snappy framed streams written by the Rust snap crate (cramjam,
# python-snappy's backend: snappy.compress = raw, snappy.stream_compress = framed), never by our code. snap compresses
# 64 KB blocks independently (fresh hash table per block), which is what the GPU compressor reproduces byte for byte.
# Inputs are procedural (gen_* from make_lz4_fixtures.py, plus gen_mixed: noise with every period-th 32-byte run
# repeated, a tunable ratio; period 4 lands between 7/8 and 6/7 of the block, so it decides the framed format's
# "stored uncompressed" rule) or real files embedded in the test assembly. Truth: the whole stream
# (<name>.raw.sz / <name>.framed.sz) when it is under 512 KiB, else its length and SHA-256 in manifest.tsv.
#
# manifest.tsv columns: name, input kind, input length, seed, raw length, raw SHA-256, raw stored, framed length,
# framed SHA-256, framed stored.
#
# Decoder (TestData/SnappyDec): what a decoder meets beyond snap's output - literal lengths in the 1..4-byte forms
# (tags 60..63), copy-4 elements, offsets reaching past 64 KB (one raw stream with no block structure), overlapping
# copies, framed streams with chunks of any size, uncompressed / padding / skippable chunks and a repeated stream
# identifier; and damaged streams each refused for a named reason. Every good stream is checked here to decode with
# snap to its content, and every damaged one to be refused by snap.
#
# SnappyDec/manifest.tsv columns: name, format (raw | framed), content kind, content length, seed, expected ("ok" or a
# SnappyError name), description. Content kinds: the generators, "far" (text + noise + the same text), "slices"
# (text(seed)).
#
# usage: python make_snappy_fixtures.py <TestData dir>     (needs cramjam; imports the generators next to this file)

import hashlib, os, struct, sys
import cramjam
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_lz4_fixtures import gen_text, gen_noise, gen_struct

def gen_mixed(n, seed, period=4):
    x = (seed | 1) & 0xFFFFFFFF
    out = bytearray(n)
    for i in range(n):
        if (i // 32) % period == period - 1 and i >= 32:
            out[i] = out[i - 32]
            continue
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out[i] = x & 0xFF
    return bytes(out)

def gen_far(n, seed):
    # text, noise, then the same text again: the second text copies from up to n * 3 / 7 bytes back
    a = n * 3 // 7
    b = n - 2 * a
    return gen_text(a, seed) + gen_noise(b, seed + 1) + gen_text(a, seed)

GEN = {"text": gen_text, "noise": gen_noise, "struct": gen_struct, "mixed": gen_mixed, "far": gen_far}

def content(kind, n, seed):
    if kind == "struct" and seed < 0: return bytes(n)
    return GEN[kind](n, seed)

def raw(data): return bytes(cramjam.snappy.compress_raw(data))
def framed(data): return bytes(cramjam.snappy.compress(data))

# ---------------------------------------------------------------------------------------------------------------
# Hand-built raw elements (decoder fixtures)

def varint(n):
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)

def lit(data, form=None):
    m = len(data) - 1
    if form is None: form = 0 if m < 60 else 1 if m < 256 else 2 if m < 65536 else 3 if m < 1 << 24 else 4
    if form == 0: return bytes([m << 2]) + data
    return bytes([(59 + form) << 2]) + m.to_bytes(form, "little") + data

def copy1(off, n): return bytes([((off >> 8) << 5) | ((n - 4) << 2) | 1, off & 0xFF])
def copy2(off, n): return bytes([((n - 1) << 2) | 2]) + struct.pack("<H", off)
def copy4(off, n): return bytes([((n - 1) << 2) | 3]) + struct.pack("<I", off)

def greedy_raw(data):
    """A raw stream over the WHOLE input (no 64 KB blocks): offsets of any distance, every literal and copy form."""
    out = bytearray(varint(len(data)))
    table = {}
    i = anchor = 0
    nlit = ncopy = 0
    n = len(data)
    def emit_lit(a, b):
        nonlocal nlit
        while a < b:
            k = min(b - a, 70000)
            # rotate through the length forms: minimal, then 1, 2, 3 and 4 bytes when they can hold it
            forms = [None, 1, 2, 3, 4]
            f = forms[nlit % 5]
            m = k - 1
            if f is not None and m >= 1 << (8 * f): f = None
            out.extend(lit(data[a:a + k], f))
            nlit += 1
            a += k
    while i + 4 <= n:
        key = data[i:i + 4]
        j = table.get(key)
        table[key] = i
        if j is None or data[j:j + 4] != key:
            i += 1
            continue
        emit_lit(anchor, i)
        L = 4
        while i + L < n and data[j + L] == data[i + L]: L += 1
        off = i - j
        p = i
        rest = L
        while rest > 0:
            k = min(rest, 64)
            if rest - k in (1, 2, 3): k = rest - 4 if rest > 64 else rest  # keep every piece >= 4 for copy1
            if off >= 65536: out.extend(copy4(off, k))
            elif 4 <= k <= 11 and off < 2048 and ncopy % 3 != 2: out.extend(copy1(off, k))
            elif ncopy % 3 == 2: out.extend(copy4(off, k))
            else: out.extend(copy2(off, k))
            ncopy += 1
            rest -= k
        i += L
        anchor = i
    emit_lit(anchor, n)
    return bytes(out)

def crc32c(data):
    c = 0xFFFFFFFF
    tab = crc32c.tab
    for b in data: c = tab[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF
crc32c.tab = []
for k in range(256):
    c = k
    for _ in range(8): c = (c >> 1) ^ (0x82F63B78 & -(c & 1))
    crc32c.tab.append(c)

def masked(data):
    c = crc32c(data)
    return ((((c >> 15) | (c << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF

STREAM_ID = b"\xff\x06\x00\x00sNaPpY"

def chunk(typ, body): return bytes([typ]) + struct.pack("<I", len(body))[:3] + body
def chunk_compressed(data, crc_of=None): return chunk(0x00, struct.pack("<I", masked(data if crc_of is None else crc_of)) + raw(data))
def chunk_stored(data): return chunk(0x01, struct.pack("<I", masked(data)) + data)

def main(testdata):
    # ------------------------------------------------------------------ compressor fixtures
    out = os.path.join(testdata, "Snappy")
    os.makedirs(out, exist_ok=True)
    rows = []
    def add(name, kind, data, seed):
        r, f = raw(data), framed(data)
        assert bytes(cramjam.snappy.decompress_raw(r)) == data and bytes(cramjam.snappy.decompress(f)) == data
        rs, fs = len(r) < 512 * 1024, len(f) < 512 * 1024
        if rs: open(os.path.join(out, name + ".raw.sz"), "wb").write(r)
        if fs: open(os.path.join(out, name + ".framed.sz"), "wb").write(f)
        rows.append((name, kind, len(data), seed, len(r), hashlib.sha256(r).hexdigest(), int(rs),
                     len(f), hashlib.sha256(f).hexdigest(), int(fs)))

    for n in (0, 1, 15, 16, 17, 18, 100, 4096, 65535, 65536, 65537, 200_000):
        add(f"text_{n}", "text", content("text", n, 7 + n % 13), 7 + n % 13)
    for n in (16, 17, 1000, 65536, 300_000):
        add(f"noise_{n}", "noise", content("noise", n, 99 + n % 7), 99 + n % 7)
    for n in (17, 5000, 65536, 1_000_003):
        add(f"struct_{n}", "struct", content("struct", n, 5 + n % 3), 5 + n % 3)
    add("zeros_200000", "struct", bytes(200_000), -1)
    for n in (65536, 200_000):
        add(f"mixed_{n}", "mixed", content("mixed", n, 3), 3)
    add("text_40000000", "text", content("text", 40_000_000, 31), 31)
    for rel in ("Big_Buck_Bunny_180_10s.webm", "bbb_first_frame.yuv", "bbb_180_2s.ivf"):
        data = open(os.path.join(testdata, rel), "rb").read()
        add("file_" + rel.replace(".", "_"), "file:" + rel, data, 0)
    # the framed "stored uncompressed" rule must be decided by mixed_65536 (between 7/8 and 6/7 of the block)
    m = content("mixed", 65536, 3)
    assert 65536 - 65536 // 7 <= len(raw(m)) < 65536 - 65536 // 8, len(raw(m))

    with open(os.path.join(out, "manifest.tsv"), "w", newline="\n") as mf:
        mf.write(f"# Rust snap crate via cramjam {cramjam.__version__}\n")
        for r in rows: mf.write("\t".join(str(x) for x in r) + "\n")
    print(f"{len(rows)} inputs -> {out} (cramjam {cramjam.__version__})")

    # ------------------------------------------------------------------ decoder fixtures
    dout = os.path.join(testdata, "SnappyDec")
    os.makedirs(dout, exist_ok=True)
    drows = []
    def dadd(name, fmt, kind, n, seed, stream, expected="ok", desc=""):
        dec = cramjam.snappy.decompress_raw if fmt == "raw" else cramjam.snappy.decompress
        if expected == "ok":
            assert bytes(dec(stream)) == content(kind, n, seed), name
        else:
            try:
                dec(stream)
                raise AssertionError(f"{name}: snap accepted a stream we expect refused ({expected})")
            except cramjam.DecompressionError:
                pass
        open(os.path.join(dout, name + ".sz"), "wb").write(stream)
        drows.append((name, fmt, kind, n, seed, expected, desc))

    # snap's own output
    t = content("text", 700_000, 41)
    dadd("text_700000_raw", "raw", "text", 700_000, 41, raw(t), desc="snap raw, 11 blocks")
    dadd("text_700000_framed", "framed", "text", 700_000, 41, framed(t), desc="snap framed")
    z = bytes(5_000_000)
    dadd("zeros_5000000_raw", "raw", "struct", 5_000_000, -1, raw(z), desc="snap raw, 77 blocks of offset-1 copies")
    nz = content("noise", 300_000, 5)
    dadd("noise_300000_framed", "framed", "noise", 300_000, 5, framed(nz), desc="incompressible: uncompressed chunks")
    dadd("empty_raw", "raw", "text", 0, 0, b"\x00", desc="empty raw stream (varint 0)")
    dadd("empty_framed", "framed", "text", 0, 0, b"", desc="empty framed stream (no chunks)")
    dadd("empty_framed_id_only", "framed", "text", 0, 0, STREAM_ID, desc="stream identifier only")

    # whole-stream greedy encoder: far offsets (copy-4), all literal forms, non-minimal copies
    f = content("far", 700_000, 9)
    dadd("far_700000_greedy_raw", "raw", "far", 700_000, 9, greedy_raw(f), desc="no block structure: copies 400 KB back, tags 60-63, copy-4")
    s = content("struct", 300_000, 2)
    dadd("struct_300000_greedy_raw", "raw", "struct", 300_000, 2, greedy_raw(s), desc="greedy, overlapping copies")
    # hand-built: content kind "file" = <name>.content beside the stream (snap's decode of it, checked against the
    # content these elements spell out)
    elems = [(lit(b"abcd"), b"abcd"), (lit(b"x", 4), b"x"), (copy2(1, 64), None), (copy4(5, 20), None),
             (lit(b"yz!", 2), b"yz!")] + [(copy4(3, 64), None)] * 1093 + [(copy2(3, 48), None), (copy1(2047, 11), None)]
    spelled = bytearray()
    body = bytearray()
    for e, data in elems:
        body += e
        if data is not None: spelled += data
        else:
            tag = e[0] & 3
            n = ((e[0] >> 2) & 7) + 4 if tag == 1 else (e[0] >> 2) + 1
            off = ((e[0] >> 5) << 8) | e[1] if tag == 1 else int.from_bytes(e[1:], "little")
            for _ in range(n): spelled.append(spelled[-off])
    hand = varint(len(spelled)) + bytes(body)
    assert bytes(cramjam.snappy.decompress_raw(hand)) == bytes(spelled)
    open(os.path.join(dout, "hand_raw.content"), "wb").write(spelled)
    open(os.path.join(dout, "hand_raw.sz"), "wb").write(hand)
    drows.append(("hand_raw", "raw", "file", len(spelled), 0, "ok", "hand-built: 2- and 4-byte literal forms, copy-4 runs, offsets 1 / 3 / 5 / 2047"))

    # framed streams of any chunk size, stored / padding / skippable chunks, a repeated identifier
    c = content("text", 200_000, 43)
    parts = [(0, 1000, "c"), (1000, 65536, "c"), (66536, 1, "s"), (66537, 30000, "s"), (96537, 3, "c"), (96540, 65536, "c")]
    st = bytearray(STREAM_ID)
    for a, k, how in parts:
        st += chunk_compressed(c[a:a + k]) if how == "c" else chunk_stored(c[a:a + k])
    st += chunk(0xFE, bytes(10)) + chunk(0x80, b"hello") + chunk(0xFD, b"")
    st += STREAM_ID
    rest = c[162076:]
    st += chunk_compressed(rest[:20000]) + chunk_stored(rest[20000:])
    dadd("text_200000_framed_mixed_chunks", "framed", "text", 200_000, 43, bytes(st),
         desc="chunks of 1 B..64 KB compressed and stored, padding + skippable chunks, a second stream identifier")

    # damaged streams
    good = framed(content("text", 200_000, 44))
    def flip(b, at, bit=1):
        b = bytearray(b); b[at] ^= bit; return bytes(b)
    dadd("bad_framed_no_identifier", "framed", "text", 200_000, 44, good[10:], "BadStreamIdentifier", "chunks without the stream identifier")
    dadd("bad_framed_identifier_body", "framed", "text", 200_000, 44, flip(good, 5), "BadStreamIdentifier", "identifier body sNaPpY damaged")
    dadd("bad_framed_reserved_chunk", "framed", "text", 200_000, 44, good[:10] + chunk(0x02, b"abc") + good[10:], "ReservedChunk", "unskippable reserved chunk type 0x02")
    dadd("bad_framed_checksum", "framed", "text", 200_000, 44, flip(good, 14), "ChunkChecksum", "first chunk's CRC-32C damaged")
    dadd("bad_framed_truncated", "framed", "text", 200_000, 44, good[:len(good) - 3], "Truncated", "last chunk cut short")
    big = content("text", 70_000, 45)
    dadd("bad_framed_compressed_too_large", "framed", "text", 70_000, 45, STREAM_ID + chunk_compressed(big), "ChunkTooLarge", "a compressed chunk that decodes to 70000 bytes")
    dadd("bad_framed_stored_too_large", "framed", "text", 70_000, 45, STREAM_ID + chunk_stored(big), "ChunkTooLarge", "an uncompressed chunk of 70000 bytes")
    small = content("text", 5000, 46)
    inner = bytearray(raw(small)); inner[0:2] = varint(4999)
    dadd("bad_framed_inner_length", "framed", "text", 5000, 46, STREAM_ID + chunk(0x00, struct.pack("<I", masked(small)) + bytes(inner)), "LengthMismatch", "chunk's raw stream declares 4999 bytes and holds 5000")

    rg = raw(content("text", 200_000, 47))
    dadd("bad_raw_offset_zero", "raw", "text", 8, 0, varint(8) + lit(b"abcd") + copy1(0, 4), "BadOffset", "copy offset 0")
    dadd("bad_raw_offset_before_start", "raw", "text", 12, 0, varint(12) + lit(b"abcd") + copy2(5, 8), "BadOffset", "copy reaching 1 byte before the output")
    dadd("bad_raw_length_short", "raw", "text", 200_000, 47, varint(200_001) + rg[len(varint(200_000)):], "LengthMismatch", "declares one byte more than it holds")
    dadd("bad_raw_length_long", "raw", "text", 200_000, 47, varint(199_999) + rg[len(varint(200_000)):], "LengthMismatch", "declares one byte less than it holds")
    dadd("bad_raw_truncated", "raw", "text", 200_000, 47, rg[:len(rg) - 5], "Truncated", "last literal cut short")
    dadd("bad_raw_varint", "raw", "text", 0, 0, b"\xff\xff\xff\xff\xff\x01", "BadVarint", "length varint longer than 5 bytes / 32 bits")
    dadd("bad_raw_empty", "raw", "text", 0, 0, b"", "Truncated", "no length varint")

    with open(os.path.join(dout, "manifest.tsv"), "w", newline="\n") as mf:
        mf.write(f"# Rust snap crate via cramjam {cramjam.__version__}\n")
        for r in drows: mf.write("\t".join(str(x) for x in r) + "\n")
    print(f"{len(drows)} streams -> {dout}")

if __name__ == "__main__":
    main(sys.argv[1])
