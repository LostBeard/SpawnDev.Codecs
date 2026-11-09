"""Pack libvpx_ref -c context dumps (one .ctx per test file) into one fixture: libvpx_contexts.gz.

Usage: pack_contexts.py <dir of <file name>.ctx> <out.gz>
(Not "x.bin.gz": "bin" is a culture code and MSBuild puts such embedded resources in a satellite assembly.)
Format (little-endian), per test file in name order: u16 name length, name (ASCII), u32 record count, then records:
u8 stage (0 = context loaded, 1 = after the compressed header), 2040 context bytes XOR the previous record's context
in this file (zeros before the first), and for stage 1 five more bytes: tx_mode, reference_mode, comp_fixed_ref,
comp_var_ref[0], comp_var_ref[1]. Byte 2039 is struct padding (libvpx FRAME_CONTEXT's int 'initialized' is at 2040).
"""
import gzip, os, struct, sys

CTX = 2040
src, out = sys.argv[1], sys.argv[2]
blob = bytearray()
for name in sorted(os.listdir(src)):
    if not name.endswith(".ctx"):
        continue
    data = open(os.path.join(src, name), "rb").read()
    records, at, prev = [], 0, bytes(CTX)
    while at < len(data):
        stage = data[at]
        ctx = data[at + 1:at + 1 + CTX]
        rec = bytes([stage]) + bytes(a ^ b for a, b in zip(ctx, prev))
        at += 1 + CTX
        if stage == 1:
            rec += data[at:at + 5]
            at += 5
        records.append(rec)
        prev = ctx
    test_name = name[:-4]
    blob += struct.pack("<H", len(test_name)) + test_name.encode() + struct.pack("<I", len(records))
    for r in records:
        blob += r
with open(out, "wb") as f:
    f.write(gzip.compress(bytes(blob), compresslevel=9, mtime=0))
print(f"{len(blob)} bytes -> {os.path.getsize(out)} gz")
