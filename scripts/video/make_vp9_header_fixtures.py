"""Synthetic VP9 header fixtures for header paths no conformance vector exercises (absolute segment data).

Takes the first frame (a profile-0 key frame without segmentation) of a VP9 IVF and re-packs only its uncompressed
header: the segmentation_enabled bit becomes a segmentation block (enabled, update_map 0, update_data 1, the given
abs_delta and features). The rest of the header and every byte after it (compressed header, tiles) are unchanged; with
update_map 0 the tile syntax does not change, so libvpx decodes the frame (with other dequantizers) and its header
dump gives the expected values.

Usage: make_vp9_header_fixtures.py <source.ivf> <out dir>
Writes vp9_hdr_<name>.ivf per case. Then run libvpx_ref -h on each for the expected header line.
"""
import os, struct, sys


class Bits:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def read(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | ((self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v


def bits_of(v, n):
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def split_header(frame):
    """Bit position of segmentation_enabled and of the end of the uncompressed header (profile-0 key frame)."""
    b = Bits(frame)
    assert b.read(2) == 2
    profile = b.read(1) | (b.read(1) << 1)
    assert profile == 0, "profile-0 source expected"
    assert b.read(1) == 0, "show_existing_frame"
    assert b.read(1) == 0, "key frame expected"
    b.read(1)  # show_frame
    error_res = b.read(1)
    assert b.read(24) == 0x498342
    color_space = b.read(3)
    if color_space != 7:
        b.read(1)
    b.read(32)  # size
    if b.read(1):
        b.read(32)  # render size
    if not error_res:
        b.read(2)  # refresh_frame_context, frame_parallel
    b.read(2)  # frame_context_idx
    b.read(9)  # filter level, sharpness
    if b.read(1):  # mode_ref_delta_enabled
        if b.read(1):  # update
            for _ in range(6):
                if b.read(1):
                    b.read(7)
    b.read(8)  # base_q_idx
    for _ in range(3):
        if b.read(1):
            b.read(5)
    seg_pos = b.pos
    assert b.read(1) == 0, "source already has segmentation"
    rest_start = b.pos
    # tile info: needs mi_cols; read through to first_partition_size by trying the remaining syntax
    return seg_pos, rest_start


def header_end(frame, rest_start, width):
    b = Bits(frame)
    b.pos = rest_start
    mi_cols = (width + 7) >> 3
    sb64 = (mi_cols + 7) >> 3
    min_log2 = 0
    while (64 << min_log2) < sb64:
        min_log2 += 1
    max_log2 = 1
    while (sb64 >> max_log2) >= 4:
        max_log2 += 1
    max_log2 -= 1
    ones = max_log2 - min_log2
    while ones > 0 and b.read(1):
        ones -= 1
    if b.read(1):
        b.read(1)
    b.read(16)  # first_partition_size
    return b.pos


def repack(frame, width, seg_bits):
    seg_pos, rest_start = split_header(frame)
    end = header_end(frame, rest_start, width)
    src = Bits(frame)
    out = [src.read(1) for _ in range(seg_pos)]
    out += seg_bits
    src.pos = rest_start
    out += [src.read(1) for _ in range(end - rest_start)]
    while len(out) % 8:
        out.append(0)
    head = bytes(int("".join(map(str, out[i:i + 8])), 2) for i in range(0, len(out), 8))
    return head + frame[(end + 7) // 8:]


def seg_block(abs_delta, features):
    """features: {segment: {feature: value}}; feature 0 ALT_Q (8 bits + sign), 1 ALT_LF (6 bits + sign), 2 REF (2 bits),
    3 SKIP (0 bits)."""
    bits = [1, 0, 1, abs_delta]  # enabled, update_map 0, update_data 1, abs_delta
    widths, signed = [8, 6, 2, 0], [1, 1, 0, 0]
    for s in range(8):
        for f in range(4):
            if f in features.get(s, {}):
                v = features[s][f]
                bits += [1] + bits_of(abs(v), widths[f])
                if signed[f]:
                    bits.append(1 if v < 0 else 0)
            else:
                bits.append(0)
    return bits


def read_first_frame(path):
    data = open(path, "rb").read()
    assert data[:4] == b"DKIF"
    hdr_len = struct.unpack_from("<H", data, 6)[0]
    width, height = struct.unpack_from("<HH", data, 12)
    size = struct.unpack_from("<I", data, hdr_len)[0]
    return data[:32], width, height, data[hdr_len + 12:hdr_len + 12 + size]


def write_ivf(path, ivf_header, frame):
    h = bytearray(ivf_header)
    struct.pack_into("<I", h, 24, 1)
    open(path, "wb").write(bytes(h) + struct.pack("<IQ", len(frame), 0) + frame)


src, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
ivf_header, width, height, frame = read_first_frame(src)
cases = {
    "segabs": seg_block(1, {0: {0: 40}, 1: {0: 200}, 2: {0: 0}, 3: {0: 255}, 4: {1: 20}, 5: {0: 90, 1: 5}}),
    "segdelta": seg_block(0, {0: {0: -30}, 1: {0: 200}, 2: {0: -255}, 6: {2: 2, 3: 0}, 7: {1: -63}}),
}
for name, bits in cases.items():
    path = os.path.join(out_dir, f"vp9_hdr_{name}.ivf")
    write_ivf(path, ivf_header, repack(frame, width, bits))
    print("wrote", path)
