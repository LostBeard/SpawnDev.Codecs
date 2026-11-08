"""Dump the EBML element tree of a Matroska / WebM file (offsets, ids, size-field widths, sizes, values).

Written from RFC 8794 (EBML) and RFC 9559 (Matroska) to inspect muxer output. Usage:
    python -I ebml_dump.py file.webm [--blocks]   (--blocks lists every SimpleBlock / BlockGroup, else summarised)
"""
import struct
import sys

NAMES = {
    0x1A45DFA3: "EBML", 0x4286: "EBMLVersion", 0x42F7: "EBMLReadVersion", 0x42F2: "EBMLMaxIDLength",
    0x42F3: "EBMLMaxSizeLength", 0x4282: "DocType", 0x4287: "DocTypeVersion", 0x4285: "DocTypeReadVersion",
    0x18538067: "Segment", 0x114D9B74: "SeekHead", 0x4DBB: "Seek", 0x53AB: "SeekID", 0x53AC: "SeekPosition",
    0xEC: "Void", 0x1549A966: "Info", 0x2AD7B1: "TimestampScale", 0x4489: "Duration", 0x4D80: "MuxingApp",
    0x5741: "WritingApp", 0x7BA9: "Title", 0x73A4: "SegmentUID", 0x4461: "DateUTC",
    0x1654AE6B: "Tracks", 0xAE: "TrackEntry", 0xD7: "TrackNumber", 0x73C5: "TrackUID", 0x83: "TrackType",
    0x9C: "FlagLacing", 0x22B59C: "Language", 0x86: "CodecID", 0x63A2: "CodecPrivate", 0x536E: "Name",
    0x88: "FlagDefault", 0x55AA: "FlagForced", 0x23E383: "DefaultDuration", 0x56AA: "CodecDelay", 0x56BB: "SeekPreRoll",
    0xE0: "Video", 0xB0: "PixelWidth", 0xBA: "PixelHeight", 0x9A: "FlagInterlaced", 0x53C0: "AlphaMode",
    0x54B0: "DisplayWidth", 0x54BA: "DisplayHeight", 0x54B2: "DisplayUnit", 0x55B0: "Colour", 0x55B1: "MatrixCoefficients",
    0x55B7: "ChromaSitingHorz", 0x55B8: "ChromaSitingVert", 0x55B9: "Range", 0x55BA: "TransferCharacteristics",
    0x55BB: "Primaries", 0xE1: "Audio", 0x9F: "Channels", 0xB5: "SamplingFrequency", 0x6264: "BitDepth",
    0x78B5: "OutputSamplingFrequency", 0x1254C367: "Tags", 0x7373: "Tag", 0x63C0: "Targets", 0x63C5: "TagTrackUID",
    0x68CA: "TargetTypeValue", 0x63CA: "TargetType",
    0x67C8: "SimpleTag", 0x45A3: "TagName", 0x4487: "TagString", 0x447A: "TagLanguage", 0x4484: "TagDefault",
    0x1F43B675: "Cluster", 0xE7: "Timestamp", 0xA3: "SimpleBlock", 0xA0: "BlockGroup", 0xA1: "Block",
    0x9B: "BlockDuration", 0xFB: "ReferenceBlock", 0x75A2: "DiscardPadding", 0x75A1: "BlockAdditions",
    0xA6: "BlockMore", 0xEE: "BlockAddID", 0xA5: "BlockAdditional", 0xAB: "PrevSize",
    0x1C53BB6B: "Cues", 0xBB: "CuePoint", 0xB3: "CueTime", 0xB7: "CueTrackPositions", 0xF7: "CueTrack",
    0xF1: "CueClusterPosition", 0xF0: "CueRelativePosition", 0xB2: "CueDuration", 0x5378: "CueBlockNumber",
    0x41E4: "BlockAdditionMapping", 0x41F0: "BlockAddIDValue", 0x41E7: "BlockAddIDType", 0x41A4: "BlockAddIDName",
}
MASTERS = {0x1A45DFA3, 0x18538067, 0x114D9B74, 0x4DBB, 0x1549A966, 0x1654AE6B, 0xAE, 0xE0, 0xE1, 0x55B0, 0x1254C367,
           0x7373, 0x63C0, 0x67C8, 0x1F43B675, 0xA0, 0x75A1, 0xA6, 0x1C53BB6B, 0xBB, 0xB7, 0x41E4}
STRINGS = {0x4282, 0x4D80, 0x5741, 0x7BA9, 0x22B59C, 0x86, 0x536E, 0x45A3, 0x4487, 0x447A, 0x63CA, 0x41A4}
FLOATS = {0x4489, 0xB5, 0x78B5}
SIGNED = {0xFB, 0x75A2}


def vint(b, p):
    first = b[p]
    n = 1
    while n <= 8 and not (first & (0x80 >> (n - 1))):
        n += 1
    val = first & ((0x80 >> (n - 1)) - 1)
    for i in range(1, n):
        val = (val << 8) | b[p + i]
    return val, n, val == (1 << (7 * n)) - 1


def elid(b, p):
    first = b[p]
    n = 1
    while not (first & (0x80 >> (n - 1))):
        n += 1
    return int.from_bytes(b[p:p + n], "big"), n


def dump(b, start, end, depth, show_blocks, seg_data):
    p = start
    nblocks = 0
    while p < end:
        i, n = elid(b, p)
        size, m, unk = vint(b, p + n)
        data = p + n + m
        e = end if unk else data + size
        name = NAMES.get(i, "0x%X" % i)
        rel = "" if seg_data is None else " seg+%d" % (p - seg_data)
        head = "%s%-8d%s %s id=%X szlen=%d size=%s" % ("  " * depth, p, rel, name, i, m, "unknown" if unk else size)
        if i in (0xA3, 0xA1):
            tn, k, _ = vint(b, data)
            ts = struct.unpack(">h", b[data + k:data + k + 2])[0]
            fl = b[data + k + 2]
            nblocks += 1
            if show_blocks:
                print(head + " track=%d rel=%d flags=%02X payload=%d" % (tn, ts, fl, e - data - k - 3))
        elif i in MASTERS:
            if not (i == 0xA0 and not show_blocks):
                print(head)
            dump(b, data, e, depth + 1, show_blocks, data if i == 0x18538067 else seg_data)
        else:
            v = b[data:e]
            if i in STRINGS:
                val = repr(v.decode("utf-8", "replace"))
            elif i in FLOATS:
                val = struct.unpack(">d", v)[0] if len(v) == 8 else struct.unpack(">f", v)[0]
            elif i in SIGNED:
                val = int.from_bytes(v, "big", signed=True)
            elif i == 0xEC or len(v) > 8:
                val = "<%d bytes>%s" % (len(v), (" " + v[:24].hex()) if i != 0xEC else "")
            else:
                val = int.from_bytes(v, "big")
            if show_blocks or i != 0xE7 or True:
                print(head + " = " + str(val))
        p = e
    if nblocks and not show_blocks:
        print("%s(%d blocks)" % ("  " * depth, nblocks))


if __name__ == "__main__":
    data = open(sys.argv[1], "rb").read()
    print("file size", len(data))
    dump(data, 0, len(data), 0, "--blocks" in sys.argv, None)
