// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 OBU assembly on the accelerator. A keyframe temporal unit is
//
//     [Temporal Delimiter OBU][Sequence Header OBU]
//     [Frame OBU = header byte, leb128(size), uncompressed header, tile group]
//
//     tile group = [one 0x00 byte when the frame has more than one tile: tile_start_and_end_present_flag = 0 and the
//                   byte alignment] then, for each tile in raster order,
//                  [tile_size_minus_1, 4 bytes little-endian: every tile except the last][the tile's bytes]
//
// Everything except the tile data, the size fields and the level depends only on (width, height, baseQIndex, tile
// grid, frame rate), so the host builds that small fixed blob (Av1KeyframeFraming.BuildFramingBlob: for every level
// the stream can signal, the TD + SH OBUs and the largest Frame OBU that level allows, then the uncompressed frame
// header: metadata, the "metadata struct setup" the cardinal rule allows) and uploads it. The data-dependent part
// runs here, in two launches:
//   1. Offsets (one thread): the running sum of the tile lengths (plus the size prefixes), i.e. where each tile's
//      region starts inside the tile group, and the group's total length; and, from the Frame OBU length that gives,
//      the level: the first candidate of the blob's table whose limit the frame is within (AV1 spec Annex A.3: the
//      frame size and bit rate limits depend on the coded bytes). Its prefix's offset and length go to the two
//      elements after the tile offsets.
//   2. Bytes (one thread per output byte): the chosen prefix, the leb128 Frame OBU size, the uncompressed header,
//      the size prefixes and the tile bytes into the caller's output. A byte finds its tile by a binary search of
//      the offsets.
// The tile bytes never visit the host.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>Assembles an AV1 keyframe temporal unit (TD + SH + Frame OBU with its tile group) on the accelerator.</summary>
public sealed class Av1ObuAssembleKernel : IDisposable
{
    /// <summary>OBU header byte of a Frame OBU with a size field (obu_type 6, obu_has_size_field 1).</summary>
    public const int FrameObuHeaderByte = (6 << 3) | 0x02;

    /// <summary>Bytes of a tile_size_minus_1 field (tile_size_bytes_minus_1 = 3 in the frame header).</summary>
    public const int TileSizeBytes = Av1KeyframeFraming.TileSizeBytes;

    /// <summary>Elements of a frame's tile offsets scratch: one per tile, the tile group's length, and the chosen prefix's offset and length.</summary>
    public static int OffsetsLength(int tileCount) => tileCount + 3;

    private readonly Action<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<byte>, int, int> _offsets;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<long>, ArrayView<int>, ArrayView<byte>, ArrayView<int>,
        int, int, int, int> _bytes;

    /// <summary>Compile the kernels.</summary>
    public Av1ObuAssembleKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _offsets = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<byte>, int, int>(Av1TileOffsetsKernel);
        _bytes = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<long>, ArrayView<int>, ArrayView<byte>, ArrayView<int>,
            int, int, int, int>(Av1ObuAssembleByteKernel);
    }

    /// <summary>
    /// Queue the assembly of one frame. <paramref name="blob"/> = the framing blob of
    /// <see cref="Av1KeyframeFraming.BuildFramingBlob"/> (the level table, the candidates' TD + SH OBUs, and the
    /// uncompressed header: <paramref name="bLen"/> bytes at <paramref name="bOffset"/>). Tile t's bytes are at
    /// <c>t * tileStride</c> in <paramref name="tiles"/> with their length in <paramref name="tileLen"/>[t].
    /// <paramref name="tileOffsets"/> (<see cref="OffsetsLength"/> ints) is this frame's scratch for the offsets and
    /// the chosen level. Writes the temporal unit to <paramref name="output"/> and its length to
    /// <paramref name="outLen"/>[0]. <paramref name="maxBytes"/> bounds the launch (the largest temporal unit possible).
    /// </summary>
    public void Run(
        ArrayView<byte> blob, ArrayView<byte> tiles, ArrayView<long> tileLen, ArrayView<int> tileOffsets,
        ArrayView<byte> output, ArrayView<int> outLen, int bOffset, int bLen, int tileCount, int tileStride, int maxBytes)
    {
        if (bOffset < Av1KeyframeFraming.LevelTableHeaderBytes + Av1KeyframeFraming.LevelTableEntryBytes || bLen < 0 || blob.Length < bOffset + bLen)
            throw new ArgumentException("blob is shorter than bOffset + bLen, or has no level table.", nameof(blob));
        if (tileCount < 1) throw new ArgumentOutOfRangeException(nameof(tileCount));
        if (outLen.Length < 1 || tileLen.Length < tileCount) throw new ArgumentException("outLen needs one element and tileLen one per tile.");
        if (tileOffsets.Length < OffsetsLength(tileCount)) throw new ArgumentException("tileOffsets needs OffsetsLength(tileCount) elements.", nameof(tileOffsets));
        if (tiles.Length < (long)tileCount * tileStride) throw new ArgumentException("tiles is shorter than tileCount * tileStride.", nameof(tiles));
        if (output.Length < maxBytes) throw new ArgumentException("output is shorter than maxBytes.", nameof(output));
        _offsets(1, tileLen, tileOffsets, blob, tileCount, bLen);
        _bytes(maxBytes, blob, tiles, tileLen, tileOffsets, output, outLen, bOffset, bLen, tileCount, tileStride);
    }

    /// <summary>Bytes of the leb128 coding of <paramref name="value"/>.</summary>
    private static int Leb128Length(int value)
    {
        int length = 1;
        int rest = value >> 7;
        while (rest > 0)
        {
            length++;
            rest >>= 7;
        }
        return length;
    }

    /// <summary>The little-endian 32-bit integer at <paramref name="at"/> of the framing blob.</summary>
    private static int BlobInt32(ArrayView<byte> blob, int at) =>
        blob[at] | (blob[at + 1] << 8) | (blob[at + 2] << 16) | (blob[at + 3] << 24);

    /// <summary>
    /// One thread: <c>tileOffsets[t]</c> = where tile t's region (its size prefix, if any, then its bytes) starts
    /// inside the tile group; <c>tileOffsets[tileCount]</c> = the group's length. At most 4096 tiles (64 x 64).
    /// Then the level: the first candidate of the blob's table that allows the Frame OBU these tiles make; its
    /// prefix's offset in the blob to <c>tileOffsets[tileCount + 1]</c>, the prefix's length to
    /// <c>tileOffsets[tileCount + 2]</c>.
    /// </summary>
    private static void Av1TileOffsetsKernel(
        Index1D _, ArrayView<long> tileLen, ArrayView<int> tileOffsets, ArrayView<byte> blob, int tileCount, int bLen)
    {
        int offset = 0;
        for (int t = 0; t < tileCount; t++)
        {
            tileOffsets[t] = offset;
            offset += (int)tileLen[t];
            if (t < tileCount - 1) offset += TileSizeBytes;
        }
        tileOffsets[tileCount] = offset;

        int payload = bLen + (tileCount > 1 ? 1 : 0) + offset;
        int frameObu = 1 + Leb128Length(payload) + payload;
        int candidates = BlobInt32(blob, 0);
        int chosen = 0;
        while (chosen < candidates - 1
            && BlobInt32(blob, Av1KeyframeFraming.LevelTableHeaderBytes + chosen * Av1KeyframeFraming.LevelTableEntryBytes) < frameObu)
            chosen++;
        int entry = Av1KeyframeFraming.LevelTableHeaderBytes + chosen * Av1KeyframeFraming.LevelTableEntryBytes;
        tileOffsets[tileCount + 1] = BlobInt32(blob, entry + 4);
        tileOffsets[tileCount + 2] = BlobInt32(blob, entry + 8);
    }

    private static void Av1ObuAssembleByteKernel(
        Index1D idx, ArrayView<byte> blob, ArrayView<byte> tiles, ArrayView<long> tileLen, ArrayView<int> tileOffsets,
        ArrayView<byte> output, ArrayView<int> outLen, int bOffset, int bLen, int tileCount, int tileStride)
    {
        int flagBytes = tileCount > 1 ? 1 : 0;
        int payload = bLen + flagBytes + tileOffsets[tileCount];
        int lebLen = Leb128Length(payload);
        int aOffset = tileOffsets[tileCount + 1];   // the chosen level: its TD + SH OBUs
        int aLen = tileOffsets[tileCount + 2];
        int total = aLen + 1 + lebLen + payload;
        int i = idx.X;
        if (i == 0) outLen[0] = total;
        if (i >= total) return;

        int value;
        if (i < aLen)
        {
            value = blob[aOffset + i];
        }
        else
        {
            int j = i - aLen;
            if (j == 0)
            {
                value = FrameObuHeaderByte;
            }
            else if (j <= lebLen)
            {
                int k = j - 1;
                int part = (payload >> (7 * k)) & 0x7F;
                if (k < lebLen - 1) part |= 0x80;
                value = part;
            }
            else
            {
                int m = j - 1 - lebLen;
                if (m < bLen)
                {
                    value = blob[bOffset + m];
                }
                else if (m - bLen < flagBytes)
                {
                    value = 0;   // tile_start_and_end_present_flag = 0, then the byte alignment
                }
                else
                {
                    int q = m - bLen - flagBytes;   // position inside the tile group's tiles
                    // the last tile whose region starts at or before q
                    int lo = 0, hi = tileCount - 1;
                    while (lo < hi)
                    {
                        int mid = (lo + hi + 1) >> 1;
                        if (tileOffsets[mid] <= q) lo = mid;
                        else hi = mid - 1;
                    }
                    int r = q - tileOffsets[lo];
                    int prefix = lo < tileCount - 1 ? TileSizeBytes : 0;
                    if (r < prefix)
                    {
                        int sizeMinus1 = (int)tileLen[lo] - 1;
                        value = (sizeMinus1 >> (8 * r)) & 0xFF;
                    }
                    else
                    {
                        value = tiles[(long)lo * tileStride + (r - prefix)];
                    }
                }
            }
        }
        output[i] = (byte)value;
    }

    /// <summary>Release the kernels.</summary>
    public void Dispose() { /* delegates are value types; nothing to release */ }
}
