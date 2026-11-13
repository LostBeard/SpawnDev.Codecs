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
// Everything except the tile data and the size fields depends only on (width, height, baseQIndex, tile grid), so the
// host builds that small fixed blob (TD + SH OBUs, then the uncompressed frame header: well under 128 bytes of
// metadata, the "metadata struct setup" the cardinal rule allows) and uploads it once per frame. The data-dependent
// part runs here, in two launches:
//   1. Offsets (one thread): the running sum of the tile lengths (plus the size prefixes), i.e. where each tile's
//      region starts inside the tile group, and the group's total length.
//   2. Bytes (one thread per output byte): the leb128 Frame OBU size, the blob, the size prefixes and the tile bytes
//      into the caller's output. A byte finds its tile by a binary search of the offsets.
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

    private readonly Action<Index1D, ArrayView<long>, ArrayView<int>, int> _offsets;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<long>, ArrayView<int>, ArrayView<byte>, ArrayView<int>,
        int, int, int, int> _bytes;

    /// <summary>Compile the kernels.</summary>
    public Av1ObuAssembleKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _offsets = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<int>, int>(Av1TileOffsetsKernel);
        _bytes = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<long>, ArrayView<int>, ArrayView<byte>, ArrayView<int>,
            int, int, int, int>(Av1ObuAssembleByteKernel);
    }

    /// <summary>
    /// Queue the assembly of one frame. <paramref name="blob"/> = TD + SH OBUs (<paramref name="aLen"/> bytes) then
    /// the uncompressed header (<paramref name="bLen"/> bytes). Tile t's bytes are at
    /// <c>t * tileStride</c> in <paramref name="tiles"/> with their length in <paramref name="tileLen"/>[t].
    /// <paramref name="tileOffsets"/> (<paramref name="tileCount"/> + 1 ints) is this frame's scratch for the
    /// offsets. Writes the temporal unit to <paramref name="output"/> and its length to <paramref name="outLen"/>[0].
    /// <paramref name="maxBytes"/> bounds the launch (the largest temporal unit possible).
    /// </summary>
    public void Run(
        ArrayView<byte> blob, ArrayView<byte> tiles, ArrayView<long> tileLen, ArrayView<int> tileOffsets,
        ArrayView<byte> output, ArrayView<int> outLen, int aLen, int bLen, int tileCount, int tileStride, int maxBytes)
    {
        if (aLen < 0 || bLen < 0 || blob.Length < aLen + bLen) throw new ArgumentException("blob is shorter than aLen + bLen.", nameof(blob));
        if (tileCount < 1) throw new ArgumentOutOfRangeException(nameof(tileCount));
        if (outLen.Length < 1 || tileLen.Length < tileCount) throw new ArgumentException("outLen needs one element and tileLen one per tile.");
        if (tileOffsets.Length < tileCount + 1) throw new ArgumentException("tileOffsets needs tileCount + 1 elements.", nameof(tileOffsets));
        if (tiles.Length < (long)tileCount * tileStride) throw new ArgumentException("tiles is shorter than tileCount * tileStride.", nameof(tiles));
        if (output.Length < maxBytes) throw new ArgumentException("output is shorter than maxBytes.", nameof(output));
        _offsets(1, tileLen, tileOffsets, tileCount);
        _bytes(maxBytes, blob, tiles, tileLen, tileOffsets, output, outLen, aLen, bLen, tileCount, tileStride);
    }

    /// <summary>
    /// One thread: <c>tileOffsets[t]</c> = where tile t's region (its size prefix, if any, then its bytes) starts
    /// inside the tile group; <c>tileOffsets[tileCount]</c> = the group's length. At most 4096 tiles (64 x 64).
    /// </summary>
    private static void Av1TileOffsetsKernel(Index1D _, ArrayView<long> tileLen, ArrayView<int> tileOffsets, int tileCount)
    {
        int offset = 0;
        for (int t = 0; t < tileCount; t++)
        {
            tileOffsets[t] = offset;
            offset += (int)tileLen[t];
            if (t < tileCount - 1) offset += TileSizeBytes;
        }
        tileOffsets[tileCount] = offset;
    }

    private static void Av1ObuAssembleByteKernel(
        Index1D idx, ArrayView<byte> blob, ArrayView<byte> tiles, ArrayView<long> tileLen, ArrayView<int> tileOffsets,
        ArrayView<byte> output, ArrayView<int> outLen, int aLen, int bLen, int tileCount, int tileStride)
    {
        int flagBytes = tileCount > 1 ? 1 : 0;
        int payload = bLen + flagBytes + tileOffsets[tileCount];
        // leb128 byte count of the payload size.
        int lebLen = 1;
        int rest = payload >> 7;
        while (rest > 0)
        {
            lebLen++;
            rest >>= 7;
        }
        int total = aLen + 1 + lebLen + payload;
        int i = idx.X;
        if (i == 0) outLen[0] = total;
        if (i >= total) return;

        int value;
        if (i < aLen)
        {
            value = blob[i];
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
                    value = blob[aLen + m];
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
